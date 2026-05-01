import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EEGTransformerEncoder(nn.Module):
    def __init__(self, input_dim=105, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1, output_dim=4096):
        super(EEGTransformerEncoder, self).__init__()
        
        # 1. 线性输入投影层 (105 -> 512)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Alignment Head (MLP 层): 512 -> 2048 -> 4096
        self.alignment_head = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )
        
        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        
        # 对 Alignment Head 进行初始化
        for m in self.alignment_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, src, src_key_padding_mask=None):
        """
        src: (batch_size, seq_len, input_dim)
        src_key_padding_mask: (batch_size, seq_len)
        """
        # 投影与位置编码
        x = self.input_projection(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer 编码
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Mean Pooling
        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask
            mask = mask.unsqueeze(-1).float()
            output = output * mask
            sum_output = torch.sum(output, dim=1)
            count = torch.sum(mask, dim=1)
            pooled_output = sum_output / count.clamp(min=1e-9)
        else:
            pooled_output = torch.mean(output, dim=1)
            
        # 对齐映射 (512 -> 4096)
        aligned_output = self.alignment_head(pooled_output)
        aligned_output = F.normalize(aligned_output, p=2, dim=-1)

        return aligned_output # (batch, 4096)

def load_frozen_llm(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    加载冻结的 LLM (Llama-3)，使用 4-bit 量化以优化显存
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"Loading LLM: {model_name} in 4-bit...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 彻底冻结权重
    for param in llm.parameters():
        param.requires_grad = False
    llm.eval()
    
    # 获取目标 Embedding 维度
    embedding_dim = llm.config.hidden_size
    print(f"LLM loaded. Target embedding dimension: {embedding_dim}")
    
    return llm, embedding_dim

if __name__ == "__main__":
    # 模拟获取 LLM 维度 (Llama-3 默认为 4096)
    target_dim = 4096 
    model = EEGTransformerEncoder(output_dim=target_dim)
    
    test_input = torch.randn(4, 15, 105)
    out = model(test_input)
    print(f"Aligned EEG Feature Shape: {out.shape}") # (4, 4096)

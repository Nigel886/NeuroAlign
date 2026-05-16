import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

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

class CenteringLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("delta", torch.zeros(int(dim), dtype=torch.float32))
        self.scaling_factor = 1.0

    @torch.no_grad()
    def set_delta(self, delta):
        if delta is None:
            self.delta.zero_()
            return
        d = delta.detach()
        if d.dim() != 1:
            d = d.view(-1)
        self.delta.copy_(d.to(device=self.delta.device, dtype=self.delta.dtype))

    def forward(self, x, delta=None):
        if delta is None:
            d = self.delta
        else:
            d = delta
            if d.dim() != 1:
                d = d.view(-1)
            d = d.to(device=x.device, dtype=x.dtype)
        return x + d * self.scaling_factor

class EEGTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=105,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        output_dim=4096,
        vocab_size=None,
        num_subjects=None,
        subject_hidden_dim=256,
        content_dim=384,
        style_dim=128,
        enable_centering=True,
    ):
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
        
        self.style_dim = int(style_dim)
        self.vocab_size = int(vocab_size) if vocab_size is not None else None

        # 4. Dual-head projection
        # proj_semantic: (d_model -> 4096) for Llama-3 space
        # proj_style: (d_model -> 128) for DANN
        self.proj_semantic = nn.Linear(d_model, output_dim)
        self.proj_style = nn.Linear(d_model, self.style_dim)
        self.centering = CenteringLayer(output_dim) if enable_centering else None
        self.aux_decoder = nn.Linear(d_model, self.vocab_size) if self.vocab_size is not None else None

        self.subject_classifier = (
            nn.Sequential(
                nn.Linear(self.style_dim, int(subject_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(subject_hidden_dim), int(num_subjects)),
            )
            if num_subjects is not None
            else None
        )
        if self.subject_classifier is not None:
            self.register_buffer("subject_grad_norm", torch.zeros(1, dtype=torch.float32))
            self.subject_classifier.register_full_backward_hook(self._subject_backward_hook)
        
        self.d_model = d_model
        self._init_weights()

    def _subject_backward_hook(self, module, grad_input, grad_output):
        if not grad_output:
            return
        g = grad_output[0]
        if not torch.is_tensor(g):
            return
        with torch.no_grad():
            self.subject_grad_norm.fill_(g.detach().to(dtype=torch.float32).norm())

    def _init_weights(self):
        initrange = 0.1
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        
        nn.init.xavier_normal_(self.proj_semantic.weight)
        nn.init.constant_(self.proj_semantic.bias, 0)
        nn.init.xavier_normal_(self.proj_style.weight)
        nn.init.constant_(self.proj_style.bias, 0)
        if self.aux_decoder is not None:
            nn.init.xavier_normal_(self.aux_decoder.weight)
            nn.init.constant_(self.aux_decoder.bias, 0)
        if self.subject_classifier is not None:
            for m in self.subject_classifier:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        src,
        src_key_padding_mask=None,
        return_subject_logits=False,
        return_aux_logits=False,
        grl_alpha=1.0,
        centering_delta=None,
        aug_mode=False,
        subject_labels=None,
    ):
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
            
        aux_logits = self.aux_decoder(pooled_output) if self.aux_decoder is not None else None

        z_semantic = self.proj_semantic(pooled_output)
        if self.centering is not None:
            z_semantic = self.centering(z_semantic, delta=centering_delta)
        z_semantic = F.normalize(z_semantic, p=2, dim=-1)

        z_style = self.proj_style(pooled_output)
        if aug_mode:
            bsz = int(z_style.size(0))
            if bsz > 1:
                device = z_style.device
                perm = torch.randperm(bsz, device=device)
                if subject_labels is not None and torch.is_tensor(subject_labels) and int(subject_labels.numel()) == bsz:
                    labels = subject_labels.to(device=device)
                    for _ in range(5):
                        same = labels == labels[perm]
                        if not bool(same.any().item()):
                            break
                        perm2 = torch.randperm(bsz, device=device)
                        perm = torch.where(same, perm2, perm)
                other = z_style[perm]
                lam = torch.rand(bsz, 1, device=device, dtype=z_style.dtype)
                z_style = lam * z_style + (1.0 - lam) * other

        if return_subject_logits:
            if self.subject_classifier is None:
                raise ValueError("Subject classifier is not enabled. Set num_subjects when constructing the model.")
            rev = ReverseLayerF.apply(z_style, grl_alpha)
            subject_logits = self.subject_classifier(rev)
            if return_aux_logits:
                return z_semantic, z_style, subject_logits, aux_logits
            return z_semantic, z_style, subject_logits

        if return_aux_logits:
            return z_semantic, z_style, aux_logits

        return z_semantic, z_style

    @torch.no_grad()
    def set_centering_delta(self, delta):
        if self.centering is not None:
            self.centering.set_delta(delta)

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
        low_cpu_mem_usage=True,
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

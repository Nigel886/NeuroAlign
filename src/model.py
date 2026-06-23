import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
import time
import urllib.request
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from src.models.subject_graph import SubjectGraphBase

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
        enable_centering=True,
        use_graph_base=False,
        num_subjects=None,
        graph_hid_dim=256,
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

        # 4. Projection to LLM space
        self.proj_semantic = nn.Linear(d_model, output_dim)
        self.centering = CenteringLayer(output_dim) if enable_centering else None
        self.graph_base = None
        if bool(use_graph_base):
            if num_subjects is None:
                raise ValueError("num_subjects is required when use_graph_base=True")
            self.graph_base = SubjectGraphBase(num_subjects=int(num_subjects), embed_dim=int(output_dim), hid_dim=int(graph_hid_dim))
        
        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        src,
        src_key_padding_mask=None,
        centering_delta=None,
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

        z_semantic = self.proj_semantic(pooled_output)
        if self.centering is not None:
            z_semantic = self.centering(z_semantic, delta=centering_delta)
        if self.graph_base is not None:
            z_semantic = self.graph_base(z_semantic, subject_labels=subject_labels)
        z_semantic = F.normalize(z_semantic, p=2, dim=-1)
        return z_semantic

    @torch.no_grad()
    def set_centering_delta(self, delta):
        if self.centering is not None:
            self.centering.set_delta(delta)

def load_frozen_llm(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    # #region debug-point A:model-load-bootstrap
    _dbg_env = os.path.join(os.getcwd(), ".dbg", "eval-llama-exit.env")
    _dbg_url = "http://127.0.0.1:7777/event"
    _dbg_session = "eval-llama-exit"
    try:
        with open(_dbg_env, "r", encoding="utf-8") as _f:
            for _l in _f.read().splitlines():
                if _l.startswith("DEBUG_SERVER_URL="):
                    _dbg_url = _l.split("=", 1)[1].strip() or _dbg_url
                elif _l.startswith("DEBUG_SESSION_ID="):
                    _dbg_session = _l.split("=", 1)[1].strip() or _dbg_session
    except Exception:
        pass
    def _dbg_send(hypothesis_id, msg, data=None, run_id="pre"):
        payload = {
            "sessionId": _dbg_session,
            "runId": run_id,
            "hypothesisId": str(hypothesis_id),
            "location": "src/model.py:load_frozen_llm",
            "msg": f"[DEBUG] {msg}",
            "data": data or {},
            "ts": int(time.time() * 1000),
        }
        try:
            req = urllib.request.Request(
                _dbg_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=1.5).read()
        except Exception:
            pass
    def _dbg_gpu(tag):
        snap = {"tag": tag, "cuda": bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(0)
                snap["gpu_free_mb"] = int(free // (1024 * 1024))
                snap["gpu_total_mb"] = int(total // (1024 * 1024))
                snap["gpu_alloc_mb"] = int(torch.cuda.memory_allocated(0) // (1024 * 1024))
                snap["gpu_reserved_mb"] = int(torch.cuda.memory_reserved(0) // (1024 * 1024))
            except Exception:
                pass
        _dbg_send("B", "gpu-snapshot", snap)
    _dbg_send("B", "enter-load-frozen-llm", {"model_name": str(model_name), "torch": getattr(torch, "__version__", None)})
    _dbg_gpu("enter")
    # #endregion
    model_name_upper = str(model_name).upper()
    is_llama_8b = ("LLAMA" in model_name_upper) and ("8B" in model_name_upper)
    if torch.cuda.is_available():
        free_mb = None
        total_mb = None
        try:
            free, total = torch.cuda.mem_get_info(0)
            free_mb = int(free // (1024 * 1024))
            total_mb = int(total // (1024 * 1024))
        except Exception:
            pass

        # if is_llama_8b and total_mb is not None and total_mb <= 9000:
        #     msg = (
        #         f"Refusing to load {model_name} on a {total_mb} MiB GPU. "
        #         "Runtime evidence from this project shows that Llama-3 8B exits during from_pretrained() "
        #         "on 8GB-class Windows/WDDM GPUs before Python can surface a traceback. "
        #         "Use a smaller LLM (for example gpt2 or a 1B-class model), or run on a higher-VRAM GPU."
        #     )
        #     _dbg_send("D", "guard-raise", {"reason": "llama-8b-on-8gb-gpu", "total_mb": total_mb, "free_mb": free_mb, "model_name": str(model_name)})
        #     raise RuntimeError(msg)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading LLM: {model_name} in 4-bit...")
        _dbg_send("B", "from_pretrained-4bit-auto", {"device_map": "auto", "free_mb": free_mb, "total_mb": total_mb})
        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="balanced",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        _dbg_gpu("after-load")
    else:
        if is_llama_8b:
            msg = (
                f"Refusing to load {model_name} on CPU. "
                "The 8B model is too large for this project's eval/training flow without a stable CUDA path, "
                "and previous runs were terminated during loading. Use a CUDA-capable environment or a smaller LLM."
            )
            _dbg_send("D", "guard-raise", {"reason": "llama-8b-on-cpu", "model_name": str(model_name)})
            raise RuntimeError(msg)
        print(f"Loading LLM: {model_name} on CPU (no 4-bit)...")
        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        llm.to(torch.device("cpu"))
    
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

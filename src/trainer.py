import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import os
import numpy as np
from tqdm import tqdm

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, eeg_features, text_features):
        """
        CLIP-style InfoNCE Loss
        eeg_features: (batch, dim)
        text_features: (batch, dim)
        """
        # 归一化特征
        eeg_features = F.normalize(eeg_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # 计算相似度矩阵
        logits_per_eeg = torch.matmul(eeg_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_eeg.t()
        
        # Ground truth: 对角线应该是最相似的
        labels = torch.arange(eeg_features.size(0), device=eeg_features.device)
        
        loss_eeg = F.cross_entropy(logits_per_eeg, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        
        return (loss_eeg + loss_text) / 2

def train_one_epoch(model, llm, dataloader, optimizer, scaler, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        # 将数据移至设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        eeg = batch['eeg'].to(device)
        eeg_mask = batch['eeg_mask'].to(device) # (batch, seq_len)
        
        # 转换为 Transformer 需要的 bool mask (True 为 padding)
        src_key_padding_mask = (eeg_mask == 0)
        
        # 1. 提取 LLM 文本特征 (仅取 EOS 位置或 Mean Pooling)
        with torch.no_grad():
            # 使用 LLM 的 Embedding 层作为目标
            # 注意: Llama-3 4-bit 加载后输出为 bfloat16 或 float16
            outputs = llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # 取最后一层的隐藏状态并对时间维度取平均 (或者取最后一个 token)
            last_hidden_state = outputs.hidden_states[-1] # (batch, seq_len, 4096)
            # Masked Mean Pooling for Text
            text_mask = attention_mask.unsqueeze(-1).float()
            text_features = torch.sum(last_hidden_state * text_mask, dim=1) / torch.sum(text_mask, dim=1).clamp(min=1e-9)
        
        # 2. 混合精度前向传播
        with autocast():
            # 提取 EEG 特征并映射到 4096 维
            eeg_features = model(eeg, src_key_padding_mask=src_key_padding_mask)
            
            # 计算对比损失
            criterion = ContrastiveLoss().to(device)
            loss = criterion(eeg_features, text_features)
            loss = loss / accumulation_steps # 梯度累积缩放
            
        # 3. 反向传播
        scaler.scale(loss).backward()
        
        # 4. 梯度累积更新
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
    return total_loss / len(dataloader)

def save_checkpoint(model, epoch, path="checkpoints/"):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, f"eeg_encoder_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved: {save_path}")

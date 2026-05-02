import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature), dtype=torch.float32))

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
        logits_per_eeg = torch.matmul(eeg_features, text_features.t()) * self.logit_scale.exp()
        logits_per_text = logits_per_eeg.t()
        
        # Ground truth: 对角线应该是最相似的
        labels = torch.arange(eeg_features.size(0), device=eeg_features.device)
        
        loss_eeg = F.cross_entropy(logits_per_eeg, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        
        return (loss_eeg + loss_text) / 2

def _build_subject_mapping(dataloader):
    dataset = getattr(dataloader, "dataset", None)
    items = getattr(dataset, "data", None) if dataset is not None else None
    if not isinstance(items, list):
        return {}
    subject_ids = []
    for it in items:
        if isinstance(it, dict) and it.get("subject_id") is not None:
            subject_ids.append(str(it["subject_id"]).upper())
    subject_ids = sorted({sid for sid in subject_ids if sid != "UNK"})
    return {sid: i for i, sid in enumerate(subject_ids)}

def _linear_warmup_to_one(progress):
    return float(max(0.0, min(1.0, progress)))

def train_one_epoch(
    model,
    llm,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch=1,
    total_epochs=1,
    accumulation_steps=4,
):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    criterion = ContrastiveLoss().to(device)
    existing_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    new_params = [p for p in criterion.parameters() if id(p) not in existing_params]
    if new_params:
        optimizer.add_param_group({"params": new_params})

    ce_loss = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    subject_to_idx = _build_subject_mapping(dataloader)
    
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
        
        denom = max(1, total_epochs * len(dataloader))
        progress = ((epoch - 1) * len(dataloader) + step) / denom
        lambda_subject = _linear_warmup_to_one(progress)

        subject_ids = batch.get("subject_ids", None)
        use_subject = (
            getattr(model, "subject_classifier", None) is not None
            and isinstance(subject_ids, list)
            and len(subject_to_idx) > 0
        )
        if use_subject:
            labels = []
            for sid in subject_ids:
                if sid is None:
                    labels.append(-1)
                else:
                    labels.append(subject_to_idx.get(str(sid).upper(), -1))
            subject_labels = torch.tensor(labels, dtype=torch.long, device=device)
        else:
            subject_labels = None

        # 2. 混合精度前向传播
        with autocast():
            if use_subject:
                eeg_features, subject_logits = model(
                    eeg,
                    src_key_padding_mask=src_key_padding_mask,
                    return_subject_logits=True,
                    grl_alpha=lambda_subject,
                )
                if subject_logits.size(-1) != len(subject_to_idx):
                    raise ValueError(
                        f"subject_logits dim={subject_logits.size(-1)} but num_subjects={len(subject_to_idx)}. "
                        "Construct model with num_subjects matching the training subjects."
                    )
                alignment_loss = criterion(eeg_features, text_features)
                subject_loss = ce_loss(subject_logits, subject_labels)
                loss = alignment_loss + lambda_subject * subject_loss
            else:
                eeg_features = model(eeg, src_key_padding_mask=src_key_padding_mask)
                alignment_loss = criterion(eeg_features, text_features)
                subject_loss = None
                loss = alignment_loss

            loss = loss / accumulation_steps
            
        # 3. 反向传播
        scaler.scale(loss).backward()
        
        # 4. 梯度累积更新
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        postfix = {
            "loss": loss.item() * accumulation_steps,
            "lambda": float(lambda_subject),
        }
        postfix["align"] = float(alignment_loss.detach().item())
        if subject_loss is not None:
            postfix["subj"] = float(subject_loss.detach().item())
        pbar.set_postfix(postfix)
        
    return total_loss / len(dataloader)

def save_checkpoint(model, epoch, path="checkpoints/"):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, f"eeg_encoder_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved: {save_path}")

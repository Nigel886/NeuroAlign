import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm

class CentroidTracker(nn.Module):
    def __init__(self, dim, momentum=0.9):
        super().__init__()
        self.momentum = float(momentum)
        self.register_buffer("eeg_centroid", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("text_centroid", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("_initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, eeg_batch, text_batch):
        eeg_mean = eeg_batch.detach().to(dtype=torch.float32).mean(dim=0)
        text_mean = text_batch.detach().to(dtype=torch.float32).mean(dim=0)
        if not bool(self._initialized.item()):
            self.eeg_centroid.copy_(eeg_mean)
            self.text_centroid.copy_(text_mean)
            self._initialized.fill_(True)
            return
        m = self.momentum
        self.eeg_centroid.mul_(m).add_(eeg_mean, alpha=(1.0 - m))
        self.text_centroid.mul_(m).add_(text_mean, alpha=(1.0 - m))

    @torch.no_grad()
    def delta(self):
        return self.text_centroid - self.eeg_centroid

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
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

    def forward_eeg_to_text_bank(self, eeg_features, text_bank, labels=None):
        eeg_features = F.normalize(eeg_features, p=2, dim=-1)
        text_bank = F.normalize(text_bank, p=2, dim=-1)
        logits = torch.matmul(eeg_features, text_bank.t()) * self.logit_scale.exp()
        if labels is None:
            labels = torch.arange(eeg_features.size(0), device=eeg_features.device)
        return F.cross_entropy(logits, labels)


class MemoryBank(nn.Module):
    def __init__(self, queue_size=1024, dim=4096):
        super().__init__()
        self.queue_size = int(queue_size)
        self.dim = int(dim)
        self.register_buffer("queue", F.normalize(torch.randn(self.queue_size, self.dim, dtype=torch.float32), p=2, dim=-1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys):
        keys = keys.detach()
        keys = F.normalize(keys.to(dtype=torch.float32), p=2, dim=-1)
        batch_size = int(keys.size(0))
        if batch_size <= 0:
            return

        ptr = int(self.queue_ptr.item())
        if batch_size >= self.queue_size:
            self.queue.copy_(keys[-self.queue_size :].to(device=self.queue.device))
            self.queue_ptr.fill_(0)
            return

        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[ptr:end].copy_(keys.to(device=self.queue.device))
        else:
            first = self.queue_size - ptr
            self.queue[ptr:].copy_(keys[:first].to(device=self.queue.device))
            self.queue[: end - self.queue_size].copy_(keys[first:].to(device=self.queue.device))
        self.queue_ptr.fill_(end % self.queue_size)

def _build_subject_mapping(dataloader):
    dataset = getattr(dataloader, "dataset", None)
    subject_to_idx = getattr(dataset, "subject_to_idx", None) if dataset is not None else None
    if isinstance(subject_to_idx, dict) and len(subject_to_idx) > 0:
        return {str(k).upper(): int(v) for k, v in subject_to_idx.items()}
    items = getattr(dataset, "data", None) if dataset is not None else None
    if not isinstance(items, list):
        return {}
    subject_ids = []
    for it in items:
        if isinstance(it, dict) and it.get("subject_id") is not None:
            subject_ids.append(str(it["subject_id"]).upper())
    subject_ids = sorted({sid for sid in subject_ids if sid != "UNK"})
    return {sid: i for i, sid in enumerate(subject_ids)}

def _dann_lambda(progress):
    p = float(max(0.0, min(1.0, progress)))
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

def train_one_epoch(
    model,
    llm,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch=1,
    total_epochs=1,
    warmup_epochs=5,
    accumulation_steps=4,
    use_centering=True,
    centering_momentum=0.9,
):
    model.train()
    total_loss = 0
    total_align = 0
    optimizer.zero_grad()

    alignment_weight = 5.0
    subject_weight = 0.5
    gamma_ortho = 0.1
    centroid_momentum = float(centering_momentum)
    criterion = ContrastiveLoss().to(device)
    existing_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    new_params = [p for p in criterion.parameters() if id(p) not in existing_params]
    if new_params:
        optimizer.add_param_group({"params": new_params})

    ce_loss = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    subject_to_idx = _build_subject_mapping(dataloader)
    if getattr(model, "memory_bank", None) is None:
        model.memory_bank = MemoryBank(queue_size=1024, dim=int(getattr(llm.config, "hidden_size", 4096))).to(device)
    
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
        current_step = (epoch - 1) * len(dataloader) + step
        progress = current_step / denom
        if epoch <= warmup_epochs:
            lambda_subject = 0.0
        else:
            lambda_subject = _dann_lambda(progress)

        subject_ids = batch.get("subject_ids", None)
        batch_subject_labels = batch.get("subject_labels", None)
        use_subject = (
            getattr(model, "subject_classifier", None) is not None
            and len(subject_to_idx) > 0
            and (torch.is_tensor(batch_subject_labels) or isinstance(subject_ids, list))
        )
        if use_subject:
            if torch.is_tensor(batch_subject_labels):
                subject_labels = batch_subject_labels.to(device=device, dtype=torch.long)
            else:
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
                eeg_features, subject_logits, content_features, style_features = model(
                    eeg,
                    src_key_padding_mask=src_key_padding_mask,
                    return_subject_logits=True,
                    return_decomposed_features=True,
                    grl_alpha=lambda_subject,
                )
                if getattr(model, "centroid_tracker", None) is None:
                    model.centroid_tracker = CentroidTracker(dim=int(text_features.size(-1)), momentum=centroid_momentum).to(device)
                model.centroid_tracker.update(eeg_features, text_features)
                delta = model.centroid_tracker.delta().to(device=device, dtype=eeg_features.dtype)
                if use_centering and hasattr(model, "set_centering_delta"):
                    model.set_centering_delta(delta)
                if subject_logits.size(-1) != len(subject_to_idx):
                    raise ValueError(
                        f"subject_logits dim={subject_logits.size(-1)} but num_subjects={len(subject_to_idx)}. "
                        "Construct model with num_subjects matching the training subjects."
                    )
                text_bank = torch.cat([text_features, model.memory_bank.queue.detach().to(device=device, dtype=text_features.dtype)], dim=0)
                labels = torch.arange(eeg_features.size(0), device=device)
                alignment_loss = criterion.forward_eeg_to_text_bank(eeg_features, text_bank, labels=labels) * alignment_weight
                total_align += float(alignment_loss.detach().item())
                subject_loss = ce_loss(subject_logits, subject_labels) * subject_weight
                content_norm = F.normalize(content_features, p=2, dim=-1)
                style_norm = F.normalize(style_features, p=2, dim=-1)
                cross = torch.matmul(content_norm.transpose(0, 1), style_norm) / float(content_norm.size(0))
                ortho_loss = torch.mean(cross ** 2)
                loss = alignment_loss + lambda_subject * subject_loss + gamma_ortho * ortho_loss
            else:
                eeg_features = model(eeg, src_key_padding_mask=src_key_padding_mask)
                if getattr(model, "centroid_tracker", None) is None:
                    model.centroid_tracker = CentroidTracker(dim=int(text_features.size(-1)), momentum=centroid_momentum).to(device)
                model.centroid_tracker.update(eeg_features, text_features)
                delta = model.centroid_tracker.delta().to(device=device, dtype=eeg_features.dtype)
                if use_centering and hasattr(model, "set_centering_delta"):
                    model.set_centering_delta(delta)
                text_bank = torch.cat([text_features, model.memory_bank.queue.detach().to(device=device, dtype=text_features.dtype)], dim=0)
                labels = torch.arange(eeg_features.size(0), device=device)
                alignment_loss = criterion.forward_eeg_to_text_bank(eeg_features, text_bank, labels=labels) * alignment_weight
                total_align += float(alignment_loss.detach().item())
                subject_loss = None
                ortho_loss = None
                loss = alignment_loss

            loss = loss / accumulation_steps
            
        # 3. 反向传播
        scaler.scale(loss).backward()
        
        # 4. 梯度累积更新
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        model.memory_bank.enqueue(text_features)
        postfix = {
            "loss": loss.item() * accumulation_steps,
            "lambda": float(lambda_subject),
        }
        postfix["align"] = float(alignment_loss.detach().item())
        if subject_loss is not None:
            postfix["subj"] = float(subject_loss.detach().item())
        if ortho_loss is not None:
            postfix["ortho"] = float(ortho_loss.detach().item())
        pbar.set_postfix(postfix)
        
    denom_batches = max(1, len(dataloader))
    return (total_loss / denom_batches), (total_align / denom_batches)

def save_checkpoint(model, epoch, path="checkpoints/"):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, f"eeg_encoder_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved: {save_path}")

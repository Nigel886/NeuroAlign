import torch
import argparse
import os
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import copy
from src.model import EEGTransformerEncoder, load_frozen_llm
from src.data_loader import get_dataloader, get_loso_loaders
from src.trainer import train_one_epoch, save_checkpoint, ContrastiveLoss, MemoryBank, CentroidTracker

class LowRankSubspaceProjector(torch.nn.Module):
    def __init__(self, embed_dim=4096, rank=16):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.rank = int(rank)
        self.down = torch.nn.Linear(self.embed_dim, self.rank, bias=False)
        self.up = torch.nn.Linear(self.rank, self.embed_dim, bias=False)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))

def _extract_text_features(llm, input_ids, attention_mask):
    with torch.no_grad():
        outputs = llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        text_mask = attention_mask.unsqueeze(-1).float()
        text_feat = torch.sum(last_hidden_state * text_mask, dim=1) / torch.sum(text_mask, dim=1).clamp(min=1e-9)
    return text_feat

def _split_support_query_by_subject(subject_ids, min_query=1):
    if subject_ids is None:
        return None, None
    subj = np.array([str(s).upper() if s is not None else "UNK" for s in subject_ids], dtype=object)
    uniq = [u for u in np.unique(subj) if u != "UNK"]
    if not uniq:
        return None, None
    chosen = uniq[np.random.randint(0, len(uniq))]
    idx = np.where(subj == chosen)[0]
    if idx.size < 2:
        return None, None
    np.random.shuffle(idx)
    qn = max(int(min_query), idx.size // 2)
    qn = min(qn, idx.size - 1)
    query_idx = idx[:qn]
    support_idx = idx[qn:]
    return support_idx, query_idx

def train_one_epoch_meta(
    model,
    llm,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch=1,
    total_epochs=1,
    accumulation_steps=4,
    use_centering=True,
    centering_momentum=0.9,
    temperature=0.05,
    margin=0.2,
    inner_steps=3,
    inner_lr=1e-4,
    inner_lambda_proto=1.0,
    tta_rank=16,
):
    model.train()
    total_loss = 0.0
    total_align = 0.0
    optimizer.zero_grad()

    alignment_weight = 15.0
    criterion = ContrastiveLoss(
        temperature=float(temperature),
        margin=float(margin),
        hard_neg_fraction=0.1,
        hard_neg_weight=1.2,
    ).to(device)
    existing_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    new_params = [p for p in criterion.parameters() if id(p) not in existing_params]
    if new_params:
        optimizer.add_param_group({"params": new_params})

    if getattr(model, "memory_bank", None) is None:
        model.memory_bank = MemoryBank(queue_size=1024, dim=int(getattr(llm.config, "hidden_size", 4096))).to(device)

    pbar = tqdm(dataloader, desc="Training(v7_0_meta)")
    for step, batch in enumerate(pbar):
        if step % accumulation_steps == 0:
            window_count = 0
            window_loss_sum = 0.0
            window_align_sum = 0.0

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        subject_ids = batch.get("subject_ids", None)

        src_key_padding_mask = (eeg_mask == 0)
        text_features = _extract_text_features(llm, input_ids, attention_mask)

        with autocast():
            z_semantic = model(eeg, src_key_padding_mask=src_key_padding_mask)
            z_semantic = F.normalize(z_semantic.to(dtype=torch.float32), p=2, dim=-1).to(dtype=z_semantic.dtype)
            text_features_norm = F.normalize(text_features.to(dtype=torch.float32), p=2, dim=-1).to(dtype=text_features.dtype)
            if getattr(model, "centroid_tracker", None) is None:
                model.centroid_tracker = CentroidTracker(dim=int(text_features_norm.size(-1)), momentum=float(centering_momentum)).to(device)
            model.centroid_tracker.update(z_semantic, text_features_norm)
            delta = model.centroid_tracker.delta().to(device=device, dtype=z_semantic.dtype)
            if use_centering and hasattr(model, "set_centering_delta"):
                model.set_centering_delta(delta)

        support_idx_np, query_idx_np = _split_support_query_by_subject(subject_ids, min_query=1)
        if support_idx_np is None or query_idx_np is None:
            bsz = int(eeg.size(0))
            if bsz < 2:
                continue
            perm = torch.randperm(bsz, device=device)
            qn = max(1, bsz // 2)
            query_idx = perm[:qn]
            support_idx = perm[qn:]
        else:
            support_idx = torch.from_numpy(support_idx_np).long().to(device)
            query_idx = torch.from_numpy(query_idx_np).long().to(device)

        projector = LowRankSubspaceProjector(embed_dim=int(text_features.size(-1)), rank=int(tta_rank)).to(device)
        projector_state = copy.deepcopy(projector.state_dict())
        inner_opt = torch.optim.AdamW(projector.parameters(), lr=float(inner_lr))

        projector.train()
        for _ in range(int(inner_steps)):
            with torch.no_grad():
                eeg_s = eeg.index_select(0, support_idx)
                eeg_mask_s = eeg_mask.index_select(0, support_idx)
                src_key_padding_mask_s = (eeg_mask_s == 0)
                z_s = model(eeg_s, src_key_padding_mask=src_key_padding_mask_s)
                z_s = F.normalize(z_s.to(dtype=torch.float32), p=2, dim=-1)
                text_s = text_features.index_select(0, support_idx).to(dtype=torch.float32)
                text_s = F.normalize(text_s, p=2, dim=-1)

            z_s_cal = projector(z_s)
            logits = torch.matmul(z_s_cal, text_s.t())
            p = F.softmax(logits / 0.05, dim=-1)
            loss_entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1).mean()
            loss_proto = 1.0 - F.cosine_similarity(z_s_cal.mean(dim=0, keepdim=True), text_s.mean(dim=0, keepdim=True)).mean()
            loss_inner = loss_entropy + float(inner_lambda_proto) * loss_proto

            inner_opt.zero_grad()
            loss_inner.backward()
            inner_opt.step()

        projector.requires_grad_(False)
        projector.eval()

        with autocast():
            eeg_q = eeg.index_select(0, query_idx)
            eeg_mask_q = eeg_mask.index_select(0, query_idx)
            src_key_padding_mask_q = (eeg_mask_q == 0)
            z_q = model(eeg_q, src_key_padding_mask=src_key_padding_mask_q)
            z_q = F.normalize(z_q.to(dtype=torch.float32), p=2, dim=-1).to(dtype=z_q.dtype)
            z_q_cal = projector(z_q.to(dtype=torch.float32)).to(dtype=z_q.dtype)

            text_q = text_features.index_select(0, query_idx)
            text_q = F.normalize(text_q.to(dtype=torch.float32), p=2, dim=-1).to(dtype=text_q.dtype)

            text_bank = torch.cat([text_q, model.memory_bank.queue.detach().to(device=device, dtype=text_q.dtype)], dim=0)
            labels = torch.arange(z_q_cal.size(0), device=device)
            alignment_loss = criterion.forward_eeg_to_text_bank(z_q_cal, text_bank, labels=labels) * alignment_weight
            total_align += float(alignment_loss.detach().item())
            loss = alignment_loss

            unscaled_loss = loss
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        projector.load_state_dict(projector_state)

        window_count += 1
        window_loss_sum += float(unscaled_loss.detach().item())
        window_align_sum += float((alignment_loss.detach().item() / alignment_weight) if alignment_weight != 0 else alignment_loss.detach().item())

        total_loss += float(unscaled_loss.detach().item())
        model.memory_bank.enqueue(text_features_norm)
        postfix = {"loss": window_loss_sum / max(1, window_count), "align": window_align_sum / max(1, window_count)}
        pbar.set_postfix(postfix)

    denom_batches = max(1, len(dataloader))
    return (total_loss / denom_batches), (total_align / denom_batches)

def main(args):
    # 1. 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 准备 DataLoader
    if args.loso_test_subject is not None:
        train_loader, _ = get_loso_loaders(
            args.data_path,
            args.llm_name,
            test_subject=args.loso_test_subject,
            train_batch_size=args.batch_size,
            test_batch_size=max(1, args.batch_size),
            subject_wise_normalize=True,
        )
    else:
        train_loader = get_dataloader(
            args.data_path,
            args.llm_name,
            batch_size=args.batch_size,
            shuffle=True,
        )

    # 3. 加载冻结的 LLM (4-bit 量化)
    # 这将作为我们的语义潜空间目标
    llm, target_dim = load_frozen_llm(args.llm_name)
    
    # 4. 初始化 EEG Transformer 编码器
    model = EEGTransformerEncoder(
        input_dim=105, 
        d_model=512, 
        nhead=8, 
        num_layers=4, 
        output_dim=target_dim,
        enable_centering=args.use_centering,
    ).to(device)

    if args.init_checkpoint is not None:
        if not os.path.exists(args.init_checkpoint):
            raise ValueError(f"init_checkpoint not found: {args.init_checkpoint}")
        raw_state = torch.load(args.init_checkpoint, map_location="cpu")
        model_sd = model.state_dict()
        filtered = {}
        dropped = 0
        for k, v in raw_state.items():
            if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape") and v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                dropped += 1
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(
            f"Initialized from checkpoint: {args.init_checkpoint} "
            f"(dropped={dropped}, missing={len(missing)}, unexpected={len(unexpected)})"
        )

    # 5. 优化器与混合精度缩放器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    # 6. 训练循环
    print(f"\nStarting NeuroAlign Training for {args.epochs} epochs...")
    print(f"Gradient Accumulation Steps: {args.grad_accum}")

    checkpoint_root = os.path.join("checkpoints", str(args.version_tag))
    os.makedirs(checkpoint_root, exist_ok=True)
    print(f"Checkpoints will be saved to: {os.path.abspath(checkpoint_root)}")
    
    best_loss = float('inf')
    best_align = float("inf")
    early_stop_patience = 5
    epochs_no_improve = 0
    
    for epoch in range(1, args.epochs + 1):
        if str(args.version_tag).lower() in {"v7_0_meta", "v7.0_meta", "v7_meta"}:
            epoch_loss, epoch_align = train_one_epoch_meta(
                model,
                llm,
                train_loader,
                optimizer,
                scaler,
                device,
                epoch=epoch,
                total_epochs=args.epochs,
                accumulation_steps=args.grad_accum,
                use_centering=args.use_centering,
                centering_momentum=args.centering_momentum,
                temperature=args.temperature,
                margin=args.margin,
                inner_steps=args.inner_steps,
                inner_lr=args.inner_lr,
                inner_lambda_proto=args.inner_lambda_proto,
                tta_rank=args.tta_rank,
            )
        else:
            epoch_loss, epoch_align = train_one_epoch(
                model,
                llm,
                train_loader,
                optimizer,
                scaler,
                device,
                epoch=epoch,
                total_epochs=args.epochs,
                accumulation_steps=args.grad_accum,
                use_centering=args.use_centering,
                centering_momentum=args.centering_momentum,
                temperature=args.temperature,
                margin=args.margin,
            )
        
        print(f"Epoch {epoch}/{args.epochs} - Total Loss: {epoch_loss:.4f} | Align Loss: {epoch_align:.4f}")
        improved = bool(epoch_align < best_align)

        if args.loso_test_subject is not None:
            if str(args.version_tag).lower() in {"v7_0_meta", "v7.0_meta", "v7_meta"}:
                latest_path = os.path.join(checkpoint_root, "eeg_encoder_meta_latest.pth")
                if args.use_centering and getattr(model, "centroid_tracker", None) is not None and hasattr(model, "set_centering_delta"):
                    model.set_centering_delta(model.centroid_tracker.delta())
                torch.save(model.state_dict(), latest_path)
                print(f"Checkpoint saved: {latest_path}")

                if epoch_align < best_align:
                    best_align = epoch_align
                    best_path = os.path.join(checkpoint_root, "eeg_encoder_meta_best.pth")
                    torch.save(model.state_dict(), best_path)
                    print(f"Checkpoint saved: {best_path}")
                    print(f"New best model saved with align loss: {best_align:.4f}")
            else:
                version_tag = str(args.version_tag)
                subj = args.loso_test_subject.upper()
                parts = ["eeg_encoder", version_tag]
                parts.append(f"loso_{subj}")
                base_name = "_".join(parts)
                print(f"Run tag: {base_name}")
                latest_path = os.path.join(checkpoint_root, f"{base_name}_latest.pth")
                if args.use_centering and getattr(model, "centroid_tracker", None) is not None and hasattr(model, "set_centering_delta"):
                    model.set_centering_delta(model.centroid_tracker.delta())
                torch.save(model.state_dict(), latest_path)
                print(f"Checkpoint saved: {latest_path}")

                if epoch_align < best_align:
                    best_align = epoch_align
                    best_path = os.path.join(checkpoint_root, f"{base_name}_best.pth")
                    torch.save(model.state_dict(), best_path)
                    print(f"Checkpoint saved: {best_path}")
                    print(f"New best model saved with align loss: {best_align:.4f}")

                if epoch in {15, 20, 25, 30}:
                    snap_path = os.path.join(checkpoint_root, f"{base_name}_epoch{epoch:02d}.pth")
                    torch.save(model.state_dict(), snap_path)
                    print(f"Checkpoint saved: {snap_path}")
        else:
            save_checkpoint(model, "latest", path=checkpoint_root)
            if epoch_align < best_align:
                best_align = epoch_align
                if str(args.version_tag).lower() in {"v7_0_meta", "v7.0_meta", "v7_meta"}:
                    best_path = os.path.join(checkpoint_root, "eeg_encoder_meta_best.pth")
                    if args.use_centering and getattr(model, "centroid_tracker", None) is not None and hasattr(model, "set_centering_delta"):
                        model.set_centering_delta(model.centroid_tracker.delta())
                    torch.save(model.state_dict(), best_path)
                    print(f"Checkpoint saved: {best_path}")
                    print(f"New best model saved with align loss: {best_align:.4f}")
                else:
                    save_checkpoint(model, "best", path=checkpoint_root)
                    print(f"New best model saved with align loss: {best_align:.4f}")
        
        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"Early stopping: align loss has not improved for {early_stop_patience} consecutive epochs "
                    f"(best_align={best_align:.4f})."
                )
                break

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroAlign: Cross-Modal EEG-to-LLM Alignment")
    
    # 路径参数
    parser.add_argument("--data_path", type=str, default="./data/preprocessed/processed_zuco_cleaned.pt",
                        help="Path to the preprocessed .pt data")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HuggingFace model name for the LLM backbone")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step (keep small for 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--loso_test_subject", type=str, default=None, help="Leave-One-Subject-Out test subject ID (e.g., ZAB)")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Initialize model weights from a checkpoint (compatible load)")
    parser.add_argument("--use_centering", action="store_true", help="Enable v1.4 centering delta in AlignmentHead")
    parser.add_argument("--no_centering", dest="use_centering", action="store_false", help="Disable v1.4 centering delta in AlignmentHead")
    parser.set_defaults(use_centering=True)
    parser.add_argument("--centering_momentum", type=float, default=0.5, help="EMA momentum for centroid tracking (v1.4)")
    parser.add_argument("--version_tag", type=str, default="v1_5", help="Version tag used to group checkpoints (e.g., v1_5, v1_6)")
    parser.add_argument("--temperature", type=float, default=0.05, help="Initial temperature for contrastive loss (trainable logit_scale)")
    parser.add_argument("--margin", type=float, default=0.2, help="Similarity margin for InfoNCE positive pairs")
    parser.add_argument("--inner_steps", type=int, default=3)
    parser.add_argument("--inner_lr", type=float, default=1e-4)
    parser.add_argument("--inner_lambda_proto", type=float, default=1.0)
    parser.add_argument("--tta_rank", type=int, default=16)
    
    args = parser.parse_args()
    
    # 确保 checkpoints 目录存在
    os.makedirs("checkpoints", exist_ok=True)
    
    main(args)

import torch
import argparse
import os
from torch.cuda.amp import GradScaler
from src.model import EEGTransformerEncoder, load_frozen_llm
from src.data_loader import get_dataloader, get_loso_loaders
from src.trainer import train_one_epoch, save_checkpoint

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

    num_subjects = None
    if args.enable_dann:
        subject_to_idx = getattr(getattr(train_loader, "dataset", None), "subject_to_idx", None)
        if isinstance(subject_to_idx, dict) and len(subject_to_idx) > 0:
            num_subjects = len(subject_to_idx)
        else:
            raise ValueError(
                "DANN is enabled but no valid subject_to_idx mapping is available. "
                "Ensure your preprocessed .pt samples contain subject_id and you are loading multiple subjects."
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
        num_subjects=num_subjects,
        subject_hidden_dim=args.subject_hidden_dim,
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

    # 5. 优化器与混合精度缩放器（Differential LR）
    subject_params = []
    main_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("subject_classifier."):
            subject_params.append(param)
        else:
            main_params.append(param)

    param_groups = [
        {"params": main_params, "lr": args.lr, "weight_decay": 1e-4},
    ]
    if len(subject_params) > 0:
        param_groups.append({"params": subject_params, "lr": args.lr_subject, "weight_decay": 1e-4})

    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler()

    # 6. 训练循环
    print(f"\nStarting NeuroAlign Training for {args.epochs} epochs...")
    print(f"Gradient Accumulation Steps: {args.grad_accum}")
    
    best_loss = float('inf')
    best_align = float("inf")
    
    for epoch in range(1, args.epochs + 1):
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
        )
        
        print(f"Epoch {epoch}/{args.epochs} - Total Loss: {epoch_loss:.4f} | Align Loss: {epoch_align:.4f}")

        if args.loso_test_subject is not None:
            version_tag = "v1_4" if args.use_centering else "v1_3"
            subj = args.loso_test_subject.upper()
            parts = ["eeg_encoder", version_tag]
            if args.enable_dann:
                parts.append("dann")
            parts.append(f"loso_{subj}")
            base_name = "_".join(parts)
            print(f"Run tag: {base_name}")
            latest_path = os.path.join("checkpoints", f"{base_name}_latest.pth")
            if args.use_centering and getattr(model, "centroid_tracker", None) is not None and hasattr(model, "set_centering_delta"):
                model.set_centering_delta(model.centroid_tracker.delta())
            torch.save(model.state_dict(), latest_path)
            print(f"Checkpoint saved: {latest_path}")

            if epoch_align < best_align:
                best_align = epoch_align
                best_path = os.path.join("checkpoints", f"{base_name}_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"Checkpoint saved: {best_path}")
                print(f"New best model saved with align loss: {best_align:.4f}")

            if epoch in {15, 20, 25, 30}:
                snap_path = os.path.join("checkpoints", f"{base_name}_epoch{epoch:02d}.pth")
                torch.save(model.state_dict(), snap_path)
                print(f"Checkpoint saved: {snap_path}")
        else:
            save_checkpoint(model, "latest")
            if epoch_align < best_align:
                best_align = epoch_align
                save_checkpoint(model, "best")
                print(f"New best model saved with align loss: {best_align:.4f}")

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
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--loso_test_subject", type=str, default=None, help="Leave-One-Subject-Out test subject ID (e.g., ZAB)")
    parser.add_argument("--enable_dann", action="store_true", help="Enable DANN-style subject-adversarial training (v1.2)")
    parser.add_argument("--subject_hidden_dim", type=int, default=256, help="Hidden dim of subject classifier MLP (v1.2)")
    parser.add_argument("--lr_subject", type=float, default=1e-5, help="Learning rate for subject_classifier branch (v1.2)")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Initialize model weights from a checkpoint (compatible load)")
    parser.add_argument("--use_centering", action="store_true", help="Enable v1.4 centering delta in AlignmentHead")
    parser.add_argument("--no_centering", dest="use_centering", action="store_false", help="Disable v1.4 centering delta in AlignmentHead")
    parser.set_defaults(use_centering=True)
    parser.add_argument("--centering_momentum", type=float, default=0.9, help="EMA momentum for centroid tracking (v1.4)")
    
    args = parser.parse_args()
    
    # 确保 checkpoints 目录存在
    os.makedirs("checkpoints", exist_ok=True)
    
    main(args)

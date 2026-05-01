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

    # 2. 加载冻结的 LLM (4-bit 量化)
    # 这将作为我们的语义潜空间目标
    llm, target_dim = load_frozen_llm(args.llm_name)
    
    # 3. 初始化 EEG Transformer 编码器
    model = EEGTransformerEncoder(
        input_dim=105, 
        d_model=512, 
        nhead=8, 
        num_layers=4, 
        output_dim=target_dim
    ).to(device)

    # 4. 准备 DataLoader
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

    # 5. 优化器与混合精度缩放器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    # 6. 训练循环
    print(f"\nStarting NeuroAlign Training for {args.epochs} epochs...")
    print(f"Gradient Accumulation Steps: {args.grad_accum}")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(
            model, 
            llm, 
            train_loader, 
            optimizer, 
            scaler, 
            device, 
            accumulation_steps=args.grad_accum
        )
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {epoch_loss:.4f}")

        if args.loso_test_subject is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_path = os.path.join(
                    "checkpoints", f"eeg_encoder_v1_1_loso_{args.loso_test_subject.upper()}.pth"
                )
                torch.save(model.state_dict(), save_path)
                print(f"Checkpoint saved: {save_path}")
                print(f"New best model saved with loss: {best_loss:.4f}")
        else:
            save_checkpoint(model, "latest")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(model, "best")
                print(f"New best model saved with loss: {best_loss:.4f}")

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
    
    args = parser.parse_args()
    
    # 确保 checkpoints 目录存在
    os.makedirs("checkpoints", exist_ok=True)
    
    main(args)

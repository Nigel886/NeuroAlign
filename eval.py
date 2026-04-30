import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from src.model import EEGTransformerEncoder, load_frozen_llm
from src.data_loader import get_dataloader
import os

def run_retrieval_eval(model, llm, dataloader, device):
    model.eval()
    llm.eval()
    
    all_eeg_features = []
    all_text_features = []
    
    print("Extracting embeddings for retrieval...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            eeg = batch['eeg'].to(device)
            eeg_mask = batch['eeg_mask'].to(device)
            
            # 1. EEG 嵌入提取
            src_key_padding_mask = (eeg_mask == 0)
            eeg_feat = model(eeg, src_key_padding_mask=src_key_padding_mask)
            eeg_feat = F.normalize(eeg_feat, p=2, dim=-1)
            
            # 2. Text 嵌入提取 (LLM)
            outputs = llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            text_mask = attention_mask.unsqueeze(-1).float()
            text_feat = torch.sum(last_hidden_state * text_mask, dim=1) / torch.sum(text_mask, dim=1).clamp(min=1e-9)
            text_feat = F.normalize(text_feat, p=2, dim=-1)
            
            all_eeg_features.append(eeg_feat.cpu())
            all_text_features.append(text_feat.cpu())
            
    all_eeg_features = torch.cat(all_eeg_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    
    # 计算相似度矩阵 (N, N)
    similarity = torch.matmul(all_eeg_features, all_text_features.t())
    
    # 计算 Top-k 准确率
    num_samples = similarity.size(0)
    labels = torch.arange(num_samples)
    
    # 检索任务：给定 EEG，找 Text
    _, indices = similarity.topk(5, dim=1)
    top1 = (indices[:, 0] == labels).float().mean().item()
    top5 = (indices == labels.view(-1, 1)).any(dim=1).float().mean().item()
    
    print(f"\nRetrieval Results (EEG -> Text):")
    print(f"Top-1 Accuracy: {top1*100:.2f}%")
    print(f"Top-5 Accuracy: {top5*100:.2f}%")
    
    return all_eeg_features.numpy(), all_text_features.numpy()

def visualize_tsne(eeg_feats, text_feats, output_path="tsne_visualization.png"):
    print("\nRunning t-SNE visualization...")
    num_samples = eeg_feats.shape[0]
    
    # 合并特征进行训练，确保在同一空间
    combined_feats = np.concatenate([eeg_feats, text_feats], axis=0)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    reduced_feats = tsne.fit_transform(combined_feats)
    
    eeg_reduced = reduced_feats[:num_samples]
    text_reduced = reduced_feats[num_samples:]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(eeg_reduced[:, 0], eeg_reduced[:, 1], c='blue', label='EEG Embeddings', alpha=0.6, s=15)
    plt.scatter(text_reduced[:, 0], text_reduced[:, 1], c='red', label='Text Embeddings', alpha=0.6, s=15)
    
    # 连线：展示同一对的对应关系（可选，如果样本太多会乱）
    if num_samples < 50:
        for i in range(num_samples):
            plt.plot([eeg_reduced[i, 0], text_reduced[i, 0]], 
                     [eeg_reduced[i, 1], text_reduced[i, 1]], 'gray', alpha=0.2)
            
    plt.title("NeuroAlign: t-SNE Visualization of EEG and Text Latent Space")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    print(f"t-SNE plot saved to {output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载 LLM (4-bit)
    llm, target_dim = load_frozen_llm()
    
    # 2. 加载训练好的 EEG 编码器
    model = EEGTransformerEncoder(output_dim=target_dim).to(device)
    checkpoint_path = "checkpoints/eeg_encoder_epoch_latest.pth" # 假设最新权重
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found. Running evaluation with random weights.")
    
    # 3. 加载测试集
    test_pt = "./data/preprocessed/processed_zuco_cleaned.pt"
    dataloader = get_dataloader(test_pt, "meta-llama/Meta-Llama-3-8B-Instruct", batch_size=4, shuffle=False)
    
    # 4. 执行评估
    eeg_feats, text_feats = run_retrieval_eval(model, llm, dataloader, device)
    
    # 5. 可视化
    visualize_tsne(eeg_feats, text_feats)

if __name__ == "__main__":
    main()

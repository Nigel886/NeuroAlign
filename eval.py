import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from src.model import EEGTransformerEncoder, load_frozen_llm
from src.data_loader import get_dataloader
import os
import argparse

def _compute_topk(similarity, k, query_indices):
    labels = torch.arange(similarity.size(0), device=similarity.device)
    _, indices = similarity.topk(k, dim=1)
    q = query_indices.to(similarity.device)
    top1 = (indices[q, 0] == labels[q]).float().mean().item()
    topk = (indices[q] == labels[q].view(-1, 1)).any(dim=1).float().mean().item()
    return top1, topk

def run_retrieval_eval(model, llm, dataloader, device, test_subject_id=None):
    model.eval()
    llm.eval()
    
    all_eeg_features = []
    all_text_features = []
    all_subject_ids = []
    
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
            all_subject_ids.extend(batch.get("subject_ids", [None] * eeg_feat.size(0)))
            
    all_eeg_features = torch.cat(all_eeg_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    all_eeg_features = F.normalize(all_eeg_features, p=2, dim=-1)
    all_text_features = F.normalize(all_text_features, p=2, dim=-1)
    
    # 计算相似度矩阵 (N, N)
    similarity = torch.matmul(all_eeg_features, all_text_features.t())
    
    num_samples = similarity.size(0)
    all_idx = torch.arange(num_samples, device=similarity.device)
    top1_all, top5_all = _compute_topk(similarity, 5, all_idx)
    print(f"\nRetrieval Results (EEG -> Text):")
    print(f"Top-1 Accuracy (All): {top1_all*100:.2f}%")
    print(f"Top-5 Accuracy (All): {top5_all*100:.2f}%")

    test_subject = str(test_subject_id).upper() if test_subject_id is not None else None
    if test_subject and any(s is not None for s in all_subject_ids):
        subj = np.array([str(s).upper() if s is not None else "UNK" for s in all_subject_ids], dtype=object)
        unseen_mask = subj == test_subject
        seen_mask = ~unseen_mask
        if unseen_mask.any():
            unseen_idx = torch.from_numpy(np.where(unseen_mask)[0]).long()
            top1_u, top5_u = _compute_topk(similarity, 5, unseen_idx)
            print(f"Top-1 Accuracy (Unseen={test_subject}): {top1_u*100:.2f}%")
            print(f"Top-5 Accuracy (Unseen={test_subject}): {top5_u*100:.2f}%")
        if seen_mask.any():
            seen_idx = torch.from_numpy(np.where(seen_mask)[0]).long()
            top1_s, top5_s = _compute_topk(similarity, 5, seen_idx)
            print(f"Top-1 Accuracy (Seen): {top1_s*100:.2f}%")
            print(f"Top-5 Accuracy (Seen): {top5_s*100:.2f}%")
    
    return all_eeg_features.numpy(), all_text_features.numpy(), all_subject_ids

def _run_tsne(eeg_feats, text_feats):
    num_samples = eeg_feats.shape[0]
    combined_feats = np.concatenate([eeg_feats, text_feats], axis=0)
    n_total = combined_feats.shape[0]
    perplexity = int(min(30, max(5, (n_total - 1) // 3)))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto")
    reduced_feats = tsne.fit_transform(combined_feats)
    eeg_reduced = reduced_feats[:num_samples]
    text_reduced = reduced_feats[num_samples:]
    return eeg_reduced, text_reduced

def visualize_tsne_loso(eeg_feats, text_feats, subject_ids, test_subject_id=None, output_prefix="tsne"):
    print("\nRunning t-SNE visualization...")
    eeg_2d, text_2d = _run_tsne(eeg_feats, text_feats)
    subj = np.array([str(s).upper() if s is not None else "UNK" for s in subject_ids], dtype=object)

    plt.figure(figsize=(10, 8))
    unique_subjects = sorted(set(subj.tolist()))
    cmap = plt.get_cmap("tab20")
    for i, sid in enumerate(unique_subjects):
        mask = subj == sid
        plt.scatter(
            eeg_2d[mask, 0],
            eeg_2d[mask, 1],
            color=cmap(i % 20),
            alpha=0.6,
            s=14,
            label=f"EEG {sid}",
        )
    plt.scatter(text_2d[:, 0], text_2d[:, 1], c="red", label="Text", alpha=0.35, s=12)
    plt.title("NeuroAlign LOSO: EEG embeddings colored by subject")
    plt.grid(True, alpha=0.3)
    plt.legend(markerscale=1.2, fontsize=8, ncol=2)
    out_all = f"{output_prefix}_subjects.png"
    plt.savefig(out_all, dpi=200, bbox_inches="tight")
    print(f"t-SNE plot saved to {out_all}")

    test_subject = str(test_subject_id).upper() if test_subject_id is not None else None
    if test_subject:
        unseen_mask = subj == test_subject
        plt.figure(figsize=(10, 8))
        plt.scatter(eeg_2d[:, 0], eeg_2d[:, 1], c="lightgray", alpha=0.25, s=12, label="EEG Seen (background)")
        if unseen_mask.any():
            plt.scatter(
                eeg_2d[unseen_mask, 0],
                eeg_2d[unseen_mask, 1],
                c="blue",
                alpha=0.8,
                s=16,
                label=f"EEG Unseen={test_subject}",
            )
        plt.scatter(text_2d[:, 0], text_2d[:, 1], c="red", alpha=0.25, s=12, label="Text")
        plt.title(f"NeuroAlign LOSO: Unseen subject highlight ({test_subject})")
        plt.grid(True, alpha=0.3)
        plt.legend(markerscale=1.2, fontsize=9)
        out_unseen = f"{output_prefix}_unseen_{test_subject}.png"
        plt.savefig(out_unseen, dpi=200, bbox_inches="tight")
        print(f"t-SNE plot saved to {out_unseen}")

def main():
    parser = argparse.ArgumentParser(description="NeuroAlign Evaluation (Retrieval + LOSO analysis)")
    parser.add_argument("--data_path", type=str, default="./data/preprocessed/processed_zuco_cleaned.pt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/eeg_encoder_epoch_best.pth")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_subject_id", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="tsne")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm, target_dim = load_frozen_llm(args.llm_name)

    model = EEGTransformerEncoder(output_dim=target_dim).to(device)
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
    else:
        print(f"Warning: checkpoint not found: {args.checkpoint}. Running evaluation with random weights.")

    dataloader = get_dataloader(
        args.data_path,
        args.tokenizer_name,
        batch_size=args.batch_size,
        shuffle=False,
        split="all",
        subject_wise_normalize=True,
    )

    eeg_feats, text_feats, subject_ids = run_retrieval_eval(
        model, llm, dataloader, device, test_subject_id=args.test_subject_id
    )
    visualize_tsne_loso(
        eeg_feats,
        text_feats,
        subject_ids,
        test_subject_id=args.test_subject_id,
        output_prefix=args.output_prefix,
    )

if __name__ == "__main__":
    main()

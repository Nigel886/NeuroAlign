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
import json
import datetime
import re

class _CentroidTracker(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("eeg_centroid", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("text_centroid", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("_initialized", torch.tensor(False))

def _infer_centroid_dim_from_state(state_dict):
    if not isinstance(state_dict, dict):
        return None
    v = state_dict.get("centroid_tracker.eeg_centroid", None)
    if v is not None and hasattr(v, "shape") and len(v.shape) == 1:
        return int(v.shape[0])
    v = state_dict.get("centroid_tracker.text_centroid", None)
    if v is not None and hasattr(v, "shape") and len(v.shape) == 1:
        return int(v.shape[0])
    return None

def _filter_state_dict_for_model(model, state_dict):
    model_sd = model.state_dict()
    filtered = {}
    dropped = 0
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape") and v.shape == model_sd[k].shape:
            filtered[k] = v
        else:
            dropped += 1
    return filtered, dropped

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
        tracker = getattr(model, "centroid_tracker", None)
        delta = None
        centering = getattr(model, "centering", None)
        if centering is not None and hasattr(centering, "delta"):
            if torch.is_tensor(centering.delta) and centering.delta.numel() > 0:
                delta = centering.delta.to(device=device, dtype=torch.float32)
        if tracker is not None and hasattr(tracker, "text_centroid") and hasattr(tracker, "eeg_centroid"):
            if delta is None:
                delta = (tracker.text_centroid - tracker.eeg_centroid).to(device=device, dtype=torch.float32)
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            eeg = batch['eeg'].to(device)
            eeg_mask = batch['eeg_mask'].to(device)
            
            # 1. EEG 嵌入提取
            src_key_padding_mask = (eeg_mask == 0)
            if delta is not None and getattr(model, "centering", None) is not None:
                z_semantic, _ = model(eeg, src_key_padding_mask=src_key_padding_mask, centering_delta=delta)
            else:
                z_semantic, _ = model(eeg, src_key_padding_mask=src_key_padding_mask)
            eeg_feat = z_semantic.detach().to(dtype=torch.float32)
            if delta is not None and getattr(model, "centering", None) is None:
                eeg_feat = eeg_feat + delta
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
    all_eeg_features = F.normalize(all_eeg_features.to(dtype=torch.float32), p=2, dim=-1)
    all_text_features = F.normalize(all_text_features.to(dtype=torch.float32), p=2, dim=-1)

    test_subject = str(test_subject_id).upper() if test_subject_id is not None else None
    if test_subject and any(s is not None for s in all_subject_ids):
        subj = np.array([str(s).upper() if s is not None else "UNK" for s in all_subject_ids], dtype=object)
        unseen_mask = subj == test_subject
        if unseen_mask.any():
            eeg_centroid_test = all_eeg_features[unseen_mask].to(dtype=torch.float32).mean(dim=0)
            text_centroid_test = all_text_features[unseen_mask].to(dtype=torch.float32).mean(dim=0)
            delta_test = text_centroid_test - eeg_centroid_test
            all_eeg_features[unseen_mask] = F.normalize(
                all_eeg_features[unseen_mask].to(dtype=torch.float32) + delta_test,
                p=2,
                dim=-1,
            ).to(dtype=all_eeg_features.dtype)
        else:
            eeg_centroid_test = all_eeg_features.to(dtype=torch.float32).mean(dim=0)
            text_centroid_test = all_text_features.to(dtype=torch.float32).mean(dim=0)
            delta_test = text_centroid_test - eeg_centroid_test
            all_eeg_features = F.normalize(all_eeg_features.to(dtype=torch.float32) + delta_test, p=2, dim=-1).to(dtype=all_eeg_features.dtype)
    else:
        eeg_centroid_test = all_eeg_features.to(dtype=torch.float32).mean(dim=0)
        text_centroid_test = all_text_features.to(dtype=torch.float32).mean(dim=0)
        delta_test = text_centroid_test - eeg_centroid_test
        all_eeg_features = F.normalize(all_eeg_features.to(dtype=torch.float32) + delta_test, p=2, dim=-1).to(dtype=all_eeg_features.dtype)
    
    # 计算相似度矩阵 (N, N)
    similarity = torch.matmul(all_eeg_features, all_text_features.t())
    
    num_samples = similarity.size(0)
    all_idx = torch.arange(num_samples, device=similarity.device)
    top1_all, top5_all = _compute_topk(similarity, 5, all_idx)
    print(f"\nRetrieval Results (EEG -> Text):")
    print(f"Top-1 Accuracy (All): {top1_all*100:.2f}%")
    print(f"Top-5 Accuracy (All): {top5_all*100:.2f}%")
    metrics = {
        "num_samples": int(num_samples),
        "top1_all": float(top1_all),
        "top5_all": float(top5_all),
    }

    if test_subject and any(s is not None for s in all_subject_ids):
        subj = np.array([str(s).upper() if s is not None else "UNK" for s in all_subject_ids], dtype=object)
        unseen_mask = subj == test_subject
        seen_mask = ~unseen_mask
        if unseen_mask.any():
            unseen_idx = torch.from_numpy(np.where(unseen_mask)[0]).long()
            top1_u, top5_u = _compute_topk(similarity, 5, unseen_idx)
            print(f"Top-1 Accuracy (Unseen={test_subject}): {top1_u*100:.2f}%")
            print(f"Top-5 Accuracy (Unseen={test_subject}): {top5_u*100:.2f}%")
            metrics["top1_unseen"] = float(top1_u)
            metrics["top5_unseen"] = float(top5_u)
            metrics["unseen_subject_id"] = str(test_subject)
            metrics["num_unseen"] = int(unseen_idx.numel())
        if seen_mask.any():
            seen_idx = torch.from_numpy(np.where(seen_mask)[0]).long()
            top1_s, top5_s = _compute_topk(similarity, 5, seen_idx)
            print(f"Top-1 Accuracy (Seen): {top1_s*100:.2f}%")
            print(f"Top-5 Accuracy (Seen): {top5_s*100:.2f}%")
            metrics["top1_seen"] = float(top1_s)
            metrics["top5_seen"] = float(top5_s)
            metrics["num_seen"] = int(seen_idx.numel())
    
    return all_eeg_features.numpy(), all_text_features.numpy(), all_subject_ids, metrics

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

def _safe_name(s: str) -> str:
    invalid = '<>:"/\\|?*'
    out = "".join("_" if c in invalid else c for c in s)
    out = out.replace(" ", "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("._")

def _infer_version_tag_from_checkpoint_path(checkpoint_path: str):
    if not checkpoint_path:
        return None
    base = os.path.basename(str(checkpoint_path))
    m = re.search(r"(v\d+(?:[._]\d+){0,3})", base, flags=re.IGNORECASE)
    if not m:
        return None
    v = m.group(1).lower().replace(".", "_")
    return _safe_name(v)

def _has_version_tag(s: str) -> bool:
    if not s:
        return False
    return re.search(r"v\d+", str(s), flags=re.IGNORECASE) is not None

def _make_output_dir(out_dir, output_prefix, test_subject_id, checkpoint_path):
    if out_dir:
        base = out_dir
    else:
        date_str = datetime.date.today().strftime("%Y%m%d")
        tag = str(output_prefix) if output_prefix else "eval"
        if tag.lower().startswith("tsne_"):
            tag = tag[5:]
        if tag == "tsne":
            tag = "eval"
        if not _has_version_tag(tag):
            ckpt_v = _infer_version_tag_from_checkpoint_path(checkpoint_path)
            if ckpt_v:
                tag = ckpt_v if tag == "eval" else f"{tag}_{ckpt_v}"
        parts = [date_str, tag]
        test_subject = str(test_subject_id).upper() if test_subject_id else None
        if test_subject and test_subject not in tag.upper():
            parts.append(f"LOSO_{test_subject}")
        run_name = _safe_name("_".join(parts))
        base = os.path.join("outputs", run_name)

    base = os.path.abspath(base)
    candidate = base
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    os.makedirs(candidate, exist_ok=True)
    return candidate

def main():
    parser = argparse.ArgumentParser(description="NeuroAlign Evaluation (Retrieval + LOSO analysis)")
    parser.add_argument("--data_path", type=str, default="./data/preprocessed/processed_zuco_cleaned.pt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/eeg_encoder_epoch_best.pth")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_subject_id", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="tsne")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm, target_dim = load_frozen_llm(args.llm_name)

    state = None
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
    model = EEGTransformerEncoder(
        output_dim=target_dim,
    ).to(device)
    centroid_dim = _infer_centroid_dim_from_state(state)
    if centroid_dim is not None:
        model.centroid_tracker = _CentroidTracker(dim=centroid_dim).to(device)

    if isinstance(state, dict):
        filtered, dropped = _filter_state_dict_for_model(model, state)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint} (dropped={dropped}, missing={len(missing)}, unexpected={len(unexpected)})")
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

    output_dir = _make_output_dir(args.out_dir, args.output_prefix, args.test_subject_id, args.checkpoint)
    print(f"\nOutputs will be saved to: {output_dir}")

    eeg_feats, text_feats, subject_ids, metrics = run_retrieval_eval(
        model, llm, dataloader, device, test_subject_id=args.test_subject_id
    )
    metrics_payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_version": _infer_version_tag_from_checkpoint_path(args.checkpoint),
        "data_path": os.path.abspath(args.data_path),
        "llm_name": args.llm_name,
        "tokenizer_name": args.tokenizer_name,
        "batch_size": int(args.batch_size),
        "test_subject_id": args.test_subject_id,
        "metrics": metrics,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    visualize_tsne_loso(
        eeg_feats,
        text_feats,
        subject_ids,
        test_subject_id=args.test_subject_id,
        output_prefix=os.path.join(output_dir, args.output_prefix),
    )

if __name__ == "__main__":
    main()

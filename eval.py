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
import copy

class _CentroidTracker(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("eeg_centroid", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("text_centroid", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("_initialized", torch.tensor(False))

def _infer_centroid_dim_from_state(state_dict):
    if not isinstance(state_dict, dict):
        return None
    sd = state_dict.get("state_dict", state_dict) if isinstance(state_dict, dict) else None
    if not isinstance(sd, dict):
        return None
    v = sd.get("centroid_tracker.eeg_centroid", None)
    if v is not None and hasattr(v, "shape") and len(v.shape) == 1:
        return int(v.shape[0])
    v = sd.get("centroid_tracker.text_centroid", None)
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

def sinkhorn_alignment(unseen_features, seen_features, reg=0.05, max_iter=100):
    norm_u = F.normalize(unseen_features, p=2, dim=-1)
    norm_s = F.normalize(seen_features, p=2, dim=-1)
    C = 1.0 - torch.mm(norm_u, norm_s.t())

    K = torch.exp(-C / float(reg))
    N_u, N_s = K.shape
    u = torch.ones(N_u, device=K.device, dtype=K.dtype) / float(N_u)
    v = torch.ones(N_s, device=K.device, dtype=K.dtype) / float(N_s)
    eps = torch.finfo(K.dtype).eps

    for _ in range(int(max_iter)):
        Kv = torch.mv(K, v).clamp(min=eps)
        u = (1.0 / float(N_u)) / Kv
        Ktu = torch.mv(K.t(), u).clamp(min=eps)
        v = (1.0 / float(N_s)) / Ktu

    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    transformed_features = torch.mm(P, seen_features) * float(N_u)
    return transformed_features

def sinkhorn_alignment(unseen_features, seen_features, reg=0.05, max_iter=100):
    """
    v3.1 最优传输(OT)测试时自适应校准核心
    利用无监督 Sinkhorn 算法，强行将未见域的扭曲流形非线性揉捏、贴合到已知域的完美拓扑上
    """
    device = unseen_features.device
    # 1. 计算两组特征之间的余弦距离代价矩阵 C [N_unseen, N_seen]
    norm_u = F.normalize(unseen_features, p=2, dim=-1)
    norm_s = F.normalize(seen_features, p=2, dim=-1)
    C = 1.0 - torch.mm(norm_u, norm_s.t()) 
    
    # 2. Sinkhorn 迭代平衡
    K = torch.exp(-C / reg)
    N_u, N_s = K.shape
    u = torch.ones(N_u, device=device) / N_u
    v = torch.ones(N_s, device=device) / N_s
    
    for _ in range(max_iter):
        u = (1.0 / N_u) / torch.clamp(torch.mv(K, v), min=1e-9)
        v = (1.0 / N_s) / torch.clamp(torch.mv(K.t(), u), min=1e-9)
        
    # 3. 计算传输矩阵 P 并通过重心投影重塑未见域特征
    P = u.unsqueeze(1) * K * v.unsqueeze(0) 
    transformed_features = torch.mm(P, seen_features) * N_u
    return transformed_features

def configure_tent_model(model):
    model.requires_grad_(False)
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)
            m.train()
    return model

class LowRankSubspaceProjector(torch.nn.Module):
    def __init__(self, dim=4096, rank=16):
        super().__init__()
        self.down = torch.nn.Linear(int(dim), int(rank), bias=False)
        self.up = torch.nn.Linear(int(rank), int(dim), bias=False)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))
    
def run_retrieval_eval(
    model,
    llm,
    dataloader,
    device,
    test_subject_id=None,
    tta_mode="none",
    reg=0.05,
    lr_tta=1e-5,
    lambda_anchor=1.0,
    lambda_elastic=10.0,
    lambda_proto=5.0,
    tta_rank=16,
    train_text_centroid=None,
):
    model.eval()
    llm.eval()
    
    all_eeg_features = []
    all_text_features = []
    all_subject_ids = []
    
    print("Extracting embeddings for retrieval...")
    if str(tta_mode).lower() in {"v6_0_cpa_lsr", "cpa_lsr"}:
        if test_subject_id is None:
            raise ValueError("--tta_mode v6_0_cpa_lsr requires --test_subject_id to define the unseen subject.")
        test_subject_id = str(test_subject_id).upper()
        if train_text_centroid is None:
            raise ValueError("v6_0_cpa_lsr requires train_text_centroid loaded from checkpoint (centroid_tracker.text_centroid).")
        train_text_centroid = train_text_centroid.to(device=device, dtype=torch.float32)

        model.requires_grad_(False)
        model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                text_mask = attention_mask.unsqueeze(-1).float()
                text_feat = torch.sum(last_hidden_state * text_mask, dim=1) / torch.sum(text_mask, dim=1).clamp(min=1e-9)
                text_feat = F.normalize(text_feat, p=2, dim=-1)
                all_text_features.append(text_feat.detach().to(dtype=torch.float32).cpu())
                all_subject_ids.extend(batch.get("subject_ids", [None] * int(text_feat.size(0))))
        all_text_features = torch.cat(all_text_features, dim=0)
        all_text_features = F.normalize(all_text_features.to(dtype=torch.float32), p=2, dim=-1)
        text_bank = all_text_features.to(device=device, dtype=torch.float32)

        dim = int(text_bank.size(-1))
        projector = LowRankSubspaceProjector(dim=dim, rank=int(tta_rank)).to(device)
        optimizer = torch.optim.Adam(projector.parameters(), lr=float(lr_tta))

        for _ in range(1):
            for batch in tqdm(dataloader, desc="CPA-LSR"):
                eeg = batch["eeg"].to(device)
                eeg_mask = batch["eeg_mask"].to(device)
                subject_ids = batch.get("subject_ids", None)
                if subject_ids is None:
                    raise ValueError("--tta_mode v6_0_cpa_lsr requires subject_ids in batch.")
                subj = np.array([str(s).upper() if s is not None else "UNK" for s in subject_ids], dtype=object)
                unseen_mask_batch = subj == test_subject_id
                if not unseen_mask_batch.any():
                    continue
                idx = torch.from_numpy(np.where(unseen_mask_batch)[0]).long().to(device)
                eeg_u = eeg.index_select(0, idx)
                eeg_mask_u = eeg_mask.index_select(0, idx)

                src_key_padding_mask = (eeg_mask_u == 0)
                with torch.no_grad():
                    raw = model(eeg_u, src_key_padding_mask=src_key_padding_mask)
                    raw = F.normalize(raw.to(dtype=torch.float32), p=2, dim=-1)
                z_calibrated = projector(raw)
                logits = torch.mm(z_calibrated, text_bank.t())
                p = F.softmax(logits / 0.05, dim=-1)
                loss_entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1).mean()
                current_unseen_centroid = z_calibrated.mean(dim=0)
                loss_proto = 1.0 - F.cosine_similarity(
                    current_unseen_centroid.unsqueeze(0),
                    train_text_centroid.unsqueeze(0),
                ).mean()
                loss_total = loss_entropy + float(lambda_proto) * loss_proto
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

        projector.eval()
        all_eeg_features = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting EEG (CPA-LSR)"):
                eeg = batch["eeg"].to(device)
                eeg_mask = batch["eeg_mask"].to(device)
                subject_ids = batch.get("subject_ids", None)
                if subject_ids is None:
                    raise ValueError("--tta_mode v6_0_cpa_lsr requires subject_ids in batch.")
                subj = np.array([str(s).upper() if s is not None else "UNK" for s in subject_ids], dtype=object)
                unseen_mask_batch = subj == test_subject_id

                src_key_padding_mask = (eeg_mask == 0)
                raw = model(eeg, src_key_padding_mask=src_key_padding_mask)
                raw = F.normalize(raw.to(dtype=torch.float32), p=2, dim=-1)
                if unseen_mask_batch.any():
                    idx = torch.from_numpy(np.where(unseen_mask_batch)[0]).long().to(device)
                    raw_u = raw.index_select(0, idx)
                    raw_cal = projector(raw_u)
                    raw.index_copy_(0, idx, F.normalize(raw_cal.to(dtype=torch.float32), p=2, dim=-1))
                all_eeg_features.append(raw.cpu())
        all_eeg_features = torch.cat(all_eeg_features, dim=0)
        all_eeg_features = F.normalize(all_eeg_features.to(dtype=torch.float32), p=2, dim=-1)

    elif str(tta_mode).lower() in {"v5_0_sga", "sga"}:
        if test_subject_id is None:
            raise ValueError("--tta_mode v5_0_sga requires --test_subject_id to define the unseen subject.")
        test_subject_id = str(test_subject_id).upper()
        if train_text_centroid is None:
            raise ValueError("v5_0_sga requires train_text_centroid loaded from checkpoint (centroid_tracker.text_centroid).")
        train_text_centroid = train_text_centroid.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                text_mask = attention_mask.unsqueeze(-1).float()
                text_feat = torch.sum(last_hidden_state * text_mask, dim=1) / torch.sum(text_mask, dim=1).clamp(min=1e-9)
                text_feat = F.normalize(text_feat, p=2, dim=-1)
                all_text_features.append(text_feat.detach().to(dtype=torch.float32).cpu())
                all_subject_ids.extend(batch.get("subject_ids", [None] * int(text_feat.size(0))))
        all_text_features = torch.cat(all_text_features, dim=0)
        all_text_features = F.normalize(all_text_features.to(dtype=torch.float32), p=2, dim=-1)
        text_bank = all_text_features.to(device=device, dtype=torch.float32)

        sga_model = configure_tent_model(copy.deepcopy(model).to(device))
        init_ln_params = {name: p.clone().detach() for name, p in sga_model.named_parameters() if "layer_norm" in name.lower()}
        if not init_ln_params:
            for module_name, m in sga_model.named_modules():
                if isinstance(m, torch.nn.LayerNorm):
                    for p_name, p in m.named_parameters(recurse=False):
                        full = f"{module_name}.{p_name}" if module_name else p_name
                        init_ln_params[full] = p.clone().detach()
        init_ln_params = {k: v.to(device=device, dtype=torch.float32) for k, v in init_ln_params.items()}

        sga_params = [p for p in sga_model.parameters() if p.requires_grad]
        if not sga_params:
            raise ValueError("No trainable LayerNorm parameters found for v5_0_sga.")
        optimizer = torch.optim.Adam(sga_params, lr=float(lr_tta))

        for _ in range(1):
            for batch in tqdm(dataloader, desc="SGA"):
                eeg = batch["eeg"].to(device)
                eeg_mask = batch["eeg_mask"].to(device)
                subject_ids = batch.get("subject_ids", [None] * int(eeg.size(0)))
                subj = np.array([str(s).upper() if s is not None else "UNK" for s in subject_ids], dtype=object)
                unseen_mask_batch = subj == test_subject_id
                if not unseen_mask_batch.any():
                    continue
                idx = torch.from_numpy(np.where(unseen_mask_batch)[0]).long().to(device)
                eeg_u = eeg.index_select(0, idx)
                eeg_mask_u = eeg_mask.index_select(0, idx)

                src_key_padding_mask = (eeg_mask_u == 0)
                eeg_features = sga_model(eeg_u, src_key_padding_mask=src_key_padding_mask)
                eeg_features = F.normalize(eeg_features.to(dtype=torch.float32), p=2, dim=-1)

                logits = torch.mm(eeg_features, text_bank.t())
                p = F.softmax(logits / 0.05, dim=-1)
                loss_entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1).mean()

                current_unseen_centroid = eeg_features.mean(dim=0)
                loss_anchor = 1.0 - F.cosine_similarity(
                    current_unseen_centroid.unsqueeze(0),
                    train_text_centroid.unsqueeze(0),
                ).mean()

                loss_elastic = 0.0
                for name, p_ln in sga_model.named_parameters():
                    if name in init_ln_params:
                        loss_elastic = loss_elastic + torch.sum((p_ln.to(dtype=torch.float32) - init_ln_params[name]) ** 2)

                loss_total = loss_entropy + float(lambda_anchor) * loss_anchor + float(lambda_elastic) * loss_elastic
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

        sga_model.eval()
        all_eeg_features = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting EEG (SGA)"):
                eeg = batch["eeg"].to(device)
                eeg_mask = batch["eeg_mask"].to(device)
                src_key_padding_mask = (eeg_mask == 0)
                z_semantic = sga_model(eeg, src_key_padding_mask=src_key_padding_mask)
                eeg_feat = F.normalize(z_semantic.detach().to(dtype=torch.float32), p=2, dim=-1)
                all_eeg_features.append(eeg_feat.cpu())
        all_eeg_features = torch.cat(all_eeg_features, dim=0)
        all_eeg_features = F.normalize(all_eeg_features.to(dtype=torch.float32), p=2, dim=-1)

    elif str(tta_mode).lower() in {"v4_0_tent", "tent"}:
        if test_subject_id is None:
            raise ValueError("--tta_mode v4_0_tent requires --test_subject_id to define the unseen subject.")
        test_subject_id = str(test_subject_id).upper()

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                text_mask = attention_mask.unsqueeze(-1).float()
                text_feat = torch.sum(last_hidden_state * text_mask, dim=1) / torch.sum(text_mask, dim=1).clamp(min=1e-9)
                text_feat = F.normalize(text_feat, p=2, dim=-1)
                all_text_features.append(text_feat.detach().to(dtype=torch.float32).cpu())
                all_subject_ids.extend(batch.get("subject_ids", [None] * int(text_feat.size(0))))
        all_text_features = torch.cat(all_text_features, dim=0)
        all_text_features = F.normalize(all_text_features.to(dtype=torch.float32), p=2, dim=-1)
        text_bank = all_text_features.to(device=device, dtype=torch.float32)

        tent_model = configure_tent_model(copy.deepcopy(model).to(device))
        tent_params = [p for p in tent_model.parameters() if p.requires_grad]
        if not tent_params:
            raise ValueError("No trainable LayerNorm parameters found for TENT.")
        optimizer = torch.optim.Adam(tent_params, lr=float(lr_tta))

        for _ in range(1):
            for batch in tqdm(dataloader, desc="TENT"):
                eeg = batch["eeg"].to(device)
                eeg_mask = batch["eeg_mask"].to(device)
                subject_ids = batch.get("subject_ids", [None] * int(eeg.size(0)))
                subj = np.array([str(s).upper() if s is not None else "UNK" for s in subject_ids], dtype=object)
                unseen_mask_batch = subj == test_subject_id
                if not unseen_mask_batch.any():
                    continue
                idx = torch.from_numpy(np.where(unseen_mask_batch)[0]).long().to(device)
                eeg_u = eeg.index_select(0, idx)
                eeg_mask_u = eeg_mask.index_select(0, idx)

                src_key_padding_mask = (eeg_mask_u == 0)
                z_semantic = tent_model(eeg_u, src_key_padding_mask=src_key_padding_mask)
                z_semantic = F.normalize(z_semantic.to(dtype=torch.float32), p=2, dim=-1)
                logits = torch.matmul(z_semantic, text_bank.t())
                p = F.softmax(logits / 0.05, dim=-1)
                loss = -torch.sum(p * torch.log(p + 1e-9), dim=-1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        tent_model.eval()
        all_eeg_features = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting EEG (TENT)"):
                eeg = batch["eeg"].to(device)
                eeg_mask = batch["eeg_mask"].to(device)
                src_key_padding_mask = (eeg_mask == 0)
                z_semantic = tent_model(eeg, src_key_padding_mask=src_key_padding_mask)
                eeg_feat = F.normalize(z_semantic.detach().to(dtype=torch.float32), p=2, dim=-1)
                all_eeg_features.append(eeg_feat.cpu())
        all_eeg_features = torch.cat(all_eeg_features, dim=0)
        all_eeg_features = F.normalize(all_eeg_features.to(dtype=torch.float32), p=2, dim=-1)
    else:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                eeg = batch['eeg'].to(device)
                eeg_mask = batch['eeg_mask'].to(device)
                
                # 1. EEG 嵌入提取
                src_key_padding_mask = (eeg_mask == 0)
                z_semantic = model(eeg, src_key_padding_mask=src_key_padding_mask)
                eeg_feat = z_semantic.detach().to(dtype=torch.float32)
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

    test_subject_id = str(test_subject_id).upper() if test_subject_id is not None else None
    subj = None
    unseen_mask = None
    seen_mask = None
    if test_subject_id and any(s is not None for s in all_subject_ids):
        subj = np.array([str(s).upper() if s is not None else "UNK" for s in all_subject_ids], dtype=object)
        unseen_mask = subj == test_subject_id
        seen_mask = ~unseen_mask

    if str(tta_mode).lower() in {"v3_1_ot", "ot"}:
        if unseen_mask is None:
            raise ValueError("--tta_mode v3_1_ot requires --test_subject_id and subject_ids in the dataset.")
        if unseen_mask.any() and seen_mask.any():
            ot_device = device
            unseen_eeg = all_eeg_features[unseen_mask].to(device=ot_device)
            seen_eeg = all_eeg_features[seen_mask].to(device=ot_device)
            calibrated_unseen = sinkhorn_alignment(unseen_eeg, seen_eeg, reg=float(reg), max_iter=100)
            all_eeg_features[unseen_mask] = F.normalize(calibrated_unseen.to(device="cpu", dtype=torch.float32), p=2, dim=-1)
    
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

    if test_subject_id and subj is not None and unseen_mask is not None and seen_mask is not None:
        if unseen_mask.any():
            unseen_idx = torch.from_numpy(np.where(unseen_mask)[0]).long()
            top1_u, top5_u = _compute_topk(similarity, 5, unseen_idx)
            print(f"Top-1 Accuracy (Unseen={test_subject_id}): {top1_u*100:.2f}%")
            print(f"Top-5 Accuracy (Unseen={test_subject_id}): {top5_u*100:.2f}%")
            metrics["top1_unseen"] = float(top1_u)
            metrics["top5_unseen"] = float(top5_u)
            metrics["unseen_subject_id"] = str(test_subject_id)
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
    parser.add_argument("--tta_mode", type=str, default="none", choices=["none", "v3_1_ot", "v4_0_tent", "v5_0_sga", "v6_0_cpa_lsr"])
    parser.add_argument("--lr_tta", type=float, default=1e-4)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--lambda_elastic", type=float, default=10.0)
    parser.add_argument("--lambda_proto", type=float, default=5.0)
    parser.add_argument("--tta_rank", type=int, default=16)
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

    state_dict = state.get("state_dict", state) if isinstance(state, dict) else None
    train_text_centroid = None
    if isinstance(state_dict, dict) and "centroid_tracker.text_centroid" in state_dict:
        train_text_centroid = state_dict["centroid_tracker.text_centroid"].detach().to(dtype=torch.float32, device=device)

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
        model,
        llm,
        dataloader,
        device,
        test_subject_id=args.test_subject_id,
        tta_mode=args.tta_mode,
        reg=args.reg,
        lr_tta=args.lr_tta,
        lambda_anchor=args.lambda_anchor,
        lambda_elastic=args.lambda_elastic,
        lambda_proto=args.lambda_proto,
        tta_rank=args.tta_rank,
        train_text_centroid=train_text_centroid,
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

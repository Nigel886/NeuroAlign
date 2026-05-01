# NeuroAlign: Technical Development & Evolution Log

This document tracks the iterative development of the NeuroAlign framework, focusing on the alignment between EEG signals and LLM semantic embeddings.

---

## [v1.1] (Upcoming) - Geometric Alignment & Hypersphere Mapping
**Objective:** Address the "Modality Gap" observed in the baseline version.

### 🔬 Planned Changes:
1. **L2 Normalization:** Integrate `torch.nn.functional.normalize` before contrastive loss computation to force embeddings onto a unit hypersphere.
2. **Non-linear Projection Head:** Replace the single-layer alignment head with a 2-layer MLP (512 -> 2048 -> 4096) with ReLU activation.
3. **Learnable Temperature:** Initialize `log_inv_tau` as a learnable parameter to optimize InfoNCE repulsion intensity.

---

## [v0.1] 2024-05-22 - Initial Baseline Convergence
**Objective:** Verify the basic connection between EEG Transformer Encoder and frozen Llama-3 embeddings.

### 📊 Results (ZuCo Task 1 - SR):
- **Top-1 Accuracy:** 19.39%
- **Top-5 Accuracy:** 43.37%

### 🔍 Diagnosis:
- **Success:** The model successfully learned to distinguish relative semantic semantic similarity, outperforming random chance (1%) by 19x.
- **Critical Issue: Modality Gap.** t-SNE visualization (see `plots/tsne_v0.png`) shows that EEG and Text embeddings are geometrically isolated. The model is clustering by modality rather than collapsing into a shared semantic manifold.

---

## [v0.0] 2024-05-15 - Data Probe & Scaffolding
- Successfully parsed ZuCo `.mat` files and built an MNE-based preprocessing pipeline.
- Established the 4-bit quantized Llama-3-8B embedding extraction interface.
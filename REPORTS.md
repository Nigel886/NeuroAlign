# NeuroAlign: Technical Development & Evolution Log

This document tracks the iterative development of the NeuroAlign framework, focusing on the alignment between EEG signals and LLM semantic embeddings.

---

## [v0.2] (Upcoming) - Geometric Alignment & Hypersphere Mapping
**Objective:** Address the "Modality Gap" observed in the initial training.

### 🔬 Planned Changes:
1. **L2 Normalization:** Integrate `torch.nn.functional.normalize` before contrastive loss computation to force embeddings onto a unit hypersphere.
2. **Non-linear Projection Head:** Replace the single-layer alignment head with a 2-layer MLP (512 -> 2048 -> 4096) with ReLU activation to filter non-semantic noise.
3. **Learnable Temperature:** Transition to a learnable temperature parameter $\tau$ to adaptively balance repulsion and attraction forces.

---

## [v0.1] 2026-05-01 - Initial Baseline Convergence (First Run)
**Objective:** Establish the first end-to-end pipeline connecting EEG Transformer Encoder with frozen Llama-3-8B embeddings.

### 📊 Results (ZuCo Reading Task):
- **Top-1 Accuracy:** 19.39%
- **Top-5 Accuracy:** 43.37%

### 🔍 Diagnosis:
- **Major Milestone:** Successfully demonstrated that neural signals during reading can be mapped to a high-dimensional LLM latent space with 19x better accuracy than random chance.
- **Critical Observation: Modality Gap.** t-SNE visualization (ref: `tsne_visualization.png`) reveals two distinct clusters. Although semantic proximity is partially captured, the EEG and Text embeddings do not yet inhabit a shared manifold.
- **System Note:** Confirmed compatibility with RTX 4070 (8GB) using 4-bit quantization and gradient accumulation.

---

## [v0.0] 2026-04-20 - Scaffolding & Data Pipeline
- Completed MNE-based preprocessing (Bandpass 0.5-50Hz, ICA artifact removal).
- Built custom PyTorch Dataset to synchronize EEG segments with ZuCo text labels.
- Verified 4-bit loading of Llama-3-8B via `bitsandbytes`.
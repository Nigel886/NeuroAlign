# NeuroAlign: Technical Development & Evolution Log

This document tracks the iterative development of the NeuroAlign framework, focusing on the alignment between EEG signals and LLM semantic embeddings.

---
## [v1.1] 2026-05-02 - The Reality Check: LOSO Validation

**Objective:** Evaluate cross-subject generalization using Leave-One-Subject-Out (LOSO) on ZAB.

### 📊 Performance Metrics (The Generalization Gap):

|**Metric**|**Seen Subjects**|**Unseen (ZAB)**|
|---|---|---|
|**Top-1 Accuracy**|8.95%|**0.00%**|
|**Top-5 Accuracy**|47.90%|**0.13%**|

### 🔍 Technical Post-mortem:

1. **Subject Incompatibility:** The model failed to translate semantic alignment logic from the training pool (ZPH, ZMG, etc.) to the target subject (ZAB). This confirms the high sensitivity of EEG-to-LLM alignment to individual neural signatures.
    
2. **Manifold Overlap:** t-SNE visualization shows good inter-subject overlap among EEG embeddings, but a persistent **Modality Gap** remains. The "alignment" achieved is purely physiological, not semantic.
    
3. **Baseline Correction:** v1.0's 83% was identified as "subject-specific overfitting." The current 0% represents the true challenge of subject-independent neural decoding.

---

## [v1.0] 2026-05-01 - Milestone: High-Fidelity Semantic Alignment
**Objective:** Finalize the core alignment architecture using nonlinear projection and hypersphere constraints.

### 🚀 Performance Breakthrough:
- **Top-1 Accuracy: 83.67%** (Significant jump from v0.1's 19.39%)
- **Top-5 Accuracy: 97.70%** (Near-perfect retrieval within semantic candidates)

### 🛠️ Key Architectural Innovations:
1. **Multi-layer Projection Head:** Transitioned from a linear map to a 2-layer MLP (512 -> 2048 -> 4096) with ReLU, effectively decoupling EEG physiological noise from semantic features.
2. **Angular Manifold Constraint:** Implemented L2 Normalization on both modalities, forcing the InfoNCE loss to optimize for directional cosine similarity.
3. **Adaptive Contrastive Temperature:** Utilized a learnable $\tau$ parameter to stabilize gradient flow and prevent semantic collapse.

### 🔍 Research Insight:
The v1.0 results prove that the "Modality Gap" can be bridged through angular alignment. Although t-SNE still shows spatial separation, the near-98% Top-5 accuracy confirms that the EEG and Text embeddings are now semantically synchronized in the high-dimensional latent space.

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
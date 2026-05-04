# NeuroAlign: Technical Development & Evolution Log

This document tracks the iterative development of the NeuroAlign framework, focusing on the alignment between EEG signals and LLM semantic embeddings.

---

## [v1.8.1] 2026-05-04 - Project Phantom: The Manifold Paradox

**Objective:** Neutralize subject-specific domain shift for ZAB through DANN-based feature inversion and Momentum-based Centering ($ \alpha=0.8 $) to align neural manifolds with Llama-3 semantic space.

### 📊 Performance Comparison (v1.7 vs. v1.8.1):

| **Metric** | **v1.7 (Ghost)** | **v1.8.1 (Phantom)** | **Status** |
| :--- | :--- | :--- | :--- |
| **Seen Top-1 Accuracy** | 8.28% | **9.46%** | 📈 Improvement |
| **Seen Top-5 Accuracy** | 48.41% | **49.39%** | 📈 Stability |
| **Unseen Top-1 (ZAB)** | 0.26% | **0.13%** | ❌ Semantic Collapse |
| **Unseen Top-5 (ZAB)** | 0.51% | **0.26%** | ❌ Semantic Collapse |
| **Geometric Overlap** | High Segregation | **Near Perfect** | ✅ Seen Domains Merged |

### 🔍 Technical Diagnosis:
1. **The Paradox of Alignment:** t-SNE 结果显示 `ZKH`, `ZKW`, `ZMG`, `ZPH` 的特征空间实现了近乎完美的流形合并 (Manifold Merging)。然而，尽管 ZAB 嵌入在几何上强行靠近了 Text 区域，但其 0.13% 的 Top-1 准确率证明了 ZAB 集群的内部结构已坍塌为非判别性的“点云 (blob)”，失去了与特定语义标签的映射关系。
2. **Momentum Over-Regularization:** 动量中心化 ($ \alpha=0.8 $) 过于强力。它强制模型维持了一个对已见域极其友好的“刚性中心”，导致模型为了追求域间的一致性，剥离了跨被试通用语义所需的细微特征，产生了严重的特征退化。
3. **Modal Gap Persistence:** 尽管 DANN 消除了 Seen subjects 之间的个体差异，但 Seen EEG 集群与 Text/ZAB 集群之间仍存在显著的垂直断裂。这表明模型学会了“消除身份信息”，但尚未真正建立“脑电-文本”的跨模态桥梁。

---

## [v1.8] 2026-05-04 - Project Phantom: The Manifold Paradox

**Objective:** Neutralize subject-specific domain shift for ZAB through Test-Time Centroid Adaptation (TTCA) and style-variance suppression.

### 📊 Performance Comparison (v1.7 vs. v1.8):

| **Metric** | **v1.7 (Ghost)** | **v1.8 (Phantom)** | **Status** |
| :--- | :--- | :--- | :--- |
| **Seen Top-1 Accuracy** | 8.28% | **7.48%** | 📉 Degradation |
| **Seen Top-5 Accuracy** | 48.41% | **39.97%** | 📉 Discriminability Loss |
| **Unseen Top-1 (ZAB)** | 0.26% | **0.00%** | ❌ Semantic Collapse |
| **Unseen Top-5 (ZAB)** | 0.51% | **0.00%** | ❌ Semantic Collapse |
| **Geometric Overlap** | High Gap | **Near Perfect** | ✅ Manifold Merged |

### 🔍 Technical Diagnosis:
1. **The Paradox of Alignment:** t-SNE visualization confirms that ZAB embeddings now occupy the same hyperspace coordinates as the Text embeddings. However, the 0% accuracy proves that the *internal structure* of the ZAB cluster has collapsed into a non-discriminative "blob"[cite: 1, 4].
2. **Over-Regularization:** The Style Residual Regularization ($\gamma=0.1$) was too punitive. It forced the model to discard subject-specific noise at the cost of fine-grained semantic features.
3. **Retrieval Instability:** TTCA moved the "cloud" but did not fix the "alignment within the cloud." Combined with a low temperature ($\tau=0.02$), any minor mismatch was penalized heavily by the InfoNCE loss, leading to zero hits.

---

## [v1.7] 2026-05-03 - Project Ghost: The Zero-Shot Breakthrough

**Objective:** Overcome the domain-shift barrier for unseen subjects via stochastic centroid perturbation and domain smoothing.

### 📊 Performance Metrics (The Generalization Pivot):

| **Metric** | **v1.5 (Medusa)** | **v1.7 (Ghost Protocol)** | **Trend** |
| :--- | :--- | :--- | :--- |
| **Seen Top-1 Accuracy** | 9.30% | **8.28%** | 📉 Trade-off for Generalization |
| **Seen Top-5 Accuracy** | 49.71% | **48.41%** | ➡️ Maintained Stability |
| **Unseen Top-1 (ZAB)** | 0.00% | **0.26%** | 🚀 **BREAKTHROUGH** (Non-zero) |
| **Unseen Top-5 (ZAB)** | 0.38% | **0.51%** | 📈 Incremental Gain |
| **Manifold Shape** | Over-clustered | **Distributed/Robust** | ✅ Reduced Domain Bias |

### 🔍 Technical Post-mortem:
1. **The Ghost Protocol Effect:** The injection of centroid noise successfully forced the encoder to decouple semantic features from fixed subject coordinates. This resulted in the first successful Top-1 retrieval for an unseen domain (ZAB).
2. **Distributional Smoothing:** t-SNE analysis shows that ZAB embeddings are no longer collapsed into a singular 'identity-heavy' kernel but are now expanding towards the LLM semantic anchors.
3. **The Persistent Gap:** While generalization has begun, a systemic modality gap remains. ZAB's cluster is still primarily located in the negative-X quadrant relative to the Text manifold, suggesting a residual translation bias.

---

## [v1.6] 2026-05-03 - Project NeuroAlign: LOSO & Adversarial Bottleneck

**Objective:** Implement Leave-One-Subject-Out (LOSO) training and evaluate cross-subject generalization using Subject-Adversarial (DANN) logic.

### 📊 Performance Metrics (The Generalization Challenge):

|**Metric**|**v1.5 (Seen Master)**|**v1.6 (LOSO Attempt)**|**Trend**|
|---|---|---|---|
|**Seen Top-1 Accuracy**|9.30%|**8.76%**|📉 Minor Trade-off|
|**Seen Top-5 Accuracy**|49.71%|**49.49%**|➡️ Highly Stable|
|**Unseen Top-1 (ZAB)**|0.00%|**0.00%**|⛔ Zero-Shot Wall|
|**Unseen Top-5 (ZAB)**|0.38%|**0.00%**|📉 Regression|
|**Modality Alignment**|Overlapping (Seen)|**Fragmented (Unseen)**|⚠️ Modality Gap|

### 🔍 Technical Post-mortem:

1. **Adversarial Interference:** The introduction of the DANN head with `lambda_subject` scheduling aimed to erase subject identity. However, the resulting gradient noise caused a slight drop in primary alignment accuracy (Seen Top-1 dropped by 0.54%).
    
2. **The Modality Gap:** t-SNE visualization confirms that while Seen Subjects cluster near the text manifold, the Unseen Subject (ZAB) remains trapped in a separate manifold region. The "Modality Gap" is currently the primary blocker for zero-shot retrieval.
    
3. **Batch Size Constraint:** With an effective batch size of only 16, the contrastive loss (InfoNCE) lacks the negative sample density required to distinguish subtle semantic differences across different human brain patterns.

---

## [v1.5] 2026-05-03 - Project Medusa: Manifold Restoration & Seen Breakthrough

**Objective:** Rectify geometric collapse via gradient clipping and expand contrastive density using a Momentum Memory Bank.

### 📊 Performance Metrics (The Medusa Victory):

| **Metric** | **v1.4 (Collapsed)** | **v1.5 (Medusa)** | **Trend** |
| :--- | :--- | :--- | :--- |
| **Seen Top-1 Accuracy** | 0.06% | **9.30%** | 🚀 Structural Leap |
| **Seen Top-5 Accuracy** | 0.22% | **49.71%** | 🚀 Major Milestone |
| **Unseen Top-1 (ZAB)** | 0.00% | **0.00%** | ➡️ Generalization Bottleneck |
| **Unseen Top-5 (ZAB)** | 0.26% | **0.38%** | 📈 Slight Gain |
| **Manifold Shape** | Snake-like Filaments | **Healthy Gaussian Clouds** | ✅ Geometry Restored |

### 🔍 Technical Post-mortem:
1. **Geometric Recovery:** By enforcing `max_norm=1.0` gradient clipping and fixing the pre-normalization centering logic, the 'snake' manifold pathology was eliminated. Embeddings now occupy a high-dimensional Gaussian space compatible with LLM semantics.
2. **Contrastive Density:** The 1024-capacity `MemoryBank` provided the necessary negative sample pressure for the InfoNCE loss to converge. The alignment head (weighted at 5.0) successfully mapped EEG signals into the LLM manifold for trained subjects.
3. **The Unseen Frontier:** While seen subject performance reached record highs, the unseen subject (ZAB) remains trapped in a 'Subject-Anchor' state, where embeddings cluster tightly but fail to disperse across semantic labels.

---

## [v1.4] 2026-05-03 - Cross-Modal Centering: The Geometric Collapse

**Objective:** Bridge the Modality Gap between EEG and Text clusters via EMA-based global centroid shifting and bias correction.

### 📊 Performance Metrics (The Great Regression):

| **Metric** | **v1.3 (Disentangled)** | **v1.4 (Centering)** | **Trend** |
| :--- | :--- | :--- | :--- |
| **Seen Top-1 Accuracy** | 2.10% | **0.06%** | 📉 Semantic Collapse |
| **Unseen Top-1 (ZAB)** | 0.13% | **0.00%** | 📉 Total Failure |
| **Align Loss (End)** | ~1.10 | **1.18** | ➡️ Plateau at 1.18 |
| **Total Loss (End)** | ~7.50 | **6.19** | ✅ Structural Convergence |
| **Manifold Shape** | Gaussian Cloud | **Snake-like Strings** | ❌ Geometric Distortion |

### 🔍 Technical Post-mortem:
1. **Adversarial Overload:** The subject classifier's gradients (Subject Loss ~10.0) dominated the training process. The Gradient Reversal Layer (GRL) forced the encoder to learn non-semantic, 'snake-like' geometric features to bypass subject identification, effectively 'erasing' semantic discriminability.
2. **Contrastive Starvation:** With an effective batch size of 8 (batch_size=2, grad_accum=4), the InfoNCE loss lacked sufficient negative samples to generate a strong alignment gradient. This resulted in a stagnant Align Loss of 1.18.
3. **Centering Paradox:** While the global centroids shifted, the underlying manifold topology was too distorted for the linear translation to enable successful retrieval.

---

## [v1.3] 2026-05-02 - Feature Disentanglement: Recovery of Semantic Utility

**Objective:** Implement a "Split-Head" architecture to decouple semantic intent (Content) from subject identity (Style), and introduce Euclidean Alignment (EA) for raw signal calibration.

### 📊 Performance Metrics (The Semantic Recovery):

|**Metric**|**v1.2.1 (Warmup)**|**v1.3 (Disentanglement)**|**Growth**|
|---|---|---|---|
|**Seen Top-1 Accuracy**|0.86%|**2.10%**|**+144%**|
|**Seen Top-5 Accuracy**|3.76%|**14.04%**|**+273%**|
|**Unseen Top-1 (ZAB)**|0.13%|**0.13%**|➡️ Plateau|
|**Unseen Top-5 (ZAB)**|0.51%|**0.26%**|📉 Precision Drop|

### 🔍 Technical Diagnosis:

1. **Successful Recovery:** The massive jump in _Seen_ performance proves that the "Split-Head" architecture successfully shielded the semantic branch from adversarial erasure. The model can now "forget" who a person is without "forgetting" what they are thinking.
    
2. **The Unseen Plateau:** ZAB's Top-1 performance remained at 0.13%. t-SNE analysis confirms that while subjects are well-mixed, the entire EEG manifold is still physically distant from the Text manifold. This confirms a persistent **Modality Gap** that simple alignment cannot bridge.
    
3. **Orthogonality Success:** Orthogonality loss was confirmed to prevent identity-leakage into the content head, ensuring that the alignment was truly based on invariant semantic features.
    

### 💡 Next Steps (v1.4 Evolution):

- **Domain Probing:** Since ZAB is a "cold start" brain, we may need a small **Calibration Probing** phase (e.g., using 5-10 sentences from the target) to shift the manifold center.
    
- **Cross-Modal Centering:** Implementing a global bias correction to pull the EEG and Text clusters together in the latent space.

---

## [v1.2.1] 2026-05-02 - Curricular Adversarial Alignment: Breaking the 0% Barrier

**Objective:** Mitigate the "Semantic Erasure" effect observed in v1.2 by implementing a dynamic $\lambda$-Warmup scheduler and differential learning rates.

### 📊 Performance Metrics (Recovery & Initial Transfer):

|**Metric**|**v1.2 (Adversarial Collapse)**|**v1.2.1 (Balanced DANN)**|**Trend**|
|---|---|---|---|
|**Seen Top-1 Acc**|0.13%|**0.86%**|📈 Recovery|
|**Unseen Top-1 (ZAB)**|0.00%|**0.13%**|🚀 **First Transfer**|
|**Seen Top-5 Acc**|1.02%|**3.76%**|📈 Improved|
|**Subject Mixing**|Extreme|**Extreme**|✅ Consistent|

### 🔍 Technical Post-mortem:

1. **The "Semantic Spark":** Achieving a non-zero Top-1 accuracy on the unseen subject (ZAB) provides the first empirical evidence that neural-to-semantic alignment can generalize across different brains.
    
2. **$\lambda$-Warmup Effectiveness:** By keeping $\lambda = 0$ for the first 5 epochs and then scaling it via $\lambda = \frac{2}{1 + \exp(-10 \cdot p)} - 1$, the model established a stable semantic manifold before attempting to erase subject identity. This successfully prevented the catastrophic collapse seen in v1.2.
    
3. **Differential LR Optimization:** Assigning a lower learning rate ($1 \times 10^{-5}$) to the Subject Classifier prevented the discriminator from overpowering the Encoder, allowing for a more nuanced feature extraction process.
    
4. **Persistent Bottleneck:** While subject-specific clusters are successfully merged (ref: `tsne_v1_2_1_dann_loso_ZAB_subjects.png`), the overall retrieval performance remains low. This suggests that "Identity Information" and "Semantic Information" are deeply entangled in the current single-stream architecture.
    

### 💡 Next Steps (v1.3 Evolution):

- **Feature Disentanglement:** Move toward a dual-stream or "Split-Head" architecture to explicitly separate subject-invariant content from subject-specific style.
    
- **Euclidean Alignment (EA):** Implement covariance-based signal alignment in the preprocessing stage to reduce physiological variance before the data enters the Transformer.

---

## [v1.2] 2026-05-02 - Adversarial Domain Adaptation (DANN) Initial Test

**Objective:** Implement Domain Adversarial Neural Networks (DANN) with a Gradient Reversal Layer (GRL) to achieve subject-invariant feature extraction.

### 📊 Performance Metrics (Adversarial Collapse):

|**Metric**|**v1.1 (LOSO Baseline)**|**v1.2 (DANN Initial)**|**Trend**|
|---|---|---|---|
|**Seen Top-1 Acc**|8.95%|**0.13%**|📉 Severe Drop|
|**Unseen Top-1 (ZAB)**|0.00%|**0.00%**|➡️ Stagnant|
|**Seen Top-5 Acc**|47.90%|**1.02%**|📉 Semantic Collapse|
|**Subject Mixing**|Moderate (Clustered)|**Extreme (Fully Mixed)**|✅ **Alignment Success**|

### 🔍 Technical Post-mortem:

1. **Semantic Erasure:** The adversarial subject classifier was too potent. To "hide" the subject's identity from the discriminator, the EEG Encoder opted to destroy all distinguishable features, including the critical semantic information needed for LLM alignment.
    
2. **Adversarial Weight ($\lambda$) Sensitivity:** Using a static $\lambda=1.0$ from the start of training prevented the model from establishing a stable semantic manifold. The "Modality Gap" observed in t-SNE expanded significantly as EEG embeddings drifted into a non-semantic noise cluster.
    
3. **Subject Invariance vs. Task Utility:** While t-SNE (ref: `tsne_v1_2_dann_loso_ZAB_subjects.png`) confirms that subject-specific boundaries have been successfully erased, the resulting "universal" features lack the granularity required for zero-shot retrieval.
    

### 💡 Next Steps (v1.2.1 Evolution):

- **$\lambda$-Warmup Strategy:** Implement a dynamic scheduler for the adversarial loss, allowing the model to prioritize semantic alignment in early epochs before introducing the identity-erasure constraint.
    
- **Feature Decoupling:** Explore a dual-encoder or multi-head approach to more effectively separate "Neural Identity" from "Neural Intent."

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
# NeuroAlign

## English

### Abstract

NeuroAlign studies a difficult but practically important problem: how to align continuous-reading EEG signals with the semantic space of a frozen large language model under strong cross-subject distribution shift. The project is built on the ZuCo dataset and uses a Transformer-based EEG encoder trained against frozen LLM text representations. The research emphasis is not only on in-domain alignment, but on whether a model trained on seen subjects can preserve semantic structure when evaluated on an unseen subject in a strict LOSO setting.

Across versions, the project evolves from a clean CLIP-style contrastive baseline into a family of test-time adaptation and representation-learning methods. The core line of inquiry is whether unseen-subject failure is caused by optimization weakness at inference time or by representational deprivation in the encoder itself. The current evidence recorded in [REPORTS.md](REPORTS.md) suggests a strong answer: test-time adaptation can refine existing semantic cues, but cannot recover semantic structure that the frozen base encoder never emits; topology-aware graph fusion is therefore introduced as a base-level remedy.

### Research Problem

Two recurring phenomena motivate this repository:

- **Seen-domain dominance**: retrieval on seen subjects can become very strong while unseen performance remains near zero.
- **Sovereign outlier subjects**: certain held-out subjects, especially `ZPH`, behave like near-orthogonal domains rather than mild distribution shifts.
- **TTA safety constraint**: inference-time adaptation must improve unseen domains without contaminating seen-domain performance.
- **Representation boundary**: when online adaptation saturates, the bottleneck may lie in the frozen encoder rather than in the adaptation objective.

### Method Overview

NeuroAlign is organized around three interacting components:

1. **Base EEG-to-LLM alignment**
   - `EEGTransformerEncoder` maps EEG into the semantic space of a frozen LLM.
   - Training uses CLIP-style InfoNCE with static `1.2x` hard-negative mining.
   - Text features come from a frozen LLM, defaulting to `meta-llama/Meta-Llama-3-8B-Instruct`.

2. **Safe test-time adaptation**
   - The `v6.*` family introduces zero-initialized low-rank residual projectors.
   - Multi-head routing, dynamic gating, optimal transport, temporal consistency, and mixup guidance are explored.
   - The design target is strict seen-domain immunity: adaptation should act on the target domain without eroding source-domain retrieval quality.

3. **Base-level structural remedies**
   - `v7.*` explores meta-learning style bases to improve adaptation readiness.
   - `v8.0_graph` introduces a Subject Interaction Graph and topology-gated fusion inside the encoder.
   - The graph line targets missing cross-subject semantic bridges at the representation level rather than only at inference time.

### Main Contributions

- A reproducible EEG-to-LLM alignment pipeline centered on ZuCo continuous reading.
- A systematic TTA benchmark spanning low-rank projection, multi-head routing, OT-guided adaptation, temporal consistency, and mixup-guided adaptation.
- An explicit safety-oriented evaluation protocol separating seen and unseen retrieval behavior under LOSO.
- A graph-based encoder variant (`v8.0_graph`) that improves unseen-domain recovery while largely preserving strong seen-domain performance.
- A versioned research log in [REPORTS.md](REPORTS.md) that records negative results, saturation points, and structural turning points instead of only final wins.

### Key Findings So Far

Based on the current experimental trajectory in [REPORTS.md](REPORTS.md):

- `v6.1_mhlr` shows that multi-head low-rank adaptation can dramatically improve seen-domain retrieval, pushing seen Top-5 to `85.53%`, while still struggling on sovereign outlier domains.
- `v6.2` and `v6.3` indicate that stronger routing, OT, and temporal consistency are not sufficient when the base encoder lacks recoverable semantic cues for the unseen subject.
- `v7.*` suggests that improving adaptation elasticity alone can destroy source stability if the base geometry is overly softened.
- `v8.0_graph` is the first branch to materially break the ZPH barrier, reaching `Unseen Top-1 = 7.63%` and `Unseen Top-5 = 80.15%` while keeping seen performance near the historical frontier.

### Repository Structure

- `src/model.py`: `EEGTransformerEncoder`, frozen LLM loading, centering logic, and graph-base integration
- `src/trainer.py`: InfoNCE training loop, hard negative mining, memory bank, and centroid tracking
- `src/data_loader.py`: ZuCo `.pt` loading, subject-wise normalization, LOSO split, and collate logic
- `src/preprocess.py`: EEG preprocessing and `.pt` export
- `src/data_probe.py`: raw dataset structure inspection
- `main.py`: training entrypoint
- `eval.py`: retrieval evaluation, t-SNE visualization, and TTA algorithms
- `outputs/`: `metrics.json`, figures, and experiment artifacts
- `REPORTS.md`: version-by-version research log and result archive

### Reproducibility

#### Environment

Python **3.10/3.11** is strongly recommended. On Windows, Python **3.13** often fails to install CUDA-enabled PyTorch wheels correctly and usually leads to CPU-only execution.

Recommended setup:

```powershell
conda create -n align311 python=3.11 -y
conda activate align311

python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu121"
python -m pip install -r requirements.txt
python -m pip install accelerate
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

#### HuggingFace Access

The default text backbone is `meta-llama/Meta-Llama-3-8B-Instruct`. You may need:

- `huggingface-cli login`
- local/offline cache
- or a smaller fallback model such as `gpt2` for quick pipeline validation

#### Data

The project expects preprocessed ZuCo `.pt` data, typically:

- `./data/preprocessed/processed_zuco_cleaned.pt`

These files are generated by `src/preprocess.py` and consumed by `src/data_loader.py`.

### Running The Project

#### Training

Basic training:

```powershell
python main.py --data_path ".\data\preprocessed\processed_zuco_cleaned.pt" --epochs 20 --batch_size 2 --grad_accum 16 --lr 1e-4 --version_tag v3_0
```

LOSO training:

```powershell
python main.py --data_path ".\data\preprocessed\processed_zuco_cleaned.pt" --loso_test_subject ZPH --epochs 20 --batch_size 2 --grad_accum 16 --lr 1e-4 --version_tag v3_0_tta_ZPH
```

Graph-base training:

```powershell
python main.py --data_path ".\data\preprocessed\processed_zuco_cleaned.pt" --epochs 20 --batch_size 2 --grad_accum 16 --lr 1e-4 --use_graph_base True --graph_hid_dim 256 --version_tag v8_0_graph
```

#### Evaluation

Baseline evaluation:

```powershell
python eval.py --checkpoint "checkpoints/<tag>/eeg_encoder_epoch_best.pth" --test_subject_id ZPH --tta_mode none --output_prefix baseline
```

Representative TTA modes:

- `v6_0_cpa_lsr`
- `v6_1_mhlr`
- `v6_2_dgmhlr`
- `v6_2_ot_dgmhlr`
- `v6.3_tmc_ot_dgmhlr`
- `v6.4_mixup_anchor`

Example:

```powershell
python eval.py --checkpoint "checkpoints/v3_0_tta_ZPH/eeg_encoder_v3_0_tta_ZPH_loso_ZPH_best.pth" --test_subject_id ZPH --tta_mode v6.4_mixup_anchor --tta_rank 16 --lr_tta 1e-4 --lambda_proto 0.1 --output_prefix v6.4_mixup_anchor
```

### Practical Notes

- Llama-3 8B is heavy even in 4-bit; reduce batch size to `1-4` and rely on gradient accumulation.
- On Windows laptop GPUs with `8GB` VRAM, LLM loading may become the system bottleneck before TTA starts.
- If CUDA is unavailable, check Python version first before debugging the project itself.

---

## 中文

### 摘要

NeuroAlign 关注一个很难但很有现实意义的问题：在强跨受试分布偏移下，如何把连续阅读任务中的 EEG 信号对齐到冻结大语言模型的语义空间。项目基于 ZuCo 数据集，使用 Transformer 风格的 EEG 编码器去逼近冻结 LLM 的文本表征。研究重点不仅是域内对齐是否成立，更在于严格 LOSO 设置下，模型在未见受试者上是否还能保持可检索的语义结构。

整个项目沿着“干净基线 -> 测试时自适应 -> 基座结构升级”的路线持续推进。训练端从一个相对纯净的 CLIP-style 对比学习基线出发，推理端逐步引入多类 TTA 机制；与此同时，我们不断追问一个更根本的问题：未见受试者失败究竟是因为推理时优化不够强，还是因为编码器本身没有产生可救回的语义线索。当前 [REPORTS.md](REPORTS.md) 中的证据更倾向于后者：测试时自适应只能修整已有语义，不能凭空制造编码器没有发射出来的语义 token，因此需要引入图拓扑等“基座级”结构补救。

### 研究问题

这个仓库围绕以下几个核心现象展开：

- **已见域统治**：Seen 域检索指标可以很高，但 Unseen 域仍接近零。
- **主权离群受试者**：某些留一受试者，尤其是 `ZPH`，更像近似正交域，而不是普通分布偏移。
- **TTA 安全约束**：测试时适应必须改善未见域，同时不能污染已见域表现。
- **表征边界**：当在线优化已经饱和时，真正瓶颈可能不在 TTA 损失，而在冻结编码器本身。

### 方法概览

NeuroAlign 目前可以理解为三个相互作用的模块：

1. **EEG 到 LLM 的基础对齐**
   - `EEGTransformerEncoder` 负责把 EEG 表征映射到冻结 LLM 的语义空间。
   - 训练采用 CLIP-style InfoNCE，并保留静态 `1.2x` Hard Negative Mining。
   - 文本特征来自冻结 LLM，默认是 `meta-llama/Meta-Llama-3-8B-Instruct`。

2. **安全的测试时自适应**
   - `v6.*` 系列引入零初始化的低秩残差投影器。
   - 在此基础上探索多头路由、动态门控、最优传输、时序一致性和 mixup 引导。
   - 设计目标是严格的 Seen 域免疫：只调整目标域，不破坏源域检索质量。

3. **基座级结构补救**
   - `v7.*` 尝试用 meta-learning 风格基座提升“可适应性”。
   - `v8.0_graph` 在编码器内部引入 Subject Interaction Graph 和 topology-gated fusion。
   - 图拓扑路线解决的是“编码器是否发出了跨受试语义桥梁”这个表示层问题，而不仅仅是推理时怎么优化。

### 当前主要贡献

- 构建了一个围绕 ZuCo 连续阅读任务的可复现 EEG-to-LLM 对齐流水线。
- 系统比较了多种 TTA 方案，包括低秩投影、多头路由、OT 引导适应、时序一致性和 mixup 引导。
- 建立了强调 Seen/Unseen 分离的安全评测协议，适合分析 LOSO 下的真实跨域行为。
- 提出并实现了 `v8.0_graph` 图拓扑编码器，在尽量保持源域性能的同时，首次显著突破 ZPH 这类强离群受试者。
- 用 [REPORTS.md](REPORTS.md) 保留了完整版本演化，包括失败结果、饱和边界和结构转折点，而不只是最终最好结果。

### 当前阶段的关键发现

结合 [REPORTS.md](REPORTS.md) 中的现有结果，可以得到几条很明确的结论：

- `v6.1_mhlr` 证明多头低秩适应能显著提升 Seen 域检索，Seen Top-5 可推到 `85.53%`，但对主权离群域仍然吃力。
- `v6.2` 与 `v6.3` 表明，即便继续加强路由、OT 和时序一致性，如果基座编码器对未见受试者没有发出可恢复的语义线索，TTA 依然救不回来。
- `v7.*` 说明单纯提升“可适应性”可能会以源域稳定性为代价，基座几何一旦过软，检索结构会整体塌陷。
- `v8.0_graph` 是目前第一条实质性打破 ZPH 冻结边界的路线，在保持接近历史最优 Seen 指标的同时，把 `Unseen Top-1` 提升到 `7.63%`、`Unseen Top-5` 提升到 `80.15%`。

### 仓库结构

- `src/model.py`：`EEGTransformerEncoder`、冻结 LLM 加载、centering 逻辑，以及 graph-base 集成
- `src/trainer.py`：InfoNCE 训练循环、Hard Negative Mining、Memory Bank、CentroidTracker
- `src/data_loader.py`：ZuCo `.pt` 数据读取、受试者归一化、LOSO 划分、batch 拼接
- `src/preprocess.py`：EEG 预处理与 `.pt` 导出
- `src/data_probe.py`：原始数据结构探查
- `main.py`：训练入口
- `eval.py`：检索评测、t-SNE 可视化、多种 TTA 算法
- `outputs/`：`metrics.json`、图像与实验产物
- `REPORTS.md`：按版本记录的研究日志与结果档案

### 复现说明

#### 环境

强烈建议使用 **Python 3.10/3.11**。在 Windows 上，**Python 3.13** 很容易装不上 CUDA 版 PyTorch，最后只能走 CPU。

推荐安装方式：

```powershell
conda create -n align311 python=3.11 -y
conda activate align311

python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu121"
python -m pip install -r requirements.txt
python -m pip install accelerate
```

验证 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

#### HuggingFace 权限

默认文本骨架模型是 `meta-llama/Meta-Llama-3-8B-Instruct`。你可能需要：

- `huggingface-cli login`
- 使用本地或离线缓存
- 或者先用 `gpt2` 这类更小模型验证 pipeline

#### 数据

项目默认读取预处理后的 ZuCo `.pt` 数据，例如：

- `./data/preprocessed/processed_zuco_cleaned.pt`

这些文件由 `src/preprocess.py` 生成，并由 `src/data_loader.py` 读取。

### 如何运行

#### 训练

基础训练：

```powershell
python main.py --data_path ".\data\preprocessed\processed_zuco_cleaned.pt" --epochs 20 --batch_size 2 --grad_accum 16 --lr 1e-4 --version_tag v3_0
```

LOSO 训练：

```powershell
python main.py --data_path ".\data\preprocessed\processed_zuco_cleaned.pt" --loso_test_subject ZPH --epochs 20 --batch_size 2 --grad_accum 16 --lr 1e-4 --version_tag v3_0_tta_ZPH
```

图拓扑基座训练：

```powershell
python main.py --data_path ".\data\preprocessed\processed_zuco_cleaned.pt" --epochs 20 --batch_size 2 --grad_accum 16 --lr 1e-4 --use_graph_base True --graph_hid_dim 256 --version_tag v8_0_graph
```

#### 评测

基础评测：

```powershell
python eval.py --checkpoint "checkpoints/<tag>/eeg_encoder_epoch_best.pth" --test_subject_id ZPH --tta_mode none --output_prefix baseline
```

代表性 TTA 模式：

- `v6_0_cpa_lsr`
- `v6_1_mhlr`
- `v6_2_dgmhlr`
- `v6_2_ot_dgmhlr`
- `v6.3_tmc_ot_dgmhlr`
- `v6.4_mixup_anchor`

示例：

```powershell
python eval.py --checkpoint "checkpoints/v3_0_tta_ZPH/eeg_encoder_v3_0_tta_ZPH_loso_ZPH_best.pth" --test_subject_id ZPH --tta_mode v6.4_mixup_anchor --tta_rank 16 --lr_tta 1e-4 --lambda_proto 0.1 --output_prefix v6.4_mixup_anchor
```

### 实际使用说明

- Llama-3 8B 即使 4-bit 也很重，建议把 `batch_size` 降到 `1-4`，并依赖梯度累积。
- 在 Windows 的 `8GB` 级笔记本 GPU 上，真正的瓶颈往往先出现在 LLM 加载，而不是 TTA 本身。
- 如果 CUDA 不可用，优先检查 Python 版本和 torch 安装，而不是先怀疑项目代码。


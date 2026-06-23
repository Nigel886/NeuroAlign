# NeuroAlign

NeuroAlign 是一个面向 ZuCo 连续阅读 EEG 数据的跨模态对齐工程：训练一个 EEG Encoder，将脑电表征映射到冻结大语言模型（默认 Llama-3）的语义嵌入空间，并在 LOSO（Leave-One-Subject-Out）设置下研究跨受试域偏移与测试时自适应（TTA）。

本仓库以“版本迭代实验”为核心工作流：训练端保持 CLIP-style InfoNCE 对齐（静态 1.2x HNM），推理端在不改动 base encoder 的前提下探索多种 TTA 机制（低秩投影、多头门控、OT、时序一致性、mixup 引导等）。详细实验轨迹与指标见 [REPORTS.md](REPORTS.md)。

## 代码结构

- `src/model.py`：EEGTransformerEncoder（Transformer 编码器 + 投影到 LLM 语义空间 + 可选 centering），以及 LLM 冻结加载逻辑
- `src/trainer.py`：训练循环（InfoNCE / Hard Negative Mining / Memory Bank / CentroidTracker）
- `src/data_loader.py`：ZuCo `.pt` 数据集封装、subject-wise z-score、LOSO 采样与 batch collate
- `src/preprocess.py`：预处理（MNE/ICA 等）与保存 `.pt`
- `eval.py`：检索评测 + t-SNE 可视化 + 多种 TTA 模式（见下）
- `main.py`：训练入口（支持 LOSO）、checkpoint 保存
- `outputs/`：评测产物（`metrics.json`、t-SNE 图）

## 环境要求（Windows / GPU 推荐）

### Python 版本

强烈建议使用 **Python 3.10/3.11**。在 Windows 上使用 **Python 3.13** 往往无法安装 CUDA 版 PyTorch（常见表现为 `No matching distribution found for torch`），会导致只能 CPU 跑。

### GPU 与依赖

- NVIDIA GPU（建议 >= 8GB 显存；4070 8GB 可以跑，但 batch 需要小）
- CUDA 驱动正常（`nvidia-smi` 可用）
- 依赖见 [requirements.txt](requirements.txt)

安装示例（conda）：

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

### HuggingFace 权限

默认使用 `meta-llama/Meta-Llama-3-8B-Instruct`。如果你没有模型访问权限或网络受限，需要先处理：

- `huggingface-cli login`
- 或设置离线缓存 / 更换可访问的 LLM（如 `--llm_name gpt2` 仅用于快速跑通流程）

## 数据准备（ZuCo）

本项目默认读取预处理后的 `.pt`（例如 `./data/preprocessed/processed_zuco_cleaned.pt`）。数据格式由 `src/preprocess.py` 产出，并由 `src/data_loader.py:ZuCoDataset` 读取。

如果你已有 `.pt`，训练与评测直接指定路径即可：

- `--data_path <your_pt_path>`

## 训练（main.py）

基础训练（对齐到冻结 LLM embedding）：

```powershell
python main.py --data_path ".\data\preprocessed\processed_zuco_cleaned.pt" --epochs 20 --batch_size 2 --grad_accum 16 --lr 1e-4 --version_tag v3_0
```

LOSO 训练（留一受试者作为测试域）：

```powershell
python main.py --data_path ".\data\preprocessed\processed_zuco_cleaned.pt" --loso_test_subject ZPH --epochs 20 --batch_size 2 --grad_accum 16 --lr 1e-4 --version_tag v3_0_tta_ZPH
```

v8.0_graph（跨受试图拓扑基底，graph-base 可开关）：

```powershell
python main.py --data_path ".\data\preprocessed\processed_zuco_cleaned.pt" --epochs 20 --batch_size 2 --grad_accum 16 --lr 1e-4 --use_graph_base True --graph_hid_dim 256 --version_tag v8_0_graph
```

checkpoint 默认保存在 `checkpoints/<version_tag>/`；当启用 `--use_graph_base True` 时，保存目录固定为 `checkpoints/v8_0_graph/`。

## 评测与 TTA（eval.py）

eval 以“检索任务”为主：从 EEG 与文本分别提取特征，计算 Top-1/Top-5，并输出 t-SNE 与 `metrics.json` 到 `outputs/`。

基础评测（无 TTA）：

```powershell
python eval.py --checkpoint "checkpoints/<tag>/eeg_encoder_epoch_best.pth" --test_subject_id ZPH --tta_mode none --output_prefix baseline
```

### 当前内置的 TTA 模式（节选）

以 `--tta_mode` 切换，所有模式均遵循“Seen 域免疫污染”的设计目标（尤其是 v6.* 系列通过零初始化残差投影实现）。

- `v6_0_cpa_lsr`：单头低秩投影（零初始化 up），只更新 projector
- `v6_1_mhlr`：多头低秩投影（MHLR）
- `v6_2_dgmhlr`：动态门控多头低秩（DGMHLR）
- `v6_2_ot_dgmhlr`：OT 引导的门控投影（Sinkhorn/Wasserstein loss）
- `v6.3_tmc_ot_dgmhlr`：时序/特征掩码一致性 + OT（Temporal Shield）
- `v6.4_mixup_anchor`：TTA 反传阶段对 unseen 做源域锚点 mixup（最终检索特征保持原生不污染）

示例（v6.4_mixup_anchor）：

```powershell
python eval.py --checkpoint "checkpoints/v3_0_tta_ZPH/eeg_encoder_v3_0_tta_ZPH_loso_ZPH_best.pth" --test_subject_id ZPH --tta_mode v6.4_mixup_anchor --tta_rank 16 --lr_tta 1e-4 --lambda_proto 0.1 --output_prefix v6.4_mixup_anchor
```

## 实验进度（来自 REPORTS.md）

最新的关键结论与轨迹都记录在 [REPORTS.md](REPORTS.md)，包括：

- v6.*：低秩投影 TTA 路线（从 LSR → 多头 → 门控 → OT → 时序一致性 → mixup 引导），强调“Seen 域指标零污染/恒等护盾”
- v7.*：meta-base（MAML 思想）对“易适应性 vs 源域稳定性”的结构张力评估
- v8.0_graph：引入受试者拓扑图基底（Subject Interaction Graph + Gated Graph Fusion）以提升跨受试桥接能力

## 常见问题

- **CUDA 不可用 / torch 是 CPU 版**：多见于 Python 3.13。请切到 Python 3.10/3.11，并安装 cu121 版 torch。
- **Llama-3 加载失败（401/403 或网络问题）**：需要 HuggingFace 登录与权限；或临时换用小模型（如 `--llm_name gpt2`）验证 pipeline。
- **显存不足**：Llama-3 8B 即使 4-bit 也很吃紧。建议将 `--batch_size` 降到 1~4，并用 `--grad_accum` 拉有效 batch。


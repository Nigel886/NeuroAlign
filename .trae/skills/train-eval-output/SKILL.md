---
name: "train-eval-output"
description: "一键串联训练→评估→定位输出：生成/执行 main.py+eval.py，并汇总 outputs 下 metrics.json 与 t-SNE 图片路径。用户要跑某个版本并快速找到结果时调用。"
---

# 训练→评估→定位输出（NeuroAlign）

当用户说“跑 vX.Y / 给我训练+eval 指令 / 结果在哪”时调用本 Skill。目标是：给出可复制的训练与评估指令，并在评估后把输出目录与关键文件路径汇总成一段可复制文本。

## 输入（从用户消息推断或用默认值）

- `version_tag`：例如 `v1_8` / `v1_8_1`
- `test_subject_id`（LOSO）：例如 `ZAB`
- 训练参数：`epochs`、`batch_size`、`grad_accum`、`lr`、`lr_subject`、`centering_momentum`
- 开关：`enable_dann`、`use_centering`
- 对比学习参数：`temperature`、`margin`
- eval 选择：用 `best` 或 `latest` checkpoint

默认策略（可按项目当前习惯调整）：
- LOSO + `--enable_dann` + `--use_centering`
- `output_prefix = tsne_<version_tag>_dann_loso_<SUBJ>`
- checkpoint：优先 `best`，失败再用 `latest`

## Step 1：生成训练指令（PowerShell 可直接复制）

```bash
python main.py `
  --data_path .\data\preprocessed `
  --loso_test_subject <SUBJ> `
  --enable_dann `
  --use_centering `
  --centering_momentum <MOMENTUM> `
  --version_tag <VERSION> `
  --epochs <EPOCHS> `
  --batch_size <BS> `
  --grad_accum <ACCUM> `
  --lr <LR> `
  --lr_subject <LR_SUBJ> `
  --temperature <TAU> `
  --margin <MARGIN>
```

## Step 2：定位 checkpoint（best / latest）

到 `.\checkpoints\<version_tag>\` 下查找：
- `eeg_encoder_<version_tag>_dann_loso_<SUBJ>_best.pth`
- `eeg_encoder_<version_tag>_dann_loso_<SUBJ>_latest.pth`

如果文件名不完全一致，就用目录内最新修改的 `.pth` 作为备选（但仍建议保持命名规范）。

## Step 3：生成 eval 指令（PowerShell 可直接复制）

best：

```bash
python eval.py `
  --data_path .\data\preprocessed `
  --checkpoint .\checkpoints\<VERSION>\eeg_encoder_<VERSION>_dann_loso_<SUBJ>_best.pth `
  --test_subject_id <SUBJ> `
  --output_prefix tsne_<VERSION>_dann_loso_<SUBJ>
```

latest：

```bash
python eval.py `
  --data_path .\data\preprocessed `
  --checkpoint .\checkpoints\<VERSION>\eeg_encoder_<VERSION>_dann_loso_<SUBJ>_latest.pth `
  --test_subject_id <SUBJ> `
  --output_prefix tsne_<VERSION>_dann_loso_<SUBJ>
```

## Step 4：评估后定位 outputs 并汇总路径

定位规则：
- 在 `.\outputs\` 下优先找名称包含 `<version_tag>` 与 `LOSO_<SUBJ>`（或包含 `dann_loso_<SUBJ>`）的最新目录
- 读取该目录的 `metrics.json`，确认其中 `checkpoint` 与本次 eval 一致

输出给用户的“可复制结果摘要”格式：

```text
Outputs: <ABS_OUTPUT_DIR>
- metrics: <ABS_OUTPUT_DIR>\metrics.json
- tsne (subjects): <ABS_OUTPUT_DIR>\tsne_<...>_subjects.png
- tsne (unseen): <ABS_OUTPUT_DIR>\tsne_<...>_unseen_<SUBJ>.png
```

可选：给一个 Windows 打开目录指令：

```bash
explorer "<ABS_OUTPUT_DIR>"
```

## 常见失败快速处理

- 如果显示 `Using device: cpu`：先确认 CUDA/torch 安装与 `torch.cuda.is_available()`，否则训练会非常慢。
- 如果 HuggingFace/tokenizer 下载超时：优先确认已登录并可访问模型；必要时使用本地缓存/离线模式。

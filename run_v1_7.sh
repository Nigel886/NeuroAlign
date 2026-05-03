#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --data_path ./data/preprocessed \
  --loso_test_subject ZAB \
  --enable_dann \
  --use_centering \
  --centering_momentum 0.8 \
  --version_tag v1_7 \
  --epochs 60 \
  --batch_size 4 \
  --grad_accum 16 \
  --lr 3e-5 \
  --lr_subject 1e-5 \
  --temperature 0.07 \
  --margin 0.2

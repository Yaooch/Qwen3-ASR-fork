#!/usr/bin/env bash
# finetuning/grpo_train.sh — GRPO 训练 wrapper（多卡数据并行）
# 用法:
#   bash finetuning/grpo_train.sh [OUTPUT_DIR] [NPROC]
#   NPROC 默认 8；单卡跑传 1（走 torchrun --nproc-per-node=1 亦可）
# effective batch = NPROC × batch_size_per_rank
set -euo pipefail
CKPT="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
DATA="/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr_dialogue_cut_mandarin.jsonl"
OUT="${1:-/cfs/data/private/WangYaoChi/model/joint_ctc_50_grpo_2}"
NPROC="${2:-8}"
PORT="${PORT:-29500}"

torchrun --nproc-per-node="$NPROC" --master-port="$PORT" -m finetuning.grpo_train \
  --ckpt "$CKPT" --data "$DATA" --output_dir "$OUT" \
  --group_size 8 --batch_size_per_rank 8 \
  --temperature 1.0 --lr 5e-5 --beta 0.04 \
  --max_steps 5000

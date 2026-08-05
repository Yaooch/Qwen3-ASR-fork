#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/.." && pwd)
export PYTHONPATH="$project_root:${PYTHONPATH:-}"
cd "$project_root"

output_dir=${1:-/cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune_subtext_v2}
nproc=${2:-8}
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} torchrun \
  --standalone \
  --nproc_per_node="$nproc" \
  finetuning/train_glclap.py \
  --output_dir "$output_dir" \
  --batch_size 8 \
  --unfreeze_audio_layers -1 \
  --unfreeze_text_layers -1 \
  --max_steps 200000 \
  --lr_projection 1e-3 \
  --lr_encoder 1e-5 \
  --weight_decay 0.01 \
  --warmup_steps 10000 \
  --eval_every 10000 \
  --save_every 10000 \
  "$@"

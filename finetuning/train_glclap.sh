#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/.." && pwd)
export PYTHONPATH="$project_root:${PYTHONPATH:-}"
cd "$project_root"

output_dir=${1:-/cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune}
nproc=${2:-2}
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5} torchrun \
  --standalone \
  --nproc_per_node="$nproc" \
  finetuning/train_glclap.py \
  --output_dir "$output_dir" \
  --batch_size 32 \
  --unfreeze_audio_layers -1 \
  --unfreeze_text_layers -1 \
  --max_steps 100000 \
  --eval_every 1000 \
  --save_every 5000 \
  "$@"

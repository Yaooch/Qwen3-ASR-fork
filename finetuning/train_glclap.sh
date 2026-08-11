#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/.." && pwd)
export PYTHONPATH="$project_root:${PYTHONPATH:-}"
cd "$project_root"

output_dir=${1:-/cfs/data/private/WangYaoChi/model/glclap/retrieval_v4}
nproc=${2:-7}
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2,3,4,5,6,7} torchrun \
  --standalone \
  --nproc_per_node="$nproc" \
  finetuning/train_glclap.py \
  --train_jsonl /cfs/data/private/WangYaoChi/train_data/all/train_700w_shuffled.jsonl \
  --eval_jsonl /cfs/data/private/WangYaoChi/train_data/all/eval_shuffled.jsonl \
  --audio_model /cfs/data/private/WangYaoChi/model/glclap/data2vec-audio-large \
  --text_model /cfs/data/private/WangYaoChi/model/glclap/bert-base-multilingual-uncased \
  --english_word_df /cfs/data/private/WangYaoChi/train_data/all/english_word_df.json \
  --output_dir "$output_dir" \
  --batch_size 9 \
  --num_workers 4 \
  --unfreeze_audio_layers -1 \
  --unfreeze_text_layers -1 \
  --max_steps 200000 \
  --lr_projection 1e-3 \
  --lr_encoder 1e-5 \
  --weight_decay 0.01 \
  --warmup_steps 10000 \
  --grad_clip 1.0 \
  --max_text_length 128 \
  --max_subtext_units 8 \
  --min_duration 0.2 \
  --max_duration 30.0 \
  --dtype bf16 \
  --seed 42 \
  --log_every 10 \
  --eval_every 10000 \
  --eval_batches 20 \
  --save_every 10000 \
  "$@"

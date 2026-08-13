#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/../.." && pwd)
export PYTHONPATH="$project_root:${PYTHONPATH:-}"
cd "$project_root"

output_dir=${1:-/cfs/data/private/WangYaoChi/model/glclap/retrieval_v5}
nproc=${2:-8}
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

case "$nproc" in
  8)
    batch_size=8
    default_devices=0,1,2,3,4,5,6,7
    ;;
  4)
    batch_size=16
    default_devices=0,1,2,3
    ;;
  *)
    echo "只支持 8 卡 × batch 8 或 4 卡 × batch 16，当前 nproc=$nproc" >&2
    exit 2
    ;;
esac

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$default_devices} torchrun \
  --standalone \
  --nproc_per_node="$nproc" \
  -m qwen_asr_ext.glclap.train \
  --train_jsonl /cfs/data/private/WangYaoChi/train_data/all/train_700w_shuffled.jsonl \
  --eval_jsonl /cfs/data/private/WangYaoChi/train_data/all/eval_shuffled.jsonl \
  --audio_backend qwen \
  --audio_model /cfs/data/private/hubk/Qwen3-ASR/Qwen/Qwen3-ASR-1___7B \
  --text_model /cfs/data/private/WangYaoChi/model/glclap/bert-base-multilingual-uncased \
  --english_word_df /cfs/data/private/WangYaoChi/train_data/all/english_word_df.json \
  --output_dir "$output_dir" \
  --batch_size "$batch_size" \
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

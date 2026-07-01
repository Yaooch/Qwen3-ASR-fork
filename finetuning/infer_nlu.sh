#!/bin/bash
# finetuning/infer_nlu.sh — NLU 批量推理 / 评测 wrapper
# 用法:
#   推理: bash finetuning/infer_nlu.sh <input.jsonl> [GPU_IDS]
#   评测: EVAL=1 bash finetuning/infer_nlu.sh <input_with_ref.jsonl> [GPU_IDS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

input_file="${1:?用法: infer_nlu.sh <input.jsonl> [GPU_IDS]}"
gpu_ids="${2:-0}"

ckpt="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
lora="/cfs/data/private/WangYaoChi/model/joint_ctc_50_nlu_lora"
output_dir="./nlu_out"

eval_flag=()
if [[ "${EVAL:-0}" == "1" ]]; then
    eval_flag=(--eval)
fi

python infer_nlu.py \
    --ckpt "$ckpt" \
    --lora "$lora" \
    --input_file "$input_file" \
    --output_dir "$output_dir" \
    --gpu_ids "$gpu_ids" \
    --batch_size 16 \
    --max_new_tokens 256 \
    "${eval_flag[@]}"

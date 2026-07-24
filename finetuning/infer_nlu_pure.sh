#!/bin/bash
# finetuning/infer_nlu_pure.sh — 纯 Qwen3 LLM NLU 推理 wrapper
# 用法:
#   推理: bash finetuning/infer_nlu_pure.sh <input.jsonl> [GPU_IDS]
#   评测: EVAL=1 bash finetuning/infer_nlu_pure.sh <input.jsonl> [GPU_IDS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

input_file="${1:-/root/asr_project/Qwen3-ASR/voyah_agent_train_20260716_sample_2k.jsonl}"
gpu_ids="${2:-6,7}"
task="${TASK:-agent}"

# ckpt="/cfs/data/private/WangYaoChi/model/Qwen3-1.7B"
ckpt="/cfs/data/private/WangYaoChi/model/qwen3_1_7b_nlu_6"
# lora="${LORA:-/cfs/data/private/WangYaoChi/model/qwen3_1_7b_nlu_2}"
lora=""
output_dir="/cfs/data/private/WangYaoChi/test_out/qwen3_1_7b_nlu_6"

eval_flag=(--eval)
if [[ "${EVAL:-1}" == "0" ]]; then
    eval_flag=()
fi

python infer_nlu_pure.py \
    --ckpt "$ckpt" \
    --lora "$lora" \
    --input_file "$input_file" \
    --output_dir "$output_dir" \
    --gpu_ids "$gpu_ids" \
    --batch_size 32 \
    --max_new_tokens 256 \
    --task "$task" \
    "${eval_flag[@]}"

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

# 用法: infer_nlu.sh [input.jsonl] [GPU_IDS]; 不传 input 用默认测试集
# default_input="/cfs/data/private/WangYaoChi/train_data/all/nlu/voyah_nlu_test_2.jsonl"
default_input="/root/asr_project/Qwen3-ASR/voyah_agent_train_split_3k.jsonl"
input_file="${1:-$default_input}"
gpu_ids="${2:-0,1,2,3}"
task="${TASK:-agent}"  # nlu=意图JSON, agent=Action&&Action Input

ckpt="/cfs/data/private/WangYaoChi/model/joint_ctc_50_nlu_9"
# lora="/cfs/data/private/WangYaoChi/model/joint_ctc_50_nlu_3"
lora=""
output_dir="/cfs/data/private/WangYaoChi/test_out/joint_ctc_50_nlu_9_train"

# 默认评测模式; EVAL=0 关闭评测只跑推理
eval_flag=(--eval)
if [[ "${EVAL:-1}" == "0" ]]; then
    eval_flag=()
fi

python infer_nlu.py \
    --ckpt "$ckpt" \
    --lora "$lora" \
    --input_file "$input_file" \
    --output_dir "$output_dir" \
    --gpu_ids "$gpu_ids" \
    --batch_size 32 \
    --max_new_tokens 256 \
    --task "$task" \
    "${eval_flag[@]}"

#!/bin/bash
# qwen_asr_ext/nlu/scripts/infer_asr.sh — ASR+NLU 评测(音频 -> 文本\n意图)
# 用法: bash qwen_asr_ext/nlu/scripts/infer_asr.sh [input.jsonl] [GPU_IDS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

# 默认用拆出的 2000 条测试集
input_file="${1:-/cfs/data/private/WangYaoChi/train_data/all/nlu/voyah_asr_nlu_test_2.jsonl}"
gpu_ids="${2:-0,1,2,3,4,5,6,7}"

ckpt="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
lora="/cfs/data/private/WangYaoChi/model/joint_ctc_50_asr_nlu_1"
output_dir="/cfs/data/private/WangYaoChi/test_out/joint_ctc_50_asr_nlu_1"

python3 -m qwen_asr_ext.nlu.infer_asr \
    --ckpt "$ckpt" \
    --lora "$lora" \
    --input_file "$input_file" \
    --output_dir "$output_dir" \
    --gpu_ids "$gpu_ids" \
    --batch_size 64 \
    --encoder_mode offline \

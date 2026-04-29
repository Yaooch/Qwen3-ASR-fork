#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

OUTPUT_DIR="/cfs/data/private/WangYaoChi/test_out/joint2_llm_2/mandarin"
REF_PATH="/cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_text_classify"

python -m qwen_asr.tools.pinyin_eval \
  --ref_path "${REF_PATH}" \
  --result_path "${OUTPUT_DIR}/results.txt" \
  --output_path "${OUTPUT_DIR}/pinyin_similarity.txt" \
  --detail_output_path "${OUTPUT_DIR}/pinyin_detail.jsonl" \
  --badcase_path "${OUTPUT_DIR}/pinyin_badcases.txt"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/qwen3-asr/bin/python}"
STAGE="all"
CKPT="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14/checkpoint-12653/"
INPUT_SCP="/cfs/data/private/WangYaoChi/open_datasets/aishell_hotword_test/wav.scp"
REF_PATH="/cfs/data/private/WangYaoChi/open_datasets/aishell_hotword_test/text"
HOTWORD_FILE="/cfs/data/private/WangYaoChi/open_datasets/aishell_hotword_test/hotword.txt"
# HOTWORD_FILE=""
OUTPUT_DIR="/cfs/data/private/WangYaoChi/test_out/joint_ctc_14/hotword_aishell/ctc"
BASELINE_DETAIL_PATH="/cfs/data/private/WangYaoChi/test_out/joint_ctc_14/hotword_aishell/ctc_no_hotword/details/results_detail.jsonl"
GPU_IDS="0,1,2,3"
BATCH_SIZE=64
DTYPE="bf16"
HOTWORD_TOPK=3
HOTWORD_PINYIN_STYLE="normal"
PROMPT="翻译语音为文本,结合语义判断如果是热词请优先使用热词."
STREAM=1
STREAM_CHUNK_SEC=0.64
STREAM_LEFT_CONTEXT_SEC=1.32
STREAM_RIGHT_CONTEXT_SEC=0.07
STREAM_FIRST_CHUNK_LEFT_PAD_SEC=0.0
STREAM_WINDOW_BATCH_SIZE=16
STREAM_WINDOW_ENCODER_BATCH_SIZE=4

usage() {
    echo "用法：bash $0 --ckpt CKPT --input_scp wav.scp --ref_path text --hotword_file hotwords.txt --output_dir out"
    echo "可选：--stage all|infer|eval --baseline_detail_path base.jsonl --gpu_ids 0,1 --batch_size 8 --hotword_topk 5 --hotword_pinyin_style normal|tone3 --prompt TEXT --no_stream"
    echo "      --stream_chunk_sec 0.64 --stream_left_context_sec 1.32 --stream_right_context_sec 0.07"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --input_scp) INPUT_SCP="$2"; shift 2 ;;
        --ref_path|--ref_dir) REF_PATH="$2"; shift 2 ;;
        --hotword_file) HOTWORD_FILE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --baseline_detail_path) BASELINE_DETAIL_PATH="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --hotword_topk) HOTWORD_TOPK="$2"; shift 2 ;;
        --hotword_pinyin_style) HOTWORD_PINYIN_STYLE="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --stream_chunk_sec) STREAM_CHUNK_SEC="$2"; shift 2 ;;
        --stream_left_context_sec) STREAM_LEFT_CONTEXT_SEC="$2"; shift 2 ;;
        --stream_right_context_sec) STREAM_RIGHT_CONTEXT_SEC="$2"; shift 2 ;;
        --stream_first_chunk_left_pad_sec) STREAM_FIRST_CHUNK_LEFT_PAD_SEC="$2"; shift 2 ;;
        --stream_window_batch_size) STREAM_WINDOW_BATCH_SIZE="$2"; shift 2 ;;
        --stream_window_encoder_batch_size) STREAM_WINDOW_ENCODER_BATCH_SIZE="$2"; shift 2 ;;
        --no_stream) STREAM=0; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数：$1"; usage; exit 1 ;;
    esac
done

if [[ -z "${OUTPUT_DIR}" || -z "${REF_PATH}" || -z "${HOTWORD_FILE}" ]]; then
    usage
    exit 1
fi
if [[ "${STAGE}" != "eval" && ( -z "${CKPT}" || -z "${INPUT_SCP}" ) ]]; then
    echo "stage=${STAGE} 时必须提供 --ckpt 和 --input_scp"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
DETAIL_PATH="${OUTPUT_DIR}/details/results_detail.jsonl"

if [[ "${STAGE}" == "all" || "${STAGE}" == "infer" ]]; then
    INFER_CMD=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py"
        --ckpt "${CKPT}"
        --mode joint
        --input_scp "${INPUT_SCP}"
        --output_dir "${OUTPUT_DIR}"
        --gpu_ids "${GPU_IDS}"
        --batch_size "${BATCH_SIZE}"
        --dtype "${DTYPE}"
        --hotword_file "${HOTWORD_FILE}"
        --hotword_topk "${HOTWORD_TOPK}"
        --hotword_pinyin_style "${HOTWORD_PINYIN_STYLE}"
        --stream_chunk_sec "${STREAM_CHUNK_SEC}"
        --stream_left_context_sec "${STREAM_LEFT_CONTEXT_SEC}"
        --stream_right_context_sec "${STREAM_RIGHT_CONTEXT_SEC}"
        --stream_first_chunk_left_pad_sec "${STREAM_FIRST_CHUNK_LEFT_PAD_SEC}"
        --stream_window_batch_size "${STREAM_WINDOW_BATCH_SIZE}"
        --stream_window_encoder_batch_size "${STREAM_WINDOW_ENCODER_BATCH_SIZE}"
    )
    if [[ -n "${PROMPT}" ]]; then
        INFER_CMD+=(--prompt "${PROMPT}")
    fi
    if [[ "${STREAM}" -eq 1 ]]; then
        INFER_CMD+=(--stream)
    fi
    "${INFER_CMD[@]}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "eval" ]]; then
    EVAL_CMD=(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/qwen_asr/tools/hotword_eval.py"
        --ref_path "${REF_PATH}"
        --detail_path "${DETAIL_PATH}"
        --hotword_file "${HOTWORD_FILE}"
        --output_path "${OUTPUT_DIR}/hotword_eval.txt"
        --detail_output_path "${OUTPUT_DIR}/hotword_eval_detail.jsonl"
        --badcase_path "${OUTPUT_DIR}/hotword_badcases.txt"
    )
    if [[ -n "${BASELINE_DETAIL_PATH}" ]]; then
        EVAL_CMD+=(--baseline_detail_path "${BASELINE_DETAIL_PATH}")
    fi
    "${EVAL_CMD[@]}"
fi

echo "完成：${OUTPUT_DIR}"

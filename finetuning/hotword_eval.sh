#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/qwen3-asr/bin/python}"
STAGE="all"
CKPT="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"

# aishell_hotword: /cfs/data/private/WangYaoChi/open_datasets/aishell_hotword_test/
# ContextASR: /cfs/data/private/WangYaoChi/open_datasets/ContextASR/hotword_test/

BASE_PATH="/cfs/data/private/WangYaoChi/open_datasets/ContextASR/hotword_test/"

INPUT_SCP="${BASE_PATH}wav.scp"
REF_PATH="${BASE_PATH}text"
HOTWORD_FILE="${BASE_PATH}hotword.txt"
UTT_HOTWORD_PATH="${BASE_PATH}utt_hotword.txt"
# HOTWORD_FILE=""
OUTPUT_DIR="/cfs/data/private/WangYaoChi/test_out/joint_ctc_14_hotword_1/ContextASR/ctc1"
BASELINE_OUTPUT_DIR=""
BASELINE_DETAIL_PATH=""
BASELINE_DETAIL_PATH_SET=0
RUN_BASELINE_INFER=0
GPU_IDS="0,1,2,3"
BATCH_SIZE=64
DTYPE="bf16"
HOTWORD_TOPK=5
HOTWORD_PINYIN_STYLE="normal"
PROMPT="转写语音，专属名词优先按列表原文输出。"
STREAM=1
STREAM_CHUNK_SEC=0.64
STREAM_LEFT_CONTEXT_SEC=1.32
STREAM_RIGHT_CONTEXT_SEC=0.07
STREAM_FIRST_CHUNK_LEFT_PAD_SEC=0.0
STREAM_WINDOW_BATCH_SIZE=16
STREAM_WINDOW_ENCODER_BATCH_SIZE=4

usage() {
    echo "用法：bash $0 --ckpt CKPT --input_scp wav.scp --ref_path text --hotword_file hotwords.txt --output_dir out"
    echo "可选：--utt_hotword_path utt_hotword.txt，每行 utt_id<TAB>热词1,热词2,..."
    echo "可选：--stage all|infer|eval --baseline_output_dir out/no_prompt --baseline_detail_path base.jsonl --no_baseline_infer"
    echo "      --gpu_ids 0,1 --batch_size 8 --hotword_topk 5 --hotword_pinyin_style normal|tone3 --prompt TEXT --no_stream"
    echo "      --stream_chunk_sec 0.64 --stream_left_context_sec 1.32 --stream_right_context_sec 0.07"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --input_scp) INPUT_SCP="$2"; shift 2 ;;
        --ref_path|--ref_dir) REF_PATH="$2"; shift 2 ;;
        --hotword_file) HOTWORD_FILE="$2"; shift 2 ;;
        --utt_hotword_path) UTT_HOTWORD_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --baseline_output_dir) BASELINE_OUTPUT_DIR="$2"; shift 2 ;;
        --baseline_detail_path) BASELINE_DETAIL_PATH="$2"; BASELINE_DETAIL_PATH_SET=1; shift 2 ;;
        --no_baseline_infer) RUN_BASELINE_INFER=0; shift 1 ;;
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

if [[ -z "${OUTPUT_DIR}" || -z "${REF_PATH}" ]]; then
    usage
    exit 1
fi
if [[ "${STAGE}" != "eval" && ( -z "${CKPT}" || -z "${INPUT_SCP}" ) ]]; then
    echo "stage=${STAGE} 时必须提供 --ckpt 和 --input_scp"
    exit 1
fi
if [[ "${STAGE}" != "eval" && -z "${HOTWORD_FILE}" ]]; then
    echo "stage=${STAGE} 时必须提供 --hotword_file 用于检索"
    exit 1
fi
if [[ "${STAGE}" == "eval" && -z "${HOTWORD_FILE}" && -z "${UTT_HOTWORD_PATH}" ]]; then
    echo "stage=eval 时必须提供 --hotword_file 或 --utt_hotword_path"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
DETAIL_PATH="${OUTPUT_DIR}/details/results_detail.jsonl"
if [[ -z "${BASELINE_OUTPUT_DIR}" ]]; then
    BASELINE_OUTPUT_DIR="${OUTPUT_DIR}/no_prompt"
fi
if [[ -z "${BASELINE_DETAIL_PATH}" ]]; then
    BASELINE_DETAIL_PATH="${BASELINE_OUTPUT_DIR}/details/results_detail.jsonl"
fi
if [[ "${BASELINE_DETAIL_PATH_SET}" -eq 1 ]]; then
    RUN_BASELINE_INFER=0
fi
if [[ "${BASELINE_OUTPUT_DIR}" == "${OUTPUT_DIR}" ]]; then
    echo "baseline_output_dir 不能和 output_dir 相同，避免覆盖热词推理结果"
    exit 1
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "infer" ]]; then
    if [[ "${RUN_BASELINE_INFER}" -eq 1 ]]; then
        BASELINE_INFER_CMD=(
            "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py"
            --ckpt "${CKPT}"
            --mode joint
            --input_scp "${INPUT_SCP}"
            --output_dir "${BASELINE_OUTPUT_DIR}"
            --gpu_ids "${GPU_IDS}"
            --batch_size "${BATCH_SIZE}"
            --dtype "${DTYPE}"
            --stream_chunk_sec "${STREAM_CHUNK_SEC}"
            --stream_left_context_sec "${STREAM_LEFT_CONTEXT_SEC}"
            --stream_right_context_sec "${STREAM_RIGHT_CONTEXT_SEC}"
            --stream_first_chunk_left_pad_sec "${STREAM_FIRST_CHUNK_LEFT_PAD_SEC}"
            --stream_window_batch_size "${STREAM_WINDOW_BATCH_SIZE}"
            --stream_window_encoder_batch_size "${STREAM_WINDOW_ENCODER_BATCH_SIZE}"
        )
        if [[ "${STREAM}" -eq 1 ]]; then
            BASELINE_INFER_CMD+=(--stream)
        fi
        echo "运行无提示词 baseline 推理：${BASELINE_OUTPUT_DIR}"
        "${BASELINE_INFER_CMD[@]}"
    fi

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
    echo "运行热词提示推理：${OUTPUT_DIR}"
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
    if [[ -f "${BASELINE_DETAIL_PATH}" ]]; then
        EVAL_CMD+=(--baseline_detail_path "${BASELINE_DETAIL_PATH}")
    else
        echo "未找到 baseline 明细，跳过 baseline 对比：${BASELINE_DETAIL_PATH}"
    fi
    if [[ -n "${UTT_HOTWORD_PATH}" ]]; then
        EVAL_CMD+=(--utt_hotword_path "${UTT_HOTWORD_PATH}")
    fi
    "${EVAL_CMD[@]}"
fi

echo "完成：${OUTPUT_DIR}"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

stage="all"
ckpt=""
input_scp=""
ref_path=""
hotword_file=""
output_dir=""
gpu_ids="0"
batch_size=16
dtype="bf16"
hotword_topk=10
hotword_pinyin_style="normal"
prompt=""
stream=1

declare -A arg_map=(
    [--stage]=stage [--ckpt]=ckpt [--input_scp]=input_scp [--ref_path]=ref_path
    [--ref_dir]=ref_path [--hotword_file]=hotword_file
    [--output_dir]=output_dir
    [--gpu_ids]=gpu_ids [--batch_size]=batch_size [--dtype]=dtype
    [--hotword_topk]=hotword_topk [--hotword_pinyin_style]=hotword_pinyin_style
    [--prompt]=prompt
)

usage() {
    echo "用法：bash $0 --ckpt CKPT --input_scp wav.scp --ref_path text --hotword_file hotwords.txt --output_dir out"
    echo "可选：--stage all|infer|eval"
    echo "      --gpu_ids 0,1 --batch_size 8 --hotword_topk 5 --hotword_pinyin_style normal|tone3 --prompt TEXT --no_stream"
}

set_arg() {
    local opt="$1"
    local value="${2:-}"
    local var="${arg_map[$opt]:-}"

    if [[ -z "${value}" || "${value}" == --* ]]; then
        echo "参数 ${opt} 缺少取值"
        usage
        exit 1
    fi

    if [[ -z "${var}" ]]; then
        echo "未知参数：${opt}"
        usage
        exit 1
    fi
    printf -v "${var}" '%s' "${value}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no_stream) stream=0; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        --*) set_arg "$1" "${2:-}"; shift 2 ;;
        *) echo "未知参数：$1"; usage; exit 1 ;;
    esac
done

if [[ -z "${output_dir}" || -z "${ref_path}" ]]; then
    usage
    exit 1
fi
if [[ "${stage}" != "eval" && ( -z "${ckpt}" || -z "${input_scp}" ) ]]; then
    echo "stage=${stage} 时必须提供 --ckpt 和 --input_scp"
    exit 1
fi
if [[ "${stage}" != "eval" && -z "${hotword_file}" ]]; then
    echo "stage=${stage} 时必须提供 --hotword_file 用于检索"
    exit 1
fi
if [[ -z "${hotword_file}" ]]; then
    echo "必须提供 --hotword_file"
    exit 1
fi

mkdir -p "${output_dir}"
detail_path="${output_dir}/details/results_detail.jsonl"

if [[ "${stage}" == "all" || "${stage}" == "infer" ]]; then
    infer_cmd=(
        python3 "${SCRIPT_DIR}/infer.py"
        --ckpt "${ckpt}"
        --mode llm,ctc
        --input_scp "${input_scp}"
        --output_dir "${output_dir}"
        --gpu_ids "${gpu_ids}"
        --batch_size "${batch_size}"
        --dtype "${dtype}"
        --hotword_file "${hotword_file}"
        --hotword_topk "${hotword_topk}"
        --hotword_pinyin_style "${hotword_pinyin_style}"
    )
    if [[ -n "${prompt}" ]]; then
        infer_cmd+=(--prompt "${prompt}")
    fi
    if [[ "${stream}" -eq 1 ]]; then
        infer_cmd+=(--stream)
    fi
    echo "运行热词提示推理：${output_dir}"
    "${infer_cmd[@]}"
fi

if [[ "${stage}" == "all" || "${stage}" == "eval" ]]; then
    eval_cmd=(
        python3 "${PROJECT_ROOT}/qwen_asr/tools/hotword_eval.py"
        --ref_path "${ref_path}"
        --detail_path "${detail_path}"
        --hotword_file "${hotword_file}"
        --output_path "${output_dir}/hotword_eval.txt"
        --badcase_path "${output_dir}/hotword_badcases.txt"
    )
    "${eval_cmd[@]}"
fi

echo "完成：${output_dir}"

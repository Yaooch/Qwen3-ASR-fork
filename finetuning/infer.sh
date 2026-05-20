#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

ckpt="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"
stage="all"
mode="llm,ctc"
input_scp="/cfs/data/private/WangYaoChi/open_datasets/aishell_hotword_test/wav.scp"
ref_dir="/cfs/data/private/WangYaoChi/open_datasets/aishell_hotword_test/text"
output_dir="/cfs/data/private/WangYaoChi/test_out/joint_ctc_14_hotword_1/hotword_aishell/ctc_no_hotword"
gpu_ids="0,1,2,3"
batch_size=128
dtype="bf16"
language=""
hotword_file=""
hotword_topk=5
hotword_pinyin_style="normal"
stream=1
wer_script="$(awk -F'"' '/^WER_SCRIPT = / {print $2; exit}' "${PROJECT_ROOT}/qwen_asr/joint/defaults.py")"
pinyin_style="tone3"
pinyin_topk_badcases=100
text_topk_badcases=100

declare -A arg_map=(
    [--ckpt]=ckpt [--stage]=stage [--mode]=mode [--input_scp]=input_scp
    [--ref_dir]=ref_dir [--output_dir]=output_dir [--gpu_ids]=gpu_ids [--batch_size]=batch_size
    [--dtype]=dtype [--language]=language
    [--hotword_file]=hotword_file [--hotword_topk]=hotword_topk [--hotword_pinyin_style]=hotword_pinyin_style
    [--wer_script]=wer_script
    [--pinyin_style]=pinyin_style [--pinyin_topk_badcases]=pinyin_topk_badcases
    [--text_topk_badcases]=text_topk_badcases
)

usage() {
    echo "用法："
    echo "  bash $0 \\"
    echo "    --ckpt /path/to/checkpoint \\"
    echo "    --mode llm,ctc \\"
    echo "    --input_scp /path/to/test.scp \\"
    echo "    --output_dir /path/to/output \\"
    echo "    --gpu_ids 0,1,2,3 \\"
    echo "    --batch_size 16 \\"
    echo "    --ref_dir /path/to/ref_dir \\"
    echo "    --stage infer,wer"
    echo ""
    echo "参数说明："
    echo "  --stage               执行阶段：all / infer / wer / pinyin / eval，支持逗号组合；默认 all"
    echo "  --ckpt                Joint checkpoint 路径"
    echo "  --mode                推理模式，逗号组合：llm / ctc / rnnt，如 llm,ctc"
    echo "  --input_scp           输入 scp 文件"
    echo "  --output_dir          输出目录"
    echo "  --gpu_ids             使用的 GPU，例如 0 或 0,1,2,3"
    echo "  --batch_size          每个进程的 batch size"
    echo "  --dtype               模型精度：bf16 / fp16 / fp32"
    echo "  --language            默认语种，可不传"
    echo "  --hotword_file        热词文件，可不传"
    echo "  --hotword_topk        热词召回数量"
    echo "  --hotword_pinyin_style 热词拼音召回风格：normal / tone3，默认 normal"
    echo "  --stream              使用 chunk-wise encoder 流式路径"
    echo "  --no_stream           关闭流式推理；原始 Qwen3-ASR 模型需配合 --mode llm 使用"
    echo "  --ref_dir             WER 参考文件路径"
    echo "  --wer_script          WER 脚本路径，默认见 qwen_asr/joint/defaults.py"
    echo "  --pinyin_style        拼音风格：normal / tone3，默认 normal"
    echo "  --pinyin_topk_badcases 拼音 badcase 输出条数，默认 100"
    echo "  --text_topk_badcases  文本 badcase 输出条数，默认 100；设为 0 输出全部"
    echo ""
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
        echo "未知参数: ${opt}"
        usage
        exit 1
    fi

    printf -v "${var}" '%s' "${value}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stream)
            stream=1
            shift 1
            ;;
        --no_stream)
            stream=0
            shift 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            set_arg "$1" "${2:-}"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

details="${output_dir}/details"

run_infer=0
run_wer=0
run_pinyin=0

IFS=',' read -ra stages <<< "${stage}"
for item in "${stages[@]}"; do
    item="$(echo "${item}" | tr '[:upper:]' '[:lower:]' | xargs)"
    case "${item}" in
        all)
            run_infer=1
            run_wer=1
            run_pinyin=1
            ;;
        eval)
            run_wer=1
            run_pinyin=1
            ;;
        infer)
            run_infer=1
            ;;
        wer)
            run_wer=1
            ;;
        pinyin)
            run_pinyin=1
            ;;
        "")
            ;;
        *)
            echo "错误：不支持的 stage: ${item}"
            echo "支持：all / infer / wer / pinyin / eval，或逗号组合如 infer,wer"
            exit 1
            ;;
    esac
done

if [[ "${run_infer}" -eq 0 && "${run_wer}" -eq 0 && "${run_pinyin}" -eq 0 ]]; then
    echo "错误：stage 为空，未选择任何执行阶段"
    exit 1
fi

if [[ -z "${output_dir}" ]]; then
    echo "错误：必须提供 --output_dir"
    exit 1
fi
if [[ "${run_infer}" -eq 1 && -z "${ckpt}" ]]; then
    echo "错误：必须提供 --ckpt"
    exit 1
fi
if [[ "${run_infer}" -eq 1 && -z "${input_scp}" ]]; then
    echo "错误：必须提供 --input_scp"
    exit 1
fi
if [[ "${run_wer}" -eq 1 || "${run_pinyin}" -eq 1 ]]; then
    if [[ -z "${ref_dir}" ]]; then
        echo "错误：必须提供 --ref_dir"
        exit 1
    fi
fi

mkdir -p "${output_dir}" "${details}"

infer_cmd=(
    python3 infer.py
    --ckpt "${ckpt}"
    --mode "${mode}"
    --input_scp "${input_scp}"
    --output_dir "${output_dir}"
    --gpu_ids "${gpu_ids}"
    --batch_size "${batch_size}"
    --dtype "${dtype}"
)

if [[ -n "${language}" ]]; then
    infer_cmd+=(--language "${language}")
fi

if [[ -n "${hotword_file}" ]]; then
    infer_cmd+=(--hotword_file "${hotword_file}")
    infer_cmd+=(--hotword_topk "${hotword_topk}")
    infer_cmd+=(--hotword_pinyin_style "${hotword_pinyin_style}")
fi

if [[ "${stream}" -eq 1 ]]; then
    infer_cmd+=(--stream)
fi

collect_result_targets() {
    targets=()
    if [[ -f "${output_dir}/results_ctc.txt" ]]; then
        targets+=("ctc:${output_dir}/results_ctc.txt")
    fi
    if [[ -f "${output_dir}/results_rnnt.txt" ]]; then
        targets+=("rnnt:${output_dir}/results_rnnt.txt")
    fi
    if [[ -f "${output_dir}/results_llm.txt" ]]; then
        targets+=("llm:${output_dir}/results_llm.txt")
    fi
    if [[ -f "${output_dir}/results_hotword_llm.txt" ]]; then
        targets+=("hotword_llm:${output_dir}/results_hotword_llm.txt")
    fi
}

if [[ "${run_infer}" -eq 1 ]]; then
    echo "============================================================"
    echo "开始推理"
    echo "============================================================"
    echo "Stage: ${stage}"
    echo "模型路径: ${ckpt}"
    echo "推理模式: ${mode}"
    echo "输入文件: ${input_scp}"
    echo "输出目录: ${output_dir}"
    echo "GPU: ${gpu_ids}"
    echo "Batch Size: ${batch_size}"
    echo "精度: ${dtype}"
    echo "Stream: ${stream}"
    echo "============================================================"

    "${infer_cmd[@]}"
fi

collect_result_targets

if [[ "${run_wer}" -eq 1 || "${run_pinyin}" -eq 1 ]]; then
if [[ "${#targets[@]}" -eq 0 ]]; then
    echo "错误：未找到推理结果文件 ${output_dir}/results_ctc.txt、results_rnnt.txt 或 results_llm.txt"
    echo "如果只跑评测，请先确认 output_dir 下已有 results_*.txt，或使用 --stage infer,wer"
    exit 1
fi
fi

if [[ "${run_wer}" -eq 1 ]]; then
echo "============================================================"
echo "开始计算 WER"
echo "============================================================"

for target in "${targets[@]}"; do
    suffix="${target%%:*}"
    result_path="${target#*:}"
    domain_path="${output_dir}/domain_${suffix}.txt"
    wer_path="${details}/wer_${suffix}.txt"
    echo "WER[${suffix}]: ${result_path}"
    python3 "${wer_script}" \
        --char=1 \
        --v=1 \
        "${ref_dir}" \
        "${result_path}" \
        "${domain_path}" \
        > "${wer_path}"
    python3 "${PROJECT_ROOT}/qwen_asr/tools/text_badcase.py" \
        --ref_path "${ref_dir}" \
        --result_path "${result_path}" \
        --badcase_path "${details}/text_badcases_${suffix}.txt" \
        --topk_badcases "${text_topk_badcases}"
done
fi

if [[ "${run_pinyin}" -eq 1 ]]; then
    echo "============================================================"
    echo "开始计算拼音评估"
    echo "============================================================"

    for target in "${targets[@]}"; do
        suffix="${target%%:*}"
        result_path="${target#*:}"
        if [[ "${suffix}" != "ctc" && "${suffix}" != "rnnt" ]]; then
            continue
        fi
        echo "拼音评估[${suffix}]: ${result_path}"
        pinyin_cmd=(
            python3 "${PROJECT_ROOT}/qwen_asr/tools/pinyin_eval.py"
            --ref_path "${ref_dir}"
            --result_path "${result_path}"
            --output_path "${output_dir}/pinyin_similarity_${suffix}.txt"
            --badcase_path "${details}/pinyin_badcases_${suffix}.txt"
            --style "${pinyin_style}"
            --topk_badcases "${pinyin_topk_badcases}"
        )

        "${pinyin_cmd[@]}"
    done
fi

echo "============================================================"
echo "执行完成"
echo "============================================================"
echo "Stage: ${stage}"
echo "明细目录: ${details}"
echo "明细文件: ${details}/results_detail.jsonl"
echo "参考文件: ${ref_dir}"
for target in "${targets[@]}"; do
    suffix="${target%%:*}"
    result_path="${target#*:}"
    echo "识别结果[${suffix}]: ${result_path}"
done
echo "WER文件: ${details}/wer_*.txt"
echo "Domain文件: ${output_dir}/domain_*.txt"
echo "拼音汇总: ${output_dir}/pinyin_similarity_*.txt"
echo "拼音Badcase: ${details}/pinyin_badcases_*.txt"
echo "文本Badcase: ${details}/text_badcases_*.txt"
echo "============================================================"

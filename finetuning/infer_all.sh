#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ckpt="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
ckpt="/cfs/data/private/hubk/Qwen3-ASR/Qwen/Qwen3-ASR-1___7B"
outdir="/cfs/data/private/WangYaoChi/test_out/qwen3_asr/offline"
mode="llm"
stage="all" 
gpu_ids="6,7"
batch_size=256
dtype="bf16"
skip_done=0
datasets_file=""
encoder_mode="offline"  # 可选：offline|stream|train_mask
wer_script="/root/scripts/compute_asr_wer_with_slu.py"
pinyin_style="tone3"
pinyin_topk_badcases=100
text_topk_badcases=300
# lora="/cfs/data/private/WangYaoChi/model/joint_ctc_50_grpo_3/lora"
# lora="/cfs/data/private/WangYaoChi/model/joint_ctc_50_nlu_2"
lora=""

# 字段：name|wav.scp|text|language。language 为空或 None 时跳过 LID 统计。
DATASETS=(
    "mandarin2|/cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_wav_2.scp|/cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_text_classify_2|Chinese"
    "yue|/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/wav.scp|/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/text|Cantonese"
    "chuan|/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/wav2.scp|/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/text2|Sichuanese"
    "aishell|/cfs/data/private/hubk/aishell_shard/chinese_test/wav.scp|/cfs/data/private/hubk/aishell_shard/chinese_test/text|Chinese"
    "aishell2|/cfs/data/private/WangYaoChi/open_datasets/aishell2/AISHELL-DEV-TEST-SET/Mic/test/wav.scp|/cfs/data/private/WangYaoChi/open_datasets/aishell2/AISHELL-DEV-TEST-SET/Mic/test/trans.txt|Chinese"
    "ws_yue|/cfs/data/private/hubk/asr_data/wenetspeech_yue/2000.scp|/cfs/data/private/hubk/asr_data/wenetspeech_yue/2000.txt|Cantonese"
    "navi|/cfs/data/private/hubk/asr_test_set/POI_ENTITY/wav.scp|/cfs/data/private/hubk/asr_test_set/POI_ENTITY/text|None"
    "media|/cfs/data/private/hubk/asr_test_set/MEDIA_ENTITY/wav.scp|/cfs/data/private/hubk/asr_test_set/MEDIA_ENTITY/text|None"
)

declare -A arg_map=(
    [--ckpt]=ckpt [--outdir]=outdir [--mode]=mode [--stage]=stage
    [--gpu_ids]=gpu_ids [--batch_size]=batch_size [--dtype]=dtype
    [--skip_done]=skip_done [--datasets_file]=datasets_file
    [--encoder_mode]=encoder_mode
    [--wer_script]=wer_script [--pinyin_style]=pinyin_style
    [--pinyin_topk_badcases]=pinyin_topk_badcases
    [--text_topk_badcases]=text_topk_badcases
    [--lora]=lora
)

usage() {
    echo "用法：bash $0 --ckpt CKPT --outdir out"
    echo "可选：--datasets_file datasets.txt 覆盖脚本内置数据集"
    echo "      --encoder_mode offline|stream|train_mask"
    echo "datasets.txt 每行：name|wav.scp|text|language"
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
        --stream) encoder_mode="stream"; shift 1 ;;
        --no_stream) encoder_mode="offline"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        --*) set_arg "$1" "${2:-}"; shift 2 ;;
        *) echo "未知参数：$1"; usage; exit 1 ;;
    esac
done

if [[ -n "${datasets_file}" ]]; then
    mapfile -t DATASETS < <(grep -v '^[[:space:]]*$' "${datasets_file}" | grep -v '^[[:space:]]*#')
fi
if [[ -z "${ckpt}" || "${#DATASETS[@]}" -eq 0 ]]; then
    usage
    exit 1
fi

summary_path="${outdir}/result.txt"

result_path() {
    local output_dir="$1"
    local kind="$2"
    echo "${output_dir}/results_${kind}.txt"
}

has_result() {
    local output_dir="$1"
    local kind="$2"
    [[ -f "$(result_path "${output_dir}" "${kind}")" ]]
}

dataset_done() {
    local output_dir="$1"
    local language="$2"
    local details="${output_dir}/details"

    [[ -f "${details}/results_detail.jsonl" ]] || return 1
    [[ -f "${details}/encoder_mode.txt" ]] || return 1
    [[ "$(tr -d '[:space:]' < "${details}/encoder_mode.txt")" == "${encoder_mode}" ]] || return 1
    IFS=',' read -ra modes <<< "${mode}"
    for item in "${modes[@]}"; do
        item="$(echo "${item}" | tr '[:upper:]' '[:lower:]' | xargs)"
        [[ -z "${item}" ]] && continue
        has_result "${output_dir}" "${item}" || return 1
        [[ -f "${output_dir}/domain_${item}.txt" ]] || return 1
        [[ -f "${details}/text_badcases_${item}.txt" ]] || return 1
        if [[ "${item}" == "ctc" || "${item}" == "rnnt" ]]; then
            [[ -f "${output_dir}/pinyin_similarity_${item}.txt" ]] || return 1
        fi
    done
    if [[ "${language}" != "None" && -n "${language}" && -f "${output_dir}/results_llm.txt" ]]; then
        [[ -f "${details}/language_llm.txt" ]] || return 1
    fi
}

language_acc() {
    local result_path="$1"
    local expected="$2"
    local output_path="$3"
    python3 - "$result_path" "$expected" "$output_path" <<'PY'
import sys

result_path, expected, output_path = sys.argv[1:4]
total = correct = 0
with open(result_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            total += 1
            correct += int(parts[2].strip() == expected)

acc = correct / total if total else 0.0
with open(output_path, "w", encoding="utf-8") as f:
    print(f"target_language: {expected}", file=f)
    print(f"total: {total}", file=f)
    print(f"correct: {correct}", file=f)
    print(f"accuracy: {acc * 100.0:.2f}%", file=f)
PY
}

metric() {
    local path="$1"
    local name="$2"
    local prefix="${3:-}"
    python3 - "$path" "$name" "$prefix" <<'PY'
import re
import sys

path, name, prefix = sys.argv[1:4]
try:
    lines = open(path, "r", encoding="utf-8").read().splitlines()
except FileNotFoundError:
    print("-")
    raise SystemExit
if prefix:
    lines = [x for x in lines if x.startswith(prefix)]
line = lines[-1] if lines else ""
match = re.search(rf"\b{re.escape(name)}\s*:\s*([0-9.]+)\s*%?", line, re.I)
print(match.group(1) if match else "-")
PY
}

maybe_lid() {
    local output_dir="$1"
    local language="$2"
    local details="${output_dir}/details"
    if [[ "${language}" == "None" || -z "${language}" || ! -f "${output_dir}/results_llm.txt" ]]; then
        return
    fi
    mkdir -p "${details}"
    language_acc "${output_dir}/results_llm.txt" "${language}" "${details}/language_llm.txt"
}

write_summary() {
    mkdir -p "${outdir}"
    {
        printf "dataset\tllm_wer\tllm_sar\tllm_lid\tctc_wer\tctc_sar\tctc_per\trnnt_wer\trnnt_sar\trnnt_per\n"
        for row in "${DATASETS[@]}"; do
            IFS='|' read -r name _scp _text _language <<< "${row}"
            local output_dir="${outdir}/${name}"
            local details="${output_dir}/details"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "${name}" \
                "$(metric "${output_dir}/domain_llm.txt" wer "Overall ->")" \
                "$(metric "${output_dir}/domain_llm.txt" sar "Overall ->")" \
                "$(metric "${details}/language_llm.txt" accuracy)" \
                "$(metric "${output_dir}/domain_ctc.txt" wer "Overall ->")" \
                "$(metric "${output_dir}/domain_ctc.txt" sar "Overall ->")" \
                "$(metric "${output_dir}/pinyin_similarity_ctc.txt" PER "Overall ->")" \
                "$(metric "${output_dir}/domain_rnnt.txt" wer "Overall ->")" \
                "$(metric "${output_dir}/domain_rnnt.txt" sar "Overall ->")" \
                "$(metric "${output_dir}/pinyin_similarity_rnnt.txt" PER "Overall ->")"
        done
    } > "${summary_path}"
}

for row in "${DATASETS[@]}"; do
    IFS='|' read -r name input_scp ref_dir language <<< "${row}"
    output_dir="${outdir}/${name}"
    echo "数据集：${name}"

    if [[ "${skip_done}" -eq 1 ]] && dataset_done "${output_dir}" "${language}"; then
        echo "已完成，跳过"
        maybe_lid "${output_dir}" "${language}"
        write_summary
        continue
    fi

    cmd=(
        bash infer.sh
        --stage "${stage}"
        --ckpt "${ckpt}"
        --mode "${mode}"
        --input_scp "${input_scp}"
        --ref_dir "${ref_dir}"
        --output_dir "${output_dir}"
        --gpu_ids "${gpu_ids}"
        --batch_size "${batch_size}"
        --dtype "${dtype}"
        --encoder_mode "${encoder_mode}"
        --pinyin_style "${pinyin_style}"
        --pinyin_topk_badcases "${pinyin_topk_badcases}"
        --text_topk_badcases "${text_topk_badcases}"
    )
    if [[ -n "${wer_script}" ]]; then
        cmd+=(--wer_script "${wer_script}")
    fi
    if [[ -n "${lora}" ]]; then
        cmd+=(--lora "${lora}")
    fi
    "${cmd[@]}"
    maybe_lid "${output_dir}" "${language}"
    write_summary
done

echo "汇总结果：${summary_path}"

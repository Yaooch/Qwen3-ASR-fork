#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 批量推理配置。按需手动修改下面这些变量和 DATASETS。
CKPT="${CKPT:-/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228/}"
EXP_CODE="${EXP_CODE:-joint_ctc_14_hotword_1}"
MODE="${MODE:-llm}"
STAGE="${STAGE:-all}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DTYPE="${DTYPE:-bf16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_PREFIX="${OUT_PREFIX:-/cfs/data/private/WangYaoChi/test_out}"
SKIP_DONE="${SKIP_DONE:-1}"
PROMPT="${PROMPT:-"转写语音，专属名词优先按列表原文输出。"}"

RNNT_MAX_SYMBOLS_PER_STEP="${RNNT_MAX_SYMBOLS_PER_STEP:-3}"
AUX_ENCODER_BATCH_SIZE="${AUX_ENCODER_BATCH_SIZE:-4}"
STREAM="${STREAM:-1}"
STREAM_CHUNK_SEC="${STREAM_CHUNK_SEC:-0.64}"
STREAM_LEFT_CONTEXT_SEC="${STREAM_LEFT_CONTEXT_SEC:-1.32}"
STREAM_RIGHT_CONTEXT_SEC="${STREAM_RIGHT_CONTEXT_SEC:-0.07}"
STREAM_FIRST_CHUNK_LEFT_PAD_SEC="${STREAM_FIRST_CHUNK_LEFT_PAD_SEC:-0.0}"
STREAM_WINDOW_BATCH_SIZE="${STREAM_WINDOW_BATCH_SIZE:-16}"
STREAM_WINDOW_ENCODER_BATCH_SIZE="${STREAM_WINDOW_ENCODER_BATCH_SIZE:-4}"

WER_SCRIPT="${WER_SCRIPT:-/root/scripts/compute_asr_wer_with_slu.py}"
PINYIN_STYLE="${PINYIN_STYLE:-tone3}"
PINYIN_TOPK_BADCASES="${PINYIN_TOPK_BADCASES:-100}"
EXP_DIR="${OUT_PREFIX}/${EXP_CODE}"
RESULT_SUMMARY_PATH="${EXP_DIR}/result.txt"

# 字段：dataset_name|scp_path|text_path|language
# language 可取：Mandarin / Cantonese / Sichuanese / English / None
DATASETS=(
    "mandarin2|/cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_wav_2.scp|/cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_text_classify_2|Chinese"
    "yue|/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/wav.scp|/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/text|Cantonese"
    "chuan|/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/wav2.scp|/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/text2|Sichuanese"
    "aishell|/cfs/data/private/hubk/aishell_shard/chinese_test/wav.scp|/cfs/data/private/hubk/aishell_shard/chinese_test/text|Chinese"
    "aishell2|/cfs/data/private/WangYaoChi/open_datasets/aishell2/AISHELL-DEV-TEST-SET/Mic/test/wav.scp|/cfs/data/private/WangYaoChi/open_datasets/aishell2/AISHELL-DEV-TEST-SET/Mic/test/trans.txt|Chinese"
    "ws_yue|/cfs/data/private/hubk/asr_data/wenetspeech_yue/2000.scp|/cfs/data/private/hubk/asr_data/wenetspeech_yue/2000.txt|Cantonese"
    # "ws_chuan|/cfs/data/private/hubk/asr_data/wenetspeech_sichuan/10000.scp|/cfs/data/private/hubk/asr_data/wenetspeech_sichuan/10000.txt|Sichuanese"
    "navi|/cfs/data/private/hubk/asr_test_set/POI_ENTITY/wav.scp|/cfs/data/private/hubk/asr_test_set/POI_ENTITY/text|None"
    "media|/cfs/data/private/hubk/asr_test_set/MEDIA_ENTITY/wav.scp|/cfs/data/private/hubk/asr_test_set/MEDIA_ENTITY/text|None"
)

language_acc() {
    local result_path="$1"
    local expected="$2"
    local output_path="$3"

    "${PYTHON_BIN}" - "$result_path" "$expected" "$output_path" <<'PY'
import sys

result_path, expected, output_path = sys.argv[1:4]
total = 0
correct = 0
with open(result_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        total += 1
        if parts[2].strip() == expected:
            correct += 1

acc = correct / total if total else 0.0
with open(output_path, "w", encoding="utf-8") as f:
    print(f"target_language: {expected}", file=f)
    print(f"total: {total}", file=f)
    print(f"correct: {correct}", file=f)
    print(f"accuracy: {acc * 100.0:.2f}%", file=f)
PY
}

extract_overall_metric() {
    local path="$1"
    local metric="$2"

    "${PYTHON_BIN}" - "$path" "$metric" <<'PY'
import re
import sys

path, metric = sys.argv[1:3]
try:
    lines = open(path, "r", encoding="utf-8").read().splitlines()
except FileNotFoundError:
    print("-")
    raise SystemExit(0)

overall = ""
for line in lines:
    if line.startswith("Overall ->"):
        overall = line

if not overall:
    print("-")
    raise SystemExit(0)

match = re.search(rf"\b{re.escape(metric)}\s*:\s*([0-9.]+)\s*%?", overall, re.IGNORECASE)
print(match.group(1) if match else "-")
PY
}

extract_language_metric() {
    local path="$1"

    "${PYTHON_BIN}" - "$path" <<'PY'
import re
import sys

path = sys.argv[1]
try:
    text = open(path, "r", encoding="utf-8").read()
except FileNotFoundError:
    print("-")
    raise SystemExit(0)

match = re.search(r"accuracy\s*:\s*([0-9.]+)\s*%?", text, re.IGNORECASE)
print(match.group(1) if match else "-")
PY
}

write_summary() {
    mkdir -p "${EXP_DIR}"
    local tmp_summary
    local target_tmp
    tmp_summary="$(mktemp "/tmp/infer_all_result.XXXXXX")"
    target_tmp="${RESULT_SUMMARY_PATH}.$$.$RANDOM.tmp"

    printf "dataset_name\tllm_wer\tllm_sar\tllm_lid\tctc_wer\tctc_sar\tctc_per\trnnt_wer\trnnt_sar\trnnt_per\n" > "${tmp_summary}"

    for item in "${DATASETS[@]}"; do
        IFS='|' read -r dataset_name _scp_path _text_path _language <<< "${item}"
        output_dir="${EXP_DIR}/${dataset_name}/stream_64_132"
        details_dir="${output_dir}/details"

        llm_wer="$(extract_overall_metric "${output_dir}/domain_llm.txt" "wer")"
        llm_sar="$(extract_overall_metric "${output_dir}/domain_llm.txt" "sar")"
        llm_lid="$(extract_language_metric "${details_dir}/language_llm.txt")"
        ctc_wer="$(extract_overall_metric "${output_dir}/domain_ctc.txt" "wer")"
        ctc_sar="$(extract_overall_metric "${output_dir}/domain_ctc.txt" "sar")"
        ctc_per="$(extract_overall_metric "${output_dir}/pinyin_similarity_ctc.txt" "PER")"
        rnnt_wer="$(extract_overall_metric "${output_dir}/domain_rnnt.txt" "wer")"
        rnnt_sar="$(extract_overall_metric "${output_dir}/domain_rnnt.txt" "sar")"
        rnnt_per="$(extract_overall_metric "${output_dir}/pinyin_similarity_rnnt.txt" "PER")"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${dataset_name}" \
            "${llm_wer}" \
            "${llm_sar}" \
            "${llm_lid}" \
            "${ctc_wer}" \
            "${ctc_sar}" \
            "${ctc_per}" \
            "${rnnt_wer}" \
            "${rnnt_sar}" \
            "${rnnt_per}" \
            >> "${tmp_summary}"
    done

    cp "${tmp_summary}" "${target_tmp}"
    mv -f "${target_tmp}" "${RESULT_SUMMARY_PATH}"
    rm -f "${tmp_summary}"
}

dataset_done() {
    local output_dir="$1"
    local language="$2"
    local details_dir="${output_dir}/details"

    case "${MODE}" in
        llm)
            [[ -f "${output_dir}/results_llm.txt" ]] || return 1
            [[ -f "${output_dir}/domain_llm.txt" ]] || return 1
            if [[ "${language}" != "None" && -n "${language}" ]]; then
                [[ -f "${details_dir}/language_llm.txt" ]] || return 1
            fi
            ;;
        ctc)
            [[ -f "${output_dir}/results_ctc.txt" ]] || return 1
            [[ -f "${output_dir}/domain_ctc.txt" ]] || return 1
            [[ -f "${output_dir}/pinyin_similarity_ctc.txt" ]] || return 1
            ;;
        rnnt)
            [[ -f "${output_dir}/results_rnnt.txt" ]] || return 1
            [[ -f "${output_dir}/domain_rnnt.txt" ]] || return 1
            [[ -f "${output_dir}/pinyin_similarity_rnnt.txt" ]] || return 1
            ;;
        joint)
            [[ -f "${details_dir}/results_detail.jsonl" ]] || return 1
            [[ -f "${output_dir}/results_llm.txt" ]] || return 1
            [[ -f "${output_dir}/domain_llm.txt" ]] || return 1
            if [[ "${language}" != "None" && -n "${language}" ]]; then
                [[ -f "${details_dir}/language_llm.txt" ]] || return 1
            fi
            if [[ -f "${output_dir}/results_ctc.txt" ]]; then
                [[ -f "${output_dir}/domain_ctc.txt" ]] || return 1
                [[ -f "${output_dir}/pinyin_similarity_ctc.txt" ]] || return 1
            elif [[ -f "${output_dir}/results_rnnt.txt" ]]; then
                [[ -f "${output_dir}/domain_rnnt.txt" ]] || return 1
                [[ -f "${output_dir}/pinyin_similarity_rnnt.txt" ]] || return 1
            else
                return 1
            fi
            ;;
        *)
            return 1
            ;;
    esac

    return 0
}

dataset_needs_lid() {
    local output_dir="$1"
    local language="$2"
    if [[ "${language}" == "None" || -z "${language}" ]]; then
        return 1
    fi
    [[ -f "${output_dir}/results_llm.txt" ]] || return 1
    [[ ! -f "${output_dir}/details/language_llm.txt" ]]
}

compute_lid_if_needed() {
    local output_dir="$1"
    local language="$2"
    local details_dir="${output_dir}/details"

    if dataset_needs_lid "${output_dir}" "${language}"; then
        mkdir -p "${details_dir}"
        language_acc \
            "${output_dir}/results_llm.txt" \
            "${language}" \
            "${details_dir}/language_llm.txt"
        echo "语种识别率: ${details_dir}/language_llm.txt"
    fi
}

for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name scp_path text_path language <<< "${item}"
    output_dir="${OUT_PREFIX}/${EXP_CODE}/${dataset_name}/stream_64_132"
    details_dir="${output_dir}/details"

    echo "============================================================"
    echo "数据集: ${dataset_name}"
    echo "输入: ${scp_path}"
    echo "参考: ${text_path}"
    echo "语种: ${language}"
    echo "输出: ${output_dir}"
    echo "============================================================"

    if [[ "${SKIP_DONE}" -eq 1 ]] && dataset_done "${output_dir}" "${language}"; then
        echo "已完成，跳过: ${dataset_name}"
        compute_lid_if_needed "${output_dir}" "${language}"
        write_summary
        echo "当前汇总结果: ${RESULT_SUMMARY_PATH}"
        continue
    fi

    cmd=(
        bash infer.sh
        --stage "${STAGE}"
        --ckpt "${CKPT}"
        --mode "${MODE}"
        --input_scp "${scp_path}"
        --ref_dir "${text_path}"
        --output_dir "${output_dir}"
        --gpu_ids "${GPU_IDS}"
        --batch_size "${BATCH_SIZE}"
        --dtype "${DTYPE}"
        --python_bin "${PYTHON_BIN}"
        --wer_script "${WER_SCRIPT}"
        --rnnt_max_symbols_per_step "${RNNT_MAX_SYMBOLS_PER_STEP}"
        --aux_encoder_batch_size "${AUX_ENCODER_BATCH_SIZE}"
        --stream_chunk_sec "${STREAM_CHUNK_SEC}"
        --stream_left_context_sec "${STREAM_LEFT_CONTEXT_SEC}"
        --stream_right_context_sec "${STREAM_RIGHT_CONTEXT_SEC}"
        --stream_first_chunk_left_pad_sec "${STREAM_FIRST_CHUNK_LEFT_PAD_SEC}"
        --stream_window_batch_size "${STREAM_WINDOW_BATCH_SIZE}"
        --stream_window_encoder_batch_size "${STREAM_WINDOW_ENCODER_BATCH_SIZE}"
        --pinyin_style "${PINYIN_STYLE}"
        --pinyin_topk_badcases "${PINYIN_TOPK_BADCASES}"
    )

    if [[ "${STREAM}" -eq 1 ]]; then
        cmd+=(--stream)
    else
        cmd+=(--no_stream)
    fi
    if [[ -n "${PROMPT}" ]]; then
        cmd+=(--prompt "${PROMPT}")
    fi

    "${cmd[@]}"

    compute_lid_if_needed "${output_dir}" "${language}"

    write_summary
    echo "当前汇总结果: ${RESULT_SUMMARY_PATH}"
done

echo "汇总结果: ${RESULT_SUMMARY_PATH}"
echo "全部数据集完成"

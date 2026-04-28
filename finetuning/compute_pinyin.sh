OUTPUT_DIR="/cfs/data/private/WangYaoChi/test_out/joint2_llm_2/mandarin"
REF_PATH="/cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_text_classify"

python compute_pinyin_similarity.py \
  --ref_path "${REF_PATH}" \
  --result_path "${OUTPUT_DIR}/results.txt" \
  --output_path "${OUTPUT_DIR}/pinyin_similarity.txt" \
  --detail_output_path "${OUTPUT_DIR}/pinyin_detail.jsonl" \
  --badcase_path "${OUTPUT_DIR}/pinyin_badcases.txt"

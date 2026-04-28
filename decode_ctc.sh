python inference_ctc.py \
    --base_model /cfs/data/private/WangYaoChi/model/qwen3-asr-finetuning-out-3/checkpoint-9375 \
    --ctc_checkpoint /cfs/data/private/WangYaoChi/model/qwen3-asr-ctc/best_model.pt \
    --scp /cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/wav2.scp \
    --output result.txt \
    --batch_size 8
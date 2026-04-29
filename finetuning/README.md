# Finetuning Entries

This directory now contains launch scripts and experiment entries only. Core
CTC/RNNT model code lives in `qwen_asr/joint/`.

## Files

```text
train.py                    Joint LLM + CTC/RNNT training entry
infer.py                    Batch inference entry
train.sh                    Training launcher
eval.sh                     Inference + WER launcher
compute_pinyin.sh           Pinyin evaluation launcher
compute_pinyin_similarity.py Compatibility wrapper
qwen3_asr_sft.py            Original Qwen3-ASR SFT baseline
```

## Train

```bash
bash train.sh "0,1,2,3"
```

Common overrides:

```bash
AUX_LOSS_TYPE=rnnt
AUX_STREAMING_TRAIN=1
BATCH_SIZE=16
GRAD_ACC=4
bash train.sh "0,1,2,3"
```

## Infer + WER

```bash
bash eval.sh \
  --ckpt /path/to/checkpoint \
  --mode rnnt \
  --input_scp /path/to/wav.scp \
  --ref_dir /path/to/text \
  --output_dir /path/to/out \
  --gpu_ids 0,1
```

Use `--stream` to enable chunk-wise streaming-style decoding.

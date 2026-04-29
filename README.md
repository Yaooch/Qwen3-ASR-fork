# Qwen3-ASR Joint Streaming Extension

This repository is based on Qwen3-ASR and adds a lightweight auxiliary ASR stack for
CTC/RNNT training, offline decoding, streaming-style decoding, and pinyin-level
evaluation.

The main goal of this fork is:

- use Qwen3-ASR as the shared audio encoder and LLM decoder;
- add CTC/RNNT heads for fast rough recognition;
- support joint training of LLM ASR loss and auxiliary CTC/RNNT loss;
- support chunk-wise streaming recognition for CTC/RNNT;
- optionally reuse chunk-wise encoder outputs for final LLM decoding;
- evaluate both text WER and pinyin-level PER/SAR.

## What Changed

Compared with the original project, this fork adds:

- `qwen_asr/joint/ctc.py`: CTC auxiliary head.
- `qwen_asr/joint/rnnt.py`: RNNT auxiliary head and greedy decoding.
- `qwen_asr/joint/model.py`: Qwen3-ASR joint wrapper for LLM/CTC/RNNT/joint modes.
- `qwen_asr/joint/hotword.py`: simple hotword retrieval used by joint decoding.
- `qwen_asr/joint/tokens.py`: shared vocabulary and SentencePiece helpers.
- `qwen_asr/tools/pinyin_eval.py`: pinyin-level evaluation tool.
- `finetuning/train.py`: joint training entry.
- `finetuning/infer.py`: batch inference entry.
- `finetuning/train.sh`: training launcher.
- `finetuning/eval.sh`: inference + WER launcher.

The auxiliary CTC/RNNT heads are now fixed to the preferred feature position:

```text
audio encoder transformer final layer -> ln_post -> auxiliary head
                                      -> proj1/proj2 -> LLM audio embeddings
```

Old checkpoint configs that contain `ctc_position` or `ctc_layer_idx` can still be
loaded, but new shell scripts no longer expose those switches.

## Current Layout

```text
qwen_asr/
  joint/
    model.py      # Joint model wrapper
    ctc.py        # CTC head
    rnnt.py       # RNNT head
    hotword.py    # Hotword retrieval
    tokens.py     # Aux vocabulary helpers
  tools/
    pinyin_eval.py

finetuning/
  train.py        # Joint training
  infer.py        # Batch inference
  train.sh        # Training launcher
  eval.sh         # Inference + WER launcher
  compute_pinyin.sh
```

## Training

Run from the repository root or from `finetuning/`:

```bash
bash finetuning/train.sh "0,1,2,3"
```

Useful environment variables:

```bash
AUX_LOSS_TYPE=rnnt      # ctc or rnnt
AUX_ONLY=0             # 1 trains only the aux head
BATCH_SIZE=16
GRAD_ACC=4
QWEN_LR=2e-5
AUX_LR=1e-3
AUX_WEIGHT=0.3
```

Streaming-style auxiliary training:

```bash
AUX_STREAMING_TRAIN=1
AUX_STREAM_CHUNK_FRAMES=64
AUX_STREAM_LEFT_CONTEXT_FRAMES=64
AUX_STREAM_RIGHT_CONTEXT_FRAMES=7
AUX_STREAM_RANDOM_LEFT=1
bash finetuning/train.sh "0,1,2,3"
```

The streaming auxiliary loss uses chunk-level encoder windows. The current chunk is
always visible. The left context can be fixed or randomly sampled; the right context
is used to compensate for limited future context.

## Inference And WER

Offline:

```bash
bash finetuning/eval.sh \
  --ckpt /path/to/checkpoint \
  --mode rnnt \
  --input_scp /path/to/wav.scp \
  --ref_dir /path/to/text \
  --output_dir /path/to/out \
  --gpu_ids 0,1,2,3
```

Streaming-style decoding:

```bash
bash finetuning/eval.sh --stream --mode rnnt
```

Modes:

```text
ctc    auxiliary CTC decode
rnnt   auxiliary RNNT decode
llm    Qwen3-ASR LLM decode
joint  auxiliary rough result + hotword context + LLM final decode
```

When `--stream` is enabled:

- `ctc` and `rnnt` decode chunk by chunk.
- `llm` encodes chunks, concatenates chunk audio embeddings, then runs one final LLM decode.
- `joint` uses streaming auxiliary recognition for rough text and streaming chunk embeddings for final LLM decode.

## Pinyin Evaluation

```bash
python -m qwen_asr.tools.pinyin_eval \
  --ref_path /path/to/ref_text \
  --result_path /path/to/results.txt \
  --output_path /path/to/pinyin.txt \
  --detail_output_path /path/to/pinyin_detail.jsonl \
  --badcase_path /path/to/pinyin_badcases.txt
```

The report contains:

- `sar`: sentence accuracy rate at pinyin-token level.
- `PER`: pinyin edit error rate.
- `N/C/S/D/I`: reference tokens, correct tokens, substitutions, deletions, insertions.

## Notes

- This fork focuses on CTC/RNNT auxiliary recognition and practical streaming-style
  experiments. The original Qwen3-ASR APIs under `qwen_asr/inference/` are kept.
- `finetuning/` contains launch scripts and training/inference entries only.
- Core joint model code lives under `qwen_asr/joint/`.
- Debug scripts and legacy standalone CTC scripts were removed during cleanup.

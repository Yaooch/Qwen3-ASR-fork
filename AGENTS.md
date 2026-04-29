# Agent Notes

This fork is a Qwen3-ASR joint CTC/RNNT streaming workspace. Treat it as an
experimental ASR codebase with production-shaped entry points.

## Main Goals

- Keep Qwen3-ASR's original inference package usable.
- Keep joint CTC/RNNT model code under `qwen_asr/joint/`.
- Keep `finetuning/` as launch and experiment entry scripts only.
- Avoid adding new debug scripts unless they are temporary and removed before handoff.

## Important Paths

```text
qwen_asr/joint/model.py       JointASR wrapper and train/infer logic
qwen_asr/joint/ctc.py         CTC head
qwen_asr/joint/rnnt.py        RNNT head
qwen_asr/joint/tokens.py      Aux vocab helpers
qwen_asr/joint/hotword.py     Hotword retrieval
qwen_asr/tools/pinyin_eval.py Pinyin-level evaluation
finetuning/train.py           Training entry
finetuning/infer.py           Batch inference entry
finetuning/train.sh           Training launcher
finetuning/eval.sh            Inference + WER launcher
```

## Current Design

The auxiliary CTC/RNNT heads should attach after the audio encoder final
`ln_post` and before `proj1/proj2`. Do not reintroduce shell-level
`CTC_POSITION` or `CTC_LAYER_IDX` switches unless the user explicitly asks for
historical experiments.

Streaming support is chunk-wise rather than a fully cached encoder:

- CTC/RNNT stream decode uses per-chunk encoder windows.
- RNNT carries predictor state across chunks.
- CTC carries previous token id across chunks for collapse.
- LLM stream mode concatenates chunk audio embeddings and runs final generation once.

## Engineering Rules

- Preserve checkpoint compatibility when practical.
- Prefer short function names plus clear docstrings over very long names.
- Keep scripts simple and explicit.
- Do not restore deleted legacy standalone CTC scripts unless requested.
- Before deleting files, check whether `finetuning/train.py`, `finetuning/infer.py`,
  or shell launchers still reference them.

## Validation

After code changes, run:

```bash
python3 -m py_compile \
  qwen_asr/joint/model.py \
  qwen_asr/joint/ctc.py \
  qwen_asr/joint/rnnt.py \
  qwen_asr/joint/tokens.py \
  qwen_asr/joint/hotword.py \
  qwen_asr/tools/pinyin_eval.py \
  finetuning/train.py \
  finetuning/infer.py

bash -n finetuning/train.sh
bash -n finetuning/eval.sh
bash -n finetuning/compute_pinyin.sh
```

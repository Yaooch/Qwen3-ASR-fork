# 开发说明

这是 Qwen3-ASR 的联合 CTC/RNNT 流式实验版。

## 重点路径

```text
qwen_asr/joint/model.py       联合模型
qwen_asr/joint/train_utils.py 流式训练窗口
qwen_asr/joint/stream.py      流式窗口和 chunk 特征
qwen_asr/joint/decode.py      推理入口
qwen_asr/joint/ctc.py         CTC 辅助头
qwen_asr/joint/rnnt.py        RNNT 辅助头
qwen_asr/joint/tokens.py      词表工具
qwen_asr/joint/hotword.py     热词召回
qwen_asr/tools/pinyin_eval.py 拼音评估
finetuning/train.py           训练入口
finetuning/infer.py           推理入口
finetuning/train.sh           训练脚本
finetuning/eval.sh            推理 + WER
```

## 代码原则

- `finetuning/` 只放入口脚本，不放核心模型。
- 核心联合模型放在 `qwen_asr/joint/`。
- 函数名尽量短，具体逻辑用中文注释说明。
- 打印输出使用简洁中文。
- 不再新增历史实验开关。
- RNNT 解码固定使用 cached greedy。
- CTC/RNNT 新训练固定接在 `ln_post` 后、`proj1/proj2` 前。

## 验证命令

```bash
python3 -m py_compile \
  qwen_asr/joint/model.py \
  qwen_asr/joint/train_utils.py \
  qwen_asr/joint/stream.py \
  qwen_asr/joint/decode.py \
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

# Finetuning 入口

这个目录只保留训练、推理和评估脚本。核心模型代码在 `qwen_asr/joint/`。

## 文件

```text
train.py          联合训练入口
infer.py          批量推理入口
train.sh          训练启动脚本
infer.sh          推理 + WER + 拼音评估脚本
infer_all.sh      多数据集批量推理和汇总脚本
hotword_eval.sh   热词专项评估脚本
qwen3_asr_sft.py  原始 SFT baseline
```

## 训练

```bash
bash train.sh "0,1,2,3"
```

常用配置直接改 `train.sh` 顶部变量。`--train` 只控制训练和 loss，不训练的已有 CTC/RNNT 头不加载，保存时原样复制。训练不再暴露额外窗口参数，短窗口由 `audio_n_window/audio_n_window_infer` 控制。
默认词表路径、SentencePiece 路径和 WER 脚本路径在 `qwen_asr/joint/defaults.py` 中维护。

## 推理

```bash
bash infer.sh \
  --ckpt /path/to/checkpoint \
  --mode llm,ctc \
  --input_scp /path/to/wav.scp \
  --ref_dir /path/to/text \
  --output_dir /path/to/out \
  --gpu_ids 0,1
```

`--mode` 只支持 `llm/ctc/rnnt` 的逗号组合，不再支持 `joint`。加 `--stream` 使用固定 chunk-wise 流式路径，具体窗口参数在 `qwen_asr/joint/defaults.py` 中修改。

热词评估会同时输出默认 LLM 和热词 prompt LLM，并比较两者的热词识别变化。

## 批量推理

```bash
bash infer_all.sh --ckpt /path/to/checkpoint --datasets_file datasets.txt --exp_dir /path/to/out
```

默认使用脚本内置数据集列表；传 `--datasets_file` 可覆盖。`datasets.txt` 每行格式：`name|wav.scp|text|language`。

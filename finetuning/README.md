# Finetuning 入口

这个目录只保留训练、推理和评估脚本。核心模型代码在 `qwen_asr/joint/`。

## 文件

```text
train.py          联合训练入口
infer.py          批量推理入口
train.sh          训练启动脚本
eval.sh           推理 + WER 脚本
compute_pinyin.sh 拼音评估脚本
qwen3_asr_sft.py  原始 SFT baseline
```

## 训练

```bash
bash train.sh "0,1,2,3"
```

常用配置：

```bash
AUX_LOSS_TYPE=rnnt
AUX_STREAMING_TRAIN=1
BATCH_SIZE=16
GRAD_ACC=4
bash train.sh "0,1,2,3"
```

## 推理

```bash
bash eval.sh \
  --ckpt /path/to/checkpoint \
  --mode rnnt \
  --input_scp /path/to/wav.scp \
  --ref_dir /path/to/text \
  --output_dir /path/to/out \
  --gpu_ids 0,1
```

加 `--stream` 使用 chunk-wise 流式路径。

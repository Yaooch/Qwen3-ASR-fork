# 开发说明

这是 Qwen3-ASR 项目，当前仓库同时包含官方 ASR 包、推理/服务入口，以及联合 CTC/RNNT 流式实验代码。

## 当前结构

```text
qwen_asr/__main__.py                         包入口提示
qwen_asr/cli/demo.py                         Gradio demo 入口
qwen_asr/cli/demo_streaming.py               流式 demo 入口
qwen_asr/cli/serve.py                        Flask 服务入口
qwen_asr/core/transformers_backend/          Transformers 后端模型、配置、processor
qwen_asr/core/vllm_backend/                  vLLM 后端适配
qwen_asr/inference/qwen3_asr.py              通用 ASR 推理封装
qwen_asr/inference/qwen3_forced_aligner.py   forced aligner 推理
qwen_asr/inference/utils.py                  推理工具函数
qwen_asr/inference/assets/                   推理所需词典等资源
qwen_asr/joint/model.py                      联合 CTC/RNNT 模型
qwen_asr/joint/train_utils.py                联合训练和流式训练窗口
qwen_asr/joint/stream.py                     流式窗口和 chunk 特征
qwen_asr/joint/decode.py                     联合模型推理入口
qwen_asr/joint/ctc.py                        CTC 辅助头
qwen_asr/joint/rnnt.py                       RNNT 辅助头和 cached greedy 解码
qwen_asr/joint/tokens.py                     词表工具
qwen_asr/joint/hotword.py                    热词召回
qwen_asr/tools/hotword_eval.py               热词评估
qwen_asr/tools/pinyin_eval.py                拼音评估
finetuning/train.py                          联合训练入口
finetuning/infer.py                          批量推理入口
finetuning/qwen3_asr_sft.py                  原始 SFT baseline
finetuning/train.sh                          训练脚本
finetuning/infer.sh                          推理脚本
finetuning/infer_all.sh                      批量推理脚本
finetuning/hotword_eval.sh                   热词评估脚本
examples/                                    Transformers/vLLM/流式/aligner 示例
docker/                                      Dockerfile
assets/                                      项目文档资源
```

## 代码原则

- `finetuning/` 只放训练、推理、评估入口脚本，不放核心模型。
- 官方包能力优先放在 `qwen_asr/core/`、`qwen_asr/inference/`、`qwen_asr/cli/`。
- 联合 CTC/RNNT 实验核心放在 `qwen_asr/joint/`。
- 工具类评估脚本放在 `qwen_asr/tools/`，shell 包装脚本放在 `finetuning/`。
- 函数名尽量短，具体逻辑用中文注释说明。
- 打印输出使用简洁中文。
- 不再新增历史实验开关。
- RNNT 解码固定使用 cached greedy。
- CTC/RNNT 新训练固定接在 `ln_post` 后、`proj1/proj2` 前。
- 每次 AI 对仓库做出代码、脚本、结构或验证流程修改后，必须同步检查并更新本 `AGENTS.md` 中相关说明，不能只改实现不改开发文档。

## 验证命令

修改 Python 代码后至少运行：

```bash
python3 -m py_compile \
  decode.py \
  decode_test.py \
  qwen_asr/__init__.py \
  qwen_asr/__main__.py \
  qwen_asr/cli/demo.py \
  qwen_asr/cli/demo_streaming.py \
  qwen_asr/cli/serve.py \
  qwen_asr/core/transformers_backend/__init__.py \
  qwen_asr/core/transformers_backend/configuration_qwen3_asr.py \
  qwen_asr/core/transformers_backend/modeling_qwen3_asr.py \
  qwen_asr/core/transformers_backend/processing_qwen3_asr.py \
  qwen_asr/core/vllm_backend/__init__.py \
  qwen_asr/core/vllm_backend/qwen3_asr.py \
  qwen_asr/inference/qwen3_asr.py \
  qwen_asr/inference/qwen3_forced_aligner.py \
  qwen_asr/inference/utils.py \
  qwen_asr/joint/__init__.py \
  qwen_asr/joint/model.py \
  qwen_asr/joint/train_utils.py \
  qwen_asr/joint/stream.py \
  qwen_asr/joint/decode.py \
  qwen_asr/joint/ctc.py \
  qwen_asr/joint/rnnt.py \
  qwen_asr/joint/tokens.py \
  qwen_asr/joint/hotword.py \
  qwen_asr/tools/__init__.py \
  qwen_asr/tools/hotword_eval.py \
  qwen_asr/tools/pinyin_eval.py \
  finetuning/train.py \
  finetuning/infer.py \
  finetuning/qwen3_asr_sft.py \
  examples/example_qwen3_asr_transformers.py \
  examples/example_qwen3_asr_vllm.py \
  examples/example_qwen3_asr_vllm_streaming.py \
  examples/example_qwen3_forced_aligner.py
```

修改 shell 脚本后运行：

```bash
bash -n finetuning/train.sh
bash -n finetuning/infer.sh
bash -n finetuning/infer_all.sh
bash -n finetuning/hotword_eval.sh
```

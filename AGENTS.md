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
qwen_asr/joint/stream.py                     流式窗口和 chunk 特征
qwen_asr/joint/decode.py                     联合模型推理入口
qwen_asr/joint/ctc.py                        CTC 辅助头
qwen_asr/joint/rnnt.py                       RNNT 辅助头和 cached greedy 解码
qwen_asr/joint/hotword.py                    热词召回
qwen_asr/joint/defaults.py                   默认提示词和内部推理常量
qwen_asr/tools/hotword_eval.py               热词评估
qwen_asr/tools/pinyin_eval.py                拼音评估
qwen_asr/tools/text_badcase.py              文本 badcase 生成
qwen_asr/tools/sample_train_stats.py        训练集随机抽样统计
qwen_asr/tools/check_stream_alignment.py    流式训练与真实流式推理逐层对齐检查
qwen_asr/tools/cut_contextasr_dialogue.py    ContextASR Dialogue 按时间戳切分并生成训练 jsonl
qwen_asr/tools/split_contextasr_dialogue_eval.py ContextASR Dialogue 抽取热词测试集和验证集
qwen_asr/tools/split_hotword_test_by_lang.py 热词测试集按中文/英文拆分
finetuning/train.py                          联合训练入口
finetuning/infer.py                          批量推理入口
finetuning/qwen3_asr_sft.py                  原始 SFT baseline
finetuning/train.sh                          训练脚本
finetuning/infer.sh                          推理脚本
finetuning/infer_all.sh                      多数据集批量推理脚本
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
- CTC adapter 支持 `mlp/moe/transformer` 配置，外部 head 仍固定叫 `ctc`；`transformer` 采用 FunASR 风格 `encoder_dim -> 2048 -> 512` 前置 MLP、残差投影通路、5 层 pre-norm `d_model=512/nhead=8/ffn=128` Transformer 和 `Linear(512 -> vocab)`，`joint_config.json` 记录 `ctc_adapter` 与相关 `ctc_*` 结构字段，旧 checkpoint 默认 `mlp`。
- CTC/RNNT 新训练固定接在 `ln_post` 后、`proj1/proj2` 前。
- 推理不再使用 `joint` 模式；`--mode` 只接受 `llm/ctc/rnnt` 的逗号组合。
- 联合训练和批量推理入口默认显式传入 `attn_implementation="flash_attention_2"`，并同步 audio encoder 配置，确保 packed batch 的 `cu_seqlens` 分段由 FA2 隔离；未安装 flash-attn 时不要盲目调大 encoder batch。
- 训练不再使用 `train_mode/aux_loss_type`；`--train` 只接受 `llm/proj/encoder/ctc/rnnt` 的逗号组合，`proj` 对应 `audio_tower.proj1/act/proj2`。
- 训练不再使用额外窗口参数；短窗口由 `audio_n_window/audio_n_window_infer` 控制。
- CTC 可通过 `finetuning/train.py --stream_train 1` 启用流式一致训练；默认关闭，训练侧使用 WeNet-like chunk mask：整条 feature 过 CNN 后用固定 640ms chunk + `STREAM_LEFT_CHUNKS` 个左侧 chunk 的块状 attention mask 编码，不使用 FA2 packed-window 折中；若 CTC 使用 transformer adapter，CTC adapter 也使用同样 chunk mask；若同时训练 LLM，则复用该流式 encoder 输出再过 `proj1/act/proj2`，不额外重跑整条 encoder；训练 batch 不携带未使用的 raw waveform；CTC-only 训练不构造 LLM `input_ids/labels`；训练期验证样例和 CER 也必须走同一 stream mask 路径。
- 批量推理使用 `--encoder_mode offline/stream/train_mask` 选择 Encoder 路径，兼容保留 `--stream/--no_stream` 作为 `stream/offline` 别名；`train_mask` 仅用于评测流式训练路径的理论上限：整条音频批量提 Mel 后复用训练侧 chunk mask Encoder，CTC transformer adapter 同样使用 chunk mask，不代表线上真实流式行为；`infer_all.sh` 默认使用 `train_mask`，其他推理入口保持原默认值。
- 推理流式细节固定在 `qwen_asr/joint/defaults.py`；真实流式批量推理需把 batch 内窗口合批送 encoder，CTC/RNNT 流式解码需先拼接 batch 内 chunk 后批量 greedy，避免逐条音频小 batch 导致 GPU 空转。
- 默认 prompt、训练词表路径、训练 SentencePiece 路径和 WER 脚本路径统一放在 `qwen_asr/joint/defaults.py`。
- 新 joint checkpoint 使用 `joint_config.json` 记录 `heads` 等结构信息；`--train` 只控制训练/loss，不训练的已有头不加载、保存时从源 checkpoint 复制。
- 训练输出根目录和 `checkpoint-*` 都必须可直接推理，保存时需要包含 processor/tokenizer 相关配置文件，并保持 `config.json` 与 `joint_config.json` 的 audio window 参数一致。
- 训练 TensorBoard 日志路径通过 `finetuning/train.py --logging_dir` 控制，`finetuning/train.sh` 顶部保留 `logging_dir` 变量；训练期验证只记录 `eval_ctc_cer/eval_rnnt_cer`，临时诊断只打印长度分桶 CER、空输出语种分布和 CTC blank 统计；CTC projection 的 blank bias 默认负初始化以减轻早期 blank collapse；CTC transformer adapter 内部关闭 PyTorch MHA eval fastpath，避免 no_grad 下 padding batch 短样本被解成空；混合精度下 CTC loss 固定用 fp32 计算，遇到 NaN/Inf 直接停止训练。
- 热词召回固定使用 pinyin，缺少 `pypinyin` 直接报错，不做静默兜底。
- 热词评估只比较默认 LLM 与热词 prompt LLM；推理热词库和评估目标热词分开，评估目标文件使用 `utt_id<TAB>热词1,热词2` 格式；badcase 保留问题分组标题，单条只输出 `utt_id/target/retrieved/ref/llm/final`。
- 热词测试集按语言拆分时，`wav.scp/text/utt_hotword.txt` 按 `utt_id` 或音频路径判断语言，默认输出到测试集目录下的 `Mandarin/` 和 `English/`，`hotword.txt` 从拆分后的目标热词重新生成。
- 拼音评估需兼容 `utt_id<TAB>文本` 和 `utt_id 空格 文本` 两种格式；中英混合文本保留 ASCII token，汉字转拼音后一起计算 PER/SAR。
- WER 阶段需同时生成文本 badcase，输出 `utt_id/WER/ref/hyp` 四项；shell 脚本不再传自定义 prompt，统一使用 `qwen_asr/joint/defaults.py` 中默认值。
- 训练集统计优先使用随机 seek 抽样和 badcase 关键词定向检索，避免对千万级 jsonl 做全量扫描。
- 流式特征状态、单条/批量返回、评测 badcase 分类等重复逻辑优先复用现有 helper，不在入口函数里重新展开。
- `--encoder_mode stream`（或 `--stream`）推理每 640ms 处理一个 waveform chunk：前端仅保留 20ms raw tail 并提交全部新增 Mel，接受末帧约 2.5ms STFT 右上下文缺失的近似；CNN 保留 8 帧左侧 Mel overlap 并丢掉重复的 1 个 CNN token，使 stride=8 网格与训练侧对齐；batch 内 active chunk 合批送 encoder KV cache；CTC 固定使用 batched greedy，若 CTC 使用 transformer adapter 则在拼接后的 encoder 输出上使用 causal mask；`--mode ctc,llm --stream` 在音频结束后复用 encoder 输出一次性 LLM 解码。
- 流式对齐检查使用 `qwen_asr/tools/check_stream_alignment.py`，依次比较整条/增量 Mel、整条/overlap CNN、chunk mask/KV cache Encoder 和最终 CTC 输出。
- 批量推理入口需在 worker 写入 `tmp_rank*.jsonl` 前创建输出目录；多 GPU `spawn` 只传轻量 rank/world size 等参数，由子进程自行读取 scp 并分片，避免通过启动 pipe 传递大 shard 导致 worker 串行拉起。
- shell 脚本参数解析避免为每个 `--arg` 手写重复 `case` 分支，优先使用通用赋值 helper，只保留布尔开关和特殊副作用分支。
- 每次 AI 对仓库做出代码、脚本、结构或验证流程修改后，必须同步检查并更新本 `AGENTS.md` 中相关说明，不能只改实现不改开发文档。

## 验证命令

按修改范围做最小必要验证，不需要每次跑全仓命令：

- 修改 Python 文件后，至少对改动过的 `.py` 文件运行 `python3 -m py_compile <file...>`。
- 修改 shell 脚本后，至少对改动过的 `.sh` 文件运行 `bash -n <file...>`。
- 如果改动影响共享接口、训练/推理主链路或跨模块行为，再按实际影响范围扩大验证。

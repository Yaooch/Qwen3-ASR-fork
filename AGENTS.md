# 开发说明

Qwen3-ASR 仓库包含官方 ASR 包、推理/服务入口，以及联合 CTC/RNNT 流式实验代码。实验环境：`conda activate qwen3-asr`。

## 目录边界

- 官方能力放在 `qwen_asr/core/`、`qwen_asr/inference/`、`qwen_asr/cli/`。
- 联合 CTC/RNNT/拼音 CTC 实验核心放在 `qwen_asr/joint/`：`model.py` 负责训练 forward 和 `transcribe` 主推理入口，`encoder.py` 负责 Encoder、长度换算和流式 KV cache，`ctc.py/rnnt.py/hotword.py/defaults.py` 分别放对应核心逻辑。
- `finetuning/` 只放训练、推理、评估入口脚本；通用评估和数据处理工具放在 `qwen_asr/tools/`。
- 默认 prompt、训练词表路径、训练 SentencePiece 路径、WER 脚本路径和流式常量统一放在 `qwen_asr/joint/defaults.py`。

## 代码原则

- 保持最小修改：一两行能解决的直接在原逻辑处写，不为单次简单逻辑新增 helper、兼容开关或历史实验开关。
- 复用现有 helper：训练/推理入口优先复用 `defaults.py`、`model.py` 的配置解析 helper 和 `encoder.py` 的长度/流式 helper，不把核心模型逻辑塞进入口脚本。
- 不再通过 Mixin/MRO 扩展 joint 推理入口；`Qwen3ASRJointModel.transcribe` 是主入口。
- 函数名尽量短，必要注释用中文；打印输出保持简洁中文。
- 只在改动较大、实现与本文档不符，或确有必要时更新 `AGENTS.md`；普通 debug 或几行修复不需要同步改文档。

## 联合训练约定

- `--train` 只接受 `llm/proj/encoder/ctc/ctc_pinyin/rnnt` 的逗号组合；`proj` 对应 `audio_tower.proj1/act/proj2`，不再使用 `train_mode/aux_loss_type`。
- 文字 CTC head 固定叫 `ctc`，拼音 CTC head 固定叫 `ctc_pinyin`，RNNT 固定 cached greedy 解码；CTC/RNNT 新训练固定接在 `ln_post` 后、`proj1/proj2` 前。
- CTC adapter 支持 `mlp/moe`，文字和拼音 CTC 结构相同、共享 encoder 输出，`joint_config.json` 记录共享 `ctc_adapter`，旧 checkpoint 默认 `mlp`。
- 拼音 CTC 词表由 `qwen_asr/tools/build_pinyin_ctc_vocab.py` 生成：中文用普通话 `pypinyin Style.TONE3`，`neutral_tone_with_five=True`，`v` 表示 `ü`；方言暂不处理；英文/ASCII token 从原训练词表保留。
- CTC-only 训练不构造 LLM `input_ids/labels`；训练 batch 不携带未使用的 raw waveform；checkpoint 保存统一走 `JointTrainer.save_model`，输出根目录和 `checkpoint-*` 都应可直接推理。
- 训练验证指标记录 `eval_ctc_cer/eval_ctc_pinyin_per/eval_rnnt_cer`，TensorBoard 日志路径由 `finetuning/train.py --logging_dir` 控制。

## 流式和推理约定

- 联合训练和批量推理默认使用 `DEFAULT_ATTN_IMPLEMENTATION="flash_attention_2"`；未安装 flash-attn 时不要盲目调大 encoder batch。
- `--stream_train 1` 使用训练侧 WeNet-like 帧级 chunk mask：默认左看 24 帧、当前 6 帧、右看 2 帧；若同时训练 LLM，复用该流式 encoder 输出再过 `proj1/act/proj2`。
- 批量推理用 `--encoder_mode offline/stream/train_mask`；`train_mask` 是训练侧 chunk mask 理论上限，不代表线上真实流式行为；兼容保留 `--stream/--no_stream` 作为别名。
- `--mode` 只接受 `llm/ctc/ctc_pinyin/rnnt` 的逗号组合，不再使用 `joint` 模式。
- 真实流式推理每 640ms 处理一个 waveform chunk，batch 内 active chunk 合批送 encoder KV cache；CTC/RNNT 流式解码固定 batched greedy。
- 流式对齐检查使用 `qwen_asr/tools/check_stream_alignment.py`；多 GPU 批量推理由子进程自行读取 scp 并分片，避免通过启动 pipe 传递大 shard。

## 热词和评估约定

- 热词召回固定使用 pinyin，缺少 `pypinyin` 直接报错；推理结果包含 `ctc_pinyin_text` 时，热词检索优先使用该有调拼音序列。
- 热词推理可用 `--keep_origin_llm 0/1` 控制是否保留默认 LLM 输出，默认保留；热词库和评估目标热词分开，评估目标格式为 `utt_id<TAB>热词1,热词2`。
- 热词测试集按语言拆分输出到测试集目录下的 `Mandarin/` 和 `English/`，`hotword.txt` 从拆分后的目标热词重新生成。
- 拼音评估兼容 `utt_id<TAB>文本` 和 `utt_id 空格 文本`；中英混合文本保留 ASCII token，汉字转拼音后一起计算 PER/SAR。
- WER 阶段同时生成文本 badcase，输出 `utt_id/WER/ref/hyp`；badcase 保留问题分组标题，单条只输出 `utt_id/target/retrieved/ref/llm/final`。

## 脚本和验证

- Shell 参数解析避免为每个 `--arg` 手写重复 `case` 分支，优先使用通用赋值 helper，只保留布尔开关和特殊副作用分支。
- 训练集统计优先使用随机 seek 抽样和 badcase 关键词定向检索，避免对千万级 jsonl 做全量扫描。
- 修改 Python 文件后，至少运行 `python3 -m py_compile <file...>`。
- 修改 shell 脚本后，至少运行 `bash -n <file...>`。
- 如果改动影响共享接口、训练/推理主链路或跨模块行为，再按实际影响范围扩大验证。

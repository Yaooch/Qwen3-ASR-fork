DEFAULT_PROMPT = ""
HOTWORD_PROMPT = "转写语音。专属名词列表仅作为候选参考：如果语音中出现某个专属名词，按列表中的完整原文输出；不要拆分、改写或组合不同候选词中的字；不要输出列表中未出现的混合专属名词；未听到的候选词不要写入结果。"
JOINT_CONFIG = "joint_config.json"
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"

TRAIN_VOCAB_PATH = "/nfsdir/hubk/sensevoice_training/wenet/examples/voyah/s0/data/dict/lang_char_large_yue.txt"
TRAIN_SP_MODEL_PATH = "/nfsdir/hubk/sensevoice_training/wenet/examples/voyah/s0/data/dict/train_960_unigram5000.model"
WER_SCRIPT = "/root/scripts/compute_asr_wer_with_slu.py"

RNNT_MAX_SYMBOLS = 3

TRAIN_MASK_LEFT_FRAMES = 24
TRAIN_MASK_CURRENT_FRAMES = 8
TRAIN_MASK_RIGHT_FRAMES = 0
STREAM_CNN_LEFT_FRAMES = 8


def hotword_prompt(words, base: str = DEFAULT_PROMPT) -> str:
    if not words:
        return base
    text = "专属名词：[" + "，".join(words) + "]"
    if "专属名词：[]" in base:
        return base.replace("专属名词：[]", text, 1)
    return ((base or HOTWORD_PROMPT).rstrip() + "\n" + text).strip()

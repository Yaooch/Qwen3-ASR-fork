DEFAULT_PROMPT = ""
HOTWORD_PROMPT = "转写语音，专属名词优先按列表原文输出。"
JOINT_CONFIG = "joint_config.json"

TRAIN_VOCAB_PATH = "/nfsdir/hubk/sensevoice_training/wenet/examples/voyah/s0/data/dict/lang_char_large_yue.txt"
TRAIN_SP_MODEL_PATH = "/nfsdir/hubk/sensevoice_training/wenet/examples/voyah/s0/data/dict/train_960_unigram5000.model"
WER_SCRIPT = "/root/scripts/compute_asr_wer_with_slu.py"

ENCODER_BATCH_SIZE = 4
RNNT_MAX_SYMBOLS = 3

STREAM_CHUNK_SEC = 0.64
STREAM_LEFT_SEC = 1.36
STREAM_RIGHT_SEC = 0.07
STREAM_FIRST_PAD_SEC = 0.0
STREAM_WINDOW_BATCH = 16
STREAM_ENCODER_BATCH = 16


def hotword_prompt(words, base: str = DEFAULT_PROMPT) -> str:
    if not words:
        return base
    text = "专属名词：[" + "，".join(words) + "]"
    if "专属名词：[]" in base:
        return base.replace("专属名词：[]", text, 1)
    return ((base or HOTWORD_PROMPT).rstrip() + "\n" + text).strip()

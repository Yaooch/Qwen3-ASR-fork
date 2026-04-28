# training/utils.py
import os
import re
import shutil
from typing import Optional

import librosa

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step, best_path = None, None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step, best_path = step, path
    return best_path


def load_audio(path: str, sr: int = 16000, max_duration: float = 30.0):
    try:
        wav, _ = librosa.load(path, sr=sr, mono=True)
        if len(wav) / sr > max_duration:
            print(f"Warning: Audio too long, skipping: {path}")
            return None
        return wav
    except Exception as e:
        print(f"Warning: Failed to load audio {path}: {e}")
        return None


def patch_outer_forward(model):
    """让 Qwen3ASRForConditionalGeneration.forward 直接转发到 thinker.forward。
    避免它自己做 audio_tower，从而让 JointModel 能复用底座。"""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: model has no `.thinker.forward`.")

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features, feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs,
        )
    cls.forward = forward
    cls._forward_patched = True


_REQUIRED_HF_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
]


def copy_required_hf_files(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for fn in _REQUIRED_HF_FILES:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))

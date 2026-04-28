# training/collator.py
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from qwen_joint.tokenize_utils import text_to_ctc_ids
from .utils import load_audio


def build_prefix_messages(prompt, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex):
        prefix_msgs = build_prefix_messages(ex.get("prompt", ""), None)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": ex.get("prompt", ""),
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }
    return _preprocess


@dataclass
class DataCollatorForJointTraining:
    processor: Any
    vocab: dict
    sp_model: Any
    sampling_rate: int = 16000

    def __post_init__(self):
        self.corrupted_count = 0

    def __call__(self, features: List[Dict[str, Any]]):
        valid = []
        for f in features:
            wav = load_audio(f["audio"], sr=self.sampling_rate)
            if wav is not None:
                f["_audio"] = wav
                valid.append(f)
            else:
                self.corrupted_count += 1
        if len(valid) == 0:
            return None

        prefix_texts = [f["prefix_text"] for f in valid]
        targets = [f["target"] for f in valid]
        audios = [f["_audio"] for f in valid]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefix_texts, targets)]

        full_inputs = self.processor(
            text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False,
        )
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full_inputs["labels"] = labels

        # CTC targets
        ctc_ids_list = []
        for t in targets:
            tt = t.split("<asr_text>")[-1] if "<asr_text>" in t else t
            ctc_ids_list.append(text_to_ctc_ids(tt, self.vocab, self.sp_model))

        B = len(valid)
        max_len = max((len(x) for x in ctc_ids_list), default=0)
        ctc_target_ids = torch.zeros(B, max_len, dtype=torch.long)
        ctc_target_lengths = torch.zeros(B, dtype=torch.long)
        for i, ids in enumerate(ctc_ids_list):
            if ids:
                ctc_target_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            ctc_target_lengths[i] = len(ids)

        full_inputs["ctc_target_ids"] = ctc_target_ids
        full_inputs["ctc_target_lengths"] = ctc_target_lengths
        full_inputs["texts"] = targets
        return full_inputs

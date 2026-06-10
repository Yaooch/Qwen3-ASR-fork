# qwen_asr/joint/model.py
import json
import os
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

from .ctc import CTC
from .defaults import (
    DEFAULT_PROMPT, JOINT_CONFIG, RNNT_MAX_SYMBOLS,
    TRAIN_MASK_CURRENT_FRAMES, TRAIN_MASK_LEFT_FRAMES, TRAIN_MASK_RIGHT_FRAMES,
    hotword_prompt,
)
from .encoder import encode_offline, encode_stream, encode_train_mask, feature_lens
from .rnnt import RNNT

ENCODER_MODES = {"offline", "stream", "train_mask"}


def names(values, allowed, label):
    values = values.split(",") if isinstance(values, str) else values
    out = []
    for value in values:
        item = str(value).strip().lower()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"不支持的 {label}: {item}")
        if item not in out:
            out.append(item)
    return tuple(out)


def read_cfg(path: Optional[str]) -> Dict:
    cfg_path = os.path.join(path or "", JOINT_CONFIG)
    if not path or not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ids_to_text(ids: List[int], vocab: Dict[int, str]) -> str:
    return "".join(vocab.get(i, "") for i in ids).replace("▁", " ").strip().lower()


class Qwen3ASRJointModel(nn.Module):
    """Qwen3-ASR + CTC/RNNT 辅助头；训练和推理主入口都在这里。"""

    def __init__(
        self,
        qwen_model,
        vocab_size: int,
        vocab: Dict[str, int],
        blank_id: int = 0,
        heads: Iterable[str] = ("ctc",),
        train_tasks: Iterable[str] = ("llm", "ctc"),
        loss_weights: Optional[Dict[str, float]] = None,
        ctc_config: Optional[Dict] = None,
        stream_train: bool = False,
    ):
        super().__init__()
        self.qwen_model = qwen_model
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self._id_to_token = {v: k for k, v in vocab.items()}
        self.heads = names(heads, {"ctc", "rnnt"}, "heads")
        self.train_tasks = names(train_tasks, {"llm", "ctc", "rnnt"}, "train_tasks")
        self.loss_weights = {"llm": 0.8, "ctc": 0.2, "rnnt": 0.2}
        if loss_weights:
            self.loss_weights.update({k: float(v) for k, v in loss_weights.items()})
        self.stream_train = stream_train
        self.encoder_output_size = qwen_model.thinker.audio_tower.config.d_model
        self.ctc = CTC(vocab_size, self.encoder_output_size, blank_id=blank_id, **(ctc_config or {})) if "ctc" in self.heads else None
        self.rnnt = RNNT(vocab_size, self.encoder_output_size, blank_id=blank_id) if "rnnt" in self.heads else None
        self.processor = None
        self._asr_wrapper = None
        for p in self.parameters():
            p.requires_grad = True

    def head(self, name: str):
        head = getattr(self, name, None)
        if head is None:
            raise RuntimeError(f"当前 checkpoint 没有 {name.upper()} 头。")
        return head

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[Union[str, dict]] = "auto",
        load_heads: bool = True,
        **kwargs,
    ) -> "Qwen3ASRJointModel":
        base = Qwen3ASRModel.from_pretrained(model_path, dtype=dtype, device_map=device_map, **kwargs)
        cfg = read_cfg(model_path)
        if not cfg:
            raise FileNotFoundError(f"未找到配置：{os.path.join(model_path, JOINT_CONFIG)}")
        if "heads" not in cfg:
            raise ValueError(f"joint checkpoint 配置缺少 heads：{os.path.join(model_path, JOINT_CONFIG)}")

        tower = base.model.thinker.audio_tower
        audio_cfg = base.model.config.thinker_config.audio_config
        # checkpoint 里的窗口参数要同步到运行时 audio tower 和 HF config。
        for key in ("audio_n_window", "audio_n_window_infer"):
            value = cfg.get(key, 0)
            if value > 0:
                attr = key.replace("audio_", "")
                setattr(tower, attr, value)
                setattr(tower.config, attr, value)
                setattr(audio_cfg, attr, value)

        model = cls(
            qwen_model=base.model,
            vocab_size=cfg["vocab_size"],
            vocab=cfg.get("vocab", {}),
            blank_id=cfg.get("blank_id", 0),
            heads=cfg["heads"],
            train_tasks=("llm", *cfg["heads"]),
            ctc_config={"adapter_type": cfg.get("ctc_adapter", "mlp")},
        )
        model.processor = base.processor
        model._asr_wrapper = base
        if load_heads:
            ref = next(base.model.parameters())
            for name in model.heads:
                path = os.path.join(model_path, f"{name}_head.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"未找到 {name.upper()} 权重：{path}")
                print(f"正在加载 {name.upper()} 头：{path}")
                model.head(name).load_state_dict(torch.load(path, map_location="cpu"), strict=True)
                model.head(name).to(device=ref.device)
        if hasattr(model.qwen_model, "tie_weights"):
            model.qwen_model.tie_weights()
        return model

    def save_aux(self, output_dir: str, heads=None, copy_heads_from: Optional[str] = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        heads = names(self.heads if heads is None else heads, {"ctc", "rnnt"}, "heads")
        source_cfg = read_cfg(copy_heads_from)
        for name in heads:
            dst = os.path.join(output_dir, f"{name}_head.pt")
            head = getattr(self, name, None)
            if head is not None:
                torch.save(head.state_dict(), dst)
                continue
            src = os.path.join(copy_heads_from or "", f"{name}_head.pt")
            if not os.path.exists(src):
                raise FileNotFoundError(f"无法保存 {name.upper()} 头，未找到可复制权重：{src}")
            shutil.copy2(src, dst)

        cfg = {
            "vocab_size": self.vocab_size,
            "encoder_output_size": self.encoder_output_size,
            "blank_id": self.blank_id,
            "heads": list(heads),
            "audio_n_window": self.qwen_model.thinker.audio_tower.n_window,
            "audio_n_window_infer": self.qwen_model.thinker.audio_tower.n_window_infer,
            "vocab": self.vocab,
        }
        if "ctc" in heads:
            cfg["ctc_adapter"] = self.ctc.adapter_type if self.ctc is not None else source_cfg.get("ctc_adapter", "mlp")
        with open(os.path.join(output_dir, JOINT_CONFIG), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def decode_aux(self, name, hs, lens, max_symbols_per_step=RNNT_MAX_SYMBOLS):
        head = self.head(name)
        ids = head.greedy_decode(hs.to(next(head.parameters()).dtype), lens, max_symbols_per_step=max_symbols_per_step)
        return [ids_to_text(x, self._id_to_token) for x in ids]

    @torch.no_grad()
    def decode_llm(self, feats_list, contexts, languages, max_new_tokens=None):
        if self._asr_wrapper is None:
            raise RuntimeError("模型未正确初始化，请使用 from_pretrained 加载。")
        thinker, processor = self.qwen_model.thinker, self.processor
        device, dtype = next(self.qwen_model.parameters()).device, next(self.qwen_model.thinker.parameters()).dtype
        token = processor.audio_token
        prompts = []
        for feats, context, language in zip(feats_list, contexts, languages):
            text = self._asr_wrapper._build_text_prompt(context=context or "", force_language=language)
            prompts.append(text.replace(token, token * int(feats.shape[0]), 1))
        old_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            tok = processor.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            processor.tokenizer.padding_side = old_side
        input_ids, attn = tok["input_ids"].to(device), tok["attention_mask"].to(device)
        embeds = thinker.get_input_embeddings()(input_ids)
        audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=embeds)
        audio_feats = torch.cat([x.to(device=device, dtype=dtype) for x in feats_list], dim=0)
        if int(audio_mask[..., 0].sum().item()) != int(audio_feats.shape[0]):
            raise RuntimeError(f"audio placeholder mismatch: prompt={int(audio_mask[..., 0].sum().item())}, features={audio_feats.shape[0]}")
        gen = self.qwen_model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            inputs_embeds=embeds.masked_scatter(audio_mask, audio_feats),
            max_new_tokens=max_new_tokens or getattr(self._asr_wrapper, "max_new_tokens", 512),
        )
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        raws = processor.batch_decode(seq[:, input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        out = []
        for raw, lang in zip(raws, languages):
            parsed_lang, text = parse_asr_output(raw, user_language=lang)
            out.append({"text": text, "language": parsed_lang or lang})
        return out

    @torch.no_grad()
    def transcribe(
        self,
        audio,
        modes: Union[str, Sequence[str]] = "llm",
        language: Optional[Union[str, List[str]]] = None,
        prompt: Optional[str] = None,
        hotword_retriever=None,
        hotword_topk: int = 10,
        keep_origin_llm: bool = True,
        stream: bool = False,
        encoder_mode: Optional[str] = None,
        max_symbols_per_step: int = RNNT_MAX_SYMBOLS,
        **kwargs,
    ):
        modes = names(modes, {"llm", "ctc", "rnnt"}, "mode")
        encoder_mode = (encoder_mode or ("stream" if stream else "offline")).strip().lower()
        if encoder_mode not in ENCODER_MODES:
            raise ValueError(f"不支持的 encoder_mode: {encoder_mode}")
        if stream and encoder_mode != "stream":
            raise ValueError("stream=True 与 encoder_mode 冲突。")
        if encoder_mode == "train_mask" and not ({"ctc", "rnnt"} & set(modes)):
            raise RuntimeError("train_mask 需要同时启用 CTC 或 RNNT，以复用流式训练 Encoder 路径。")
        if encoder_mode == "stream" and "llm" in modes and "ctc" not in modes:
            raise RuntimeError("流式 LLM 需要同时启用 CTC，以复用 CTC 流式 Encoder 输出。")
        for name in ("ctc", "rnnt"):
            if name in modes:
                self.head(name)

        items = [audio] if isinstance(audio, (str, np.ndarray)) else audio
        wavs = [
            librosa.load(x, sr=16000, mono=True)[0].astype(np.float32, copy=False)
            if isinstance(x, str) else x.astype(np.float32, copy=False)
            for x in items
        ]
        langs = [None] * len(wavs) if language is None else ([language] * len(wavs) if isinstance(language, str) else list(language))
        records = [{"text": "", "language": lang, "hotwords": []} for lang in langs]
        need_llm = "llm" in modes
        # 三种 encoder path 只在这里分流，后面的 CTC/RNNT/LLM 解码共用同一套结果。
        if encoder_mode == "stream":
            ref = next(self.qwen_model.parameters())
            chunks, llm_list = encode_stream(
                self.qwen_model.thinker.audio_tower, self.processor.feature_extractor, wavs, ref, need_llm,
            )
            seqs = [torch.cat(x, dim=0) for x in chunks]
            lens = torch.tensor([x.shape[0] for x in seqs], dtype=torch.long, device=seqs[0].device)
            hs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        else:
            ref = next(self.qwen_model.parameters())
            fe = self.processor.feature_extractor
            batch = fe(
                wavs,
                sampling_rate=int(getattr(fe, "sampling_rate", 16000) or 16000),
                return_tensors="pt",
                padding=True,
                truncation=False,
                return_attention_mask=True,
            )
            feats = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
            mask = batch.get("feature_attention_mask", batch.get("attention_mask"))
            mask = mask.to(device=ref.device) if mask is not None else None
            tower = self.qwen_model.thinker.audio_tower
            feat_lengths = feature_lens(feats, mask)
            if encoder_mode == "train_mask":
                hs, llm, lens = encode_train_mask(
                    tower, feats, feat_lengths,
                    TRAIN_MASK_LEFT_FRAMES, TRAIN_MASK_CURRENT_FRAMES, TRAIN_MASK_RIGHT_FRAMES,
                    need_llm,
                )
            else:
                hs, llm, lens = encode_offline(tower, feats, feat_lengths, need_llm)
            llm_list = [] if not need_llm else [llm[sum(lens[:i]).item():sum(lens[:i + 1]).item()] for i in range(len(lens))]

        for name in ("ctc", "rnnt"):
            if name in modes:
                for rec, text in zip(records, self.decode_aux(name, hs, lens, max_symbols_per_step)):
                    rec[f"{name}_text"] = text

        base_prompt = prompt or DEFAULT_PROMPT
        if need_llm and (hotword_retriever is None or keep_origin_llm):
            for rec, out in zip(records, self.decode_llm(llm_list, [base_prompt] * len(wavs), langs, kwargs.get("max_new_tokens"))):
                rec["llm_text"] = out["text"]
                rec["language"] = out["language"] or rec["language"]

        if need_llm and hotword_retriever is not None:
            contexts = []
            for rec in records:
                src = next((name for name in ("ctc", "rnnt") if rec.get(f"{name}_text")), "")
                words = hotword_retriever.retrieve(rec.get(f"{src}_text", ""), topk=hotword_topk) if src else []
                rec["hotwords"] = words
                rec["hotword_source"] = src
                contexts.append(hotword_prompt(words, base_prompt))
            for rec, out in zip(records, self.decode_llm(llm_list, contexts, langs, kwargs.get("max_new_tokens"))):
                rec["hotword_llm_text"] = out["text"]
                rec["language"] = out["language"] or rec["language"]

        for rec in records:
            rec["text"] = rec.get("hotword_llm_text") or rec.get("llm_text") or rec.get("ctc_text") or rec.get("rnnt_text") or ""
        return records[0] if isinstance(audio, str) else records

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        ctc_target_ids=None,
        ctc_target_lengths=None,
        texts=None,
        tasks=None,
        **kwargs,
    ):
        tasks = names(tasks or self.train_tasks, {"llm", "ctc", "rnnt"}, "tasks")
        aux_tasks = [x for x in ("ctc", "rnnt") if x in tasks]
        need_llm = "llm" in tasks
        # 训练时 CTC/RNNT 和可选 LLM 复用同一次 encoder 输出。
        tower = self.qwen_model.thinker.audio_tower
        feat_lengths = feature_lens(input_features, feature_attention_mask)
        if self.stream_train and aux_tasks:
            hs, llm, lens = encode_train_mask(
                tower, input_features, feat_lengths,
                TRAIN_MASK_LEFT_FRAMES, TRAIN_MASK_CURRENT_FRAMES, TRAIN_MASK_RIGHT_FRAMES,
                need_llm,
            )
        else:
            hs, llm, lens = encode_offline(tower, input_features, feat_lengths, need_llm)

        outputs, losses = {"output_lengths": lens}, []
        for name in aux_tasks:
            head = self.head(name)
            head_hs = hs.to(next(head.parameters()).dtype)
            loss = head(head_hs, lens, ctc_target_ids, ctc_target_lengths) if ctc_target_ids is not None else torch.tensor(0.0, device=hs.device)
            outputs[f"{name}_loss"] = loss
            losses.append(self.loss_weights.get(name, 1.0) * loss)
            if name == "ctc" and not self.training:
                outputs["log_probs"] = self.ctc.log_softmax(head_hs, lens)

        if need_llm:
            embeds = self.qwen_model.thinker.get_input_embeddings()(input_ids)
            mask = self.qwen_model.thinker.get_placeholder_mask(input_ids, embeds)
            out = self.qwen_model.thinker(inputs_embeds=embeds.masked_scatter(mask, llm.to(embeds.dtype)), attention_mask=attention_mask, labels=labels)
            outputs["llm_loss"] = out.loss
            losses.append(self.loss_weights.get("llm", 1.0) * out.loss)
        if not losses:
            raise RuntimeError("tasks 为空，无法计算 loss。")
        outputs["loss"] = sum(losses)
        return outputs

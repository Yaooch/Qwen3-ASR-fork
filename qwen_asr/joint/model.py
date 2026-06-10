# qwen_asr/joint/model.py
import json
import os
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

from .ctc import CTC
from .defaults import (
    DEFAULT_PROMPT, ENCODER_BATCH_SIZE, JOINT_CONFIG, RNNT_MAX_SYMBOLS, STREAM_CNN_LEFT_FRAMES,
    TRAIN_MASK_CURRENT_FRAMES, TRAIN_MASK_LEFT_FRAMES, TRAIN_MASK_RIGHT_FRAMES, hotword_prompt,
)
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


def out_lens(feat_lens: torch.Tensor) -> torch.Tensor:
    leave = feat_lens % 100
    x = (leave - 1) // 2 + 1
    return ((x - 1) // 2 + 1 - 1) // 2 + 1 + (feat_lens // 100) * 13


def enc_len(n: int) -> int:
    for _ in range(3):
        n = (n + 1) // 2 if n > 0 else 0
    return n


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
        self.encoder_batch_size = ENCODER_BATCH_SIZE
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

    def _feat_lens(self, feats, mask=None):
        if mask is not None:
            return mask.sum(dim=1).long()
        return torch.full((feats.shape[0],), feats.shape[2], dtype=torch.long, device=feats.device)

    def encode_offline(self, feats, mask=None, need_llm=True, encoder_batch_size=None):
        bsz = feats.shape[0]
        encoder_batch_size = self.encoder_batch_size if encoder_batch_size is None else encoder_batch_size
        if encoder_batch_size and bsz > encoder_batch_size:
            hs_parts, llm_parts, out_parts, feat_parts = [], [], [], []
            for start in range(0, bsz, encoder_batch_size):
                end = start + encoder_batch_size
                sub = mask[start:end] if mask is not None else None
                hs, llm, lens, flens = self.encode_offline(feats[start:end], sub, need_llm, 0)
                hs_parts.append(hs); out_parts.append(lens); feat_parts.append(flens)
                if llm is not None:
                    llm_parts.append(llm)
            lens = torch.cat(out_parts)
            hs_pad = hs_parts[0].new_zeros(bsz, int(lens.max().item()), self.encoder_output_size)
            offset = 0
            for hs, cur_lens in zip(hs_parts, out_parts):
                for i, length in enumerate(cur_lens.tolist()):
                    hs_pad[offset + i, :int(length)] = hs[i, :int(length)]
                offset += hs.shape[0]
            return hs_pad, torch.cat(llm_parts) if llm_parts else None, lens, torch.cat(feat_parts)

        feat_lens = self._feat_lens(feats, mask)
        tower = self.qwen_model.thinker.audio_tower
        # 离线路径按有效长度拼接整批特征，依赖 FA2/cu_seqlens 做样本隔离。
        enc, aux = tower(
            input_features=torch.cat([feats[i, :, :feat_lens[i]] for i in range(bsz)], dim=1),
            feature_lens=feat_lens,
            return_pre_proj=True,
        )
        lens = out_lens(feat_lens)
        hs_pad = aux.new_zeros(bsz, int(lens.max().item()), self.encoder_output_size)
        offset = 0
        for i, length in enumerate(lens.tolist()):
            length = int(length)
            hs_pad[i, :length] = aux[offset:offset + length]
            offset += length
        llm = tower.proj2(tower.act(tower.proj1(enc.last_hidden_state))) if need_llm else None
        return hs_pad, llm, lens, feat_lens

    def encode_train_mask(self, feats, mask=None):
        # train_mask 是整条 Mel 上的训练侧 chunk mask，不是真实在线 KV cache。
        feat_lens = self._feat_lens(feats, mask)
        hs, lens = self.qwen_model.thinker.audio_tower.forward_stream_mask(
            feats,
            feat_lens,
            left_frames=TRAIN_MASK_LEFT_FRAMES,
            current_frames=TRAIN_MASK_CURRENT_FRAMES,
            right_frames=TRAIN_MASK_RIGHT_FRAMES,
        )
        return hs, lens, feat_lens

    def project_llm(self, hs, lens):
        tower = self.qwen_model.thinker.audio_tower
        x = torch.cat([hs[i, :int(length)] for i, length in enumerate(lens.tolist())], dim=0)
        return tower.proj2(tower.act(tower.proj1(x)))

    def decode_head(self, name, hs, lens, max_symbols_per_step=RNNT_MAX_SYMBOLS):
        head = self.head(name)
        ids = head.greedy_decode(hs.to(next(head.parameters()).dtype), lens, max_symbols_per_step=max_symbols_per_step)
        return [ids_to_text(x, self._id_to_token) for x in ids]

    @torch.no_grad()
    def decode_feats(
        self,
        input_features,
        feature_attention_mask=None,
        head="ctc",
        max_symbols_per_step=RNNT_MAX_SYMBOLS,
        encoder_batch_size=None,
    ):
        ref = next(self.qwen_model.parameters())
        feats = input_features.to(device=ref.device, dtype=ref.dtype)
        mask = feature_attention_mask.to(device=ref.device) if feature_attention_mask is not None else None
        if self.stream_train:
            hs, lens, _ = self.encode_train_mask(feats, mask)
        else:
            hs, _, lens, _ = self.encode_offline(feats, mask, False, encoder_batch_size)
        return self.decode_head(head, hs, lens, max_symbols_per_step)

    def _wavs(self, audio):
        import librosa
        import numpy as np

        items = [audio] if isinstance(audio, (str, np.ndarray)) else audio
        wavs = []
        for item in items:
            if isinstance(item, str):
                wavs.append(librosa.load(item, sr=16000, mono=True)[0].astype(np.float32, copy=False))
            elif isinstance(item, np.ndarray):
                wavs.append(item.astype(np.float32, copy=False))
            else:
                raise TypeError(f"不支持的音频项类型：{type(item)}")
        return wavs

    def _feature_batch(self, wavs):
        fe = self.processor.feature_extractor
        batch = fe(
            wavs,
            sampling_rate=int(getattr(fe, "sampling_rate", 16000) or 16000),
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        if "feature_attention_mask" not in batch and "attention_mask" in batch:
            batch["feature_attention_mask"] = batch["attention_mask"]
        return batch

    @torch.no_grad()
    def encode_stream(self, wavs, need_llm):
        import numpy as np

        ref = next(self.qwen_model.parameters())
        fe = self.processor.feature_extractor
        sr = int(getattr(fe, "sampling_rate", 16000) or 16000)
        hop = int(getattr(fe, "hop_length", 160) or 160)
        tail_limit = ((((int(getattr(fe, "n_fft", 400) or 400) + 1) // 2) + hop - 1) // hop) * hop
        chunk_samples = max(1, int(round(0.64 * sr)))
        cache_size = 7 * enc_len(max(1, int(round(0.64 * 16000 / hop))))
        tower = self.qwen_model.thinker.audio_tower
        states = [dict(wav=np.asarray(w, dtype=np.float32), pos=0, raw_tail=None, mel_tail=None, cache=None, offset=0, chunks=[]) for w in wavs]

        # 每轮取 batch 内仍 active 的 640ms waveform chunk，一起送前端和 encoder。
        while any(s["pos"] < len(s["wav"]) for s in states):
            pending = []
            for i, s in enumerate(states):
                if s["pos"] >= len(s["wav"]):
                    continue
                end = min(len(s["wav"]), s["pos"] + chunk_samples)
                cur = s["wav"][s["pos"]:end]
                # raw_tail 只为 STFT 左上下文服务，left 用来丢掉重复 Mel。
                seg = cur if s["raw_tail"] is None else np.concatenate([s["raw_tail"], cur])
                left = 0 if s["raw_tail"] is None else int(s["raw_tail"].shape[0]) // hop
                s["raw_tail"] = seg[-min(len(seg), tail_limit):].copy() if tail_limit > 0 else None
                s["pos"] = end
                pending.append((i, seg, left))

            fb = fe([x[1] for x in pending], sampling_rate=sr, return_tensors="pt", padding=True, truncation=False, return_attention_mask=True)
            mask = fb.get("feature_attention_mask", fb.get("attention_mask"))
            rows, max_len = [], 0
            for row, (i, _seg, left) in enumerate(pending):
                valid = int(mask[row].sum().item()) if mask is not None else fb["input_features"].shape[-1]
                mel = fb["input_features"][row, :, left:valid]
                if mel.shape[1] == 0:
                    continue
                s = states[i]
                mel = mel.to(device=ref.device, dtype=ref.dtype)
                # mel_tail 给 CNN 左上下文，drop 丢掉 overlap 产生的重复 encoder token。
                cnn_in = mel if s["mel_tail"] is None else torch.cat([s["mel_tail"], mel], dim=1)
                drop = 0 if s["mel_tail"] is None else enc_len(STREAM_CNN_LEFT_FRAMES)
                s["mel_tail"] = cnn_in[:, -STREAM_CNN_LEFT_FRAMES:].detach() if STREAM_CNN_LEFT_FRAMES > 0 else None
                rows.append((i, cnn_in, drop)); max_len = max(max_len, int(cnn_in.shape[1]))
            if not rows:
                continue

            batch = rows[0][1].new_zeros((len(rows), rows[0][1].shape[0], max_len))
            feat_lens, drops, caches, offsets = [], [], [], []
            for r, (i, cnn_in, drop) in enumerate(rows):
                batch[r, :, :cnn_in.shape[1]] = cnn_in
                feat_lens.append(int(cnn_in.shape[1])); drops.append(drop)
                caches.append(states[i]["cache"]); offsets.append(states[i]["offset"])
            chunks, caches = tower.forward_stream_batch_chunks(
                batch,
                torch.tensor(feat_lens, dtype=torch.long, device=ref.device),
                torch.tensor(drops, dtype=torch.long, device=ref.device),
                kv_caches=caches,
                cache_size=cache_size,
                detach_cache=True,
                position_offsets=torch.tensor(offsets, dtype=torch.long, device=ref.device),
            )
            for (i, _cnn, _drop), chunk, cache in zip(rows, chunks, caches):
                states[i]["cache"] = cache
                if chunk.numel() > 0:
                    states[i]["chunks"].append(chunk)
                    states[i]["offset"] += int(chunk.shape[0])

        chunk_lists, llm = [], []
        for i, s in enumerate(states):
            if not s["chunks"]:
                raise RuntimeError(f"No streaming auxiliary features were produced for item {i}.")
            chunk_lists.append(s["chunks"])
            if need_llm:
                seq = torch.cat(s["chunks"], dim=0).to(device=ref.device, dtype=ref.dtype)
                llm.append(tower.proj2(tower.act(tower.proj1(seq))))
        return chunk_lists, llm

    def _prompt(self, n_audio_tokens, context, language):
        if self._asr_wrapper is None:
            raise RuntimeError("模型未正确初始化，请使用 from_pretrained 加载。")
        prompt = self._asr_wrapper._build_text_prompt(context=context or "", force_language=language)
        token = self.processor.audio_token
        if token not in prompt:
            raise RuntimeError(f"Prompt does not contain audio token {token!r}: {prompt!r}")
        return prompt.replace(token, token * n_audio_tokens, 1)

    @torch.no_grad()
    def generate_llm(self, feats_list, contexts, languages, max_new_tokens=None):
        thinker, processor = self.qwen_model.thinker, self.processor
        device, dtype = next(self.qwen_model.parameters()).device, next(self.qwen_model.thinker.parameters()).dtype
        prompts = [self._prompt(int(x.shape[0]), ctx or "", lang) for x, ctx, lang in zip(feats_list, contexts, languages)]
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

        wavs = self._wavs(audio)
        langs = [None] * len(wavs) if language is None else ([language] * len(wavs) if isinstance(language, str) else list(language))
        records = [{"text": "", "language": lang, "hotwords": []} for lang in langs]
        need_llm = "llm" in modes
        # 三种 encoder path 只在这里分流，后面的 CTC/RNNT/LLM 解码共用同一套结果。
        if encoder_mode == "stream":
            chunks, llm_list = self.encode_stream(wavs, need_llm)
            seqs = [torch.cat(x, dim=0) for x in chunks]
            lens = torch.tensor([x.shape[0] for x in seqs], dtype=torch.long, device=seqs[0].device)
            hs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        else:
            ref = next(self.qwen_model.parameters())
            batch = self._feature_batch(wavs)
            feats = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
            mask = batch.get("feature_attention_mask")
            mask = mask.to(device=ref.device) if mask is not None else None
            if encoder_mode == "train_mask":
                hs, lens, _ = self.encode_train_mask(feats, mask)
                llm = self.project_llm(hs, lens) if need_llm else None
            else:
                hs, llm, lens, _ = self.encode_offline(feats, mask, need_llm, ENCODER_BATCH_SIZE)
            llm_list = [] if not need_llm else [llm[sum(lens[:i]).item():sum(lens[:i + 1]).item()] for i in range(len(lens))]

        for name in ("ctc", "rnnt"):
            if name in modes:
                for rec, text in zip(records, self.decode_head(name, hs, lens, max_symbols_per_step)):
                    rec[f"{name}_text"] = text

        base_prompt = prompt or DEFAULT_PROMPT
        if need_llm:
            for rec, out in zip(records, self.generate_llm(llm_list, [base_prompt] * len(wavs), langs, kwargs.get("max_new_tokens"))):
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
            for rec, out in zip(records, self.generate_llm(llm_list, contexts, langs, kwargs.get("max_new_tokens"))):
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
        if self.stream_train and aux_tasks:
            hs, lens, _ = self.encode_train_mask(input_features, feature_attention_mask)
            llm = self.project_llm(hs, lens) if need_llm else None
        else:
            hs, llm, lens, _ = self.encode_offline(input_features, feature_attention_mask, need_llm)

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

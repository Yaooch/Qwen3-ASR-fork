# qwen_asr/joint/stream.py
from typing import List, Tuple

import torch

from .defaults import (
    STREAM_CHUNK_SEC,
    STREAM_LEFT_CHUNKS,
    STREAM_CNN_LEFT_FRAMES,
    STREAM_FEATURE_HOLD_FRAMES,
)


class StreamingFeatureState:
    """连续接收 waveform，只对短 tail + 新 chunk 提取 feature。"""

    def __init__(self, feature_extractor, hold_frames: int = STREAM_FEATURE_HOLD_FRAMES):
        self.feature_extractor = feature_extractor
        self.hold_frames = max(0, int(hold_frames))
        self.hop_length = int(getattr(feature_extractor, "hop_length", 160) or 160)
        self.n_fft = int(getattr(feature_extractor, "n_fft", 400) or 400)
        self.sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000) or 16000)
        raw_tail = self.n_fft + (self.hold_frames + 2) * self.hop_length
        self.tail_limit = ((raw_tail + self.hop_length - 1) // self.hop_length) * self.hop_length
        self.tail_samples = None
        self.total_samples = 0
        self.emitted_frames = 0

    def prepare(self, wav):
        import numpy as np

        wav = np.asarray(wav, dtype=np.float32)
        old_total = self.total_samples
        self.total_samples += int(wav.shape[0])
        if self.tail_samples is None:
            return wav, old_total
        segment_start = old_total - int(self.tail_samples.shape[0])
        segment = np.concatenate([self.tail_samples, wav]) if wav.size > 0 else self.tail_samples
        return segment, segment_start

    def finish(self, input_features: torch.Tensor, valid: int, segment_start: int, final: bool) -> torch.Tensor:
        frame_start = max(0, int(segment_start) // self.hop_length)
        valid = int(valid)
        valid_end = frame_start + valid
        emit_end = valid_end if final else max(self.emitted_frames, valid_end - self.hold_frames)
        local_start = max(0, self.emitted_frames - frame_start)
        local_end = max(local_start, min(valid, emit_end - frame_start))
        out = input_features[:, local_start:local_end] if local_end > local_start else input_features[:, :0]
        self.emitted_frames = max(self.emitted_frames, frame_start + local_end)
        return out

    def set_tail(self, segment):
        keep = min(int(segment.shape[0]), self.tail_limit)
        self.tail_samples = segment[-keep:].copy() if keep > 0 else None


class StreamMixin:
    """流式窗口、音频特征和 chunk 拼接。"""

    def _audio_list(self, audio) -> List:
        if isinstance(audio, str):
            return [audio]
        if isinstance(audio, list):
            return audio

        import numpy as np
        if isinstance(audio, np.ndarray):
            return [audio]

        raise TypeError(f"不支持的音频类型：{type(audio)}")

    def _feature_batch(self, waveform_list: List):
        feature_extractor = self.processor.feature_extractor
        batch = feature_extractor(
            waveform_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        if "feature_attention_mask" not in batch and "attention_mask" in batch:
            batch["feature_attention_mask"] = batch.pop("attention_mask")
        return batch

    def _enc_len(self, feature_len: int) -> int:
        if feature_len <= 0:
            return 0
        for _ in range(3):
            feature_len = (feature_len + 1) // 2
        return feature_len

    def _sec_to_feature_count(self, seconds: float, min_value: int = 0) -> int:
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        hop_length = int(getattr(feature_extractor, "hop_length", 160) or 160)
        value = int(round(float(seconds) * 16000 / hop_length))
        return max(min_value, value)

    def _stream_waveform_feats(
        self,
        wavs: List,
        need_llm: bool,
        chunk_sec: float = STREAM_CHUNK_SEC,
        **_kwargs,
    ) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
        """CTC/RNNT cache 流式路径：增量前端 + batch active chunk encoder。"""
        if chunk_sec <= 0:
            raise ValueError(f"chunk_sec must be > 0, got {chunk_sec}")

        import numpy as np

        ref = next(self.qwen_model.parameters())
        feature_extractor = self.processor.feature_extractor
        sr = int(getattr(feature_extractor, "sampling_rate", 16000) or 16000)
        chunk_samples = max(1, int(round(chunk_sec * sr)))
        feature_chunk = self._sec_to_feature_count(chunk_sec, min_value=1)
        cnn_left = max(0, int(STREAM_CNN_LEFT_FRAMES))
        cache_size = max(0, int(STREAM_LEFT_CHUNKS)) * self._enc_len(feature_chunk)
        audio_tower = self.qwen_model.thinker.audio_tower

        states = []
        for wav in wavs:
            states.append({
                "wav": np.asarray(wav, dtype=np.float32),
                "pos": 0,
                "frontend": StreamingFeatureState(feature_extractor),
                "cache": None,
                "chunks": [],
                "feat_tail": None,
                "feat_offset": 0,
                "pos_offset": 0,
                "done": False,
            })

        while any(not state["done"] for state in states):
            pending = []
            for idx, state in enumerate(states):
                if state["done"]:
                    continue
                wav = state["wav"]
                start = state["pos"]
                end = min(len(wav), start + chunk_samples)
                final = end >= len(wav)
                segment, segment_start = state["frontend"].prepare(wav[start:end])
                state["pos"] = end
                state["done"] = final
                if segment.size > 0:
                    pending.append((idx, segment, segment_start, final))

            if not pending:
                continue

            feature_batch = feature_extractor(
                [item[1] for item in pending],
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                truncation=False,
                return_attention_mask=True,
            )
            mask = feature_batch.get("feature_attention_mask", feature_batch.get("attention_mask"))

            batch_items = []
            max_len = 0
            for row, (idx, segment, segment_start, final) in enumerate(pending):
                state = states[idx]
                valid = int(mask[row].sum().item()) if mask is not None else feature_batch["input_features"].shape[-1]
                new_feat = state["frontend"].finish(feature_batch["input_features"][row], valid, segment_start, final)
                state["frontend"].set_tail(segment)
                if new_feat.shape[-1] == 0:
                    continue

                new_feat = new_feat.to(device=ref.device, dtype=ref.dtype)
                feat_tail = state["feat_tail"]
                feat = new_feat if feat_tail is None else torch.cat([feat_tail, new_feat], dim=1)
                left = 0 if feat_tail is None else feat_tail.shape[1]
                feat_start = max(0, int(state["feat_offset"]) - left)
                feat_end = int(state["feat_offset"]) + int(new_feat.shape[1])
                keep_start = self._enc_len(int(state["feat_offset"])) - self._enc_len(feat_start)
                keep_end = self._enc_len(feat_end) - self._enc_len(feat_start)
                batch_items.append((idx, feat, keep_start, keep_end))
                max_len = max(max_len, int(feat.shape[1]))
                state["feat_offset"] = feat_end

                tail_src = feat if feat.shape[1] <= cnn_left else feat[:, -cnn_left:]
                state["feat_tail"] = tail_src.detach()

            if not batch_items:
                continue

            feat_dim = batch_items[0][1].shape[0]
            batch_feats = batch_items[0][1].new_zeros((len(batch_items), feat_dim, max_len))
            feat_lens = []
            keep_starts = []
            keep_ends = []
            caches = []
            pos_offsets = []
            for row, (idx, feat, keep_start, keep_end) in enumerate(batch_items):
                batch_feats[row, :, : feat.shape[1]] = feat
                feat_lens.append(int(feat.shape[1]))
                keep_starts.append(int(keep_start))
                keep_ends.append(int(keep_end))
                caches.append(states[idx]["cache"])
                pos_offsets.append(int(states[idx]["pos_offset"]))

            cur_chunks, new_caches = audio_tower.forward_stream_batch_chunks(
                batch_feats,
                torch.tensor(feat_lens, dtype=torch.long, device=ref.device),
                torch.tensor(keep_starts, dtype=torch.long, device=ref.device),
                torch.tensor(keep_ends, dtype=torch.long, device=ref.device),
                kv_caches=caches,
                cache_size=cache_size,
                detach_cache=True,
                position_offsets=torch.tensor(pos_offsets, dtype=torch.long, device=ref.device),
            )
            for (idx, _feat, _keep_start, _keep_end), cur, cache in zip(batch_items, cur_chunks, new_caches):
                states[idx]["cache"] = cache
                if cur.numel() > 0:
                    states[idx]["chunks"].append(cur)
                    states[idx]["pos_offset"] += int(cur.shape[0])

        aux_chunks = []
        llm_features = []
        for wav_idx, state in enumerate(states):
            chunks = state["chunks"]
            if not chunks:
                raise RuntimeError(f"No streaming auxiliary features were produced for item {wav_idx}.")
            aux_chunks.append(chunks)
            if need_llm:
                seq = torch.cat(chunks, dim=0).to(device=ref.device, dtype=ref.dtype)
                llm_features.append(audio_tower.proj2(audio_tower.act(audio_tower.proj1(seq))))
        return aux_chunks, llm_features

    @torch.no_grad()
    def _stream_batch_feats(
        self,
        wavs: List,
        need_llm: bool,
        chunk_sec: float = STREAM_CHUNK_SEC,
        **kwargs,
    ) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
        return self._stream_waveform_feats(wavs, need_llm, chunk_sec=chunk_sec, **kwargs)

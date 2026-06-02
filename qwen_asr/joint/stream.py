# qwen_asr/joint/stream.py
from typing import List, Tuple

import torch

from .defaults import STREAM_CHUNK_SEC, STREAM_CNN_LEFT_FRAMES, STREAM_LEFT_CHUNKS


class StreamingFeatureState:
    """连续接收 waveform，仅重算短 raw tail 和当前 chunk 的 Mel。"""

    def __init__(self, feature_extractor):
        self.hop_length = int(getattr(feature_extractor, "hop_length", 160) or 160)
        self.n_fft = int(getattr(feature_extractor, "n_fft", 400) or 400)
        self.sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000) or 16000)
        raw_left = (self.n_fft + 1) // 2
        self.tail_limit = ((raw_left + self.hop_length - 1) // self.hop_length) * self.hop_length
        self.tail_samples = None

    def append(self, wav):
        import numpy as np

        wav = np.asarray(wav, dtype=np.float32)
        if self.tail_samples is None:
            segment = wav
            left_frames = 0
        else:
            segment = np.concatenate([self.tail_samples, wav])
            left_frames = int(self.tail_samples.shape[0]) // self.hop_length
        keep = min(int(segment.shape[0]), self.tail_limit)
        self.tail_samples = segment[-keep:].copy() if keep > 0 else None
        return segment, left_frames


class StreamMixin:
    """流式 waveform、Mel、CNN overlap 和 Encoder KV cache。"""

    def _audio_list(self, audio) -> List:
        if isinstance(audio, str):
            return [audio]
        if isinstance(audio, list):
            return audio

        import numpy as np
        if isinstance(audio, np.ndarray):
            return [audio]

        raise TypeError(f"不支持的音频类型：{type(audio)}")

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

    @torch.no_grad()
    def _encode_stream_waveforms(
        self,
        wavs: List,
        need_llm: bool,
    ) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
        """按 640ms 增量提 Mel，CNN 对齐后批量推进 Encoder cache。"""
        import numpy as np

        ref = next(self.qwen_model.parameters())
        feature_extractor = self.processor.feature_extractor
        sr = int(getattr(feature_extractor, "sampling_rate", 16000) or 16000)
        chunk_samples = max(1, int(round(STREAM_CHUNK_SEC * sr)))
        feature_chunk = self._sec_to_feature_count(STREAM_CHUNK_SEC, min_value=1)
        cnn_left = STREAM_CNN_LEFT_FRAMES
        cache_size = max(0, int(STREAM_LEFT_CHUNKS)) * self._enc_len(feature_chunk)
        audio_tower = self.qwen_model.thinker.audio_tower

        states = [{
            "wav": np.asarray(wav, dtype=np.float32),
            "wav_pos": 0,
            "frontend": StreamingFeatureState(feature_extractor),
            "mel_tail": None,
            "encoder_cache": None,
            "position_offset": 0,
            "chunks": [],
        } for wav in wavs]

        while any(state["wav_pos"] < len(state["wav"]) for state in states):
            pending = []
            for idx, state in enumerate(states):
                start = state["wav_pos"]
                if start >= len(state["wav"]):
                    continue
                end = min(len(state["wav"]), start + chunk_samples)
                segment, left_frames = state["frontend"].append(state["wav"][start:end])
                state["wav_pos"] = end
                pending.append((idx, segment, left_frames))

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
            for row, (idx, _segment, left_frames) in enumerate(pending):
                state = states[idx]
                valid = int(mask[row].sum().item()) if mask is not None else feature_batch["input_features"].shape[-1]
                new_mel = feature_batch["input_features"][row, :, left_frames:valid]
                if new_mel.shape[1] == 0:
                    continue
                new_mel = new_mel.to(device=ref.device, dtype=ref.dtype)
                mel_tail = state["mel_tail"]
                cnn_input = new_mel if mel_tail is None else torch.cat([mel_tail, new_mel], dim=1)
                drop_prefix = 0 if mel_tail is None else self._enc_len(cnn_left)
                batch_items.append((idx, cnn_input, drop_prefix))
                max_len = max(max_len, int(cnn_input.shape[1]))
                state["mel_tail"] = cnn_input[:, -cnn_left:].detach() if cnn_left > 0 else None

            if not batch_items:
                continue

            feat_dim = batch_items[0][1].shape[0]
            batch_feats = batch_items[0][1].new_zeros((len(batch_items), feat_dim, max_len))
            feat_lens = []
            drop_prefixes = []
            caches = []
            pos_offsets = []
            for row, (idx, cnn_input, drop_prefix) in enumerate(batch_items):
                batch_feats[row, :, : cnn_input.shape[1]] = cnn_input
                feat_lens.append(int(cnn_input.shape[1]))
                drop_prefixes.append(drop_prefix)
                caches.append(states[idx]["encoder_cache"])
                pos_offsets.append(states[idx]["position_offset"])

            cur_chunks, new_caches = audio_tower.forward_stream_batch_chunks(
                batch_feats,
                torch.tensor(feat_lens, dtype=torch.long, device=ref.device),
                torch.tensor(drop_prefixes, dtype=torch.long, device=ref.device),
                kv_caches=caches,
                cache_size=cache_size,
                detach_cache=True,
                position_offsets=torch.tensor(pos_offsets, dtype=torch.long, device=ref.device),
            )
            for (idx, _cnn_input, _drop_prefix), cur, cache in zip(batch_items, cur_chunks, new_caches):
                states[idx]["encoder_cache"] = cache
                if cur.numel() > 0:
                    states[idx]["chunks"].append(cur)
                    states[idx]["position_offset"] += int(cur.shape[0])

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

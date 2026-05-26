# qwen_asr/joint/stream.py
from typing import Dict, List, Optional, Tuple

import torch

from .defaults import (
    RNNT_MAX_SYMBOLS,
    STREAM_CHUNK_SEC,
    STREAM_ENCODER_BATCH,
    STREAM_FIRST_PAD_SEC,
    STREAM_LEFT_SEC,
    STREAM_RIGHT_SEC,
    STREAM_WINDOW_BATCH,
)


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

    def _feat_len(self, num_samples: int, cache: Optional[Dict[int, int]] = None) -> int:
        """Feature extractor valid length for a waveform with this many samples."""
        if num_samples <= 0:
            return 0
        num_samples = int(num_samples)
        if cache is not None and num_samples in cache:
            return cache[num_samples]

        feature_extractor = self.processor.feature_extractor
        hop_length = int(getattr(feature_extractor, "hop_length", 0) or 0)
        if hop_length > 0:
            value = (num_samples + hop_length - 1) // hop_length
        else:
            import numpy as np

            feat = feature_extractor(
                np.zeros((num_samples,), dtype=np.float32),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                truncation=False,
                return_attention_mask=True,
            )
            value = int(feat["attention_mask"].sum().item())

        if cache is not None:
            cache[num_samples] = value
        return value

    def _enc_len(self, feature_len: int, device: torch.device = None) -> int:
        if feature_len <= 0:
            return 0
        leave = feature_len % 100
        feat_lengths = (leave - 1) // 2 + 1
        return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_len // 100) * 13

    def _windows(
        self,
        wav,
        chunk_sec: float,
        left_context_sec: float,
        right_context_sec: float,
        first_chunk_left_pad_sec: float,
    ):
        """Build current-chunk windows for streaming aux decoding."""
        if chunk_sec <= 0:
            raise ValueError(f"chunk_sec must be > 0, got {chunk_sec}")

        import numpy as np

        sr = 16000
        total_samples = int(wav.shape[0])
        chunk_samples = max(1, int(round(chunk_sec * sr)))
        left_samples = max(0, int(round(left_context_sec * sr)))
        right_samples = max(0, int(round(right_context_sec * sr)))
        first_left_pad_samples = max(0, int(round(first_chunk_left_pad_sec * sr)))
        if first_left_pad_samples >= chunk_samples:
            raise ValueError(
                "first_chunk_left_pad_sec must be smaller than chunk_sec, got "
                f"{first_chunk_left_pad_sec} >= {chunk_sec}"
            )

        windows = []
        start = 0
        first_chunk = True
        while start < total_samples:
            cur_chunk_samples = chunk_samples
            left_pad_samples = 0
            if first_chunk and first_left_pad_samples > 0:
                cur_chunk_samples = chunk_samples - first_left_pad_samples
                left_pad_samples = first_left_pad_samples

            end = min(total_samples, start + cur_chunk_samples)
            enc_start = max(0, start - left_samples)
            enc_end = min(total_samples, end + right_samples)
            window_wav = wav[enc_start:enc_end]
            if left_pad_samples > 0:
                pad = np.zeros(left_pad_samples, dtype=window_wav.dtype)
                window_wav = np.concatenate([pad, window_wav], axis=0)

            windows.append((start, end, enc_start, enc_end, left_pad_samples, window_wav))
            start = end
            first_chunk = False
        return windows

    @torch.no_grad()
    def _enc_aux_windows(
        self,
        window_wavs: List,
        encoder_batch_size: int = 1,
    ) -> List[torch.Tensor]:
        """Encode audio windows and return one aux-feature tensor per window."""
        if not window_wavs:
            return []

        ref = next(self.qwen_model.parameters())
        batch = self._feature_batch(window_wavs)
        input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
        feature_attention_mask = batch["feature_attention_mask"].to(device=ref.device)

        hs_pad, _, out_lens, _ = self._enc_joint(
            input_features,
            feature_attention_mask,
            need_llm_features=False,
            encoder_batch_size=encoder_batch_size,
        )
        return [hs_pad[i, : int(length)] for i, length in enumerate(out_lens.tolist())]

    @torch.no_grad()
    def _enc_joint_windows(
        self,
        window_wavs: List,
        encoder_batch_size: int = 1,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode audio windows and return aux + LLM features per window."""
        if not window_wavs:
            return [], []

        ref = next(self.qwen_model.parameters())
        batch = self._feature_batch(window_wavs)
        input_features = batch["input_features"].to(device=ref.device, dtype=ref.dtype)
        feature_attention_mask = batch["feature_attention_mask"].to(device=ref.device)

        hs_pad, llm_features, out_lens, _ = self._enc_joint(
            input_features,
            feature_attention_mask,
            need_llm_features=True,
            encoder_batch_size=encoder_batch_size,
        )
        aux_results = []
        llm_results = []
        offset = 0
        for i, length in enumerate(out_lens.tolist()):
            cur_len = int(length)
            aux_results.append(hs_pad[i, :cur_len])
            llm_results.append(llm_features[offset: offset + cur_len])
            offset += cur_len
        return aux_results, llm_results

    def _window_batch_size(self, windows: List, window_batch_size: int) -> int:
        if window_batch_size is None or window_batch_size <= 0:
            return len(windows)
        return window_batch_size

    def _current_chunk_slice(
        self,
        window,
        feature_count: int,
        feature_len_cache: Optional[Dict[int, int]],
        device: torch.device,
    ) -> slice:
        start, end, enc_start, _enc_end, left_pad_samples, _ = window
        keep_start_samples = left_pad_samples + start - enc_start
        keep_end_samples = left_pad_samples + end - enc_start
        keep_start_feat = self._feat_len(keep_start_samples, feature_len_cache)
        keep_end_feat = self._feat_len(keep_end_samples, feature_len_cache)
        keep_start_idx = self._enc_len(keep_start_feat, device)
        keep_end_idx = self._enc_len(keep_end_feat, device)

        keep_start_idx = max(0, min(keep_start_idx, feature_count))
        keep_end_idx = max(keep_start_idx, min(keep_end_idx, feature_count))
        return slice(keep_start_idx, keep_end_idx)

    @torch.no_grad()
    def _stream_aux_chunks(
        self,
        wav,
        chunk_sec: float = STREAM_CHUNK_SEC,
        left_context_sec: float = STREAM_LEFT_SEC,
        right_context_sec: float = STREAM_RIGHT_SEC,
        first_chunk_left_pad_sec: float = STREAM_FIRST_PAD_SEC,
        window_batch_size: int = STREAM_WINDOW_BATCH,
        window_encoder_batch_size: int = STREAM_ENCODER_BATCH,
        feature_len_cache: Optional[Dict[int, int]] = None,
    ) -> List[torch.Tensor]:
        """Encode overlap windows and keep only the newly arrived chunk frames."""
        device = next(self.qwen_model.parameters()).device
        windows = self._windows(
            wav,
            chunk_sec=chunk_sec,
            left_context_sec=left_context_sec,
            right_context_sec=right_context_sec,
            first_chunk_left_pad_sec=first_chunk_left_pad_sec,
        )
        window_batch_size = self._window_batch_size(windows, window_batch_size)

        kept_chunks = []
        for batch_start in range(0, len(windows), window_batch_size):
            batch_windows = windows[batch_start: batch_start + window_batch_size]
            window_features_batch = self._enc_aux_windows(
                [x[5] for x in batch_windows],
                encoder_batch_size=window_encoder_batch_size,
            )

            for window, window_features in zip(batch_windows, window_features_batch):
                cur_slice = self._current_chunk_slice(
                    window,
                    window_features.shape[0],
                    feature_len_cache,
                    device,
                )
                kept = window_features[cur_slice]
                if kept.numel() > 0:
                    kept_chunks.append(kept)

        if not kept_chunks:
            raise RuntimeError("No streaming auxiliary features were produced.")
        return kept_chunks

    @torch.no_grad()
    def _stream_joint_feats(
        self,
        wav,
        chunk_sec: float = STREAM_CHUNK_SEC,
        left_context_sec: float = STREAM_LEFT_SEC,
        right_context_sec: float = STREAM_RIGHT_SEC,
        first_chunk_left_pad_sec: float = STREAM_FIRST_PAD_SEC,
        window_batch_size: int = STREAM_WINDOW_BATCH,
        window_encoder_batch_size: int = STREAM_ENCODER_BATCH,
        feature_len_cache: Optional[Dict[int, int]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """流式窗口只跑一次 encoder，同时保留 aux chunk 和 LLM 特征。"""
        device = next(self.qwen_model.parameters()).device
        windows = self._windows(
            wav,
            chunk_sec=chunk_sec,
            left_context_sec=left_context_sec,
            right_context_sec=right_context_sec,
            first_chunk_left_pad_sec=first_chunk_left_pad_sec,
        )
        window_batch_size = self._window_batch_size(windows, window_batch_size)

        aux_chunks = []
        llm_chunks = []
        for batch_start in range(0, len(windows), window_batch_size):
            batch_windows = windows[batch_start: batch_start + window_batch_size]
            aux_features_batch, llm_features_batch = self._enc_joint_windows(
                [x[5] for x in batch_windows],
                encoder_batch_size=window_encoder_batch_size,
            )

            for window, aux_features, llm_features in zip(
                batch_windows,
                aux_features_batch,
                llm_features_batch,
            ):
                cur_slice = self._current_chunk_slice(
                    window,
                    aux_features.shape[0],
                    feature_len_cache,
                    device,
                )
                aux_kept = aux_features[cur_slice]
                llm_kept = llm_features[cur_slice]
                if aux_kept.numel() > 0:
                    aux_chunks.append(aux_kept)
                if llm_kept.numel() > 0:
                    llm_chunks.append(llm_kept)

        if not aux_chunks:
            raise RuntimeError("No streaming auxiliary features were produced.")
        if not llm_chunks:
            raise RuntimeError("No streaming LLM audio features were produced.")
        return aux_chunks, torch.cat(llm_chunks, dim=0)

    @torch.no_grad()
    def _stream_batch_feats(
        self,
        wavs: List,
        need_llm: bool,
        chunk_sec: float = STREAM_CHUNK_SEC,
        left_context_sec: float = STREAM_LEFT_SEC,
        right_context_sec: float = STREAM_RIGHT_SEC,
        first_chunk_left_pad_sec: float = STREAM_FIRST_PAD_SEC,
        window_batch_size: int = STREAM_WINDOW_BATCH,
        window_encoder_batch_size: int = STREAM_ENCODER_BATCH,
        feature_len_cache: Optional[Dict[int, int]] = None,
    ) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
        """把一个 batch 内所有流式窗口合批编码，避免逐条音频喂 GPU。"""
        device = next(self.qwen_model.parameters()).device
        all_windows = []
        for wav_idx, wav in enumerate(wavs):
            windows = self._windows(
                wav,
                chunk_sec=chunk_sec,
                left_context_sec=left_context_sec,
                right_context_sec=right_context_sec,
                first_chunk_left_pad_sec=first_chunk_left_pad_sec,
            )
            all_windows.extend((wav_idx, window) for window in windows)

        aux_chunks = [[] for _ in wavs]
        llm_chunks = [[] for _ in wavs]
        window_batch_size = self._window_batch_size(all_windows, window_batch_size)

        for batch_start in range(0, len(all_windows), window_batch_size):
            batch_items = all_windows[batch_start: batch_start + window_batch_size]
            batch_windows = [item[1] for item in batch_items]
            window_wavs = [window[5] for window in batch_windows]
            if need_llm:
                aux_batch, llm_batch = self._enc_joint_windows(
                    window_wavs,
                    encoder_batch_size=window_encoder_batch_size,
                )
            else:
                aux_batch = self._enc_aux_windows(
                    window_wavs,
                    encoder_batch_size=window_encoder_batch_size,
                )
                llm_batch = [None] * len(aux_batch)

            for (wav_idx, window), aux_features, llm_features in zip(batch_items, aux_batch, llm_batch):
                cur_slice = self._current_chunk_slice(
                    window,
                    aux_features.shape[0],
                    feature_len_cache,
                    device,
                )
                aux_kept = aux_features[cur_slice]
                if aux_kept.numel() > 0:
                    aux_chunks[wav_idx].append(aux_kept)
                if need_llm:
                    llm_kept = llm_features[cur_slice]
                    if llm_kept.numel() > 0:
                        llm_chunks[wav_idx].append(llm_kept)

        llm_features_list = []
        for idx, chunks in enumerate(aux_chunks):
            if not chunks:
                raise RuntimeError(f"No streaming auxiliary features were produced for item {idx}.")
            if need_llm:
                if not llm_chunks[idx]:
                    raise RuntimeError(f"No streaming LLM audio features were produced for item {idx}.")
                llm_features_list.append(torch.cat(llm_chunks[idx], dim=0))
        return aux_chunks, llm_features_list

    @torch.no_grad()
    def _rnnt_stream_decode(
        self,
        chunks: List[torch.Tensor],
        max_symbols_per_step: int = RNNT_MAX_SYMBOLS,
    ) -> List[int]:
        """Stateful RNNT greedy decode. Predictor state is carried across chunks."""
        if self.rnnt is None:
            raise RuntimeError("当前 checkpoint 没有 RNNT 头。")
        if max_symbols_per_step <= 0:
            raise ValueError(f"max_symbols_per_step must be positive, got {max_symbols_per_step}")

        rnnt = self.rnnt
        device = next(rnnt.parameters()).device
        rnnt_dtype = next(rnnt.parameters()).dtype
        state_dtype = rnnt.embed.weight.dtype
        hidden_size = rnnt.pred_rnn.hidden_size
        num_layers = rnnt.pred_rnn.num_layers

        h = torch.zeros(num_layers, 1, hidden_size, device=device, dtype=state_dtype)
        c = torch.zeros(num_layers, 1, hidden_size, device=device, dtype=state_dtype)

        start_token = torch.full((1, 1), rnnt.blank_id, dtype=torch.long, device=device)
        pred_step, (h, c) = rnnt.pred_rnn(rnnt.embed(start_token), (h, c))
        pred = rnnt.pred_proj(pred_step[:, -1, :])

        emitted = []
        for chunk in chunks:
            enc = rnnt.enc_proj(chunk.to(device=device, dtype=rnnt_dtype).unsqueeze(0))[0]
            for t in range(enc.shape[0]):
                symbols = 0
                while symbols < max_symbols_per_step:
                    joint = torch.tanh(enc[t].unsqueeze(0) + pred)
                    next_id = int(rnnt.joiner(joint).argmax(dim=-1).item())
                    if next_id == rnnt.blank_id:
                        break

                    emitted.append(next_id)
                    token = torch.tensor([[next_id]], dtype=torch.long, device=device)
                    pred_step, (h, c) = rnnt.pred_rnn(rnnt.embed(token), (h, c))
                    pred = rnnt.pred_proj(pred_step[:, -1, :])
                    symbols += 1

        return emitted

    @torch.no_grad()
    def _ctc_stream_decode(self, chunks: List[torch.Tensor]) -> List[int]:
        """Streaming CTC greedy decode with CTC collapse state carried across chunks."""
        if self.ctc is None:
            raise RuntimeError("当前 checkpoint 没有 CTC 头。")

        emitted = []
        prev_id = -1
        for chunk in chunks:
            hs = chunk.to(device=next(self.ctc.parameters()).device, dtype=next(self.ctc.parameters()).dtype)
            log_probs = self.ctc.log_softmax(hs.unsqueeze(0))[0]
            pred_ids = log_probs.argmax(dim=-1).tolist()
            for idx in pred_ids:
                if idx != self.ctc.blank_id and idx != prev_id:
                    emitted.append(idx)
                prev_id = idx
        return emitted

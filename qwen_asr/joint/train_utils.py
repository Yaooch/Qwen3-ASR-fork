# qwen_asr/joint/train_utils.py
from typing import Dict, List, Optional, Tuple

import torch


class TrainMixin:
    """流式训练窗口和辅助头训练特征。"""

    def _train_windows(
        self,
        input_features: torch.Tensor,
        feat_lens: torch.Tensor,
    ):
        """Build feature-level streaming windows for aux loss training.

        Each window contains a current chunk, a random amount of left context,
        and a fixed right context. Only the current chunk frames are kept for
        the final aux sequence.
        """
        windows = []
        chunk_frames = max(1, int(self.aux_stream_chunk_frames))
        max_left_frames = max(0, int(self.aux_stream_left_context_frames))
        right_frames = max(0, int(self.aux_stream_right_context_frames))
        random_left = bool(self.aux_stream_random_left)
        device = input_features.device

        for b in range(input_features.shape[0]):
            total = int(feat_lens[b].item())
            start = 0
            while start < total:
                end = min(total, start + chunk_frames)
                allowed_left = min(max_left_frames, start)
                if random_left and allowed_left > 0 and self.training:
                    left = int(torch.randint(0, allowed_left + 1, (1,), device=device).item())
                else:
                    left = allowed_left
                enc_start = start - left
                enc_end = min(total, end + right_frames)
                windows.append(
                    {
                        "sample_idx": b,
                        "start": start,
                        "end": end,
                        "enc_start": enc_start,
                        "features": input_features[b, :, enc_start:enc_end],
                    }
                )
                start = end
        return windows

    def _pad_windows(self, windows: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not windows:
            raise RuntimeError("No feature windows were produced.")

        batch_size = len(windows)
        channels = windows[0]["features"].shape[0]
        max_len = max(int(w["features"].shape[1]) for w in windows)
        ref = windows[0]["features"]
        padded = torch.zeros(batch_size, channels, max_len, dtype=ref.dtype, device=ref.device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=ref.device)
        for i, item in enumerate(windows):
            cur = item["features"]
            cur_len = int(cur.shape[1])
            padded[i, :, :cur_len] = cur
            mask[i, :cur_len] = 1
        return padded, mask

    def _enc_train_stream(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode aux features with streaming-style random left context for training."""
        if feature_attention_mask is not None:
            feat_lens = feature_attention_mask.sum(dim=1).long()
        else:
            feat_lens = torch.full(
                (input_features.shape[0],),
                input_features.shape[2],
                dtype=torch.long,
                device=input_features.device,
            )

        windows = self._train_windows(input_features, feat_lens)
        per_sample_chunks = [[] for _ in range(input_features.shape[0])]
        window_batch_size = max(1, int(self.aux_stream_window_batch_size))
        device = input_features.device

        for batch_start in range(0, len(windows), window_batch_size):
            batch_windows = windows[batch_start: batch_start + window_batch_size]
            window_features, window_mask = self._pad_windows(batch_windows)
            hs_pad, _, out_lens, _ = self._enc_joint(
                window_features,
                window_mask,
                need_llm_features=False,
                encoder_batch_size=self.aux_encoder_batch_size,
            )
            hs_pad = hs_pad.to(next(self.aux_head.parameters()).dtype)

            for i, item in enumerate(batch_windows):
                keep_start_frames = item["start"] - item["enc_start"]
                keep_end_frames = item["end"] - item["enc_start"]
                keep_start_idx = self._enc_len(keep_start_frames, device)
                keep_end_idx = self._enc_len(keep_end_frames, device)
                cur_len = int(out_lens[i].item())
                keep_start_idx = max(0, min(keep_start_idx, cur_len))
                keep_end_idx = max(keep_start_idx, min(keep_end_idx, cur_len))
                kept = hs_pad[i, keep_start_idx:keep_end_idx]
                if kept.numel() > 0:
                    per_sample_chunks[item["sample_idx"]].append(kept)

        seqs = []
        lens = []
        for chunks in per_sample_chunks:
            if not chunks:
                raise RuntimeError("Streaming aux training produced an empty sample.")
            seq = torch.cat(chunks, dim=0)
            seqs.append(seq)
            lens.append(seq.shape[0])

        max_len = max(lens)
        hs_out = torch.zeros(
            len(seqs),
            max_len,
            self.encoder_output_size,
            dtype=seqs[0].dtype,
            device=input_features.device,
        )
        for i, seq in enumerate(seqs):
            hs_out[i, : seq.shape[0]] = seq
        out_lens = torch.tensor(lens, dtype=torch.long, device=input_features.device)
        return hs_out, out_lens


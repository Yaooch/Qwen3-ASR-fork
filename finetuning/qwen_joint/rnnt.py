# qwen_joint/rnnt.py
import torch
import torch.nn as nn


class RNNT(nn.Module):
    """轻量 RNNT 分支：encoder projection + predictor + joiner。"""

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int,
        blank_id: int = 0,
        pred_embed_dim: int = 512,
        pred_hidden_dim: int = 768,
        joint_dim: int = 768,
        max_logit_elements: int = 20_000_000_000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.max_logit_elements = max_logit_elements

        self.enc_proj = nn.Linear(encoder_dim, joint_dim)
        self.embed = nn.Embedding(vocab_size, pred_embed_dim, padding_idx=blank_id)
        self.pred_rnn = nn.LSTM(pred_embed_dim, pred_hidden_dim, num_layers=1, batch_first=True)
        self.pred_proj = nn.Linear(pred_hidden_dim, joint_dim)
        self.joiner = nn.Linear(joint_dim, vocab_size)

    def forward_logits(self, hs_pad: torch.Tensor, ys_in_pad: torch.Tensor) -> torch.Tensor:
        """返回 RNNT logits: [B, T, U+1, V]。"""
        enc = self.enc_proj(hs_pad)
        pred = self.embed(ys_in_pad)
        pred, _ = self.pred_rnn(pred)
        pred = self.pred_proj(pred)

        joint = torch.tanh(enc.unsqueeze(2) + pred.unsqueeze(1))
        return self.joiner(joint)

    def forward(
        self,
        hs_pad: torch.Tensor,
        encoder_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from torchaudio.functional import rnnt_loss
        except ImportError as e:
            raise ImportError("当前环境缺少 torchaudio.functional.rnnt_loss，需安装支持 RNNT loss 的 torchaudio。") from e

        batch_size, max_u = labels.shape
        if labels.numel() > 0:
            if labels.min().item() < 0 or labels.max().item() >= self.vocab_size:
                raise ValueError(
                    f"RNNT targets out of range: min={labels.min().item()}, "
                    f"max={labels.max().item()}, vocab_size={self.vocab_size}"
                )
        if (label_lengths < 0).any() or (label_lengths > max_u).any():
            raise ValueError(f"Invalid RNNT label_lengths: max_u={max_u}, lengths={label_lengths.tolist()}")
        if (encoder_lengths <= 0).any() or (encoder_lengths > hs_pad.size(1)).any():
            raise ValueError(
                f"Invalid RNNT encoder_lengths: max_t={hs_pad.size(1)}, lengths={encoder_lengths.tolist()}"
            )

        logit_elements = batch_size * hs_pad.size(1) * (max_u + 1) * self.vocab_size
        if logit_elements > self.max_logit_elements:
            raise RuntimeError(
                "RNNT logits would be too large: "
                f"B={batch_size}, T={hs_pad.size(1)}, U={max_u}, V={self.vocab_size}, "
                f"elements={logit_elements:,}. Reduce BATCH_SIZE, max audio length, or target length."
            )

        ys_in = torch.full(
            (batch_size, max_u + 1),
            self.blank_id,
            dtype=torch.long,
            device=labels.device,
        )
        ys_in[:, 1:] = labels

        logits = self.forward_logits(hs_pad, ys_in)

        return rnnt_loss(
            logits=logits.float(),
            targets=labels.int(),
            logit_lengths=encoder_lengths.int(),
            target_lengths=label_lengths.int(),
            blank=self.blank_id,
            reduction="mean",
            fused_log_softmax=True,
        )

    @torch.no_grad()
    def greedy_decode(
        self,
        hs_pad: torch.Tensor,
        hs_lengths: torch.Tensor,
        max_symbols_per_step: int = 5,
        decode_strategy: str = "cached",
    ):
        if decode_strategy == "legacy":
            return self.greedy_decode_legacy(
                hs_pad,
                hs_lengths,
                max_symbols_per_step=max_symbols_per_step,
            )
        if decode_strategy == "cached":
            return self.greedy_decode_cached(
                hs_pad,
                hs_lengths,
                max_symbols_per_step=max_symbols_per_step,
            )
        raise ValueError(f"Unsupported RNNT decode_strategy: {decode_strategy}")

    @torch.no_grad()
    def greedy_decode_legacy(
        self,
        hs_pad: torch.Tensor,
        hs_lengths: torch.Tensor,
        max_symbols_per_step: int = 5,
    ):
        """Original RNNT greedy decode used by training-time eval."""
        enc = self.enc_proj(hs_pad)
        results = []

        for b in range(enc.size(0)):
            emitted = []
            ys = [self.blank_id]
            cur_len = int(hs_lengths[b].item())

            for t in range(cur_len):
                symbols = 0

                while symbols < max_symbols_per_step:
                    ys_tensor = torch.tensor([ys], dtype=torch.long, device=enc.device)
                    pred = self.embed(ys_tensor)
                    pred, _ = self.pred_rnn(pred)
                    pred = self.pred_proj(pred)[:, -1, :]

                    joint = torch.tanh(enc[b, t].unsqueeze(0) + pred)
                    next_id = int(self.joiner(joint).argmax(dim=-1).item())

                    if next_id == self.blank_id:
                        break

                    emitted.append(next_id)
                    ys.append(next_id)
                    symbols += 1

            results.append(emitted)

        return results

    @torch.no_grad()
    def greedy_decode_cached(
        self,
        hs_pad: torch.Tensor,
        hs_lengths: torch.Tensor,
        max_symbols_per_step: int = 5,
    ):
        """Batched RNNT greedy decode with cached predictor states.

        The predictor state represents the emitted token history for each sample.
        It is updated only when a non-blank token is emitted, so blank frames do
        not repeatedly rerun the LSTM over the same history.
        """
        if max_symbols_per_step <= 0:
            raise ValueError(f"max_symbols_per_step must be positive, got {max_symbols_per_step}")

        enc = self.enc_proj(hs_pad)
        batch_size, max_t, _ = enc.shape
        if batch_size == 0:
            return []

        device = enc.device
        results = [[] for _ in range(batch_size)]

        state_dtype = self.embed.weight.dtype
        num_layers = self.pred_rnn.num_layers
        hidden_size = self.pred_rnn.hidden_size
        h = torch.zeros(num_layers, batch_size, hidden_size, device=device, dtype=state_dtype)
        c = torch.zeros(num_layers, batch_size, hidden_size, device=device, dtype=state_dtype)

        start_tokens = torch.full((batch_size, 1), self.blank_id, dtype=torch.long, device=device)
        pred, (h, c) = self.pred_rnn(self.embed(start_tokens), (h, c))
        pred = self.pred_proj(pred[:, -1, :])

        lengths = hs_lengths.to(device=device)
        for t in range(max_t):
            frame_active = lengths > t
            if not bool(frame_active.any()):
                break

            step_active = frame_active.clone()
            emitted_this_frame = torch.zeros(batch_size, device=device, dtype=torch.long)

            while bool(step_active.any()):
                active_idx = step_active.nonzero(as_tuple=False).squeeze(1)
                joint = torch.tanh(enc[active_idx, t, :] + pred[active_idx])
                next_ids = self.joiner(joint).argmax(dim=-1)

                nonblank = next_ids != self.blank_id
                blank_idx = active_idx[~nonblank]
                if blank_idx.numel() > 0:
                    step_active[blank_idx] = False

                emit_idx = active_idx[nonblank]
                if emit_idx.numel() == 0:
                    continue

                emit_ids = next_ids[nonblank]
                for sample_idx, token_id in zip(emit_idx.tolist(), emit_ids.tolist()):
                    results[sample_idx].append(token_id)

                emb = self.embed(emit_ids.view(-1, 1))
                pred_step, new_state = self.pred_rnn(
                    emb,
                    (h[:, emit_idx, :].contiguous(), c[:, emit_idx, :].contiguous()),
                )
                h[:, emit_idx, :] = new_state[0]
                c[:, emit_idx, :] = new_state[1]
                pred[emit_idx] = self.pred_proj(pred_step[:, -1, :])

                emitted_this_frame[emit_idx] += 1
                finished_idx = emit_idx[emitted_this_frame[emit_idx] >= max_symbols_per_step]
                if finished_idx.numel() > 0:
                    step_active[finished_idx] = False

        return results

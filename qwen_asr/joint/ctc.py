# qwen_asr/joint/ctc.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCAdapter(nn.Module):
    """CTC 分支的轻量残差适配层。

    输入已经过 audio tower 的 ln_post。depthwise 时序卷积提供帧间上下文，
    MLP 负责把共享特征轻量转到 CTC 空间。
    """

    def __init__(self, dim: int, dropout: float = 0.1, hidden_mult: int = 2):
        super().__init__()
        hidden_dim = dim * hidden_mult
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class CTCMoEAdapter(nn.Module):
    """CTC 分支的轻量 MoE 残差适配层。"""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        hidden_mult: int = 2,
        num_experts: int = 8,
        top_k: int = 2,
        router_type: str = "frame",
    ):
        super().__init__()
        hidden_dim = dim * hidden_mult
        self.top_k = top_k
        self.router_type = router_type
        self.norm = nn.LayerNorm(dim)
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )
        # self.scale = nn.Parameter(torch.tensor(0.1))

    def _mean(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        if lengths is None:
            return x.mean(dim=1)

        t = x.size(1)
        mask = torch.arange(t, device=x.device).unsqueeze(0) < lengths.to(x.device).unsqueeze(1)
        denom = mask.sum(dim=1).clamp_min(1).to(x.dtype).unsqueeze(1)
        return (x * mask.unsqueeze(2).to(x.dtype)).sum(dim=1) / denom

    def _topk(self, logits: torch.Tensor) -> torch.Tensor:
        top_values, top_indices = logits.topk(self.top_k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(-1, top_indices, top_values)
        return F.softmax(masked, dim=-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        z = self.norm(x)
        expert_out = torch.stack([expert(z) for expert in self.experts], dim=2)

        if self.router_type == "frame":
            weights = self._topk(self.router(z)).unsqueeze(3)
        else:
            router_in = self._mean(z, lengths)
            weights = self._topk(self.router(router_in)).unsqueeze(1).unsqueeze(3)

        return x + (expert_out * weights).sum(dim=2)


class CTCTransformerAdapter(nn.Module):
    """CTC Transformer adaptor。"""

    def __init__(
        self,
        dim: int,
        model_dim: int,
        ffn_dim: int,
        num_layers: int,
        num_heads: int,
        layer_ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._config = {
            "transformer_dim": model_dim,
            "transformer_ffn_dim": ffn_dim,
            "transformer_layers": num_layers,
            "transformer_heads": num_heads,
            "transformer_layer_ffn_dim": layer_ffn_dim,
        }
        self.input_norm = nn.LayerNorm(dim)
        self.skip_proj = nn.Identity() if dim == model_dim else nn.Linear(dim, model_dim)
        # self.adapter_scale = nn.Parameter(torch.tensor(0.1))
        self.in_proj = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, model_dim),
        )
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=layer_ffn_dim,
                    dropout=dropout,
                    activation="relu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.Identity()

    def _padding_mask(self, lengths: torch.Tensor, size: int, device) -> torch.Tensor:
        if lengths is None:
            return None
        pos = torch.arange(size, device=device).unsqueeze(0)
        return pos >= lengths.to(device=device).unsqueeze(1)

    def _attention_mask(self, size: int, device, mask_mode: str, chunk_size: int, left_chunks: int) -> torch.Tensor:
        if mask_mode == "offline":
            return None
        if mask_mode == "causal":
            return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
        if mask_mode != "chunk":
            raise ValueError(f"不支持的 CTC mask_mode: {mask_mode}")

        chunk_size = max(1, int(chunk_size))
        left_chunks = max(0, int(left_chunks))
        pos = torch.arange(size, device=device)
        q_chunk = pos // chunk_size
        k_pos = pos.unsqueeze(0)
        start = (q_chunk - left_chunks).clamp_min(0).unsqueeze(1) * chunk_size
        end = ((q_chunk + 1) * chunk_size).unsqueeze(1)
        return ~((k_pos >= start) & (k_pos < end))

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor = None,
        mask_mode: str = "offline",
        chunk_size: int = 1,
        left_chunks: int = 0,
    ) -> torch.Tensor:
        # z = self.skip_proj(x) + self.adapter_scale * self.in_proj(self.input_norm(x))
        z = self.skip_proj(x) + self.in_proj(self.input_norm(x))
        size = z.size(1)
        padding_mask = self._padding_mask(lengths, size, z.device)
        attention_mask = self._attention_mask(size, z.device, mask_mode, chunk_size, left_chunks)
        mha = getattr(torch.backends, "mha", None)
        old_fastpath = None
        if mha is not None and hasattr(mha, "get_fastpath_enabled") and hasattr(mha, "set_fastpath_enabled"):
            old_fastpath = mha.get_fastpath_enabled()
            mha.set_fastpath_enabled(False)
        try:
            for layer in self.layers:
                z = layer(z, src_mask=attention_mask, src_key_padding_mask=padding_mask)
        finally:
            if old_fastpath is not None:
                mha.set_fastpath_enabled(old_fastpath)
        return z

    def config(self) -> dict:
        return dict(self._config)


class CTC(nn.Module):
    """CTC 分支：shared_encoder_output -> adapter -> classifier -> log_softmax -> CTC Loss。"""

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        blank_id: int = 0,
        dropout: float = 0.1,
        bottleneck_dim: int = 1024,
        adapter_type: str = "mlp",
        transformer_dim: int = 1024,
        transformer_ffn_dim: int = 1024,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_layer_ffn_dim: int = 2048,
        blank_bias: float = -2.0,
    ):
        super().__init__()
        self.adapter_type = adapter_type.lower()
        self.bottleneck_dim = bottleneck_dim
        self.blank_bias = float(blank_bias)
        self.blank_id = blank_id

        if self.adapter_type == "transformer":
            self.adapter = CTCTransformerAdapter(
                encoder_output_size,
                model_dim=transformer_dim,
                ffn_dim=transformer_ffn_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                layer_ffn_dim=transformer_layer_ffn_dim,
                dropout=dropout,
            )
            self.ctc_lo = nn.Linear(transformer_dim, odim)
        else:
            if self.adapter_type == "moe":
                self.adapter = CTCMoEAdapter(encoder_output_size, dropout=dropout)
            else:
                self.adapter = CTCAdapter(encoder_output_size, dropout=dropout)
            self.ctc_lo = nn.Sequential(
                nn.Linear(encoder_output_size, bottleneck_dim),
                nn.SiLU(),
                nn.Linear(bottleneck_dim, odim),
            )

        self._init_blank_bias()
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def _init_blank_bias(self) -> None:
        linear = self.ctc_lo if isinstance(self.ctc_lo, nn.Linear) else self.ctc_lo[-1]
        with torch.no_grad():
            linear.bias[self.blank_id].fill_(self.blank_bias)

    def config(self) -> dict:
        cfg = {
            "adapter_type": self.adapter_type,
            "bottleneck_dim": self.bottleneck_dim,
            "blank_bias": self.blank_bias,
        }
        if hasattr(self.adapter, "config"):
            cfg.update(self.adapter.config())
        return cfg

    def log_softmax(
        self,
        hs_pad: torch.Tensor,
        hs_lengths: torch.Tensor = None,
        mask_mode: str = "offline",
        chunk_size: int = 1,
        left_chunks: int = 0,
    ) -> torch.Tensor:
        if self.adapter_type == "moe":
            ctc_hidden = self.adapter(hs_pad, hs_lengths)
        elif self.adapter_type == "transformer":
            ctc_hidden = self.adapter(hs_pad, hs_lengths, mask_mode=mask_mode, chunk_size=chunk_size, left_chunks=left_chunks)
        else:
            ctc_hidden = self.adapter(hs_pad)
        return F.log_softmax(self.ctc_lo(ctc_hidden), dim=2)

    def forward(
        self,
        hs_pad: torch.Tensor,
        hs_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        mask_mode: str = "offline",
        chunk_size: int = 1,
        left_chunks: int = 0,
    ) -> torch.Tensor:
        """计算 CTC loss。

        参数：
        - hs_pad: [B, T, D]
        - hs_lengths: [B]
        - targets: CTC 标签
        - target_lengths: 标签长度
        """
        log_probs = self.log_softmax(
            hs_pad,
            hs_lengths,
            mask_mode=mask_mode,
            chunk_size=chunk_size,
            left_chunks=left_chunks,
        ).transpose(0, 1).float()
        return self.ctc_loss(log_probs, targets, hs_lengths, target_lengths)

    @torch.no_grad()
    def greedy_decode(
        self,
        hs_pad: torch.Tensor,
        hs_lengths: torch.Tensor,
        mask_mode: str = "offline",
        chunk_size: int = 1,
        left_chunks: int = 0,
    ):
        """CTC 贪心解码。

        返回：
        - List[List[int]]
        - 已经做了 blank 去除
        - 已经做了连续重复 token 去重
        """
        log_probs = self.log_softmax(
            hs_pad,
            hs_lengths,
            mask_mode=mask_mode,
            chunk_size=chunk_size,
            left_chunks=left_chunks,
        )   # [B, T, V]
        preds = log_probs.argmax(dim=-1)       # [B, T]

        results = []
        for b in range(preds.size(0)):
            cur_len = int(hs_lengths[b].item())
            ids = preds[b, :cur_len].cpu().tolist()

            dedup_ids = []
            prev_id = -1
            for idx in ids:
                if idx != self.blank_id and idx != prev_id:
                    dedup_ids.append(idx)
                prev_id = idx

            results.append(dedup_ids)

        return results

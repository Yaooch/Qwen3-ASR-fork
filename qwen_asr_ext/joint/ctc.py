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


def build_ctc_adapter(adapter_type: str, dim: int, dropout: float = 0.1, hidden_mult: int = 2) -> nn.Module:
    if adapter_type == "moe":
        return CTCMoEAdapter(dim, dropout=dropout, hidden_mult=hidden_mult)
    return CTCAdapter(dim, dropout=dropout, hidden_mult=hidden_mult)


class CTC(nn.Module):
    """CTC 分支。

    整体结构：
        shared_encoder_output
            -> ctc_adapter
            -> linear classifier
            -> log_softmax
            -> CTC Loss

    说明：
    - adapter 用来把共享特征轻量适配到更适合 CTC 的空间
    - ctc_lo 负责把每个时间步映射到词表维度
    - blank_id 默认为 0
    """

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        blank_id: int = 0,
        dropout: float = 0.1,
        bottleneck_dim: int = 1024,
        adapter_type: str = "mlp",
    ):
        super().__init__()
        self.dropout = dropout
        self.bottleneck_dim = bottleneck_dim
        self.adapter_type = adapter_type
        self.adapter = build_ctc_adapter(
            adapter_type,
            encoder_output_size,
            dropout=dropout,
        )
        self.ctc_lo = nn.Sequential(
            nn.Linear(encoder_output_size, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, odim),
        )
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def get_ctc_hidden(self, hs_pad: torch.Tensor, hs_lengths: torch.Tensor = None) -> torch.Tensor:
        """对共享 encoder 输出做一层轻量适配。"""
        if self.adapter_type == "moe":
            return self.adapter(hs_pad, hs_lengths)
        return self.adapter(hs_pad)

    def log_softmax(self, hs_pad: torch.Tensor, hs_lengths: torch.Tensor = None) -> torch.Tensor:
        """输出每个时间步对词表的对数概率。"""
        ctc_hidden = self.get_ctc_hidden(hs_pad, hs_lengths)
        logits = self.ctc_lo(ctc_hidden)
        return F.log_softmax(logits, dim=2)

    def forward(
        self,
        hs_pad: torch.Tensor,
        hs_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """计算 CTC loss。

        参数：
        - hs_pad: [B, T, D]
        - hs_lengths: [B]
        - targets: CTC 标签
        - target_lengths: 标签长度
        """
        log_probs = self.log_softmax(hs_pad, hs_lengths).transpose(0, 1).float()
        return self.ctc_loss(log_probs, targets, hs_lengths, target_lengths)

    @torch.no_grad()
    def greedy_decode(self, hs_pad: torch.Tensor, hs_lengths: torch.Tensor, max_symbols_per_step: int = 0):
        """CTC 贪心解码（max_symbols_per_step 仅用于接口统一，CTC 忽略）。

        返回：
        - List[List[int]]
        - 已经做了 blank 去除
        - 已经做了连续重复 token 去重
        """
        log_probs = self.log_softmax(hs_pad, hs_lengths)   # [B, T, V]
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

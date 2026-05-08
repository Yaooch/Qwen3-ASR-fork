# qwen_asr/joint/ctc.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCAdapter(nn.Module):
    """CTC 分支的轻量残差适配层。

    输入已经过 audio tower 的 ln_post，这里不再做时序卷积，避免流式
    chunk 边界引入补零伪影。MLP 只负责把共享特征轻量转到 CTC 空间。
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
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.ffn(self.norm(x))


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
    ):
        super().__init__()
        self.dropout = dropout
        self.bottleneck_dim = bottleneck_dim
        self.adapter = CTCAdapter(encoder_output_size, dropout=dropout)
        self.ctc_lo = nn.Sequential(
            nn.Linear(encoder_output_size, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, odim),
        )
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def get_ctc_hidden(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """对共享 encoder 输出做一层轻量适配。"""
        return self.adapter(hs_pad)

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """输出每个时间步对词表的对数概率。"""
        ctc_hidden = self.get_ctc_hidden(hs_pad)
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
        log_probs = self.log_softmax(hs_pad).transpose(0, 1).float()
        return self.ctc_loss(log_probs, targets, hs_lengths, target_lengths)

    @torch.no_grad()
    def greedy_decode(self, hs_pad: torch.Tensor, hs_lengths: torch.Tensor):
        """CTC 贪心解码。

        返回：
        - List[List[int]]
        - 已经做了 blank 去除
        - 已经做了连续重复 token 去重
        """
        log_probs = self.log_softmax(hs_pad)   # [B, T, V]
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

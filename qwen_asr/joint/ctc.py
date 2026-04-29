# qwen_asr/joint/ctc.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCAdapter(nn.Module):
    """给 CTC 分支使用的小型残差适配层。

    设计目的：
    - 尽量不强行改动共享 encoder 的表示空间
    - 让 CTC 分支自己学习一小段“表征修正”
    - 比纯线性头更有表达能力，但参数量仍然可控

    结构：
        x -> Linear -> SiLU -> Dropout -> Linear
        -> Residual Add -> LayerNorm
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + residual
        x = self.norm(x)
        return x


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
    ):
        super().__init__()
        self.adapter = CTCAdapter(encoder_output_size, dropout=dropout)
        self.ctc_lo = nn.Linear(encoder_output_size, odim)
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

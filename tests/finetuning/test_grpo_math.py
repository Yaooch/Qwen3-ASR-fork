import torch

from finetuning.grpo_math import group_advantages, grpo_loss


def test_group_advantages_zero_mean_per_group():
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    a = group_advantages(r)
    assert a.shape == (1, 4)
    assert abs(float(a.sum())) < 1e-6


def test_grpo_loss_finite_and_sign():
    logp = torch.zeros(2, 4, requires_grad=True)
    old = torch.zeros(2, 4)
    adv = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    ref = torch.zeros(2, 4)
    loss = grpo_loss(logp, old, adv, ref, beta=0.04)
    assert torch.isfinite(loss)
    # 正优势 + logp==old → ratio=1 → surrogate=adv，loss=-mean(adv) < 0
    assert float(loss.detach()) < 0
    loss.backward()
    assert logp.grad is not None

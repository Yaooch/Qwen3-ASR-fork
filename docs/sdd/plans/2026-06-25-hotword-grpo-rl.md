# 热词 GRPO 强化学习 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use sdd:subagent-driven-development (recommended) or sdd:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 GRPO 强化学习训练 Qwen3-ASR talker（LoRA），使其在注入「真热词 + 干扰词」混合列表时选对并原样输出真正被说到的热词、不误注入干扰词、且非热词部分不退化。

**Architecture:** 自写轻量 GRPO loop（不套 trl）。每条样本用数据预给的热词 prompt，冻结 thinker 音频侧 / CTC / RNNT / lm_head，只 LoRA thinker 文本解码器；每条采样 G=8 个 rollout，用可验证奖励（热词召回 − 误注入 − 非热词 CER − 格式兜底）组内归一化作优势，clip 损失 + KL 正则更新 LoRA。

**Tech Stack:** PyTorch、transformers 4.57.6、peft（LoRA）、editdistance（CER）、librosa、datasets、accelerate、pytest。

## Global Constraints

- 基线分支 `new-rag`，工作分支 `rl/hotword-grpo`（已创建）。
- 训练数据：`/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl`（172443 行，每行 `{"audio","text","prompt"}`）。
- 权重 ckpt：`/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228`（`Qwen3ASRForConditionalGeneration`，`model_type=qwen3_asr`）。
- LoRA target：`qwen_asr` thinker 文本解码器 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，`r=16,alpha=32,dropout=0.05`，bf16。冻结 audio_tower / CTC / RNNT / lm_head / 原始权重。
- 模型输出原生带 `language X<asr_text>` 前缀，奖励前用 `parse_asr_output`（`qwen_asr.inference.utils`）剥前缀，再去标点。
- 依赖 `peft`、`pytest` 未安装，需 `pip install -i https://pypi.org/simple/ peft pytest`（默认镜像 SSL 故障，用 pypi 直连）。
- 不动现有 `transcribe` 推理路径；RL 训完 LoRA 可选 merge 或挂载进 `infer`（本期不做 merge 流程）。
- 遵循 `.claude/rules/common/coding-style.md`：不可变、小文件、显式错误处理、无硬编码 magic、外科式编辑。

## File Structure

| 文件 | 责任 | 新增/修改 |
|------|------|----------|
| `qwen_asr/tools/hotword_reward.py` | 奖励纯函数：归一化、解析样本、T/D 划分、recall/fp/cer_nh、合成奖励 | 新增 |
| `finetuning/grpo_data.py` | 读 jsonl → 缓存训练样本（audio, G_text, H, T, D, prompt），切 eval | 新增 |
| `finetuning/grpo_math.py` | GRPO 纯数学：组内优势、clip surrogate 损失、KL | 新增 |
| `finetuning/grpo_lora.py` | LoRA 装配：挂 LoRA、冻结其余、暴露 π_ref 路径 | 新增 |
| `finetuning/grpo_rollout.py` | 音频 embedding 缓存 + G 路采样 + token logprob | 新增 |
| `finetuning/grpo_train.py` | GRPO 主循环：装配模型/数据/LoRA、step、ckpt、argparse | 新增 |
| `finetuning/grpo_eval.py` | 评测：热词召回/误注入/非热词 CER/整体 CER，对比基线 | 新增 |
| `finetuning/grpo_train.sh` | 训练 shell wrapper | 新增 |
| `finetuning/grpo_eval.sh` | 评测 shell wrapper | 新增 |
| `tests/tools/test_hotword_reward.py` | 奖励单测 | 新增 |
| `tests/finetuning/test_grpo_data.py` | 数据单测 | 新增 |
| `tests/finetuning/test_grpo_math.py` | GRPO 数学单测 | 新增 |
| `tests/conftest.py` | pytest 路径根 | 新增 |
| `AGENTS.md` | 记录 RL 子包与开关 | 修改 |

---

### Task 1: 环境与测试脚手架

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`, `tests/tools/__init__.py`, `tests/finetuning/__init__.py`（空文件，使 pytest 发现包）

**Interfaces:**
- Produces: 可运行的 pytest（`python -m pytest`）

- [ ] **Step 1: 安装依赖**

Run: `pip install -i https://pypi.org/simple/ peft pytest`
Expected: `Successfully installed peft-0.19.1 pytest-...`

- [ ] **Step 2: 建空 __init__.py**

```python
# tests/__init__.py  (空)
# tests/tools/__init__.py  (空)
# tests/finetuning/__init__.py  (空)
```

- [ ] **Step 3: 建 conftest.py（把仓库根加入 sys.path）**

```python
# tests/conftest.py
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
```

- [ ] **Step 4: 验证 pytest 可跑**

Run: `python -m pytest tests/ -q`
Expected: `no tests ran`（无错），退出码 5 或 0。

- [ ] **Step 5: Commit**

```bash
git add tests/ && git commit -m "feat(grpo): add pytest scaffold and install peft/pytest deps"
```

---

### Task 2: 奖励纯函数（TDD）

**Files:**
- Create: `qwen_asr/tools/hotword_reward.py`
- Test: `tests/tools/test_hotword_reward.py`

**Interfaces:**
- Produces:
  - `normalize(text: str) -> str`：去标点、小写、压缩空白
  - `parse_text_field(raw: str) -> str`：从 `text` 字段剥 `language X<asr_text>` 得纯转写
  - `parse_hotword_list(prompt: str) -> list[str]`：从 prompt 抠 `专属名词：[...]` 词表
  - `split_truth(hotwords: list[str], gt_text: str) -> tuple[set[str], set[str]]`：返回 `(T, D)`，基于 normalize 后子串匹配
  - `hotword_recall(output: str, true_hw: set[str]) -> float`
  - `false_injection_rate(output: str, distractors: set[str]) -> float`
  - `non_hotword_cer(output: str, gt_text: str, hotwords: list[str]) -> float`
  - `compute_reward(output: str, gt_text: str, hotwords: list[str], weights: dict|None=None) -> float`

- [ ] **Step 1: 写失败测试**

```python
# tests/tools/test_hotword_reward.py
from qwen_asr.tools.hotword_reward import (
    normalize, parse_text_field, parse_hotword_list, split_truth,
    hotword_recall, false_injection_rate, non_hotword_cer, compute_reward,
)

def test_normalize_strips_punct_and_lowercases():
    assert normalize("Hello, 世界！ WORLD.") == "hello 世界 world"

def test_parse_text_field_strips_prefix():
    raw = "language Chinese<asr_text>你们看高志森那部电影了吗"
    assert parse_text_field(raw) == "你们看高志森那部电影了吗"

def test_parse_hotword_list():
    prompt = "转写语音，专属名词优先按列表原文输出。\n专属名词：[高志森，小鬼三个爸，洪金宝]"
    assert parse_hotword_list(prompt) == ["高志森", "小鬼三个爸", "洪金宝"]

def test_split_truth_separates_spoken_vs_distractor():
    hw = ["高志森", "小鬼三个爸", "洪金宝", "伊佐美纪"]  # 伊佐美纪 没说到
    gt = "你们看高志森那部小鬼三个爸了吗 洪金宝演的"
    T, D = split_truth(hw, gt)
    assert T == {"高志森", "小鬼三个爸", "洪金宝"}
    assert D == {"伊佐美纪"}

def test_recall_and_fp():
    out = "你们看高志森那部小鬼三个爸了吗"
    T = {"高志森", "小鬼三个爸", "洪金宝"}
    D = {"伊佐美纪"}
    assert hotword_recall(out, T) == 2/3
    assert false_injection_rate(out, D) == 0.0
    # 误注入
    out2 = "你们看高志森那部伊佐美纪了吗"
    assert false_injection_rate(out2, D) == 1.0

def test_non_hotword_cer_isolated():
    out = "你们看高志森那部电影了吗"
    gt = "你们看高志森那部电影了吗"
    hw = ["高志森", "伊佐美纪"]
    assert non_hotword_cer(out, gt, hw) == 0.0
    # 非热词部分错一字
    out2 = "你们看高志森那部电形了吗"
    assert non_hotword_cer(out2, gt, hw) > 0.0

def test_compute_reward_shape():
    out = "你们看高志森那部小鬼三个爸了吗 洪金宝演的"
    gt = "你们看高志森那部小鬼三个爸了吗 洪金宝演的"
    hw = ["高志森", "小鬼三个爸", "洪金宝", "伊佐美纪"]
    r = compute_reward(out, gt, hw)
    # 召回满、无误注入、CER 0 → 接近 1.0
    assert r > 0.9

def test_compute_reward_empty_truth_no_injection():
    out = "今天天气不错"
    gt = "今天天气不错"
    hw = ["伊佐美纪"]  # 无真热词，且未注入
    r = compute_reward(out, gt, hw)
    assert r >= 0.0  # 不奖不罚 recall 项，CER 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/tools/test_hotword_reward.py -v`
Expected: FAIL（ModuleNotFoundError / 函数未定义）。

- [ ] **Step 3: 写实现**

```python
# qwen_asr/tools/hotword_reward.py
"""热词 GRPO 奖励：纯函数，可验证。"""
import re
from typing import List, Sequence, Set, Tuple

# CJK + ASCII 标点（保留字母数字汉字与空格）
_PUNCT_RE = re.compile(
    r"[!-/:-@\[-`{-~　-〿＀-￯ -⁯]"
)
_WS_RE = re.compile(r"\s+")

DEFAULT_WEIGHTS = {"w_r": 1.0, "w_f": 1.0, "w_c": 0.5, "w_fmt": 0.5}


def normalize(text: str) -> str:
    if not text:
        return ""
    s = _PUNCT_RE.sub(" ", str(text))
    s = _WS_RE.sub(" ", s).strip()
    return s.lower()


def parse_text_field(raw: str) -> str:
    """从训练数据 text 字段剥 language X<asr_text> 前缀，返回归一化纯转写。"""
    if raw is None:
        return ""
    s = str(raw).strip()
    tag = "<asr_text>"
    if tag in s:
        s = s.split(tag, 1)[1]
    return normalize(s)


def parse_hotword_list(prompt: str) -> List[str]:
    """从 prompt 抠 专属名词：[a，b，c] 词表，去标点归一。"""
    if not prompt:
        return []
    m = re.search(r"专属名词：\[([^\]]*)\]", str(prompt))
    if not m:
        return []
    inner = m.group(1)
    parts = re.split(r"[，,]", inner)
    words = [normalize(p) for p in parts]
    return [w for w in words if w]


def _substring_match(hotword: str, text: str) -> bool:
    return bool(hotword) and hotword in text


def split_truth(
    hotwords: Sequence[str], gt_text: str
) -> Tuple[Set[str], Set[str]]:
    """T = 列表中出现在 gt 的；D = 其余。基于 normalize 后子串。"""
    gt = normalize(gt_text)
    true_set, dist = set(), set()
    for h in hotwords:
        h = normalize(h)
        (true_set if _substring_match(h, gt) else dist).add(h)
    return true_set, dist


def hotword_recall(output: str, true_hw: Set[str]) -> float:
    if not true_hw:
        return 0.0
    out = normalize(output)
    hit = sum(1 for h in true_hw if _substring_match(h, out))
    return hit / max(1, len(true_hw))


def false_injection_rate(output: str, distractors: Set[str]) -> float:
    if not distractors:
        return 0.0
    out = normalize(output)
    hit = sum(1 for h in distractors if _substring_match(h, out))
    return hit / max(1, len(distractors))


def _mask_all(text: str, words: Sequence[str]) -> str:
    s = normalize(text)
    # 按长度降序避免短词遮蔽长词
    for w in sorted({normalize(w) for w in words if normalize(w)}, key=len, reverse=True):
        if w:
            s = s.replace(w, " ")
    return _WS_RE.sub(" ", s).strip()


def non_hotword_cer(output: str, gt_text: str, hotwords: Sequence[str]) -> float:
    pred = _mask_all(output, hotwords)
    ref = _mask_all(gt_text, hotwords)
    if not ref:
        return 0.0
    import editdistance
    return editdistance.eval(pred, ref) / len(ref)


def _format_ok(output: str) -> bool:
    s = str(output or "").strip()
    if not s:
        return False
    if "专属名词" in s or "[" in s and "]" in s:
        return False
    return True


def compute_reward(
    output: str,
    gt_text: str,
    hotwords: Sequence[str],
    weights: dict | None = None,
) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    T, D = split_truth(hotwords, gt_text)
    recall = hotword_recall(output, T)
    fp = false_injection_rate(output, D)
    cer_nh = non_hotword_cer(output, gt_text, hotwords)
    fmt = 1.0 if _format_ok(output) else 0.0
    return (
        w["w_r"] * recall
        - w["w_f"] * fp
        - w["w_c"] * cer_nh
        - w["w_fmt"] * (1.0 - fmt)
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/tools/test_hotword_reward.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add qwen_asr/tools/hotword_reward.py tests/tools/test_hotword_reward.py
git commit -m "feat(grpo): add pure hotword reward functions with tests"
```

---

### Task 3: 数据准备（TDD）

**Files:**
- Create: `finetuning/grpo_data.py`
- Test: `tests/finetuning/test_grpo_data.py`

**Interfaces:**
- Produces:
  - `GrpoSample`（dataclass）：`audio: str`、`gt_text: str`（parse 后归一）、`hotwords: list[str]`、`prompt: str`（原 prompt 原样，供 rollout 拼 LLM 输入）
  - `load_samples(jsonl_path: str, limit: int|None=None) -> list[GrpoSample]`
  - `split_eval(samples: list[GrpoSample], eval_ratio: float=0.02, seed: int=42) -> tuple[list, list]`

- [ ] **Step 1: 写失败测试**

```python
# tests/finetuning/test_grpo_data.py
import json, os, tempfile
from finetuning.grpo_data import load_samples, split_eval

def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def test_load_samples_parses_fields():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        _write(f.name, [
            {"audio": "/a.wav", "text": "language Chinese<asr_text>洪金宝演的",
             "prompt": "转写语音。\n专属名词：[洪金宝，伊佐美纪]"},
        ])
        path = f.name
    samples = load_samples(path)
    os.unlink(path)
    assert len(samples) == 1
    s = samples[0]
    assert s.audio == "/a.wav"
    assert s.gt_text == "洪金宝演的"  # normalize 后
    assert s.hotwords == ["洪金宝", "伊佐美纪"]
    assert "专属名词" in s.prompt

def test_split_eval_disjoint_and_ratio():
    rows = [{"audio": f"/{i}.wav", "text": f"language Chinese<asr_text>x{i}",
             "prompt": "专属名词：[a，b]"} for i in range(200)]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        _write(f.name, rows); path = f.name
    samples = load_samples(path)
    os.unlink(path)
    tr, ev = split_eval(samples, eval_ratio=0.1, seed=42)
    assert len(ev) == 20
    assert len(tr) == 180
    assert set(id(x) for x in tr).isdisjoint(set(id(x) for x in ev))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/finetuning/test_grpo_data.py -v`
Expected: FAIL（ModuleNotFoundError）。

- [ ] **Step 3: 写实现**

```python
# finetuning/grpo_data.py
"""读 ContextASR jsonl → GRPO 训练样本，切 eval。"""
import json
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from qwen_asr.tools.hotword_reward import parse_hotword_list, parse_text_field


@dataclass(frozen=True)
class GrpoSample:
    audio: str
    gt_text: str          # parse + normalize 后纯转写
    hotwords: List[str]   # normalize 后热词列表
    prompt: str           # 原 prompt 字段（rollout 时原样用作 context）


def load_samples(jsonl_path: str, limit: Optional[int] = None) -> List[GrpoSample]:
    samples: List[GrpoSample] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            samples.append(GrpoSample(
                audio=r["audio"],
                gt_text=parse_text_field(r.get("text", "")),
                hotwords=parse_hotword_list(r.get("prompt", "")),
                prompt=r.get("prompt", ""),
            ))
    return samples


def split_eval(
    samples: List[GrpoSample], eval_ratio: float = 0.02, seed: int = 42
) -> Tuple[List[GrpoSample], List[GrpoSample]]:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_eval = int(len(idx) * eval_ratio)
    eval_idx = set(idx[:n_eval])
    train = [samples[i] for i in range(len(samples)) if i not in eval_idx]
    eval_ = [samples[i] for i in idx[:n_eval]]
    return train, eval_
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/finetuning/test_grpo_data.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add finetuning/grpo_data.py tests/finetuning/test_grpo_data.py
git commit -m "feat(grpo): add data loader and eval split with tests"
```

---

### Task 4: GRPO 纯数学（TDD）

**Files:**
- Create: `finetuning/grpo_math.py`
- Test: `tests/finetuning/test_grpo_math.py`

**Interfaces:**
- Produces:
  - `group_advantages(rewards: torch.Tensor) -> torch.Tensor`：shape `(B, G)` → 同形状优势 `(r - r.mean(dim=1,keepdim)) / (r.std(dim=1,keepdim) + 1e-8)`
  - `grpo_loss(logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor, ref_logp: torch.Tensor, beta: float=0.04) -> torch.Tensor`：clip surrogate（ratio=`exp(logp-old_logp)`，`clip(ratio,1-ε,1+ε)`，ε=0.2）+ KL(`logp - ref_logp` 简化估计）取负均值

- [ ] **Step 1: 写失败测试**

```python
# tests/finetuning/test_grpo_math.py
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
    # 正优势 + logp==old → ratio=1 → surrogate = adv，loss=-mean(adv) < 0
    assert float(loss) < 0
    loss.backward()
    assert logp.grad is not None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/finetuning/test_grpo_math.py -v`
Expected: FAIL。

- [ ] **Step 3: 写实现**

```python
# finetuning/grpo_math.py
"""GRPO 纯数学：组内优势 + clip surrogate + KL。"""
import torch

EPS_CLIP = 0.2


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """rewards: (B, G) → 组内归一化优势 (B, G)。"""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + 1e-8)


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    ref_logp: torch.Tensor,
    beta: float = 0.04,
) -> torch.Tensor:
    """token-level：logp/old_logp/ref_logp/advantages 同 shape。
    返回标量 loss = -mean(clip_surrogate) + beta * mean(KL)。
    KL 简化为 (logp - ref_logp).detach() * logp 的近似的反向：用 (logp - ref_logp) 作为惩罚项梯度。"""
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP)
    surrogate = torch.min(ratio * advantages, clipped * advantages)
    kl = (logp - ref_logp)
    return -surrogate.mean() + beta * kl.mean()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/finetuning/test_grpo_math.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add finetuning/grpo_math.py tests/fineturing/test_grpo_math.py
git commit -m "feat(grpo): add advantage and clip-surrogate loss math with tests"
```

---

### Task 5: LoRA 装配

**Files:**
- Create: `finetuning/grpo_lora.py`
- Test: `tests/finetuning/test_grpo_lora.py`（集成：需真实 ckpt，标 `@pytest.mark.skip(reason="需真实 ckpt")` 默认跳过，提供手动 smoke 命令）

**Interfaces:**
- Consumes: `Qwen3ASRJointModel.from_pretrained(ckpt)`
- Produces:
  - `apply_lora(joint, r=16, alpha=32, dropout=0.05) -> peft.PeftModel`：挂 LoRA 到 thinker 文本解码器，冻结其余
  - `assert_only_text_decoder_trainable(peft_model)`：断言可训参数路径全在 `thinker.model`

- [ ] **Step 1: 写实现（LoRA 装配 + 断言）**

```python
# finetuning/grpo_lora.py
"""挂 LoRA 到 Qwen3-ASR thinker 文本解码器，冻结其余。"""
from typing import Optional

TEXT_DECODER_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def apply_lora(joint, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=TEXT_DECODER_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # 冻结全部，再由 peft 只解冻 LoRA
    for p in joint.parameters():
        p.requires_grad_(False)
    peft_model = get_peft_model(joint, cfg)
    assert_only_text_decoder_trainable(peft_model)
    return peft_model


def assert_only_text_decoder_trainable(peft_model) -> None:
    bad = []
    for name, p in peft_model.named_parameters():
        if p.requires_grad and "thinker.model" not in name:
            bad.append(name)
    if bad:
        raise RuntimeError(
            f"LoRA 误挂到非文本解码器: {bad[:5]}（共 {len(bad)} 个）"
        )
```

- [ ] **Step 2: 写集成 smoke 测试（默认 skip）**

```python
# tests/finetuning/test_grpo_lora.py
import pytest

CKPT = "/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"


@pytest.mark.skip(reason="需真实 ckpt，手动跑: pytest -m integration")
def test_apply_lora_only_text_decoder_trainable():
    import torch
    from qwen_asr.joint import Qwen3ASRJointModel
    from finetuning.grpo_lora import apply_lora, assert_only_text_decoder_trainable

    joint = Qwen3ASRJointModel.from_pretrained(CKPT, dtype="bf16", device_map="cuda")
    peft_model = apply_lora(joint)
    assert_only_text_decoder_trainable(peft_model)
    n = sum(1 for p in peft_model.parameters() if p.requires_grad)
    assert n > 0
```

- [ ] **Step 3: 在 pytest.ini 注册 marker**

Create `pytest.ini`:
```ini
[pytest]
markers =
    integration: 需真实模型 ckpt 的集成测试（默认不收集）
addopts = -m "not integration"
```

- [ ] **Step 4: 跑单测确认 import 通过、集成默认跳过**

Run: `python -m pytest tests/finetuning/test_grpo_lora.py -v`
Expected: 1 skipped。

- [ ] **Step 5: 手动集成 smoke（可选，有 GPU 时）**

Run: `python -m pytest tests/finetuning/test_grpo_lora.py -m integration -v`
Expected: PASS（LoRA 仅挂 thinker.model）。

- [ ] **Step 6: Commit**

```bash
git add finetuning/grpo_lora.py tests/finetuning/test_grpo_lora.py pytest.ini
git commit -m "feat(grpo): apply LoRA to thinker text decoder, freeze rest"
```

---

### Task 6: Rollout 模块（采样 + logprob）

**Files:**
- Create: `finetuning/grpo_rollout.py`
- Test: 集成 smoke 脚本 `finetuning/grpo_rollout_smoke.py`（手动跑，不进 pytest 默认）

**Interfaces:**
- Consumes: `Qwen3ASRJointModel`（已挂 LoRA）、`GrpoSample`
- Produces:
  - `RolloutResult`（dataclass）：`ids: torch.LongTensor (1, T)`、`text: str`、`logp_ref: torch.Tensor (T,)`（base 关闭 LoRA 的 logprob）
  - `class RolloutSampler`：
    - `__init__(self, joint_peft, processor, asr_wrapper, group_size=8, temperature=0.8, max_new_tokens=512, device="cuda")`
    - `audio_embedding(self, audio_path: str) -> torch.Tensor`：缓存（同音频多 rollout 只算一次）
    - `build_inputs(self, sample) -> (input_ids, attn, audio_embeds, audio_mask)`
    - `sample(self, sample) -> list[RolloutResult]`：用 LoRA-on 采 G 条；对每条用 LoRA-off 算 ref logp
    - `token_logp(self, input_ids, audio_embeds, audio_mask, gen_ids) -> torch.Tensor`：LoRA-on 前向算生成 token logprob（训练用）

- [ ] **Step 1: 写实现**

```python
# finetuning/grpo_rollout.py
"""GRPO rollout：音频 embedding 缓存 + G 路采样 + token logprob。"""
from dataclasses import dataclass
from typing import List

import librosa
import torch
import torch.nn.functional as F


@dataclass
class RolloutResult:
    ids: torch.LongTensor       # (1, T_gen) 生成 token
    text: str
    logp_ref: torch.Tensor      # (T_gen,) base(LoRA-off) logprob，detach


class RolloutSampler:
    def __init__(
        self,
        joint_peft,
        processor,
        asr_wrapper,
        group_size: int = 8,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ):
        self.joint = joint_peft
        self.processor = processor
        self.asr_wrapper = asr_wrapper
        self.G = group_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._audio_cache = {}

    def audio_embedding(self, audio_path: str) -> torch.Tensor:
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        wav = wav.astype("float32")
        fe = self.processor.feature_extractor
        batch = fe(
            [wav], sampling_rate=16000, return_tensors="pt",
            padding=True, truncation=False, return_attention_mask=True,
        )
        feats = batch["input_features"]
        mask = batch.get("feature_attention_mask", batch.get("attention_mask"))
        ref = next(self.joint.parameters())
        feats = feats.to(device=ref.device, dtype=ref.dtype)
        if mask is not None:
            mask = mask.to(device=ref.device)
        from qwen_asr.joint.encoder import encode_offline, feature_lens
        lens = feature_lens(feats, mask)
        tower = self.joint.qwen_model.thinker.audio_tower
        _, llm, _ = encode_offline(tower, feats, lens, need_llm=True)
        self._audio_cache[audio_path] = llm[0]
        return llm[0]

    def build_inputs(self, sample):
        thinker = self.joint.qwen_model.thinker
        processor = self.processor
        token = processor.audio_token
        base = sample.prompt or ""
        text = self.asr_wrapper._build_text_prompt(context=base, force_language=None)
        # 拼 audio placeholder
        llm = self.audio_embedding(sample.audio)
        text = text.replace(token, token * int(llm.shape[0]), 1)
        old = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            tok = processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        finally:
            processor.tokenizer.padding_side = old
        input_ids = tok["input_ids"].to(self.device)
        attn = tok["attention_mask"].to(self.device)
        embeds = thinker.get_input_embeddings()(input_ids)
        audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=embeds)
        audio_embeds = llm.to(dtype=embeds.dtype)
        return input_ids, attn, embeds, audio_mask, audio_embeds

    @torch.no_grad()
    def _generate_one(self, input_ids, attn, embeds, audio_mask, audio_embeds):
        thinker = self.joint.qwen_model.thinker
        inputs_embeds = embeds.masked_scatter(audio_mask, audio_embeds)
        gen = self.joint.qwen_model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            return_dict_in_generate=True,
            output_scores=False,
        )
        seq = gen.sequences
        gen_ids = seq[:, input_ids.shape[1]:]
        from qwen_asr.inference.utils import parse_asr_output
        raw = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        _, text = parse_asr_output(raw)
        return gen_ids[0], text

    def _logp_of(self, input_ids, attn, embeds, audio_mask, audio_embeds, gen_ids, use_lora: bool):
        """前向算 gen_ids 的 token logprob。use_lora=False 时关 LoRA（ref）。"""
        from contextlib import nullcontext
        thinker = self.joint.qwen_model.thinker
        ctx = self.joint.disable_adapter() if (not use_lora and hasattr(self.joint, "disable_adapter")) else nullcontext()
        full_ids = torch.cat([input_ids, gen_ids.unsqueeze(0)], dim=1)
        full_attn = torch.ones_like(full_ids)
        inputs_embeds = thinker.get_input_embeddings()(full_ids)
        # 重建 audio_mask 与 audio_embeds 对齐（位置不变）
        am = thinker.get_placeholder_mask(full_ids, inputs_embeds=inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(am, audio_embeds)
        with ctx:
            logits = self.joint.qwen_model.thinker(
                input_ids=full_ids,
                attention_mask=full_attn,
                inputs_embeds=inputs_embeds,
            ).logits
        # logits: (1, T, V)；取生成段
        log_logits = F.log_softmax(logits[:, input_ids.shape[1]-1:-1, :], dim=-1)
        gen_ids = gen_ids.to(self.device)
        logp = log_logits.gather(-1, gen_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        return logp.squeeze(0)

    def sample(self, sample) -> List[RolloutResult]:
        input_ids, attn, embeds, audio_mask, audio_embeds = self.build_inputs(sample)
        results = []
        for _ in range(self.G):
            gen_ids, text = self._generate_one(input_ids, attn, embeds, audio_mask, audio_embeds)
            ref_logp = self._logp_of(input_ids, attn, embeds, audio_mask, audio_embeds, gen_ids, use_lora=False).detach()
            results.append(RolloutResult(ids=gen_ids.detach(), text=text, logp_ref=ref_logp))
        return results

    def token_logp(self, sample, gen_ids) -> torch.Tensor:
        """训练时用 LoRA-on 重算 gen_ids 的 logp（带梯度）。"""
        input_ids, attn, embeds, audio_mask, audio_embeds = self.build_inputs(sample)
        return self._logp_of(input_ids, attn, embeds, audio_mask, audio_embeds, gen_ids, use_lora=True)
```

> **注意**：`self.joint.qwen_model.thinker(...)` 的前向签名以仓库内 `modeling_qwen3_asr.py` 的 `Qwen3ASRThinkerForConditionalGeneration.forward` 为准（传 `input_ids`/`attention_mask`/`inputs_embeds` 返回含 `.logits`）。实现时若签名不符，按实际源码调整调用——这是本任务最大的不确定性点，集成 smoke 必须验证。

- [ ] **Step 2: 写 smoke 脚本**

```python
# finetuning/grpo_rollout_smoke.py
"""手动跑：python -m finetuning.grpo_rollout_smoke --ckpt <ckpt> --audio <wav>"""
import argparse, torch
from qwen_asr.joint import Qwen3ASRJointModel
from finetuning.grpo_data import GrpoSample
from finetuning.grpo_lora import apply_lora
from finetuning.grpo_rollout import RolloutSampler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--prompt", default="专属名词：[测试词]")
    args = ap.parse_args()
    joint = Qwen3ASRJointModel.from_pretrained(args.ckpt, dtype="bf16", device_map="cuda")
    joint.setup_asr_wrapper()  # 若该名不存在，见 model.py 实际初始化 _asr_wrapper 的方法
    peft = apply_lora(joint)
    sampler = RolloutSampler(peft, joint.processor, joint._asr_wrapper, group_size=2)
    s = GrpoSample(audio=args.audio, gt_text="", hotwords=["测试词"], prompt=args.prompt)
    rs = sampler.sample(s)
    for i, r in enumerate(rs):
        print(i, repr(r.text), "ref_logp", tuple(float(x) for x in r.logp_ref[:4]))
    logp = sampler.token_logp(s, rs[0].ids)
    print("train logp finite:", bool(torch.isfinite(logp).all()))

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 手动 smoke（有 GPU 时）**

Run: `python -m finetuning.grpo_rollout_smoke --ckpt <CKPT> --audio <一条 ContextASR wav>`
Expected: 打印 2 条不同采样文本 + ref_logp 数值 + `train logp finite: True`。若 thinker 前向签名不符，按报错调整 `_logp_of` 调用。

- [ ] **Step 4: Commit**

```bash
git add finetuning/grpo_rollout.py finetuning/grpo_rollout_smoke.py
git commit -m "feat(grpo): rollout sampler with audio embedding cache, sampling, logprob"
```

---

### Task 7: GRPO 训练主循环

**Files:**
- Create: `finetuning/grpo_train.py`
- Test: 集成 smoke：`python -m finetuning.grpo_train --smoke ...`

**Interfaces:**
- Consumes: Task 2/3/4/5/6 全部
- Produces: `finetuning/grpo_train.py` 带 argparse，可 `--smoke` 跑 2 样本 1 step 自检

- [ ] **Step 1: 写实现**

```python
# finetuning/grpo_train.py
"""GRPO 训练主循环。"""
import argparse
import os
import torch
from torch.optim import AdamW

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.tools.hotword_reward import compute_reward
from finetuning.grpo_data import load_samples, split_eval
from finetuning.grpo_lora import apply_lora
from finetuning.grpo_math import group_advantages, grpo_loss
from finetuning.grpo_rollout import RolloutSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--smoke", action="store_true", help="2 样本 1 step 自检")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    joint = Qwen3ASRJointModel.from_pretrained(args.ckpt, dtype="bf16", device_map="cuda")
    # 初始化 _asr_wrapper（见 model.py 实际方法名）
    if joint._asr_wrapper is None:
        # Qwen3ASRModel 自带 wrapper；按现有 transcribe 路径同样初始化
        joint._asr_wrapper = joint.qwen_model  # 占位，若不符用 model.py 里的真实初始化
    peft = apply_lora(joint)
    sampler = RolloutSampler(
        peft, joint.processor, joint._asr_wrapper,
        group_size=args.group_size, temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    trainable = [p for p in peft.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr)

    samples = load_samples(args.data, limit=4 if args.smoke else None)
    train, eval_ = split_eval(samples, eval_ratio=0.02)
    if args.smoke:
        train = train[:2]; eval_ = eval_[:2]; args.max_steps = 1

    step = 0
    for sample in train:
        if step >= args.max_steps:
            break
        # 1) rollout（no_grad）
        with torch.no_grad():
            rollouts = sampler.sample(sample)
        # 2) reward
        rewards = torch.tensor([
            compute_reward(r.text, sample.gt_text, sample.hotwords) for r in rollouts
        ], device="cuda")
        if rewards.std() < 1e-6:
            step += 1
            continue  # 组内无区分，跳过
        adv = group_advantages(rewards.unsqueeze(0)).squeeze(0)  # (G,)
        # 3) LoRA-on logp（带梯度）
        opt.zero_grad()
        loss = 0.0
        for r, a in zip(rollouts, adv):
            logp = sampler.token_logp(sample, r.ids)  # (T,)
            old_logp = logp.detach()
            loss = loss + grpo_loss(
                logp, old_logp, a.expand_as(logp), r.logp_ref.to(logp.dtype), beta=args.beta,
            )
        loss = loss / len(rollouts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"step {step} loss {float(loss):.4f} reward_mean {float(rewards.mean()):.3f}")
        step += 1

    peft.save_pretrained(os.path.join(args.output_dir, "lora"))
    print("saved LoRA to", os.path.join(args.output_dir, "lora"))


if __name__ == "__main__":
    main()
```

> **注意**：`joint._asr_wrapper` 的初始化方式以 `qwen_asr/joint/model.py` 实际代码为准（`from_pretrained` 是否已设 wrapper）。实现时若 `joint.qwen_model` 不是 wrapper，按 `model.py` 中 `transcribe` 使用 `_asr_wrapper` 的同一初始化路径修正。

- [ ] **Step 2: smoke 自检**

Run: `python -m finetuning.grpo_train --ckpt <CKPT> --data <jsonl> --output_dir /tmp/grpo_smoke --smoke`
Expected: 打印 `step 0 loss <有限> reward_mean <数>` 与 `saved LoRA to /tmp/grpo_smoke/lora`，无异常。loss finite、LoRA 文件落盘。

- [ ] **Step 3: Commit**

```bash
git add finetuning/grpo_train.py
git commit -m "feat(grpo): GRPO training loop with smoke self-check"
```

---

### Task 8: 评测脚本

**Files:**
- Create: `finetuning/grpo_eval.py`
- Create: `finetuning/grpo_eval.sh`

**Interfaces:**
- Consumes: Task 2（奖励分量）+ `infer.py` 推理路径（或直接用 joint.transcribe）
- Produces: 打印热词召回 / 误注入率 / 非热词 CER / 整体 CER，对比 RL 前后

- [ ] **Step 1: 写实现**

```python
# finetuning/grpo_eval.py
"""评测 GRPO LoRA：热词召回 / 误注入 / 非热词 CER / 整体 CER。"""
import argparse
import json
import torch

from qwen_asr.joint import Qwen3ASRJointModel
from qwen_asr.tools.hotword_reward import (
    split_truth, hotword_recall, false_injection_rate, non_hotword_cer, normalize,
)
from finetuning.grpo_data import load_samples
import editdistance


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--lora", default=None, help="RL 训出的 LoRA 目录；不传则评基线")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--mode", default="llm")
    return p.parse_args()


def overall_cer(output, gt):
    o, g = normalize(output), normalize(gt)
    return editdistance.eval(o, g) / max(1, len(g))


def main():
    args = parse_args()
    joint = Qwen3ASRJointModel.from_pretrained(args.ckpt, dtype="bf16", device_map="cuda")
    if args.lora:
        from peft import PeftModel
        joint.qwen_model = PeftModel.from_pretrained(joint.qwen_model, args.lora)
    samples = load_samples(args.data, limit=args.limit)
    agg = {"recall": [], "fp": [], "cer_nh": [], "cer": []}
    for s in samples:
        rec = joint.transcribe(s.audio, modes="llm", prompt=s.prompt)
        out = rec.get("text", "")
        T, D = split_truth(s.hotwords, s.gt_text)
        agg["recall"].append(hotword_recall(out, T))
        agg["fp"].append(false_injection_rate(out, D))
        agg["cer_nh"].append(non_hotword_cer(out, s.gt_text, s.hotwords))
        agg["cer"].append(overall_cer(out, s.gt_text))
    for k in agg:
        v = agg[k]
        print(f"{k}: mean={sum(v)/len(v):.4f} n={len(v)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写 shell wrapper**

```bash
# finetuning/grpo_eval.sh
#!/usr/bin/env bash
set -euo pipefail
CKPT="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"
DATA="/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl"
LORA="${1:-}"        # 传 LoRA 目录评 RL 后；不传评基线
LIMIT="${2:-200}"
if [ -n "$LORA" ]; then
  python -m finetuning.grpo_eval --ckpt "$CKPT" --data "$DATA" --lora "$LORA" --limit "$LIMIT"
else
  python -m finetuning.grpo_eval --ckpt "$CKPT" --data "$DATA" --limit "$LIMIT"
fi
```

- [ ] **Step 3: 跑基线评测（验证脚本通）**

Run: `bash finetuning/grpo_eval.sh "" 50`
Expected: 打印 recall/fp/cer_nh/cer 四行数值，无异常。

- [ ] **Step 4: Commit**

```bash
git add finetuning/grpo_eval.py finetuning/grpo_eval.sh
git commit -m "feat(grpo): eval script for hotword recall/fp/non-hotword cer"
```

---

### Task 9: 训练 shell wrapper + 文档 + 收尾

**Files:**
- Create: `finetuning/grpo_train.sh`
- Modify: `AGENTS.md`（记录 RL 子包与脚本）

- [ ] **Step 1: 写训练 shell**

```bash
# finetuning/grpo_train.sh
#!/usr/bin/env bash
set -euo pipefail
CKPT="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"
DATA="/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl"
OUT="/cfs/data/private/WangYaoChi/model/grpo_lora_out"
python -m finetuning.grpo_train \
  --ckpt "$CKPT" --data "$DATA" --output_dir "$OUT" \
  --group_size 8 --temperature 0.8 --lr 1e-5 --beta 0.04 \
  --max_steps 1000 --batch_size 2
```

- [ ] **Step 2: 更新 AGENTS.md**

在 AGENTS.md 记录 RL 子包：`finetuning/grpo_*.py`（训练/评测/数学/rollout/LoRA）、`qwen_asr/tools/hotword_reward.py`、`tests/`，以及 RL 开关说明（LoRA 目录经 `grpo_eval.sh <lora_dir>` 挂载评测）。

- [ ] **Step 3: 全量单测**

Run: `python -m pytest tests/ -q`
Expected: 纯单测全 PASS（集成 skip）。

- [ ] **Step 4: Commit**

```bash
git add finetuning/grpo_train.sh AGENTS.md
git commit -m "feat(grpo): training shell wrapper and docs"
```

---

## 成功标准（spec 对齐）

- 热词召回相对基线提升。
- 误注入率下降。
- 非热词 CER 不劣化（≤ 基线 +0.5% 绝对）。
- 三者同时达成才算通过（`grpo_eval.sh` 基线 vs RL 后对比）。

## 风险与不确定点（实现时验证）

1. `Qwen3ASRThinkerForConditionalGeneration.forward` 的前向签名（Task 6 `_logp_of`）——以仓库 `modeling_qwen3_asr.py` 源码为准，smoke 必须验证。
2. `joint._asr_wrapper` 初始化路径（Task 7）——以 `model.py` 实际为准。
3. peft LoRA `target_modules` 是否误挂 audio_tower（Task 5 `assert_only_text_decoder_trainable` 兜底）。
4. verbatim 子串匹配对中文同形/标点边界噪声——去标点归一缓解，可接受。

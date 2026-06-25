# finetuning/grpo_lora.py
"""挂 LoRA 到 Qwen3-ASR thinker 文本解码器，冻结其余。"""
from typing import Optional

# thinker 文本解码器的注意力与 MLP 投影。audio_tower 的注意力层同名(q_proj 等)，
# 故用正则限定到 thinker.model.layers 下，避免误挂音频侧。
TEXT_DECODER_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
# peft 用 re.fullmatch 匹配整条模块路径。前缀可选：训练时包 joint（有 base_model.model.qwen_model. 前缀），
# 评测时包 qwen_model（无前缀，键为 thinker.model...），两种都需命中；audio_tower 同名层靠 `.model.layers` 排除。
TEXT_DECODER_TARGET_REGEX = (
    r"(?:.*\.)?thinker\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))"
)


def apply_lora(
    joint,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=TEXT_DECODER_TARGET_REGEX,
        bias="none",
        task_type=None,  # Qwen3-ASR 自带 generate，不走 peft 的 CausalLM 生成路径
    )
    # 先冻结全部，peft 再解冻 LoRA
    for p in joint.parameters():
        p.requires_grad_(False)
    peft_model = get_peft_model(joint, cfg)
    assert_only_text_decoder_trainable(peft_model)
    return peft_model


def assert_only_text_decoder_trainable(peft_model) -> None:
    """可训参数必须全在 thinker.model 下（文本解码器），不得误挂 audio_tower / heads。"""
    bad = []
    for name, p in peft_model.named_parameters():
        if p.requires_grad and "thinker.model" not in name:
            bad.append(name)
    if bad:
        raise RuntimeError(
            f"LoRA 误挂到非文本解码器: {bad[:5]}（共 {len(bad)} 个）"
        )

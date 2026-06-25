import pytest

CKPT = "/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"


@pytest.mark.integration
def test_apply_lora_only_text_decoder_trainable():
    import torch

    from qwen_asr.joint import Qwen3ASRJointModel
    from qwen_asr.joint.defaults import DEFAULT_ATTN_IMPLEMENTATION
    from finetuning.grpo_lora import apply_lora, assert_only_text_decoder_trainable

    joint = Qwen3ASRJointModel.from_pretrained(
        CKPT,
        dtype=torch.bfloat16,
        device_map=None,
        load_heads=False,
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    ).to("cuda")
    peft_model = apply_lora(joint)
    assert_only_text_decoder_trainable(peft_model)
    n = sum(1 for p in peft_model.parameters() if p.requires_grad)
    assert n > 0

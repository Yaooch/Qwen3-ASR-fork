"""Joint LLM 输入构造的共享 helper。"""


def audio_prompts(asr_wrapper, audio_token, audio_lengths, contexts, languages):
    """按音频 token 长度构造 ASR prompt。"""
    prompts = []
    for length, context, language in zip(audio_lengths, contexts, languages):
        text = asr_wrapper._build_text_prompt(
            context=context or "", force_language=language
        )
        prompts.append(text.replace(audio_token, audio_token * int(length), 1))
    return prompts


def left_tokenize(tokenizer, texts, **kwargs):
    """临时使用 left padding，不改变 tokenizer 的全局状态。"""
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        return tokenizer(texts, **kwargs)
    finally:
        tokenizer.padding_side = old_side


def inject_audio(thinker, input_ids, audio_features):
    """把音频特征写入 input_ids 对应的 audio placeholder。"""
    embeds = thinker.get_input_embeddings()(input_ids)
    audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=embeds)
    prompt_count = int(audio_mask[..., 0].sum().item())
    feature_count = audio_features.numel() // audio_features.shape[-1]
    if prompt_count != feature_count:
        raise RuntimeError(
            f"audio placeholder mismatch: prompt={prompt_count}, features={feature_count}"
        )
    return embeds.masked_scatter(
        audio_mask, audio_features.to(device=embeds.device, dtype=embeds.dtype)
    )

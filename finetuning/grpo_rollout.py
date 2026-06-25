# finetuning/grpo_rollout.py
"""GRPO rollout：音频 embedding 缓存 + G 路采样 + token logprob。

与 qwen_asr/joint/model.py 的 decode_llm 路径对齐：
- 音频 embedding 由 encode_offline 产出（thinker.audio_tower，冻结）
- prompt 由 _asr_wrapper._build_text_prompt 构造，audio placeholder 用 processor.audio_token
- 采样走 qwen_model.generate（do_sample），logprob 走 thinker.forward 的 logits
"""
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List

import librosa
import torch
import torch.nn.functional as F

from qwen_asr.inference.utils import parse_asr_output
from qwen_asr.joint.encoder import encode_offline, feature_lens


@dataclass
class RolloutResult:
    ids: torch.LongTensor   # (T_gen,) 生成 token
    text: str
    logp_ref: torch.Tensor  # (T_gen,) base(LoRA-off) logprob，detach


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
        """返回该音频的 LLM 输入 embedding (n_audio_tokens, hidden)，缓存。"""
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        wav = wav.astype("float32")
        fe = self.processor.feature_extractor
        batch = fe(
            [wav],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        feats = batch["input_features"]
        mask = batch.get("feature_attention_mask", batch.get("attention_mask"))
        ref = next(self.joint.parameters())
        feats = feats.to(device=ref.device, dtype=ref.dtype)
        if mask is not None:
            mask = mask.to(device=ref.device)
        lens = feature_lens(feats, mask)
        tower = self.joint.qwen_model.thinker.audio_tower
        _, llm, _ = encode_offline(tower, feats, lens, need_llm=True)
        # 单条音频：llm 即 (n_audio, hidden) 的 LLM 音频 embedding
        emb = llm[: int(lens[0])]
        self._audio_cache[audio_path] = emb
        return emb

    def build_inputs(self, sample):
        """对齐 decode_llm：拼 prompt + audio placeholder，返回 input_ids 与 audio embedding。"""
        thinker = self.joint.qwen_model.thinker
        processor = self.processor
        token = processor.audio_token
        context = sample.prompt or ""
        text = self.asr_wrapper._build_text_prompt(context=context, force_language=None)
        audio_embeds = self.audio_embedding(sample.audio)
        text = text.replace(token, token * int(audio_embeds.shape[0]), 1)
        old = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            tok = processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        finally:
            processor.tokenizer.padding_side = old
        input_ids = tok["input_ids"].to(self.device)
        attn = tok["attention_mask"].to(self.device)
        return input_ids, attn, audio_embeds

    @torch.no_grad()
    def _generate_one(self, input_ids, attn, audio_embeds):
        thinker = self.joint.qwen_model.thinker
        embeds = thinker.get_input_embeddings()(input_ids)
        audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=embeds)
        inputs_embeds = embeds.masked_scatter(audio_mask, audio_embeds.to(dtype=embeds.dtype))
        gen = self.joint.qwen_model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
        )
        seq = gen.sequences
        gen_ids = seq[0, input_ids.shape[1]:]
        raw = self.processor.batch_decode(seq[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        _, text = parse_asr_output(raw)
        return gen_ids, text

    def _logp_of(self, input_ids, audio_embeds, gen_ids, use_lora: bool) -> torch.Tensor:
        """前向算 gen_ids 的 token logprob。use_lora=False 时关 LoRA（ref，detach）。"""
        thinker = self.joint.qwen_model.thinker
        gen_ids = gen_ids.to(self.device)
        full_ids = torch.cat([input_ids, gen_ids.unsqueeze(0)], dim=1)
        full_attn = torch.ones_like(full_ids)
        inputs_embeds = thinker.get_input_embeddings()(full_ids)
        audio_mask = thinker.get_placeholder_mask(full_ids, inputs_embeds=inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_mask, audio_embeds.to(dtype=inputs_embeds.dtype)
        )
        ctx = (
            self.joint.disable_adapter()
            if (not use_lora and hasattr(self.joint, "disable_adapter"))
            else nullcontext()
        )
        with ctx:
            out = thinker(
                input_ids=full_ids,
                attention_mask=full_attn,
                inputs_embeds=inputs_embeds,
            )
        logits = out.logits  # (1, T, V)
        prompt_len = input_ids.shape[1]
        # 位置 i 的 logits 预测 token i+1；生成段预测位置 [prompt_len-1 : prompt_len-1+G]
        log_logits = F.log_softmax(logits[:, prompt_len - 1:-1, :], dim=-1)
        logp = log_logits.gather(-1, gen_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        return logp.squeeze(0)

    @torch.no_grad()
    def sample(self, sample) -> List[RolloutResult]:
        input_ids, attn, audio_embeds = self.build_inputs(sample)
        results = []
        for _ in range(self.G):
            gen_ids, text = self._generate_one(input_ids, attn, audio_embeds)
            ref_logp = self._logp_of(
                input_ids, audio_embeds, gen_ids, use_lora=False
            ).detach()
            results.append(RolloutResult(ids=gen_ids.detach(), text=text, logp_ref=ref_logp))
        return results

    def token_logp(self, sample, gen_ids) -> torch.Tensor:
        """训练时用 LoRA-on 重算 gen_ids 的 logp（带梯度）。"""
        input_ids, _, audio_embeds = self.build_inputs(sample)
        return self._logp_of(input_ids, audio_embeds, gen_ids, use_lora=True)

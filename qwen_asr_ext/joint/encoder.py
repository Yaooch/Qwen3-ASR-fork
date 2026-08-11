"""三种 Encoder 路径：离线、流式训练 chunk mask、真实流式增量。"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .defaults import STREAM_CNN_LEFT_FRAMES


def feature_lens(feats, mask=None):
    if mask is not None:
        return mask.sum(dim=1).long()
    return torch.full((feats.shape[0],), feats.shape[2], dtype=torch.long, device=feats.device)


def out_lens(feat_lens: torch.Tensor) -> torch.Tensor:
    """Mel 帧数 → encoder 输出帧数（仅 encode_offline 用）。"""
    leave = feat_lens % 100
    x = (leave - 1) // 2 + 1
    return ((x - 1) // 2 + 1 - 1) // 2 + 1 + (feat_lens // 100) * 13


def conv_len(lengths):
    """3 层 stride-2 CNN 后的帧数（encode_train_mask / encode_stream 用）。"""
    for _ in range(3):
        lengths = (lengths + 1) // 2
    return lengths


def conv_subsample_batch(tower, input_features: torch.Tensor) -> torch.Tensor:
    """CNN 前端：conv2d×3 + GELU → conv_out 展平。"""
    x = input_features.unsqueeze(1)
    pieces = []
    for chunk in x.split(tower.conv_chunksize, dim=0):
        chunk = F.gelu(tower.conv2d1(chunk))
        chunk = F.gelu(tower.conv2d2(chunk))
        chunk = F.gelu(tower.conv2d3(chunk))
        pieces.append(chunk)
    x = torch.cat(pieces, dim=0)
    b, c, f, t = x.size()
    return tower.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))


# ===== Encoder 路径 =====


def encode_offline(tower, feats: torch.Tensor, feat_lens: torch.Tensor, need_llm: bool):
    """离线整段编码：拼接有效帧 → tower() → aux 特征 + 可选 LLM 投影。"""
    bsz = feats.shape[0]
    enc, aux = tower(
        input_features=torch.cat([feats[i, :, :feat_lens[i]] for i in range(bsz)], dim=1),
        feature_lens=feat_lens,
        return_pre_proj=True,
    )
    lens = out_lens(feat_lens)
    hs_pad = aux.new_zeros(bsz, int(lens.max().item()), tower.config.d_model)
    offset = 0
    for i, length in enumerate(lens.tolist()):
        length = int(length)
        hs_pad[i, :length] = aux[offset:offset + length]
        offset += length
    llm = tower.proj2(tower.act(tower.proj1(enc.last_hidden_state))) if need_llm else None
    return hs_pad, llm, lens


def encode_train_mask(
    tower,
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
    left_frames: int,
    current_frames: int,
    right_frames: int,
    need_llm: bool,
):
    """流式训练用的 chunk mask 编码：整段 Mel → aux 特征 + 可选 LLM 投影。"""
    hidden_states = conv_subsample_batch(tower, feats)
    t = hidden_states.shape[1]
    pos_emb = tower.positional_embedding.positional_embedding[:t, :].unsqueeze(0)
    hidden_states = hidden_states + pos_emb.to(hidden_states.device, dtype=hidden_states.dtype)

    # 构建 chunk 级 attention mask
    lens = conv_len(feat_lens)
    current_frames = max(1, int(current_frames))
    left_frames = max(0, int(left_frames))
    right_frames = max(0, int(right_frames))
    pos = torch.arange(t, device=hidden_states.device)
    q_chunk = pos // current_frames
    k_pos = pos.unsqueeze(0)
    block_start = q_chunk * current_frames
    start = (block_start - left_frames).clamp_min(0).unsqueeze(1)
    end = (block_start + current_frames + right_frames).unsqueeze(1)
    chunk_mask = (k_pos >= start) & (k_pos < end)
    valid = pos.unsqueeze(0) < lens.unsqueeze(1)
    attention_mask = chunk_mask.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)

    for layer in tower.layers:
        # self-attention with mask
        residual = hidden_states
        normed = layer.self_attn_layer_norm(hidden_states)
        batch_size, seq_length, _ = normed.size()
        attn = layer.self_attn
        q = attn.q_proj(normed).reshape(batch_size, seq_length, attn.num_heads, -1).transpose(1, 2)
        k = attn.k_proj(normed).reshape(batch_size, seq_length, attn.num_heads, -1).transpose(1, 2)
        v = attn.v_proj(normed).reshape(batch_size, seq_length, attn.num_heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask.unsqueeze(1),
            dropout_p=0.0 if not attn.training else attn.attention_dropout,
            scale=attn.scaling,
        )
        hidden_states = residual + attn.out_proj(
            out.transpose(1, 2).reshape(batch_size, seq_length, -1).contiguous()
        )

        # FFN
        residual = hidden_states
        hidden_states = layer.final_layer_norm(hidden_states)
        hidden_states = layer.fc1(hidden_states)
        hidden_states = layer.activation_fn(hidden_states)
        hidden_states = layer.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    hs = tower.ln_post(hidden_states)
    if not need_llm:
        return hs, None, lens
    x = torch.cat([hs[i, :int(length)] for i, length in enumerate(lens.tolist())], dim=0)
    llm = tower.proj2(tower.act(tower.proj1(x)))
    return hs, llm, lens


def encode_stream(tower, feature_extractor, wavs, ref, need_llm: bool):
    """真实流式编码：640ms 增量 Mel → CNN overlap → KV cache 批量推进。"""
    sr = int(getattr(feature_extractor, "sampling_rate", 16000) or 16000)
    # 帧移点数（160点 10ms）
    hop = int(getattr(feature_extractor, "hop_length", 160) or 160)
    # 做fft的窗长点数（400点 25ms）
    n_fft = int(getattr(feature_extractor, "n_fft", 400) or 400)
    # 尾巴的最长长度
    tail_limit = ((((n_fft + 1) // 2) + hop - 1) // hop) * hop
    # 一个chunk的点数, 现在硬编码成了640ms的窗长, 后面可以更改
    chunk_samples = max(1, int(round(0.64 * sr)))
    # 这里表示cache7个chunk的encoder帧数, 也是硬编码了.
    cache_size = 7 * conv_len(max(1, int(round(0.64 * 16000 / hop))))
    # 一个bathc中的音频每条音频一个状态, 构成一个列表
    states = [
        dict(wav=np.asarray(w, dtype=np.float32), pos=0, raw_tail=None, mel_tail=None, cache=None, offset=0, chunks=[])
        for w in wavs
    ]

    #只要这个batch中有音频还没结束就继续
    while any(s["pos"] < len(s["wav"]) for s in states):
        pending = []
        for i, state in enumerate(states):
            if state["pos"] >= len(state["wav"]):
                continue
            # 当前音频的结束位置（点数）
            end = min(len(state["wav"]), state["pos"] + chunk_samples)
            # 当前音频
            cur = state["wav"][state["pos"]:end]
            # 当前音频加上上个chunk的尾巴
            seg = cur if state["raw_tail"] is None else np.concatenate([state["raw_tail"], cur])
            # 上个chunk的尾巴对应的帧数
            left = 0 if state["raw_tail"] is None else int(state["raw_tail"].shape[0]) // hop
            # 更新尾巴的状态为当前chunk的尾巴
            state["raw_tail"] = seg[-min(len(seg), tail_limit):].copy() if tail_limit > 0 else None
            # 更新起始位置的状态为当前chunk的结束位置
            state["pos"] = end
            # 记录一个batch中的每条音频拆分的chunk
            pending.append((i, seg, left))
        # 对当前batch中的每个chunk提取特征mel谱, batch["input_features"]返回一个三维张量[B,D,T],B表示整个batch中的chunk数,D表示mel维度,T表示时间帧数（padding到最大）
        batch = feature_extractor(
            [x[1] for x in pending],
            sampling_rate=sr,
            return_tensors="pt", padding=True, truncation=False, return_attention_mask=True,
        )
        # 这个mask标记mel帧中哪些是真实帧, 哪些是padding
        mask = batch.get("feature_attention_mask", batch.get("attention_mask"))
        rows, max_len = [], 0
        for row, (i, _seg, left) in enumerate(pending):
            # 计算每个chunk的有效帧数
            valid = int(mask[row].sum().item()) if mask is not None else batch["input_features"].shape[-1]
            # 提取当前chunk对应的真实mel谱, 前left帧是上个chunk的tail. valid-left是当前帧对应的帧长
            mel = batch["input_features"][row, :, left:valid]
            if mel.shape[1] == 0:
                continue
            state = states[i]
            mel = mel.to(device=ref.device, dtype=ref.dtype)
            # 拼接上个chunk的mel帧的尾巴, 防止cnn下采样出现pad
            cnn_in = mel if state["mel_tail"] is None else torch.cat([state["mel_tail"], mel], dim=1)
            # cnn之后需要丢掉的左侧pad的encoder帧长
            drop = 0 if state["mel_tail"] is None else conv_len(STREAM_CNN_LEFT_FRAMES)
            # 更新mel尾巴
            state["mel_tail"] = cnn_in[:, -STREAM_CNN_LEFT_FRAMES:].detach() if STREAM_CNN_LEFT_FRAMES > 0 else None
            rows.append((i, cnn_in, drop))
            # 记录当前轮batch内chunk的最长mel帧长
            max_len = max(max_len, int(cnn_in.shape[1]))
        if not rows:
            continue

        # 合批送 encoder
        # 继承自rows[0][1]的device和dtype. batch的形状[B,D,T], 此时batch是全零张量
        batch = rows[0][1].new_zeros((len(rows), rows[0][1].shape[0], max_len))
        feat_lens, drops, caches, offsets = [], [], [], []
        # 把这些量真正赋值
        for row, (i, cnn_in, drop) in enumerate(rows):
            batch[row, :, :cnn_in.shape[1]] = cnn_in
            feat_lens.append(int(cnn_in.shape[1]))
            drops.append(drop)
            caches.append(states[i]["cache"])
            offsets.append(states[i]["offset"])
        # 送入encoder. 输出分别是lnpost之后的encoder输出和更新后的kv cache
        chunks, caches = _stream_encode_chunks(
            tower, batch,
            torch.tensor(feat_lens, dtype=torch.long, device=ref.device),
            torch.tensor(drops, dtype=torch.long, device=ref.device),
            kv_caches=caches, cache_size=cache_size, detach_cache=True,
            position_offsets=torch.tensor(offsets, dtype=torch.long, device=ref.device),
        )
        for (i, _cnn, _drop), chunk, cache in zip(rows, chunks, caches):
            states[i]["cache"] = cache
            if chunk.numel() > 0:
                states[i]["chunks"].append(chunk)
                states[i]["offset"] += int(chunk.shape[0])

    chunk_lists, llm = [], []
    for i, state in enumerate(states):
        if not state["chunks"]:
            raise RuntimeError(f"No streaming auxiliary features were produced for item {i}.")
        chunk_lists.append(state["chunks"])
        if need_llm:
            seq = torch.cat(state["chunks"], dim=0).to(device=ref.device, dtype=ref.dtype)
            llm.append(tower.proj2(tower.act(tower.proj1(seq))))
    return chunk_lists, llm


# ===== 流式 chunk 编码内核 =====


def _stream_encode_chunks(
    tower,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    drop_prefixes: torch.Tensor,
    kv_caches: Optional[list] = None,
    cache_size: int = 0,
    detach_cache: bool = True,
    position_offsets: Optional[torch.Tensor] = None,
):
    """不等长 chunk 合批 → CNN → position encoding → KV cache attention → ln_post。"""
    batch_size = input_features.shape[0]
    if batch_size == 0:
        return [], []
    if kv_caches is None:
        kv_caches = [[None] * len(tower.layers) for _ in range(batch_size)]
    else:
        kv_caches = [cache if cache is not None else [None] * len(tower.layers) for cache in kv_caches]

    lens_after_cnn = conv_len(feature_lens)
    padded_embed = conv_subsample_batch(tower, input_features)
    if position_offsets is None:
        position_offsets = torch.zeros(batch_size, dtype=torch.long, device=input_features.device)

    hidden_states_list = []
    for idx, length in enumerate(lens_after_cnn.tolist()):
        cur_len = int(length)
        start = max(0, min(int(drop_prefixes[idx].item()), cur_len))
        hidden_states = padded_embed[idx, start:cur_len]
        pos_start = max(0, int(position_offsets[idx].item()))
        pos_end = pos_start + hidden_states.shape[0]
        pos_emb = tower.positional_embedding.positional_embedding[pos_start:pos_end, :]
        hidden_states = hidden_states + pos_emb.to(hidden_states.device, dtype=hidden_states.dtype)
        hidden_states_list.append(hidden_states)

    new_all_caches = [[None] * len(tower.layers) for _ in range(batch_size)]
    for layer_idx, layer in enumerate(tower.layers):
        layer_caches = [cache[layer_idx] for cache in kv_caches]

        # ---- KV cache attention ----
        residuals = hidden_states_list
        normed = [layer.self_attn_layer_norm(x) for x in hidden_states_list]
        q_lens = [int(x.shape[0]) for x in normed]
        max_q = max(q_lens)
        if max_q > 0:
            attn = layer.self_attn
            cur = torch.cat(normed, dim=0)
            query_all = attn.q_proj(cur).reshape(cur.shape[0], attn.num_heads, -1)
            key_cur_all = attn.k_proj(cur).reshape(cur.shape[0], attn.num_heads, -1)
            value_cur_all = attn.v_proj(cur).reshape(cur.shape[0], attn.num_heads, -1)
            query_split = query_all.split(q_lens, dim=0)
            key_cur_split = key_cur_all.split(q_lens, dim=0)
            value_cur_split = value_cur_all.split(q_lens, dim=0)

            key_list, value_list, k_lens, new_caches = [], [], [], []
            for key_cur, value_cur, cache in zip(key_cur_split, value_cur_split, layer_caches):
                if cache is not None:
                    key_all = torch.cat([cache[0].to(key_cur.dtype), key_cur], dim=0)
                    value_all = torch.cat([cache[1].to(value_cur.dtype), value_cur], dim=0)
                else:
                    key_all, value_all = key_cur, value_cur
                key_list.append(key_all); value_list.append(value_all)
                k_lens.append(int(key_all.shape[0]))
                new_key = key_all[-cache_size:] if cache_size and cache_size > 0 else key_all[:0]
                new_value = value_all[-cache_size:] if cache_size and cache_size > 0 else value_all[:0]
                if detach_cache:
                    new_key, new_value = new_key.detach(), new_value.detach()
                new_caches.append((new_key, new_value))

            max_k = max(k_lens)
            q = query_all.new_zeros((batch_size, attn.num_heads, max_q, attn.head_dim))
            k = query_all.new_zeros((batch_size, attn.num_heads, max_k, attn.head_dim))
            v = query_all.new_zeros((batch_size, attn.num_heads, max_k, attn.head_dim))
            attn_mask = torch.zeros((batch_size, 1, 1, max_k), dtype=torch.bool, device=query_all.device)
            for i, (qry, key, val, q_len, k_len) in enumerate(zip(query_split, key_list, value_list, q_lens, k_lens)):
                q[i, :, :q_len] = qry.transpose(0, 1)
                k[i, :, :k_len] = key.transpose(0, 1)
                v[i, :, :k_len] = val.transpose(0, 1)
                attn_mask[i, :, :, :k_len] = True

            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=0.0 if not attn.training else attn.attention_dropout,
                scale=attn.scaling,
            )
            attn_outputs = []
            for i, q_len in enumerate(q_lens):
                cur_out = out[i, :, :q_len].transpose(0, 1).reshape(q_len, -1).contiguous()
                attn_outputs.append(attn.out_proj(cur_out))

            hidden_states_list = [r + a for r, a in zip(residuals, attn_outputs)]
        layer_caches = new_caches

        # ---- FFN ----
        for i in range(len(hidden_states_list)):
            x = hidden_states_list[i]
            residual = x
            x = layer.final_layer_norm(x)
            x = layer.fc1(x)
            x = layer.activation_fn(x)
            x = layer.fc2(x)
            x = residual + x
            if x.dtype == torch.float16:
                clamp_value = torch.finfo(x.dtype).max - 1000
                x = torch.clamp(x, min=-clamp_value, max=clamp_value)
            hidden_states_list[i] = x

        for item_idx, layer_cache in enumerate(layer_caches):
            new_all_caches[item_idx][layer_idx] = layer_cache

    return [tower.ln_post(x) for x in hidden_states_list], new_all_caches

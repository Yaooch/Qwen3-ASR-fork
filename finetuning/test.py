import sentencepiece as spm

# ========== 配置路径（改成你的实际路径）==========
SP_MODEL_PATH = "/nfsdir/hubk/sensevoice_training/wenet/examples/voyah/s0/data/dict/train_960_unigram5000.model"
VOCAB_PATH = "/nfsdir/hubk/sensevoice_training/wenet/examples/voyah/s0/data/dict/lang_char_large_yue.txt"

# ========== 1. 加载 SP 模型 ==========
print("=" * 50)
print("1. 测试 SP 模型输出")
print("=" * 50)

sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

# 测试样本
test_samples = [
    "属蛇",           # 中文
    "枢",             # 单字中文
    "ZERO",           # 英文
    "Hello World",    # 英文句子
    "TAKE",           # 可能是词中片段
]

for text in test_samples:
    pieces = sp.encode_as_pieces(text.upper())  # 转大写，因为 vocab 是大写
    print(f"输入: '{text}' -> SP输出: {pieces}")

# ========== 2. 加载 Vocab 并检查格式 ==========
print("\n" + "=" * 50)
print("2. 检查 Vocab 文件格式")
print("=" * 50)

vocab = {}
with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                # 格式: id token
                idx = int(parts[0])
                token = parts[1]
            except ValueError:
                # 格式: token id
                token = parts[0]
                idx = int(parts[1])
            vocab[token] = idx
        
        # 只打印前 20 行看看格式
        if i < 20:
            print(f"行{i}: {line.strip()}")

print(f"\nVocab 总大小: {len(vocab)}")

# ========== 3. 关键验证：SP 输出 vs Vocab ==========
print("\n" + "=" * 50)
print("3. 匹配验证（核心）")
print("=" * 50)

for text in test_samples:
    pieces = sp.encode_as_pieces(text.upper())
    print(f"\n输入: '{text}'")
    
    for piece in pieces:
        # 检查三种情况
        in_vocab_direct = piece in vocab
        clean_piece = piece.replace("▁", "")
        in_vocab_clean = clean_piece in vocab
        with_underscore = "▁" + piece if not piece.startswith("▁") else piece
        in_vocab_with = with_underscore in vocab
        
        status = []
        if in_vocab_direct:
            status.append(f"✅ 直接匹配 (ID:{vocab[piece]})")
        if in_vocab_clean:
            status.append(f"✅ 去▁后匹配 (ID:{vocab[clean_piece]})")
        if in_vocab_with:
            status.append(f"✅ 加▁后匹配 (ID:{vocab[with_underscore]})")
        if not status:
            status.append("❌ 未找到 -> 将变成 <unk>")
        
        print(f"  SP输出: '{piece}' -> {' | '.join(status)}")

# ========== 4. 统计：有多少 SP 输出会失败 ==========
print("\n" + "=" * 50)
print("4. 抽样统计（随机检查 100 个 vocab 中的 token）")
print("=" * 50)

import random
vocab_tokens = list(vocab.keys())
sample_tokens = random.sample(vocab_tokens, min(100, len(vocab_tokens)))

missing_count = 0
for token in sample_tokens:
    # 模拟 SP 会怎么输出这个 token
    # 注意：这里只是简单测试，实际要看 SP 对包含该 token 的句子的输出
    if token.startswith("▁"):
        # 如果 vocab 里有 ▁token，检查 SP 是否输出 ▁token
        pass  # 复杂场景，略过
    else:
        # vocab 里没 ▁，但 SP 可能输出 ▁token
        piece_with = "▁" + token
        if piece_with not in vocab and token not in vocab:
            missing_count += 1

print(f"随机检查中，有 {missing_count}/100 个 token 可能需要 ▁ 转换")


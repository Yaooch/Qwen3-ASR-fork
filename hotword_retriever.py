#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热词检索模块

基于CTC粗识别结果(拼音序列)，使用编辑距离从热词库中检索Top-K热词

使用流程:
1. CTC输出拼音序列 (如: ['ni', 'hao', 'a', 'li', 'ya'])
2. 将热词库转换为拼音 (使用 pypinyin)
3. 计算编辑距离，找出Top-K匹配
4. 将匹配的热词作为prompt送入Qwen3-ASR
"""

import json
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class HotwordMatch:
    """热词匹配结果"""
    hotword: str          # 原始热词文本
    pinyin: str          # 热词拼音
    distance: int        # 编辑距离
    score: float         # 相似度分数 (0-1)


class HotwordRetriever:
    """
    热词检索器
    
    支持两种匹配模式:
    1. 拼音级编辑距离 (推荐，容错性好)
    2. 字符级编辑距离 (精确匹配)
    
    支持两种输入:
    1. CTC输出的汉字ID序列 (字符词表)
    2. CTC输出的拼音ID序列 (拼音词表)
    """
    
    def __init__(
        self,
        hotword_list: List[str],
        vocab: Dict[str, int],
        match_mode: str = "pinyin",
        max_distance: int = 3,
        input_mode: str = "char",  # "char" 或 "pinyin"
    ):
        """
        初始化热词检索器
        
        Args:
            hotword_list: 热词列表
            vocab: CTC词表 (token -> id)
            match_mode: 匹配模式 ("pinyin" 或 "char")
            max_distance: 最大编辑距离阈值
            input_mode: 输入模式 ("char"=汉字, "pinyin"=拼音)
        """
        self.hotword_list = hotword_list
        self.vocab = vocab
        self.match_mode = match_mode
        self.max_distance = max_distance
        self.input_mode = input_mode
        self.id_to_token = {v: k for k, v in vocab.items()}
        
        # 预计算热词的拼音
        self.hotword_pinyins = self._precompute_pinyins(hotword_list)
        
    def _precompute_pinyins(self, hotword_list: List[str]) -> List[str]:
        """预计算热词的拼音表示"""
        try:
            from pypinyin import pinyin, Style
        except ImportError:
            raise ImportError("使用拼音匹配需要安装 pypinyin: pip install pypinyin")
        
        pinyins = []
        for word in hotword_list:
            # 转换为拼音列表，然后连接成字符串
            py_list = pinyin(word, style=Style.NORMAL, strict=False)
            py_str = " ".join([item[0] for item in py_list])
            pinyins.append(py_str)
        
        return pinyins
    
    def ctc_ids_to_text(self, ctc_ids: List[int]) -> str:
        """
        将CTC输出的ID序列转换为文本字符串
        
        Args:
            ctc_ids: CTC解码后的ID序列
            
        Returns:
            文本字符串 (如 "阿里巴巴")
        """
        chars = []
        for idx in ctc_ids:
            token = self.id_to_token.get(idx, "")
            # 跳过特殊token
            if token and token not in ["<blank>", "<unk>", "<s>", "</s>", "<pad>"]:
                chars.append(token)
        
        return "".join(chars)
    
    def ctc_ids_to_pinyin(self, ctc_ids: List[int]) -> str:
        """
        将CTC输出的ID序列转换为拼音字符串
        
        Args:
            ctc_ids: CTC解码后的ID序列 (拼音词表)
            
        Returns:
            拼音字符串 (如 "ni hao a li ya")
        """
        pinyins = []
        for idx in ctc_ids:
            token = self.id_to_token.get(idx, "")
            # 跳过特殊token
            if token and token not in ["<blank>", "<unk>", "<s>", "</s>", "<pad>"]:
                pinyins.append(token)
        
        return " ".join(pinyins)
    
    def text_to_pinyin(self, text: str) -> str:
        """
        将汉字文本转换为拼音字符串
        
        Args:
            text: 汉字文本
            
        Returns:
            拼音字符串
        """
        try:
            from pypinyin import pinyin, Style
        except ImportError:
            raise ImportError("需要安装 pypinyin: pip install pypinyin")
        
        py_list = pinyin(text, style=Style.NORMAL, strict=False)
        return " ".join([item[0] for item in py_list])
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def retrieve(
        self,
        ctc_ids: List[int],
        top_k: int = 5,
    ) -> List[HotwordMatch]:
        """
        检索Top-K热词
        
        支持多种组合:
        - input_mode="char" + match_mode="pinyin": CTC输出汉字，拼音匹配热词
        - input_mode="char" + match_mode="char": CTC输出汉字，字符匹配热词
        - input_mode="pinyin" + match_mode="pinyin": CTC输出拼音，拼音匹配热词
        
        Args:
            ctc_ids: CTC解码后的ID序列
            top_k: 返回前K个匹配结果
            
        Returns:
            匹配结果列表，按分数降序排列
        """
        matches = []
        
        # 情况1: 输入是汉字，用拼音匹配 (推荐)
        if self.input_mode == "char" and self.match_mode == "pinyin":
            # CTC输出转汉字
            ctc_text = self.ctc_ids_to_text(ctc_ids)
            if not ctc_text:
                return []
            # 汉字转拼音
            ctc_pinyin = self.text_to_pinyin(ctc_text)
            
            for hotword, hw_pinyin in zip(self.hotword_list, self.hotword_pinyins):
                distance = self._levenshtein_distance(ctc_pinyin, hw_pinyin)
                if distance > self.max_distance:
                    continue
                max_len = max(len(ctc_pinyin), len(hw_pinyin))
                score = 1.0 - distance / max_len if max_len > 0 else 0.0
                matches.append(HotwordMatch(hotword, hw_pinyin, distance, score))
        
        # 情况2: 输入是汉字，用字符匹配
        elif self.input_mode == "char" and self.match_mode == "char":
            ctc_text = self.ctc_ids_to_text(ctc_ids)
            if not ctc_text:
                return []
            
            for hotword in self.hotword_list:
                distance = self._levenshtein_distance(ctc_text, hotword)
                if distance > self.max_distance:
                    continue
                max_len = max(len(ctc_text), len(hotword))
                score = 1.0 - distance / max_len if max_len > 0 else 0.0
                # 获取热词拼音用于显示
                hw_pinyin = self.hotword_pinyins[self.hotword_list.index(hotword)]
                matches.append(HotwordMatch(hotword, hw_pinyin, distance, score))
        
        # 情况3: 输入是拼音，用拼音匹配
        elif self.input_mode == "pinyin" and self.match_mode == "pinyin":
            ctc_pinyin = self.ctc_ids_to_pinyin(ctc_ids)
            if not ctc_pinyin:
                return []
            
            for hotword, hw_pinyin in zip(self.hotword_list, self.hotword_pinyins):
                distance = self._levenshtein_distance(ctc_pinyin, hw_pinyin)
                if distance > self.max_distance:
                    continue
                max_len = max(len(ctc_pinyin), len(hw_pinyin))
                score = 1.0 - distance / max_len if max_len > 0 else 0.0
                matches.append(HotwordMatch(hotword, hw_pinyin, distance, score))
        
        # 按分数降序排列，然后按距离升序
        matches.sort(key=lambda x: (-x.score, x.distance))
        
        return matches[:top_k]
    
    def build_prompt(self, matches: List[HotwordMatch], max_tokens: int = 50) -> str:
        """
        构建Qwen3-ASR的prompt
        
        Args:
            matches: 匹配的热词列表
            max_tokens: 最大token数限制
            
        Returns:
            prompt字符串
        """
        if not matches:
            return ""
        
        # 选择高置信度的热词
        selected_words = []
        current_len = 0
        
        for match in matches:
            word_len = len(match.hotword)
            if current_len + word_len > max_tokens:
                break
            selected_words.append(match.hotword)
            current_len += word_len
        
        if not selected_words:
            return ""
        
        # 构建prompt
        prompt = "关注热词: " + ", ".join(selected_words)
        return prompt


class SlidingWindowRetriever(HotwordRetriever):
    """
    滑动窗口热词检索器
    
    对于长音频的CTC输出，使用滑动窗口进行局部匹配，
    避免整句匹配导致局部热词被忽略
    """
    
    def __init__(
        self,
        hotword_list: List[str],
        vocab: Dict[str, int],
        window_size: int = 20,  # 窗口大小 (拼音token数)
        stride: int = 10,       # 滑动步长
        **kwargs
    ):
        super().__init__(hotword_list, vocab, **kwargs)
        self.window_size = window_size
        self.stride = stride
    
    def retrieve(
        self,
        ctc_ids: List[int],
        top_k: int = 5,
    ) -> List[HotwordMatch]:
        """
        滑动窗口检索
        
        对长序列分段匹配，然后合并结果
        """
        if len(ctc_ids) <= self.window_size:
            return super().retrieve(ctc_ids, top_k)
        
        all_matches = []
        
        # 滑动窗口
        for start in range(0, len(ctc_ids), self.stride):
            end = start + self.window_size
            window_ids = ctc_ids[start:end]
            
            window_matches = super().retrieve(window_ids, top_k=top_k*2)
            all_matches.extend(window_matches)
        
        # 去重并按分数排序
        seen_words = set()
        unique_matches = []
        
        for match in sorted(all_matches, key=lambda x: -x.score):
            if match.hotword not in seen_words:
                unique_matches.append(match)
                seen_words.add(match.hotword)
        
        return unique_matches[:top_k]


def demo():
    """演示热词检索流程 (使用字符词表)"""
    
    # 示例热词库
    hotwords = [
        "阿里巴巴",
        "支付宝", 
        "淘宝",
        "天猫",
        "菜鸟驿站",
        "盒马鲜生",
        "飞猪旅行",
        "饿了么",
        "高德地图",
        "钉钉",
    ]
    
    # 加载词表 (使用ctc_vocab.json汉字词表)
    vocab_path = "ctc_vocab.json"
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
    except FileNotFoundError:
        print(f"词表文件不存在: {vocab_path}")
        print(f"请确保ctc_vocab.json在当前目录")
        return
    
    print(f"已加载词表，大小: {len(vocab)}")
    
    # ========== 方案1: CTC输出汉字，拼音匹配热词 (推荐) ==========
    print("\n" + "="*60)
    print("方案1: CTC输出汉字 → 拼音匹配热词")
    print("="*60)
    
    retriever = HotwordRetriever(
        hotword_list=hotwords,
        vocab=vocab,
        match_mode="pinyin",      # 拼音匹配
        input_mode="char",        # 输入是汉字
        max_distance=5,
    )
    
    # 模拟CTC输出: "阿里妈妈逊" (接近"阿里巴巴"，但CTC识别错误)
    sample_text = "阿里妈妈逊"
    sample_ids = [vocab.get(c, 1) for c in sample_text]  # 1是<unk>
    
    print(f"CTC粗识别结果: {sample_text}")
    print(f"CTC输出ID序列: {sample_ids}")
    
    # 检索
    matches = retriever.retrieve(sample_ids, top_k=3)
    
    print("\nTop-3 热词匹配结果:")
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match.hotword} (拼音: {match.pinyin})")
        print(f"   编辑距离: {match.distance}, 相似度: {match.score:.3f}")
    
    # 构建prompt
    prompt = retriever.build_prompt(matches)
    print(f"\n生成的Prompt: {prompt}")
    
    # ========== 方案2: CTC输出汉字，字符匹配热词 ==========
    print("\n" + "="*60)
    print("方案2: CTC输出汉字 → 字符匹配热词")
    print("="*60)
    
    retriever2 = HotwordRetriever(
        hotword_list=hotwords,
        vocab=vocab,
        match_mode="char",        # 字符匹配
        input_mode="char",        # 输入是汉字
        max_distance=2,
    )
    
    # 检索
    matches2 = retriever2.retrieve(sample_ids, top_k=3)
    
    print(f"CTC粗识别结果: {sample_text}")
    print("\nTop-3 热词匹配结果:")
    for i, match in enumerate(matches2, 1):
        print(f"{i}. {match.hotword} (拼音: {match.pinyin})")
        print(f"   编辑距离: {match.distance}, 相似度: {match.score:.3f}")
    
    print(f"\n说明:")
    print(f"  - 方案1(拼音匹配)容错性更好，可以匹配发音相似的热词")
    print(f"  - 方案2(字符匹配)更精确，但对CTC识别错误敏感")
    print(f"  - 推荐方案1，配合Qwen3-ASR可以有效纠正CTC的错误")


if __name__ == "__main__":
    demo()

import re
from pathlib import Path


class FeatureExtractor:
    """
    基于词典的特征提取器
    提取：情绪渲染词命中数、模糊信源词命中数
    """

    def __init__(self, emotion_path: str, vague_path: str):
        self.emotion_words = self._load(emotion_path)
        self.vague_words = self._load(vague_path)

    def _load(self, path: str) -> list:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"词典文件不存在：{path}")
        with open(p, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def emotion_score(self, text: str) -> int:
        count = 0
        for w in self.emotion_words:
            count += len(re.findall(re.escape(w), text))
        return count

    def vague_score(self, text: str) -> int:
        count = 0
        for w in self.vague_words:
            count += len(re.findall(re.escape(w), text))
        return count

    def emotion_hits(self, text: str) -> list:
        """返回命中的情绪渲染词列表（去重）"""
        return list({w for w in self.emotion_words if w in text})

    def vague_hits(self, text: str) -> list:
        """返回命中的模糊信源词列表（去重）"""
        return list({w for w in self.vague_words if w in text})

    def extract(self, text: str) -> list:
        """返回特征向量 [情绪得分, 信源得分]"""
        return [self.emotion_score(text), self.vague_score(text)]

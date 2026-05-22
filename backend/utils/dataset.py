import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer


class NewsDataset(Dataset):
    """
    新闻风险数据集
    读取 Excel 文件，支持多列格式：
        - title       新闻标题
        - text        新闻正文
        - emotion_level  情绪风险标签（-1/0/1/2 → 训练时统一 +1 偏移为 0/1/2/3）
        - clarity_level  信源清晰度标签（0=模糊, 1=清晰）
    """

    TOKENIZER_NAME = "bert-base-chinese"

    def __init__(self, path: str, feature_extractor, max_length: int = 512):
        df = pd.read_excel(path)

        # 填充缺失值并转字符串
        df["title"] = df["title"].fillna("").astype(str)
        df["text"] = df["text"].fillna("").astype(str)

        # 标题 + 正文拼接
        self.texts = (df["title"] + " " + df["text"]).tolist()

        # 标签（允许空缺，默认 0）
        self.emotion_labels = df["emotion_level"].fillna(0).astype(int).tolist()
        self.source_labels = df["clarity_level"].fillna(0).astype(int).tolist()

        # 手工特征
        self.features = [feature_extractor.extract(t) for t in self.texts]

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.TOKENIZER_NAME)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids":      token["input_ids"].squeeze(0),
            "attention_mask": token["attention_mask"].squeeze(0),
            "features":       torch.tensor(self.features[idx], dtype=torch.float),
            "emotion_label":  torch.tensor(self.emotion_labels[idx], dtype=torch.long),
            "source_label":   torch.tensor(self.source_labels[idx], dtype=torch.long),
        }

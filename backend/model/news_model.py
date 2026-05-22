import torch
import torch.nn as nn
from transformers import BertModel


class NewsRiskModel(nn.Module):
    """
    新闻风险识别模型（多任务分类）
    输入：BERT token + 手工特征（情绪词数、模糊信源数）
    输出：情绪风险等级（4类）+ 信源风险等级（2类）
    """

    def __init__(self, feature_dim=2):
        super().__init__()

        # 中文 BERT 编码器
        self.bert = BertModel.from_pretrained("bert-base-chinese")

        # 手工特征处理层
        self.feature_fc = nn.Linear(feature_dim, 32)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        # 文本 + 特征融合层
        self.fc = nn.Linear(768 + 32, 256)

        # 多任务分类头
        self.emotion_classifier = nn.Linear(256, 4)   # 情绪风险：0~3
        self.source_classifier = nn.Linear(256, 2)    # 信源风险：0~1

    def forward(self, input_ids, attention_mask, features):
        # BERT 编码，取 [CLS] 向量
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = bert_output.pooler_output           # [B, 768]

        # 手工特征编码
        f = self.gelu(self.feature_fc(features))  # [B, 32]
        f = self.dropout(f)

        # 拼接融合
        x = torch.cat([cls, f], dim=1)            # [B, 800]
        x = self.gelu(self.fc(x))                 # [B, 256]
        x = self.dropout(x)

        # 多任务输出
        emotion_logits = self.emotion_classifier(x)  # [B, 4]
        source_logits = self.source_classifier(x)    # [B, 2]

        return emotion_logits, source_logits

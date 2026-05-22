"""
train.py —— 模型训练入口
用法：
    python train.py
    python train.py --epochs 10 --batch_size 16 --lr 2e-5
"""

import argparse
import sys
import os
from pathlib import Path

# 将 backend/ 目录加入 sys.path，无论从哪里运行都能正确导入
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)   # 将工作目录切换到 backend/，使相对路径全部生效

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

from model.news_model import NewsRiskModel
from utils.dataset import NewsDataset
from utils.feature_extractor import FeatureExtractor

# ─── 超参数 ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="新闻风险识别模型训练")
parser.add_argument("--data",       default=str(BASE_DIR.parent / "data" / "news.xlsx"), help="训练数据路径")
parser.add_argument("--emotion_lex",default="lexicon/emotion_words.txt",help="情绪词典路径")
parser.add_argument("--vague_lex",  default="lexicon/vague_sources.txt",help="信源词典路径")
parser.add_argument("--epochs",     type=int,   default=5,              help="训练轮数")
parser.add_argument("--batch_size", type=int,   default=8,              help="批次大小")
parser.add_argument("--lr",         type=float, default=2e-5,           help="学习率")
parser.add_argument("--save",       default="news_model.pth",           help="模型保存路径")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备：{device}")

# ─── 数据加载 ─────────────────────────────────────────────────────────────────
feature_extractor = FeatureExtractor(args.emotion_lex, args.vague_lex)
dataset = NewsDataset(args.data, feature_extractor)
loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print(f"数据集大小：{len(dataset)} 条")

# ─── 模型 ─────────────────────────────────────────────────────────────────────
model     = NewsRiskModel().to(device)
optimizer = Adam(model.parameters(), lr=args.lr)
loss_fn   = nn.CrossEntropyLoss()

# ─── 训练循环 ─────────────────────────────────────────────────────────────────
for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        mask      = batch["attention_mask"].to(device)
        features  = batch["features"].to(device)

        # 标签偏移：emotion_level 值域 -1/0/1/2 → 0/1/2/3
        emotion_label = (batch["emotion_label"] + 1).to(device)
        source_label  = batch["source_label"].to(device)

        emotion_pred, source_pred = model(input_ids, mask, features)

        loss = loss_fn(emotion_pred, emotion_label) + loss_fn(source_pred, source_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}  Step {step+1}  Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(len(loader), 1)
    print(f"[Epoch {epoch+1}] 平均损失：{avg_loss:.4f}")

torch.save(model.state_dict(), args.save)
print(f"\n模型已保存至 {args.save}")

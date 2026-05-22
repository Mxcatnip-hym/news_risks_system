"""
predict.py —— 命令行预测入口
用法：
    python predict.py                        # 交互式输入 URL
    python predict.py --url https://...      # 直接指定 URL
    python predict.py --text "新闻正文..."   # 直接输入文本
"""

import argparse
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

import torch
from transformers import BertTokenizer

from model.news_model import NewsRiskModel
from utils.feature_extractor import FeatureExtractor
from utils.web_parser import get_news_text

# ─── 风险等级映射 ──────────────────────────────────────────────────────────────
EMOTION_LEVELS = {0: "无风险", 1: "低风险", 2: "中风险", 3: "高风险"}
SOURCE_LEVELS  = {0: "信源模糊（高风险）", 1: "信源清晰（低风险）"}

# ─── 参数 ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="新闻风险识别预测")
parser.add_argument("--model",      default="news_model.pth",           help="模型权重路径")
parser.add_argument("--emotion_lex",default="lexicon/emotion_words.txt",help="情绪词典路径")
parser.add_argument("--vague_lex",  default="lexicon/vague_sources.txt",help="信源词典路径")
parser.add_argument("--url",        default=None,                       help="新闻 URL")
parser.add_argument("--text",       default=None,                       help="直接输入文本")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 加载 ─────────────────────────────────────────────────────────────────────
model = NewsRiskModel()
model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device).eval()

tokenizer         = BertTokenizer.from_pretrained("bert-base-chinese")
feature_extractor = FeatureExtractor(args.emotion_lex, args.vague_lex)

# ─── 获取文本 ─────────────────────────────────────────────────────────────────
if args.text:
    text = args.text
elif args.url:
    print(f"正在抓取：{args.url}")
    text = get_news_text(args.url)
else:
    url  = input("请输入新闻 URL：").strip()
    text = get_news_text(url)

print(f"\n文本长度：{len(text)} 字")

# ─── 特征提取 ─────────────────────────────────────────────────────────────────
emotion_hits = feature_extractor.emotion_hits(text)
vague_hits   = feature_extractor.vague_hits(text)
features     = torch.tensor(feature_extractor.extract(text)).unsqueeze(0).float()

token = tokenizer(
    text,
    max_length=512,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

# ─── 推理 ─────────────────────────────────────────────────────────────────────
with torch.no_grad():
    emotion_logits, source_logits = model(
        token["input_ids"].to(device),
        token["attention_mask"].to(device),
        features.to(device)
    )

emotion_pred = torch.argmax(emotion_logits).item()
source_pred  = torch.argmax(source_logits).item()

# ─── 输出 ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("新闻风险识别结果")
print("=" * 50)
print(f"情绪过度渲染风险：{EMOTION_LEVELS[emotion_pred]}")
print(f"信息源清晰度：    {SOURCE_LEVELS[source_pred]}")
print(f"\n命中情绪词（{len(emotion_hits)} 个）：{', '.join(emotion_hits[:10]) or '无'}")
print(f"命中模糊信源（{len(vague_hits)} 个）：{', '.join(vague_hits[:5]) or '无'}")
print("=" * 50)

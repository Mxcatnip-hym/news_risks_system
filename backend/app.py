"""
app.py —— Flask API 服务
提供前端所需的 RESTful 接口

接口列表：
    POST /api/analyze/text    文本直接分析
    POST /api/analyze/url     URL 抓取并分析
    GET  /api/health          服务健康检查
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer

from model.news_model import NewsRiskModel
from utils.feature_extractor import FeatureExtractor
from utils.web_parser import get_news_text

# ─── 初始化 ───────────────────────────────────────────────────────────────────
app    = Flask(__name__)
CORS(app)   # 允许跨域，前端直接调用

device = "cuda" if torch.cuda.is_available() else "cpu"

EMOTION_LEVELS = ["无风险", "低风险", "中风险", "高风险"]
SOURCE_LEVELS  = ["信源模糊", "信源清晰"]

# ─── 全局模型（延迟加载，首次请求时初始化）────────────────────────────────────
_model             = None
_tokenizer         = None
_feature_extractor = None

MODEL_PATH   = os.environ.get("MODEL_PATH",   "news_model.pth")
EMOTION_LEX  = os.environ.get("EMOTION_LEX",  "lexicon/emotion_words.txt")
VAGUE_LEX    = os.environ.get("VAGUE_LEX",    "lexicon/vague_sources.txt")
USE_BERT     = os.environ.get("USE_BERT", "false").lower() == "true"


def get_model():
    global _model, _tokenizer, _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor(EMOTION_LEX, VAGUE_LEX)
    if USE_BERT and _model is None and os.path.exists(MODEL_PATH):
        _model = NewsRiskModel()
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        _model.to(device).eval()
        _tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    return _model, _tokenizer, _feature_extractor


def analyze(text: str) -> dict:
    """核心分析函数，返回结构化结果"""
    model, tokenizer, fe = get_model()

    # 词典特征
    emotion_hits = fe.emotion_hits(text)
    vague_hits   = fe.vague_hits(text)
    emotion_raw  = fe.emotion_score(text)
    vague_raw    = fe.vague_score(text)

    # 风险评分（词典层面，0~100）
    emotion_score = min(100, emotion_raw * 8)
    source_score  = min(100, vague_raw * 12)
    total_score   = min(100, round(emotion_score * 0.6 + source_score * 0.4))

    def level(score):
        if score <= 25: return "低"
        if score <= 55: return "中"
        return "高"

    result = {
        "emotion_score":  emotion_score,
        "source_score":   source_score,
        "total_score":    total_score,
        "emotion_level":  level(emotion_score),
        "source_level":   level(source_score),
        "total_level":    level(total_score),
        "emotion_hits":   emotion_hits,
        "vague_hits":     vague_hits,
        "text_preview":   text[:200],
    }

    # 若模型已加载，追加 BERT 预测
    if model is not None:
        token = tokenizer(
            text, max_length=512, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        features = torch.tensor(fe.extract(text)).unsqueeze(0).float()
        with torch.no_grad():
            e_logits, s_logits = model(
                token["input_ids"].to(device),
                token["attention_mask"].to(device),
                features.to(device)
            )
        result["bert_emotion"] = EMOTION_LEVELS[torch.argmax(e_logits).item()]
        result["bert_source"]  = SOURCE_LEVELS[torch.argmax(s_logits).item()]

    return result


# ─── 路由 ─────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})


@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text 不能为空"}), 400
    try:
        return jsonify(analyze(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze/url", methods=["POST"])
def analyze_url():
    data = request.get_json(force=True)
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url 不能为空"}), 400
    try:
        text = get_news_text(url)
        if not text:
            return jsonify({"error": "无法从该链接提取正文，请直接粘贴文本"}), 400
        result = analyze(text)
        result["source_url"] = url
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"抓取失败：{str(e)}"}), 500


if __name__ == "__main__":
    print(f"服务启动，设备：{device}")
    print(f"模型状态：{'已加载' if os.path.exists(MODEL_PATH) else '未找到权重文件（仅词典模式）'}")
    app.run(host="0.0.0.0", port=5000, debug=False)

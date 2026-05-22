# 新闻风险识别系统

基于 BERT + 词典融合的多任务中文新闻风险识别系统，可识别**情绪过度渲染**与**模糊信息源**两类信息风险，并提供可视化 Web 界面。

---

## 目录结构

```
news_risk_system/
├── backend/
│   ├── app.py                  # Flask API 服务（前后端接口）
│   ├── train.py                # 模型训练脚本
│   ├── predict.py              # 命令行预测脚本
│   ├── model/
│   │   └── news_model.py       # 多任务 BERT 分类模型
│   ├── utils/
│   │   ├── dataset.py          # 数据集加载
│   │   ├── feature_extractor.py# 词典特征提取
│   │   └── web_parser.py       # 网页正文抓取
│   └── lexicon/
│       ├── emotion_words.txt   # 情绪渲染词典
│       └── vague_sources.txt   # 模糊信源词典
├── frontend/
│   └── index.html              # 可视化 Web 界面
├── data/
│   └── news.xlsx               # 标注数据集
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型（可选）

```bash
cd backend
python train.py
# 可选参数：
# python train.py --epochs 10 --batch_size 16 --lr 2e-5
```

训练完成后会在 `backend/` 目录下生成 `news_model.pth`。

### 3. 启动 API 服务

```bash
cd backend
python app.py
```

服务默认运行在 `http://localhost:5000`。

若已训练好模型，可开启 BERT 预测：

```bash
USE_BERT=true python app.py
```

### 4. 打开前端界面

直接用浏览器打开 `frontend/index.html` 即可使用。

> 若后端未启动，前端会自动回退到纯词典匹配模式，无需任何配置。

---

## 系统功能

| 功能 | 说明 |
|------|------|
| 文本输入分析 | 粘贴新闻文本，即时检测风险 |
| URL 链接分析 | 自动抓取新闻正文并分析 |
| 情绪渲染检测 | 基于 170+ 词条的情绪词典匹配 |
| 模糊信源检测 | 基于 110+ 词条的模糊信源词典匹配 |
| 风险评分 | 情绪评分、信源评分、综合风险指数（0~100） |
| 风险等级 | 低 / 中 / 高 三级判定 |
| BERT 预测 | 加载模型后输出情绪风险等级（4类）+ 信源风险（2类） |
| AI 深度解读 | 调用大语言模型生成人工智能分析报告 |

---

## API 接口说明

### 健康检查

```
GET /api/health
```

### 文本分析

```
POST /api/analyze/text
Content-Type: application/json

{"text": "新闻正文内容"}
```

### URL 分析

```
POST /api/analyze/url
Content-Type: application/json

{"url": "https://example.com/news/..."}
```

**返回示例：**

```json
{
  "emotion_score": 56,
  "source_score": 36,
  "total_score": 48,
  "emotion_level": "中",
  "source_level": "中",
  "total_level": "中",
  "emotion_hits": ["震惊", "惊天内幕", "濒临崩溃"],
  "vague_hits": ["知情人士", "业内人士称"],
  "bert_emotion": "中风险",
  "bert_source": "信源模糊"
}
```

---

## 数据集格式

`data/news.xlsx` 需包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| title | 新闻标题 | 震惊！专家称… |
| text | 新闻正文 | 昨日，据消息人士… |
| emotion_level | 情绪风险标签 | -1 / 0 / 1 / 2 |
| clarity_level | 信源清晰度标签 | 0（模糊）/ 1（清晰）|

---

## 技术架构

```
输入文本
   │
   ├─ 词典匹配层（FeatureExtractor）
   │     ├─ 情绪词典匹配 → 命中词列表 + 计数
   │     └─ 信源词典匹配 → 命中词列表 + 计数
   │
   └─ BERT 编码层（bert-base-chinese）
         ├─ [CLS] 向量 (768维)
         ├─ 拼接词典特征 (32维)
         └─ 多任务分类头
               ├─ 情绪风险分类（4类）
               └─ 信源风险分类（2类）
```

---

## 命令行预测

```bash
cd backend

# 直接输入文本
python predict.py --text "震惊！知情人士透露重大内幕，据悉局势已濒临失控"

# 输入 URL
python predict.py --url https://example.com/news/

# 交互式输入
python predict.py
```

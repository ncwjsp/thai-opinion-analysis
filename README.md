# 🇹🇭 Thai Online Media Public Opinion Analysis System

> A graduation thesis project (西华大学 / Xihua University) — Computer Science & Technology

---

## 📖 Overview

This system collects public opinion data from Thai online media platforms (news websites and social media), processes it using Thai-specific NLP, and presents the results through an interactive web-based visualization dashboard.

Users can enter keywords to explore public discussions, sentiment trends, and popular topics in Thai online media — targeting researchers, students, government analysts, and organizations operating in Thailand's digital media environment.

---

## 🏗️ System Architecture

The system is organized into four layers:

```
┌─────────────────────────────────────────────────────┐
│              Data Collection Layer                   │
│         Scrapy (static sites) + Selenium (JS)        │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│              Processing Layer                        │
│   Text cleaning → PyThaiNLP tokenization (newmm)    │
│   Stopword removal → NER → MD5 + MinHash/LSH dedup  │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│           Sentiment Analysis Layer                   │
│  WangchanBERTa (primary) · XLM-RoBERTa (secondary)  │
│       Positive / Neutral / Negative + confidence     │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│            Visualization Layer                       │
│  Trend charts · Sentiment pie/bar · Word cloud       │
│  Geographic choropleth map (Thai province level)     │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer                  | Technology                                       |
| ---------------------- | ------------------------------------------------ |
| Web crawling (static)  | Scrapy 2.x                                       |
| Web crawling (dynamic) | Selenium + ChromeDriver                          |
| Thai NLP               | PyThaiNLP (newmm tokenizer, NER, stopwords)      |
| Deduplication          | MD5 exact + MinHash LSH near-duplicate           |
| Sentiment model        | XLM-RoBERTa (primary) / WangchanBERTa (optional) |
| ML framework           | HuggingFace Transformers                         |
| Backend API            | FastAPI + SQLAlchemy + SQLite                    |
| Frontend               | Vanilla JS + Chart.js                            |

## Setup
### *This project requires Python 3.10+

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python run.py
```

Open http://localhost:8000 in your browser.

## Data Sources

| Source      | Type                 | Crawler          |
| ----------- | -------------------- | ---------------- |
| Sanook News | Thai news portal     | Scrapy spider    |
| Khaosod     | Thai news portal     | Scrapy spider    |
| Pantip      | Thai community forum | Selenium crawler |

## Sentiment Models

### XLM-RoBERTa (default)

- Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- Multilingual — works on Thai out-of-the-box
- Labels: positive / neutral / negative

### WangchanBERTa (optional)

- Model: `airesearch/wangchanberta-base-att-spm-uncased`
- Thai-specific BERT, ~94.57% accuracy (Nokkaew et al. 2023)
- Select in the UI dropdown or set `SENTIMENT_MODEL=wangchanberta` in `.env`

## API Reference

| Method | Path                     | Description                       |
| ------ | ------------------------ | --------------------------------- |
| POST   | `/api/search`            | Crawl + analyze keyword           |
| GET    | `/api/history`           | List previously searched keywords |
| GET    | `/api/results/{keyword}` | Fetch stored results              |
| GET    | `/health`                | Liveness check                    |

### POST /api/search body

```json
{
  "keyword": "รัฐบาล",
  "sources": ["sanook", "khaosod", "pantip"],
  "max_items_per_source": 30,
  "model": "xlm-roberta"
}
```

## Project Structure

```
thai-opinion-analysis/
├── config.py                  — settings
├── database.py                — SQLAlchemy models
├── run.py                     — server entry point
├── crawler/
│   ├── base.py                — shared RawItem dataclass
│   ├── scrapy_runner.py       — async Scrapy integration
│   ├── pantip_crawler.py      — Selenium crawler
│   └── spiders/
│       ├── sanook_spider.py
│       └── khaosod_spider.py
├── preprocessing/
│   └── processor.py           — clean → tokenize → NER → dedup
├── sentiment/
│   └── analyzer.py            — HuggingFace pipeline wrapper
├── api/
│   ├── main.py                — FastAPI routes
│   └── schemas.py             — Pydantic request/response models
└── frontend/
    └── index.html             — single-page UI
```

# Thai Online Opinion Analysis System

Keyword-driven Thai language public opinion collection and sentiment analysis.

## Tech Stack

| Layer | Technology |
|---|---|
| Web crawling (static) | Scrapy 2.x |
| Web crawling (dynamic) | Selenium + ChromeDriver |
| Thai NLP | PyThaiNLP (newmm tokenizer, NER, stopwords) |
| Deduplication | MD5 exact + MinHash LSH near-duplicate |
| Sentiment model | XLM-RoBERTa (primary) / WangchanBERTa (optional) |
| ML framework | HuggingFace Transformers |
| Backend API | FastAPI + SQLAlchemy + SQLite |
| Frontend | Vanilla JS + Chart.js |

## Setup

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

| Source | Type | Crawler |
|---|---|---|
| Sanook News | Thai news portal | Scrapy spider |
| Khaosod | Thai news portal | Scrapy spider |
| Pantip | Thai community forum | Selenium crawler |

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

| Method | Path | Description |
|---|---|---|
| POST | `/api/search` | Crawl + analyze keyword |
| GET | `/api/history` | List previously searched keywords |
| GET | `/api/results/{keyword}` | Fetch stored results |
| GET | `/health` | Liveness check |

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

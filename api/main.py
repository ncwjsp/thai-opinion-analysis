"""
FastAPI application — Thai Opinion Analysis System

Endpoints:
  POST /api/search        — crawl + analyze; returns full result set
  GET  /api/history       — list previously searched keywords
  GET  /api/results/{kw}  — fetch stored results for a keyword
  GET  /health            — liveness check
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from collections import Counter
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from config import settings
from database import Article, init_db, get_db
from crawler.base import RawItem
from crawler.google_news_crawler import crawl_google_news
from crawler.pantip_crawler import crawl_pantip
from preprocessing.processor import preprocess, Deduplicator, tokenize
from sentiment.analyzer import get_analyzer

from api.schemas import (
    SearchRequest, SearchResponse, ArticleOut,
    SentimentSummary, PlatformBreakdown, RegionCount, TopKeyword,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Thai Opinion Analysis API",
    version="1.0.0",
    description="Keyword-driven Thai social media & news sentiment analysis",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("Database initialised.")


# ── Static frontend ───────────────────────────────────────────────────────────
_FRONTEND = pathlib.Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(_FRONTEND / "index.html"))


# ════════════════════════════════════════════════════════════════════════════
# POST /api/search
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest, db: AsyncSession = Depends(get_db)):
    keyword   = req.keyword.strip()
    analyzer  = get_analyzer(req.model)
    dedup     = Deduplicator(threshold=settings.NEAR_DUP_THRESHOLD)
    max_items = req.max_items_per_source
    loop      = asyncio.get_event_loop()

    # ── 1. Crawl (run blocking crawlers in thread-pool) ───────────────────────
    crawl_tasks = []

    # "sanook" and "khaosod" are sourced via Google News RSS
    # (individual sites block automated requests with Cloudflare)
    if "sanook" in req.sources or "khaosod" in req.sources:
        crawl_tasks.append(
            loop.run_in_executor(None, crawl_google_news, keyword, max_items * 2)
        )

    if "pantip" in req.sources:
        crawl_tasks.append(
            loop.run_in_executor(None, crawl_pantip, keyword, max_items)
        )

    raw_batches: list[list[RawItem]] = await asyncio.gather(*crawl_tasks)
    raw_items: list[RawItem] = [item for batch in raw_batches for item in batch]
    total_crawled = len(raw_items)

    # ── 2. Preprocess + deduplicate ──────────────────────────────────────────
    clean_items = preprocess(raw_items, dedup)
    total_after_dedup = len(clean_items)

    if not clean_items:
        if total_crawled == 0:
            raise HTTPException(
                status_code=503,
                detail="Could not fetch data from any source. Check your internet connection and try again.",
            )
        raise HTTPException(
            status_code=422,
            detail=f"Crawled {total_crawled} items but all were removed by deduplication. Try a different keyword.",
        )

    # ── 3. Sentiment analysis (batched) ──────────────────────────────────────
    texts      = [item.text_content for item in clean_items]
    sentiments = await loop.run_in_executor(None, analyzer.predict_batch, texts)

    # ── 4. Persist to DB ─────────────────────────────────────────────────────
    for item, sentiment in zip(clean_items, sentiments):
        content_hash = Article.make_hash(item.text_content)
        existing = await db.execute(
            select(Article).where(Article.content_hash == content_hash)
        )
        if existing.scalar_one_or_none():
            continue
        db.add(Article(
            text_content     = item.text_content,
            sentiment_label  = sentiment.label,
            confidence_score = sentiment.score,
            source_platform  = item.source_platform,
            published_at     = item.published_at,
            region           = item.region,
            keyword          = keyword,
            content_hash     = content_hash,
            url              = item.url,
            title            = item.title,
        ))

    await db.commit()

    # ── 5. Build aggregated response from DB ─────────────────────────────────
    result = await db.execute(
        select(Article).where(Article.keyword == keyword)
    )
    all_articles: list[Article] = result.scalars().all()

    return SearchResponse(
        keyword             = keyword,
        total_crawled       = total_crawled,
        total_after_dedup   = total_after_dedup,
        sentiment_summary   = _summarise_sentiments(all_articles),
        platform_breakdown  = _platform_breakdown(all_articles),
        region_distribution = _region_distribution(all_articles),
        top_keywords        = _top_keywords(all_articles),
        articles            = [ArticleOut.model_validate(a) for a in all_articles],
    )


# ── GET /api/history ─────────────────────────────────────────────────────────

@app.get("/api/history")
async def history(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Article.keyword, func.count(Article.id).label("count"))
        .group_by(Article.keyword)
        .order_by(func.count(Article.id).desc())
    )
    return [{"keyword": r.keyword, "count": r.count} for r in result.all()]


# ── GET /api/results/{keyword} ───────────────────────────────────────────────

@app.get("/api/results/{keyword}", response_model=SearchResponse)
async def get_results(keyword: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Article).where(Article.keyword == keyword)
    )
    articles = result.scalars().all()
    if not articles:
        raise HTTPException(status_code=404, detail="No results found.")
    return SearchResponse(
        keyword             = keyword,
        total_crawled       = len(articles),
        total_after_dedup   = len(articles),
        sentiment_summary   = _summarise_sentiments(articles),
        platform_breakdown  = _platform_breakdown(articles),
        region_distribution = _region_distribution(articles),
        top_keywords        = _top_keywords(articles),
        articles            = [ArticleOut.model_validate(a) for a in articles],
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ════════════════════════════════════════════════════════════════════════════

def _summarise_sentiments(articles: list[Article]) -> SentimentSummary:
    pos = sum(1 for a in articles if a.sentiment_label == "positive")
    neg = sum(1 for a in articles if a.sentiment_label == "negative")
    neu = sum(1 for a in articles if a.sentiment_label == "neutral")
    return SentimentSummary(positive=pos, negative=neg, neutral=neu, total=len(articles))


def _platform_breakdown(articles: list[Article]) -> list[PlatformBreakdown]:
    platforms: dict[str, dict] = {}
    for a in articles:
        p = a.source_platform
        if p not in platforms:
            platforms[p] = {"count": 0, "positive": 0, "negative": 0, "neutral": 0}
        platforms[p]["count"] += 1
        platforms[p][a.sentiment_label or "neutral"] += 1
    return [PlatformBreakdown(platform=k, **v) for k, v in platforms.items()]


def _region_distribution(articles: list[Article]) -> list[RegionCount]:
    regions: dict[str, dict] = {}
    for a in articles:
        r = a.region or "ไม่ระบุ"
        if r not in regions:
            regions[r] = {"count": 0, "positive": 0, "negative": 0, "neutral": 0}
        regions[r]["count"] += 1
        regions[r][a.sentiment_label or "neutral"] += 1
    return sorted(
        [RegionCount(region=k, **v) for k, v in regions.items()],
        key=lambda x: x.count, reverse=True,
    )


def _top_keywords(articles: list[Article], top_n: int = 30) -> list[TopKeyword]:
    counter: Counter = Counter()
    for a in articles:
        counter.update(tokenize(a.text_content))
    return [TopKeyword(word=w, count=c) for w, c in counter.most_common(top_n)]

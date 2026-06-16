"""
Subsystem: HTTP API & pipeline orchestration (FastAPI).
--------------------------------------------------------------------------
Function:   Expose the system over REST and orchestrate the end-to-end flow:
            crawl -> preprocess/dedup -> sentiment -> persist -> aggregate.
Algorithms & techniques:
  * FastAPI with async routes; the synchronous, I/O- and CPU-bound stages
    (crawlers, transformer inference) are dispatched to a thread-pool executor
    so the event loop is never blocked.
  * The two crawlers run concurrently via asyncio.gather(return_exceptions);
    a failure in one source is logged and skipped, not fatal to the request.
  * Results are aggregated from the database (not just the fresh crawl) so
    repeated searches accumulate, and responses carry sentiment, platform, and
    province breakdowns plus the most frequent keywords.
Role in pipeline:
  * This is the top-level entry point that wires every other subsystem together
    in execution order: config -> database -> crawlers -> preprocessing ->
    sentiment -> persistence -> aggregation -> JSON response.
  * Failure isolation is layered: a single crawler error is skipped, a
    sentiment failure returns HTTP 500, an empty crawl returns 503, and a
    fully-deduplicated result returns 422 — each with an actionable message.
  * The aggregation helpers (_summarise_sentiments, _platform_breakdown,
    _region_distribution, _top_keywords) are pure functions over the stored
    rows, which keeps the route logic readable and the helpers easy to reason
    about in isolation.
--------------------------------------------------------------------------

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

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from config import settings, TOP_KEYWORDS_N, CACHE_TTL_SECONDS
from cache import TTLCache, make_search_key
from exceptions import (
    AppError, DataSourceUnavailable, NoUsableData, ModelError,
    PersistenceError, NoResultsFound,
)
from database import Article, init_db, get_db
from crawler.base import RawItem
from crawler.google_news_crawler import crawl_google_news
from crawler.pantip_crawler import crawl_pantip
from preprocessing.processor import preprocess, Deduplicator, tokenize
from sentiment.analyzer import get_analyzer

from api.schemas import (
    SearchRequest, SearchResponse, ArticleOut,
    SentimentSummary, PlatformBreakdown, RegionCount, TopKeyword,
    TimelinePoint, ConfidenceBucket, HistoryItem, HealthResponse,
)

logger = logging.getLogger(__name__)

# Process-wide cache: identical searches within the TTL skip the pipeline.
_search_cache = TTLCache(CACHE_TTL_SECONDS)

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


@app.exception_handler(AppError)
async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Translate any :class:`AppError` into its JSON HTTP response.

    Centralising this means route handlers can simply raise a domain exception
    and rely on its ``status_code``/``detail`` being applied consistently.
    """
    logger.warning("AppError on %s: %s", request.url.path, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.on_event("startup")
async def startup():
    """Create database tables before the app begins serving requests."""
    await init_db()
    logger.info("Database initialised.")


# ── Static frontend ───────────────────────────────────────────────────────────
_FRONTEND = pathlib.Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the single-page frontend at the site root."""
        return FileResponse(str(_FRONTEND / "index.html"))


# ════════════════════════════════════════════════════════════════════════════
# POST /api/search
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest, db: AsyncSession = Depends(get_db)):
    """Run the full pipeline for a keyword and return aggregated results.

    Stages: crawl the selected sources concurrently (blocking crawlers run in
    the thread-pool), preprocess and deduplicate the results, classify
    sentiment in a batch, persist new rows, then read back and aggregate the
    stored articles for the selected sources.

    Args:
        req: Validated search request.
        db: Injected database session.

    Returns:
        A :class:`SearchResponse` with summary, breakdowns, and articles.

    Raises:
        DataSourceUnavailable: ``503`` if nothing could be crawled.
        NoUsableData: ``422`` if every crawled item was removed by dedup.
        ModelError: ``500`` if sentiment analysis fails.
        PersistenceError: ``500`` if results cannot be saved.
    """
    keyword   = req.keyword.strip()
    analyzer  = get_analyzer(req.model)
    dedup     = Deduplicator(threshold=settings.NEAR_DUP_THRESHOLD)
    max_items = req.max_items_per_source
    loop      = asyncio.get_event_loop()
    logger.info("Search %r sources=%s model=%s", keyword, req.sources, req.model)

    # ── 0. Return a cached response for an identical recent search ────────────
    cache_key = make_search_key(keyword, req.model, req.sources)
    cached = _search_cache.get(cache_key)
    if cached is not None:
        logger.info("Cache hit for %s", cache_key)
        return cached

    # ── 1. Crawl (run blocking crawlers in thread-pool) ───────────────────────
    crawl_tasks = []

    # Google News RSS covers Thai news broadly (incl. Sanook, Khaosod, etc.)
    if "google_news" in req.sources:
        crawl_tasks.append(
            loop.run_in_executor(None, crawl_google_news, keyword, max_items * 2)
        )

    if "pantip" in req.sources:
        crawl_tasks.append(
            loop.run_in_executor(None, crawl_pantip, keyword, max_items)
        )

    # gather(return_exceptions) so one failing crawler does not abort the other
    raw_batches = await asyncio.gather(*crawl_tasks, return_exceptions=True)
    raw_items: list[RawItem] = []
    for batch in raw_batches:
        if isinstance(batch, Exception):
            logger.error("A crawler failed: %s", batch)
            continue
        raw_items.extend(batch)
    total_crawled = len(raw_items)
    logger.info("Crawled %d raw items for %r", total_crawled, keyword)

    # ── 2. Preprocess + deduplicate ──────────────────────────────────────────
    clean_items = preprocess(raw_items, dedup)
    total_after_dedup = len(clean_items)

    if not clean_items:
        if total_crawled == 0:
            raise DataSourceUnavailable()
        raise NoUsableData(
            f"Crawled {total_crawled} items but all were removed by deduplication. "
            "Try a different keyword."
        )

    # ── 3. Sentiment analysis (batched) ──────────────────────────────────────
    texts = [item.text_content for item in clean_items]
    try:
        sentiments = await loop.run_in_executor(None, analyzer.predict_batch, texts)
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Sentiment analysis failed for %r", keyword)
        raise ModelError() from exc
    logger.info("Classified %d items for %r", len(sentiments), keyword)

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

    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("Failed to persist results for %r", keyword)
        raise PersistenceError() from exc
    logger.info("Persisted results for %r", keyword)

    # ── 5. Build aggregated response from DB (only selected sources) ─────────
    # source_platform is stored as e.g. "google_news(Ch7.com)" or "pantip"
    # so we match by prefix using LIKE
    source_filters = []
    for src in req.sources:
        source_filters.append(Article.source_platform.like(f"{src}%"))

    result = await db.execute(
        select(Article).where(
            Article.keyword == keyword,
            or_(*source_filters) if source_filters else Article.keyword == keyword,
        )
    )
    all_articles: list[Article] = result.scalars().all()

    response = SearchResponse(
        keyword             = keyword,
        total_crawled       = total_crawled,
        total_after_dedup   = total_after_dedup,
        sentiment_summary   = _summarise_sentiments(all_articles),
        platform_breakdown  = _platform_breakdown(all_articles),
        region_distribution = _region_distribution(all_articles),
        top_keywords        = _top_keywords(all_articles),
        sentiment_timeline  = _sentiment_timeline(all_articles),
        confidence_distribution = _confidence_distribution(all_articles),
        articles            = [ArticleOut.model_validate(a) for a in all_articles],
    )
    _search_cache.set(cache_key, response)
    return response


# ── GET /api/history ─────────────────────────────────────────────────────────

@app.get("/api/history", response_model=list[HistoryItem])
async def history(db: AsyncSession = Depends(get_db)):
    """List previously searched keywords with their stored article counts.

    Args:
        db: Injected database session.

    Returns:
        A list of :class:`HistoryItem`, most-searched first.
    """
    result = await db.execute(
        select(Article.keyword, func.count(Article.id).label("count"))
        .group_by(Article.keyword)
        .order_by(func.count(Article.id).desc())
    )
    return [HistoryItem(keyword=r.keyword, count=r.count) for r in result.all()]


# ── GET /api/results/{keyword} ───────────────────────────────────────────────

@app.get("/api/results/{keyword}", response_model=SearchResponse)
async def get_results(keyword: str, db: AsyncSession = Depends(get_db)):
    """Return stored results for a previously searched keyword.

    Unlike :func:`search`, this does not crawl — it only reads what is already
    persisted.

    Args:
        keyword: The keyword to look up.
        db: Injected database session.

    Returns:
        A :class:`SearchResponse` built from the stored articles.

    Raises:
        NoResultsFound: ``404`` if no stored results exist for ``keyword``.
    """
    result = await db.execute(
        select(Article).where(Article.keyword == keyword)
    )
    articles = result.scalars().all()
    if not articles:
        raise NoResultsFound()
    return SearchResponse(
        keyword             = keyword,
        total_crawled       = len(articles),
        total_after_dedup   = len(articles),
        sentiment_summary   = _summarise_sentiments(articles),
        platform_breakdown  = _platform_breakdown(articles),
        region_distribution = _region_distribution(articles),
        top_keywords        = _top_keywords(articles),
        sentiment_timeline  = _sentiment_timeline(articles),
        confidence_distribution = _confidence_distribution(articles),
        articles            = [ArticleOut.model_validate(a) for a in articles],
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe — returns ``ok`` and the current server time."""
    return HealthResponse(status="ok", timestamp=datetime.utcnow().isoformat())


# ════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ════════════════════════════════════════════════════════════════════════════

def _summarise_sentiments(articles: list[Article]) -> SentimentSummary:
    """Count positive/neutral/negative labels across ``articles``."""
    pos = sum(1 for a in articles if a.sentiment_label == "positive")
    neg = sum(1 for a in articles if a.sentiment_label == "negative")
    neu = sum(1 for a in articles if a.sentiment_label == "neutral")
    return SentimentSummary(positive=pos, negative=neg, neutral=neu, total=len(articles))


def _platform_breakdown(articles: list[Article]) -> list[PlatformBreakdown]:
    """Group ``articles`` by source platform with per-label counts."""
    platforms: dict[str, dict] = {}
    for a in articles:
        p = a.source_platform
        if p not in platforms:
            platforms[p] = {"count": 0, "positive": 0, "negative": 0, "neutral": 0}
        platforms[p]["count"] += 1
        platforms[p][a.sentiment_label or "neutral"] += 1
    return [PlatformBreakdown(platform=k, **v) for k, v in platforms.items()]


def _region_distribution(articles: list[Article]) -> list[RegionCount]:
    """Group ``articles`` by province (descending), with per-label counts.

    Articles with no detected province are bucketed under ``"ไม่ระบุ"``
    (unspecified).
    """
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


def _sentiment_timeline(articles: list[Article]) -> list[TimelinePoint]:
    """Group ``articles`` by publication date with per-label counts.

    Powers the "sentiment trend over time" view: articles are bucketed by the
    date part of ``published_at`` (those without a date are skipped) and the
    resulting points are returned in chronological order.

    Args:
        articles: Articles to bucket.

    Returns:
        One :class:`TimelinePoint` per date, ascending by date.
    """
    days: dict[str, dict] = {}
    for a in articles:
        if a.published_at is None:
            continue
        day = a.published_at.strftime("%Y-%m-%d")
        bucket = days.setdefault(day, {"positive": 0, "neutral": 0, "negative": 0})
        bucket[a.sentiment_label or "neutral"] += 1
    return [
        TimelinePoint(date=d, **counts)
        for d, counts in sorted(days.items())
    ]


def _confidence_distribution(articles: list[Article]) -> list[ConfidenceBucket]:
    """Bucket prediction confidences into fixed 20%-wide ranges.

    Args:
        articles: Articles whose ``confidence_score`` is examined.

    Returns:
        Five :class:`ConfidenceBucket` entries covering 0–20% … 80–100%.
    """
    edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    counts = [0] * len(edges)
    for a in articles:
        score = a.confidence_score or 0.0
        for i, (lo, hi) in enumerate(edges):
            if lo <= score < hi:
                counts[i] += 1
                break
    return [ConfidenceBucket(range=labels[i], count=counts[i]) for i in range(len(edges))]


def _top_keywords(articles: list[Article], top_n: int = TOP_KEYWORDS_N) -> list[TopKeyword]:
    """Return the ``top_n`` most frequent tokens across ``articles``.

    Each article's text is re-tokenized (with stopwords removed) and the token
    frequencies are pooled.

    Args:
        articles: Articles to scan.
        top_n: How many of the most common tokens to return.

    Returns:
        The most frequent tokens, paired with their counts, most-common first.
    """
    counter: Counter = Counter()
    for a in articles:
        counter.update(tokenize(a.text_content))
    return [TopKeyword(word=w, count=c) for w, c in counter.most_common(top_n)]

"""
Subsystem: API data contracts (Pydantic schemas).
--------------------------------------------------------------------------
Function:   Define and validate the request/response shapes exchanged between
            the frontend and the FastAPI backend.
Algorithms & techniques:
  * Pydantic v2 models give automatic type coercion, validation, and OpenAPI
    documentation generation for every endpoint.
  * Field validators harden input at the boundary: the keyword is trimmed and
    rejected if empty, and the source list is de-duplicated while preserving
    order, so malformed requests fail fast with clear messages.
Role in pipeline:
  * Defines the single request type (SearchRequest) and the response types the
    API returns, so the contract between frontend and backend lives in one
    place and is validated automatically on every call.
--------------------------------------------------------------------------
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

from validators import clean_keyword, dedupe_preserving_order


# ── Request ──────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Body of ``POST /api/search``.

    Attributes:
        keyword: Thai keyword to search for (whitespace is trimmed; must be
            non-empty after trimming).
        sources: Which platforms to crawl. Duplicates are removed while order
            is preserved.
        max_items_per_source: Per-source crawl cap (1–100).
        model: Which sentiment model to run.
    """
    keyword: str = Field(..., min_length=1, description="Thai keyword to search")
    sources: list[Literal["google_news", "pantip"]] = Field(
        default=["google_news", "pantip"],
        description="Which platforms to crawl",
    )
    max_items_per_source: int = Field(default=30, ge=1, le=100)
    model: Literal["xlm-roberta", "wangchanberta"] = Field(
        default="xlm-roberta",
        description="Sentiment model to use",
    )

    @field_validator("keyword")
    @classmethod
    def _strip_keyword(cls, v: str) -> str:
        """Trim surrounding whitespace and reject whitespace-only keywords."""
        return clean_keyword(v)

    @field_validator("sources")
    @classmethod
    def _dedupe_sources(cls, v: list[str]) -> list[str]:
        """Remove duplicate sources while preserving order; reject empty."""
        return dedupe_preserving_order(v)


# ── Response ─────────────────────────────────────────────────────────────────

class ArticleOut(BaseModel):
    """A single analysed article as returned to the client.

    Mirrors the persisted :class:`database.Article` (``from_attributes`` lets
    it be built directly from an ORM row).
    """
    id: int
    text_content: str
    sentiment_label: Optional[str]
    confidence_score: Optional[float]
    source_platform: str
    published_at: Optional[datetime]
    region: Optional[str]
    url: Optional[str]
    title: Optional[str]

    class Config:
        from_attributes = True


class SentimentSummary(BaseModel):
    """Counts of each sentiment label plus the overall total."""
    positive: int
    neutral: int
    negative: int
    total: int


class PlatformBreakdown(BaseModel):
    """Per-platform article count with a sentiment breakdown."""
    platform: str
    count: int
    positive: int
    neutral: int
    negative: int


class RegionCount(BaseModel):
    """Per-province article count with a sentiment breakdown."""
    region: str
    count: int
    positive: int
    negative: int
    neutral: int


class TopKeyword(BaseModel):
    """A frequent token and how many times it appeared."""
    word: str
    count: int


class TimelinePoint(BaseModel):
    """Sentiment counts for a single publication date (for the trend chart)."""
    date: str
    positive: int
    neutral: int
    negative: int


class ConfidenceBucket(BaseModel):
    """How many predictions fell into a given confidence range."""
    range: str
    count: int


class SearchResponse(BaseModel):
    """Aggregated result returned by the search and results endpoints.

    Attributes:
        keyword: The keyword these results are for.
        total_crawled: Items fetched before deduplication.
        total_after_dedup: Items remaining after deduplication.
        sentiment_summary: Overall label counts.
        platform_breakdown: Counts grouped by source platform.
        region_distribution: Counts grouped by province (descending).
        top_keywords: Most frequent tokens across the articles.
        sentiment_timeline: Sentiment counts grouped by publication date.
        confidence_distribution: Prediction counts grouped by confidence range.
        articles: The individual analysed articles.
    """
    keyword: str
    total_crawled: int
    total_after_dedup: int
    sentiment_summary: SentimentSummary
    platform_breakdown: list[PlatformBreakdown]
    region_distribution: list[RegionCount]
    top_keywords: list[TopKeyword]
    sentiment_timeline: list[TimelinePoint]
    confidence_distribution: list[ConfidenceBucket]
    articles: list[ArticleOut]


class HistoryItem(BaseModel):
    """One entry in the search-history list: a keyword and its stored count."""
    keyword: str
    count: int


class HealthResponse(BaseModel):
    """Body returned by the ``/health`` liveness probe."""
    status: str
    timestamp: str

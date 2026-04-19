from __future__ import annotations
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    keyword: str = Field(..., min_length=1, description="Thai keyword to search")
    sources: list[Literal["sanook", "khaosod", "pantip"]] = Field(
        default=["sanook", "khaosod", "pantip"],
        description="Which platforms to crawl",
    )
    max_items_per_source: int = Field(default=30, ge=1, le=100)
    model: Literal["xlm-roberta", "wangchanberta"] = Field(
        default="xlm-roberta",
        description="Sentiment model to use",
    )


# ── Response ─────────────────────────────────────────────────────────────────

class ArticleOut(BaseModel):
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
    positive: int
    neutral: int
    negative: int
    total: int


class PlatformBreakdown(BaseModel):
    platform: str
    count: int
    positive: int
    neutral: int
    negative: int


class RegionCount(BaseModel):
    region: str
    count: int
    positive: int
    negative: int
    neutral: int


class TopKeyword(BaseModel):
    word: str
    count: int


class SearchResponse(BaseModel):
    keyword: str
    total_crawled: int
    total_after_dedup: int
    sentiment_summary: SentimentSummary
    platform_breakdown: list[PlatformBreakdown]
    region_distribution: list[RegionCount]
    top_keywords: list[TopKeyword]
    articles: list[ArticleOut]

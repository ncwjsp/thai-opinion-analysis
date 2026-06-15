"""
Subsystem: Persistence layer (SQLAlchemy + SQLite).
--------------------------------------------------------------------------
Function:   Define the storage schema for analysed records and provide async
            engine/session helpers to the rest of the application.
Algorithms & techniques:
  * SQLAlchemy 2.x async engine over SQLite (aiosqlite) so all database I/O is
    awaitable and never blocks the FastAPI event loop.
  * A UNIQUE content_hash (MD5 of the text) enforces exact deduplication at the
    database level, complementing the in-memory dedup done in preprocessing.
  * Composite and single-column indexes back the three hot query paths:
    keyword+platform filtering, and aggregation by sentiment and by region.
Role in pipeline:
  * The persistence boundary. The API writes one Article per analysed item and
    later reads them back to build aggregated responses, so results accumulate
    across repeated searches rather than being recomputed each time.
  * The unique content_hash means a re-run of the same keyword is idempotent:
    already-seen items are silently skipped instead of duplicated.
--------------------------------------------------------------------------

The application stores every crawled-and-analysed record as an :class:`Article`
row in a local SQLite database (a Railway volume path is used automatically in
production). Access is async via SQLAlchemy's asyncio engine so database I/O
never blocks the FastAPI event loop.
"""

import hashlib
import logging
from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, Text, Integer, Index
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Article(Base):
    """One crawled and analysed record.

    Attributes:
        id: Auto-increment primary key.
        text_content: Cleaned text that was classified.
        sentiment_label: ``positive`` / ``neutral`` / ``negative`` (nullable
            until analysis has run).
        confidence_score: Model confidence for the label.
        source_platform: Origin label, e.g. ``google_news(...)`` or ``pantip``.
        published_at: Original publication time, if known.
        region: Thai province extracted by NER, if any.
        keyword: Search keyword that produced the record.
        content_hash: MD5 of the text, unique — enforces exact dedup at the DB
            level so repeated searches do not insert duplicates.
        url: Link to the original content.
        title: Original title.
        created_at: Row insertion time.

    The composite and single-column indexes support the API's three hot query
    paths: filtering by keyword+platform, and aggregating by sentiment and by
    region.
    """
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_content   = Column(Text,    nullable=False)
    sentiment_label = Column(String(16), nullable=True)   # positive/negative/neutral
    confidence_score = Column(Float,  nullable=True)
    source_platform  = Column(String(64), nullable=False)  # google rss / pantip
    published_at     = Column(DateTime, nullable=True)
    region           = Column(String(128), nullable=True)  # Thai province from NER
    keyword          = Column(String(256), nullable=False)
    content_hash     = Column(String(64),  nullable=False, unique=True)  # MD5 for exact dedup
    url              = Column(String(512), nullable=True)
    title            = Column(String(512), nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_keyword_platform", "keyword", "source_platform"),
        Index("ix_sentiment", "sentiment_label"),
        Index("ix_region", "region"),
    )

    @staticmethod
    def make_hash(text: str) -> str:
        """Return the MD5 hex digest of ``text`` for the unique content hash.

        Args:
            text: The text to hash (typically the cleaned content).

        Returns:
            A 32-character hexadecimal MD5 digest.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── Async engine (used by FastAPI) ──────────────────────────────────────────
async_engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    """Create all tables if they do not already exist.

    Idempotent — safe to call on every start-up.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema ensured.")


async def get_db():
    """Yield a database session for FastAPI dependency injection.

    Rolls back and re-raises on error so a failed request never leaves a
    half-applied transaction, and always closes the session afterwards.

    Yields:
        An :class:`AsyncSession` bound to the request scope.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            logger.exception("Database session rolled back due to error")
            raise

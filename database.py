import hashlib
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Text, Integer, Index
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from config import settings


class Base(DeclarativeBase):
    pass


class Article(Base):
    """One crawled + analyzed record."""
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_content   = Column(Text,    nullable=False)
    sentiment_label = Column(String(16), nullable=True)   # positive/negative/neutral
    confidence_score = Column(Float,  nullable=True)
    source_platform  = Column(String(64), nullable=False)  # sanook / khaosod / pantip
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
        return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── Async engine (used by FastAPI) ──────────────────────────────────────────
async_engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

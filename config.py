import os
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # Database - supports Railway volume mount
    @property
    def DATABASE_URL(self) -> str:
        """Returns database URL with Railway volume path if available"""
        volume_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", ".")
        return f"sqlite+aiosqlite:///{volume_path}/opinion_data.db"
    
    @property
    def DATABASE_SYNC_URL(self) -> str:
        """Returns sync database URL with Railway volume path if available"""
        volume_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", ".")
        return f"sqlite:///{volume_path}/opinion_data.db"

    # Sentiment model — swap to "airesearch/wangchanberta-base-att-spm-uncased"
    # for WangchanBERTa (requires fine-tuned checkpoint for sentiment)
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    SENTIMENT_LABELS: list[str] = ["negative", "neutral", "positive"]

    # Crawling limits
    MAX_ITEMS_PER_SOURCE: int = 30
    CRAWL_TIMEOUT: int = 30            # seconds per page request
    SELENIUM_HEADLESS: bool = True

    # Deduplication
    NEAR_DUP_THRESHOLD: float = 0.85  # MinHash similarity cutoff

    # CORS (for frontend dev)
    CORS_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()

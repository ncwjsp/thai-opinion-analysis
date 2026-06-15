"""
Subsystem: Central configuration & logging.
--------------------------------------------------------------------------
Function:   Provide one authoritative place for runtime settings, tunable
            constants, and logging setup shared by every other subsystem.
Algorithms & techniques:
  * Pydantic BaseSettings loads per-deployment values from environment
    variables and an optional .env file, so the same code runs locally and on
    Railway (where the database path comes from a mounted volume).
  * Implementation constants (retry counts, timeouts, batch sizes, MinHash
    parameters) are named here instead of being scattered as magic numbers.
Role in pipeline:
  * Loaded first, before any other subsystem. setup_logging() is invoked at
    process start so every subsystem's logger shares one format and level.
  * Centralising the tunables here means a reviewer can see, in one place, the
    knobs that govern crawling politeness, deduplication strictness, and model
    inference batching.
--------------------------------------------------------------------------

Central application configuration and tunable constants.

All runtime settings live on the :class:`Settings` object, which reads values
from environment variables (and an optional ``.env`` file) so that deployment
targets such as Railway can override them without code changes. Constants that
are implementation details rather than per-deployment settings — retry counts,
timeouts, model batch sizes — are defined here as module-level values so they
can be imported and reused instead of being hard-coded ("magic numbers")
throughout the crawler, preprocessing, and sentiment modules.

This module also exposes :func:`setup_logging`, the single place where the
application's logging format and level are configured.
"""

import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent


# ════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging for the whole application.

    Called once at process start-up (see ``run.py``). Every module obtains its
    own logger via ``logging.getLogger(__name__)`` and inherits this
    configuration, so log lines are formatted consistently and share a single
    output stream.

    Args:
        level: Optional log level name (e.g. ``"DEBUG"``, ``"INFO"``). When
            ``None``, the value of the ``LOG_LEVEL`` environment variable is
            used, defaulting to ``"INFO"``.
    """
    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
    )
    # Quiet down noisy third-party loggers that would otherwise flood output.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)


# ════════════════════════════════════════════════════════════════════════════
# HTTP / crawling tunables
# ════════════════════════════════════════════════════════════════════════════

# Number of attempts for any outbound HTTP request before giving up.
HTTP_RETRIES = 3
# Base for exponential backoff between retries: sleep = BACKOFF_BASE ** attempt.
BACKOFF_BASE = 2
# Per-request socket timeout in seconds.
REQUEST_TIMEOUT = 20

# Pantip is aggressively rate-limited, so its crawler is capped tightly.
PANTIP_MAX_PAGES = 3        # stop after this many search-result pages
PANTIP_PAGE_DELAY = 3.0     # seconds to wait between pages (avoid HTTP 429)
RATE_LIMIT_WAIT = 10        # base seconds to back off after a 429 response


# ════════════════════════════════════════════════════════════════════════════
# Preprocessing tunables
# ════════════════════════════════════════════════════════════════════════════

# PyThaiNLP NER engine used to extract province (LOC) entities.
NER_ENGINE = "thainer"
# MinHash configuration for near-duplicate detection.
MINHASH_NUM_PERM = 128      # permutations — higher = more accurate, slower
MINHASH_NGRAM = 3           # character n-gram size used to build the MinHash


# ════════════════════════════════════════════════════════════════════════════
# Sentiment / aggregation tunables
# ════════════════════════════════════════════════════════════════════════════

# Local HuggingFace pipeline batch size for inference.
SENTIMENT_BATCH_SIZE = 16
# Token cap fed to the model (longer inputs are truncated).
MODEL_MAX_LENGTH = 416
# Maximum inputs per HuggingFace Inference API call.
HF_API_BATCH = 100
# How many of the most frequent keywords to return in a search response.
TOP_KEYWORDS_N = 30


class Settings(BaseSettings):
    """Per-deployment settings, overridable via environment / ``.env``.

    Attributes:
        SENTIMENT_MODEL: Default HuggingFace model id used when no model is
            specified on a request.
        SENTIMENT_LABELS: Canonical sentiment label ordering.
        MAX_ITEMS_PER_SOURCE: Default crawl cap per source.
        CRAWL_TIMEOUT: Per-request crawl timeout in seconds.
        SELENIUM_HEADLESS: Retained for backward compatibility with older
            crawler configurations.
        NEAR_DUP_THRESHOLD: MinHash Jaccard-similarity cutoff above which two
            documents are treated as near-duplicates.
        CORS_ORIGINS: Allowed CORS origins for the frontend.
    """

    @property
    def DATABASE_URL(self) -> str:
        """Async SQLAlchemy URL, honoring a Railway volume mount if present."""
        volume_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", ".")
        return f"sqlite+aiosqlite:///{volume_path}/opinion_data.db"

    @property
    def DATABASE_SYNC_URL(self) -> str:
        """Synchronous SQLAlchemy URL (used by tooling that lacks async)."""
        volume_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", ".")
        return f"sqlite:///{volume_path}/opinion_data.db"

    # Sentiment model — swap to "airesearch/wangchanberta-base-att-spm-uncased"
    # for WangchanBERTa (requires fine-tuned checkpoint for sentiment)
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    SENTIMENT_LABELS: list[str] = ["negative", "neutral", "positive"]

    # Crawling limits
    MAX_ITEMS_PER_SOURCE: int = 30
    CRAWL_TIMEOUT: int = REQUEST_TIMEOUT   # seconds per page request
    SELENIUM_HEADLESS: bool = True

    # Deduplication
    NEAR_DUP_THRESHOLD: float = 0.85  # MinHash similarity cutoff

    # CORS (for frontend dev)
    CORS_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()

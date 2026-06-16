"""
Subsystem: In-memory result cache.
--------------------------------------------------------------------------
Function:   Provide a small time-to-live (TTL) cache so that repeating an
            identical search within a short window returns the previous result
            instead of re-crawling and re-classifying.
Algorithms & techniques:
  * A plain dictionary keyed by a deterministic search key, storing the value
    together with its insertion timestamp; entries older than the TTL are
    treated as absent and lazily evicted on access.
  * Access is guarded by a lock so the cache is safe to share across the
    FastAPI thread-pool workers.
Role in pipeline:
  * Sits in front of the search pipeline. It changes nothing about the shape of
    the response — only whether the work is recomputed — so callers are
    unaffected apart from latency.
--------------------------------------------------------------------------
"""

import threading
import time
from typing import Any, Optional


class TTLCache:
    """A minimal thread-safe time-to-live cache.

    Attributes:
        ttl: Number of seconds an entry remains valid after insertion.
    """

    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for ``key``, or ``None`` if absent/expired.

        Args:
            key: Cache key to look up.

        Returns:
            The stored value if present and still within its TTL, else ``None``.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            inserted_at, value = entry
            if time.monotonic() - inserted_at > self.ttl:
                # Expired — drop it and report a miss.
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` with the current timestamp.

        Expired entries are swept on each write so the cache cannot grow
        without bound when many distinct keys are searched over time.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        self._evict_expired()
        with self._lock:
            self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()

    def _evict_expired(self) -> None:
        """Drop every entry whose TTL has elapsed (housekeeping helper)."""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, (ts, _) in self._store.items() if now - ts > self.ttl]
            for k in expired:
                del self._store[k]


def make_search_key(keyword: str, model: str, sources: list[str]) -> str:
    """Build a deterministic cache key for a search request.

    The key is order-independent in the source list so that selecting the same
    sources in a different order still hits the same cache entry.

    Args:
        keyword: The (already trimmed) search keyword.
        model: The selected model name.
        sources: The selected source names.

    Returns:
        A stable string key.

    Example:
        ``make_search_key("cat", "xlm-roberta", ["pantip", "google_news"])``
        and the same call with the sources reversed both produce the key
        ``"cat|xlm-roberta|google_news,pantip"``.
    """
    sources_part = ",".join(sorted(sources))
    return f"{keyword}|{model}|{sources_part}"

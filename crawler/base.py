"""
Shared data contract for all crawlers.

Every crawler in this package — regardless of whether it reads an RSS feed or
scrapes HTML — converts its source-specific payload into a single uniform
:class:`RawItem`. Normalising at the collection boundary means the rest of the
pipeline (preprocessing, deduplication, sentiment analysis, persistence) only
ever has to understand one structure, no matter where the data came from.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RawItem:
    """A single crawled record in normalised form.

    Attributes:
        text_content: The raw text to be analysed (title, body, or both).
        source_platform: Origin label, e.g. ``"google_news(Thairath)"`` or
            ``"pantip"``. The prefix before any parenthesis identifies the
            source family and is what the API filters on.
        keyword: The search keyword that produced this item.
        url: Canonical link to the original content, if available.
        title: Human-readable title, if available.
        published_at: Publication timestamp (naive) or ``None`` when the source
            does not expose one.
        region: Thai province name. Left ``None`` by the crawler and filled in
            later by the NER step in preprocessing.
    """
    text_content: str
    source_platform: str
    keyword: str
    url: str = ""
    title: str = ""
    published_at: Optional[datetime] = None
    region: Optional[str] = None        # filled by NER in preprocessing


# A desktop User-Agent plus a Thai-first Accept-Language header. Some Thai news
# and forum sites vary their markup or block requests that look automated, so we
# present as an ordinary browser to receive the same HTML a human visitor would.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
}

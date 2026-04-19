from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RawItem:
    """Uniform item schema produced by every crawler."""
    text_content: str
    source_platform: str
    keyword: str
    url: str = ""
    title: str = ""
    published_at: Optional[datetime] = None
    region: Optional[str] = None        # filled by NER in preprocessing


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
}

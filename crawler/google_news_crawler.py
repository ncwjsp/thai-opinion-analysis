"""
Google News RSS crawler for Thai news.

Fetches up to max_items news articles matching a Thai keyword from Google News RSS.
This aggregates articles from Sanook, Khaosod, Matichon, Thairath, and all major
Thai news outlets — without hitting individual site bot-detection.

RSS URL: https://news.google.com/rss/search?q=KEYWORD&hl=th&gl=TH&ceid=TH:th
"""

import re
import html
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import requests

from crawler.base import RawItem, HEADERS

_RSS_URL = "https://news.google.com/rss/search?q={kw}&hl=th&gl=TH&ceid=TH:th"
_RE_HTML = re.compile(r"<[^>]+>")


def crawl_google_news(keyword: str, max_items: int = 30) -> list[RawItem]:
    """Synchronous RSS crawl — runs in FastAPI's thread-pool executor."""
    url = _RSS_URL.format(kw=quote(keyword))
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            break
        except Exception as exc:
            import logging, time
            logging.getLogger(__name__).warning("Google News attempt %d failed: %s", attempt+1, exc)
            if attempt == 2:
                return []
            time.sleep(2 ** attempt)

    root = ET.fromstring(resp.content)
    items = root.findall(".//item")

    collected: list[RawItem] = []
    for item in items[:max_items]:
        title   = html.unescape(item.findtext("title", "").strip())
        raw_desc = item.findtext("description", "")
        # description is an HTML <a> tag; strip to get plain text
        desc    = html.unescape(_RE_HTML.sub("", raw_desc)).strip()
        href    = item.findtext("link", "").strip()
        source  = item.findtext("source", "").strip()
        pub_raw = item.findtext("pubDate", "")

        text = f"{title} {desc}".strip()
        if not text:
            continue

        published_at = _parse_rfc2822(pub_raw)
        collected.append(RawItem(
            text_content    = text,
            source_platform = f"google_news({source})" if source else "google_news",
            keyword         = keyword,
            url             = href,
            title           = title,
            published_at    = published_at,
        ))

    return collected


def _parse_rfc2822(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).replace(tzinfo=None)
    except Exception:
        return None

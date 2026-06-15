"""
Subsystem: Data collection — Google News RSS crawler for Thai news.
--------------------------------------------------------------------------
Function:   Fetch news articles matching a Thai keyword and normalise them to
            the shared RawItem schema.
Algorithms & techniques:
  * Uses Google News' RSS aggregation endpoint, which already merges Sanook,
    Khaosod, Matichon, Thairath and other outlets — avoiding per-site
    bot-detection and HTML scraping entirely.
  * XML is parsed with the standard library ElementTree; HTML entities in the
    description are unescaped and stripped to plain text.
  * Network calls use bounded retries with exponential backoff and degrade to
    an empty result rather than raising, so one bad source never fails a search.
Role in pipeline:
  * One of the two data-collection sources. It contributes mainstream news
    coverage and, because Google News already aggregates many outlets, a single
    request yields broad source diversity at low cost.
  * The per-item source_platform is tagged as google_news(<outlet>) so the API
    can still report which outlet each article came from.
--------------------------------------------------------------------------

Fetches up to ``max_items`` news articles matching a Thai keyword from Google
News RSS. This aggregates articles from Sanook, Khaosod, Matichon, Thairath,
and all major Thai news outlets — without hitting individual site
bot-detection.

RSS URL: https://news.google.com/rss/search?q=KEYWORD&hl=th&gl=TH&ceid=TH:th
"""

import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import requests

from config import BACKOFF_BASE, HTTP_RETRIES, REQUEST_TIMEOUT
from crawler.base import RawItem, HEADERS

logger = logging.getLogger(__name__)

_RSS_URL = "https://news.google.com/rss/search?q={kw}&hl=th&gl=TH&ceid=TH:th"
_RE_HTML = re.compile(r"<[^>]+>")


def crawl_google_news(keyword: str, max_items: int = 30) -> list[RawItem]:
    """Crawl Google News RSS for a Thai keyword.

    Synchronous by design — it is intended to run inside FastAPI's thread-pool
    executor so the event loop is never blocked. Network failures are handled
    gracefully: the request is retried with exponential backoff, and an empty
    list is returned rather than raising if the feed cannot be fetched or
    parsed.

    Args:
        keyword: Thai search term.
        max_items: Maximum number of articles to return.

    Returns:
        A list of :class:`RawItem`, at most ``max_items`` long. Empty if the
        feed could not be retrieved or contained no usable entries.
    """
    url = _RSS_URL.format(kw=quote(keyword))
    logger.info("Crawling Google News for %r (max %d)", keyword, max_items)

    resp = None
    for attempt in range(HTTP_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            logger.warning(
                "Google News attempt %d/%d failed: %s",
                attempt + 1, HTTP_RETRIES, exc,
            )
            if attempt == HTTP_RETRIES - 1:
                logger.error("Google News: giving up after %d attempts", HTTP_RETRIES)
                return []
            time.sleep(BACKOFF_BASE ** attempt)

    if resp is None:
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        logger.error("Google News: failed to parse RSS XML: %s", exc)
        return []

    items = root.findall(".//item")
    collected: list[RawItem] = []
    for item in items[:max_items]:
        title    = html.unescape(item.findtext("title", "").strip())
        raw_desc = item.findtext("description", "")
        # description is an HTML <a> tag; strip to get plain text
        desc    = html.unescape(_RE_HTML.sub("", raw_desc)).strip()
        href    = item.findtext("link", "").strip()
        source  = item.findtext("source", "").strip()
        pub_raw = item.findtext("pubDate", "")

        text = f"{title} {desc}".strip()
        if not text:
            continue

        collected.append(RawItem(
            text_content    = text,
            source_platform = f"google_news({source})" if source else "google_news",
            keyword         = keyword,
            url             = href,
            title           = title,
            published_at    = _parse_rfc2822(pub_raw),
        ))

    logger.info("Google News: collected %d items for %r", len(collected), keyword)
    return collected


def _parse_rfc2822(raw: str) -> datetime | None:
    """Parse an RSS ``pubDate`` (RFC-2822) into a naive datetime.

    Args:
        raw: The raw ``pubDate`` string, e.g. ``"Mon, 19 Apr 2026 08:00:00 GMT"``.

    Returns:
        A timezone-naive :class:`datetime`, or ``None`` if ``raw`` is empty or
        cannot be parsed.
    """
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None

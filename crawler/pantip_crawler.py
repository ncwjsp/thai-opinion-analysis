"""
Subsystem: Data collection — Pantip community forum crawler.
--------------------------------------------------------------------------
Function:   Scrape Pantip topic search results for a Thai keyword and
            normalise them to the shared RawItem schema.
Algorithms & techniques:
  * Plain requests + BeautifulSoup (lxml) — Pantip serves server-rendered
    HTML, so a heavyweight headless browser (Selenium) is unnecessary.
  * Politeness/anti-rate-limit policy: bounded retries with exponential
    backoff, explicit HTTP 429 handling with a widening wait, an inter-page
    delay, and a hard page cap.
  * Thai Buddhist-Era dates with abbreviated month names are converted to the
    Gregorian calendar (subtract 543, expand 2-digit years into the 25xx range).
Role in pipeline:
  * One of the two data-collection sources. It contributes community-forum
    opinion (as opposed to the news coverage from the Google News crawler),
    emitting the same RawItem schema so downstream stages treat both alike.
  * Paging stops on the first of: reaching max_items, an empty result list, or
    the page cap — whichever comes first, to stay polite to the source.
--------------------------------------------------------------------------

Pantip returns server-rendered HTML for search results, so no Selenium is
needed. Search URL: https://pantip.com/search?q=KEYWORD

Confirmed selectors (verified 2026-04-19):
  Result list:   ul.pt-list__sr
  Each card:     li.pt-list-item
  Title link:    a[href*="/topic/"]      — text is the full topic title
  Date:          span.pt-sm-toggle-date-hide — Thai Buddhist Era date

Pantip rate-limits aggressively, so the crawler is deliberately conservative:
it backs off on HTTP 429, waits between pages, and stops after a small page cap
(see ``PANTIP_MAX_PAGES`` in ``config``).
"""

import logging
import re
import time as _time
from datetime import datetime
from typing import Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from config import (
    BACKOFF_BASE,
    HTTP_RETRIES,
    PANTIP_MAX_PAGES,
    PANTIP_PAGE_DELAY,
    RATE_LIMIT_WAIT,
    REQUEST_TIMEOUT,
)
from crawler.base import RawItem, HEADERS

logger = logging.getLogger(__name__)

# Map Thai abbreviated month names to month numbers (Pantip prints dates like
# "19 เม.ย. 69", where the year is a 2-digit Buddhist-Era shorthand).
_THAI_MONTHS = {
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4,
    "พ.ค.": 5, "มิ.ย.": 6, "ก.ค.": 7, "ส.ค.": 8,
    "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
}


def crawl_pantip(keyword: str, max_items: int = 30) -> list[RawItem]:
    """Crawl Pantip search results for a Thai keyword.

    Synchronous by design — intended to run inside FastAPI's thread-pool
    executor. Paging continues until ``max_items`` is reached, the result list
    runs out, or the ``PANTIP_MAX_PAGES`` cap is hit. Whatever has been
    collected so far is returned on persistent network failure rather than
    raising.

    Args:
        keyword: Thai search term.
        max_items: Maximum number of topics to return.

    Returns:
        A list of :class:`RawItem`, at most ``max_items`` long.
    """
    logger.info("Crawling Pantip for %r (max %d)", keyword, max_items)
    collected: list[RawItem] = []
    page = 1

    while len(collected) < max_items:
        url = f"https://pantip.com/search?q={quote(keyword)}&page={page}"
        resp = _fetch_page(url, page)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "lxml")
        ul = soup.select_one("ul.pt-list__sr")
        if not ul:
            logger.info("Pantip: no result list on page %d — stopping", page)
            break

        cards = ul.select("li.pt-list-item")
        if not cards:
            logger.info("Pantip: no cards on page %d — stopping", page)
            break

        logger.debug("Pantip: %d cards on page %d", len(cards), page)
        for card in cards:
            if len(collected) >= max_items:
                break
            item = _parse_card(card, keyword)
            if item is not None:
                collected.append(item)

        page += 1
        if page > PANTIP_MAX_PAGES:    # cap pages to reduce rate-limiting
            logger.info("Pantip: reached page cap (%d)", PANTIP_MAX_PAGES)
            break
        _time.sleep(PANTIP_PAGE_DELAY)   # longer delay to avoid 429

    logger.info("Pantip: collected %d items for %r", len(collected), keyword)
    return collected


def _fetch_page(url: str, page: int) -> Optional[requests.Response]:
    """Fetch a single Pantip search page with retries and 429 backoff.

    Args:
        url: Fully-formed search URL for the page.
        page: Page number, used only for log context.

    Returns:
        The successful :class:`requests.Response`, or ``None`` if all attempts
        failed.
    """
    for attempt in range(HTTP_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = RATE_LIMIT_WAIT * (attempt + 1)
                logger.warning(
                    "Pantip page %d attempt %d: HTTP 429, waiting %ds",
                    page, attempt + 1, wait,
                )
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            logger.warning(
                "Pantip page %d attempt %d/%d failed: %s",
                page, attempt + 1, HTTP_RETRIES, exc,
            )
            if attempt == HTTP_RETRIES - 1:
                logger.error("Pantip: giving up on page %d", page)
                return None
            _time.sleep(BACKOFF_BASE ** attempt)
    return None


def _parse_card(card, keyword: str) -> Optional[RawItem]:
    """Extract a :class:`RawItem` from one Pantip result card.

    Args:
        card: A BeautifulSoup element for a single ``li.pt-list-item``.
        keyword: The search keyword (stored on the item).

    Returns:
        A :class:`RawItem`, or ``None`` if the card has no topic link or title.
    """
    a = card.select_one("a[href*='/topic/']")
    if not a:
        return None

    title = a.get_text(strip=True)
    if not title:
        return None

    href = a.get("href", "")
    if not href.startswith("http"):
        href = "https://pantip.com" + href

    date_el = card.select_one("span.pt-sm-toggle-date-hide, time, [class*=date]")
    date_str = date_el.get_text(strip=True) if date_el else ""

    return RawItem(
        text_content    = title,
        source_platform = "pantip",
        keyword         = keyword,
        url             = href,
        title           = title,
        published_at    = _parse_thai_date(date_str),
    )


def _parse_thai_date(raw: str) -> Optional[datetime]:
    """Parse a Thai-formatted Pantip date into a Gregorian datetime.

    Pantip prints dates such as ``"19 เม.ย. 69"`` using abbreviated Thai month
    names and a Buddhist-Era (BE) year. This converts both: BE years are mapped
    to the Gregorian calendar by subtracting 543, and 2-digit shorthand years
    are first expanded into the 25xx BE range.

    Args:
        raw: The raw date text from a result card.

    Returns:
        A :class:`datetime`, or ``None`` if no recognisable date is present.
    """
    if not raw:
        return None
    match = re.search(r"(\d{1,2})\s+([\w.]+)\s+(\d{2,4})", raw)
    if match:
        day, month_str, year_str = match.groups()
        month = _THAI_MONTHS.get(month_str)
        if month:
            year = int(year_str)
            if year < 100:           # 2-digit year shorthand (e.g. 69 → 2569 BE)
                year += 2500
            if year > 2400:          # Buddhist Era → Gregorian
                year -= 543
            try:
                return datetime(year, month, int(day))
            except ValueError:
                pass
    return None

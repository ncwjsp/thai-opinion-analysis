"""
Pantip crawler using plain requests + BeautifulSoup.

Pantip returns server-rendered HTML for search results, so no Selenium is needed.
Search URL: https://pantip.com/search?q=KEYWORD

Confirmed selectors (verified 2026-04-19):
  Result list:   ul.pt-list__sr
  Each card:     li.pt-list-item
  Title link:    a[href*="/topic/"]      — text is the full topic title
  Date:          span.pt-sm-toggle-date-hide — Thai Buddhist Era date
"""

import re
import time as _time
from datetime import datetime
from typing import Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from crawler.base import RawItem, HEADERS

_THAI_MONTHS = {
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4,
    "พ.ค.": 5, "มิ.ย.": 6, "ก.ค.": 7, "ส.ค.": 8,
    "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
}


def crawl_pantip(keyword: str, max_items: int = 30) -> list[RawItem]:
    """Synchronous HTTP crawl — runs in FastAPI's thread-pool executor."""
    collected: list[RawItem] = []
    page = 1

    while len(collected) < max_items:
        url = f"https://pantip.com/search?q={quote(keyword)}&page={page}"
        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=20)
                resp.raise_for_status()
                break
            except Exception as exc:
                import time as _t
                import logging
                logging.getLogger(__name__).warning("Pantip page %d attempt %d failed: %s", page, attempt+1, exc)
                if attempt == 2:
                    return collected
                _t.sleep(2 ** attempt)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "lxml")
        ul = soup.select_one("ul.pt-list__sr")
        if not ul:
            break

        cards = ul.select("li.pt-list-item")
        if not cards:
            break

        for card in cards:
            if len(collected) >= max_items:
                break

            a = card.select_one("a[href*='/topic/']")
            if not a:
                continue

            title = a.get_text(strip=True)
            href  = a.get("href", "")
            if not href.startswith("http"):
                href = "https://pantip.com" + href

            date_el = card.select_one("span.pt-sm-toggle-date-hide, time, [class*=date]")
            date_str = date_el.get_text(strip=True) if date_el else ""

            if not title:
                continue

            collected.append(RawItem(
                text_content    = title,
                source_platform = "pantip",
                keyword         = keyword,
                url             = href,
                title           = title,
                published_at    = _parse_thai_date(date_str),
            ))

        page += 1
        if page > 5:    # cap at 5 pages
            break
        _time.sleep(0.5)   # polite delay

    return collected


def _parse_thai_date(raw: str) -> Optional[datetime]:
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

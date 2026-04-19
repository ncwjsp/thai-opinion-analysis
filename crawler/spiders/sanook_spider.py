"""
Scrapy spider for Sanook News (www.sanook.com/news).

Search URL pattern: https://www.sanook.com/search/?q=KEYWORD&cat=news

SELECTOR NOTES — verify with browser DevTools if the site changes layout:
  Article cards:  div.list-layout__item  (or article.sanook-news-card)
  Title:          h2.news-title a  (text + href)
  Snippet:        p.news-description
  Date:           time[datetime] attribute
"""

import re
from datetime import datetime
from typing import Generator
from urllib.parse import quote

import scrapy
from scrapy.http import Response

from crawler.base import RawItem, HEADERS


class SanookSpider(scrapy.Spider):
    name = "sanook"
    custom_settings = {
        "USER_AGENT": HEADERS["User-Agent"],
        "DOWNLOAD_TIMEOUT": 30,
        "ROBOTSTXT_OBEY": False,
        "LOG_LEVEL": "ERROR",
        "FEEDS": {},                      # disable auto-export
        "ITEM_PIPELINES": {},
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 5.0,
    }

    def __init__(self, keyword: str = "", max_items: int = 30, collected: list = None, **kwargs):
        super().__init__(**kwargs)
        self.keyword = keyword
        self.max_items = int(max_items)
        self.collected: list[RawItem] = collected if collected is not None else []
        self.count = 0

    def start_requests(self):
        url = f"https://www.sanook.com/search/?q={quote(self.keyword)}&cat=news"
        yield scrapy.Request(url, headers=HEADERS, callback=self.parse)

    def parse(self, response: Response) -> Generator:
        # ── Article card selectors (adjust if site layout changes) ──────────
        cards = (
            response.css("div.list-layout__item")
            or response.css("article")
            or response.css("div.news-item")
        )

        for card in cards:
            if self.count >= self.max_items:
                return

            title = (
                card.css("h2 a::text, h3 a::text, .news-title::text").get("").strip()
                or card.css("a::text").get("").strip()
            )
            snippet = card.css(
                "p.news-description::text, .news-desc::text, p::text"
            ).get("").strip()
            href = card.css("a::attr(href)").get("")
            date_str = card.css("time::attr(datetime), time::text").get("")

            text = f"{title} {snippet}".strip()
            if not text:
                continue

            published_at = _parse_date(date_str)
            item = RawItem(
                text_content=text,
                source_platform="sanook",
                keyword=self.keyword,
                url=href,
                title=title,
                published_at=published_at,
            )
            self.collected.append(item)
            self.count += 1

        # ── Follow pagination up to max_items ────────────────────────────────
        if self.count < self.max_items:
            next_page = response.css("a.pagination__next::attr(href), a[rel=next]::attr(href)").get()
            if next_page:
                yield response.follow(next_page, headers=HEADERS, callback=self.parse)


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw[:19], fmt[:len(raw[:19])])
        except ValueError:
            pass
    return None

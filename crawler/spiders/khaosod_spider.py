"""
Scrapy spider for Khaosod (www.khaosod.co.th).

Search URL pattern: https://www.khaosod.co.th/?s=KEYWORD

SELECTOR NOTES — verify with browser DevTools if the site changes layout:
  Article cards:  article.post  (WordPress standard)
  Title:          h2.entry-title a
  Snippet:        div.entry-summary p
  Date:           time.entry-date[datetime]
"""

from datetime import datetime
from typing import Generator
from urllib.parse import quote

import scrapy
from scrapy.http import Response

from crawler.base import RawItem, HEADERS


class KhaoSodSpider(scrapy.Spider):
    name = "khaosod"
    custom_settings = {
        "USER_AGENT": HEADERS["User-Agent"],
        "DOWNLOAD_TIMEOUT": 30,
        "ROBOTSTXT_OBEY": False,
        "LOG_LEVEL": "ERROR",
        "FEEDS": {},
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
        url = f"https://www.khaosod.co.th/?s={quote(self.keyword)}"
        yield scrapy.Request(url, headers=HEADERS, callback=self.parse)

    def parse(self, response: Response) -> Generator:
        # WordPress theme — standard selectors
        cards = (
            response.css("article.post")
            or response.css("article")
            or response.css("div.td-module-container")
        )

        for card in cards:
            if self.count >= self.max_items:
                return

            title = card.css(
                "h2.entry-title a::text, h3.entry-title a::text, "
                ".td-module-title a::text, a::text"
            ).get("").strip()

            snippet = card.css(
                "div.entry-summary p::text, .td-excerpt::text, p::text"
            ).get("").strip()

            href = card.css(
                "h2.entry-title a::attr(href), h3 a::attr(href), "
                ".td-module-title a::attr(href)"
            ).get("")

            date_str = card.css(
                "time.entry-date::attr(datetime), time::attr(datetime), time::text"
            ).get("")

            text = f"{title} {snippet}".strip()
            if not text:
                continue

            published_at = _parse_date(date_str)
            item = RawItem(
                text_content=text,
                source_platform="khaosod",
                keyword=self.keyword,
                url=href,
                title=title,
                published_at=published_at,
            )
            self.collected.append(item)
            self.count += 1

        if self.count < self.max_items:
            next_page = response.css(
                "a.next.page-numbers::attr(href), a[rel=next]::attr(href)"
            ).get()
            if next_page:
                yield response.follow(next_page, headers=HEADERS, callback=self.parse)


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[:19], fmt[:len(raw[:19])])
        except ValueError:
            pass
    return None

"""
Standalone script — run ONE Scrapy spider and write results to a JSON file.

Called by scrapy_runner.py via subprocess so that Twisted's signal handlers
run in a real main thread (not a thread-pool worker).

Usage:
    python crawler/run_spider.py --spider sanook --keyword รัฐบาล
                                 --max 30 --output /tmp/results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Make sure the project root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scrapy.crawler import CrawlerProcess

from crawler.base import RawItem
from crawler.spiders.sanook_spider import SanookSpider
from crawler.spiders.khaosod_spider import KhaoSodSpider

SPIDERS = {
    "sanook":  SanookSpider,
    "khaosod": KhaoSodSpider,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider",  required=True, choices=list(SPIDERS))
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--max",     type=int, default=30)
    parser.add_argument("--output",  required=True)
    args = parser.parse_args()

    collected: list[RawItem] = []

    process = CrawlerProcess(settings={
        "LOG_LEVEL": "ERROR",
        "ROBOTSTXT_OBEY": False,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
    })
    process.crawl(
        SPIDERS[args.spider],
        keyword=args.keyword,
        max_items=args.max,
        collected=collected,
    )
    process.start()

    # Serialize to JSON (datetime → ISO string)
    data = []
    for item in collected:
        data.append({
            "text_content":   item.text_content,
            "source_platform": item.source_platform,
            "keyword":        item.keyword,
            "url":            item.url,
            "title":          item.title,
            "published_at":   item.published_at.isoformat() if item.published_at else None,
            "region":         item.region,
        })

    Path(args.output).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

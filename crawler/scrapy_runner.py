"""
Run Scrapy spiders safely from inside FastAPI.

Problem: Twisted installs signal handlers via signal.signal(), which Python
only permits from the main thread. FastAPI's thread-pool workers are not the
main thread, so running CrawlerProcess there raises ValueError.

Solution: launch each spider as a *subprocess* (its own main thread).
Results are written to a temp JSON file and read back here.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Type
from datetime import datetime

import scrapy

from crawler.base import RawItem

_RUNNER = Path(__file__).parent / "run_spider.py"

# Map spider class → name string used by run_spider.py
_SPIDER_NAMES: dict[str, str] = {
    "SanookSpider":  "sanook",
    "KhaoSodSpider": "khaosod",
}


async def run_spider_async(
    spider_cls: Type[scrapy.Spider],
    keyword: str,
    max_items: int,
) -> list[RawItem]:
    """
    Spawn a subprocess to run one Scrapy spider, then parse its JSON output.
    Fully async — awaitable from any FastAPI route.
    """
    spider_name = _SPIDER_NAMES.get(spider_cls.__name__, spider_cls.name)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = tmp.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_RUNNER),
            "--spider",  spider_name,
            "--keyword", keyword,
            "--max",     str(max_items),
            "--output",  output_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            # Log but don't crash — fall back to empty results
            import logging
            logging.getLogger(__name__).warning(
                "Spider %s exited %d: %s",
                spider_name, proc.returncode,
                stderr.decode(errors="replace")[-400:],
            )
            return []

        raw = json.loads(Path(output_path).read_text(encoding="utf-8"))
        return [_dict_to_item(d) for d in raw]

    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Spider %s failed: %s", spider_name, exc)
        return []
    finally:
        Path(output_path).unlink(missing_ok=True)


def _dict_to_item(d: dict) -> RawItem:
    pub = d.get("published_at")
    return RawItem(
        text_content    = d["text_content"],
        source_platform = d["source_platform"],
        keyword         = d["keyword"],
        url             = d.get("url", ""),
        title           = d.get("title", ""),
        published_at    = datetime.fromisoformat(pub) if pub else None,
        region          = d.get("region"),
    )

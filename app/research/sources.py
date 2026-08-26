"""Free, public topic sources for AI/tech discovery.

No scraping; only official RSS/Atom feeds and public APIs that allow this use.
Each source returns a list of candidate topics with minimal metadata.
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

from app.research.youtube_trending import fetch_trending_topics, TrendingTopic
from app.utils.logging import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

# Max length for a topic summary/description (stripped from RSS <description>).
MAX_DESC_LENGTH = 500


class _HTMLStripper(HTMLParser):
    """Minimal HTML stripper: removes tags, discards <style>/<script> content entirely."""

    def __init__(self):
        super().__init__()
        self._text_parts: list[str] = []
        self._discard = False  # True while inside <style> or <script>

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("style", "script"):
            self._discard = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script"):
            self._discard = False

    def handle_data(self, data: str) -> None:
        if not self._discard:
            self._text_parts.append(data)

    def get_text(self) -> str:
        # Collapse whitespace and trim
        text = "".join(self._text_parts)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_DESC_LENGTH]


def _strip_html(text: str) -> str:
    """Remove HTML tags and <style>/<script> content; collapse whitespace; truncate."""
    if not text:
        return ""
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


class TopicCandidate:
    def __init__(self, title: str, url: str, source: str,
                 summary: str = "", published: str | None = None):
        self.title = title
        self.url = url
        self.source = source
        self.summary = summary
        self.published = published or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "summary": self.summary,
            "published": self.published,
        }


# Curated list of free, policy-compliant feeds for AI/tech.
# Each has a parser function below.
FEEDS: list[dict] = [
    {
        "name": "google_ai_blog",
        "url": "https://research.google/blog/rss",
        "parser": "parse_generic_rss",
    },
    {
        "name": "openai_blog",
        "url": "https://openai.com/blog/rss.xml",
        "parser": "parse_generic_rss",
    },
    {
        "name": "anthropic_news",
        "url": "https://tim-hilde.github.io/anthropic-rss/rss.xml",
        "parser": "parse_generic_rss",
    },
    {
        "name": "github_blog",
        "url": "https://github.blog/feed/",
        "parser": "parse_generic_rss",
    },
    {
        "name": "huggingface_blog",
        "url": "https://huggingface.co/blog/feed.xml",
        "parser": "parse_generic_rss",
    },
    {
        "name": "microsoft_research",
        "url": "https://www.microsoft.com/en-us/research/feed/",
        "parser": "parse_generic_rss",
    },
    {
        "name": "arxiv_cs_ai",
        "url": "http://export.arxiv.org/rss/cs.AI",
        "parser": "parse_arxiv",
    },
    {
        "name": "arxiv_cs_cl",
        "url": "http://export.arxiv.org/rss/cs.CL",
        "parser": "parse_arxiv",
    },
    {
        "name": "arxiv_cs_cv",
        "url": "http://export.arxiv.org/rss/cs.CV",
        "parser": "parse_arxiv",
    },
    {
        "name": "arxiv_cs_lg",
        "url": "http://export.arxiv.org/rss/cs.LG",
        "parser": "parse_arxiv",
    },
]


def _fetch(url: str) -> str:
    return retry(lambda: _fetch_once(url), max_attempts=2,
                 retry_on=(urllib.error.URLError, TimeoutError))

def _fetch_once(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "YTShortsBot/0.1"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", "ignore")


def parse_generic_rss(xml: str, source: str) -> list[TopicCandidate]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        raw_desc = (item.findtext("description") or "").strip()
        # Strip HTML tags, remove <style>/<script> content, collapse whitespace, truncate
        desc = _strip_html(raw_desc)
        pub = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date") or ""
        if title and link:
            items.append(TopicCandidate(title, link, source, desc, pub))
    return items


def parse_google_ai(xml: str, source: str) -> list[TopicCandidate]:
    # Google AI Blog uses Atom with namespace
    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml)
    items = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        link_el = entry.find("atom:link", ns)
        link = link_el.get("href", "") if link_el is not None else ""
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=ns)
        if title and link:
            items.append(TopicCandidate(title, link, source, summary, published))
    return items


def parse_arxiv(xml: str, source: str) -> list[TopicCandidate]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub = item.findtext("pubDate") or ""
        if title and link:
            # arxiv title often has "Title - Authors" format; keep as-is.
            items.append(TopicCandidate(title, link, source, desc, pub))
    return items


def discover_candidates(max_per_source: int = 5, include_trending: bool = True) -> list[TopicCandidate]:
    """Fetch all configured feeds + YouTube trending, return deduplicated candidates."""
    all_cands: list[TopicCandidate] = []
    failed_count = 0

    # 1. RSS/Atom feeds
    for feed in FEEDS:
        try:
            xml = _fetch(feed["url"])
            parser = globals()[feed["parser"]]
            cands = parser(xml, feed["name"])
            all_cands.extend(cands[:max_per_source])
            logger.info(f"source {feed['name']}: {len(cands)} items",
                        extra={"stage": "discover", "status": "ok"})
        except Exception as exc:
            failed_count += 1
            logger.warning(f"source {feed['name']} failed: {exc}",
                           extra={"stage": "discover", "status": "error", "error": str(exc)})

    # 2. YouTube trending topics (converted to TopicCandidate format)
    if include_trending:
        try:
            trending = fetch_trending_topics(max_results=max_per_source * 2)
            for t in trending:
                # Convert TrendingTopic -> TopicCandidate
                # Use keywords as summary; no real URL since it's a topic not a video
                all_cands.append(TopicCandidate(
                    title=t.title,
                    url=f"youtube_trending:{t.category_id}:{t.title[:40]}",
                    source="youtube_trending",
                    summary=", ".join(t.keywords[:8]),
                    published=datetime.now(timezone.utc).isoformat(),
                ))
            logger.info(f"youtube_trending: {len(trending)} topics",
                        extra={"stage": "discover", "status": "ok"})
        except Exception as exc:
            logger.warning(f"youtube_trending failed: {exc}",
                           extra={"stage": "discover", "status": "error", "error": str(exc)})

    # All-feeds-failed warning: distinct from "feeds worked but nothing new"
    total_sources = len(FEEDS) + (1 if include_trending else 0)
    if failed_count == total_sources and total_sources > 0:
        logger.warning(
            f"all {total_sources} research sources failed — check for dead/moved URLs or API key",
            extra={"stage": "discover", "status": "all_failed", "error": "all_sources_failed"},
        )

    # Deduplicate by URL
    seen = set()
    unique = []
    for c in all_cands:
        if c.url not in seen:
            seen.add(c.url)
            unique.append(c)
    logger.info(f"total unique candidates: {len(unique)}",
                extra={"stage": "discover", "status": "done"})
    return unique
"""Fetch trending topics from YouTube (no video download, just metadata).

Pure topic research: gets titles, tags, categories from trending videos.
Feeds into the existing generator → original scripts + assets.
"""
from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrendingTopic:
    """A topic candidate from YouTube trending."""
    title: str
    keywords: list[str]
    category_id: str
    view_count: int
    source: str = "youtube_trending"


def _get_api_key() -> Optional[str]:
    return os.getenv("YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY")


def fetch_trending_topics_api(max_results: int = 20, region: str = "US") -> list[TrendingTopic]:
    """Fetch trending videos via YouTube Data API v3.

    Requires YOUTUBE_API_KEY (or GOOGLE_API_KEY) with YouTube Data API v3 enabled.
    Cost: 1 quota unit per call. Free tier: 10,000 units/day → plenty.
    """
    key = _get_api_key()
    if not key:
        logger.warning("YOUTUBE_API_KEY not set; skipping API trending fetch")
        return []

    try:
        import urllib.request
        import urllib.parse

        url = (
            "https://www.googleapis.com/youtube/v3/videos"
            f"?part=snippet,statistics,topicDetails"
            f"&chart=mostPopular"
            f"&regionCode={region}"
            f"&maxResults={max_results}"
            f"&key={key}"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        topics: list[TrendingTopic] = []
        for item in data.get("items", []):
            snip = item.get("snippet", {})
            stats = item.get("statistics", {})
            title = snip.get("title", "").strip()
            if not title:
                continue
            # Keywords: tags + topic IDs (fallback to title words)
            tags = snip.get("tags", [])
            topic_ids = []
            for td in item.get("topicDetails", {}).get("topicCategories", []):
                topic_ids.append(td.split("/")[-1].replace("_", " "))
            keywords = list(dict.fromkeys(tags + topic_ids + title.split()[:8]))[:12]
            view_count = int(stats.get("viewCount", "0"))
            topics.append(TrendingTopic(
                title=title,
                keywords=keywords,
                category_id=snip.get("categoryId", ""),
                view_count=view_count,
                source="youtube_trending_api",
            ))
        logger.info(f"fetched {len(topics)} trending topics from YouTube API")
        return topics

    except Exception as exc:
        logger.error(f"YouTube trending API fetch failed: {exc}")
        return []


def fetch_trending_topics_scrape(region: str = "US") -> list[TrendingTopic]:
    """Fallback: scrape YouTube trending page (no API key needed).

    WARNING: HTML structure changes break this. Use API when possible.
    """
    try:
        import urllib.request
        import re

        url = f"https://www.youtube.com/feed/trending?gl={region.lower()}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; YouTubeTrendingBot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode()

        # Extract video titles from ytInitialData JSON blob
        match = re.search(r"var ytInitialData = ({.*?});", html, re.DOTALL)
        if not match:
            logger.warning("ytInitialData not found in trending page")
            return []

        data = json.loads(match.group(1))
        topics: list[TrendingTopic] = []

        # Navigate the nested structure (brittle, but works as of 2024)
        try:
            contents = (
                data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]
                ["tabRenderer"]["content"]["richGridRenderer"]["contents"]
            )
        except (KeyError, IndexError, TypeError):
            logger.warning("trending page structure changed")
            return []

        for item in contents:
            if "richItemRenderer" not in item:
                continue
            content = item["richItemRenderer"]["content"]
            if "videoRenderer" not in content:
                continue
            vr = content["videoRenderer"]
            title = vr.get("title", {}).get("runs", [{}])[0].get("text", "").strip()
            if not title:
                continue
            # View count text
            view_text = vr.get("viewCountText", {}).get("simpleText", "0")
            view_count = int("".join(filter(str.isdigit, view_text)) or "0")
            keywords = title.split()[:8]
            topics.append(TrendingTopic(
                title=title,
                keywords=keywords,
                category_id="",
                view_count=view_count,
                source="youtube_trending_scrape",
            ))

        logger.info(f"fetched {len(topics)} trending topics from scrape")
        return topics

    except Exception as exc:
        logger.error(f"YouTube trending scrape failed: {exc}")
        return []


def fetch_trending_topics(
    max_results: int = 20,
    region: str = "US",
    prefer_api: bool = True,
) -> list[TrendingTopic]:
    """Main entry: try API first, fall back to scrape."""
    if prefer_api and _get_api_key():
        return fetch_trending_topics_api(max_results, region)
    return fetch_trending_topics_scrape(region)
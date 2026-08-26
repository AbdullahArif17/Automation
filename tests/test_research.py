"""Phase 4 tests: sources, researcher, verifier, topic selector."""
import json
from unittest.mock import patch, MagicMock

import pytest

from app.research.sources import TopicCandidate, discover_candidates, parse_generic_rss
from app.research.researcher import Researcher, ResearchResult
from app.research.verifier import FactVerifier, VerifiedFact
from app.content.topic_selector import TopicSelector, TopicScore
from app.ai.provider import MockProvider


def test_topic_candidate():
    c = TopicCandidate("Test", "http://example.com", "src", "summary", "2026-01-01")
    assert c.title == "Test"
    d = c.to_dict()
    assert d["url"] == "http://example.com"


def test_parse_generic_rss():
    xml = """<rss><channel><item><title>T</title><link>http://u</link>
<description>D</description><pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate></item></channel></rss>"""
    items = parse_generic_rss(xml, "test")
    assert len(items) == 1
    assert items[0].title == "T"
    assert items[0].url == "http://u"


def test_parse_generic_rss_strips_html():
    """parse_generic_rss strips HTML tags, removes <style>/<script> content,
    collapses whitespace, and truncates to MAX_DESC_LENGTH (500)."""
    from app.research.sources import MAX_DESC_LENGTH
    xml = """<rss><channel><item><title>HTML Topic</title><link>http://u</link>
<description><![CDATA[<p>Plain text with <strong>bold</strong> and <em>italic</em>.</p>
<style>body { color: red; font-size: 12px; }</style>
<script>console.log("evil");</script>
<p>More text after scripts.</p>]]></description>
<pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate></item></channel></rss>"""
    items = parse_generic_rss(xml, "test")
    assert len(items) == 1
    desc = items[0].summary
    # HTML tags removed
    assert "<p>" not in desc
    assert "<strong>" not in desc
    assert "<em>" not in desc
    # style/script content entirely removed
    assert "color: red" not in desc
    assert "console.log" not in desc
    # Only readable text remains, whitespace collapsed
    assert "Plain text with bold and italic." in desc
    assert "More text after scripts." in desc
    # Length cap
    assert len(desc) <= MAX_DESC_LENGTH

    # Plain text description passes through unchanged
    xml_plain = """<rss><channel><item><title>Plain</title><link>http://u</link>
<description>Simple summary without HTML.</description>
<pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate></item></channel></rss>"""
    items2 = parse_generic_rss(xml_plain, "test")
    assert items2[0].summary == "Simple summary without HTML."


def test_researcher_mock():
    p = MockProvider([json.dumps({
        "summary": "AI is advancing.",
        "sources": ["http://a"],
        "facts": ["Fact 1", "Fact 2"],
        "publication_dates": ["2026-01-01"],
        "confidence": 0.8
    })])
    r = Researcher(p)
    cands = [TopicCandidate("Topic", "http://a", "src", "sum", "2026-01-01")]
    res = r.research("Topic", cands)
    assert isinstance(res, ResearchResult)
    assert len(res.facts) == 2
    assert res.confidence == 0.8


def test_verifier_mock():
    p = MockProvider([json.dumps({
        "verified_facts": [
            {"fact": "Fact 1", "verified": True, "confidence": 0.9, "notes": "ok"},
            {"fact": "Fact 2", "verified": False, "confidence": 0.2, "notes": "no source"},
        ]
    })])
    v = FactVerifier(p)
    research = ResearchResult("t", "s", [], ["Fact 1", "Fact 2"], [], 0.8)
    verified = v.verify(research)
    assert len(verified) == 2
    assert verified[0].verified is True
    assert verified[1].verified is False


def test_topic_scoring():
    p = MockProvider([json.dumps({
        "trend": 0.8, "interest": 0.7, "novelty": 0.6,
        "visual": 0.5, "shorts": 0.6, "source_quality": 0.8
    })])
    # Need a DB mock for duplicate penalty
    mock_db = MagicMock()
    mock_db.fetchall.return_value = []
    sel = TopicSelector(p, db=mock_db)
    c = TopicCandidate("New AI Tool", "http://x", "src", "sum", "2026-01-01")
    research = ResearchResult("t", "s", ["http://x"], ["fact"], [], 0.8)
    verified = [VerifiedFact("fact", True, 0.9, "")]
    score = sel.score(c, research, verified)
    assert isinstance(score, TopicScore)
    assert 0 <= score.final <= 1
    assert score.duplicate_penalty == 0.0


def test_topic_selector_select_best():
    # Two candidates; second should win with higher scores
    scores_1 = json.dumps({"trend": 0.5, "interest": 0.5, "novelty": 0.5,
                           "visual": 0.5, "shorts": 0.5, "source_quality": 0.5})
    scores_2 = json.dumps({"trend": 0.9, "interest": 0.9, "novelty": 0.9,
                           "visual": 0.8, "shorts": 0.8, "source_quality": 0.9})
    p = MockProvider([scores_1, scores_2])
    mock_db = MagicMock()
    mock_db.fetchall.return_value = []
    sel = TopicSelector(p, db=mock_db)

    c1 = TopicCandidate("Topic A", "http://a", "src", "sum", "2026-01-01")
    c2 = TopicCandidate("Topic B", "http://b", "src", "sum", "2026-01-01")
    research = ResearchResult("t", "s", [], ["f"], [], 0.8)
    verified = [VerifiedFact("f", True, 0.9, "")]

    # Just test scoring two and picking best directly
    s1 = sel.score(c1, research, verified)
    s2 = sel.score(c2, research, verified)
    assert s2.final > s1.final


def test_duplicate_penalty_exact():
    mock_db = MagicMock()
    mock_db.fetchall.return_value = [{"topic": "Exact Same Topic"}]
    p = MockProvider([json.dumps({
        "trend": 0.5, "interest": 0.5, "novelty": 0.5,
        "visual": 0.5, "shorts": 0.5, "source_quality": 0.5
    })])
    sel = TopicSelector(p, db=mock_db)
    c = TopicCandidate("Exact Same Topic", "http://x", "src", "sum", "2026-01-01")
    score = sel.score(c, ResearchResult("t","s",[],["f"],[],0.8), [])
    assert score.duplicate_penalty == 1.0
    assert score.final == 0.0


# Integration test: discover_candidates is network-bound; skip in unit tests
# @pytest.mark.integration
# def test_discover_candidates():
#     cands = discover_candidates(max_per_source=2)
#     assert isinstance(cands, list)
#     assert all(isinstance(c, TopicCandidate) for c in cands)


# --- feed URL and all-feeds-failed warning tests -----------------------------

def test_feed_urls_updated():
    """Verify the dead feed URLs have been replaced with working ones."""
    from app.research.sources import FEEDS
    feed_names = {f["name"] for f in FEEDS}
    assert "google_ai_blog" in feed_names
    assert "anthropic_news" in feed_names
    # google_ai_blog should now use research.google/blog/rss with generic parser
    google_feed = next(f for f in FEEDS if f["name"] == "google_ai_blog")
    assert google_feed["url"] == "https://research.google/blog/rss"
    assert google_feed["parser"] == "parse_generic_rss"
    # anthropic_news should use the community mirror
    anthropic_feed = next(f for f in FEEDS if f["name"] == "anthropic_news")
    assert anthropic_feed["url"] == "https://tim-hilde.github.io/anthropic-rss/rss.xml"
    assert anthropic_feed["parser"] == "parse_generic_rss"


def test_all_feeds_failed_warning(caplog):
    """When every feed fails, a distinct WARNING is logged."""
    import logging
    from app.research.sources import discover_candidates, FEEDS
    from unittest.mock import patch

    caplog.set_level(logging.WARNING, logger="app.research.sources")

    # In tests: no YOUTUBE_API_KEY -> trending uses scrape, which catches all exceptions
    # and returns [], so it never "fails" (doesn't raise). Only RSS feeds can fail.
    # So total failable sources = len(FEEDS) only.
    with patch("app.research.sources._fetch", side_effect=Exception("404 Not Found")):
        result = discover_candidates(max_per_source=2)
    assert result == []
    total_sources = len(FEEDS)  # trending scrape never raises in tests
    all_failed_logs = [r for r in caplog.records
                       if "all" in r.message.lower() and "failed" in r.message.lower()]
    assert len(all_failed_logs) == 1
    assert "all" in all_failed_logs[0].message
    assert str(total_sources) in all_failed_logs[0].message

    # Partial failures should NOT trigger this warning - they log per-source
    caplog.clear()
    with patch("app.research.sources._fetch") as mock_fetch:
        mock_fetch.side_effect = [
            '<rss><channel><item><title>OK</title><link>http://ok</link></item></channel></rss>',
            Exception("404 Not Found"),
        ]
        discover_candidates(max_per_source=2)
    partial_all_failed = [r for r in caplog.records
                          if "all" in r.message.lower() and "failed" in r.message.lower()]
    assert len(partial_all_failed) == 0
"""Duplicate detection using text similarity and hashing.

Compares new content (topic, script, title, visual concepts) against
historical content in the database. Rejects highly similar content.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from app.storage.database import Database
from app.utils.hashing import normalized_hash, sha256_text
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Similarity thresholds
TOPIC_SIM_THRESHOLD = 0.85
SCRIPT_SIM_THRESHOLD = 0.80
TITLE_SIM_THRESHOLD = 0.85


@dataclass
class DuplicateCheckResult:
    is_duplicate: bool
    reason: str
    similar_items: list[dict]  # {type, id, similarity, content_preview}


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity on word tokens."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def check_topic_duplicate(db: Database, topic: str, threshold: float = TOPIC_SIM_THRESHOLD) -> Optional[dict]:
    """Check if topic is too similar to past topics."""
    rows = db.fetchall("SELECT id, topic FROM topics WHERE final_score IS NOT NULL")
    topic_norm = normalized_hash(topic)
    for row in rows:
        past = row["topic"]
        if normalized_hash(past) == topic_norm:
            return {"type": "topic", "id": row["id"], "similarity": 1.0, "content": past[:100]}
        sim = jaccard_similarity(topic, past)
        if sim >= threshold:
            return {"type": "topic", "id": row["id"], "similarity": sim, "content": past[:100]}
    return None


def check_script_duplicate(db: Database, script: str, threshold: float = SCRIPT_SIM_THRESHOLD) -> Optional[dict]:
    """Check if script is too similar to past scripts."""
    rows = db.fetchall("SELECT id, content FROM scripts WHERE score IS NOT NULL")
    script_norm = normalized_hash(script)
    for row in rows:
        past = row["content"]
        if normalized_hash(past) == script_norm:
            return {"type": "script", "id": row["id"], "similarity": 1.0, "content": past[:100]}
        sim = jaccard_similarity(script, past)
        if sim >= threshold:
            return {"type": "script", "id": row["id"], "similarity": sim, "content": past[:100]}
    return None


def check_title_duplicate(db: Database, title: str, threshold: float = TITLE_SIM_THRESHOLD) -> Optional[dict]:
    """Check if title is too similar to past video titles."""
    rows = db.fetchall("SELECT id, title FROM videos WHERE youtube_video_id IS NOT NULL")
    for row in rows:
        past = row["title"] or ""
        sim = jaccard_similarity(title, past)
        if sim >= threshold:
            return {"type": "title", "id": row["id"], "similarity": sim, "content": past[:100]}
    return None


def check_visual_concept_duplicate(db: Database, visual_plan_json: str, threshold: float = 0.75) -> Optional[dict]:
    """Check if visual plan (scene queries) is too similar."""
    try:
        plan = json.loads(visual_plan_json)
        queries = " ".join(s.get("visual_query", "") for s in plan.get("scenes", []))
        queries_norm = normalized_hash(queries)
    except Exception:
        return None

    rows = db.fetchall("SELECT id, topic FROM topics WHERE final_score IS NOT NULL")
    for row in rows:
        # We don't store visual plans directly, so check topic similarity as proxy
        sim = jaccard_similarity(queries, row["topic"])
        if sim >= threshold:
            return {"type": "visual", "id": row["id"], "similarity": sim, "content": row["topic"][:100]}
    return None


def run_duplicate_checks(
    db: Database,
    topic: str,
    script: str,
    title: str,
    visual_plan_json: str,
    job_id: Optional[str] = None,
) -> DuplicateCheckResult:
    """Run all duplicate checks, return combined result."""
    similar = []

    for check_fn, label in [
        (lambda: check_topic_duplicate(db, topic), "topic"),
        (lambda: check_script_duplicate(db, script), "script"),
        (lambda: check_title_duplicate(db, title), "title"),
        (lambda: check_visual_concept_duplicate(db, visual_plan_json), "visual"),
    ]:
        result = check_fn()
        if result:
            similar.append(result)
            logger.warning(f"duplicate detected: {label} (sim={result['similarity']:.2f})",
                           extra={"job_id": job_id, "stage": "duplicate_check", "status": "duplicate"})

    is_dup = len(similar) > 0
    reason = "; ".join(f"{s['type']} sim={s['similarity']:.2f}" for s in similar) if similar else "no duplicates"

    return DuplicateCheckResult(
        is_duplicate=is_dup,
        reason=reason,
        similar_items=similar,
    )
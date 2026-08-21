"""Visual scene planner: breaks a script into timed scenes with visual queries.

Output JSON matches the spec (section 15):
{
  "duration": 42,
  "scenes": [
    {"start": 0, "end": 4, "visual_query": "...", "visual_type": "image", "motion": "zoom_in"},
    ...
  ]
}

Every scene must support the narration. No unrelated stock footage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional

from app.ai.prompts import render
from app.ai.provider import LLMProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Scene:
    start: float
    end: float
    visual_query: str
    visual_type: str  # "image" | "video" | "text" | "graphic"
    motion: str       # "zoom_in" | "zoom_out" | "pan" | "static" | "ken_burns"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VisualPlan:
    duration: float
    scenes: list[Scene]

    def to_json(self) -> str:
        return json.dumps({"duration": self.duration, "scenes": [s.to_dict() for s in self.scenes]})


class VisualPlanner:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def plan(self, script: str, topic: str, duration_estimate: float,
             job_id: Optional[str] = None) -> VisualPlan:
        prompt = render("visual_plan", script=script, topic=topic)
        result = self.provider.generate_json(prompt, temperature=0.4)

        duration = float(result.get("duration", duration_estimate))
        scenes_data = result.get("scenes", [])

        scenes = []
        for s in scenes_data:
            scenes.append(Scene(
                start=float(s.get("start", 0)),
                end=float(s.get("end", 0)),
                visual_query=str(s.get("visual_query", "")),
                visual_type=str(s.get("visual_type", "image")),
                motion=str(s.get("motion", "static")),
            ))

        # Validate: scenes should cover duration, no gaps > 1s
        if scenes:
            scenes.sort(key=lambda s: s.start)
            # Ensure first scene starts at 0
            if scenes[0].start > 0.5:
                scenes[0].start = 0
            # Ensure last scene ends at duration
            if abs(scenes[-1].end - duration) > 1.0:
                scenes[-1].end = duration

        plan = VisualPlan(duration=duration, scenes=scenes)
        logger.info(f"visual plan: {len(scenes)} scenes, {duration:.1f}s",
                    extra={"job_id": job_id, "stage": "visual_plan", "status": "done"})
        return plan
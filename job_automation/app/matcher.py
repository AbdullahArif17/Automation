from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
from pathlib import Path

try:
    from app.fetcher import JobPosting
except ImportError:
    from job_automation.app.fetcher import JobPosting

# Hard location disqualifiers for remote international candidates
LOCATION_DISQUALIFIERS = [
    r"\bus only\b",
    r"\bu\.s\. only\b",
    r"\bunited states only\b",
    r"\bcanada only\b",
    r"\beu only\b",
    r"\buk only\b",
    r"\bsecurity clearance\b",
    r"\bus citizen\b",
    r"\bgreen card\b",
    r"\bno visa sponsorship\b",
    r"\bmust reside in\b",
    r"\bmust be based in the us\b",
    r"\bon-site\b",
    r"\bhybrid\b",
]

# Tech keywords that indicate relevant developer roles
RELEVANT_ROLE_KEYWORDS = [
    "developer",
    "engineer",
    "full stack",
    "full-stack",
    "frontend",
    "front end",
    "front-end",
    "backend",
    "back end",
    "back-end",
    "software",
    "web",
    "python",
    "javascript",
    "typescript",
    "react",
    "next.js",
    "fastapi",
    "ai",
    "machine learning",
    "rag",
    "llm",
]

# Roles that are definitely not software engineering
EXCLUDED_ROLES = [
    "sales",
    "marketing",
    "accountant",
    "recruiter",
    "hr",
    "manager",
    "director",
    "vp",
    "legal",
    "customer support",
    "executive assistant",
    "content writer",
]


@dataclass
class MatchResult:
    job_id: str
    match_score: float
    is_eligible: bool
    rejection_reason: str = ""
    matched_skills: List[str] = None
    summary_reason: str = ""

    def __post_init__(self):
        if self.matched_skills is None:
            self.matched_skills = []


class JobMatcher:
    def __init__(self, profile_path: str = "config/profile.yaml", gemini_api_key: Optional[str] = None):
        self.profile = self._load_profile(profile_path)
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            for env_candidate in [Path(".env"), Path("/home/ubuntu/JobAutomation/.env"), Path("/home/ubuntu/Automation/.env")]:
                if env_candidate.exists():
                    for line in env_candidate.read_text(encoding="utf-8").splitlines():
                        if line.startswith("GEMINI_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                    if self.api_key:
                        break

    def _load_profile(self, path_str: str) -> Dict:
        path = Path(path_str)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def pre_filter(self, job: JobPosting) -> tuple[bool, str]:
        """Fast rule-based filter before calling LLM."""
        title_lower = job.title.lower()
        desc_lower = job.description.lower()
        loc_lower = (job.location or "").lower()

        # 1. Title exclusion
        for exc in EXCLUDED_ROLES:
            if re.search(r"\b" + re.escape(exc) + r"\b", title_lower) and not any(
                k in title_lower for k in ["engineer", "developer", "software"]
            ):
                return False, f"Non-engineering title: {job.title}"

        # 2. Must match at least one relevant tech keyword
        if not any(k in title_lower for k in RELEVANT_ROLE_KEYWORDS):
            return False, f"Title not matching developer keywords: {job.title}"

        # 3. Location disqualifiers
        full_text = f"{loc_lower} {desc_lower}"
        for disq in LOCATION_DISQUALIFIERS:
            if re.search(disq, full_text):
                # Exception: if it explicitly says "or Worldwide" / "open to remote anywhere"
                if "anywhere" not in loc_lower and "worldwide" not in loc_lower:
                    return False, f"Location restriction: matched pattern '{disq}'"

        return True, "Passed pre-filter"

    def evaluate_job(self, job: JobPosting) -> MatchResult:
        """Evaluates job posting match against Abdullah Arif's profile."""
        passes, reason = self.pre_filter(job)
        if not passes:
            return MatchResult(
                job_id=job.job_id,
                match_score=0.0,
                is_eligible=False,
                rejection_reason=reason,
            )

        # If Gemini API key is available, use LLM evaluation for deep matching
        if self.api_key:
            return self._llm_evaluate(job)

        # Fallback heuristic scoring if no LLM key
        return self._heuristic_evaluate(job)

    def _heuristic_evaluate(self, job: JobPosting) -> MatchResult:
        text = f"{job.title} {job.description}".lower()
        core_skills = [
            "python",
            "fastapi",
            "next.js",
            "react",
            "typescript",
            "javascript",
            "tailwind",
            "postgresql",
            "supabase",
            "rest",
            "api",
            "docker",
            "ai",
            "llm",
            "rag",
            "vector",
        ]
        matched = [s for s in core_skills if s in text]
        score = min(100.0, len(matched) * 12.0 + 20.0)

        is_eligible = score >= 70.0
        return MatchResult(
            job_id=job.job_id,
            match_score=round(score, 1),
            is_eligible=is_eligible,
            rejection_reason="" if is_eligible else f"Match score {score:.1f}% below 70% threshold",
            matched_skills=matched,
            summary_reason=f"Matched skills: {', '.join(matched)}",
        )

    def _llm_evaluate(self, job: JobPosting) -> MatchResult:
        """Uses Gemini Flash to analyze alignment between CV and job description."""
        prompt = f"""
You are an expert technical recruiter assessing whether this job posting is a suitable match for the candidate.

CANDIDATE PROFILE:
Name: Abdullah Arif
Role: Full-Stack Developer
Core Skills: Next.js, React, TypeScript, JavaScript, Python, FastAPI, REST APIs, AI/LLM Integration (Gemini API, RAG, Qdrant Vector Search), Supabase, PostgreSQL, Tailwind CSS, Docker, Linux.
Location: Karachi, Pakistan (Seeking Remote Worldwide / Global Contractor roles)
Experience: Early-career / Mid-level full-stack engineer who builds and ships end-to-end production web & AI apps.

JOB POSTING:
Title: {job.title}
Company: {job.company}
Location: {job.location}
Description (truncated):
{job.description[:2500]}

TASK:
1. Determine if the candidate has the required tech stack (Next.js/React, TypeScript, Python/FastAPI, REST APIs, or AI/LLM).
2. Confirm the candidate can apply as an international remote contractor from Pakistan (disqualify if strictly US/EU citizenship or on-site presence is required).
3. Provide a match score from 0 to 100.
4. Set "is_eligible" to true ONLY if match score >= 75 AND location is remote/contractor-friendly.

Return ONLY a valid JSON object:
{{
  "match_score": 85,
  "is_eligible": true,
  "matched_skills": ["TypeScript", "Next.js", "Python"],
  "rejection_reason": "",
  "summary_reason": "Strong alignment with full-stack TypeScript/Next.js and Python backend requirements."
}}
"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            # Clean markdown code blocks if any
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            data = json.loads(raw_text)
            return MatchResult(
                job_id=job.job_id,
                match_score=float(data.get("match_score", 0)),
                is_eligible=bool(data.get("is_eligible", False)),
                rejection_reason=data.get("rejection_reason", ""),
                matched_skills=data.get("matched_skills", []),
                summary_reason=data.get("summary_reason", ""),
            )
        except Exception as e:
            print(f"[Matcher] LLM evaluation error ({e}); falling back to heuristic")
            return self._heuristic_evaluate(job)

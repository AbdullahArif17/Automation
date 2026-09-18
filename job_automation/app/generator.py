from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Dict, Optional

try:
    from app.fetcher import JobPosting
except ImportError:
    from job_automation.app.fetcher import JobPosting


class ApplicationGenerator:
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

    def generate_cover_letter(self, job: JobPosting) -> str:
        """Generates a concise, impactful 2-paragraph cover letter for the role."""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = f"""
Write a professional, crisp, and compelling 2-paragraph cover letter for Abdullah Arif applying to:
Role: {job.title}
Company: {job.company}

Candidate Summary:
Full-Stack Developer skilled in Next.js, React, TypeScript, FastAPI, Python, and AI/LLM integration (RAG pipelines, Qdrant vector search).
Notable projects:
1. AI-Powered RAG Chatbot (FastAPI, Qdrant, Gemini API, Supabase, HuggingFace Spaces)
2. Taskflow AI (Next.js, FastAPI, PostgreSQL, Vercel)
3. Personal AI Employee / Automation Agent (Playwright, Gemini API, LinkedIn API)
Links: Portfolio (https://my-portfolio-nine-chi-62.vercel.app/), GitHub (https://github.com/AbdullahArif17)
Work Mode: Available immediately for remote worldwide / contractor work.

Tone: Confident, technical, concise, no fluff or generic corporate buzzwords. Keep it strictly under 180 words.
Do not include address headers or placeholders like [Date] or [Hiring Manager Name]. Start directly with "Dear {job.company} Team,".
"""
                resp = model.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                print(f"[Generator] LLM cover letter error ({e}); using template")

        # High-quality fallback template
        return (
            f"Dear {job.company} Team,\n\n"
            f"I am writing to express my strong enthusiasm for the {job.title} role. As a Full-Stack Developer "
            f"specializing in Next.js, TypeScript, and FastAPI, I build and independently deploy production-ready web platforms "
            f"and AI systems. My recent work includes architecting an AI-Powered RAG Chatbot featuring a vector search pipeline "
            f"(FastAPI, Qdrant, Supabase) and building Taskflow AI, an end-to-end task management platform with Next.js and PostgreSQL.\n\n"
            f"I thrive in fast-paced environments where ownership and shipping speed are prioritized. Available to start immediately "
            f"as a remote developer, I would love the opportunity to contribute directly to {job.company}'s engineering goals. "
            f"You can explore my open-source code at github.com/AbdullahArif17 and view my interactive projects at "
            f"my-portfolio-nine-chi-62.vercel.app.\n\n"
            f"Best regards,\nAbdullah Arif\n+92 336 2725979 | abdullaharif893@gmail.com"
        )

    def answer_question(self, question: str, job: JobPosting) -> str:
        """Answers custom ATS application questions."""
        q_lower = question.lower()

        # Handle common standard questions
        std = self.profile.get("standard_answers", {})
        if "salary" in q_lower or "compensation" in q_lower:
            return std.get("salary_expectations", "Open to market rate for remote developers")
        if "notice" in q_lower or "start date" in q_lower or "when can you start" in q_lower:
            return std.get("notice_period", "Immediately available")
        if "sponsorship" in q_lower or "visa" in q_lower:
            return std.get("visa_sponsorship", "No visa sponsorship required (remote contractor)")
        if "authorized" in q_lower or "legally authorized" in q_lower:
            return std.get("authorized_to_work", "Yes, fully authorized for remote contractor roles")
        if "how did you hear" in q_lower:
            return std.get("hear_about_us", "Online Job Board / Careers Page")

        # LLM generated answer for technical / cultural questions
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = f"""
Candidate: Abdullah Arif (Full-Stack Developer: Next.js, TypeScript, FastAPI, Python, AI/LLM RAG)
Role: {job.title} at {job.company}

Job Application Question:
"{question}"

Provide a concise, direct, authentic answer (1 to 3 sentences maximum) representing Abdullah's background and projects.
Do not use bullet points or preamble.
"""
                resp = model.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                print(f"[Generator] LLM question error ({e})")

        return (
            "I bring strong hands-on experience building full-stack applications with Next.js, FastAPI, and PostgreSQL, "
            "as well as integrating AI/LLM pipelines. I take full ownership from architectural design to deployment."
        )

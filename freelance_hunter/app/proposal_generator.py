from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional
import yaml

try:
    from app.poller import FreelanceProject
except ImportError:
    from freelance_hunter.app.poller import FreelanceProject


class ProposalGenerator:
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

    def generate_proposal(self, project: FreelanceProject) -> str:
        """Generates a high-converting, tailored freelance proposal in seconds."""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = f"""
Write a high-converting, winning freelance proposal for Abdullah Arif.

CLIENT PROJECT:
Title: {project.title}
Platform: {project.source}
Budget: {project.budget_str}
Description:
{project.description[:1500]}

CANDIDATE STACK & RELEVANT PROOFS:
Name: Abdullah Arif
Role: Full-Stack & Python Automation Developer
Key Skills: Next.js, React, TypeScript, FastAPI, Python, Playwright web automation, Gemini API/RAG, Supabase, PostgreSQL.
Portfolio: https://my-portfolio-nine-chi-62.vercel.app/
GitHub: https://github.com/AbdullahArif17
Notable Projects:
- AI-Powered RAG Chatbot (FastAPI, Qdrant, Gemini API, Supabase)
- Personal AI Employee (Playwright browser automation, Gemini API, LinkedIn integration)
- Taskflow AI (Next.js, FastAPI, PostgreSQL)

UPWORK / FREELANCE PROPOSAL RULES:
1. DO NOT use generic greetings like "Dear Hiring Manager" or "I am a skilled developer with 5 years experience".
2. First sentence MUST directly address their specific technical problem or requirement.
3. Include 2-3 concise bullet points outlining the technical implementation.
4. Mention 1 relevant project from Abdullah's portfolio/GitHub as proof of work.
5. End with a crisp call to action proposing a quick chat or first milestone kickoff.
6. Keep the entire proposal under 150 words.
"""
                resp = model.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                print(f"[ProposalGenerator] LLM generation error ({e}); using template")

        # High-impact fallback proposal
        return (
            f"Hi,\n\n"
            f"I reviewed your requirements for \"{project.title}\" and can deliver this cleanly using "
            f"FastAPI, Python, and Next.js.\n\n"
            f"Here is how I will approach this:\n"
            f"• Implement clean, modular backend logic with async error handling and database persistence.\n"
            f"• Handle all integration/automation requirements reliably using battle-tested libraries.\n"
            f"• Provide clear documentation and assist with deployment.\n\n"
            f"You can view my live work and code on GitHub: github.com/AbdullahArif17 and my interactive "
            f"portfolio at my-portfolio-nine-chi-62.vercel.app.\n\n"
            f"I can start immediately and deliver the first milestone quickly. When would be a convenient time for a quick discussion?\n\n"
            f"Best,\nAbdullah Arif"
        )

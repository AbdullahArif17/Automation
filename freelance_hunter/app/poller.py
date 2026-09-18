from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from email.utils import parsedate_to_datetime


@dataclass
class FreelanceProject:
    project_id: str
    source: str
    title: str
    description: str
    url: str
    budget_str: str = ""
    skills: List[str] = field(default_factory=list)
    posted_at: Optional[datetime] = None


def fetch_freelancer_projects(queries: List[str], max_age_minutes: int = 60) -> List[FreelanceProject]:
    """Fetches fresh active projects from Freelancer.com public API."""
    projects: List[FreelanceProject] = []
    seen_ids = set()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

    for q in queries:
        encoded_q = urllib.parse.quote(q)
        url = f"https://www.freelancer.com/api/projects/0.1/projects/active?query={encoded_q}&limit=10&compact=true"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for p in data.get("result", {}).get("projects", []):
                    pid = str(p.get("id"))
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    title = p.get("title", "").strip()
                    desc = p.get("preview_description", "").strip()

                    # Require at least one core developer keyword
                    dev_keywords = [
                        "developer", "engineer", "python", "fastapi", "next.js", "react", 
                        "scraping", "scraper", "automation", "bot", "api", "full-stack", 
                        "full stack", "backend", "frontend", "ai", "rag", "llm", "script"
                    ]
                    combined_text = f"{title} {desc}".lower()
                    if not any(k in combined_text for k in dev_keywords):
                        continue

                    # Time check
                    submit_time = p.get("time_submitted")
                    posted_at = None
                    if submit_time:
                        posted_at = datetime.fromtimestamp(submit_time, tz=timezone.utc)
                    if posted_at and posted_at < cutoff:
                        continue
                    b_info = p.get("budget", {})
                    b_min = b_info.get("minimum")
                    b_max = b_info.get("maximum")
                    if b_min and b_max:
                        budget_str = f"${b_min:.0f} - ${b_max:.0f}"
                    elif b_min:
                        budget_str = f"From ${b_min:.0f}"
                    else:
                        budget_str = "Negotiable / Hourly"

                    seo_url = p.get("seo_url", pid)
                    project_url = f"https://www.freelancer.com/projects/{seo_url}"

                    projects.append(
                        FreelanceProject(
                            project_id=f"freelancer_{pid}",
                            source="Freelancer.com",
                            title=p.get("title", "").strip(),
                            description=p.get("preview_description", "").strip(),
                            url=project_url,
                            budget_str=budget_str,
                            skills=[],
                            posted_at=posted_at,
                        )
                    )
        except Exception as e:
            print(f"[Poller] Freelancer error for query '{q}': {e}")

    return projects


def fetch_wwr_contract_projects(max_age_minutes: int = 180) -> List[FreelanceProject]:
    """Fetches contract and freelance gigs from WeWorkRemotely RSS."""
    projects: List[FreelanceProject] = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    rss_url = "https://weworkremotely.com/categories/remote-programming-jobs.rss"

    try:
        req = urllib.request.Request(rss_url, headers={"User-Agent": "JobHunter/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            root = ET.fromstring(resp.read())
            for item in root.findall("./channel/item"):
                title_elem = item.find("title")
                link_elem = item.find("link")
                pub_date_elem = item.find("pubDate")
                desc_elem = item.find("description")

                if title_elem is None or link_elem is None:
                    continue

                full_title = title_elem.text or ""
                link = link_elem.text or ""

                pub_date = None
                if pub_date_elem is not None and pub_date_elem.text:
                    try:
                        pub_date = parsedate_to_datetime(pub_date_elem.text)
                    except Exception:
                        pass

                if pub_date and pub_date < cutoff:
                    continue

                raw_desc = desc_elem.text if desc_elem is not None and desc_elem.text else ""
                clean_desc = re.sub(r"<[^>]+>", " ", raw_desc)
                clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

                # Filter for contract or freelance mentions
                if not any(k in f"{full_title} {clean_desc}".lower() for k in ["contract", "freelance", "part-time", "contractor"]):
                    continue

                pid = link.split("/")[-1] if "/" in link else link

                projects.append(
                    FreelanceProject(
                        project_id=f"wwr_{pid}",
                        source="WeWorkRemotely",
                        title=full_title,
                        description=clean_desc[:1200],
                        url=link,
                        budget_str="Contract Rate",
                        skills=[],
                        posted_at=pub_date,
                    )
                )
    except Exception as e:
        print(f"[Poller] WWR error: {e}")

    return projects


def fetch_all_fresh_projects(queries: List[str], max_age_minutes: int = 60) -> List[FreelanceProject]:
    """Combines all live sources and sorts newest first."""
    all_projects: List[FreelanceProject] = []
    seen = set()

    for p in fetch_freelancer_projects(queries=queries, max_age_minutes=max_age_minutes):
        if p.project_id not in seen:
            seen.add(p.project_id)
            all_projects.append(p)

    for p in fetch_wwr_contract_projects(max_age_minutes=max_age_minutes * 2):
        if p.project_id not in seen:
            seen.add(p.project_id)
            all_projects.append(p)

    all_projects.sort(key=lambda x: x.posted_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return all_projects

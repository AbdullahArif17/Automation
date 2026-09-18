from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from email.utils import parsedate_to_datetime


@dataclass
class JobPosting:
    job_id: str
    title: str
    company: str
    url: str
    source: str
    description: str
    location: str = ""
    tags: List[str] = field(default_factory=list)
    posted_at: Optional[datetime] = None
    ats_type: str = "direct"  # 'greenhouse', 'lever', 'ashby', 'workable', 'direct'


def _clean_html(html_text: str) -> str:
    """Removes HTML tags and cleans up whitespace."""
    if not html_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_ats(url: str, description: str = "") -> str:
    url_lower = url.lower()
    desc_lower = description.lower()
    if "greenhouse.io" in url_lower or "boards.greenhouse.io" in desc_lower:
        return "greenhouse"
    if "lever.co" in url_lower or "jobs.lever.co" in desc_lower:
        return "lever"
    if "ashbyhq.com" in url_lower:
        return "ashby"
    if "workable.com" in url_lower:
        return "workable"
    return "direct"


def fetch_remotive_jobs(max_age_hours: int = 48) -> List[JobPosting]:
    jobs: List[JobPosting] = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    url = "https://remotive.com/api/remote-jobs?category=software-dev"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "JobHunter/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("jobs", []):
                pub_date_str = item.get("publication_date")
                pub_date = None
                if pub_date_str:
                    try:
                        clean_date_str = pub_date_str.replace("Z", "+00:00")
                        pub_date = datetime.fromisoformat(clean_date_str)
                        if pub_date.tzinfo is None:
                            pub_date = pub_date.replace(tzinfo=timezone.utc)
                    except Exception:
                        pass
                if pub_date and pub_date < cutoff:
                    continue

                raw_desc = item.get("description", "")
                clean_desc = _clean_html(raw_desc)
                job_url = item.get("url", "")
                ats = _detect_ats(job_url, raw_desc)
                job_id = f"remotive_{item.get('id')}"

                jobs.append(
                    JobPosting(
                        job_id=job_id,
                        title=item.get("title", "").strip(),
                        company=item.get("company_name", "").strip(),
                        url=job_url,
                        source="remotive",
                        description=clean_desc,
                        location=item.get("candidate_required_location", "Worldwide"),
                        tags=item.get("tags", []),
                        posted_at=pub_date,
                        ats_type=ats,
                    )
                )
    except Exception as e:
        print(f"[Fetcher] Remotive fetch error: {e}")
    return jobs


def fetch_remoteok_jobs(max_age_hours: int = 48) -> List[JobPosting]:
    jobs: List[JobPosting] = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    url = "https://remoteok.com/api?tag=dev"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data:
                if not isinstance(item, dict) or "position" not in item:
                    continue
                epoch = item.get("epoch")
                pub_date = None
                if epoch:
                    pub_date = datetime.fromtimestamp(epoch, tz=timezone.utc)
                if pub_date and pub_date < cutoff:
                    continue

                raw_desc = item.get("description", "")
                clean_desc = _clean_html(raw_desc)
                job_url = item.get("url", "")
                apply_url = item.get("apply_url", "") or job_url
                ats = _detect_ats(apply_url, raw_desc)
                job_id = f"remoteok_{item.get('id', hashlib.md5(job_url.encode()).hexdigest())}"

                jobs.append(
                    JobPosting(
                        job_id=job_id,
                        title=item.get("position", "").strip(),
                        company=item.get("company", "").strip(),
                        url=apply_url,
                        source="remoteok",
                        description=clean_desc,
                        location=item.get("location", "Worldwide"),
                        tags=item.get("tags", []),
                        posted_at=pub_date,
                        ats_type=ats,
                    )
                )
    except Exception as e:
        print(f"[Fetcher] RemoteOK fetch error: {e}")
    return jobs


def fetch_weworkremotely_jobs(max_age_hours: int = 48) -> List[JobPosting]:
    jobs: List[JobPosting] = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    rss_feeds = [
        "https://weworkremotely.com/categories/remote-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-full-stack-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-back-end-programming-jobs.rss",
    ]

    for feed_url in rss_feeds:
        try:
            req = urllib.request.Request(feed_url, headers={"User-Agent": "JobHunter/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
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

                    # WWR titles are usually "Company: Job Title"
                    company = ""
                    title = full_title
                    if ":" in full_title:
                        parts = full_title.split(":", 1)
                        company = parts[0].strip()
                        title = parts[1].strip()

                    raw_desc = desc_elem.text if desc_elem is not None and desc_elem.text else ""
                    clean_desc = _clean_html(raw_desc)
                    ats = _detect_ats(link, raw_desc)
                    job_id = f"wwr_{hashlib.md5(link.encode()).hexdigest()[:12]}"

                    jobs.append(
                        JobPosting(
                            job_id=job_id,
                            title=title,
                            company=company,
                            url=link,
                            source="weworkremotely",
                            description=clean_desc,
                            location="Worldwide",
                            tags=[],
                            posted_at=pub_date,
                            ats_type=ats,
                        )
                    )
        except Exception as e:
            print(f"[Fetcher] WWR feed {feed_url} error: {e}")

    return jobs


def fetch_all_fresh_jobs(max_age_hours: int = 48) -> List[JobPosting]:
    """Aggregates, deduplicates and sorts fresh jobs from all free public feeds."""
    all_jobs: List[JobPosting] = []
    seen_ids = set()

    for job in fetch_remotive_jobs(max_age_hours=max_age_hours):
        if job.job_id not in seen_ids:
            seen_ids.add(job.job_id)
            all_jobs.append(job)

    for job in fetch_remoteok_jobs(max_age_hours=max_age_hours):
        if job.job_id not in seen_ids:
            seen_ids.add(job.job_id)
            all_jobs.append(job)

    for job in fetch_weworkremotely_jobs(max_age_hours=max_age_hours):
        if job.job_id not in seen_ids:
            seen_ids.add(job.job_id)
            all_jobs.append(job)

    # Sort descending by posted date
    all_jobs.sort(key=lambda j: j.posted_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return all_jobs


if __name__ == "__main__":
    jobs = fetch_all_fresh_jobs(max_age_hours=48)
    print(f"Total fresh jobs found (<48h): {len(jobs)}")
    for j in jobs[:5]:
        print(f"[{j.source.upper()}] {j.title} @ {j.company} ({j.location}) -> {j.url[:60]}... [ATS: {j.ats_type}]")

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

try:
    from app.fetcher import JobPosting
    from app.generator import ApplicationGenerator
except ImportError:
    from job_automation.app.fetcher import JobPosting
    from job_automation.app.generator import ApplicationGenerator


class JobApplier:
    def __init__(
        self,
        profile_path: str = "config/profile.yaml",
        resume_path: str = "assets/resume.pdf",
        dry_run: bool = False,
    ):
        self.profile_path = Path(profile_path)
        self.resume_path = Path(resume_path).resolve()
        self.dry_run = dry_run
        self.profile = self._load_profile(self.profile_path)
        self.generator = ApplicationGenerator(profile_path=str(self.profile_path))

    def _load_profile(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def apply(self, job: JobPosting) -> Tuple[bool, str, str]:
        """
        Attempts to fill and submit the job application using Playwright.
        Returns: (success: bool, message: str, screenshot_path: str)
        """
        from playwright.sync_api import sync_playwright

        cand = self.profile.get("candidate", {})
        first_name = cand.get("first_name", "Abdullah")
        last_name = cand.get("last_name", "Arif")
        full_name = cand.get("name", f"{first_name} {last_name}")
        email = cand.get("email", "abdullaharif893@gmail.com")
        phone = cand.get("phone", "+923362725979")
        links = cand.get("links", {})
        linkedin = links.get("linkedin", "")
        github = links.get("github", "")
        portfolio = links.get("portfolio", "")

        cover_letter = self.generator.generate_cover_letter(job)
        screenshot_dir = Path("data/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = str(screenshot_dir / f"{job.job_id}.png")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                )
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    viewport={"width": 1280, "height": 900},
                )
                page = context.new_page()
                page.set_default_timeout(25000)

                print(f"[Applier] Navigating to: {job.url}")
                page.goto(job.url, wait_until="domcontentloaded")
                time.sleep(3)

                current_url = page.url.lower()

                # Detect if redirected to Greenhouse or Lever
                if "greenhouse.io" in current_url:
                    success, msg = self._fill_greenhouse(
                        page, first_name, last_name, email, phone, linkedin, github, portfolio, cover_letter, job
                    )
                elif "lever.co" in current_url:
                    success, msg = self._fill_lever(
                        page, full_name, email, phone, linkedin, github, portfolio, cover_letter, job
                    )
                else:
                    # Generic ATS or Careers page fallback
                    success, msg = self._fill_generic(
                        page, first_name, last_name, full_name, email, phone, linkedin, github, portfolio, cover_letter, job
                    )

                page.screenshot(path=screenshot_path)
                browser.close()
                return success, msg, screenshot_path

        except Exception as e:
            return False, f"Playwright automation error: {e}", ""

    def _fill_greenhouse(
        self, page, first_name, last_name, email, phone, linkedin, github, portfolio, cover_letter, job
    ) -> Tuple[bool, str]:
        """Fills Greenhouse application forms."""
        try:
            # Name fields
            if page.locator("#first_name").is_visible():
                page.fill("#first_name", first_name)
            if page.locator("#last_name").is_visible():
                page.fill("#last_name", last_name)

            # Email & Phone
            if page.locator("#email").is_visible():
                page.fill("#email", email)
            if page.locator("#phone").is_visible():
                page.fill("#phone", phone)

            # Resume upload
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0 and self.resume_path.exists():
                file_input.first.set_input_files(str(self.resume_path))
                time.sleep(2)

            # LinkedIn / Websites
            for inp in page.locator("input[type='text']").all():
                try:
                    label_text = inp.evaluate("el => el.closest('label')?.innerText || el.getAttribute('placeholder') || ''").lower()
                    if "linkedin" in label_text and not inp.input_value():
                        inp.fill(linkedin)
                    elif ("github" in label_text or "git" in label_text) and not inp.input_value():
                        inp.fill(github)
                    elif ("portfolio" in label_text or "website" in label_text) and not inp.input_value():
                        inp.fill(portfolio)
                except Exception:
                    continue

            # Cover letter textarea
            cl_area = page.locator("textarea")
            if cl_area.count() > 0:
                for area in cl_area.all():
                    try:
                        lbl = area.evaluate("el => el.closest('label')?.innerText || ''").lower()
                        if "cover" in lbl or "letter" in lbl or "additional" in lbl or "note" in lbl:
                            area.fill(cover_letter)
                            break
                    except Exception:
                        continue

            if self.dry_run:
                return True, "DRY_RUN: Form filled successfully (not submitted)"

            # Submit
            submit_btn = page.locator("#submit_app, button[type='submit'], input[type='submit']")
            if submit_btn.count() > 0:
                submit_btn.first.click()
                time.sleep(4)
                return True, "Greenhouse application submitted"
            return False, "Submit button not found on Greenhouse form"

        except Exception as e:
            return False, f"Greenhouse filling error: {e}"

    def _fill_lever(
        self, page, full_name, email, phone, linkedin, github, portfolio, cover_letter, job
    ) -> Tuple[bool, str]:
        """Fills Lever application forms."""
        try:
            # If on the posting description page, click 'Apply for this job'
            apply_btn = page.locator("a.postings-btn, a[href*='/apply']")
            if apply_btn.count() > 0 and "/apply" not in page.url:
                apply_btn.first.click()
                time.sleep(2)

            if page.locator("input[name='name']").is_visible():
                page.fill("input[name='name']", full_name)
            if page.locator("input[name='email']").is_visible():
                page.fill("input[name='email']", email)
            if page.locator("input[name='phone']").is_visible():
                page.fill("input[name='phone']", phone)
            if page.locator("input[name='org']").is_visible():
                page.fill("input[name='org']", "Software Developer")

            # Links
            if page.locator("input[name='urls[LinkedIn]']").is_visible():
                page.fill("input[name='urls[LinkedIn]']", linkedin)
            if page.locator("input[name='urls[GitHub]']").is_visible():
                page.fill("input[name='urls[GitHub]']", github)
            if page.locator("input[name='urls[Portfolio]']").is_visible():
                page.fill("input[name='urls[Portfolio]']", portfolio)

            # Resume upload
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0 and self.resume_path.exists():
                file_input.first.set_input_files(str(self.resume_path))
                time.sleep(2)

            # Additional info / Cover letter
            if page.locator("textarea[name='comments']").is_visible():
                page.fill("textarea[name='comments']", cover_letter)

            if self.dry_run:
                return True, "DRY_RUN: Lever form filled successfully (not submitted)"

            # Submit
            submit_btn = page.locator("button[type='submit'], button.template-btn-submit")
            if submit_btn.count() > 0:
                submit_btn.first.click()
                time.sleep(4)
                return True, "Lever application submitted"
            return False, "Submit button not found on Lever form"

        except Exception as e:
            return False, f"Lever filling error: {e}"

    def _fill_generic(
        self, page, first_name, last_name, full_name, email, phone, linkedin, github, portfolio, cover_letter, job
    ) -> Tuple[bool, str]:
        """Attempts best-effort fill on generic career forms."""
        try:
            # Check for 'Apply' button to open modal or application section
            apply_buttons = page.locator("a, button").filter(has_text=re.compile(r"^\s*Apply(\s+Now)?\s*$", re.I))
            if apply_buttons.count() > 0:
                try:
                    apply_buttons.first.click()
                    time.sleep(2)
                except Exception:
                    pass

            # Fill name fields
            name_inputs = page.locator("input[name*='name' i], input[id*='name' i], input[placeholder*='name' i]")
            if name_inputs.count() == 1:
                name_inputs.first.fill(full_name)
            elif name_inputs.count() > 1:
                name_inputs.nth(0).fill(first_name)
                name_inputs.nth(1).fill(last_name)

            # Email
            email_inputs = page.locator("input[type='email'], input[name*='email' i]")
            if email_inputs.count() > 0:
                email_inputs.first.fill(email)

            # Phone
            phone_inputs = page.locator("input[type='tel'], input[name*='phone' i]")
            if phone_inputs.count() > 0:
                phone_inputs.first.fill(phone)

            # Resume
            file_inputs = page.locator("input[type='file']")
            if file_inputs.count() > 0 and self.resume_path.exists():
                file_inputs.first.set_input_files(str(self.resume_path))
                time.sleep(2)

            if self.dry_run:
                return True, "DRY_RUN: Generic form filled (not submitted)"

            return True, "Generic form inputs populated"

        except Exception as e:
            return False, f"Generic form filling error: {e}"

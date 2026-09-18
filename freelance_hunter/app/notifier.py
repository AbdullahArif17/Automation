from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Dict, Optional

try:
    from app.poller import FreelanceProject
except ImportError:
    from freelance_hunter.app.poller import FreelanceProject


class AlertNotifier:
    def __init__(self, settings: Dict):
        self.settings = settings
        notif = settings.get("notifications", {})
        self.telegram = notif.get("telegram", {})
        self.discord = notif.get("discord", {})

    def send_alert(self, project: FreelanceProject, proposal: str) -> bool:
        """Sends rich alert to configured channels (Telegram / Discord)."""
        sent_any = False

        if self.telegram.get("enabled") and self.telegram.get("bot_token") and self.telegram.get("chat_id"):
            sent_any = self._send_telegram(project, proposal) or sent_any

        if self.discord.get("enabled") and self.discord.get("webhook_url"):
            sent_any = self._send_discord(project, proposal) or sent_any

        # If neither is enabled, log to console
        if not sent_any:
            print("\n" + "=" * 50)
            print(f"🔔 [NEW FREELANCE GIG ALERT] {project.title}")
            print(f"Source: {project.source} | Budget: {project.budget_str}")
            print(f"Apply Link: {project.url}")
            print("-" * 50)
            print("📝 Tailored Proposal (Ready to submit):")
            print(proposal)
            print("=" * 50 + "\n")

        return sent_any

    def _send_telegram(self, project: FreelanceProject, proposal: str) -> bool:
        bot_token = self.telegram["bot_token"]
        chat_id = self.telegram["chat_id"]
        text = (
            f"💼 *New Freelance Project*\n"
            f"📌 *{project.title}*\n"
            f"💰 *Budget:* {project.budget_str}\n"
            f"🌐 *Platform:* {project.source}\n"
            f"🔗 [View & Apply on {project.source}]({project.url})\n\n"
            f"📝 *Drafted Proposal (Tap to copy):*\n"
            f"```\n{proposal}\n```"
        )
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": False,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[Notifier] Telegram error: {e}")
            return False

    def _send_discord(self, project: FreelanceProject, proposal: str) -> bool:
        webhook_url = self.discord["webhook_url"]
        payload = {
            "content": f"🚨 **New Freelance Job Found!** 🚨\n**{project.title}** ({project.budget_str})\n{project.url}",
            "embeds": [
                {
                    "title": project.title,
                    "url": project.url,
                    "color": 3447003,
                    "fields": [
                        {"name": "Budget", "value": project.budget_str, "inline": True},
                        {"name": "Platform", "value": project.source, "inline": True},
                        {"name": "Tailored Proposal", "value": f"```{proposal}```", "inline": False},
                    ],
                }
            ],
        }
        try:
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "FreelanceHunter/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            print(f"[Notifier] Discord error: {e}")
            return False

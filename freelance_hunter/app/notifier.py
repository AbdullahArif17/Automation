from __future__ import annotations

import json
import smtplib
import urllib.parse
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional

try:
    from app.poller import FreelanceProject
except ImportError:
    from freelance_hunter.app.poller import FreelanceProject


class AlertNotifier:
    def __init__(self, settings: Dict):
        self.settings = settings
        notif = settings.get("notifications", {})
        self.whatsapp = notif.get("whatsapp", {})
        self.email = notif.get("email", {})
        self.telegram = notif.get("telegram", {})
        self.discord = notif.get("discord", {})

    def send_alert(self, project: FreelanceProject, proposal: str) -> bool:
        """Sends rich alert to configured channels (WhatsApp / Email / Telegram / Discord)."""
        sent_any = False

        if self.whatsapp.get("enabled") and self.whatsapp.get("api_key"):
            sent_any = self._send_whatsapp(project, proposal) or sent_any

        if self.email.get("enabled") and self.email.get("sender_password"):
            sent_any = self._send_email(project, proposal) or sent_any

        if self.telegram.get("enabled") and self.telegram.get("bot_token") and self.telegram.get("chat_id"):
            sent_any = self._send_telegram(project, proposal) or sent_any

        if self.discord.get("enabled") and self.discord.get("webhook_url"):
            sent_any = self._send_discord(project, proposal) or sent_any

        # If none enabled, print to terminal/logs
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

    def _send_whatsapp(self, project: FreelanceProject, proposal: str) -> bool:
        """Sends WhatsApp alert via CallMeBot free API."""
        phone = self.whatsapp.get("phone_number", "").replace("+", "").replace(" ", "").replace("-", "")
        apikey = self.whatsapp.get("api_key", "")
        text = (
            f"💼 *New Freelance Project*\n"
            f"📌 {project.title}\n"
            f"💰 Budget: {project.budget_str}\n"
            f"🌐 Platform: {project.source}\n"
            f"🔗 {project.url}\n\n"
            f"📝 *Proposal:*\n{proposal}"
        )
        encoded_text = urllib.parse.quote_plus(text)
        url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={encoded_text}&apikey={apikey}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FreelanceHunter/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[Notifier] WhatsApp error: {e}")
            return False

    def _send_email(self, project: FreelanceProject, proposal: str) -> bool:
        """Sends an HTML formatted email alert via SMTP (e.g. Gmail App Password)."""
        smtp_server = self.email.get("smtp_server", "smtp.gmail.com")
        smtp_port = int(self.email.get("smtp_port", 465))
        sender = self.email.get("sender_email", "")
        password = self.email.get("sender_password", "")
        recipient = self.email.get("recipient_email", sender)

        subject = f"💼 [Freelance Alert] {project.title} ({project.budget_str})"

        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="background-color: #f4f7f6; padding: 20px; border-radius: 8px;">
                <h2 style="color: #2c3e50; margin-top: 0;">New Freelance Project Found</h2>
                <p><strong>Project:</strong> {project.title}</p>
                <p><strong>Budget:</strong> <span style="color: #27ae60; font-weight: bold;">{project.budget_str}</span></p>
                <p><strong>Platform:</strong> {project.source}</p>
                <p><a href="{project.url}" style="background-color: #3498db; color: white; padding: 10px 18px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0;">View & Apply on {project.source}</a></p>
                <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                <h3 style="color: #34495e;">Tailored Winning Proposal (Ready to Copy):</h3>
                <div style="background: white; padding: 15px; border-left: 4px solid #3498db; border-radius: 4px; white-space: pre-wrap; font-family: monospace; font-size: 13px;">{proposal}</div>
            </div>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        msg.attach(MIMEText(html_content, "html"))

        try:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=12) as server:
                server.login(sender, password)
                server.sendmail(sender, recipient, msg.as_string())
            print(f"[Notifier] Email alert sent to {recipient}")
            return True
        except Exception as e:
            print(f"[Notifier] Email error: {e}")
            return False

    def _send_telegram(self, project: FreelanceProject, proposal: str) -> bool:
        bot_token = self.telegram.get("bot_token", "")
        chat_id = self.telegram.get("chat_id", "")
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
        webhook_url = self.discord.get("webhook_url", "")
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

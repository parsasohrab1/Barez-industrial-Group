"""Notification Service با Escalation Matrix
کانال‌ها: Telegram، Email (SMTP)، SMS (generic provider)
"""
import logging
import smtplib
from email.mime.text import MIMEText
from enum import Enum
from typing import Optional

import httpx

from tvdi_industrial.config import get_settings

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


ESCALATION = {
    AlertSeverity.INFO: ["operator"],
    AlertSeverity.WARNING: ["operator", "quality_manager"],
    AlertSeverity.CRITICAL: ["operator", "quality_manager", "maintenance", "admin"],
}


class NotificationService:
    def __init__(self, settings=None):
        self.s = settings or get_settings()

    async def send(self, severity: AlertSeverity, title: str, body: str, channels=None):
        channels = channels or ["telegram", "email"]
        results = {}
        if "telegram" in channels:
            results["telegram"] = await self._telegram(title, body)
        if "email" in channels:
            results["email"] = self._email(title, body)
        if "sms" in channels:
            results["sms"] = await self._sms(title, body)
        logger.info("alert sent severity=%s channels=%s results=%s",
                    severity.value, channels, results)
        return results

    async def _telegram(self, title: str, body: str) -> bool:
        if not self.s.telegram_bot_token:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.post(
                    f"https://api.telegram.org/bot{self.s.telegram_bot_token}/sendMessage",
                    json={"chat_id": self.s.telegram_chat_id,
                          "text": f"*{title}*\n{body}", "parse_mode": "Markdown"},
                )
                return r.status_code == 200
        except Exception as exc:
            logger.error("telegram failed: %s", exc)
            return False

    def _email(self, title: str, body: str) -> bool:
        if not self.s.smtp_host:
            return False
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = title
            msg["From"] = self.s.smtp_user
            msg["To"] = self.s.smtp_user
            with smtplib.SMTP(self.s.smtp_host, self.s.smtp_port) as srv:
                srv.starttls()
                if self.s.smtp_user:
                    srv.login(self.s.smtp_user, self.s.smtp_password)
                srv.send_message(msg)
            return True
        except Exception as exc:
            logger.error("email failed: %s", exc)
            return False

    async def _sms(self, title: str, body: str) -> bool:
        if not self.s.sms_api_key:
            return False
        # اینجا آداپتور کاوه‌نگار/پیامک‌رسان قرار می‌گیرد
        logger.info("sms stub: %s — %s", title, body)
        return True

    def escalate(self, severity: AlertSeverity) -> list[str]:
        return ESCALATION.get(severity, ["admin"])

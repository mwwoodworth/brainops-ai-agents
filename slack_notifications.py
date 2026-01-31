#!/usr/bin/env python3
"""
Slack Notifications for BrainOps AI OS
======================================
Sends critical alerts and system notifications to Slack.

Part of BrainOps OS Total Completion Protocol.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

from safe_task import create_safe_task

# Slack webhook URL from environment
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#brainops-alerts")


class AlertColor(Enum):
    """Slack attachment colors for different severity levels"""
    INFO = "#2196F3"      # Blue
    WARNING = "#FFC107"   # Amber
    ERROR = "#FF5722"     # Deep Orange
    CRITICAL = "#F44336"  # Red
    SUCCESS = "#4CAF50"   # Green


@dataclass
class SlackMessage:
    """Represents a Slack message to send"""
    text: str
    channel: Optional[str] = None
    attachments: Optional[list[dict]] = None
    blocks: Optional[list[dict]] = None
    thread_ts: Optional[str] = None


class SlackNotifier:
    """
    Sends notifications to Slack via webhook.

    Features:
    - Rate limiting to prevent spam
    - Message formatting for alerts
    - Async sending for non-blocking operation
    """

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or SLACK_WEBHOOK_URL
        self.enabled = bool(self.webhook_url)
        self._rate_limit_window = 60  # seconds
        self._max_messages_per_window = 10
        self._message_times: list[datetime] = []

        if self.enabled:
            logger.info("Slack notifications enabled")
        else:
            logger.warning("Slack notifications disabled (no SLACK_WEBHOOK_URL)")

    def _is_rate_limited(self) -> bool:
        """Check if we're being rate limited"""
        now = datetime.now(timezone.utc)
        # Remove old timestamps
        self._message_times = [
            t for t in self._message_times
            if (now - t).total_seconds() < self._rate_limit_window
        ]
        return len(self._message_times) >= self._max_messages_per_window

    def _record_message(self):
        """Record a message being sent"""
        self._message_times.append(datetime.now(timezone.utc))

    async def send_message(self, message: SlackMessage) -> bool:
        """Send a message to Slack"""
        if not self.enabled:
            logger.debug("Slack notifications disabled, skipping message")
            return False

        if self._is_rate_limited():
            logger.warning("Rate limited, skipping Slack message")
            return False

        payload = {"text": message.text}
        if message.channel:
            payload["channel"] = message.channel
        if message.attachments:
            payload["attachments"] = message.attachments
        if message.blocks:
            payload["blocks"] = message.blocks
        if message.thread_ts:
            payload["thread_ts"] = message.thread_ts

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload
                )
                response.raise_for_status()
                self._record_message()
                logger.info(f"Slack message sent: {message.text[:50]}...")
                return True
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False

    def send_message_sync(self, message: SlackMessage) -> bool:
        """
        Synchronous wrapper for send_message.

        Note: if called from inside a running asyncio loop, this cannot block.
        In that case we schedule a safe background task and return True to mean
        "scheduled" (the actual send result will be logged by the task).
        """
        try:
            loop = asyncio.get_running_loop()
            create_safe_task(self.send_message(message), name="slack.send_message")
            return True
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(self.send_message(message))

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        fields: Optional[dict[str, str]] = None,
        url: Optional[str] = None
    ) -> bool:
        """Send an alert notification to Slack"""

        # Map severity to color
        color_map = {
            "info": AlertColor.INFO.value,
            "warning": AlertColor.WARNING.value,
            "error": AlertColor.ERROR.value,
            "critical": AlertColor.CRITICAL.value,
            "success": AlertColor.SUCCESS.value
        }
        color = color_map.get(severity.lower(), AlertColor.INFO.value)

        # Build attachment
        attachment = {
            "color": color,
            "title": title,
            "text": message,
            "footer": "BrainOps AI OS",
            "ts": int(datetime.now(timezone.utc).timestamp())
        }

        if url:
            attachment["title_link"] = url

        if fields:
            attachment["fields"] = [
                {"title": k, "value": str(v), "short": True}
                for k, v in fields.items()
            ]

        slack_message = SlackMessage(
            text=f"*{severity.upper()}*: {title}",
            attachments=[attachment]
        )

        return await self.send_message(slack_message)

    def send_alert_sync(
        self,
        title: str,
        message: str,
        severity: str = "info",
        fields: Optional[dict[str, str]] = None,
        url: Optional[str] = None
    ) -> bool:
        """
        Synchronous wrapper for send_alert.

        Note: if called from inside a running asyncio loop, this cannot block.
        In that case we schedule a safe background task and return True to mean
        "scheduled" (the actual send result will be logged by the task).
        """
        try:
            loop = asyncio.get_running_loop()
            create_safe_task(
                self.send_alert(title, message, severity, fields, url),
                name="slack.send_alert",
            )
            return True
        except RuntimeError:
            return asyncio.run(
                self.send_alert(title, message, severity, fields, url)
            )


# Singleton instance
_notifier: Optional[SlackNotifier] = None


def get_slack_notifier() -> SlackNotifier:
    """Get the singleton Slack notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = SlackNotifier()
    return _notifier


def create_alert_handler():
    """
    Create an alert handler for the AlertingSystem.

    Usage:
        from ai_system_enhancements import get_alerting
        from slack_notifications import create_alert_handler

        alerting = get_alerting()
        alerting.add_handler(create_alert_handler())
    """
    notifier = get_slack_notifier()

    def handler(alert):
        """Handle alerts by sending to Slack"""
        # Only send WARNING, ERROR, and CRITICAL to Slack
        if alert.severity.value in ("warning", "error", "critical"):
            notifier.send_alert_sync(
                title=f"Alert: {alert.alert_type}",
                message=alert.message,
                severity=alert.severity.value,
                fields={
                    "Module": alert.module,
                    "Metric": alert.metric,
                    "Current Value": str(alert.current_value),
                    "Threshold": str(alert.threshold),
                    "Alert ID": alert.id
                }
            )

    return handler


# Initialize alerting integration
def setup_slack_alerting():
    """
    Setup Slack alerting integration with the AlertingSystem.

    Call this during app startup to enable Slack notifications.
    """
    if not SLACK_WEBHOOK_URL:
        logger.info("Slack alerting not configured (no SLACK_WEBHOOK_URL)")
        return False

    try:
        from ai_system_enhancements import get_alerting
        alerting = get_alerting()
        handler = create_alert_handler()
        alerting.add_handler(handler)
        logger.info("Slack alerting handler registered")
        return True
    except Exception as e:
        logger.error(f"Failed to setup Slack alerting: {e}")
        return False


# Convenience functions for direct notifications
async def notify_critical(title: str, message: str, **kwargs) -> bool:
    """Send a critical notification to Slack"""
    return await get_slack_notifier().send_alert(
        title, message, severity="critical", **kwargs
    )


async def notify_warning(title: str, message: str, **kwargs) -> bool:
    """Send a warning notification to Slack"""
    return await get_slack_notifier().send_alert(
        title, message, severity="warning", **kwargs
    )


async def notify_success(title: str, message: str, **kwargs) -> bool:
    """Send a success notification to Slack"""
    return await get_slack_notifier().send_alert(
        title, message, severity="success", **kwargs
    )


async def notify_info(title: str, message: str, **kwargs) -> bool:
    """Send an info notification to Slack"""
    return await get_slack_notifier().send_alert(
        title, message, severity="info", **kwargs
    )


__all__ = [
    "SlackNotifier",
    "SlackMessage",
    "AlertColor",
    "get_slack_notifier",
    "create_alert_handler",
    "setup_slack_alerting",
    "notify_critical",
    "notify_warning",
    "notify_success",
    "notify_info"
]

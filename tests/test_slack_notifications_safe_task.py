import asyncio
import sys
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_send_message_sync_schedules_task_inside_running_loop(monkeypatch):
    # Import inside test so we use the current module code.
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

    from slack_notifications import SlackNotifier, SlackMessage

    class BoomNotifier(SlackNotifier):
        async def send_message(self, message: SlackMessage) -> bool:  # type: ignore[override]
            raise RuntimeError("boom")

    notifier = BoomNotifier(webhook_url="https://example.com/webhook")

    # When running inside a loop, this should not raise and should return a bool.
    assert notifier.send_message_sync(SlackMessage(text="hi")) is True

    # Give the scheduled task a chance to run.
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_send_alert_sync_schedules_task_inside_running_loop(monkeypatch):
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

    from slack_notifications import SlackNotifier

    class BoomNotifier(SlackNotifier):
        async def send_alert(self, *args, **kwargs) -> bool:  # type: ignore[override]
            raise RuntimeError("boom")

    notifier = BoomNotifier(webhook_url="https://example.com/webhook")
    assert notifier.send_alert_sync("t", "m", "info") is True
    await asyncio.sleep(0)

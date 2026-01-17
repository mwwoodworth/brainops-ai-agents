"""
Email Scheduler Daemon
======================
Background daemon that processes scheduled emails from ai_email_queue
and sends them via Resend (primary) or SendGrid (fallback) at the appropriate times.

Handles:
- Nurture campaign emails (drip sequences)
- Follow-up emails triggered by lead actions
- Scheduled outreach emails
- Retry logic for failed sends

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Email provider configuration (Resend is primary in prod)
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")

# SendGrid fallback configuration (optional)
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY', '')
SENDGRID_FROM_EMAIL = os.getenv('SENDGRID_FROM_EMAIL', 'noreply@brainops.ai')
SENDGRID_FROM_NAME = os.getenv('SENDGRID_FROM_NAME', 'BrainOps AI')

RESEND_RATE_LIMIT_PER_SECOND = float(os.getenv("RESEND_RATE_LIMIT_PER_SECOND", "2"))
_RESEND_MIN_INTERVAL = 1.0 / RESEND_RATE_LIMIT_PER_SECOND if RESEND_RATE_LIMIT_PER_SECOND > 0 else 0.0
_resend_rate_lock = asyncio.Lock()
_last_resend_request_at = 0.0

async def _throttle_resend() -> None:
    """Best-effort throttle to avoid hammering Resend rate limits."""
    global _last_resend_request_at
    if _RESEND_MIN_INTERVAL <= 0:
        return
    async with _resend_rate_lock:
        now = time.monotonic()
        elapsed = now - _last_resend_request_at
        if elapsed < _RESEND_MIN_INTERVAL:
            await asyncio.sleep(_RESEND_MIN_INTERVAL - elapsed)
        _last_resend_request_at = time.monotonic()


TEST_EMAIL_SUFFIXES = (".test", ".example", ".invalid")
TEST_EMAIL_TOKENS = ("@example.", "@test.", "@demo.", "@invalid.")
TEST_EMAIL_DOMAINS = ("test.com", "example.com", "localhost", "demo@roofing.com")

OUTBOUND_EMAIL_MODE = os.getenv("OUTBOUND_EMAIL_MODE", "disabled").strip().lower()


def _parse_csv_env(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


_OUTBOUND_EMAIL_ALLOWLIST_RECIPIENTS = _parse_csv_env(os.getenv("OUTBOUND_EMAIL_ALLOWLIST", ""))
_OUTBOUND_EMAIL_ALLOWLIST_DOMAINS = _parse_csv_env(os.getenv("OUTBOUND_EMAIL_ALLOWLIST_DOMAINS", ""))


def _is_allowlisted_recipient(recipient: str | None) -> bool:
    if not recipient:
        return False
    lowered = recipient.strip().lower()
    if lowered in _OUTBOUND_EMAIL_ALLOWLIST_RECIPIENTS:
        return True
    if "@" not in lowered:
        return False
    domain = lowered.split("@", 1)[1]
    if domain in _OUTBOUND_EMAIL_ALLOWLIST_DOMAINS:
        return True
    return any(domain.endswith(f".{allowed}") for allowed in _OUTBOUND_EMAIL_ALLOWLIST_DOMAINS)


def _is_test_recipient(recipient: str | None) -> bool:
    if not recipient:
        return True
    lowered = recipient.lower().strip()
    if any(lowered.endswith(suffix) for suffix in TEST_EMAIL_SUFFIXES):
        return True
    if any(token in lowered for token in TEST_EMAIL_TOKENS):
        return True
    return any(domain in lowered for domain in TEST_EMAIL_DOMAINS)


def _get_skip_reason(email: "EmailJob") -> str | None:
    # Global outbound safety rail (fail closed by default).
    if OUTBOUND_EMAIL_MODE != "live" and not _is_allowlisted_recipient(email.recipient):
        return "not_allowlisted" if OUTBOUND_EMAIL_MODE == "allowlist" else "outbound_disabled"

    if email.metadata.get("is_test") is True or email.metadata.get("test") is True:
        return "is_test_metadata"
    if _is_test_recipient(email.recipient):
        return "test_recipient"
    return None


@dataclass
class EmailJob:
    """Represents an email job from the queue"""
    id: str
    recipient: str
    subject: str
    body: str
    scheduled_for: datetime
    status: str
    metadata: dict[str, Any]
    created_at: datetime


class EmailSchedulerDaemon:
    """
    Background daemon that processes scheduled emails.

    Polls the ai_email_queue table for emails due to be sent
    and dispatches them via SendGrid.
    """

    def __init__(self, poll_interval: int = 30, batch_size: int = 50, max_retries: int = 3):
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._running = False
        self._stats = {
            "emails_sent": 0,
            "emails_failed": 0,
            "emails_retried": 0,
            "last_poll": None
        }

    async def start(self):
        """Start the email scheduler daemon"""
        if self._running:
            logger.warning("EmailSchedulerDaemon already running")
            return

        self._running = True
        logger.info("📧 EmailSchedulerDaemon started")

        # Start processing loop
        await self._process_loop()

    async def stop(self):
        """Stop the email scheduler daemon"""
        self._running = False
        logger.info("🛑 EmailSchedulerDaemon stopped")

    async def _process_loop(self):
        """Main processing loop"""
        poll_count = 0
        while self._running:
            try:
                poll_count += 1
                # Fetch due emails
                emails = await self._fetch_due_emails()

                if emails:
                    logger.info(f"📬 Processing {len(emails)} scheduled emails")
                    for email in emails:
                        await self._send_email(email)
                elif poll_count == 1 or poll_count % 60 == 0:
                    # Log periodically to show the daemon is alive
                    logger.info(f"📧 Email scheduler heartbeat - no pending emails (poll #{poll_count})")

                self._stats["last_poll"] = datetime.now(timezone.utc).isoformat()
                self._stats["poll_count"] = poll_count

            except Exception as e:
                logger.error(f"Error in email processing loop: {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)

    async def _fetch_due_emails(self) -> list[EmailJob]:
        """Fetch emails that are due to be sent"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()

            async with pool.acquire() as conn:
                # Lock and fetch due emails
                rows = await conn.fetch("""
                    SELECT id, recipient, subject, body, scheduled_for,
                           status, metadata, created_at
                    FROM ai_email_queue
                    WHERE status = 'queued'
                      AND (scheduled_for IS NULL OR scheduled_for <= NOW())
                    ORDER BY scheduled_for ASC NULLS FIRST, created_at ASC
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                """, self.batch_size)

                # Mark as processing
                if rows:
                    email_ids = [row['id'] for row in rows]
                    await conn.execute("""
                        UPDATE ai_email_queue
                        SET status = 'processing'
                        WHERE id = ANY($1)
                    """, email_ids)

                emails = []
                for row in rows:
                    # Handle metadata - could be dict, string, or None
                    metadata = row['metadata']
                    if metadata is None:
                        metadata = {}
                    elif isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}

                    emails.append(EmailJob(
                        id=str(row['id']),
                        recipient=row['recipient'],
                        subject=row['subject'],
                        body=row['body'],
                        scheduled_for=row['scheduled_for'],
                        status=row['status'],
                        metadata=metadata,
                        created_at=row['created_at']
                    ))
                return emails

        except Exception as e:
            logger.error(f"Failed to fetch due emails: {e}")
            return []

    async def _send_email(self, email: EmailJob):
        """Send a single email via Resend (primary) or SendGrid (fallback)."""
        try:
            skip_reason = _get_skip_reason(email)
            if skip_reason:
                logger.info(f"Skipping email {email.id} ({skip_reason})")
                await self._update_email_status(
                    email.id,
                    'skipped',
                    {
                        **email.metadata,
                        'skip_reason': skip_reason,
                        'skipped_at': datetime.now(timezone.utc).isoformat()
                    }
                )
                return

            provider: str | None = None
            provider_meta: dict[str, Any] = {}
            success = False
            message = ""

            if RESEND_API_KEY:
                provider = "resend"
                success, message, provider_meta = await self._send_via_resend(email)
            elif SENDGRID_API_KEY:
                provider = "sendgrid"
                success, message, provider_meta = await self._send_via_sendgrid(email)
            else:
                logger.critical("No email provider configured (need RESEND_API_KEY or SENDGRID_API_KEY)")
                await self._update_email_status(
                    email.id,
                    "failed",
                    {
                        **email.metadata,
                        "final_error": "No email provider configured (need RESEND_API_KEY or SENDGRID_API_KEY)",
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self._stats["emails_failed"] += 1
                return

            if success:
                # Success
                await self._update_email_status(email.id, 'sent', {
                    **email.metadata,
                    "provider": provider,
                    **provider_meta,
                    'sent_at': datetime.now(timezone.utc).isoformat(),
                    "send_message": message,
                })
                self._stats["emails_sent"] += 1
                logger.info(f"✅ Email {email.id} sent to {email.recipient}")

                # Optional: Record delivery for tracking (table may not exist)
                try:
                    await self._record_delivery(email)
                except Exception as delivery_err:
                    logger.debug(f"Delivery tracking skipped for {email.id}: {delivery_err}")
            else:
                # Failed
                await self._handle_send_failure(
                    email,
                    f"{provider or 'provider'} failure: {message}".strip()
                )

        except Exception as e:
            await self._handle_send_failure(email, str(e))

    async def _send_via_resend(self, email: EmailJob) -> tuple[bool, str, dict[str, Any]]:
        if not RESEND_API_KEY:
            return False, "RESEND_API_KEY not configured", {}

        try:
            import httpx

            await _throttle_resend()
            payload: dict[str, Any] = {
                "from": RESEND_FROM_EMAIL,
                "to": [email.recipient],
                "subject": email.subject,
            }

            if "<" in email.body and ">" in email.body:
                payload["html"] = email.body
            else:
                payload["text"] = email.body

            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    "https://api.resend.com/emails",
                    headers={
                        "Authorization": f"Bearer {RESEND_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

            if response.status_code in (200, 201):
                resend_id: str | None = None
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        resend_id = data.get("id")
                except Exception:
                    resend_id = None
                return True, f"Resend {response.status_code}", {
                    "resend_status": response.status_code,
                    **({"resend_id": resend_id} if resend_id else {}),
                }

            retry_after = response.headers.get("Retry-After")
            retry_note = f" retry_after={retry_after}" if retry_after else ""
            return False, f"Resend {response.status_code}{retry_note}: {response.text}", {
                "resend_status": response.status_code,
                **({"retry_after": retry_after} if retry_after else {}),
            }
        except Exception as exc:
            return False, f"Resend error: {exc}", {}

    async def _send_via_sendgrid(self, email: EmailJob) -> tuple[bool, str, dict[str, Any]]:
        if not SENDGRID_API_KEY:
            return False, "SENDGRID_API_KEY not configured", {}
        try:
            # Import SendGrid
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Content, CustomArg, Email, Mail, To

            # Determine content type - check if body contains HTML tags
            content_type = "text/html" if "<" in email.body and ">" in email.body else "text/plain"

            # Build message
            message = Mail(
                from_email=Email(SENDGRID_FROM_EMAIL, SENDGRID_FROM_NAME),
                to_emails=To(email.recipient),
                subject=email.subject,
                html_content=Content(content_type, email.body)
            )

            # Add tracking metadata as custom args for webhooks
            message.custom_arg = CustomArg('email_id', email.id)
            if email.metadata.get('campaign_id'):
                message.add_custom_arg(CustomArg('campaign_id', email.metadata.get('campaign_id', '')))
            if email.metadata.get('lead_id'):
                message.add_custom_arg(CustomArg('lead_id', email.metadata.get('lead_id', '')))

            sg = SendGridAPIClient(SENDGRID_API_KEY)
            response = sg.send(message)

            if response.status_code in [200, 201, 202]:
                return True, f"SendGrid {response.status_code}", {"sendgrid_status": response.status_code}
            return False, f"SendGrid returned {response.status_code}", {"sendgrid_status": response.status_code}
        except Exception as exc:
            return False, f"SendGrid error: {exc}", {}

    async def _handle_send_failure(self, email: EmailJob, error: str):
        """Handle email send failure with retry logic"""
        retry_count = email.metadata.get('retry_count', 0)

        if retry_count < self.max_retries:
            # Schedule retry with exponential backoff
            retry_delay = (2 ** retry_count) * 60  # 1m, 2m, 4m
            next_retry = datetime.now(timezone.utc) + timedelta(seconds=retry_delay)

            await self._update_email_status(email.id, 'queued', {
                **email.metadata,
                'retry_count': retry_count + 1,
                'last_error': error,
                'next_retry': next_retry.isoformat()
            }, next_retry)

            self._stats["emails_retried"] += 1
            logger.warning(f"⚠️ Email {email.id} failed, retry {retry_count + 1}/{self.max_retries}: {error}")
        else:
            # Max retries exceeded
            await self._update_email_status(email.id, 'failed', {
                **email.metadata,
                'final_error': error,
                'failed_at': datetime.now(timezone.utc).isoformat()
            })

            self._stats["emails_failed"] += 1
            logger.error(f"❌ Email {email.id} permanently failed after {self.max_retries} retries: {error}")

    async def _update_email_status(
        self,
        email_id: str,
        status: str,
        metadata: dict[str, Any],
        scheduled_for: Optional[datetime] = None
    ):
        """Update email status in database"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()

            async with pool.acquire() as conn:
                if status == 'sent':
                    await conn.execute("""
                        UPDATE ai_email_queue
                        SET status = $1,
                            sent_at = NOW(),
                            metadata = $2
                        WHERE id = $3
                    """, status, json.dumps(metadata), email_id)
                elif scheduled_for:
                    await conn.execute("""
                        UPDATE ai_email_queue
                        SET status = $1,
                            scheduled_for = $2,
                            metadata = $3
                        WHERE id = $4
                    """, status, scheduled_for, json.dumps(metadata), email_id)
                else:
                    await conn.execute("""
                        UPDATE ai_email_queue
                        SET status = $1,
                            metadata = $2
                        WHERE id = $3
                    """, status, json.dumps(metadata), email_id)

        except Exception as e:
            logger.error(f"Failed to update email status: {e}")

    async def _record_delivery(self, email: EmailJob):
        """Record email delivery for tracking"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()

            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ai_email_deliveries
                    (email_id, recipient, delivered_at, status, metadata)
                    VALUES ($1, $2, NOW(), 'delivered', $3)
                """, email.id, email.recipient, json.dumps({
                    'subject': email.subject,
                    'campaign_id': email.metadata.get('campaign_id'),
                    'lead_id': email.metadata.get('lead_id')
                }))
        except Exception as e:
            logger.warning(f"Failed to record delivery for email {email.id}: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics"""
        return {
            **self._stats,
            "running": self._running,
            "poll_interval": self.poll_interval,
            "batch_size": self.batch_size
        }


# ===========================================
# NURTURE CAMPAIGN HELPERS
# ===========================================

async def schedule_nurture_email(
    recipient: str,
    subject: str,
    body: str,
    delay_minutes: int = 0,
    metadata: Optional[dict[str, Any]] = None
) -> str:
    """Schedule an email for a nurture campaign"""
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        scheduled_for = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes) if delay_minutes > 0 else None

        async with pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO ai_email_queue
                (recipient, subject, body, scheduled_for, status, metadata)
                VALUES ($1, $2, $3, $4, 'queued', $5)
                RETURNING id
            """, recipient, subject, body, scheduled_for, json.dumps(metadata or {}))

            email_id = str(result['id'])
            logger.info(f"📧 Scheduled nurture email {email_id} for {recipient}")
            return email_id

    except Exception as e:
        logger.error(f"Failed to schedule nurture email: {e}")
        return ""


async def schedule_campaign_sequence(
    recipient: str,
    campaign_id: str,
    lead_id: str,
    sequence: list[dict[str, Any]]
) -> list[str]:
    """Schedule a sequence of emails for a nurture campaign"""
    email_ids = []
    cumulative_delay = 0

    for step in sequence:
        delay_minutes = step.get('delay_minutes', 0)
        cumulative_delay += delay_minutes

        email_id = await schedule_nurture_email(
            recipient=recipient,
            subject=step['subject'],
            body=step['body'],
            delay_minutes=cumulative_delay,
            metadata={
                'campaign_id': campaign_id,
                'lead_id': lead_id,
                'step_number': step.get('step_number', 0),
                'sequence_name': step.get('sequence_name', '')
            }
        )

        if email_id:
            email_ids.append(email_id)

    logger.info(f"📋 Scheduled {len(email_ids)} emails for campaign {campaign_id}")
    return email_ids


# ===========================================
# SINGLETON MANAGEMENT
# ===========================================

_daemon: Optional[EmailSchedulerDaemon] = None
_daemon_task: Optional[asyncio.Task] = None


def get_email_scheduler() -> EmailSchedulerDaemon:
    """Get or create the EmailSchedulerDaemon singleton"""
    global _daemon
    if _daemon is None:
        _daemon = EmailSchedulerDaemon()
    return _daemon


async def start_email_scheduler():
    """Start the email scheduler as a background task"""
    global _daemon_task
    daemon = get_email_scheduler()

    if _daemon_task is None or _daemon_task.done():
        _daemon_task = asyncio.create_task(daemon.start())
        logger.info("📧 Email scheduler started as background task")

    return daemon


async def stop_email_scheduler():
    """Stop the email scheduler"""
    global _daemon_task
    daemon = get_email_scheduler()
    await daemon.stop()

    if _daemon_task:
        _daemon_task.cancel()
        try:
            await _daemon_task
        except asyncio.CancelledError:
            logger.debug("Email scheduler task cancelled")
        _daemon_task = None

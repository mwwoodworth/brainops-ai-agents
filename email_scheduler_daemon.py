"""
Email Scheduler Daemon
======================
Background daemon that processes scheduled emails from ai_email_queue
and sends them via SendGrid at the appropriate times.

Handles:
- Nurture campaign emails (drip sequences)
- Follow-up emails triggered by lead actions
- Scheduled outreach emails
- Retry logic for failed sends

Author: BrainOps AI System
Version: 1.0.0
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# SendGrid configuration
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY', '')
SENDGRID_FROM_EMAIL = os.getenv('SENDGRID_FROM_EMAIL', 'noreply@brainops.ai')
SENDGRID_FROM_NAME = os.getenv('SENDGRID_FROM_NAME', 'BrainOps AI')


@dataclass
class EmailJob:
    """Represents an email job from the queue"""
    id: str
    recipient: str
    subject: str
    body: str
    scheduled_for: datetime
    status: str
    metadata: Dict[str, Any]
    created_at: datetime


class EmailSchedulerDaemon:
    """
    Background daemon that processes scheduled emails.

    Polls the ai_email_queue table for emails due to be sent
    and dispatches them via SendGrid.
    """

    def __init__(self, poll_interval: int = 60, batch_size: int = 50, max_retries: int = 3):
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
        while self._running:
            try:
                # Fetch due emails
                emails = await self._fetch_due_emails()

                if emails:
                    logger.info(f"📬 Processing {len(emails)} scheduled emails")
                    for email in emails:
                        await self._send_email(email)

                self._stats["last_poll"] = datetime.now(timezone.utc).isoformat()

            except Exception as e:
                logger.error(f"Error in email processing loop: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _fetch_due_emails(self) -> List[EmailJob]:
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

                return [
                    EmailJob(
                        id=str(row['id']),
                        recipient=row['recipient'],
                        subject=row['subject'],
                        body=row['body'],
                        scheduled_for=row['scheduled_for'],
                        status=row['status'],
                        metadata=row['metadata'] if row['metadata'] else {},
                        created_at=row['created_at']
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to fetch due emails: {e}")
            return []

    async def _send_email(self, email: EmailJob):
        """Send a single email via SendGrid"""
        try:
            if not SENDGRID_API_KEY:
                logger.warning(f"SendGrid not configured - skipping email {email.id}")
                await self._update_email_status(email.id, 'skipped', {'reason': 'no_api_key'})
                return

            # Import SendGrid
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail, Email, To, Content

            # Build message
            message = Mail(
                from_email=Email(SENDGRID_FROM_EMAIL, SENDGRID_FROM_NAME),
                to_emails=To(email.recipient),
                subject=email.subject,
                html_content=Content("text/html", email.body)
            )

            # Add tracking metadata
            message.custom_args = {
                'email_id': email.id,
                'campaign_id': email.metadata.get('campaign_id', ''),
                'lead_id': email.metadata.get('lead_id', '')
            }

            # Send via SendGrid
            sg = SendGridAPIClient(SENDGRID_API_KEY)
            response = sg.send(message)

            if response.status_code in [200, 201, 202]:
                # Success
                await self._update_email_status(email.id, 'sent', {
                    'sendgrid_response': response.status_code,
                    'sent_at': datetime.now(timezone.utc).isoformat()
                })
                self._stats["emails_sent"] += 1
                logger.info(f"✅ Email {email.id} sent to {email.recipient}")

                # Record delivery for tracking
                await self._record_delivery(email)
            else:
                # Failed
                await self._handle_send_failure(email, f"SendGrid returned {response.status_code}")

        except Exception as e:
            await self._handle_send_failure(email, str(e))

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
        metadata: Dict[str, Any],
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

    def get_stats(self) -> Dict[str, Any]:
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
    metadata: Optional[Dict[str, Any]] = None
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
    sequence: List[Dict[str, Any]]
) -> List[str]:
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
            pass
        _daemon_task = None

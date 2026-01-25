"""
Email Sender - Production Email Sending System
===============================================
Processes the ai_email_queue table and sends emails via SendGrid or SMTP.
Integrates with the agent scheduler for periodic processing.

Features:
- SendGrid primary (when SENDGRID_API_KEY is set)
- SMTP fallback (when SMTP_* env vars are set)
- Retry logic with exponential backoff
- Proper status tracking (queued -> processing -> sent/failed)
- Delivery logging in ai_email_deliveries

Author: BrainOps AI System
Version: 1.0.0
"""

import json
import logging
import os
import smtplib
import threading
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Environment configuration - Resend is PRIMARY (it works!)
RESEND_API_KEY = os.getenv('RESEND_API_KEY', '')
_RESEND_FROM_EMAIL_RAW = os.getenv('RESEND_FROM_EMAIL', 'onboarding@resend.dev')

def _sanitize_from_email(from_email: str) -> str:
    """
    Sanitize the from email to ensure it's in a valid format.
    Handles cases like "Matt @ BrainStack <email>" by removing @ from display name.
    """
    import re
    # Check if it's in "Display Name <email>" format
    match = re.match(r'^(.+?)\s*<([^>]+)>$', from_email.strip())
    if match:
        display_name = match.group(1).strip()
        email_addr = match.group(2).strip()
        # Remove @ symbols from display name (they cause Resend validation errors)
        display_name = display_name.replace('@', ' at ')
        # Also handle other problematic characters
        display_name = re.sub(r'[<>]', '', display_name)
        return f"{display_name} <{email_addr}>"
    # If it's just an email address, return as-is
    if '@' in from_email and '<' not in from_email:
        return from_email.strip()
    # Fallback: return as-is
    return from_email.strip()

RESEND_FROM_EMAIL = _sanitize_from_email(_RESEND_FROM_EMAIL_RAW)

# SendGrid as secondary option
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY', '')
SENDGRID_FROM_EMAIL = os.getenv('SENDGRID_FROM_EMAIL', 'noreply@myroofgenius.com')
SENDGRID_FROM_NAME = os.getenv('SENDGRID_FROM_NAME', 'MyRoofGenius AI')

# SMTP fallback configuration
SMTP_HOST = os.getenv('SMTP_HOST', '')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
SMTP_FROM_EMAIL = os.getenv('SMTP_FROM_EMAIL', SENDGRID_FROM_EMAIL)
SMTP_FROM_NAME = os.getenv('SMTP_FROM_NAME', SENDGRID_FROM_NAME)
SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'

# Test data safeguards
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


def _outbound_block_reason(recipient: str, metadata: dict[str, Any] | None) -> str | None:
    """
    Centralized outbound safety rail.

    Modes:
      - disabled (default): nothing sends; queue entries are marked skipped.
      - allowlist: only allowlisted recipients/domains send.
      - live: sends (still skips test recipients by heuristic).
    """
    mode = OUTBOUND_EMAIL_MODE
    if mode == "live":
        return None

    if _is_allowlisted_recipient(recipient):
        return None

    if mode == "allowlist":
        return "not_allowlisted"

    # disabled (or unknown): fail closed
    if metadata and (metadata.get("allow_outbound") is True):
        # Explicit per-email override (still subject to allowlist above). Keep for future use.
        return "outbound_disabled"
    return "outbound_disabled"


# Database configuration - supports DATABASE_URL or individual vars
def _get_db_config():
    """Get database configuration from environment variables."""
    # First try DATABASE_URL (preferred for Render/Supabase)
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        # Parse DATABASE_URL: postgresql://user:pass@host:port/dbname
        import re
        match = re.match(
            r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)',
            database_url
        )
        if match:
            return {
                'host': match.group(3),
                'database': match.group(5),
                'user': match.group(1),
                'password': match.group(2),
                'port': int(match.group(4))
            }

    # Fallback to individual environment variables
    db_host = os.getenv('DB_HOST')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_port = os.getenv('DB_PORT', '5432')

    missing = []
    if not db_host:
        missing.append('DB_HOST')
    if not db_name:
        missing.append('DB_NAME')
    if not db_user:
        missing.append('DB_USER')
    if not db_password:
        missing.append('DB_PASSWORD')

    if missing:
        raise RuntimeError(
            f"Required environment variables not set: {', '.join(missing)}. "
            "Set DATABASE_URL or these variables before using email sender."
        )

    return {
        'host': db_host,
        'database': db_name,
        'user': db_user,
        'password': db_password,
        'port': int(db_port)
    }



# Processing configuration
BATCH_SIZE = int(os.getenv('EMAIL_BATCH_SIZE', '10'))
MAX_RETRIES = int(os.getenv('EMAIL_MAX_RETRIES', '3'))
RESEND_RATE_LIMIT_PER_SECOND = float(os.getenv('RESEND_RATE_LIMIT_PER_SECOND', '2'))
RESEND_RATE_LIMIT_MAX_RETRIES = int(os.getenv('EMAIL_RATE_LIMIT_MAX_RETRIES', '5'))
RESEND_RATE_LIMIT_BASE_DELAY_SECONDS = int(os.getenv('EMAIL_RATE_LIMIT_BASE_DELAY_SECONDS', '30'))
RESEND_RATE_LIMIT_MAX_DELAY_SECONDS = int(os.getenv('EMAIL_RATE_LIMIT_MAX_DELAY_SECONDS', '600'))

_RESEND_MIN_INTERVAL = 1.0 / RESEND_RATE_LIMIT_PER_SECOND if RESEND_RATE_LIMIT_PER_SECOND > 0 else 0.0
_resend_rate_lock = threading.Lock()
_last_resend_request_at = 0.0


def _throttle_resend() -> None:
    global _last_resend_request_at
    if _RESEND_MIN_INTERVAL <= 0:
        return
    with _resend_rate_lock:
        now = time.monotonic()
        elapsed = now - _last_resend_request_at
        if elapsed < _RESEND_MIN_INTERVAL:
            time.sleep(_RESEND_MIN_INTERVAL - elapsed)
        _last_resend_request_at = time.monotonic()


def _is_rate_limited_error(message: str | None) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return "429" in lowered or "rate limit" in lowered or "too many requests" in lowered


def _extract_retry_after_seconds(message: str | None) -> int | None:
    if not message:
        return None
    lowered = message.lower()
    marker = "retry_after="
    idx = lowered.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    digits = []
    while start < len(lowered) and lowered[start].isdigit():
        digits.append(lowered[start])
        start += 1
    if not digits:
        return None
    return int("".join(digits))


def _calculate_rate_limit_delay_seconds(retry_count: int, retry_after: int | None) -> int:
    base_delay = RESEND_RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** retry_count)
    if retry_after:
        base_delay = max(base_delay, retry_after)
    if RESEND_RATE_LIMIT_MAX_DELAY_SECONDS:
        base_delay = min(base_delay, RESEND_RATE_LIMIT_MAX_DELAY_SECONDS)
    if base_delay < 1:
        return 1
    return int(base_delay)


def _is_test_recipient(recipient: str | None, metadata: dict[str, Any] | None = None) -> bool:
    if metadata and (
        metadata.get("is_test") is True
        or metadata.get("is_demo") is True
        or metadata.get("test") is True
    ):
        return True
    if not recipient:
        return True
    lowered = recipient.lower().strip()
    if any(lowered.endswith(suffix) for suffix in TEST_EMAIL_SUFFIXES):
        return True
    if any(token in lowered for token in TEST_EMAIL_TOKENS):
        return True
    return any(domain in lowered for domain in TEST_EMAIL_DOMAINS)


def get_db_connection():
    """Get database connection"""
    db_config = _get_db_config()
    return psycopg2.connect(**db_config)


def send_via_resend(recipient: str, subject: str, body: str, metadata: dict = None) -> tuple[bool, str]:
    """Send email via Resend API - PRIMARY METHOD (proven working)"""
    if not RESEND_API_KEY:
        return False, "RESEND_API_KEY not configured"

    try:
        import requests

        _throttle_resend()
        response = requests.post(
            'https://api.resend.com/emails',
            headers={
                'Authorization': f'Bearer {RESEND_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'from': RESEND_FROM_EMAIL,
                'to': [recipient],
                'subject': subject,
                'html': body,
            }
        )

        if response.status_code in [200, 201]:
            logger.info(f"Resend sent email to {recipient}: status {response.status_code}")
            return True, f"Sent via Resend (status: {response.status_code})"
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After', '').strip()
            if retry_after:
                error_msg = f"Resend rate limited (429), retry_after={retry_after}: {response.text}"
            else:
                error_msg = f"Resend rate limited (429): {response.text}"
            logger.warning(error_msg)
            return False, error_msg
        else:
            error_msg = f"Resend returned status {response.status_code}: {response.text}"
            logger.error(error_msg)
            return False, error_msg

    except Exception as e:
        error_msg = f"Resend error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def send_via_sendgrid(recipient: str, subject: str, body: str, metadata: dict = None) -> tuple[bool, str]:
    """Send email via SendGrid API"""
    if not SENDGRID_API_KEY:
        return False, "SENDGRID_API_KEY not configured"

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Content, Email, Mail, To

        # Determine if body is HTML or plain text
        content_type = "text/html" if "<" in body and ">" in body else "text/plain"

        message = Mail(
            from_email=Email(SENDGRID_FROM_EMAIL, SENDGRID_FROM_NAME),
            to_emails=To(recipient),
            subject=subject,
            html_content=Content(content_type, body)
        )

        # Send via SendGrid
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)

        if response.status_code in [200, 201, 202]:
            logger.info(f"SendGrid sent email to {recipient}: status {response.status_code}")
            return True, f"Sent via SendGrid (status: {response.status_code})"
        else:
            error_msg = f"SendGrid returned status {response.status_code}"
            logger.error(error_msg)
            return False, error_msg

    except ImportError:
        return False, "sendgrid package not installed"
    except Exception as e:
        error_msg = f"SendGrid error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def send_via_smtp(recipient: str, subject: str, body: str, metadata: dict = None) -> tuple[bool, str]:
    """Send email via SMTP as fallback"""
    if not SMTP_HOST or not SMTP_USER:
        return False, "SMTP not configured (SMTP_HOST and SMTP_USER required)"

    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
        msg['To'] = recipient

        # Determine content type
        content_type = "html" if "<" in body and ">" in body else "plain"
        msg.attach(MIMEText(body, content_type))

        # Connect and send
        if SMTP_USE_TLS:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)

        if SMTP_PASSWORD:
            server.login(SMTP_USER, SMTP_PASSWORD)

        server.sendmail(SMTP_FROM_EMAIL, recipient, msg.as_string())
        server.quit()

        logger.info(f"SMTP sent email to {recipient}")
        return True, "Sent via SMTP"

    except Exception as e:
        error_msg = f"SMTP error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def send_email(recipient: str, subject: str, body: str, metadata: dict = None) -> tuple[bool, str]:
    """Send email using available method (Resend > SendGrid > SMTP)"""
    last_error = None
    rate_limit_error = None
    # Try Resend FIRST - proven working!
    if RESEND_API_KEY:
        success, message = send_via_resend(recipient, subject, body, metadata)
        if success:
            return True, message
        last_error = message
        if _is_rate_limited_error(message):
            rate_limit_error = message
        logger.warning(f"Resend failed, trying SendGrid: {message}")

    # Try SendGrid second
    if SENDGRID_API_KEY:
        success, message = send_via_sendgrid(recipient, subject, body, metadata)
        if success:
            return True, message
        last_error = message
        if _is_rate_limited_error(message) and not rate_limit_error:
            rate_limit_error = message
        logger.warning(f"SendGrid failed, trying SMTP: {message}")

    # Try SMTP as fallback
    if SMTP_HOST:
        success, message = send_via_smtp(recipient, subject, body, metadata)
        if success:
            return True, message
        last_error = message
        if _is_rate_limited_error(message) and not rate_limit_error:
            rate_limit_error = message
        logger.warning(f"SMTP failed: {message}")

    if rate_limit_error:
        return False, rate_limit_error
    if last_error:
        return False, last_error
    return False, "No email provider configured (need RESEND_API_KEY, SENDGRID_API_KEY, or SMTP_HOST)"


def process_email_queue(batch_size: int = None, dry_run: bool = False) -> dict[str, Any]:
    """
    Process pending emails from the queue.

    Args:
        batch_size: Number of emails to process (default: BATCH_SIZE env var)
        dry_run: If True, don't actually send emails, just report what would be sent

    Returns:
        Dict with processing statistics
    """
    batch_size = batch_size or BATCH_SIZE
    stats = {
        "processed": 0,
        "sent": 0,
        "failed": 0,
        "skipped": 0,
        "retried": 0,
        "reset_failed": 0,
        "emails": [],
        "dry_run": dry_run,
        "provider": "resend" if RESEND_API_KEY else ("sendgrid" if SENDGRID_API_KEY else ("smtp" if SMTP_HOST else "none"))
    }

    # Validate DB config early
    try:
        _get_db_config()
    except RuntimeError as e:
        logger.error(f"Database not configured: {e}")
        stats["error"] = str(e)
        return stats

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Reset stuck 'processing' emails back to 'queued'
        # Either they have processing_started_at > 1 hour ago, or they don't have it (old data)
        cursor.execute("""
            UPDATE ai_email_queue
            SET status = 'queued',
                metadata = metadata || '{"reset_from_processing": true}'::jsonb
            WHERE status = 'processing'
            AND (
                metadata->>'processing_started_at' IS NULL
                OR (metadata->>'processing_started_at')::timestamp < NOW() - INTERVAL '1 hour'
            )
        """)
        reset_count = cursor.rowcount
        if reset_count > 0:
            logger.info(f"Reset {reset_count} stuck 'processing' emails back to 'queued'")
            conn.commit()

        # Reset failed rate-limited emails back to 'queued' for retry
        cursor.execute("""
            UPDATE ai_email_queue
            SET status = 'queued',
                scheduled_for = NOW(),
                metadata = metadata || %s
            WHERE status = 'failed'
            AND (
                COALESCE(metadata->>'final_error', '') ILIKE %s
                OR COALESCE(metadata->>'last_error', '') ILIKE %s
                OR COALESCE(metadata->>'error', '') ILIKE %s
                OR COALESCE(metadata->>'final_error', '') ILIKE %s
                OR COALESCE(metadata->>'last_error', '') ILIKE %s
                OR COALESCE(metadata->>'error', '') ILIKE %s
            )
        """, (
            json.dumps({
                'reset_from_failed': True,
                'reset_reason': 'rate_limit',
                'reset_at': datetime.now(timezone.utc).isoformat(),
                'retry_count': 0,
                'rate_limit_retry_count': 0
            }),
            '%429%',
            '%429%',
            '%429%',
            '%rate limit%',
            '%rate limit%',
            '%rate limit%'
        ))
        reset_failed_count = cursor.rowcount
        if reset_failed_count > 0:
            stats["reset_failed"] = reset_failed_count
            logger.info(f"Reset {reset_failed_count} rate-limited failed emails back to 'queued'")
            conn.commit()

        # Fetch emails ready to send:
        # - Status is 'queued' or 'scheduled' (with due time)
        # - scheduled_for is NULL or in the past
        cursor.execute("""
            SELECT id, recipient, subject, body, scheduled_for, status, metadata, created_at
            FROM ai_email_queue
            WHERE status IN ('queued', 'scheduled')
              AND (scheduled_for IS NULL OR scheduled_for <= NOW())
            ORDER BY
                CASE WHEN metadata->>'priority' = 'high' THEN 0 ELSE 1 END,
                scheduled_for ASC NULLS FIRST,
                created_at ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        """, (batch_size,))

        emails = cursor.fetchall()
        logger.info(f"Found {len(emails)} emails to process")

        for email in emails:
            email_id = str(email['id'])
            recipient = email['recipient']
            subject = email['subject']
            body = email['body'] or ''
            metadata = email['metadata'] or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Mark as processing with timestamp
            if not dry_run:
                cursor.execute("""
                    UPDATE ai_email_queue
                    SET status = 'processing',
                        metadata = metadata || %s
                    WHERE id = %s
                """, (json.dumps({'processing_started_at': datetime.now(timezone.utc).isoformat()}), email_id))
                conn.commit()

            stats["processed"] += 1

            # Validate email
            if not recipient or '@' not in recipient:
                logger.warning(f"Invalid recipient for email {email_id}: {recipient}")
                if not dry_run:
                    cursor.execute("""
                        UPDATE ai_email_queue
                        SET status = 'failed',
                            metadata = metadata || %s
                        WHERE id = %s
                    """, (json.dumps({'error': 'Invalid recipient', 'failed_at': datetime.now(timezone.utc).isoformat()}), email_id))
                    conn.commit()
                stats["failed"] += 1
                stats["emails"].append({"id": email_id, "recipient": recipient, "status": "failed", "reason": "invalid_recipient"})
                continue

            block_reason = _outbound_block_reason(recipient, metadata)
            if block_reason:
                logger.warning("Outbound email blocked (%s) for %s", block_reason, recipient)
                if not dry_run:
                    cursor.execute(
                        """
                        UPDATE ai_email_queue
                        SET status = 'skipped',
                            metadata = metadata || %s
                        WHERE id = %s
                        """,
                        (
                            json.dumps(
                                {
                                    "skip_reason": block_reason,
                                    "skipped_at": datetime.now(timezone.utc).isoformat(),
                                }
                            ),
                            email_id,
                        ),
                    )
                    conn.commit()
                stats["skipped"] += 1
                stats["emails"].append(
                    {"id": email_id, "recipient": recipient, "status": "skipped", "reason": block_reason}
                )
                continue

            if _is_test_recipient(recipient, metadata):
                logger.info(f"Skipping test email {email_id}: {recipient}")
                if not dry_run:
                    cursor.execute("""
                        UPDATE ai_email_queue
                        SET status = 'skipped',
                            metadata = metadata || %s
                        WHERE id = %s
                    """, (json.dumps({'skip_reason': 'test_email', 'skipped_at': datetime.now(timezone.utc).isoformat()}), email_id))
                    conn.commit()
                stats["skipped"] += 1
                stats["emails"].append({"id": email_id, "recipient": recipient, "status": "skipped", "reason": "test_email"})
                continue

            # Send email
            if dry_run:
                success = True
                message = "Dry run - would send"
            else:
                success, message = send_email(recipient, subject, body, metadata)

            if success:
                logger.info(f"Email {email_id} sent to {recipient}")
                if not dry_run:
                    # Update queue status
                    cursor.execute("""
                        UPDATE ai_email_queue
                        SET status = 'sent',
                            sent_at = NOW(),
                            metadata = metadata || %s
                        WHERE id = %s
                    """, (json.dumps({'send_message': message, 'sent_at': datetime.now(timezone.utc).isoformat()}), email_id))

                    # Record delivery (optional - table may not exist yet)
                    try:
                        cursor.execute("""
                            INSERT INTO ai_email_deliveries
                            (email_id, recipient, status, delivered_at)
                            VALUES (%s, %s, 'delivered', NOW())
                        """, (email_id, recipient))
                    except Exception as delivery_err:
                        logger.debug(f"Delivery tracking skipped for {email_id}: {delivery_err}")

                    conn.commit()

                stats["sent"] += 1
                stats["emails"].append({"id": email_id, "recipient": recipient, "status": "sent", "message": message})
            else:
                # Handle failure with retry logic
                retry_count = metadata.get('retry_count', 0)

                if _is_rate_limited_error(message):
                    rate_limit_retry_count = metadata.get('rate_limit_retry_count', 0)
                    if rate_limit_retry_count < RESEND_RATE_LIMIT_MAX_RETRIES:
                        retry_after_seconds = _extract_retry_after_seconds(message)
                        delay_seconds = _calculate_rate_limit_delay_seconds(rate_limit_retry_count, retry_after_seconds)
                        next_retry = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)

                        logger.warning(
                            f"Email {email_id} rate limited, scheduling retry "
                            f"{rate_limit_retry_count + 1}/{RESEND_RATE_LIMIT_MAX_RETRIES} in {delay_seconds}s: {message}"
                        )
                        if not dry_run:
                            cursor.execute("""
                                UPDATE ai_email_queue
                                SET status = 'queued',
                                    scheduled_for = %s,
                                    metadata = metadata || %s
                                WHERE id = %s
                            """, (
                                next_retry,
                                json.dumps({
                                    'rate_limit_retry_count': rate_limit_retry_count + 1,
                                    'last_error': message,
                                    'rate_limit_retry_scheduled_for': next_retry.isoformat(),
                                    'rate_limited_at': datetime.now(timezone.utc).isoformat()
                                }),
                                email_id
                            ))
                            conn.commit()

                        stats["retried"] += 1
                        stats["emails"].append({
                            "id": email_id,
                            "recipient": recipient,
                            "status": "rate_limit_retry_scheduled",
                            "retry": rate_limit_retry_count + 1,
                            "delay_seconds": delay_seconds,
                            "error": message
                        })
                        continue
                    logger.warning(
                        f"Email {email_id} rate limit retries exceeded, falling back to standard retry: {message}"
                    )

                if retry_count < MAX_RETRIES:
                    # Schedule retry with exponential backoff
                    retry_delay_minutes = (2 ** retry_count) * 5  # 5, 10, 20 minutes
                    next_retry = datetime.now(timezone.utc) + timedelta(minutes=retry_delay_minutes)

                    logger.warning(f"Email {email_id} failed, scheduling retry {retry_count + 1}/{MAX_RETRIES}: {message}")
                    if not dry_run:
                        cursor.execute("""
                            UPDATE ai_email_queue
                            SET status = 'queued',
                                scheduled_for = %s,
                                metadata = metadata || %s
                            WHERE id = %s
                        """, (
                            next_retry,
                            json.dumps({
                                'retry_count': retry_count + 1,
                                'last_error': message,
                                'retry_scheduled_for': next_retry.isoformat()
                            }),
                            email_id
                        ))
                        conn.commit()

                    stats["retried"] += 1
                    stats["emails"].append({"id": email_id, "recipient": recipient, "status": "retry_scheduled", "retry": retry_count + 1, "error": message})
                else:
                    # Max retries exceeded
                    logger.error(f"Email {email_id} permanently failed after {MAX_RETRIES} retries: {message}")
                    if not dry_run:
                        cursor.execute("""
                            UPDATE ai_email_queue
                            SET status = 'failed',
                                metadata = metadata || %s
                            WHERE id = %s
                        """, (json.dumps({'final_error': message, 'failed_at': datetime.now(timezone.utc).isoformat(), 'retry_count': retry_count}), email_id))
                        conn.commit()

                    stats["failed"] += 1
                    stats["emails"].append({"id": email_id, "recipient": recipient, "status": "failed", "error": message})

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error processing email queue: {e}")
        stats["error"] = str(e)

    logger.info(
        "Email queue processing complete: "
        f"{stats['sent']} sent, {stats['failed']} failed, {stats['skipped']} skipped, "
        f"{stats['retried']} retried, {stats['reset_failed']} reset"
    )
    return stats


def get_queue_status() -> dict[str, Any]:
    """Get current email queue status"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("""
            SELECT
                status,
                COUNT(*) as count,
                MIN(created_at) as oldest,
                MAX(created_at) as newest
            FROM ai_email_queue
            GROUP BY status
            ORDER BY count DESC
        """)

        status_counts = cursor.fetchall()

        cursor.execute("""
            SELECT COUNT(*) as total,
                   COUNT(*) FILTER (WHERE status = 'sent') as sent,
                   COUNT(*) FILTER (WHERE status = 'failed') as failed,
                   COUNT(*) FILTER (WHERE status = 'queued') as queued,
                   COUNT(*) FILTER (WHERE status = 'processing') as processing,
                   COUNT(*) FILTER (WHERE status = 'scheduled') as scheduled,
                   COUNT(*) FILTER (WHERE status = 'skipped') as skipped
            FROM ai_email_queue
        """)

        totals = cursor.fetchone()

        cursor.close()
        conn.close()

        return {
            "totals": dict(totals),
            "by_status": [dict(row) for row in status_counts],
            "provider": "resend" if RESEND_API_KEY else ("sendgrid" if SENDGRID_API_KEY else ("smtp" if SMTP_HOST else "none")),
            "provider_configured": bool(RESEND_API_KEY or SENDGRID_API_KEY or SMTP_HOST)
        }

    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return {"error": str(e)}


# Singleton for scheduled execution
_email_processor_running = False


def run_email_processor():
    """Run email processor - called by agent scheduler"""
    global _email_processor_running

    if _email_processor_running:
        logger.warning("Email processor already running, skipping")
        return {"skipped": True, "reason": "already_running"}

    try:
        _email_processor_running = True
        result = process_email_queue()
        return result
    finally:
        _email_processor_running = False


# CLI interface for testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "status":
            status = get_queue_status()
            print(json.dumps(status, indent=2, default=str))

        elif command == "process":
            dry_run = "--dry-run" in sys.argv
            batch_size = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else BATCH_SIZE
            result = process_email_queue(batch_size=batch_size, dry_run=dry_run)
            print(json.dumps(result, indent=2, default=str))

        elif command == "test":
            # Send a test email
            if len(sys.argv) < 3:
                print("Usage: python email_sender.py test <email>")
                sys.exit(1)
            test_email = sys.argv[2]
            success, message = send_email(
                test_email,
                "Test Email from BrainOps AI",
                "<h1>Test Email</h1><p>This is a test email from the BrainOps AI email system.</p>",
                {}
            )
            print(f"Result: {'Success' if success else 'Failed'} - {message}")

        else:
            print("Usage: python email_sender.py [status|process|test <email>]")
    else:
        # Default: show status and process queue
        print("Email Queue Status:")
        status = get_queue_status()
        print(json.dumps(status, indent=2, default=str))

        print("\nProcessing queue (dry run):")
        result = process_email_queue(dry_run=True)
        print(json.dumps(result, indent=2, default=str))

"""
Email Capture API - Lead Generation for Gumroad Products
=========================================================
Captures email addresses and adds them to nurture sequences.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/email", tags=["email-capture"])


class EmailCaptureRequest(BaseModel):
    email: EmailStr
    source: Optional[str] = "landing_page"
    name: Optional[str] = None


class EmailCaptureResponse(BaseModel):
    success: bool
    message: str


def _get_db_connection():
    """Get database connection"""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        # Prefer DATABASE_URL (Render/Supabase); ensure SSL is required.
        database_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
        if database_url:
            if "sslmode=" not in database_url:
                sep = "&" if "?" in database_url else "?"
                database_url = f"{database_url}{sep}sslmode=require"
            return psycopg2.connect(database_url, cursor_factory=RealDictCursor)

        # Fallback to individual environment variables (Supabase requires SSL).
        db_host = os.getenv("DB_HOST")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME", "postgres")
        db_port = os.getenv("DB_PORT", "6543")

        return psycopg2.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            dbname=db_name,
            port=int(db_port),
            sslmode="require",
            cursor_factory=RealDictCursor,
        )
    except Exception as e:
        # Do not leak connection details; keep error high-level.
        logger.error("Database connection failed for email_capture", exc_info=True)
        raise


@router.post("/capture", response_model=EmailCaptureResponse)
async def capture_email(request: EmailCaptureRequest):
    """
    Capture an email for lead nurturing.
    - Stores in email_captures table
    - Triggers welcome sequence
    - Sends free starter kit
    """
    try:
        conn = _get_db_connection()
        cur = conn.cursor()

        # Check if email already exists
        cur.execute(
            "SELECT id FROM email_captures WHERE email = %s",
            (request.email,)
        )
        existing = cur.fetchone()

        if existing:
            conn.close()
            return EmailCaptureResponse(
                success=True,
                message="You're already on our list! Check your inbox for the starter kit."
            )

        # Insert new capture
        cur.execute("""
            INSERT INTO email_captures (email, source, name, created_at)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (request.email, request.source, request.name, datetime.now(timezone.utc)))

        capture_id = cur.fetchone()['id']
        conn.commit()

        # Queue welcome email with free starter kit
        cur.execute("""
            INSERT INTO ai_email_queue (recipient, subject, body, status, scheduled_for, metadata)
            VALUES (%s, %s, %s, 'queued', NOW(), %s)
        """, (
            request.email,
            "Your Free AI Automation Starter Kit is Ready!",
            f"""Hi there!

Thanks for signing up for the BrainStack AI Automation Starter Kit.

Here's what's inside:

## 5 Production-Ready AI Prompts
1. Lead Qualification Prompt - Instantly score and qualify incoming leads
2. Content Generation Prompt - Create SEO-optimized blog posts
3. Email Response Prompt - Craft professional responses in seconds
4. Data Extraction Prompt - Pull structured data from unstructured text
5. Code Review Prompt - Get instant feedback on your code

## 2 Automation Scripts
1. lead_qualifier.py - Automatic lead scoring based on engagement
2. data_extractor.py - Extract key information from any document

Download your kit here: https://brainstack.gumroad.com/l/free-starter-kit

## Ready for More?
Check out our full product lineup:
- AI Prompt Engineering Pack (200+ prompts) - $29
- Business Automation Toolkit - $29
- MCP Server Starter Kit - $49

Use code STARTER20 for 20% off your first purchase!

Happy automating,
Matt @ BrainStack

P.S. Reply to this email if you have any questions. I read every message.
""",
            '{"source": "email_capture", "capture_id": "' + str(capture_id) + '"}'
        ))
        conn.commit()
        conn.close()

        logger.info(f"Email captured: {request.email} from {request.source}")

        return EmailCaptureResponse(
            success=True,
            message="Check your inbox! Your free starter kit is on its way."
        )

    except Exception as e:
        logger.error(f"Email capture error: {e}")
        raise HTTPException(status_code=500, detail="Failed to capture email")


@router.get("/stats")
async def get_capture_stats():
    """Get email capture statistics"""
    try:
        conn = _get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                COUNT(*) as total_captures,
                COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as last_7_days,
                COUNT(CASE WHEN created_at > NOW() - INTERVAL '30 days' THEN 1 END) as last_30_days
            FROM email_captures
        """)
        stats = cur.fetchone()
        conn.close()

        return {
            "total_captures": stats['total_captures'],
            "last_7_days": stats['last_7_days'],
            "last_30_days": stats['last_30_days']
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"total_captures": 0, "last_7_days": 0, "last_30_days": 0}

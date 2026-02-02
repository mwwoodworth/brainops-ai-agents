"""
Email Capture API - Lead Generation for Gumroad Products
=========================================================
Captures email addresses and adds them to nurture sequences.
"""

import os
import logging
import ipaddress
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/email", tags=["email-capture"])


class EmailCaptureRequest(BaseModel):
    email: EmailStr
    source: Optional[str] = "landing_page"
    name: Optional[str] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None


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
async def capture_email(payload: EmailCaptureRequest, http_request: Request):
    """
    Capture an email for lead nurturing.
    - Stores in email_captures table
    - Triggers welcome sequence
    - Sends free starter kit
    """
    try:
        from psycopg2.extras import Json

        conn = _get_db_connection()
        cur = conn.cursor()

        # Check if email already exists
        cur.execute(
            "SELECT id FROM email_captures WHERE email = %s",
            (payload.email,)
        )
        existing = cur.fetchone()

        if existing:
            conn.close()
            return EmailCaptureResponse(
                success=True,
                message="You're already on our list! Check your inbox for the starter kit."
            )

        # Capture IP/user-agent safely (do not error if headers are missing/invalid).
        forwarded_for = (http_request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        client_ip = forwarded_for or (http_request.client.host if http_request.client else "")
        try:
            ipaddress.ip_address(client_ip)
        except Exception:
            client_ip = None

        user_agent = http_request.headers.get("user-agent")

        meta_data = {"captured_via": "email_capture_api"}
        if payload.name:
            meta_data["name"] = payload.name

        # Insert new capture
        cur.execute("""
            INSERT INTO email_captures (
              email,
              source,
              meta_data,
              utm_source,
              utm_medium,
              utm_campaign,
              ip_address,
              user_agent
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            payload.email,
            payload.source,
            Json(meta_data),
            payload.utm_source,
            payload.utm_medium,
            payload.utm_campaign,
            client_ip,
            user_agent,
        ))

        capture_id = cur.fetchone()['id']

        # Queue welcome email with free starter kit
        email_metadata = {"source": "email_capture", "capture_id": str(capture_id)}
        if payload.utm_source:
            email_metadata["utm_source"] = payload.utm_source
        if payload.utm_medium:
            email_metadata["utm_medium"] = payload.utm_medium
        if payload.utm_campaign:
            email_metadata["utm_campaign"] = payload.utm_campaign

        cur.execute("""
            INSERT INTO ai_email_queue (recipient, subject, body, status, scheduled_for, metadata)
            VALUES (%s, %s, %s, 'queued', NOW(), %s)
        """, (
            payload.email,
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
            Json(email_metadata),
        ))
        conn.commit()
        conn.close()

        logger.info("Email captured: %s from %s", payload.email, payload.source)

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

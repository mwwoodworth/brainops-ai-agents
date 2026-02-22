"""
Communications API
==================
REST endpoints for sending communications from Weathercraft ERP.

Handles estimate, invoice, and other document delivery to customers.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/communications", tags=["Communications"])


class SendEstimateRequest(BaseModel):
    """Request to send an estimate to a customer."""
    estimate_id: str
    estimate_number: Optional[str] = None
    customer_email: EmailStr
    customer_name: str
    project_name: Optional[str] = "Roofing Project"
    total_amount: Optional[float] = None
    valid_until: Optional[str] = None
    tenant_id: str


def _format_currency(amount: Optional[float]) -> str:
    """Format amount as USD currency."""
    if amount is None:
        return "TBD"
    return f"${amount:,.2f}"


def _generate_estimate_email_html(request: SendEstimateRequest, view_link: str) -> str:
    """Generate HTML email content for estimate."""
    total_formatted = _format_currency(request.total_amount)
    valid_until_text = ""
    if request.valid_until:
        try:
            valid_date = datetime.fromisoformat(request.valid_until.replace("Z", "+00:00"))
            valid_until_text = f"<p><strong>Valid Until:</strong> {valid_date.strftime('%B %d, %Y')}</p>"
        except (ValueError, TypeError):
            pass

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Estimate from Weathercraft Roofing</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #1a365d; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f7fafc; padding: 30px; border: 1px solid #e2e8f0; }}
        .estimate-box {{ background: white; border: 2px solid #1a365d; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .amount {{ font-size: 28px; font-weight: bold; color: #1a365d; }}
        .cta-button {{ display: inline-block; background: #2563eb; color: white !important; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 20px 0; }}
        .cta-button:hover {{ background: #1d4ed8; }}
        .footer {{ text-align: center; padding: 20px; color: #718096; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Weathercraft Roofing</h1>
        <p>Your Estimate is Ready</p>
    </div>
    <div class="content">
        <p>Dear {request.customer_name},</p>

        <p>Thank you for choosing Weathercraft Roofing! We're pleased to provide you with an estimate for your project.</p>

        <div class="estimate-box">
            <p><strong>Estimate #:</strong> {request.estimate_number or request.estimate_id[:8].upper()}</p>
            <p><strong>Project:</strong> {request.project_name}</p>
            <p class="amount">Total: {total_formatted}</p>
            {valid_until_text}
        </div>

        <p>Please review your estimate and let us know if you have any questions. When you're ready to proceed, simply click the button below to view the full details and approve.</p>

        <center>
            <a href="{view_link}" class="cta-button">View Full Estimate</a>
        </center>

        <p>If you have any questions, please don't hesitate to contact us.</p>

        <p>Best regards,<br>
        <strong>Weathercraft Roofing Team</strong></p>
    </div>
    <div class="footer">
        <p>Weathercraft Roofing &bull; Professional Roofing Services</p>
        <p>This estimate is valid for 30 days unless otherwise specified.</p>
    </div>
</body>
</html>
"""


def _generate_estimate_email_text(request: SendEstimateRequest, view_link: str) -> str:
    """Generate plain text email content for estimate."""
    total_formatted = _format_currency(request.total_amount)

    return f"""
Dear {request.customer_name},

Thank you for choosing Weathercraft Roofing! We're pleased to provide you with an estimate for your project.

ESTIMATE DETAILS
----------------
Estimate #: {request.estimate_number or request.estimate_id[:8].upper()}
Project: {request.project_name}
Total: {total_formatted}

Please review your estimate at: {view_link}

If you have any questions, please don't hesitate to contact us.

Best regards,
Weathercraft Roofing Team
"""


@router.post("/send-estimate")
async def send_estimate(request: SendEstimateRequest) -> dict[str, Any]:
    """
    Send an estimate to a customer via email.

    This endpoint is called by the Weathercraft ERP when a user clicks
    "Send Estimate" to deliver the estimate to the customer.
    """
    from database.async_connection import get_pool

    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    now = datetime.now(timezone.utc)

    # Generate view link (ERP portal)
    # The ERP has its own customer portal where estimates can be viewed
    view_link = f"https://weathercraft-erp.vercel.app/portal/estimates/{request.estimate_id}"

    # Generate email content
    html_content = _generate_estimate_email_html(request, view_link)

    # Prepare email metadata
    email_metadata = {
        "source": "weathercraft_erp",
        "type": "estimate",
        "estimate_id": request.estimate_id,
        "estimate_number": request.estimate_number,
        "tenant_id": request.tenant_id,
        "customer_name": request.customer_name,
        "project_name": request.project_name,
        "total_amount": float(request.total_amount) if request.total_amount else None,
        "view_link": view_link,
        "sent_at": now.isoformat(),
    }

    try:
        # Queue the email
        email_id = uuid.uuid4()
        subject = f"Your Estimate from Weathercraft Roofing - {request.estimate_number or request.estimate_id[:8].upper()}"

        await pool.execute("""
            INSERT INTO ai_email_queue (id, recipient, subject, body, status, scheduled_for, created_at, metadata)
            VALUES ($1, $2, $3, $4, 'queued', $5, $5, $6::jsonb)
        """,
            email_id,
            request.customer_email,
            subject,
            html_content,
            now,
            json.dumps(email_metadata),
        )

        logger.info(
            f"Estimate email queued: {request.estimate_number or request.estimate_id[:8]} -> {request.customer_email[:3]}***"
        )

        return {
            "success": True,
            "message": f"Estimate sent to {request.customer_email}",
            "email_id": str(email_id),
            "estimate_id": request.estimate_id,
            "view_link": view_link,
        }

    except Exception as e:
        logger.error(f"Failed to queue estimate email: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{email_id}")
async def get_email_status(email_id: str) -> dict[str, Any]:
    """
    Get the status of a sent email.
    """
    from database.async_connection import get_pool

    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        email = await pool.fetchrow("""
            SELECT id, recipient, subject, status, scheduled_for, sent_at, created_at, metadata
            FROM ai_email_queue
            WHERE id = $1
        """, uuid.UUID(email_id))

        if not email:
            raise HTTPException(status_code=404, detail="Email not found")

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "email": {
                "id": str(email["id"]),
                "recipient_masked": email["recipient"][:3] + "***@" + email["recipient"].split("@")[1] if "@" in email["recipient"] else "***",
                "subject": email["subject"],
                "status": email["status"],
                "scheduled_for": email["scheduled_for"].isoformat() if email["scheduled_for"] else None,
                "sent_at": email["sent_at"].isoformat() if email.get("sent_at") else None,
                "created_at": email["created_at"].isoformat() if email["created_at"] else None,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get email status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

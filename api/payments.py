"""
Payments API
============
REST endpoints for payment capture and invoice management.

Part of Revenue Perfection Session.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from payment_capture import (
    get_payment_capture,
    ensure_invoices_table,
    STRIPE_AVAILABLE,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payments", tags=["Payment Capture"])

PAYMENTS_CONTACT_EMAIL = (
    os.getenv("PAYMENTS_CONTACT_EMAIL", "").strip()
    or os.getenv("SUPPORT_EMAIL", "").strip()
    or "support@brainstackstudio.com"
)


class MarkPaidRequest(BaseModel):
    """Request to mark an invoice as paid."""
    payment_method: str = "manual"
    payment_reference: Optional[str] = None
    verified_by: str


@router.on_event("startup")
async def startup():
    """Ensure invoices table exists on startup."""
    await ensure_invoices_table()


@router.get("/status")
async def get_payment_status() -> dict[str, Any]:
    """
    Get payment system status.
    """
    pc = get_payment_capture()
    summary = await pc.get_revenue_summary(real_only=True)

    stripe_client = pc._get_stripe()
    stripe_mode = "unavailable"
    if stripe_client and STRIPE_AVAILABLE:
        stripe_mode = "live" if "live" in str(STRIPE_AVAILABLE).lower() else "test"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stripe_available": STRIPE_AVAILABLE,
        "stripe_mode": stripe_mode,
        "payment_methods": ["stripe", "manual"] if STRIPE_AVAILABLE else ["manual"],
        "revenue_summary": summary
    }


@router.post("/invoice/from-proposal/{proposal_id}")
async def create_invoice_from_proposal(
    proposal_id: str,
    due_days: int = Query(default=14, ge=7, le=90)
) -> dict[str, Any]:
    """
    Create an invoice from an approved/sent proposal.

    Generates a Stripe checkout link if Stripe is available,
    otherwise provides a manual payment link.
    """
    pc = get_payment_capture()

    success, message, invoice = await pc.create_invoice(proposal_id, due_days)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "invoice_id": invoice.id,
        "amount": float(invoice.amount),
        "currency": invoice.currency,
        "payment_link": invoice.payment_link,
        "stripe_checkout": bool(invoice.stripe_checkout_id),
        "due_date": invoice.due_date.isoformat(),
        "next_step": f"POST /payments/invoice/{invoice.id}/send to send to client"
    }


@router.post("/invoice/{invoice_id}/send")
async def send_invoice(invoice_id: str) -> dict[str, Any]:
    """
    Send an invoice to the client.

    Queues email with payment link.
    """
    pc = get_payment_capture()

    success, message = await pc.send_invoice(invoice_id)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "invoice_id": invoice_id,
        "status": "sent",
        "note": "Email queued. Lead state updated to INVOICED."
    }


@router.post("/invoice/{invoice_id}/mark-paid")
async def mark_invoice_paid(
    invoice_id: str,
    request: MarkPaidRequest
) -> dict[str, Any]:
    """
    Manually mark an invoice as paid.

    CRITICAL: This creates REAL REVENUE in the system.
    Use only when payment has been verified.
    """
    pc = get_payment_capture()

    success, message = await pc.mark_paid(
        invoice_id,
        payment_method=request.payment_method,
        payment_reference=request.payment_reference,
        verified_by=request.verified_by
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "invoice_id": invoice_id,
        "status": "paid",
        "note": "REAL REVENUE CAPTURED. Lead state updated to PAID."
    }


@router.get("/invoice/{invoice_id}")
async def get_invoice(invoice_id: str) -> dict[str, Any]:
    """Get invoice details."""
    from database.async_connection import get_pool
    import uuid

    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    invoice = await pool.fetchrow("""
        SELECT i.*, p.offer_name, p.client_company
        FROM ai_invoices i
        JOIN ai_proposals p ON i.proposal_id = p.id
        WHERE i.id = $1
    """, uuid.UUID(invoice_id))

    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "invoice": {
            "id": str(invoice["id"]),
            "proposal_id": str(invoice["proposal_id"]),
            "lead_id": str(invoice["lead_id"]),
            "offer_name": invoice["offer_name"],
            "client_company": invoice["client_company"],
            "amount": float(invoice["amount"]),
            "currency": invoice["currency"],
            "status": invoice["status"],
            "payment_link": invoice["payment_link"],
            "due_date": invoice["due_date"].isoformat() if invoice["due_date"] else None,
            "paid_at": invoice["paid_at"].isoformat() if invoice.get("paid_at") else None,
            "created_at": invoice["created_at"].isoformat() if invoice["created_at"] else None
        }
    }


@router.get("/invoices/pending")
async def get_pending_invoices() -> dict[str, Any]:
    """Get all pending/sent invoices awaiting payment."""
    from database.async_connection import get_pool

    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    rows = await pool.fetch("""
        SELECT i.id, i.amount, i.status, i.due_date, p.offer_name, p.client_company
        FROM ai_invoices i
        JOIN ai_proposals p ON i.proposal_id = p.id
        WHERE i.status IN ('pending', 'sent')
        ORDER BY i.due_date ASC
    """)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_pending": len(rows),
        "total_value": sum(float(r["amount"]) for r in rows),
        "invoices": [
            {
                "id": str(r["id"]),
                "offer_name": r["offer_name"],
                "client_company": r["client_company"],
                "amount": float(r["amount"]),
                "status": r["status"],
                "due_date": r["due_date"].isoformat() if r["due_date"] else None
            }
            for r in rows
        ]
    }


@router.get("/revenue/real")
async def get_real_revenue() -> dict[str, Any]:
    """
    Get real revenue from PAID invoices.

    This is the ground truth for actual revenue collected.
    """
    pc = get_payment_capture()
    summary = await pc.get_revenue_summary(real_only=True)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **summary,
        "note": "Revenue only from PAID invoices linked to REAL leads"
    }


# Success/cancel pages for Stripe checkout
@router.get("/success")
async def payment_success(invoice_id: str) -> dict[str, Any]:
    """Handle successful payment redirect from Stripe."""
    return {
        "success": True,
        "message": "Payment successful! Thank you.",
        "invoice_id": invoice_id,
        "note": "Payment confirmation will be processed via webhook."
    }


@router.get("/cancel")
async def payment_cancelled(invoice_id: str) -> dict[str, Any]:
    """Handle cancelled payment redirect from Stripe."""
    return {
        "cancelled": True,
        "message": "Payment was cancelled.",
        "invoice_id": invoice_id,
        "note": "You can try again using the payment link in your email."
    }


@router.get("/manual/{invoice_id}")
async def manual_payment_page(invoice_id: str) -> dict[str, Any]:
    """
    Manual payment instructions page.

    Used when Stripe is not available.
    """
    from database.async_connection import get_pool
    import uuid

    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    invoice = await pool.fetchrow("""
        SELECT i.*, p.offer_name, p.client_company
        FROM ai_invoices i
        JOIN ai_proposals p ON i.proposal_id = p.id
        WHERE i.id = $1
    """, uuid.UUID(invoice_id))

    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "invoice_id": str(invoice["id"]),
        "amount": float(invoice["amount"]),
        "currency": invoice["currency"],
        "offer_name": invoice["offer_name"],
        "client_company": invoice["client_company"],
        "due_date": invoice["due_date"].isoformat() if invoice["due_date"] else None,
        "payment_instructions": {
            "method_1": {
                "name": "Bank Transfer",
                "details": f"Contact {PAYMENTS_CONTACT_EMAIL} for wire transfer instructions"
            },
            "method_2": {
                "name": "PayPal",
                "details": f"Send to {PAYMENTS_CONTACT_EMAIL} with invoice ID in notes"
            },
            "method_3": {
                "name": "Crypto",
                "details": "Contact for BTC/ETH address"
            }
        },
        "note": "After payment, your invoice will be marked as paid within 24 hours."
    }

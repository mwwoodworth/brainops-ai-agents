"""
Stripe Webhook Handler
======================
Handles Stripe payment events with proper signature verification.
This endpoint is NOT protected by API key - it uses Stripe's signature verification.
Records all events to stripe_events table and updates revenue_leads pipeline.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

logger = logging.getLogger(__name__)

# Configuration - check both possible env var names
STRIPE_WEBHOOK_SECRET = (
    os.getenv("STRIPE_WEBHOOK_SECRET_AIAGENTS") or
    os.getenv("STRIPE_WEBHOOK_SECRET") or
    ""
)
ENVIRONMENT = os.getenv("ENVIRONMENT", "production").strip().lower()

router = APIRouter(
    prefix="/stripe",
    tags=["stripe", "payments", "webhooks"]
)


def _get_db_pool():
    """Get the async database pool for recording events."""
    from database.async_connection import get_pool
    return get_pool()


async def _record_stripe_event(event_type: str, stripe_id: str, customer_email: str | None,
                                amount_cents: int, currency: str, status: str,
                                raw_data: dict) -> bool:
    """Record a Stripe event to the database. Returns True on success."""
    try:
        pool = _get_db_pool()
        # Check for duplicate first (no unique constraint on stripe_id)
        existing = await pool.fetchval(
            "SELECT id FROM stripe_events WHERE stripe_id = $1", stripe_id
        )
        if existing:
            logger.info(f"Stripe event already recorded: {stripe_id}")
            return True

        await pool.execute("""
            INSERT INTO stripe_events (event_type, stripe_id, customer_email, amount_cents,
                                       currency, status, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, NOW())
        """, event_type, stripe_id, customer_email, amount_cents, currency, status,
            json.dumps(raw_data, default=str))
        logger.info(f"Recorded stripe event: {event_type} ({stripe_id})")
        return True
    except Exception as e:
        logger.error(f"Failed to record stripe event: {e}")
        return False


async def _update_lead_to_won(customer_email: str, amount: float) -> bool:
    """Update a revenue lead to WON stage when payment succeeds."""
    if not customer_email:
        return False
    try:
        pool = _get_db_pool()
        result = await pool.execute("""
            UPDATE revenue_leads
            SET stage = 'won', status = 'customer',
                value_estimate = GREATEST(COALESCE(value_estimate, 0), $2),
                converted_at = NOW(), last_interaction_at = NOW()
            WHERE LOWER(email) = LOWER($1) AND stage != 'won'
        """, customer_email, amount)
        if result and "UPDATE" in str(result):
            logger.info(f"Updated revenue lead {customer_email} to WON (${amount})")
        return True
    except Exception as e:
        logger.error(f"Failed to update lead: {e}")
        return False


def verify_stripe_signature(payload: bytes, sig_header: str, webhook_secret: str) -> dict:
    """Verify Stripe webhook signature and return the event."""
    import hashlib
    import hmac
    import time

    if not webhook_secret:
        logger.error("STRIPE_WEBHOOK_SECRET_AIAGENTS not configured")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")

    elements = {}
    for element in sig_header.split(","):
        key, value = element.split("=", 1)
        elements[key] = value

    timestamp = elements.get("t")
    signature = elements.get("v1")

    if not timestamp or not signature:
        raise HTTPException(status_code=400, detail="Invalid signature format")

    try:
        timestamp_int = int(timestamp)
        if abs(time.time() - timestamp_int) > 300:
            raise HTTPException(status_code=400, detail="Timestamp too old")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp")

    signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
    expected_sig = hmac.new(
        webhook_secret.encode("utf-8"),
        signed_payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_sig, signature):
        logger.warning("Invalid Stripe signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")


async def process_stripe_event(event: dict, background_tasks: BackgroundTasks) -> dict:
    """Process a verified Stripe event."""
    event_type = event.get("type", "unknown")
    event_id = event.get("id", "unknown")
    data = event.get("data", {}).get("object", {})
    if isinstance(data, dict) and "livemode" not in data and "livemode" in event:
        data["livemode"] = event["livemode"]

    logger.info(f"Processing Stripe event: {event_type} ({event_id})")

    try:
        if event_type == "checkout.session.completed":
            return await handle_checkout_completed(data, background_tasks)
        elif event_type == "customer.subscription.created":
            return await handle_subscription_created(data)
        elif event_type == "customer.subscription.updated":
            return await handle_subscription_updated(data)
        elif event_type == "customer.subscription.deleted":
            return await handle_subscription_deleted(data)
        elif event_type == "invoice.paid":
            return await handle_invoice_paid(data)
        elif event_type == "invoice.payment_failed":
            return await handle_payment_failed(data)
        elif event_type == "charge.succeeded":
            return await handle_charge_succeeded(data)
        elif event_type == "charge.refunded":
            return await handle_charge_refunded(data)
        elif event_type == "charge.dispute.created":
            return await handle_dispute_created(data)
        else:
            logger.info(f"Unhandled event type: {event_type}")
            return {"status": "ignored", "event_type": event_type}

    except Exception as e:
        logger.error(f"Error processing {event_type}: {e}")
        return {"status": "error", "event_type": event_type, "error": str(e)}


async def handle_checkout_completed(data: dict, background_tasks: BackgroundTasks) -> dict:
    """Handle checkout.session.completed - records event and updates lead to WON."""
    customer_email = data.get("customer_email") or data.get("customer_details", {}).get("email")
    amount_total = data.get("amount_total", 0) / 100
    currency = data.get("currency", "usd").upper()
    payment_status = data.get("payment_status")
    session_id = data.get("id")

    logger.info(f"Checkout completed: {customer_email}, ${amount_total} {currency}")

    await _record_stripe_event(
        "checkout.session.completed", session_id, customer_email,
        int(amount_total * 100), currency, payment_status or "completed", data
    )

    if customer_email and payment_status == "paid":
        await _update_lead_to_won(customer_email, amount_total)

    return {
        "status": "processed",
        "event_type": "checkout.session.completed",
        "customer_email": customer_email,
        "amount": amount_total
    }


async def handle_subscription_created(data: dict) -> dict:
    """Handle customer.subscription.created - records new subscription."""
    sub_id = data.get("id")
    customer_id = data.get("customer")
    status = data.get("status", "active")
    plan = data.get("plan", {})
    amount = plan.get("amount", 0) / 100
    interval = plan.get("interval", "month")

    logger.info(f"Subscription created: {sub_id}, ${amount}/{interval}")

    await _record_stripe_event(
        "customer.subscription.created", sub_id, None,
        int(amount * 100), plan.get("currency", "usd").upper(), status, data
    )

    # Record in mrg_subscriptions for MRR tracking
    try:
        pool = _get_db_pool()
        existing = await pool.fetchval(
            "SELECT id FROM mrg_subscriptions WHERE stripe_subscription_id = $1", sub_id
        )
        if existing:
            await pool.execute("""
                UPDATE mrg_subscriptions SET status = $2, amount = $3, updated_at = NOW()
                WHERE stripe_subscription_id = $1
            """, sub_id, status, amount)
        else:
            await pool.execute("""
                INSERT INTO mrg_subscriptions (stripe_subscription_id, status, amount,
                                               billing_cycle, created_at)
                VALUES ($1, $2, $3, $4, NOW())
            """, sub_id, status, amount, interval)
    except Exception as e:
        logger.error(f"Failed to record subscription: {e}")

    return {"status": "processed", "event_type": "customer.subscription.created",
            "subscription_id": sub_id, "amount": amount}


async def handle_subscription_updated(data: dict) -> dict:
    """Handle customer.subscription.updated - updates subscription status."""
    sub_id = data.get("id")
    status = data.get("status", "active")
    cancel_at = data.get("cancel_at")
    plan = data.get("plan", {})
    amount = plan.get("amount", 0) / 100

    logger.info(f"Subscription updated: {sub_id}, status={status}")

    await _record_stripe_event(
        "customer.subscription.updated", sub_id, None,
        int(amount * 100), plan.get("currency", "usd").upper(), status, data
    )

    try:
        pool = _get_db_pool()
        await pool.execute("""
            UPDATE mrg_subscriptions SET status = $2, amount = $3, updated_at = NOW()
            WHERE stripe_subscription_id = $1
        """, sub_id, status, amount)
    except Exception as e:
        logger.error(f"Failed to update subscription: {e}")

    return {"status": "processed", "event_type": "customer.subscription.updated",
            "subscription_id": sub_id}


async def handle_subscription_deleted(data: dict) -> dict:
    """Handle customer.subscription.deleted - marks subscription as canceled."""
    sub_id = data.get("id")
    logger.info(f"Subscription deleted: {sub_id}")

    await _record_stripe_event(
        "customer.subscription.deleted", sub_id, None, 0, "USD", "canceled", data
    )

    try:
        pool = _get_db_pool()
        await pool.execute("""
            UPDATE mrg_subscriptions SET status = 'canceled', canceled_at = NOW(), updated_at = NOW()
            WHERE stripe_subscription_id = $1
        """, sub_id)
    except Exception as e:
        logger.error(f"Failed to cancel subscription: {e}")

    return {"status": "processed", "event_type": "customer.subscription.deleted",
            "subscription_id": sub_id}


async def handle_invoice_paid(data: dict) -> dict:
    """Handle invoice.paid - records successful payment."""
    invoice_id = data.get("id")
    customer_email = data.get("customer_email")
    amount = data.get("amount_paid", 0) / 100
    sub_id = data.get("subscription")

    logger.info(f"Invoice paid: {invoice_id}, ${amount}, customer={customer_email}")

    await _record_stripe_event(
        "invoice.paid", invoice_id, customer_email,
        int(amount * 100), data.get("currency", "usd").upper(), "paid", data
    )

    if customer_email:
        await _update_lead_to_won(customer_email, amount)

    return {"status": "processed", "event_type": "invoice.paid",
            "amount": amount, "customer_email": customer_email}


async def handle_payment_failed(data: dict) -> dict:
    """Handle invoice.payment_failed - records failed payment for dunning."""
    invoice_id = data.get("id")
    customer_email = data.get("customer_email")
    amount = data.get("amount_due", 0) / 100
    attempt_count = data.get("attempt_count", 0)

    logger.warning(f"Payment failed: {invoice_id}, ${amount}, attempt={attempt_count}")

    await _record_stripe_event(
        "invoice.payment_failed", invoice_id, customer_email,
        int(amount * 100), data.get("currency", "usd").upper(), "failed", data
    )

    return {"status": "processed", "event_type": "invoice.payment_failed",
            "amount": amount, "attempt_count": attempt_count}


async def handle_charge_succeeded(data: dict) -> dict:
    """Handle charge.succeeded - records successful charge."""
    charge_id = data.get("id")
    amount = data.get("amount", 0) / 100
    customer_email = data.get("billing_details", {}).get("email")

    logger.info(f"Charge succeeded: {charge_id}, ${amount}")

    await _record_stripe_event(
        "charge.succeeded", charge_id, customer_email,
        int(amount * 100), data.get("currency", "usd").upper(), "succeeded", data
    )

    if customer_email:
        await _update_lead_to_won(customer_email, amount)

    return {"status": "processed", "event_type": "charge.succeeded", "amount": amount}


async def handle_charge_refunded(data: dict) -> dict:
    """Handle charge.refunded event."""
    charge_id = data.get("id")
    amount_refunded = data.get("amount_refunded", 0) / 100

    logger.info(f"Charge refunded: {charge_id}, ${amount_refunded}")

    await _record_stripe_event(
        "charge.refunded", charge_id, None,
        int(amount_refunded * 100), data.get("currency", "usd").upper(), "refunded", data
    )

    return {"status": "processed", "event_type": "charge.refunded",
            "amount_refunded": amount_refunded}


async def handle_dispute_created(data: dict) -> dict:
    """Handle charge.dispute.created event."""
    dispute_id = data.get("id")
    amount = data.get("amount", 0) / 100
    reason = data.get("reason")

    logger.warning(f"Dispute created: {dispute_id}, ${amount}, reason={reason}")

    await _record_stripe_event(
        "charge.dispute.created", dispute_id, None,
        int(amount * 100), data.get("currency", "usd").upper(), "disputed", data
    )

    return {"status": "processed", "event_type": "charge.dispute.created", "amount": amount}


@router.post("/webhook")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Stripe webhooks with signature verification."""
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature", "")

    event = verify_stripe_signature(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    result = await process_stripe_event(event, background_tasks)

    return {
        "received": True,
        "event_id": event.get("id"),
        "event_type": event.get("type"),
        **result
    }


@router.get("/health")
async def stripe_health():
    """Check Stripe webhook configuration status."""
    secret_source = None
    if os.getenv("STRIPE_WEBHOOK_SECRET_AIAGENTS"):
        secret_source = "STRIPE_WEBHOOK_SECRET_AIAGENTS"
    elif os.getenv("STRIPE_WEBHOOK_SECRET"):
        secret_source = "STRIPE_WEBHOOK_SECRET"

    return {
        "status": "healthy",
        "webhook_secret_configured": bool(STRIPE_WEBHOOK_SECRET),
        "secret_source": secret_source,
        "environment": ENVIRONMENT
    }

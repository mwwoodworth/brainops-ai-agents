"""
Stripe Webhook Handler
======================
Handles Stripe payment events with proper signature verification.
This endpoint is NOT protected by API key - it uses Stripe's signature verification.
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


def verify_stripe_signature(payload: bytes, sig_header: str, webhook_secret: str) -> dict:
    """
    Verify Stripe webhook signature and return the event.
    Uses Stripe's signature verification algorithm.
    """
    import hashlib
    import hmac
    import time

    if not webhook_secret:
        logger.error("STRIPE_WEBHOOK_SECRET_AIAGENTS not configured")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")

    # Parse the signature header
    # Format: t=timestamp,v1=signature
    elements = {}
    for element in sig_header.split(","):
        key, value = element.split("=", 1)
        elements[key] = value

    timestamp = elements.get("t")
    signature = elements.get("v1")

    if not timestamp or not signature:
        raise HTTPException(status_code=400, detail="Invalid signature format")

    # Check timestamp tolerance (5 minutes)
    try:
        timestamp_int = int(timestamp)
        if abs(time.time() - timestamp_int) > 300:
            raise HTTPException(status_code=400, detail="Timestamp too old")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp")

    # Compute expected signature
    signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
    expected_sig = hmac.new(
        webhook_secret.encode("utf-8"),
        signed_payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    # Constant-time comparison
    if not hmac.compare_digest(expected_sig, signature):
        logger.warning("Invalid Stripe signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Parse and return the event
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

    # INJECT INTO BRAIN MEMORY
    try:
        from unified_memory_manager import get_memory_manager, Memory, MemoryType
        
        manager = get_memory_manager()
        
        # Try to find tenant_id in metadata, otherwise system default
        tenant_id = data.get("metadata", {}).get("tenant_id") or "00000000-0000-0000-0000-000000000000"
        
        memory = Memory(
            memory_type=MemoryType.EPISODIC,
            content={
                "event_type": event_type,
                "stripe_id": event_id,
                "amount": data.get("amount", data.get("amount_total", 0)),
                "currency": data.get("currency", "usd"),
                "customer": data.get("customer") or data.get("customer_email"),
                "status": data.get("status")
            },
            source_system="stripe",
            source_agent="webhook_handler",
            created_by="system",
            importance_score=0.9 if "succeeded" in event_type or "paid" in event_type else 0.5,
            tags=["revenue", "stripe", event_type, "financial"],
            tenant_id=tenant_id
        )
        
        manager.store(memory)
        logger.info(f"🧠 Injected Stripe event {event_id} into Brain Memory")
        
    except Exception as e:
        logger.error(f"Failed to inject Stripe event into Brain Memory: {e}")

    try:
        # Handle different event types
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
        # Don't raise - we want to return 200 to Stripe to prevent retries
        return {"status": "error", "event_type": event_type, "error": str(e)}


async def handle_checkout_completed(data: dict, background_tasks: BackgroundTasks) -> dict:
    """Handle checkout.session.completed event."""
    customer_email = data.get("customer_email") or data.get("customer_details", {}).get("email")
    amount_total = data.get("amount_total", 0) / 100  # Convert from cents
    currency = data.get("currency", "usd").upper()
    payment_status = data.get("payment_status")
    session_id = data.get("id")

    logger.info(f"Checkout completed: {customer_email}, ${amount_total} {currency}")

    # Record in database
    try:
        from supabase_client import get_supabase_client
        supabase = await get_supabase_client()

        # 1. Record event
        await supabase.table("stripe_events").insert({
            "event_type": "checkout.session.completed",
            "stripe_id": session_id,
            "customer_email": customer_email,
            "amount_cents": int(amount_total * 100),
            "currency": currency,
            "status": payment_status,
            "metadata": data,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        # 2. Update Revenue Lead (The Synapse)
        if customer_email and payment_status == "paid":
            logger.info(f"Updating revenue lead for {customer_email} to WON")
            await supabase.table("revenue_leads").update({
                "stage": "won",
                "status": "customer",
                "value_estimate": amount_total,
                "converted_at": datetime.utcnow().isoformat(),
                "last_interaction_at": datetime.utcnow().isoformat()
            }).eq("email", customer_email).execute()

    except Exception as e:
        logger.error(f"Failed to record checkout/update lead: {e}")

    return {
        "status": "processed",
        "event_type": "checkout.session.completed",
        "customer_email": customer_email,
        "amount": amount_total
    }

# ... (rest of file) ...

async def handle_charge_succeeded(data: dict) -> dict:
    """Handle charge.succeeded event."""
    charge_id = data.get("id")
    amount = data.get("amount", 0) / 100
    customer_email = data.get("billing_details", {}).get("email")

    logger.info(f"Charge succeeded: {charge_id}, ${amount}")

    try:
        from supabase_client import get_supabase_client
        supabase = await get_supabase_client()

        # 1. Record event
        await supabase.table("stripe_events").insert({
            "event_type": "charge.succeeded",
            "stripe_id": charge_id,
            "customer_email": customer_email,
            "amount_cents": int(amount * 100),
            "currency": data.get("currency", "usd").upper(),
            "status": "succeeded",
            "metadata": data,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        # 2. Update Revenue Lead (The Synapse)
        if customer_email:
            logger.info(f"Updating revenue lead for {customer_email} to WON (Charge Succeeded)")
            # Only update if not already won? Or update value?
            # Let's simple update to WON to ensure conversion tracking.
            await supabase.table("revenue_leads").update({
                "stage": "won",
                "status": "customer",
                # Don't overwrite value if it's larger, but ensure it's tracked
                "last_interaction_at": datetime.utcnow().isoformat()
            }).eq("email", customer_email).execute()

    except Exception as e:
        logger.error(f"Failed to record charge: {e}")

    return {"status": "processed", "event_type": "charge.succeeded", "amount": amount}


async def handle_charge_refunded(data: dict) -> dict:
    """Handle charge.refunded event."""
    charge_id = data.get("id")
    amount_refunded = data.get("amount_refunded", 0) / 100

    logger.info(f"Charge refunded: {charge_id}, ${amount_refunded}")

    return {"status": "processed", "event_type": "charge.refunded", "amount_refunded": amount_refunded}


async def handle_dispute_created(data: dict) -> dict:
    """Handle charge.dispute.created event."""
    dispute_id = data.get("id")
    amount = data.get("amount", 0) / 100
    reason = data.get("reason")

    logger.warning(f"Dispute created: {dispute_id}, ${amount}, reason={reason}")

    return {"status": "processed", "event_type": "charge.dispute.created", "amount": amount}


@router.post("/webhook")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle Stripe webhooks with signature verification.

    This endpoint does NOT require API key authentication.
    Security is provided via Stripe's webhook signature verification.
    """
    # Get raw body and signature
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature", "")

    # Verify signature and parse event
    event = verify_stripe_signature(payload, sig_header, STRIPE_WEBHOOK_SECRET)

    # Process the event
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
    # Determine which env var is being used
    secret_source = None
    if os.getenv("STRIPE_WEBHOOK_SECRET_AIAGENTS"):
        secret_source = "STRIPE_WEBHOOK_SECRET_AIAGENTS"
    elif os.getenv("STRIPE_WEBHOOK_SECRET"):
        secret_source = "STRIPE_WEBHOOK_SECRET"

    return {
        "status": "healthy",
        "webhook_secret_REDACTED": bool(STRIPE_WEBHOOK_SECRET),
        "secret_source": secret_source,
        "secret_prefix": STRIPE_WEBHOOK_SECRET[:10] + "..." if STRIPE_WEBHOOK_SECRET else None,
        "environment": ENVIRONMENT
    }

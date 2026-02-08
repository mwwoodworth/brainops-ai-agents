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
import uuid
from datetime import datetime, timezone
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


def _parse_uuid(value: Any) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(str(value).strip())
    except Exception:
        return None


def _utc_from_unix(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except Exception:
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_tier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"starter", "professional", "enterprise", "free", "demo"}:
        return text
    if "starter" in text:
        return "starter"
    if "pro" in text or "professional" in text:
        return "professional"
    if "enterprise" in text:
        return "enterprise"
    if "free" in text:
        return "free"
    return text


def _extract_price_details(subscription: dict) -> tuple[str | None, float, str, str]:
    """
    Extract price_id, amount (USD), currency, interval from a Stripe subscription object.

    Stripe modern objects store pricing under `items.data[0].price`.
    Legacy objects sometimes include a top-level `plan`.
    """
    price_obj: dict[str, Any] = {}
    items = (subscription.get("items") or {}).get("data") or []
    if items and isinstance(items[0], dict):
        price_obj = items[0].get("price") or items[0].get("plan") or {}
    if not price_obj:
        price_obj = subscription.get("plan") or {}

    price_id = price_obj.get("id")
    unit_amount = price_obj.get("unit_amount")
    if unit_amount is None:
        unit_amount = price_obj.get("amount")
    amount = float(unit_amount or 0) / 100

    currency = str(price_obj.get("currency") or "usd").upper()
    recurring = price_obj.get("recurring") or {}
    interval = str(recurring.get("interval") or price_obj.get("interval") or "month")
    return price_id, amount, currency, interval


async def _resolve_plan_name_and_tier(price_id: str | None) -> tuple[str | None, str | None]:
    if not price_id:
        return None, None
    try:
        pool = _get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT name
            FROM mrg_subscription_plans
            WHERE stripe_price_id_monthly = $1 OR stripe_price_id_yearly = $1
            LIMIT 1
            """,
            price_id,
        )
        if not row:
            return None, None
        name = str(row["name"] or "").strip()
        return name, _normalize_tier(name)
    except Exception as e:
        logger.error(f"Failed to resolve plan for price_id={price_id}: {e}")
        return None, None


async def _resolve_tenant_id(metadata: dict[str, Any] | None, stripe_customer_id: str | None) -> uuid.UUID | None:
    md = metadata or {}
    for key in ("tenant_id", "tenantId", "tenant", "tenant_uuid", "tenantUUID"):
        tid = _parse_uuid(md.get(key))
        if tid:
            return tid
    if stripe_customer_id:
        try:
            pool = _get_db_pool()
            tenant_id = await pool.fetchval(
                "SELECT id FROM tenants WHERE stripe_customer_id = $1 LIMIT 1",
                stripe_customer_id,
            )
            if tenant_id:
                return tenant_id
        except Exception as e:
            logger.error(f"Failed to resolve tenant by stripe_customer_id={stripe_customer_id}: {e}")
    return None


async def _upsert_subscription_row(
    *,
    stripe_subscription_id: str,
    stripe_customer_id: str | None,
    tenant_id: uuid.UUID | None,
    status: str | None,
    amount: float | None,
    currency: str | None,
    interval: str | None,
    current_period_start: datetime | None,
    current_period_end: datetime | None,
    cancel_at: datetime | None,
    trial_end: datetime | None,
    livemode: bool,
    plan_id: str | None,
    plan_name: str | None,
    plan_tier: str | None,
    metadata: dict[str, Any] | None,
) -> None:
    pool = _get_db_pool()
    is_test = not livemode

    payload = {
        "stripe": {
            "subscription_id": stripe_subscription_id,
            "customer_id": stripe_customer_id,
            "livemode": livemode,
            "plan_id": plan_id,
            "plan_name": plan_name,
        },
        "metadata": metadata or {},
    }

    try:
        existing = await pool.fetchrow(
            "SELECT id FROM subscriptions WHERE stripe_subscription_id = $1 LIMIT 1",
            stripe_subscription_id,
        )
        if existing:
            await pool.execute(
                """
                UPDATE subscriptions
                SET
                    stripe_customer_id = COALESCE($2, stripe_customer_id),
                    tenant_id = COALESCE($3, tenant_id),
                    status = COALESCE($4, status),
                    amount = COALESCE($5, amount),
                    billing_cycle = COALESCE($6, billing_cycle),
                    current_period_start = COALESCE($7, current_period_start),
                    current_period_end = COALESCE($8, current_period_end),
                    cancel_at = COALESCE($9, cancel_at),
                    trial_end = COALESCE($10, trial_end),
                    is_test = $11,
                    plan_id = COALESCE($12, plan_id),
                    plan_name = COALESCE($13, plan_name),
                    meta_data = COALESCE($14::jsonb, meta_data),
                    updated_at = NOW()
                WHERE stripe_subscription_id = $1
                """,
                stripe_subscription_id,
                stripe_customer_id,
                tenant_id,
                status,
                amount,
                interval,
                current_period_start,
                current_period_end,
                cancel_at,
                trial_end,
                is_test,
                plan_id,
                plan_tier or plan_name,
                json.dumps(payload, default=str),
            )
            return

        await pool.execute(
            """
            INSERT INTO subscriptions (
                stripe_subscription_id,
                stripe_customer_id,
                tenant_id,
                status,
                amount,
                billing_cycle,
                current_period_start,
                current_period_end,
                cancel_at,
                trial_end,
                is_test,
                plan_id,
                plan_name,
                meta_data
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14::jsonb
            )
            """,
            stripe_subscription_id,
            stripe_customer_id,
            tenant_id,
            status,
            amount,
            interval,
            current_period_start,
            current_period_end,
            cancel_at,
            trial_end,
            is_test,
            plan_id,
            plan_tier or plan_name,
            json.dumps(payload, default=str),
        )
    except Exception as e:
        logger.error(f"Failed to upsert subscriptions row for {stripe_subscription_id}: {e}")


async def _update_tenant_subscription(
    *,
    tenant_id: uuid.UUID,
    stripe_customer_id: str | None,
    stripe_subscription_id: str | None,
    status: str | None,
    plan_tier: str | None,
    livemode: bool,
) -> None:
    pool = _get_db_pool()
    try:
        await pool.execute(
            """
            UPDATE tenants
            SET
                stripe_customer_id = COALESCE($2, stripe_customer_id),
                stripe_subscription_id = COALESCE($3, stripe_subscription_id),
                subscription_status = COALESCE($4, subscription_status),
                subscription_tier = COALESCE($5, subscription_tier),
                is_test = $6,
                updated_at = NOW()
            WHERE id = $1
            """,
            tenant_id,
            stripe_customer_id,
            stripe_subscription_id,
            status,
            plan_tier,
            (not livemode),
        )
    except Exception as e:
        logger.error(f"Failed to update tenant subscription for tenant_id={tenant_id}: {e}")


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
    livemode = _coerce_bool(event.get("livemode"))
    if isinstance(data, dict) and "livemode" not in data:
        data["livemode"] = livemode

    logger.info(f"Processing Stripe event: {event_type} ({event_id})")

    try:
        if event_type == "checkout.session.completed":
            return await handle_checkout_completed(data, background_tasks, livemode=livemode)
        elif event_type == "customer.subscription.created":
            return await handle_subscription_created(data, livemode=livemode)
        elif event_type == "customer.subscription.updated":
            return await handle_subscription_updated(data, livemode=livemode)
        elif event_type == "customer.subscription.deleted":
            return await handle_subscription_deleted(data, livemode=livemode)
        elif event_type in {"invoice.paid", "invoice.payment_succeeded"}:
            return await handle_invoice_paid(data, livemode=livemode)
        elif event_type == "invoice.payment_failed":
            return await handle_payment_failed(data, livemode=livemode)
        elif event_type == "charge.succeeded":
            return await handle_charge_succeeded(data, livemode=livemode)
        elif event_type == "charge.refunded":
            return await handle_charge_refunded(data, livemode=livemode)
        elif event_type == "charge.dispute.created":
            return await handle_dispute_created(data, livemode=livemode)
        else:
            logger.info(f"Unhandled event type: {event_type}")
            return {"status": "ignored", "event_type": event_type}

    except Exception as e:
        logger.error(f"Error processing {event_type}: {e}")
        return {"status": "error", "event_type": event_type, "error": str(e)}


async def handle_checkout_completed(data: dict, background_tasks: BackgroundTasks, *, livemode: bool) -> dict:
    """Handle checkout.session.completed - records event and updates lead to WON."""
    customer_email = data.get("customer_email") or data.get("customer_details", {}).get("email")
    amount_total = data.get("amount_total", 0) / 100
    currency = data.get("currency", "usd").upper()
    payment_status = data.get("payment_status")
    session_id = data.get("id")
    metadata = data.get("metadata") or {}

    logger.info(f"Checkout completed: {customer_email}, ${amount_total} {currency}")

    await _record_stripe_event(
        "checkout.session.completed", session_id, customer_email,
        int(amount_total * 100), currency, payment_status or "completed", data
    )

    if customer_email and payment_status == "paid":
        await _update_lead_to_won(customer_email, amount_total)

    # Best-effort: attribute Stripe customer/subscription IDs to a tenant when metadata includes tenant_id.
    stripe_customer_id = data.get("customer")
    stripe_subscription_id = data.get("subscription")
    tenant_id = await _resolve_tenant_id(metadata if isinstance(metadata, dict) else None, stripe_customer_id)
    plan_tier = _normalize_tier(
        (metadata or {}).get("subscription_tier")
        or (metadata or {}).get("subscriptionTier")
        or (metadata or {}).get("tier")
    )
    if tenant_id:
        await _update_tenant_subscription(
            tenant_id=tenant_id,
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=stripe_subscription_id,
            status="active" if payment_status == "paid" else None,
            plan_tier=plan_tier,
            livemode=livemode,
        )
        if stripe_subscription_id:
            # Create a placeholder subscription row so downstream dashboards have a stable mapping immediately.
            await _upsert_subscription_row(
                stripe_subscription_id=stripe_subscription_id,
                stripe_customer_id=stripe_customer_id,
                tenant_id=tenant_id,
                status="active" if payment_status == "paid" else None,
                amount=None,
                currency=currency,
                interval=None,
                current_period_start=None,
                current_period_end=None,
                cancel_at=None,
                trial_end=None,
                livemode=livemode,
                plan_id=None,
                plan_name=None,
                plan_tier=plan_tier,
                metadata=metadata if isinstance(metadata, dict) else None,
            )

    return {
        "status": "processed",
        "event_type": "checkout.session.completed",
        "customer_email": customer_email,
        "amount": amount_total
    }


async def handle_subscription_created(data: dict, *, livemode: bool) -> dict:
    """Handle customer.subscription.created - records new subscription."""
    sub_id = data.get("id")
    status = data.get("status", "active")
    stripe_customer_id = data.get("customer")
    price_id, amount, currency, interval = _extract_price_details(data)
    metadata = data.get("metadata") or {}
    tenant_id = await _resolve_tenant_id(metadata if isinstance(metadata, dict) else None, stripe_customer_id)
    plan_name, plan_tier = await _resolve_plan_name_and_tier(price_id)

    logger.info(f"Subscription created: {sub_id}, ${amount}/{interval}")

    await _record_stripe_event(
        "customer.subscription.created", sub_id, None,
        int(amount * 100), currency, status, data
    )

    await _upsert_subscription_row(
        stripe_subscription_id=sub_id,
        stripe_customer_id=stripe_customer_id,
        tenant_id=tenant_id,
        status=status,
        amount=amount,
        currency=currency,
        interval=interval,
        current_period_start=_utc_from_unix(data.get("current_period_start")),
        current_period_end=_utc_from_unix(data.get("current_period_end")),
        cancel_at=_utc_from_unix(data.get("cancel_at")),
        trial_end=_utc_from_unix(data.get("trial_end")),
        livemode=livemode,
        plan_id=price_id,
        plan_name=plan_name,
        plan_tier=plan_tier,
        metadata=metadata if isinstance(metadata, dict) else None,
    )
    if tenant_id:
        await _update_tenant_subscription(
            tenant_id=tenant_id,
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=sub_id,
            status=status,
            plan_tier=plan_tier,
            livemode=livemode,
        )

    return {"status": "processed", "event_type": "customer.subscription.created",
            "subscription_id": sub_id, "amount": amount}


async def handle_subscription_updated(data: dict, *, livemode: bool) -> dict:
    """Handle customer.subscription.updated - updates subscription status."""
    sub_id = data.get("id")
    status = data.get("status", "active")
    stripe_customer_id = data.get("customer")
    price_id, amount, currency, interval = _extract_price_details(data)
    metadata = data.get("metadata") or {}
    tenant_id = await _resolve_tenant_id(metadata if isinstance(metadata, dict) else None, stripe_customer_id)
    plan_name, plan_tier = await _resolve_plan_name_and_tier(price_id)

    logger.info(f"Subscription updated: {sub_id}, status={status}")

    await _record_stripe_event(
        "customer.subscription.updated", sub_id, None,
        int(amount * 100), currency, status, data
    )

    await _upsert_subscription_row(
        stripe_subscription_id=sub_id,
        stripe_customer_id=stripe_customer_id,
        tenant_id=tenant_id,
        status=status,
        amount=amount,
        currency=currency,
        interval=interval,
        current_period_start=_utc_from_unix(data.get("current_period_start")),
        current_period_end=_utc_from_unix(data.get("current_period_end")),
        cancel_at=_utc_from_unix(data.get("cancel_at")),
        trial_end=_utc_from_unix(data.get("trial_end")),
        livemode=livemode,
        plan_id=price_id,
        plan_name=plan_name,
        plan_tier=plan_tier,
        metadata=metadata if isinstance(metadata, dict) else None,
    )
    if tenant_id:
        await _update_tenant_subscription(
            tenant_id=tenant_id,
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=sub_id,
            status=status,
            plan_tier=plan_tier,
            livemode=livemode,
        )

    return {"status": "processed", "event_type": "customer.subscription.updated",
            "subscription_id": sub_id}


async def handle_subscription_deleted(data: dict, *, livemode: bool) -> dict:
    """Handle customer.subscription.deleted - marks subscription as canceled."""
    sub_id = data.get("id")
    stripe_customer_id = data.get("customer")
    metadata = data.get("metadata") or {}
    tenant_id = await _resolve_tenant_id(metadata if isinstance(metadata, dict) else None, stripe_customer_id)
    logger.info(f"Subscription deleted: {sub_id}")

    await _record_stripe_event(
        "customer.subscription.deleted", sub_id, None, 0, "USD", "canceled", data
    )

    await _upsert_subscription_row(
        stripe_subscription_id=sub_id,
        stripe_customer_id=stripe_customer_id,
        tenant_id=tenant_id,
        status="canceled",
        amount=None,
        currency=None,
        interval=None,
        current_period_start=_utc_from_unix(data.get("current_period_start")),
        current_period_end=_utc_from_unix(data.get("current_period_end")),
        cancel_at=_utc_from_unix(data.get("cancel_at")) or datetime.now(timezone.utc),
        trial_end=_utc_from_unix(data.get("trial_end")),
        livemode=livemode,
        plan_id=None,
        plan_name=None,
        plan_tier=None,
        metadata=metadata if isinstance(metadata, dict) else None,
    )
    if tenant_id:
        await _update_tenant_subscription(
            tenant_id=tenant_id,
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=sub_id,
            status="canceled",
            plan_tier=None,
            livemode=livemode,
        )

    return {"status": "processed", "event_type": "customer.subscription.deleted",
            "subscription_id": sub_id}


async def handle_invoice_paid(data: dict, *, livemode: bool) -> dict:
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

    if sub_id:
        try:
            pool = _get_db_pool()
            await pool.execute(
                "UPDATE subscriptions SET last_payment_at = NOW(), updated_at = NOW() WHERE stripe_subscription_id = $1",
                sub_id,
            )
        except Exception as e:
            logger.error(f"Failed to update last_payment_at for subscription {sub_id}: {e}")

    return {"status": "processed", "event_type": "invoice.paid",
            "amount": amount, "customer_email": customer_email}


async def handle_payment_failed(data: dict, *, livemode: bool) -> dict:
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

    sub_id = data.get("subscription")
    if sub_id:
        try:
            pool = _get_db_pool()
            await pool.execute(
                "UPDATE subscriptions SET status = 'past_due', updated_at = NOW() WHERE stripe_subscription_id = $1",
                sub_id,
            )
        except Exception as e:
            logger.error(f"Failed to update subscription past_due for {sub_id}: {e}")

    return {"status": "processed", "event_type": "invoice.payment_failed",
            "amount": amount, "attempt_count": attempt_count}


async def handle_charge_succeeded(data: dict, *, livemode: bool) -> dict:
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


async def handle_charge_refunded(data: dict, *, livemode: bool) -> dict:
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


async def handle_dispute_created(data: dict, *, livemode: bool) -> dict:
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

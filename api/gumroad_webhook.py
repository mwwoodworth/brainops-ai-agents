"""
Gumroad Sales Funnel Webhook Integration
Handles Gumroad sales and integrates with ConvertKit, Stripe, SendGrid, and Supabase
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from config import config
from utils.outbound import email_block_reason

# Configuration
CONVERTKIT_API_KEY = os.getenv("CONVERTKIT_API_KEY", "")
CONVERTKIT_API_SECRET = os.getenv("CONVERTKIT_API_SECRET", "")
CONVERTKIT_FORM_ID = os.getenv("CONVERTKIT_FORM_ID", "")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")
GUMROAD_WEBHOOK_SECRET = os.getenv("GUMROAD_WEBHOOK_SECRET", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production").strip().lower()

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/gumroad",
    tags=["gumroad", "income", "sales"]
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def _require_internal_api_key(api_key: str = Security(_api_key_header)) -> bool:
    """
    Protect non-webhook Gumroad endpoints.

    `/gumroad/webhook` must remain publicly reachable for Gumroad, and is protected
    via HMAC signature verification. Everything else is internal-only.
    """
    if not config.security.auth_required:
        return True

    provided = (api_key or "").strip()
    if not provided or provided not in config.security.valid_api_keys:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True

def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}

def _parse_gumroad_timestamp(value: Any) -> datetime:
    """
    Gumroad commonly sends ISO timestamps with a trailing 'Z' (UTC).
    Python's datetime.fromisoformat() does not accept 'Z', so normalize it.
    """
    if value is None:
        return datetime.now(timezone.utc)
    text = str(value).strip()
    if not text:
        return datetime.now(timezone.utc)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        # Last-resort: accept unix seconds (string/number)
        try:
            parsed = datetime.fromtimestamp(float(text), tz=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed

def _parse_price(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    text = str(value).strip().replace("$", "").replace(",", "")
    if not text:
        return Decimal("0")
    try:
        return Decimal(text)
    except Exception:
        return Decimal("0")

def _is_test_sale(sale_data: dict[str, Any]) -> bool:
    # Gumroad includes a `test` field for test purchases.
    if _coerce_bool(sale_data.get("test")):
        return True
    sale_id = str(sale_data.get("sale_id") or "").strip().upper()
    if sale_id.startswith("TEST-"):
        return True
    email = str(sale_data.get("email") or "").strip().lower()
    if any(token in email for token in ("test", "example", "demo", "sample", "fake", "placeholder", "localhost")):
        return True
    return False

# Product mapping - keyed by Gumroad permalink (uppercase) or product code
PRODUCT_MAPPING = {
    # === LIVE GUMROAD PRODUCTS (Published) - Prices synced 2026-01-15 ===
    'HJHMSM': {
        'name': 'MCP Server Starter Kit',
        'price': 49,  # Live price on Gumroad
        'type': 'code_kit',
        'convertkit_tag': 'mcp-kit-buyer',
        'gumroad_url': 'https://woodworthia.gumroad.com/l/hjhmsm'
    },
    'VJXCEW': {
        'name': 'SaaS Automation Scripts',
        'price': 37,  # Live price on Gumroad
        'type': 'code_kit',
        'convertkit_tag': 'saas-scripts-buyer',
        'gumroad_url': 'https://woodworthia.gumroad.com/l/vjxcew'
    },
    'XGFKP': {
        'name': 'AI Prompt Engineering Pack',
        'price': 29,  # Live price on Gumroad
        'type': 'prompt_pack',
        'convertkit_tag': 'ai-prompts-buyer',
        'gumroad_url': 'https://woodworthia.gumroad.com/l/xgfkp'
    },
    'GSAAVB': {
        'name': 'AI Orchestration Framework',
        'price': 97,  # Live price on Gumroad
        'type': 'code_kit',
        'convertkit_tag': 'ai-orchestration-buyer',
        'gumroad_url': 'https://woodworthia.gumroad.com/l/gsaavb'
    },
    'UPSYKR': {
        'name': 'Command Center UI Kit',
        'price': 149,  # Live price on Gumroad
        'type': 'code_kit',
        'convertkit_tag': 'ui-kit-buyer',
        'gumroad_url': 'https://woodworthia.gumroad.com/l/upsykr'
    },
    'CAWVO': {
        'name': 'Business Automation Toolkit',
        'price': 29,  # Live price on Gumroad
        'type': 'prompt_pack',
        'convertkit_tag': 'business-toolkit-buyer',
        'gumroad_url': 'https://woodworthia.gumroad.com/l/cawvo'
    },
    # === LEGACY/FUTURE PRODUCTS ===
    'GR-ROOFINT': {
        'name': 'Commercial Roofing Intelligence Bundle',
        'price': 97,
        'type': 'prompt_pack',
        'convertkit_tag': 'roofing-intelligence-buyer'
    },
    'GR-PMACC': {
        'name': 'AI-Enhanced Project Management Accelerator',
        'price': 127,
        'type': 'prompt_pack',
        'convertkit_tag': 'pm-accelerator-buyer'
    },
    'GR-LAUNCH': {
        'name': 'Digital Product Launch Optimizer',
        'price': 147,
        'type': 'prompt_pack',
        'convertkit_tag': 'launch-optimizer-buyer'
    },
    'GR-ONBOARD': {
        'name': 'Intelligent Client Onboarding System',
        'price': 297,
        'type': 'automation',
        'convertkit_tag': 'onboarding-system-buyer'
    },
    'GR-CONTENT': {
        'name': 'AI-Powered Content Production Pipeline',
        'price': 347,
        'type': 'automation',
        'convertkit_tag': 'content-pipeline-buyer'
    },
    'GR-ROOFVAL': {
        'name': 'Commercial Roofing Estimation Validator',
        'price': 497,
        'type': 'automation',
        'convertkit_tag': 'roofing-validator-buyer'
    },
    'GR-ERP-START': {
        'name': 'SaaS ERP Starter Kit',
        'price': 197,
        'type': 'code_kit',
        'convertkit_tag': 'erp-starter-buyer'
    },
    'GR-AI-ORCH': {
        'name': 'BrainOps AI Orchestrator Framework',
        'price': 147,
        'type': 'code_kit',
        'convertkit_tag': 'ai-orchestrator-buyer'
    },
    'GR-UI-KIT': {
        'name': 'Modern Command Center UI Kit',
        'price': 97,
        'type': 'code_kit',
        'convertkit_tag': 'ui-kit-buyer'
    },
    'GR-ULTIMATE': {
        'name': 'Ultimate All-Access Bundle',
        'price': 997,
        'type': 'bundle',
        'convertkit_tag': 'ultimate-bundle-buyer'
    }
}

class GumroadSale(BaseModel):
    email: str
    full_name: Optional[str] = ""
    product_name: Optional[str] = ""
    product_permalink: Optional[str] = ""
    sale_id: Optional[str] = ""
    sale_timestamp: Optional[str] = ""
    price: Optional[str] = ""
    currency: Optional[str] = "USD"
    download_url: Optional[str] = ""

def _as_convertkit_tag_name(value: str) -> str:
    return value.strip().replace("_", " ").replace("-", " ").strip()

async def _resolve_convertkit_tag_id(client: httpx.AsyncClient, tag_ref: str) -> Optional[str]:
    """
    ConvertKit tag references can be either:
    - A numeric tag id (preferred)
    - A human-readable tag name (we'll look up / create)
    """
    tag_ref = (tag_ref or "").strip()
    if not tag_ref:
        return None
    if tag_ref.isdigit():
        return tag_ref

    if not CONVERTKIT_API_KEY:
        logger.warning("ConvertKit tagging requested but CONVERTKIT_API_KEY is missing")
        return None

    # Lookup by name
    try:
        response = await client.get(
            "https://api.convertkit.com/v3/tags",
            params={"api_key": CONVERTKIT_API_KEY},
        )
        if response.status_code == 200:
            data = response.json()
            tags = data.get("tags", []) if isinstance(data, dict) else []
            desired = _as_convertkit_tag_name(tag_ref).lower()
            for tag in tags:
                if not isinstance(tag, dict):
                    continue
                if str(tag.get("name", "")).strip().lower() == desired:
                    tag_id = tag.get("id")
                    if tag_id is not None:
                        return str(tag_id)
        else:
            logger.warning("ConvertKit tag lookup failed: HTTP %s", response.status_code)
    except Exception as exc:
        logger.warning("ConvertKit tag lookup error: %s", exc)

    # Create tag if API secret is available
    if not CONVERTKIT_API_SECRET:
        logger.warning("ConvertKit tag '%s' missing and CONVERTKIT_API_SECRET not set (cannot create)", tag_ref)
        return None

    try:
        create_resp = await client.post(
            "https://api.convertkit.com/v3/tags",
            data={"api_secret": CONVERTKIT_API_SECRET, "tag[name]": _as_convertkit_tag_name(tag_ref)},
        )
        if create_resp.status_code in (200, 201):
            payload = create_resp.json()
            tag = payload.get("tag") if isinstance(payload, dict) else None
            if isinstance(tag, dict) and tag.get("id") is not None:
                return str(tag.get("id"))
        logger.warning("ConvertKit tag create failed: HTTP %s %s", create_resp.status_code, create_resp.text)
    except Exception as exc:
        logger.warning("ConvertKit tag create error: %s", exc)

    return None

async def add_to_convertkit(email: str, first_name: str, last_name: str, product_code: str):
    """Add subscriber to ConvertKit with product tagging"""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            # Subscribe to form
            if not (CONVERTKIT_API_KEY and CONVERTKIT_FORM_ID):
                logger.warning("ConvertKit not configured; skipping ConvertKit sync for %s", email)
                return False

            response = await client.post(
                f"https://api.convertkit.com/v3/forms/{CONVERTKIT_FORM_ID}/subscribe",
                data={
                    "api_key": CONVERTKIT_API_KEY,
                    "email": email,
                    "first_name": first_name,
                    "fields[last_name]": last_name,
                    "fields[gumroad_customer]": "true",
                    "fields[last_purchase]": datetime.utcnow().isoformat(),
                    "fields[purchased_products]": product_code,
                },
            )

            if response.status_code == 200:
                # Add product tag
                product = PRODUCT_MAPPING.get(product_code)
                if product and "convertkit_tag" in product:
                    tag_ref = str(product.get("convertkit_tag") or "").strip()
                    tag_id = await _resolve_convertkit_tag_id(client, tag_ref)
                    if tag_id:
                        await client.post(
                            f"https://api.convertkit.com/v3/tags/{tag_id}/subscribe",
                            data={"api_key": CONVERTKIT_API_KEY, "email": email},
                        )
                    else:
                        logger.warning("ConvertKit tag not available for product_code=%s tag_ref=%s", product_code, tag_ref)
                logger.info(f"Added {email} to ConvertKit with tag {product_code}")
                return True
            else:
                logger.error(f"ConvertKit error: {response.text}")
                return False

    except Exception as e:
        logger.error(f"ConvertKit integration error: {e}")
        return False

async def record_sale_to_database(sale_data: dict[str, Any]):
    """Record sale in Supabase database"""
    try:
        from database.async_connection import get_pool

        pool = get_pool()
        is_test = _is_test_sale(sale_data)
        sale_timestamp = _parse_gumroad_timestamp(sale_data.get("sale_timestamp"))
        await pool.execute("""
            INSERT INTO gumroad_sales (
                sale_id, email, customer_name, product_code,
                product_name, price, currency, sale_timestamp,
                convertkit_synced, stripe_synced, sendgrid_sent, is_test, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (sale_id) DO UPDATE SET
                email = EXCLUDED.email,
                customer_name = EXCLUDED.customer_name,
                product_code = EXCLUDED.product_code,
                product_name = EXCLUDED.product_name,
                price = EXCLUDED.price,
                currency = EXCLUDED.currency,
                sale_timestamp = EXCLUDED.sale_timestamp,
                metadata = EXCLUDED.metadata,
                is_test = EXCLUDED.is_test,
                convertkit_synced = gumroad_sales.convertkit_synced OR EXCLUDED.convertkit_synced,
                stripe_synced = gumroad_sales.stripe_synced OR EXCLUDED.stripe_synced,
                sendgrid_sent = gumroad_sales.sendgrid_sent OR EXCLUDED.sendgrid_sent,
                updated_at = NOW()
        """,
            sale_data.get('sale_id'),
            sale_data.get('email'),
            sale_data.get('full_name'),
            sale_data.get('product_code'),
            sale_data.get('product_name'),
            _parse_price(sale_data.get("price")),
            sale_data.get('currency', 'USD'),
            sale_timestamp,
            False,  # convertkit_synced (set true only after confirmed)
            False,  # stripe_synced
            False,  # sendgrid_sent
            is_test,
            sale_data
        )
        logger.info(f"Recorded sale {sale_data.get('sale_id')} to database")
        return True

    except Exception as e:
        logger.error(f"Database recording error: {e}")
        return False

async def _update_sale_flags(
    sale_id: str,
    *,
    convertkit_synced: Optional[bool] = None,
    stripe_synced: Optional[bool] = None,
    sendgrid_sent: Optional[bool] = None,
) -> None:
    if not sale_id:
        return

    updates: list[str] = []
    params: list[Any] = [sale_id]

    def _add_flag(column: str, value: Optional[bool]) -> None:
        if value is None:
            return
        if value is True:
            params.append(True)
            updates.append(f"{column} = {column} OR ${len(params)}")
        else:
            # Never downgrade flags (keep audit semantics monotonic)
            return

    _add_flag("convertkit_synced", convertkit_synced)
    _add_flag("stripe_synced", stripe_synced)
    _add_flag("sendgrid_sent", sendgrid_sent)

    if not updates:
        return

    from database.async_connection import get_pool

    pool = get_pool()
    query = f"""
        UPDATE gumroad_sales
        SET {", ".join(updates)}, updated_at = NOW()
        WHERE sale_id = $1
    """
    await pool.execute(query, *params)

async def send_purchase_email(email: str, name: str, product_name: str, download_url: str):
    """Send purchase confirmation via Resend (primary) or SendGrid (fallback)."""

    block_reason = email_block_reason(email, {"source": "gumroad_webhook", "product": product_name})
    if block_reason:
        logger.warning("Purchase email blocked (%s) for %s", block_reason, email)
        return False

    html_body = f"""
      <h2>Hi {name or 'there'},</h2>
      <p>Thank you for purchasing <strong>{product_name}</strong>!</p>
      <p><a href="{download_url}" style="background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Download Your Product</a></p>
      <p>If you have any questions, reply to this email.</p>
      <p>Best,<br>BrainOps Team</p>
    """

    try:
        if RESEND_API_KEY:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    "https://api.resend.com/emails",
                    headers={
                        "Authorization": f"Bearer {RESEND_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "from": RESEND_FROM_EMAIL,
                        "to": [email],
                        "subject": f"Your {product_name} is ready!",
                        "html": html_body,
                    },
                )

            if response.status_code in (200, 201):
                logger.info("Sent purchase email via Resend to %s", email)
                return True
            logger.error("Resend error: HTTP %s %s", response.status_code, response.text)
            return False

        if not SENDGRID_API_KEY:
            logger.warning("No email provider configured for purchase email (need RESEND_API_KEY or SENDGRID_API_KEY)")
            return False

        async with httpx.AsyncClient(timeout=20.0) as client:
            email_data = {
                "personalizations": [{
                    "to": [{"email": email}],
                    "subject": f"Your {product_name} is ready!"
                }],
                "from": {"email": SENDGRID_FROM_EMAIL or "support@myroofgenius.com"},
                "content": [{
                    "type": "text/html",
                    "value": html_body,
                }]
            }

            response = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={
                    "Authorization": f"Bearer {SENDGRID_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=email_data
            )

        if response.status_code in [200, 202]:
            logger.info("Sent purchase email via SendGrid to %s", email)
            return True
        logger.error("SendGrid error: %s", response.text)
        return False

    except Exception as e:
        logger.error(f"Email sending error: {e}")
        return False

@router.post("/webhook")
async def handle_gumroad_webhook(request: Request, background_tasks: BackgroundTasks):
    """Main webhook handler for Gumroad sales"""
    try:
        # Get raw body for signature verification
        body = await request.body()

        # Webhook signature verification is mandatory in production.
        signature_required = ENVIRONMENT == "production"
        if signature_required and not GUMROAD_WEBHOOK_SECRET:
            logger.critical("GUMROAD_WEBHOOK_SECRET is not set; refusing to process webhooks in production.")
            raise HTTPException(status_code=503, detail="Webhook not configured")

        if GUMROAD_WEBHOOK_SECRET:
            signature = request.headers.get("x-gumroad-signature")
            if not signature:
                raise HTTPException(status_code=401, detail="Missing signature")

            expected_sig = hmac.new(
                GUMROAD_WEBHOOK_SECRET.encode(),
                body,
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_sig):
                logger.warning("Invalid webhook signature")
                raise HTTPException(status_code=401, detail="Invalid signature")
        elif signature_required:
            raise HTTPException(status_code=503, detail="Webhook not configured")

        # Parse webhook data (Gumroad commonly posts form-encoded payloads).
        content_type = (request.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            raw_data = await request.json()
        else:
            form = await request.form()
            raw_data = dict(form)

        if not isinstance(raw_data, dict):
            raise HTTPException(status_code=400, detail="Invalid payload")

        sale = GumroadSale(**raw_data)

        # Extract product code
        product_code = sale.product_permalink.upper() if sale.product_permalink else ""
        if not product_code and sale.product_name:
            for code in PRODUCT_MAPPING.keys():
                if code in sale.product_name.upper():
                    product_code = code
                    break

        # Parse customer name
        name_parts = sale.full_name.split(" ") if sale.full_name else []
        first_name = name_parts[0] if name_parts else ""
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        redacted_email = (
            f"{sale.email[:1]}***@{sale.email.split('@')[-1]}"
            if sale.email and "@" in sale.email
            else "<missing>"
        )
        logger.info(f"Processing Gumroad sale: {sale.sale_id} for {redacted_email} - Product: {product_code}")

        # Persist the full raw payload in metadata (includes fields like `test`)
        sale_dict = {**raw_data, **sale.dict(), "product_code": product_code}

        # Always record to DB synchronously (fail closed) so Gumroad retries on failure.
        recorded = await record_sale_to_database(sale_dict)
        if not recorded:
            raise HTTPException(status_code=500, detail="Failed to record sale")

        # Automations (ConvertKit, nurture sequences, upsells) are explicitly gated.
        automations_enabled = os.getenv("GUMROAD_AUTOMATIONS_ENABLED", "").strip().lower() in {"1", "true", "yes"}
        if automations_enabled and not _is_test_sale(sale_dict):
            background_tasks.add_task(
                process_sale,
                sale_dict,
                product_code,
                first_name,
                last_name
            )

        return {
            "success": True,
            "sale_id": sale.sale_id,
            "recorded": True,
            "automations_enabled": automations_enabled,
            "message": "Sale recorded"
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Webhook processing error")
        raise HTTPException(status_code=500, detail="Internal error") from None

async def enroll_in_nurture_sequence(
    email: str,
    first_name: str,
    product_name: str,
    product_type: str,
    product_code: str,
    download_url: str
) -> bool:
    """Enroll buyer in post-purchase nurture sequence"""
    try:
        from gumroad_sequences import enroll_buyer_in_sequence

        email_ids = await enroll_buyer_in_sequence(
            email=email,
            first_name=first_name,
            product_name=product_name,
            product_type=product_type,
            product_code=product_code,
            download_url=download_url
        )

        if email_ids:
            logger.info(f"Enrolled {email} in nurture sequence: {len(email_ids)} emails scheduled")
            return True
        return False

    except Exception as e:
        logger.error(f"Failed to enroll in nurture sequence: {e}")
        return False


async def process_sale(sale_data: dict[str, Any], product_code: str, first_name: str, last_name: str):
    """Process sale through all systems"""
    # Safety: do not run post-purchase automations for test purchases.
    if _is_test_sale(sale_data):
        logger.info("Skipping Gumroad automations for test sale_id=%s", sale_data.get("sale_id"))
        return

    product_info = PRODUCT_MAPPING.get(product_code, {})
    product_name = product_info.get('name', sale_data.get('product_name', 'Product'))
    product_type = product_info.get('type', 'code_kit')
    download_url = sale_data.get('download_url', 'https://gumroad.com/library')

    # Import upsell engine
    try:
        from upsell_engine import process_purchase_for_upsell
        upsell_task = process_purchase_for_upsell(
            email=sale_data['email'],
            first_name=first_name,
            product_code=product_code,
            product_name=product_name
        )
    except ImportError:
        upsell_task = asyncio.sleep(0)  # No-op if not available

    results = await asyncio.gather(
        add_to_convertkit(sale_data['email'], first_name, last_name, product_code),
        enroll_in_nurture_sequence(
            email=sale_data['email'],
            first_name=first_name,
            product_name=product_name,
            product_type=product_type,
            product_code=product_code,
            download_url=download_url
        ),
        upsell_task,
        return_exceptions=True
    )

    convertkit_ok = results[0] is True
    # Nurture + upsell currently do not map cleanly to a gumroad_sales boolean flag.
    try:
        await _update_sale_flags(sale_data.get("sale_id", ""), convertkit_synced=convertkit_ok)
    except Exception as exc:
        logger.warning("Failed updating gumroad_sale flags for sale_id=%s: %s", sale_data.get("sale_id"), exc)

    logger.info(
        "Sale processing results - ConvertKit: %s, NurtureSequence: %s, Upsell: %s",
        results[0],
        results[1],
        results[2],
    )

@router.get("/analytics", dependencies=[Depends(_require_internal_api_key)])
async def get_sales_analytics():
    """Get sales analytics from database"""
    try:
        from database.async_connection import get_pool

        pool = get_pool()

        # Get total sales
        total_sales = await pool.fetchval(
            "SELECT COUNT(*) FROM gumroad_sales WHERE created_at > NOW() - INTERVAL '30 days'"
        )

        # Get total revenue
        total_revenue = await pool.fetchval(
            "SELECT COALESCE(SUM(price), 0) FROM gumroad_sales WHERE created_at > NOW() - INTERVAL '30 days'"
        )

        # Get product breakdown
        product_stats = await pool.fetch("""
            SELECT
                product_code,
                COUNT(*) as units_sold,
                SUM(price) as revenue
            FROM gumroad_sales
            WHERE created_at > NOW() - INTERVAL '30 days'
            GROUP BY product_code
            ORDER BY revenue DESC
        """)

        # Get recent sales
        recent_sales = await pool.fetch("""
            SELECT
                sale_id, email, product_name, price, created_at
            FROM gumroad_sales
            ORDER BY created_at DESC
            LIMIT 10
        """)

        return {
            "total_sales": total_sales or 0,
            "total_revenue": float(total_revenue) if total_revenue else 0,
            "product_stats": [dict(row) for row in product_stats],
            "recent_sales": [dict(row) for row in recent_sales]
        }

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/test", dependencies=[Depends(_require_internal_api_key)])
async def test_webhook(background_tasks: BackgroundTasks):
    """Test endpoint for webhook"""
    if ENVIRONMENT == "production":
        raise HTTPException(status_code=404, detail="Not found")

    test_sale = {
        "email": "test@example.com",
        "full_name": "Test User",
        "product_name": "Ultimate All-Access Bundle (GR-ULTIMATE)",
        "product_permalink": "gr-ultimate",
        "sale_id": f"TEST-{datetime.utcnow().timestamp()}",
        "sale_timestamp": datetime.utcnow().isoformat(),
        "price": "997.00",
        "currency": "USD",
        "download_url": "https://gumroad.com/library"
    }

    background_tasks.add_task(
        process_sale,
        test_sale,
        "GR-ULTIMATE",
        "Test",
        "User"
    )

    return {
        "success": True,
        "message": "Test sale initiated",
        "test_data": test_sale
    }

@router.get("/products", dependencies=[Depends(_require_internal_api_key)])
async def get_products():
    """Get list of configured products"""
    return {
        "products": PRODUCT_MAPPING,
        "total": len(PRODUCT_MAPPING)
    }

"""
Intelligent Upsell Engine
=========================
Automatically suggests related products after purchase,
tracks customer history, and sends personalized upsell emails.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Base upsell timing: 3 days after purchase (in minutes).
BASE_UPSELL_DELAY_MINUTES = 60 * 24 * 3

# Product relationships for upselling
PRODUCT_UPSELLS = {
    "HJHMSM": {  # MCP Server Starter Kit
        "related": ["GSAAVB", "GR-AI-ORCH"],
        "upsell_subject": "Level up your MCP setup with AI Orchestration",
        "upsell_message": "Since you're building with MCP servers, you'll love the AI Orchestration Framework - it turns your MCP setup into a fully autonomous system."
    },
    "VJXCEW": {  # SaaS Automation Scripts
        "related": ["GR-ERP-START", "GR-LAUNCH"],
        "upsell_subject": "Take your automation to the next level",
        "upsell_message": "Your automation scripts pair perfectly with the SaaS ERP Starter Kit - get a complete business automation stack."
    },
    "XGFKP": {  # AI Prompt Engineering Pack
        "related": ["GR-CONTENT", "GSAAVB"],
        "upsell_subject": "Turn your prompts into a content machine",
        "upsell_message": "With your prompt engineering skills, you're ready for the AI Content Production Pipeline - automate your entire content workflow."
    },
    "GSAAVB": {  # AI Orchestration Framework
        "related": ["GR-ULTIMATE", "HJHMSM"],
        "upsell_subject": "Unlock the full BrainOps ecosystem",
        "upsell_message": "You've mastered orchestration! The Ultimate Bundle gives you every tool in our arsenal at a massive discount."
    },
    "UPSYKR": {  # Command Center UI Kit
        "related": ["GR-ERP-START", "GSAAVB"],
        "upsell_subject": "Build the backend for your Command Center",
        "upsell_message": "Your UI is ready - now power it with the SaaS ERP Starter Kit for a complete full-stack solution."
    },
    "CAWVO": {  # Business Automation Toolkit
        "related": ["GR-LAUNCH", "GR-ONBOARD"],
        "upsell_subject": "Automate your entire client journey",
        "upsell_message": "Love the automation? The Client Onboarding System takes it further with intelligent client management."
    },
    "GR-ROOFINT": {  # Commercial Roofing Intelligence Bundle
        "related": ["GR-ROOFVAL", "GR-PMACC"],
        "upsell_subject": "Complete your roofing tech stack",
        "upsell_message": "Add the Estimation Validator to your Intelligence Bundle for bulletproof pricing confidence."
    },
    "GR-PMACC": {  # AI Project Management Accelerator
        "related": ["GR-ONBOARD", "GR-CONTENT"],
        "upsell_subject": "Supercharge your project delivery",
        "upsell_message": "Great projects need great onboarding - pair your PM tools with our Client Onboarding System."
    },
    "GR-LAUNCH": {  # Digital Product Launch Optimizer
        "related": ["GR-CONTENT", "GR-ULTIMATE"],
        "upsell_subject": "Scale your launches with content automation",
        "upsell_message": "Ready to launch faster? The Content Production Pipeline creates all your launch assets automatically."
    },
    "GR-ONBOARD": {  # Intelligent Client Onboarding System
        "related": ["GR-PMACC", "GR-ERP-START"],
        "upsell_subject": "From onboarding to ongoing management",
        "upsell_message": "Smooth onboarding is just the start - the SaaS ERP Starter Kit handles everything that comes next."
    },
    "GR-CONTENT": {  # AI Content Production Pipeline
        "related": ["GR-LAUNCH", "GR-ULTIMATE"],
        "upsell_subject": "Launch your content at scale",
        "upsell_message": "Content + Launch = unstoppable. Add the Digital Product Launch Optimizer to your toolkit."
    },
    "GR-ROOFVAL": {  # Commercial Roofing Estimation Validator
        "related": ["GR-ROOFINT", "GR-PMACC"],
        "upsell_subject": "Add intelligence to your estimates",
        "upsell_message": "Pair your validator with the Roofing Intelligence Bundle for data-driven decision making."
    },
    "GR-ERP-START": {  # SaaS ERP Starter Kit
        "related": ["GR-ULTIMATE", "UPSYKR"],
        "upsell_subject": "Complete your SaaS platform",
        "upsell_message": "Your ERP needs a stunning frontend - the Command Center UI Kit is the perfect match."
    },
    "GR-AI-ORCH": {  # BrainOps AI Orchestrator Framework
        "related": ["GR-ULTIMATE", "HJHMSM"],
        "upsell_subject": "Expand your AI capabilities",
        "upsell_message": "Take orchestration further with the MCP Server Starter Kit for true multi-model AI."
    },
    "GR-UI-KIT": {  # Modern Command Center UI Kit
        "related": ["GR-ERP-START", "GSAAVB"],
        "upsell_subject": "Add backend power to your UI",
        "upsell_message": "Beautiful UI deserves smart backend - pair with the SaaS ERP Starter Kit."
    },
    "GR-ULTIMATE": {  # Ultimate All-Access Bundle
        "related": [],  # Already has everything
        "upsell_subject": None,
        "upsell_message": None
    }
}

# Product prices and names for upselling
PRODUCT_INFO = {
    "HJHMSM": {"name": "MCP Server Starter Kit", "price": 97},
    "VJXCEW": {"name": "SaaS Automation Scripts", "price": 67},
    "XGFKP": {"name": "AI Prompt Engineering Pack", "price": 47},
    "GSAAVB": {"name": "AI Orchestration Framework", "price": 147},
    "UPSYKR": {"name": "Command Center UI Kit", "price": 149},
    "CAWVO": {"name": "Business Automation Toolkit", "price": 49},
    "GR-ROOFINT": {"name": "Commercial Roofing Intelligence Bundle", "price": 97},
    "GR-PMACC": {"name": "AI Project Management Accelerator", "price": 127},
    "GR-LAUNCH": {"name": "Digital Product Launch Optimizer", "price": 147},
    "GR-ONBOARD": {"name": "Intelligent Client Onboarding System", "price": 297},
    "GR-CONTENT": {"name": "AI Content Production Pipeline", "price": 347},
    "GR-ROOFVAL": {"name": "Commercial Roofing Estimation Validator", "price": 497},
    "GR-ERP-START": {"name": "SaaS ERP Starter Kit", "price": 197},
    "GR-AI-ORCH": {"name": "BrainOps AI Orchestrator Framework", "price": 147},
    "GR-UI-KIT": {"name": "Modern Command Center UI Kit", "price": 97},
    "GR-ULTIMATE": {"name": "Ultimate All-Access Bundle", "price": 997}
}

# Legacy mapping for backwards compatibility
PRODUCT_PRICES = {k: v["price"] for k, v in PRODUCT_INFO.items()}


def _normalize_product_code(product_code: str) -> str:
    return (product_code or "").strip().upper()


def _product_name(product_code: str) -> str:
    code = _normalize_product_code(product_code)
    name = str(PRODUCT_INFO.get(code, {}).get("name") or "").strip()
    return name or code or "Product"


def _product_price(product_code: str) -> int:
    code = _normalize_product_code(product_code)
    try:
        return int(PRODUCT_INFO.get(code, {}).get("price") or 0)
    except Exception:
        return 0


def _gumroad_product_url(product_code: str) -> str:
    code = (product_code or "").strip()
    return f"https://brainstack.gumroad.com/l/{code}" if code else "https://gumroad.com/library"


async def get_customer_purchase_history(email: str) -> list[dict]:
    """Get all products a customer has purchased."""
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        rows = await pool.fetch("""
            SELECT product_code, product_name, price, sale_timestamp
            FROM gumroad_sales
            WHERE email = $1
            ORDER BY sale_timestamp DESC NULLS LAST, created_at DESC
        """, email)

        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error getting purchase history: {e}")
        return []


async def get_recommended_upsells(email: str, last_product_code: str) -> list[dict]:
    """Get personalized upsell recommendations based on purchase history."""
    try:
        # Get what they've already bought
        history = await get_customer_purchase_history(email)
        purchased_codes = {
            _normalize_product_code(p.get("product_code", ""))
            for p in history
            if p.get("product_code")
        }
        trigger_code = _normalize_product_code(last_product_code)
        if trigger_code:
            purchased_codes.add(trigger_code)

        # Get upsell config for last product
        upsell_config = PRODUCT_UPSELLS.get(trigger_code, {})
        related_products = list(upsell_config.get("related", []) or [])

        # Fallback: if there is no explicit mapping, default to the Ultimate bundle (if not owned).
        if not related_products and "GR-ULTIMATE" not in purchased_codes:
            related_products = ["GR-ULTIMATE"]

        # Filter out already purchased
        recommendations = []
        prior_purchase_count = max(len(purchased_codes) - 1, 0)
        for product_code in related_products:
            code = _normalize_product_code(product_code)
            if not code or code in purchased_codes:
                continue

            price = _product_price(code)
            discount_pct = min(prior_purchase_count * 5, 20)  # 5% per prior purchase, max 20%
            discounted_price = int(price * (100 - discount_pct) / 100) if price else 0

            recommendations.append({
                "product_code": code,
                "product_name": _product_name(code),
                "product_url": _gumroad_product_url(code),
                "original_price": price,
                "discount_percent": discount_pct,
                "discounted_price": discounted_price,
                "reason": f"Pairs well with {_product_name(trigger_code)}",
            })

        return recommendations[:2]  # Max 2 recommendations
    except Exception as e:
        logger.error(f"Error getting upsell recommendations: {e}")
        return []


async def calculate_optimal_upsell_timing(email: str) -> int:
    """
    Calculate optimal delay (minutes) for the upsell email based on engagement.

    Constraints:
    - Never schedule sooner than 3 days after purchase (BASE_UPSELL_DELAY_MINUTES).

    Engagement signals (best-effort, based on what we have in DB today):
    - Repeat purchases (gumroad_sales) => earlier on day 3
    - High recent email volume (ai_email_queue) => later on day 3
    """
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        purchase_stats = await pool.fetchrow("""
            SELECT
                COUNT(*)::int AS purchase_count,
                MAX(sale_timestamp) AS last_purchase_at,
                COALESCE(SUM(price), 0) AS lifetime_value
            FROM gumroad_sales
            WHERE email = $1
        """, email)

        email_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'sent' AND sent_at > NOW() - INTERVAL '60 days')::int AS emails_sent_60d,
                MAX(sent_at) AS last_email_sent_at
            FROM ai_email_queue
            WHERE recipient = $1
        """, email)

        purchase_count = int(purchase_stats.get("purchase_count") or 0) if purchase_stats else 0
        emails_sent_60d = int(email_stats.get("emails_sent_60d") or 0) if email_stats else 0

        # Repeat buyers tend to convert better: send closer to the 3-day mark.
        if purchase_count >= 3:
            return BASE_UPSELL_DELAY_MINUTES + 60  # +1h
        if purchase_count == 2:
            return BASE_UPSELL_DELAY_MINUTES + 240  # +4h

        # Avoid piling on if the customer is already receiving lots of emails.
        if emails_sent_60d >= 8:
            return BASE_UPSELL_DELAY_MINUTES + 720  # +12h

        return BASE_UPSELL_DELAY_MINUTES + 480  # +8h default
    except Exception as e:
        logger.error(f"Error calculating timing: {e}")
        return BASE_UPSELL_DELAY_MINUTES + 480


async def _avoid_email_collision(recipient: str, scheduled_for: datetime) -> datetime:
    """
    Best-effort collision avoidance to prevent sending multiple emails at the same time.

    If there is an existing queued/scheduled email within +/- 45 minutes, push by 2 hours.
    """
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        candidate = scheduled_for
        for _ in range(3):
            window_start = candidate - timedelta(minutes=45)
            window_end = candidate + timedelta(minutes=45)
            collision_count = await pool.fetchval("""
                SELECT COUNT(*)::int
                FROM ai_email_queue
                WHERE recipient = $1
                  AND status IN ('queued', 'scheduled', 'processing')
                  AND scheduled_for IS NOT NULL
                  AND scheduled_for BETWEEN $2 AND $3
            """, recipient, window_start, window_end)
            if not collision_count:
                return candidate
            candidate = candidate + timedelta(hours=2)

        return candidate
    except Exception:
        return scheduled_for


async def schedule_upsell_email(
    email: str,
    first_name: str,
    last_product_code: str,
    last_product_name: str
) -> Optional[str]:
    """Schedule a personalized upsell email."""
    try:
        trigger_code = _normalize_product_code(last_product_code)

        # Get upsell config
        upsell_config = PRODUCT_UPSELLS.get(trigger_code)
        if not upsell_config or not upsell_config.get('related'):
            logger.info(f"No upsells configured for {last_product_code}")
            return None

        # Get recommendations
        recommendations = await get_recommended_upsells(email, trigger_code)
        if not recommendations:
            logger.info(f"No upsell recommendations for {email} (may already own related)")
            return None

        # Calculate optimal timing
        delay_minutes = await calculate_optimal_upsell_timing(email)

        # Build email content
        subject = upsell_config.get('upsell_subject', f"A special offer for you, {first_name}")

        blocks: list[str] = []
        recommended_codes: list[str] = []
        max_discount_percent = 0

        for rec in recommendations:
            code = str(rec.get("product_code") or "").strip()
            recommended_codes.append(code)

            product_name = str(rec.get("product_name") or _product_name(code))
            product_url = str(rec.get("product_url") or _gumroad_product_url(code))
            reason = str(rec.get("reason") or "").strip()

            original_price = int(rec.get("original_price") or 0)
            discounted_price = int(rec.get("discounted_price") or 0)
            discount_percent = int(rec.get("discount_percent") or 0)
            max_discount_percent = max(max_discount_percent, discount_percent)

            price_html = ""
            if original_price and discounted_price and discounted_price < original_price:
                price_html = (
                    f"<span style=\"text-decoration: line-through; color: #999;\">${original_price}</span> "
                    f"<strong>${discounted_price}</strong>"
                )
            elif original_price:
                price_html = f"<strong>${original_price}</strong>"

            discount_text = f"{discount_percent}% off for existing customers" if discount_percent else ""

            blocks.append(f"""
    <div style="background: #f5f5f5; padding: 18px; border-radius: 8px; margin: 14px 0;">
        <h3 style="margin: 0 0 8px 0;">{product_name}</h3>
        {f'<p style=\"margin: 0 0 10px 0; color: #555;\">{reason}</p>' if reason else ''}
        {f'<p style=\"margin: 0 0 12px 0; font-size: 20px; color: #2563eb;\">{price_html}</p>' if price_html else ''}
        {f'<p style=\"margin: 0 0 12px 0; color: #16a34a;\">{discount_text}</p>' if discount_text else ''}
        <a href="{product_url}"
           style="background: #2563eb; color: white; padding: 12px 18px;
                  text-decoration: none; border-radius: 6px; display: inline-block;">
            View {product_name}
        </a>
    </div>
""")

        body = f"""
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Hope you're enjoying <strong>{last_product_name}</strong>!</p>

    <p>{upsell_config.get('upsell_message', '')}</p>

    <h3 style="margin-top: 22px;">Recommended next steps</h3>
    {''.join(blocks)}

    <p>This offer is just for existing customers like you. Thanks for being part of the BrainStack community!</p>

    <p>Best,<br>Matt @ BrainStack</p>
</div>
"""

        # Schedule the email
        from email_scheduler_daemon import schedule_nurture_email
        desired_send_at = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)
        desired_send_at = await _avoid_email_collision(email, desired_send_at)
        delay_minutes = max(0, int((desired_send_at - datetime.now(timezone.utc)).total_seconds() / 60))

        email_id = await schedule_nurture_email(
            recipient=email,
            subject=subject,
            body=body,
            delay_minutes=delay_minutes,
            metadata={
                "source": "upsell_engine",
                "trigger_product": trigger_code,
                "trigger_product_name": last_product_name,
                "recommended_products": [c for c in recommended_codes if c],
                "recommended_product": next((c for c in recommended_codes if c), ""),
                "discount_percent": max_discount_percent,
            }
        )

        logger.info(
            "Scheduled upsell email for %s: %s in %s minutes",
            email,
            ",".join([c for c in recommended_codes if c]),
            delay_minutes,
        )
        return email_id

    except Exception as e:
        logger.error(f"Error scheduling upsell email: {e}")
        return None


async def process_purchase_for_upsell(
    email: str,
    first_name: str,
    product_code: str,
    product_name: str
) -> dict:
    """
    Main entry point: Process a new purchase and schedule upsell.
    Call this from gumroad_webhook after a successful sale.
    """
    result = {
        "processed": True,
        "email": email,
        "product_code": product_code,
        "upsell_scheduled": False,
        "upsell_email_id": None,
        "recommendations": []
    }

    try:
        trigger_code = _normalize_product_code(product_code)

        # Best-effort duplicate protection: avoid scheduling multiple upsells for the same trigger product.
        try:
            from database.async_connection import get_pool
            pool = get_pool()
            existing = await pool.fetchval("""
                SELECT COUNT(*)::int
                FROM ai_email_queue
                WHERE recipient = $1
                  AND metadata->>'source' = 'upsell_engine'
                  AND metadata->>'trigger_product' = $2
                  AND created_at > NOW() - INTERVAL '14 days'
            """, email, trigger_code)
            if existing and int(existing) > 0:
                result["skipped_reason"] = "duplicate_recent_upsell"
                return result
        except Exception:
            # If the DB check fails, proceed (queue-level locks still reduce duplicate sends in practice).
            pass

        # Get recommendations
        recommendations = await get_recommended_upsells(email, trigger_code)
        result["recommendations"] = recommendations

        if recommendations:
            # Schedule upsell email
            email_id = await schedule_upsell_email(
                email=email,
                first_name=first_name or "there",
                last_product_code=trigger_code,
                last_product_name=product_name
            )

            if email_id:
                result["upsell_scheduled"] = True
                result["upsell_email_id"] = email_id

        return result

    except Exception as e:
        logger.error(f"Error processing purchase for upsell: {e}")
        result["error"] = str(e)
        return result


# Scheduled job: Process abandoned upsell opportunities
async def process_missed_upsells(days_back: int = 7, limit: int = 50) -> dict:
    """
    Find recent purchases that didn't get upsell emails and schedule them.
    Run this as a scheduled job.
    """
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        # Find purchases without upsell emails
        purchases = await pool.fetch("""
            SELECT DISTINCT ON (gs.email)
                gs.email, gs.product_code, gs.product_name, gs.sale_timestamp,
                COALESCE(gs.customer_name, 'there') as first_name
            FROM gumroad_sales gs
            LEFT JOIN ai_email_queue eq ON eq.recipient = gs.email
                AND eq.metadata->>'source' = 'upsell_engine'
                AND eq.metadata->>'trigger_product' = gs.product_code
            WHERE gs.sale_timestamp > NOW() - make_interval(days => $1)
                AND eq.id IS NULL
                AND gs.email NOT LIKE '%test%'
                AND gs.email NOT LIKE '%example%'
            ORDER BY gs.email, gs.sale_timestamp DESC
            LIMIT $2
        """, days_back, limit)

        scheduled = 0
        for purchase in purchases:
            result = await process_purchase_for_upsell(
                email=purchase['email'],
                first_name=purchase['first_name'],
                product_code=purchase['product_code'],
                product_name=purchase['product_name']
            )
            if result.get('upsell_scheduled'):
                scheduled += 1

        return {
            "success": True,
            "purchases_found": len(purchases),
            "upsells_scheduled": scheduled,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Error processing missed upsells: {e}")
        return {"success": False, "error": str(e)}

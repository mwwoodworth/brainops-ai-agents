"""
Intelligent Upsell Engine
=========================
Automatically suggests related products after purchase,
tracks customer history, and sends personalized upsell emails.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

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


def _get_db_config():
    """Get database configuration from environment variables."""
    database_url = os.getenv('DATABASE_URL')
    if database_url:
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
    return None


async def get_customer_purchase_history(email: str) -> list[dict]:
    """Get all products a customer has purchased."""
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        rows = await pool.fetch("""
            SELECT product_code, product_name, price, sale_timestamp
            FROM gumroad_sales
            WHERE email = $1
            ORDER BY sale_timestamp DESC
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
        purchased_codes = {p['product_code'] for p in history}

        # Get upsell config for last product
        upsell_config = PRODUCT_UPSELLS.get(last_product_code, {})
        related_products = upsell_config.get('related', [])

        # Filter out already purchased
        recommendations = []
        for product_code in related_products:
            if product_code not in purchased_codes:
                price = PRODUCT_PRICES.get(product_code, 0)
                # Calculate loyalty discount based on purchase count
                discount_pct = min(len(purchased_codes) * 5, 20)  # 5% per purchase, max 20%
                discounted_price = int(price * (100 - discount_pct) / 100)

                recommendations.append({
                    'product_code': product_code,
                    'original_price': price,
                    'discount_percent': discount_pct,
                    'discounted_price': discounted_price,
                    'reason': f"Recommended based on your {last_product_code} purchase"
                })

        return recommendations[:2]  # Max 2 recommendations
    except Exception as e:
        logger.error(f"Error getting upsell recommendations: {e}")
        return []


async def calculate_optimal_upsell_timing(email: str) -> int:
    """Calculate optimal delay for upsell email based on engagement."""
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        # Check email engagement
        engagement = await pool.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'sent') as emails_received,
                COUNT(*) FILTER (WHERE metadata->>'clicked' = 'true') as emails_clicked
            FROM ai_email_queue
            WHERE recipient = $1
        """, email)

        if engagement:
            received = engagement['emails_received'] or 0
            clicked = engagement['emails_clicked'] or 0

            if received > 0 and clicked / received > 0.3:
                # High engagement - send sooner (2 days)
                return 2880  # minutes
            elif received > 5:
                # Low engagement - wait longer (5 days)
                return 7200

        # Default: 3 days
        return 4320
    except Exception as e:
        logger.error(f"Error calculating timing: {e}")
        return 4320  # Default 3 days


async def schedule_upsell_email(
    email: str,
    first_name: str,
    last_product_code: str,
    last_product_name: str
) -> Optional[str]:
    """Schedule a personalized upsell email."""
    try:
        # Get upsell config
        upsell_config = PRODUCT_UPSELLS.get(last_product_code)
        if not upsell_config or not upsell_config.get('related'):
            logger.info(f"No upsells configured for {last_product_code}")
            return None

        # Get recommendations
        recommendations = await get_recommended_upsells(email, last_product_code)
        if not recommendations:
            logger.info(f"No upsell recommendations for {email} (may already own related)")
            return None

        # Calculate optimal timing
        delay_minutes = await calculate_optimal_upsell_timing(email)

        # Build email content
        subject = upsell_config.get('upsell_subject', f"A special offer for you, {first_name}")

        rec = recommendations[0]
        rec_product_name = PRODUCT_INFO.get(rec['product_code'], {}).get('name', rec['product_code'])
        discount_text = f"As a valued customer, you get {rec['discount_percent']}% off!" if rec['discount_percent'] > 0 else ""

        body = f"""
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Hope you're enjoying <strong>{last_product_name}</strong>!</p>

    <p>{upsell_config.get('upsell_message', '')}</p>

    <div style="background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3 style="margin-top: 0;">Special Offer for You</h3>
        <p><strong>{rec_product_name}</strong></p>
        <p style="font-size: 24px; color: #2563eb;">
            <span style="text-decoration: line-through; color: #999;">${rec['original_price']}</span>
            <strong>${rec['discounted_price']}</strong>
        </p>
        {f'<p style="color: #16a34a;">{discount_text}</p>' if discount_text else ''}
        <a href="https://brainstack.gumroad.com/l/{rec['product_code']}"
           style="background: #2563eb; color: white; padding: 12px 24px;
                  text-decoration: none; border-radius: 5px; display: inline-block;">
            Get It Now
        </a>
    </div>

    <p>This offer is just for existing customers like you. Thanks for being part of the BrainStack community!</p>

    <p>Best,<br>Matt @ BrainStack</p>
</div>
"""

        # Schedule the email
        from email_scheduler_daemon import schedule_nurture_email
        email_id = await schedule_nurture_email(
            recipient=email,
            subject=subject,
            body=body,
            delay_minutes=delay_minutes,
            metadata={
                "source": "upsell_engine",
                "trigger_product": last_product_code,
                "recommended_product": rec['product_code'],
                "discount_percent": rec['discount_percent']
            }
        )

        logger.info(f"Scheduled upsell email for {email}: {rec['product_code']} in {delay_minutes} minutes")
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
        # Get recommendations
        recommendations = await get_recommended_upsells(email, product_code)
        result["recommendations"] = recommendations

        if recommendations:
            # Schedule upsell email
            email_id = await schedule_upsell_email(
                email=email,
                first_name=first_name or "there",
                last_product_code=product_code,
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

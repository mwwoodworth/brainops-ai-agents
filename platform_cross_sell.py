"""
Platform Cross-Sell Sequences
==============================
Cross-sell sequences for users from different platforms:
- MRG (MyRoofGenius) users -> Roofing Validator product
- BSS (BrainStack Studio) users -> Developer products

This module identifies platform users and enrolls them in targeted cross-sell sequences.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Cross-sell delay: 7 days after signup/last activity
CROSS_SELL_DELAY_MINUTES = 60 * 24 * 7

GUMROAD_SELLER_HOST = os.getenv("GUMROAD_SELLER_HOST", "woodworthia.gumroad.com").strip() or "woodworthia.gumroad.com"

# MRG User -> Roofing Validator cross-sell sequence
MRG_ROOFING_VALIDATOR_SEQUENCE = {
    "name": "MRG to Roofing Validator",
    "product_code": "GR-ROOFVAL",
    "product_name": "Commercial Roofing Estimation Validator",
    "product_url": f"https://{GUMROAD_SELLER_HOST}/l/gr-roofval",
    "price": 497,
    "emails": [
        {
            "delay_minutes": CROSS_SELL_DELAY_MINUTES,  # Day 7
            "subject": "Stop leaving money on the table with your estimates",
            "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>I noticed you're using MyRoofGenius for your roofing business. Nice!</p>

    <p>Quick question: <strong>How confident are you in your commercial estimates?</strong></p>

    <p>I built something that might help. The <strong>Commercial Roofing Estimation Validator</strong> uses AI trained on 18,000+ real commercial roofing estimates to catch pricing errors before you send the bid.</p>

    <p>It validates:</p>
    <ul>
        <li>Labor rates against regional benchmarks</li>
        <li>Material costs for TPO, EPDM, metal, and built-up</li>
        <li>50+ industry-standard validation rules</li>
        <li>Common underpricing mistakes</li>
    </ul>

    <p>One contractor told me it caught a $12,000 underpricing error on his first estimate.</p>

    <p><a href="{product_url}" style="background: #e94560; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: bold;">See How It Works - $497</a></p>

    <p>If you're doing commercial work, this pays for itself on the first job.</p>

    <p>Best,<br>Matt @ BrainStack</p>
</div>
"""
        },
        {
            "delay_minutes": CROSS_SELL_DELAY_MINUTES + 4320,  # Day 10
            "subject": "Real numbers from contractors using the Validator",
            "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Following up on the Roofing Estimation Validator.</p>

    <p>Here's what contractors are telling me:</p>

    <ul>
        <li>"Caught a labor rate error that would have cost me $8,200"</li>
        <li>"Finally confident my TPO estimates are competitive but profitable"</li>
        <li>"Wish I had this 5 years ago"</li>
    </ul>

    <p>The tool runs your estimate through 50+ validation rules built from analyzing 18,000+ real commercial bids.</p>

    <p>If you're doing any commercial work, this is a no-brainer at $497.</p>

    <p><a href="{product_url}" style="background: #e94560; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: bold;">Get the Validator</a></p>

    <p>Best,<br>Matt</p>
</div>
"""
        }
    ]
}

# BSS User -> Developer Products cross-sell sequence
BSS_DEV_PRODUCTS_SEQUENCE = {
    "name": "BSS to Dev Products",
    "emails": [
        {
            "delay_minutes": CROSS_SELL_DELAY_MINUTES,  # Day 7
            "subject": "Turn your AI experiments into production systems",
            "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>I see you've been exploring BrainStack Studio. Love that you're building with AI!</p>

    <p>Quick thought: What if you could skip months of infrastructure work and go straight to building?</p>

    <p>I've packaged up the exact systems I use to build and ship AI products:</p>

    <div style="background: #f5f5f5; padding: 18px; border-radius: 8px; margin: 14px 0;">
        <h3 style="margin: 0 0 8px 0; color: #a855f7;">AI Content Production Pipeline - $347</h3>
        <p style="margin: 0 0 10px 0; color: #555;">Scale your content 10x with multi-stage AI pipeline, SEO optimization, and publishing integrations.</p>
        <a href="https://woodworthia.gumroad.com/l/gr-content" style="color: #a855f7; font-weight: bold;">Learn more</a>
    </div>

    <div style="background: #f5f5f5; padding: 18px; border-radius: 8px; margin: 14px 0;">
        <h3 style="margin: 0 0 8px 0; color: #0ea5e9;">Intelligent Client Onboarding - $297</h3>
        <p style="margin: 0 0 10px 0; color: #555;">Automate your entire client intake with smart forms, e-signatures, and CRM integrations.</p>
        <a href="https://woodworthia.gumroad.com/l/gr-onboard" style="color: #0ea5e9; font-weight: bold;">Learn more</a>
    </div>

    <div style="background: #f5f5f5; padding: 18px; border-radius: 8px; margin: 14px 0;">
        <h3 style="margin: 0 0 8px 0; color: #22c55e;">SaaS ERP Starter Kit - $197</h3>
        <p style="margin: 0 0 10px 0; color: #555;">Production-ready multi-tenant foundation with auth, CRM, invoicing, and AI features.</p>
        <a href="https://woodworthia.gumroad.com/l/gr-erp-start" style="color: #22c55e; font-weight: bold;">Learn more</a>
    </div>

    <p>All include full source code, documentation, and lifetime updates.</p>

    <p>Best,<br>Matt @ BrainStack</p>
</div>
"""
        },
        {
            "delay_minutes": CROSS_SELL_DELAY_MINUTES + 4320,  # Day 10
            "subject": "Which product fits your build?",
            "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Still figuring out which developer product would help most? Here's my quick guide:</p>

    <p><strong>Building a SaaS?</strong><br>
    Start with the <a href="https://woodworthia.gumroad.com/l/gr-erp-start" style="color: #22c55e;">SaaS ERP Starter Kit ($197)</a> - it handles auth, multi-tenancy, billing, and gives you a solid foundation.</p>

    <p><strong>Need to scale content?</strong><br>
    The <a href="https://woodworthia.gumroad.com/l/gr-content" style="color: #a855f7;">AI Content Pipeline ($347)</a> automates research, writing, SEO, and publishing. I use it for everything.</p>

    <p><strong>Client-facing service business?</strong><br>
    <a href="https://woodworthia.gumroad.com/l/gr-onboard" style="color: #0ea5e9;">Intelligent Client Onboarding ($297)</a> handles intake, contracts, and kickoff automatically.</p>

    <p><strong>Managing complex projects?</strong><br>
    The <a href="https://woodworthia.gumroad.com/l/gr-pmcmd" style="color: #f59e0b;">AI Project Command Center ($197)</a> is a complete Notion workspace with 8 linked databases and 30+ automations.</p>

    <p>Questions? Just reply - I read every email.</p>

    <p>Best,<br>Matt</p>
</div>
"""
        }
    ]
}


async def enroll_mrg_user_in_cross_sell(
    email: str,
    first_name: str,
    user_id: Optional[str] = None
) -> list[str]:
    """
    Enroll an MRG user in the Roofing Validator cross-sell sequence.
    Returns list of scheduled email IDs.
    """
    try:
        from email_scheduler_daemon import schedule_nurture_email

        # Check if user already purchased the product
        from database.async_connection import get_pool
        pool = get_pool()

        existing = await pool.fetchval("""
            SELECT COUNT(*)::int
            FROM gumroad_sales
            WHERE email = $1
              AND UPPER(product_code) = 'GR-ROOFVAL'
        """, email)

        if existing and int(existing) > 0:
            logger.info(f"Skipping MRG cross-sell for {email} - already owns Roofing Validator")
            return []

        # Check for existing cross-sell emails
        existing_cross_sell = await pool.fetchval("""
            SELECT COUNT(*)::int
            FROM ai_email_queue
            WHERE recipient = $1
              AND metadata->>'source' = 'mrg_cross_sell'
              AND created_at > NOW() - INTERVAL '30 days'
        """, email)

        if existing_cross_sell and int(existing_cross_sell) > 0:
            logger.info(f"Skipping MRG cross-sell for {email} - already enrolled recently")
            return []

        sequence = MRG_ROOFING_VALIDATOR_SEQUENCE
        email_ids = []

        for email_template in sequence["emails"]:
            subject = email_template["subject"]
            body = email_template["body"].format(
                first_name=first_name or "there",
                product_url=sequence["product_url"]
            )

            email_id = await schedule_nurture_email(
                recipient=email,
                subject=subject,
                body=body,
                delay_minutes=email_template["delay_minutes"],
                metadata={
                    "source": "mrg_cross_sell",
                    "sequence_name": sequence["name"],
                    "product_code": sequence["product_code"],
                    "user_id": user_id,
                    "delay_minutes": email_template["delay_minutes"]
                }
            )

            if email_id:
                email_ids.append(email_id)

        logger.info(f"Enrolled MRG user {email} in Roofing Validator cross-sell: {len(email_ids)} emails")
        return email_ids

    except Exception as e:
        logger.error(f"Failed to enroll MRG user in cross-sell: {e!r}")
        return []


async def enroll_bss_user_in_cross_sell(
    email: str,
    first_name: str,
    user_id: Optional[str] = None
) -> list[str]:
    """
    Enroll a BSS user in the developer products cross-sell sequence.
    Returns list of scheduled email IDs.
    """
    try:
        from email_scheduler_daemon import schedule_nurture_email
        from database.async_connection import get_pool

        pool = get_pool()

        # Check for existing cross-sell emails
        existing_cross_sell = await pool.fetchval("""
            SELECT COUNT(*)::int
            FROM ai_email_queue
            WHERE recipient = $1
              AND metadata->>'source' = 'bss_cross_sell'
              AND created_at > NOW() - INTERVAL '30 days'
        """, email)

        if existing_cross_sell and int(existing_cross_sell) > 0:
            logger.info(f"Skipping BSS cross-sell for {email} - already enrolled recently")
            return []

        sequence = BSS_DEV_PRODUCTS_SEQUENCE
        email_ids = []

        for email_template in sequence["emails"]:
            subject = email_template["subject"]
            body = email_template["body"].format(
                first_name=first_name or "there"
            )

            email_id = await schedule_nurture_email(
                recipient=email,
                subject=subject,
                body=body,
                delay_minutes=email_template["delay_minutes"],
                metadata={
                    "source": "bss_cross_sell",
                    "sequence_name": sequence["name"],
                    "user_id": user_id,
                    "delay_minutes": email_template["delay_minutes"]
                }
            )

            if email_id:
                email_ids.append(email_id)

        logger.info(f"Enrolled BSS user {email} in dev products cross-sell: {len(email_ids)} emails")
        return email_ids

    except Exception as e:
        logger.error(f"Failed to enroll BSS user in cross-sell: {e!r}")
        return []


async def process_platform_cross_sells(limit: int = 100) -> dict:
    """
    Scheduled job: Find platform users who should receive cross-sell sequences.
    Run this as a daily scheduled job.
    """
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        results = {
            "mrg_enrolled": 0,
            "bss_enrolled": 0,
            "skipped": 0,
            "errors": 0
        }

        # Find MRG users who signed up 7+ days ago and haven't been cross-sold
        mrg_users = await pool.fetch("""
            SELECT DISTINCT u.email, u.name, u.id
            FROM users u
            LEFT JOIN ai_email_queue eq ON eq.recipient = u.email
                AND eq.metadata->>'source' = 'mrg_cross_sell'
            LEFT JOIN gumroad_sales gs ON gs.email = u.email
                AND UPPER(gs.product_code) = 'GR-ROOFVAL'
            WHERE u.created_at < NOW() - INTERVAL '7 days'
              AND u.marketing_consent = true
              AND eq.id IS NULL
              AND gs.id IS NULL
              AND u.email NOT LIKE '%test%'
              AND u.email NOT LIKE '%example%'
            ORDER BY u.created_at DESC
            LIMIT $1
        """, limit // 2)

        for user in mrg_users:
            try:
                first_name = (user['name'] or '').split()[0] if user.get('name') else ''
                emails = await enroll_mrg_user_in_cross_sell(
                    email=user['email'],
                    first_name=first_name,
                    user_id=str(user.get('id', ''))
                )
                if emails:
                    results["mrg_enrolled"] += 1
                else:
                    results["skipped"] += 1
            except Exception as e:
                logger.error(f"Error enrolling MRG user {user.get('email')}: {e!r}")
                results["errors"] += 1

        # BSS users would be found from a different database/table
        # For now, this is a placeholder - BSS user identification would need to be added
        # when BSS has a user database

        return {
            "success": True,
            **results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Error processing platform cross-sells: {e!r}")
        return {"success": False, "error": str(e)}


# API endpoint functions for manual triggering
async def trigger_mrg_cross_sell_for_user(email: str, first_name: str = "") -> dict:
    """Manually trigger MRG cross-sell for a specific user"""
    email_ids = await enroll_mrg_user_in_cross_sell(email, first_name)
    return {
        "success": bool(email_ids),
        "email": email,
        "emails_scheduled": len(email_ids),
        "email_ids": email_ids
    }


async def trigger_bss_cross_sell_for_user(email: str, first_name: str = "") -> dict:
    """Manually trigger BSS cross-sell for a specific user"""
    email_ids = await enroll_bss_user_in_cross_sell(email, first_name)
    return {
        "success": bool(email_ids),
        "email": email,
        "emails_scheduled": len(email_ids),
        "email_ids": email_ids
    }

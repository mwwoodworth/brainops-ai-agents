"""
Outreach Executor
=================
Bridges ai_scheduled_outreach → ai_email_queue.

Processes scheduled outreach entries by:
1. Resolving email addresses from customers/revenue_leads tables
2. Generating email content from message templates
3. Queuing emails via schedule_nurture_email()
4. Marking outreach entries as executed

Also runs outreach campaigns for new revenue_leads.

Author: BrainOps AI System
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Template content for ai_scheduled_outreach entries
OUTREACH_TEMPLATES = {
    "estimate_followup": {
        "subject": "Following up on your roofing estimate",
        "body": """Hi {customer_name},

I wanted to follow up on the estimate we prepared for you. We know choosing a roofing contractor is a big decision, and we're here to answer any questions.

If you'd like to discuss the estimate or schedule a time to review it together, just reply to this email.

Best regards,
Matt Woodworth
Weathercraft Roofing"""
    },
    "reengagement": {
        "subject": "It's been a while - how's your roof holding up?",
        "body": """Hi {customer_name},

It's been a while since we last worked together, and I wanted to check in.

Colorado weather can be tough on roofs. If you've noticed any issues or just want a free inspection, we'd be happy to take a look.

Just reply to this email or give us a call.

Best regards,
Matt Woodworth
Weathercraft Roofing"""
    },
    "product_led_growth_v1": {
        "subject": "AI-powered roofing estimates for {company}",
        "body": """Hi there,

I noticed {company} does roofing work in {city}. Quick question:

How much time does your team spend on estimates each week?

We built MyRoofGenius - an AI tool that generates accurate roof estimates in under 2 minutes from satellite imagery.

Contractors using it are cutting estimate time by 70% and winning more bids.

Plans start at $49/month with a free trial: https://myroofgenius.com/pricing

Worth a look?

Best,
Matt Woodworth
MyRoofGenius"""
    },
}


async def process_scheduled_outreach(batch_size: int = 25) -> dict[str, Any]:
    """
    Process ai_scheduled_outreach entries that are due.

    Resolves email addresses, generates content, and queues to ai_email_queue.
    """
    try:
        from database.async_connection import get_pool
        from email_scheduler_daemon import schedule_nurture_email
        from utils.outbound import email_block_reason
    except ImportError as e:
        logger.error("Required modules not available: %s", e)
        return {"error": str(e)}

    pool = get_pool()
    if not pool:
        return {"error": "Database pool not available"}

    stats = {
        "processed": 0,
        "queued": 0,
        "skipped_no_email": 0,
        "skipped_test": 0,
        "skipped_blocked": 0,
        "skipped_duplicate": 0,
        "errors": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Fetch due outreach entries
        rows = await pool.fetch("""
            SELECT id, target_id, channel, message_template, personalization,
                   scheduled_for, status, metadata
            FROM ai_scheduled_outreach
            WHERE status = 'scheduled'
              AND scheduled_for <= NOW()
            ORDER BY scheduled_for ASC
            LIMIT $1
        """, batch_size)

        if not rows:
            logger.debug("No due outreach entries found")
            return stats

        logger.info("Processing %d scheduled outreach entries", len(rows))

        for row in rows:
            outreach_id = row["id"]
            target_id = row["target_id"]
            template_name = row["message_template"]
            personalization = row["personalization"] or {}
            metadata = row["metadata"] or {}

            if isinstance(personalization, str):
                try:
                    personalization = json.loads(personalization)
                except (json.JSONDecodeError, TypeError):
                    personalization = {}

            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            stats["processed"] += 1

            try:
                # Resolve email address from revenue_leads ONLY.
                # CRITICAL: The ERP customers table is ALL demo/seed data.
                # NEVER look up emails from the customers table for outreach.
                email = None
                resolved_name = personalization.get("customer_name", "")

                # Only use revenue_leads (real prospect data)
                if target_id:
                    lead = await pool.fetchrow("""
                        SELECT company_name, contact_name, email
                        FROM revenue_leads
                        WHERE id::text = $1
                        AND (is_test = false OR is_test IS NULL)
                        AND (is_demo = false OR is_demo IS NULL)
                    """, target_id)
                    if lead and lead["email"]:
                        email = lead["email"]
                        if not resolved_name:
                            resolved_name = lead["contact_name"] or lead["company_name"] or ""

                if not email:
                    stats["skipped_no_email"] += 1
                    await pool.execute("""
                        UPDATE ai_scheduled_outreach
                        SET status = 'skipped', executed_at = NOW(),
                            metadata = COALESCE(metadata, '{}'::jsonb) || '{"skip_reason": "no_email"}'::jsonb
                        WHERE id = $1
                    """, outreach_id)
                    continue

                # Check test email patterns
                email_lower = email.lower().strip()
                test_patterns = ("@test.", "@example.", "@demo.", "@invalid.", "@localhost",
                                 ".test", ".example", "weathercraft.net", "newcustomer.com")
                if any(p in email_lower for p in test_patterns):
                    stats["skipped_test"] += 1
                    await pool.execute("""
                        UPDATE ai_scheduled_outreach
                        SET status = 'skipped', executed_at = NOW(),
                            metadata = COALESCE(metadata, '{}'::jsonb) || '{"skip_reason": "test_email"}'::jsonb
                        WHERE id = $1
                    """, outreach_id)
                    continue

                # Check outbound email block
                block = email_block_reason(email)
                if block:
                    stats["skipped_blocked"] += 1
                    await pool.execute("""
                        UPDATE ai_scheduled_outreach
                        SET status = 'skipped', executed_at = NOW(),
                            metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb
                        WHERE id = $1
                    """, outreach_id, json.dumps({"skip_reason": block}))
                    continue

                # Check for duplicate - don't send same template to same recipient
                existing = await pool.fetchval("""
                    SELECT id FROM ai_email_queue
                    WHERE recipient = $1
                    AND metadata->>'outreach_template' = $2
                    AND status IN ('queued', 'sent', 'processing')
                    LIMIT 1
                """, email, template_name)
                if existing:
                    stats["skipped_duplicate"] += 1
                    await pool.execute("""
                        UPDATE ai_scheduled_outreach
                        SET status = 'skipped', executed_at = NOW(),
                            metadata = COALESCE(metadata, '{}'::jsonb) || '{"skip_reason": "duplicate"}'::jsonb
                        WHERE id = $1
                    """, outreach_id)
                    continue

                # Generate email content from template
                template = OUTREACH_TEMPLATES.get(template_name)
                if not template:
                    logger.warning("Unknown template: %s", template_name)
                    stats["errors"] += 1
                    await pool.execute("""
                        UPDATE ai_scheduled_outreach
                        SET status = 'failed', executed_at = NOW(),
                            metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb
                        WHERE id = $1
                    """, outreach_id, json.dumps({"error": f"unknown_template:{template_name}"}))
                    continue

                # Build template variables
                template_vars = {
                    "customer_name": resolved_name or "there",
                    "company": personalization.get("company", resolved_name or "your company"),
                    "city": personalization.get("city", "your area"),
                    "product": personalization.get("product", "MyRoofGenius"),
                }

                try:
                    subject = template["subject"].format(**template_vars)
                    body = template["body"].format(**template_vars)
                except KeyError as fmt_err:
                    logger.warning("Template format error for %s: %s", template_name, fmt_err)
                    subject = template["subject"]
                    body = template["body"]

                # Queue email
                email_id = await schedule_nurture_email(
                    recipient=email,
                    subject=subject,
                    body=body,
                    delay_minutes=0,
                    metadata={
                        "source": "outreach_executor",
                        "outreach_id": str(outreach_id),
                        "outreach_template": template_name,
                        "target_id": target_id,
                    },
                )

                if email_id:
                    stats["queued"] += 1
                    await pool.execute("""
                        UPDATE ai_scheduled_outreach
                        SET status = 'executed', executed_at = NOW(),
                            response = $2,
                            metadata = COALESCE(metadata, '{}'::jsonb) || $3::jsonb
                        WHERE id = $1
                    """, outreach_id, f"queued:{email_id}", json.dumps({
                        "email_queue_id": email_id,
                        "recipient": email,
                    }))
                    logger.info("Queued outreach %s → %s (%s)", str(outreach_id)[:8], email, template_name)
                else:
                    stats["errors"] += 1

            except Exception as entry_err:
                stats["errors"] += 1
                logger.error("Error processing outreach %s: %s", outreach_id, entry_err)
                try:
                    await pool.execute("""
                        UPDATE ai_scheduled_outreach
                        SET status = 'failed', executed_at = NOW(),
                            metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb
                        WHERE id = $1
                    """, outreach_id, json.dumps({"error": str(entry_err)[:500]}))
                except Exception:
                    pass

    except Exception as e:
        logger.error("Outreach executor failed: %s", e, exc_info=True)
        stats["error"] = str(e)

    logger.info(
        "Outreach executor: processed=%d queued=%d skipped=%d errors=%d",
        stats["processed"], stats["queued"],
        stats["skipped_no_email"] + stats["skipped_test"] + stats["skipped_blocked"] + stats["skipped_duplicate"],
        stats["errors"],
    )
    return stats


async def run_outreach_cycle() -> dict[str, Any]:
    """
    Full outreach cycle:
    1. Process due ai_scheduled_outreach entries
    2. Run campaigns for new revenue_leads
    """
    results = {}

    # Step 1: Process existing ai_scheduled_outreach entries
    try:
        outreach_stats = await process_scheduled_outreach(batch_size=25)
        results["outreach_executor"] = outreach_stats
    except Exception as e:
        logger.error("Outreach executor step failed: %s", e)
        results["outreach_executor"] = {"error": str(e)}

    # Step 2: Run campaign enrollment for new leads
    try:
        from outreach_campaigns import run_campaign_for_new_leads
        campaign_stats = await run_campaign_for_new_leads(
            campaign_id="roofing_saas_intro",
            limit=10,
            exclude_contacted=True,
        )
        results["campaign_enrollment"] = campaign_stats
    except Exception as e:
        logger.error("Campaign enrollment step failed: %s", e)
        results["campaign_enrollment"] = {"error": str(e)}

    return results

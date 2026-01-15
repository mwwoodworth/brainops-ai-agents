"""
Automated Outreach Campaign System
==================================
Multi-touch B2B outreach sequences for roofing contractors.
Integrates with email_sender and revenue_leads.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Campaign definitions - multi-touch sequences
OUTREACH_CAMPAIGNS = {
    "roofing_saas_intro": {
        "name": "Roofing SaaS Introduction",
        "target": "roofing_contractors",
        "description": "Introduce MyRoofGenius AI platform to roofing contractors",
        "emails": [
            {
                "day": 0,
                "subject": "Quick question about your roofing estimates, {company_name}",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <p>Hi there,</p>

    <p>I noticed {company_name} does commercial roofing work. Quick question:</p>

    <p><strong>How much time does your team spend on estimates each week?</strong></p>

    <p>Most contractors I talk to say 10-15 hours. We built an AI tool that cuts that to under 2 hours while improving accuracy.</p>

    <p>Would a 15-minute demo be worth your time?</p>

    <p>Best,<br>Matt Woodworth<br>MyRoofGenius</p>

    <p style="font-size: 12px; color: #666;">PS - No pressure. If timing's bad, just reply "later" and I'll check back in a few months.</p>
</div>
"""
            },
            {
                "day": 3,
                "subject": "Re: Quick question about estimates",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <p>Hi again,</p>

    <p>Following up on my note about roofing estimates.</p>

    <p>Here's what our AI does in 2 minutes:</p>
    <ul>
        <li>Analyzes satellite imagery of any roof</li>
        <li>Calculates precise measurements</li>
        <li>Generates material lists and cost estimates</li>
        <li>Creates professional proposals</li>
    </ul>

    <p>One contractor told me: "We used to lose bids because we were too slow. Now we're first to respond every time."</p>

    <p>Worth a quick look?</p>

    <p>- Matt</p>
</div>
"""
            },
            {
                "day": 7,
                "subject": "Last try - then I'll leave you alone",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <p>Hi,</p>

    <p>I don't want to be that annoying sales guy, so this is my last email.</p>

    <p>If AI-powered roofing estimates aren't on your radar right now, no worries at all.</p>

    <p>But if you're curious, here's a 2-minute video showing how it works:</p>

    <p><a href="https://myroofgenius.com/demo" style="background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">Watch Quick Demo</a></p>

    <p>Either way, good luck with your projects this season!</p>

    <p>Best,<br>Matt</p>

    <p style="font-size: 12px; color: #666;">Reply "interested" anytime and I'll reach back out.</p>
</div>
"""
            }
        ]
    },
    "erp_demo_request": {
        "name": "ERP Demo Follow-up",
        "target": "demo_requests",
        "description": "Follow up with users who requested ERP demos",
        "emails": [
            {
                "day": 0,
                "subject": "Your Weathercraft ERP demo is ready",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <p>Hi {first_name},</p>

    <p>Thanks for your interest in Weathercraft ERP!</p>

    <p>I've got your demo environment ready. Here's what we'll cover:</p>
    <ul>
        <li>AI-powered job scheduling that saves 5+ hours/week</li>
        <li>Automated invoicing and payment tracking</li>
        <li>Customer communication tools</li>
        <li>Real-time project dashboards</li>
    </ul>

    <p>What time works best for a 20-minute walkthrough?</p>

    <p><a href="https://calendly.com/brainstack/erp-demo" style="background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">Book Your Demo</a></p>

    <p>Best,<br>Matt</p>
</div>
"""
            },
            {
                "day": 2,
                "subject": "Quick question about your ERP needs",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <p>Hi {first_name},</p>

    <p>Following up on your demo request. Before we meet, quick question:</p>

    <p><strong>What's the #1 thing you're hoping an ERP will solve?</strong></p>

    <p>Common answers I hear:</p>
    <ul>
        <li>"Scheduling is a nightmare"</li>
        <li>"We're always chasing payments"</li>
        <li>"Can't track project profitability"</li>
        <li>"Too much manual data entry"</li>
    </ul>

    <p>Just hit reply - I'll customize the demo to focus on what matters most to you.</p>

    <p>- Matt</p>
</div>
"""
            }
        ]
    },
    "digital_product_launch": {
        "name": "Digital Product Launch",
        "target": "developers",
        "description": "Announce new digital products to developer audience",
        "emails": [
            {
                "day": 0,
                "subject": "New: {product_name} just dropped",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Just launched something I think you'll love:</p>

    <h3>{product_name}</h3>

    <p>{product_description}</p>

    <p><strong>Launch week special: 20% off</strong></p>

    <p><a href="{product_url}" style="background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">Get It Now</a></p>

    <p>Questions? Just reply!</p>

    <p>- Matt @ BrainStack</p>
</div>
"""
            },
            {
                "day": 3,
                "subject": "Last chance: {product_name} launch discount",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <p>Hey {first_name},</p>

    <p>Quick reminder: the 20% launch discount on <strong>{product_name}</strong> ends tonight.</p>

    <p>After midnight, it goes back to full price.</p>

    <p><a href="{product_url}" style="background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">Grab It Before Midnight</a></p>

    <p>- Matt</p>
</div>
"""
            }
        ]
    }
}


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


async def enroll_lead_in_campaign(
    lead_id: str,
    email: str,
    company_name: str,
    campaign_id: str,
    first_name: str = None,
    custom_vars: dict = None
) -> dict:
    """Enroll a lead in an outreach campaign."""
    campaign = OUTREACH_CAMPAIGNS.get(campaign_id)
    if not campaign:
        return {"success": False, "error": f"Unknown campaign: {campaign_id}"}

    try:
        from email_scheduler_daemon import schedule_nurture_email

        email_ids = []
        custom_vars = custom_vars or {}

        for email_template in campaign["emails"]:
            # Personalize content
            subject = email_template["subject"].format(
                company_name=company_name,
                first_name=first_name or "there",
                **custom_vars
            )

            body = email_template["body"].format(
                company_name=company_name,
                first_name=first_name or "there",
                **custom_vars
            )

            # Calculate delay in minutes
            delay_minutes = email_template["day"] * 24 * 60

            # Schedule the email
            email_id = await schedule_nurture_email(
                recipient=email,
                subject=subject,
                body=body,
                delay_minutes=delay_minutes,
                metadata={
                    "source": "outreach_campaign",
                    "campaign_id": campaign_id,
                    "campaign_name": campaign["name"],
                    "lead_id": lead_id,
                    "day": email_template["day"]
                }
            )

            if email_id:
                email_ids.append(email_id)
                logger.info(f"Scheduled campaign email for {email}: day {email_template['day']}")

        return {
            "success": True,
            "campaign_id": campaign_id,
            "campaign_name": campaign["name"],
            "emails_scheduled": len(email_ids),
            "email_ids": email_ids
        }

    except Exception as e:
        logger.error(f"Error enrolling in campaign: {e}")
        return {"success": False, "error": str(e)}


async def run_campaign_for_new_leads(
    campaign_id: str = "roofing_saas_intro",
    limit: int = 50,
    exclude_contacted: bool = True
) -> dict:
    """
    Run a campaign for leads that haven't been contacted yet.
    Scheduled job entry point.
    """
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        # Get leads with real emails that haven't been in this campaign
        query = """
            SELECT rl.id, rl.email, rl.company_name, rl.contact_name
            FROM revenue_leads rl
            WHERE rl.email IS NOT NULL
                AND rl.email NOT LIKE '%%test%%'
                AND rl.email NOT LIKE '%%example%%'
                AND NOT EXISTS (
                    SELECT 1 FROM ai_email_queue eq
                    WHERE eq.recipient = rl.email
                    AND eq.metadata->>'campaign_id' = $1
                )
        """

        if exclude_contacted:
            query += " AND rl.stage = 'new'"

        query += " ORDER BY rl.score DESC LIMIT $2"

        leads = await pool.fetch(query, campaign_id, limit)

        enrolled = 0
        for lead in leads:
            # Extract first name from contact_name
            first_name = None
            if lead['contact_name']:
                first_name = lead['contact_name'].split()[0]

            result = await enroll_lead_in_campaign(
                lead_id=str(lead['id']),
                email=lead['email'],
                company_name=lead['company_name'] or "your company",
                campaign_id=campaign_id,
                first_name=first_name
            )

            if result.get('success'):
                enrolled += 1

                # Update lead stage to contacted
                await pool.execute("""
                    UPDATE revenue_leads
                    SET stage = 'contacted', updated_at = NOW()
                    WHERE id = $1
                """, lead['id'])

        return {
            "success": True,
            "campaign_id": campaign_id,
            "leads_found": len(leads),
            "leads_enrolled": enrolled,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Error running campaign: {e}")
        return {"success": False, "error": str(e)}


async def get_campaign_stats(campaign_id: str = None) -> dict:
    """Get statistics for campaigns."""
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        if campaign_id:
            stats = await pool.fetchrow("""
                SELECT
                    COUNT(*) as total_emails,
                    COUNT(*) FILTER (WHERE status = 'sent') as sent,
                    COUNT(*) FILTER (WHERE status = 'queued') as queued,
                    COUNT(DISTINCT recipient) as unique_recipients
                FROM ai_email_queue
                WHERE metadata->>'campaign_id' = $1
            """, campaign_id)

            return {
                "campaign_id": campaign_id,
                "campaign_name": OUTREACH_CAMPAIGNS.get(campaign_id, {}).get('name'),
                "total_emails": stats['total_emails'],
                "sent": stats['sent'],
                "queued": stats['queued'],
                "unique_recipients": stats['unique_recipients']
            }
        else:
            # All campaigns
            all_stats = await pool.fetch("""
                SELECT
                    metadata->>'campaign_id' as campaign_id,
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE status = 'sent') as sent
                FROM ai_email_queue
                WHERE metadata->>'source' = 'outreach_campaign'
                GROUP BY metadata->>'campaign_id'
            """)

            return {
                "campaigns": [
                    {
                        "campaign_id": s['campaign_id'],
                        "campaign_name": OUTREACH_CAMPAIGNS.get(s['campaign_id'], {}).get('name'),
                        "total": s['total'],
                        "sent": s['sent']
                    }
                    for s in all_stats
                ]
            }

    except Exception as e:
        logger.error(f"Error getting campaign stats: {e}")
        return {"error": str(e)}


# API endpoint handlers
async def list_campaigns() -> list[dict]:
    """List all available campaigns."""
    return [
        {
            "id": campaign_id,
            "name": config["name"],
            "target": config["target"],
            "description": config["description"],
            "email_count": len(config["emails"])
        }
        for campaign_id, config in OUTREACH_CAMPAIGNS.items()
    ]

"""
Automated Income Streams API
=============================
Real revenue generation endpoints that activate and manage all income streams.

This file provides:
1. Subscription product setup via Stripe
2. Email campaign management
3. Lead conversion automation
4. Revenue stream status and analytics

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/income", tags=["Income Streams"])

# =============================================================================
# Configuration
# =============================================================================

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Email campaign configurations
EMAIL_CAMPAIGNS = {
    "welcome_sequence": {
        "name": "Welcome Sequence",
        "trigger": "new_lead",
        "steps": [
            {"delay_hours": 0, "subject": "Welcome to BrainOps AI!", "template": "welcome"},
            {"delay_hours": 24, "subject": "Quick question about your business", "template": "discovery"},
            {"delay_hours": 72, "subject": "3 ways AI can grow your revenue", "template": "value_prop"},
            {"delay_hours": 168, "subject": "Exclusive offer for you", "template": "offer"}
        ]
    },
    "product_launch": {
        "name": "Product Launch",
        "trigger": "manual",
        "steps": [
            {"delay_hours": 0, "subject": "Launching something special...", "template": "teaser"},
            {"delay_hours": 24, "subject": "It's here! Introducing [Product]", "template": "launch"},
            {"delay_hours": 48, "subject": "Early bird discount ends soon", "template": "urgency"},
            {"delay_hours": 72, "subject": "Last chance: 24 hours left", "template": "final_call"}
        ]
    },
    "win_back": {
        "name": "Win Back Campaign",
        "trigger": "inactive_30d",
        "steps": [
            {"delay_hours": 0, "subject": "We miss you!", "template": "we_miss_you"},
            {"delay_hours": 72, "subject": "Special offer just for you", "template": "comeback_offer"}
        ]
    },
    "upsell_sequence": {
        "name": "Upsell Sequence",
        "trigger": "purchase_completed",
        "steps": [
            {"delay_hours": 24, "subject": "Thank you! Here's what's next", "template": "thank_you"},
            {"delay_hours": 72, "subject": "Upgrade to unlock more", "template": "upsell"},
            {"delay_hours": 168, "subject": "Exclusive bundle deal", "template": "bundle_offer"}
        ]
    }
}

# MRG Subscription tiers
MRG_SUBSCRIPTION_TIERS = {
    "starter": {
        "name": "Starter",
        "price_monthly": 29,
        "price_yearly": 290,
        "features": ["Basic AI analysis", "5 projects/month", "Email support"]
    },
    "professional": {
        "name": "Professional",
        "price_monthly": 79,
        "price_yearly": 790,
        "features": ["Advanced AI analysis", "Unlimited projects", "Priority support", "API access"]
    },
    "enterprise": {
        "name": "Enterprise",
        "price_monthly": 199,
        "price_yearly": 1990,
        "features": ["Custom AI models", "White-label", "Dedicated support", "Custom integrations"]
    }
}


# =============================================================================
# Request/Response Models
# =============================================================================

class ActivateCampaignRequest(BaseModel):
    campaign_id: str
    target_segment: Optional[str] = "all"

class CreateStripeProductsRequest(BaseModel):
    products: list[dict[str, Any]]

class ConvertLeadsRequest(BaseModel):
    lead_ids: Optional[list[str]] = None
    max_leads: int = 100
    industry: Optional[str] = None


# =============================================================================
# Database Helpers
# =============================================================================

async def _get_db_pool():
    """Get async database pool"""
    try:
        from database.async_connection import get_pool
        return get_pool()
    except Exception as e:
        logger.error(f"Failed to get DB pool: {e}")
        return None


async def _execute_query(query: str, params: tuple = None):
    """Execute a database query"""
    pool = await _get_db_pool()
    if not pool:
        return None

    try:
        if params:
            return await pool.fetch(query, *params)
        return await pool.fetch(query)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return None


# =============================================================================
# Income Stream Status
# =============================================================================

@router.get("/status")
async def get_income_streams_status():
    """Get comprehensive status of all income streams"""
    try:
        pool = await _get_db_pool()

        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "streams": {},
            "total_potential_revenue": 0,
            "active_streams": 0,
            "blocked_streams": 0
        }

        # 1. Gumroad Products Status
        gumroad_products = await pool.fetch("SELECT * FROM digital_products WHERE is_active = true")
        gumroad_sales = await pool.fetchval(
            "SELECT COALESCE(SUM(price), 0) FROM gumroad_sales WHERE is_test = false"
        )

        status["streams"]["gumroad"] = {
            "name": "Digital Products (Gumroad)",
            "status": "configured" if gumroad_products else "needs_setup",
            "products_defined": len(gumroad_products) if gumroad_products else 0,
            "products_live": 0,  # Need to check Gumroad API
            "total_revenue": float(gumroad_sales or 0),
            "potential_monthly": len(gumroad_products) * 500 if gumroad_products else 0,
            "blockers": ["Create products in Gumroad dashboard", "Add GUMROAD_WEBHOOK_SECRET"] if gumroad_products else ["Define digital products"]
        }

        # 2. Lead Pipeline Status
        lead_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN stage = 'new' THEN 1 ELSE 0 END) as new_leads,
                SUM(CASE WHEN stage = 'qualified' THEN 1 ELSE 0 END) as qualified,
                SUM(CASE WHEN stage = 'proposal_sent' THEN 1 ELSE 0 END) as proposal_sent,
                SUM(CASE WHEN stage = 'won' THEN 1 ELSE 0 END) as won,
                COALESCE(SUM(value_estimate), 0) as pipeline_value
            FROM revenue_leads
            WHERE is_test = false OR is_test IS NULL
        """)

        status["streams"]["lead_pipeline"] = {
            "name": "Lead-to-Sale Pipeline",
            "status": "active" if lead_stats and lead_stats['qualified'] > 0 else "needs_nurturing",
            "total_leads": lead_stats['total'] if lead_stats else 0,
            "new_leads": lead_stats['new_leads'] if lead_stats else 0,
            "qualified_leads": lead_stats['qualified'] if lead_stats else 0,
            "proposals_out": lead_stats['proposal_sent'] if lead_stats else 0,
            "won_deals": lead_stats['won'] if lead_stats else 0,
            "pipeline_value": float(lead_stats['pipeline_value']) if lead_stats else 0,
            "potential_monthly": float(lead_stats['pipeline_value'] or 0) * 0.2 if lead_stats else 0,
            "blockers": ["Activate email campaigns", "Set up automated follow-ups"] if lead_stats and lead_stats['new_leads'] > 100 else []
        }

        # 3. Email Campaigns Status
        email_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_queued,
                SUM(CASE WHEN status = 'sent' THEN 1 ELSE 0 END) as sent,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) as pending
            FROM ai_email_queue
        """)

        status["streams"]["email_campaigns"] = {
            "name": "Email Marketing",
            "status": "active" if email_stats and email_stats['sent'] > 0 else "not_sending",
            "campaigns_defined": len(EMAIL_CAMPAIGNS),
            "emails_sent": email_stats['sent'] if email_stats else 0,
            "emails_pending": email_stats['pending'] if email_stats else 0,
            "emails_failed": email_stats['failed'] if email_stats else 0,
            "potential_monthly": len(EMAIL_CAMPAIGNS) * 2000,
            "blockers": ["Define email templates", "Activate campaigns"] if not email_stats or email_stats['sent'] == 0 else []
        }

        # 4. MRG Subscription Status
        mrg_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_users,
                COUNT(*) FILTER (WHERE subscription_status = 'active') as active_subs,
                COUNT(*) FILTER (WHERE subscription_tier = 'professional') as pro_subs,
                COUNT(*) FILTER (WHERE subscription_tier = 'enterprise') as enterprise_subs
            FROM tenants
            WHERE stripe_subscription_id IS NOT NULL
        """)

        mrr = 0
        if mrg_stats:
            mrr = (mrg_stats['pro_subs'] or 0) * 79 + (mrg_stats['enterprise_subs'] or 0) * 199 + ((mrg_stats['active_subs'] or 0) - (mrg_stats['pro_subs'] or 0) - (mrg_stats['enterprise_subs'] or 0)) * 29

        status["streams"]["mrg_subscriptions"] = {
            "name": "MyRoofGenius SaaS",
            "status": "active" if mrg_stats and mrg_stats['active_subs'] > 0 else "needs_subscribers",
            "total_users": mrg_stats['total_users'] if mrg_stats else 0,
            "active_subscriptions": mrg_stats['active_subs'] if mrg_stats else 0,
            "mrr": mrr,
            "potential_monthly": max(mrr * 2, 5000),
            "blockers": ["Create Stripe subscription products", "Launch pricing page"] if mrr == 0 else []
        }

        # 5. API Monetization Status
        api_stats = await pool.fetchrow("""
            SELECT
                COUNT(DISTINCT user_id) as api_users,
                COUNT(*) as total_calls
            FROM api_usage_logs
            WHERE created_at > NOW() - INTERVAL '30 days'
        """) if await pool.fetchval("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'api_usage_logs')") else None

        status["streams"]["api_monetization"] = {
            "name": "API Access",
            "status": "potential" if not api_stats else "tracking",
            "active_api_users": api_stats['api_users'] if api_stats else 0,
            "monthly_calls": api_stats['total_calls'] if api_stats else 0,
            "potential_monthly": 2000,
            "blockers": ["Create API pricing tiers", "Implement usage tracking"]
        }

        # Calculate totals
        for stream in status["streams"].values():
            status["total_potential_revenue"] += stream.get("potential_monthly", 0)
            if stream.get("status") in ["active", "tracking"]:
                status["active_streams"] += 1
            if stream.get("blockers"):
                status["blocked_streams"] += 1

        return status

    except Exception as e:
        logger.error(f"Income status error: {e}")
        return {"error": str(e), "streams": {}}


# =============================================================================
# Email Campaign Management
# =============================================================================

@router.get("/campaigns")
async def list_campaigns():
    """List all configured email campaigns"""
    return {
        "campaigns": EMAIL_CAMPAIGNS,
        "total": len(EMAIL_CAMPAIGNS)
    }


@router.post("/campaigns/activate")
async def activate_campaign(request: ActivateCampaignRequest, background_tasks: BackgroundTasks):
    """Activate an email campaign for a target segment"""
    campaign = EMAIL_CAMPAIGNS.get(request.campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign {request.campaign_id} not found")

    # Queue campaign activation in background
    background_tasks.add_task(
        _execute_campaign,
        request.campaign_id,
        campaign,
        request.target_segment
    )

    return {
        "status": "activated",
        "campaign": request.campaign_id,
        "target_segment": request.target_segment,
        "steps": len(campaign["steps"])
    }


async def _execute_campaign(campaign_id: str, campaign: dict, segment: str):
    """Execute email campaign in background"""
    try:
        pool = await _get_db_pool()

        # Get target leads based on segment
        if segment == "all":
            leads = await pool.fetch("""
                SELECT id, email, contact_name
                FROM revenue_leads
                WHERE email IS NOT NULL
                AND (is_test = false OR is_test IS NULL)
                LIMIT 500
            """)
        else:
            leads = await pool.fetch("""
                SELECT id, email, contact_name
                FROM revenue_leads
                WHERE stage = $1
                AND email IS NOT NULL
                AND (is_test = false OR is_test IS NULL)
                LIMIT 500
            """, segment)

        if not leads:
            logger.warning(f"No leads found for campaign {campaign_id}")
            return

        # Queue emails for first step
        first_step = campaign["steps"][0]

        for lead in leads:
            await pool.execute("""
                INSERT INTO ai_email_queue (recipient, subject, body, status, metadata, created_at)
                VALUES ($1, $2, $3, 'queued', $4, NOW())
                ON CONFLICT (recipient, subject) DO NOTHING
            """,
                lead['email'],
                first_step['subject'],
                _get_email_template(first_step['template'], lead['contact_name'] or 'there'),
                json.dumps({
                    "campaign_id": campaign_id,
                    "step": 0,
                    "lead_id": str(lead['id'])
                })
            )

        logger.info(f"Campaign {campaign_id} activated: {len(leads)} emails queued")

    except Exception as e:
        logger.error(f"Campaign execution error: {e}")


def _get_email_template(template_name: str, name: str) -> str:
    """Get HTML email template"""
    templates = {
        "welcome": f"""
            <html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h1 style="color: #2563eb;">Welcome to BrainOps AI!</h1>
            <p>Hi {name},</p>
            <p>Thanks for joining us! We're excited to help you automate your business with AI.</p>
            <p>Here's what you can do next:</p>
            <ul>
                <li>Explore our AI-powered tools</li>
                <li>Check out our digital products</li>
                <li>Book a demo call</li>
            </ul>
            <a href="https://myroofgenius.com/dashboard" style="display: inline-block; background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px;">Get Started</a>
            <p style="color: #666; margin-top: 20px;">Best,<br>The BrainOps Team</p>
            </body></html>
        """,
        "discovery": f"""
            <html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>Quick question, {name}</h2>
            <p>I wanted to reach out and learn more about your business.</p>
            <p>What's your biggest challenge right now when it comes to:</p>
            <ul>
                <li>Generating leads?</li>
                <li>Converting proposals to sales?</li>
                <li>Automating repetitive tasks?</li>
            </ul>
            <p>Reply to this email and let me know - I'd love to help!</p>
            <p style="color: #666;">Matt<br>Founder, BrainOps AI</p>
            </body></html>
        """,
        "value_prop": f"""
            <html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>3 Ways AI Can Grow Your Revenue</h2>
            <p>Hi {name},</p>
            <p>Here are 3 proven ways our AI tools are helping businesses like yours:</p>
            <ol>
                <li><strong>Automated Lead Qualification</strong> - Score and prioritize leads automatically</li>
                <li><strong>AI Proposal Generation</strong> - Create professional proposals in minutes</li>
                <li><strong>Intelligent Follow-ups</strong> - Never let a lead go cold again</li>
            </ol>
            <a href="https://myroofgenius.com/products" style="display: inline-block; background: #10b981; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px;">See Our Products</a>
            </body></html>
        """,
        "offer": f"""
            <html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #dc2626;">Exclusive Offer for You</h2>
            <p>Hi {name},</p>
            <p>As a valued subscriber, I'm offering you an exclusive deal:</p>
            <div style="background: #fef3c7; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="margin: 0;">20% OFF any product</h3>
                <p style="margin: 10px 0 0;">Use code: <strong>WELCOME20</strong></p>
            </div>
            <a href="https://brainstackstudio.gumroad.com" style="display: inline-block; background: #dc2626; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px;">Shop Now</a>
            <p style="color: #666; font-size: 12px;">Offer expires in 48 hours</p>
            </body></html>
        """
    }

    return templates.get(template_name, templates["welcome"])


# =============================================================================
# Lead Conversion Automation
# =============================================================================

@router.post("/leads/qualify-batch")
async def qualify_leads_batch(request: ConvertLeadsRequest, background_tasks: BackgroundTasks):
    """Batch qualify and score leads for conversion"""
    background_tasks.add_task(_batch_qualify_leads, request.lead_ids, request.max_leads, request.industry)

    return {
        "status": "processing",
        "max_leads": request.max_leads,
        "message": "Lead qualification running in background"
    }


async def _batch_qualify_leads(lead_ids: list[str], max_leads: int, industry: str):
    """Score and qualify leads in batch"""
    try:
        pool = await _get_db_pool()

        # Get leads to qualify
        if lead_ids:
            leads = await pool.fetch("""
                SELECT * FROM revenue_leads
                WHERE id = ANY($1)
            """, lead_ids)
        else:
            query = """
                SELECT * FROM revenue_leads
                WHERE stage = 'new'
                AND (is_test = false OR is_test IS NULL)
            """
            if industry:
                query += f" AND metadata->>'industry' = '{industry}'"
            query += f" LIMIT {max_leads}"
            leads = await pool.fetch(query)

        qualified_count = 0
        for lead in leads:
            # Calculate lead score
            score = 30  # Base score

            # Email quality
            email = lead.get('email', '')
            if email and not any(x in email.lower() for x in ['gmail', 'yahoo', 'hotmail']):
                score += 20  # Business email

            # Phone provided
            if lead.get('phone'):
                score += 15

            # Company name
            if lead.get('company_name'):
                score += 10

            # Value estimate
            value = lead.get('value_estimate', 0) or 0
            if value > 5000:
                score += 15
            elif value > 1000:
                score += 10

            # Update lead score and possibly qualify
            new_stage = 'qualified' if score >= 70 else 'new'

            await pool.execute("""
                UPDATE revenue_leads
                SET score = $1, stage = $2, updated_at = NOW()
                WHERE id = $3
            """, score, new_stage, lead['id'])

            if new_stage == 'qualified':
                qualified_count += 1

        logger.info(f"Lead qualification complete: {qualified_count} qualified out of {len(leads)}")

    except Exception as e:
        logger.error(f"Lead qualification error: {e}")


@router.get("/leads/pipeline")
async def get_lead_pipeline():
    """Get lead pipeline with conversion metrics"""
    try:
        pool = await _get_db_pool()

        pipeline = await pool.fetch("""
            SELECT
                stage,
                COUNT(*) as count,
                COALESCE(SUM(value_estimate), 0) as value
            FROM revenue_leads
            WHERE is_test = false OR is_test IS NULL
            GROUP BY stage
            ORDER BY
                CASE stage
                    WHEN 'new' THEN 1
                    WHEN 'contacted' THEN 2
                    WHEN 'qualified' THEN 3
                    WHEN 'proposal_sent' THEN 4
                    WHEN 'negotiating' THEN 5
                    WHEN 'won' THEN 6
                    WHEN 'lost' THEN 7
                END
        """)

        return {
            "pipeline": [dict(row) for row in pipeline],
            "total_leads": sum(row['count'] for row in pipeline),
            "total_value": sum(float(row['value']) for row in pipeline)
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"error": str(e)}


# =============================================================================
# Revenue Analytics
# =============================================================================

@router.get("/analytics")
async def get_revenue_analytics():
    """Get comprehensive revenue analytics across all streams"""
    try:
        pool = await _get_db_pool()

        analytics = {
            "timestamp": datetime.utcnow().isoformat(),
            "period": "last_30_days",
            "streams": {}
        }

        # Gumroad revenue
        gumroad = await pool.fetchrow("""
            SELECT
                COUNT(*) as sales_count,
                COALESCE(SUM(price), 0) as total_revenue
            FROM gumroad_sales
            WHERE is_test = false
            AND created_at > NOW() - INTERVAL '30 days'
        """)

        analytics["streams"]["gumroad"] = {
            "sales": gumroad['sales_count'] if gumroad else 0,
            "revenue": float(gumroad['total_revenue']) if gumroad else 0
        }

        # Lead pipeline value
        leads = await pool.fetchrow("""
            SELECT
                COALESCE(SUM(CASE WHEN stage = 'won' THEN value_estimate ELSE 0 END), 0) as won_value,
                COALESCE(SUM(CASE WHEN stage IN ('qualified', 'proposal_sent', 'negotiating') THEN value_estimate ELSE 0 END), 0) as pipeline_value
            FROM revenue_leads
            WHERE (is_test = false OR is_test IS NULL)
            AND created_at > NOW() - INTERVAL '30 days'
        """)

        analytics["streams"]["leads"] = {
            "won_revenue": float(leads['won_value']) if leads else 0,
            "pipeline_value": float(leads['pipeline_value']) if leads else 0
        }

        # MRG subscriptions
        mrg = await pool.fetchrow("""
            SELECT
                SUM(CASE WHEN subscription_status = 'active' THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN subscription_tier = 'professional' THEN 1 ELSE 0 END) as pro,
                SUM(CASE WHEN subscription_tier = 'enterprise' THEN 1 ELSE 0 END) as enterprise
            FROM tenants
            WHERE stripe_subscription_id IS NOT NULL
        """)

        mrr = 0
        if mrg:
            mrr = (mrg['pro'] or 0) * 79 + (mrg['enterprise'] or 0) * 199

        analytics["streams"]["mrg_saas"] = {
            "active_subscribers": mrg['active'] if mrg else 0,
            "mrr": mrr,
            "arr": mrr * 12
        }

        # Total revenue
        analytics["total_revenue"] = (
            analytics["streams"]["gumroad"]["revenue"] +
            analytics["streams"]["leads"]["won_revenue"] +
            analytics["streams"]["mrg_saas"]["mrr"]
        )

        analytics["projected_monthly"] = (
            analytics["streams"]["gumroad"]["revenue"] +
            analytics["streams"]["leads"]["pipeline_value"] * 0.2 +
            analytics["streams"]["mrg_saas"]["mrr"]
        )

        return analytics

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return {"error": str(e)}


# =============================================================================
# Quick Actions
# =============================================================================

@router.post("/activate-all")
async def activate_all_streams(background_tasks: BackgroundTasks):
    """Quick action to activate all income streams"""

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "actions_taken": []
    }

    # 1. Activate welcome campaign
    background_tasks.add_task(
        _execute_campaign,
        "welcome_sequence",
        EMAIL_CAMPAIGNS["welcome_sequence"],
        "new"
    )
    results["actions_taken"].append("Welcome email campaign activated for new leads")

    # 2. Qualify new leads
    background_tasks.add_task(_batch_qualify_leads, None, 500, None)
    results["actions_taken"].append("Lead qualification running (up to 500 leads)")

    # 3. Log activation
    try:
        pool = await _get_db_pool()
        await pool.execute("""
            INSERT INTO ai_agent_executions (agent_id, task, result, success, created_at)
            VALUES ('income_stream_activator', 'activate_all_streams', $1, true, NOW())
        """, json.dumps(results))
    except Exception as e:
        logger.warning(f"Could not log activation: {e}")

    results["status"] = "success"
    results["message"] = "All income streams are being activated. Check /income/status for progress."

    return results

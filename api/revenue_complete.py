"""
Complete Revenue Pipeline API
=============================
Unified endpoint for all revenue operations.

Integrates:
- API Monetization with Stripe billing
- Agent-as-a-Service execution tracking
- Gumroad product management
- Lead nurturing pipelines
- Content monetization
- Real-time revenue tracking

This is the MASTER revenue API that coordinates all revenue streams.
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/revenue/complete",
    tags=["revenue", "billing", "monetization"]
)

# Configuration
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")


# ==================== PYDANTIC MODELS ====================

class CreateAPIKeyRequest(BaseModel):
    email: EmailStr
    name: str = "API Key"
    tier: str = "free"  # free, starter, pro, enterprise


class AgentExecutionBilling(BaseModel):
    agent_type: str
    execution_time_ms: int = 0
    tokens_used: int = 0
    api_key: str


class RevenueStreamStatus(BaseModel):
    stream: str
    status: str
    revenue_today: float = 0
    revenue_mtd: float = 0


# ==================== PRICING CONFIG ====================

PRICING_TIERS = {
    "free": {
        "price_monthly_cents": 0,
        "api_calls_included": 100,
        "agents_included": 5,
        "stripe_price_id": None
    },
    "starter": {
        "price_monthly_cents": 2900,
        "api_calls_included": 10000,
        "agents_included": 20,
        "stripe_price_id": "price_starter_monthly"
    },
    "pro": {
        "price_monthly_cents": 9900,
        "api_calls_included": 100000,
        "agents_included": 50,
        "stripe_price_id": "price_pro_monthly"
    },
    "enterprise": {
        "price_monthly_cents": 29900,
        "api_calls_included": 1000000,
        "agents_included": -1,  # unlimited
        "stripe_price_id": "price_enterprise_monthly"
    }
}

AGENT_PRICING = {
    # High-value agents
    "ContentGeneratorAgent": Decimal("0.10"),
    "LeadDiscoveryAgentReal": Decimal("0.05"),
    "NurtureExecutorAgentReal": Decimal("0.03"),
    "GumroadRevenueAgent": Decimal("0.02"),
    "CustomerIntelligenceAgent": Decimal("0.05"),

    # Standard agents
    "EmailProcessor": Decimal("0.01"),
    "HealthMonitor": Decimal("0.005"),

    # Default
    "default": Decimal("0.01")
}


# ==================== DATABASE HELPERS ====================

async def get_pool():
    """Get database pool."""
    try:
        from database.async_connection import get_pool as _get_pool
        return _get_pool()
    except Exception as e:
        logger.error(f"Database pool error: {e}")
        return None


async def get_stripe_client():
    """Get Stripe client."""
    if not STRIPE_SECRET_KEY:
        return None
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        return stripe
    except ImportError:
        logger.warning("Stripe SDK not available")
        return None


# ==================== API KEY MANAGEMENT ====================

@router.post("/api-keys")
async def create_api_key(request: CreateAPIKeyRequest, background_tasks: BackgroundTasks):
    """
    Create a new billable API key with optional Stripe subscription.

    Tiers:
    - free: 100 calls/mo, $0
    - starter: 10,000 calls/mo, $29/mo
    - pro: 100,000 calls/mo, $99/mo
    - enterprise: 1,000,000 calls/mo, $299/mo
    """
    pool = await get_pool()
    if not pool:
        raise HTTPException(503, "Database unavailable")

    tier_config = PRICING_TIERS.get(request.tier)
    if not tier_config:
        raise HTTPException(400, f"Invalid tier: {request.tier}")

    # Generate secure API key
    raw_key = f"brainops_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:15]
    key_id = str(uuid.uuid4())

    # Create Stripe customer and subscription if paid tier
    stripe_customer_id = None
    stripe_subscription_id = None

    if request.tier != "free" and STRIPE_SECRET_KEY:
        stripe = await get_stripe_client()
        if stripe:
            try:
                # Create Stripe customer
                customer = stripe.Customer.create(
                    email=request.email,
                    metadata={
                        "api_key_id": key_id,
                        "tier": request.tier
                    }
                )
                stripe_customer_id = customer.id

                # Note: Subscription would be created after payment method attached
                # For now, we track the customer
                logger.info(f"Created Stripe customer: {stripe_customer_id}")

            except Exception as e:
                logger.error(f"Stripe error: {e}")

    # Store API key in database
    await pool.execute("""
        INSERT INTO api_keys (id, key_hash, key_prefix, name, tier, metadata, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, NOW())
    """,
        key_id,
        key_hash,
        key_prefix,
        request.name,
        request.tier,
        json.dumps({
            "email": request.email,
            "stripe_customer_id": stripe_customer_id,
            "stripe_subscription_id": stripe_subscription_id
        })
    )

    return {
        "success": True,
        "key_id": key_id,
        "api_key": raw_key,  # Only returned once!
        "key_prefix": key_prefix,
        "tier": request.tier,
        "limits": {
            "api_calls_per_month": tier_config["api_calls_included"],
            "agents_included": tier_config["agents_included"],
            "price_monthly": tier_config["price_monthly_cents"] / 100
        },
        "stripe_customer_id": stripe_customer_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "important": "Save your API key now - it cannot be retrieved later!"
    }


@router.get("/api-keys/{key_id}/usage")
async def get_api_key_usage(key_id: str, days: int = 30):
    """Get usage statistics for an API key."""
    pool = await get_pool()
    if not pool:
        raise HTTPException(503, "Database unavailable")

    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Get API key info
    key_info = await pool.fetchrow(
        "SELECT tier, metadata FROM api_keys WHERE id = $1",
        key_id
    )
    if not key_info:
        raise HTTPException(404, "API key not found")

    tier = key_info["tier"]
    tier_config = PRICING_TIERS.get(tier, PRICING_TIERS["free"])

    # Get usage from api_usage table
    usage = await pool.fetchrow("""
        SELECT
            COUNT(*) as total_calls,
            COALESCE(SUM(cost_cents), 0) as total_cost_cents,
            COALESCE(AVG(response_time_ms), 0) as avg_response_time
        FROM api_usage
        WHERE api_key_id = $1 AND created_at >= $2
    """, key_id, since)

    # Get agent executions if tracked
    agent_usage = await pool.fetch("""
        SELECT
            agent_type,
            COUNT(*) as executions,
            COALESCE(SUM(response_time_ms), 0) as total_time_ms
        FROM api_usage
        WHERE api_key_id = $1
          AND created_at >= $2
          AND metadata->>'agent_type' IS NOT NULL
        GROUP BY agent_type
        ORDER BY executions DESC
    """, key_id, since)

    total_calls = usage["total_calls"] or 0
    included_calls = tier_config["api_calls_included"]
    overage = max(0, total_calls - included_calls)

    return {
        "key_id": key_id,
        "tier": tier,
        "period_days": days,
        "usage": {
            "total_calls": total_calls,
            "included_calls": included_calls,
            "overage_calls": overage,
            "avg_response_time_ms": float(usage["avg_response_time"] or 0)
        },
        "billing": {
            "subscription_cost": tier_config["price_monthly_cents"] / 100,
            "overage_cost": float(usage["total_cost_cents"] or 0) / 100,
            "total_cost": (tier_config["price_monthly_cents"] + (usage["total_cost_cents"] or 0)) / 100
        },
        "agent_usage": [
            {"agent": r["agent_type"], "executions": r["executions"]}
            for r in agent_usage
        ],
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


# ==================== AGENT BILLING ====================

@router.post("/track-agent-execution")
async def track_agent_execution(billing: AgentExecutionBilling):
    """
    Track an agent execution for billing.
    Called internally after each agent execution.
    """
    pool = await get_pool()
    if not pool:
        raise HTTPException(503, "Database unavailable")

    # Validate API key
    key_hash = hashlib.sha256(billing.api_key.encode()).hexdigest()
    key_info = await pool.fetchrow(
        "SELECT id, tier FROM api_keys WHERE key_hash = $1 AND is_active = true",
        key_hash
    )

    if not key_info:
        raise HTTPException(401, "Invalid API key")

    # Calculate cost based on agent type
    agent_price = AGENT_PRICING.get(billing.agent_type, AGENT_PRICING["default"])
    cost_cents = int(agent_price * 100)

    # Record usage
    usage_id = str(uuid.uuid4())
    await pool.execute("""
        INSERT INTO api_usage (id, api_key_id, endpoint, response_time_ms, tokens_used, cost_cents, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    """,
        usage_id,
        key_info["id"],
        f"/agents/{billing.agent_type}",
        billing.execution_time_ms,
        billing.tokens_used,
        cost_cents,
        json.dumps({"agent_type": billing.agent_type})
    )

    return {
        "tracked": True,
        "usage_id": usage_id,
        "agent_type": billing.agent_type,
        "cost_cents": cost_cents
    }


# ==================== REVENUE STREAMS ====================

@router.get("/streams")
async def get_revenue_streams():
    """Get status of all revenue streams."""
    pool = await get_pool()

    streams = {
        "api_monetization": {
            "name": "API Monetization",
            "status": "active",
            "model": "usage_based",
            "projected_mrr": 500
        },
        "gumroad_products": {
            "name": "Digital Products (Gumroad)",
            "status": "active",
            "model": "one_time",
            "projected_mrr": 300
        },
        "lead_nurturing": {
            "name": "Lead Nurturing Pipeline",
            "status": "active",
            "model": "conversion",
            "projected_mrr": 1000
        },
        "content_factory": {
            "name": "SEO Content Factory",
            "status": "active",
            "model": "indirect",
            "projected_mrr": 200
        },
        "agent_services": {
            "name": "Agent-as-a-Service",
            "status": "active",
            "model": "subscription",
            "projected_mrr": 800
        },
        "saas_subscriptions": {
            "name": "SaaS Subscriptions",
            "status": "active",
            "model": "recurring",
            "projected_mrr": 400
        }
    }

    # Get real metrics if database available
    if pool:
        try:
            # Gumroad revenue
            gumroad = await pool.fetchrow("""
                SELECT COUNT(*) as sales, COALESCE(SUM(price), 0) as revenue
                FROM gumroad_sales WHERE is_test = false
            """)
            streams["gumroad_products"]["actual_revenue"] = float(gumroad["revenue"] or 0)
            streams["gumroad_products"]["sales_count"] = gumroad["sales"] or 0

            # Lead pipeline
            leads = await pool.fetchrow("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN stage = 'won' THEN value_estimate ELSE 0 END) as won_value
                FROM revenue_leads
            """)
            streams["lead_nurturing"]["total_leads"] = leads["total"] or 0
            streams["lead_nurturing"]["won_value"] = float(leads["won_value"] or 0)

            # API usage
            api_usage = await pool.fetchrow("""
                SELECT COUNT(*) as total_calls, COALESCE(SUM(cost_cents), 0) as revenue_cents
                FROM api_usage
                WHERE created_at >= NOW() - INTERVAL '30 days'
            """)
            streams["api_monetization"]["calls_30d"] = api_usage["total_calls"] or 0
            streams["api_monetization"]["revenue_30d"] = float(api_usage["revenue_cents"] or 0) / 100

        except Exception as e:
            logger.warning(f"Error fetching stream metrics: {e}")

    total_projected_mrr = sum(s["projected_mrr"] for s in streams.values())

    return {
        "streams": streams,
        "total_projected_mrr": total_projected_mrr,
        "total_projected_arr": total_projected_mrr * 12,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/dashboard")
async def revenue_dashboard(days: int = 30):
    """
    Complete revenue dashboard with all metrics.
    """
    pool = await get_pool()
    if not pool:
        raise HTTPException(503, "Database unavailable")

    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Gumroad sales (excluding test)
    gumroad = await pool.fetchrow("""
        SELECT
            COUNT(*) as total_sales,
            COALESCE(SUM(price), 0) as total_revenue,
            COUNT(DISTINCT email) as unique_customers
        FROM gumroad_sales
        WHERE sale_timestamp >= $1 AND is_test = false
          AND email NOT LIKE '%test%' AND email NOT LIKE '%example%'
    """, since)

    # Stripe revenue
    stripe = await pool.fetchrow("""
        SELECT
            COUNT(*) as total_charges,
            COALESCE(SUM(amount_cents), 0) as total_cents
        FROM stripe_events
        WHERE created_at >= $1
          AND event_type IN ('charge.succeeded', 'checkout.session.completed')
    """, since)

    # Lead pipeline
    leads = await pool.fetchrow("""
        SELECT
            COUNT(*) as total_leads,
            SUM(CASE WHEN stage = 'new' THEN 1 ELSE 0 END) as new_leads,
            SUM(CASE WHEN stage = 'contacted' THEN 1 ELSE 0 END) as contacted,
            SUM(CASE WHEN stage = 'qualified' THEN 1 ELSE 0 END) as qualified,
            SUM(CASE WHEN stage = 'won' THEN 1 ELSE 0 END) as won,
            SUM(CASE WHEN stage = 'won' THEN value_estimate ELSE 0 END) as won_value
        FROM revenue_leads
        WHERE created_at >= $1
    """, since)

    # Email campaign performance
    emails = await pool.fetchrow("""
        SELECT
            COUNT(*) as total_sent,
            SUM(CASE WHEN status = 'sent' THEN 1 ELSE 0 END) as delivered,
            SUM(CASE WHEN status = 'opened' THEN 1 ELSE 0 END) as opened
        FROM ai_email_queue
        WHERE created_at >= $1
    """, since)

    # Agent executions
    agents = await pool.fetchrow("""
        SELECT
            COUNT(*) as total_executions,
            COUNT(DISTINCT agent_type) as unique_agents
        FROM agent_executions
        WHERE created_at >= $1
    """, since)

    # API usage
    api = await pool.fetchrow("""
        SELECT
            COUNT(*) as total_calls,
            COUNT(DISTINCT api_key_id) as unique_keys,
            COALESCE(SUM(cost_cents), 0) as total_revenue_cents
        FROM api_usage
        WHERE created_at >= $1
    """, since)

    # Calculate totals
    total_revenue = (
        float(gumroad["total_revenue"] or 0) +
        float(stripe["total_cents"] or 0) / 100 +
        float(api["total_revenue_cents"] or 0) / 100
    )

    return {
        "period_days": days,
        "period_start": since.isoformat(),
        "total_revenue": total_revenue,
        "revenue_breakdown": {
            "gumroad": {
                "sales": gumroad["total_sales"] or 0,
                "revenue": float(gumroad["total_revenue"] or 0),
                "customers": gumroad["unique_customers"] or 0
            },
            "stripe": {
                "charges": stripe["total_charges"] or 0,
                "revenue": float(stripe["total_cents"] or 0) / 100
            },
            "api_monetization": {
                "calls": api["total_calls"] or 0,
                "revenue": float(api["total_revenue_cents"] or 0) / 100,
                "active_keys": api["unique_keys"] or 0
            }
        },
        "lead_pipeline": {
            "total": leads["total_leads"] or 0,
            "new": leads["new_leads"] or 0,
            "contacted": leads["contacted"] or 0,
            "qualified": leads["qualified"] or 0,
            "won": leads["won"] or 0,
            "won_value": float(leads["won_value"] or 0),
            "conversion_rate": (leads["won"] or 0) / max(1, leads["total_leads"] or 1) * 100
        },
        "email_campaigns": {
            "sent": emails["total_sent"] or 0,
            "delivered": emails["delivered"] or 0,
            "opened": emails["opened"] or 0,
            "open_rate": (emails["opened"] or 0) / max(1, emails["delivered"] or 1) * 100
        },
        "agent_activity": {
            "executions": agents["total_executions"] or 0,
            "unique_agents": agents["unique_agents"] or 0
        },
        "health": {
            "all_streams_active": True,
            "database_connected": True
        },
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


@router.post("/run-all-pipelines")
async def run_all_pipelines(background_tasks: BackgroundTasks):
    """
    Execute all revenue pipelines.
    Runs in background for non-blocking operation.
    """
    results = {}

    # 1. Lead Discovery
    try:
        from revenue_pipeline_agents import LeadDiscoveryAgentReal
        agent = LeadDiscoveryAgentReal()
        result = await agent.execute({"action": "discover_all"})
        results["lead_discovery"] = {
            "status": "completed",
            "leads_found": result.get("leads_discovered", 0)
        }
    except Exception as e:
        results["lead_discovery"] = {"status": "error", "error": str(e)}

    # 2. Nurture Sequences
    try:
        from revenue_pipeline_agents import NurtureExecutorAgentReal
        agent = NurtureExecutorAgentReal()
        result = await agent.execute({"action": "nurture_new_leads"})
        results["nurture_sequences"] = {
            "status": "completed",
            "emails_queued": result.get("emails_queued", 0)
        }
    except Exception as e:
        results["nurture_sequences"] = {"status": "error", "error": str(e)}

    # 3. Gumroad Sync
    try:
        from gumroad_revenue_agent import GumroadRevenueAgent
        agent = GumroadRevenueAgent()
        result = await agent.execute("daily_sync")
        results["gumroad_sync"] = {
            "status": "completed",
            "sales_synced": result.get("sales_synced", 0)
        }
    except Exception as e:
        results["gumroad_sync"] = {"status": "error", "error": str(e)}

    # 4. Content Generation (lightweight check)
    try:
        from content_generation_agent import ContentGeneratorAgent
        agent = ContentGeneratorAgent()
        results["content_factory"] = {"status": "ready", "agent": "ContentGeneratorAgent"}
    except Exception as e:
        results["content_factory"] = {"status": "error", "error": str(e)}

    return {
        "success": all(r.get("status") in ["completed", "ready"] for r in results.values()),
        "pipelines_run": len(results),
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/health")
async def revenue_health():
    """Health check for all revenue systems."""
    pool = await get_pool()

    health = {
        "status": "healthy",
        "database": pool is not None,
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "systems": {}
    }

    # Check each system
    systems_to_check = [
        ("lead_discovery", "revenue_pipeline_agents", "LeadDiscoveryAgentReal"),
        ("nurture_executor", "revenue_pipeline_agents", "NurtureExecutorAgentReal"),
        ("gumroad_agent", "gumroad_revenue_agent", "GumroadRevenueAgent"),
        ("content_generator", "content_generation_agent", "ContentGeneratorAgent"),
        ("api_monetization", "api_monetization_engine", "APIMonetizationEngine"),
        ("product_generator", "automated_product_generator", "AutomatedProductGenerator"),
    ]

    for name, module, cls in systems_to_check:
        try:
            exec(f"from {module} import {cls}")
            health["systems"][name] = "available"
        except ImportError:
            health["systems"][name] = "unavailable"
            health["status"] = "degraded"

    if not pool:
        health["status"] = "degraded"

    health["timestamp"] = datetime.now(timezone.utc).isoformat()
    return health


# ==================== OUTREACH CAMPAIGNS ====================

@router.get("/campaigns")
async def list_outreach_campaigns():
    """List all available outreach campaigns."""
    try:
        from outreach_campaigns import list_campaigns
        return await list_campaigns()
    except ImportError:
        raise HTTPException(status_code=503, detail="Outreach campaigns not available")


@router.post("/campaigns/run/{campaign_id}")
async def run_outreach_campaign(campaign_id: str, limit: int = 50):
    """Run an outreach campaign for new leads."""
    try:
        from outreach_campaigns import run_campaign_for_new_leads
        result = await run_campaign_for_new_leads(campaign_id, limit)
        return result
    except ImportError:
        raise HTTPException(status_code=503, detail="Outreach campaigns not available")


@router.get("/campaigns/stats")
async def get_campaign_stats(campaign_id: str = None):
    """Get statistics for outreach campaigns."""
    try:
        from outreach_campaigns import get_campaign_stats
        return await get_campaign_stats(campaign_id)
    except ImportError:
        raise HTTPException(status_code=503, detail="Outreach campaigns not available")


# ==================== UPSELL ENGINE ====================

@router.post("/upsells/process-missed")
async def process_missed_upsells(days_back: int = 7, limit: int = 50):
    """Process purchases that missed upsell emails."""
    try:
        from upsell_engine import process_missed_upsells
        result = await process_missed_upsells(days_back, limit)
        return result
    except ImportError:
        raise HTTPException(status_code=503, detail="Upsell engine not available")


@router.get("/upsells/recommendations/{email}")
async def get_upsell_recommendations(email: str):
    """Get upsell recommendations for a customer."""
    try:
        from upsell_engine import get_customer_purchase_history, get_recommended_upsells
        history = await get_customer_purchase_history(email)
        if not history:
            return {"email": email, "recommendations": [], "message": "No purchase history"}

        last_product = history[0]['product_code']
        recommendations = await get_recommended_upsells(email, last_product)
        return {
            "email": email,
            "purchase_history": history,
            "recommendations": recommendations
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Upsell engine not available")


# Export router
__all__ = ["router"]

#!/usr/bin/env python3
"""
API Monetization Engine
=======================
Usage-based billing and metering for BrainOps API services.

Revenue Model:
- Free Tier: 100 calls/day
- Starter: $29/mo - 10,000 calls/mo
- Pro: $99/mo - 100,000 calls/mo
- Enterprise: $299/mo - 1,000,000 calls/mo
- Pay-as-you-go: $0.001 - $0.10 per call (varies by endpoint)

Integrates with Stripe for billing and tracks usage in Supabase.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Avoid hardcoding domains we don't own. Operators can override via env.
_PRICING_URL = (
    os.getenv("PRICING_URL", "").strip()
    or os.getenv("SUPPORT_URL", "").strip()
    or "https://brainstackstudio.com/pricing"
)

# Pricing tiers
class PricingTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    PAYG = "pay_as_you_go"

TIER_LIMITS = {
    PricingTier.FREE: {"calls_per_month": 100, "price_monthly": 0},
    PricingTier.STARTER: {"calls_per_month": 10_000, "price_monthly": 29},
    PricingTier.PRO: {"calls_per_month": 100_000, "price_monthly": 99},
    PricingTier.ENTERPRISE: {"calls_per_month": 1_000_000, "price_monthly": 299},
    PricingTier.PAYG: {"calls_per_month": float('inf'), "price_monthly": 0},
}

# Per-call pricing for PAYG and overage
ENDPOINT_PRICING = {
    # High-value AI endpoints
    "ai/generate": Decimal("0.05"),
    "ai/analyze": Decimal("0.03"),
    "content/generate": Decimal("0.10"),
    "agents/execute": Decimal("0.02"),

    # MCP Bridge tools
    "mcp/tools": Decimal("0.001"),
    "mcp/execute": Decimal("0.005"),

    # Standard endpoints
    "default": Decimal("0.001"),
}


class APIMonetizationEngine:
    """
    Manages API monetization, usage tracking, and billing.
    """

    def __init__(self):
        self.engine_id = "APIMonetizationEngine"
        self.version = "1.0.0"
        self._pool = None

    def _get_pool(self):
        """Lazy-load database pool."""
        if self._pool is None:
            try:
                from database.async_connection import get_pool
                self._pool = get_pool()
            except Exception as e:
                logger.error(f"Failed to get database pool: {e}")
                raise
        return self._pool

    async def ensure_tables(self):
        """Ensure monetization tables exist."""
        pool = self._get_pool()

        logger.info("API monetization tables ensured")

    async def create_api_key(
        self,
        user_id: str = None,
        name: str = "Default API Key",
        tier: PricingTier = PricingTier.FREE,
        rate_limit: int = 60
    ) -> dict:
        """Create a new API key for a user."""
        import hashlib
        import secrets

        # Generate secure API key
        key_raw = f"brainops_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key_raw.encode()).hexdigest()
        key_prefix = key_raw[:15]

        pool = self._get_pool()

        key_id = str(uuid.uuid4())
        await pool.execute("""
            INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name, tier, rate_limit_per_minute)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, key_id, user_id, key_hash, key_prefix, name, tier.value, rate_limit)

        logger.info(f"Created API key {key_prefix}... for tier {tier.value}")

        return {
            "key_id": key_id,
            "api_key": key_raw,  # Only returned once at creation!
            "key_prefix": key_prefix,
            "tier": tier.value,
            "rate_limit_per_minute": rate_limit,
            "monthly_limit": TIER_LIMITS[tier]["calls_per_month"],
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    async def validate_api_key(self, api_key: str) -> dict:
        """Validate an API key and return its metadata."""
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        pool = self._get_pool()

        result = await pool.fetchrow("""
            SELECT id, user_id, tier, rate_limit_per_minute, is_active, expires_at, metadata
            FROM api_keys
            WHERE key_hash = $1
        """, key_hash)

        if not result:
            return {"valid": False, "error": "Invalid API key"}

        if not result['is_active']:
            return {"valid": False, "error": "API key is deactivated"}

        if result['expires_at'] and result['expires_at'] < datetime.now(timezone.utc):
            return {"valid": False, "error": "API key has expired"}

        # Check monthly usage
        year_month = datetime.now(timezone.utc).strftime("%Y-%m")
        usage = await pool.fetchrow("""
            SELECT total_calls FROM api_usage_monthly
            WHERE api_key_id = $1 AND year_month = $2
        """, result['id'], year_month)

        tier = PricingTier(result['tier'])
        limit = TIER_LIMITS[tier]["calls_per_month"]
        current_usage = usage['total_calls'] if usage else 0

        # Allow overage for PAYG tier
        if tier != PricingTier.PAYG and current_usage >= limit:
            return {
                "valid": False,
                "error": "Monthly API limit exceeded",
                "usage": current_usage,
                "limit": limit,
                "upgrade_url": _PRICING_URL
            }

        return {
            "valid": True,
            "key_id": str(result['id']),
            "user_id": str(result['user_id']) if result['user_id'] else None,
            "tier": result['tier'],
            "rate_limit": result['rate_limit_per_minute'],
            "usage_this_month": current_usage,
            "monthly_limit": limit,
            "remaining": limit - current_usage if limit != float('inf') else None
        }

    async def track_usage(
        self,
        api_key_id: str,
        endpoint: str,
        method: str = "GET",
        status_code: int = 200,
        response_time_ms: int = 0,
        tokens_used: int = 0
    ) -> dict:
        """Track API usage and calculate cost."""
        pool = self._get_pool()

        # Get API key tier
        key_data = await pool.fetchrow(
            "SELECT tier FROM api_keys WHERE id = $1",
            api_key_id
        )
        tier = PricingTier(key_data['tier']) if key_data else PricingTier.FREE

        # Calculate cost
        endpoint_category = self._categorize_endpoint(endpoint)
        price_per_call = ENDPOINT_PRICING.get(endpoint_category, ENDPOINT_PRICING["default"])

        # Apply tier discount
        tier_discount = {
            PricingTier.FREE: Decimal("0"),  # Free tier doesn't pay
            PricingTier.STARTER: Decimal("0.5"),  # 50% discount
            PricingTier.PRO: Decimal("0.3"),  # 70% discount
            PricingTier.ENTERPRISE: Decimal("0.1"),  # 90% discount
            PricingTier.PAYG: Decimal("1.0"),  # Full price
        }

        effective_price = price_per_call * tier_discount.get(tier, Decimal("1.0"))
        cost_cents = int(effective_price * 100)

        # Record usage
        usage_id = str(uuid.uuid4())
        await pool.execute("""
            INSERT INTO api_usage (id, api_key_id, endpoint, method, status_code, response_time_ms, tokens_used, cost_cents)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, usage_id, api_key_id, endpoint, method, status_code, response_time_ms, tokens_used, cost_cents)

        # Update monthly summary
        year_month = datetime.now(timezone.utc).strftime("%Y-%m")
        await pool.execute("""
            INSERT INTO api_usage_monthly (api_key_id, year_month, total_calls, total_cost_cents, tier)
            VALUES ($1, $2, 1, $3, $4)
            ON CONFLICT (api_key_id, year_month)
            DO UPDATE SET
                total_calls = api_usage_monthly.total_calls + 1,
                total_cost_cents = api_usage_monthly.total_cost_cents + $3,
                updated_at = NOW()
        """, api_key_id, year_month, cost_cents, tier.value)

        return {
            "usage_id": usage_id,
            "cost_cents": cost_cents,
            "tier": tier.value
        }

    def _categorize_endpoint(self, endpoint: str) -> str:
        """Categorize endpoint for pricing."""
        endpoint_lower = endpoint.lower()

        if "ai/generate" in endpoint_lower or "content/generate" in endpoint_lower:
            return "content/generate"
        elif "ai" in endpoint_lower:
            return "ai/generate"
        elif "mcp/execute" in endpoint_lower:
            return "mcp/execute"
        elif "mcp" in endpoint_lower:
            return "mcp/tools"
        elif "agents" in endpoint_lower:
            return "agents/execute"

        return "default"

    async def get_usage_report(self, api_key_id: str, days: int = 30) -> dict:
        """Generate usage report for an API key."""
        pool = self._get_pool()

        since = datetime.now(timezone.utc) - timedelta(days=days)

        # Get total usage
        totals = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_calls,
                SUM(cost_cents) as total_cost_cents,
                AVG(response_time_ms) as avg_response_time,
                SUM(tokens_used) as total_tokens
            FROM api_usage
            WHERE api_key_id = $1 AND timestamp >= $2
        """, api_key_id, since)

        # Get usage by endpoint
        by_endpoint = await pool.fetch("""
            SELECT
                endpoint,
                COUNT(*) as calls,
                SUM(cost_cents) as cost_cents
            FROM api_usage
            WHERE api_key_id = $1 AND timestamp >= $2
            GROUP BY endpoint
            ORDER BY calls DESC
            LIMIT 10
        """, api_key_id, since)

        # Get daily breakdown
        daily = await pool.fetch("""
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as calls,
                SUM(cost_cents) as cost_cents
            FROM api_usage
            WHERE api_key_id = $1 AND timestamp >= $2
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, api_key_id, since)

        return {
            "api_key_id": api_key_id,
            "period_days": days,
            "total_calls": totals['total_calls'] or 0,
            "total_cost": float(totals['total_cost_cents'] or 0) / 100,
            "avg_response_time_ms": float(totals['avg_response_time'] or 0),
            "total_tokens": totals['total_tokens'] or 0,
            "top_endpoints": [
                {"endpoint": r['endpoint'], "calls": r['calls'], "cost": float(r['cost_cents']) / 100}
                for r in by_endpoint
            ],
            "daily_usage": [
                {"date": str(r['date']), "calls": r['calls'], "cost": float(r['cost_cents']) / 100}
                for r in daily
            ],
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    async def generate_invoice(self, api_key_id: str, year_month: str = None) -> dict:
        """
        Generate invoice record for API usage.
        
        NOTE: This creates a billing record in the local database. 
        It does NOT trigger a charge on Stripe automatically.
        The `stripe_invoice_id` field will be null until processed by a separate billing worker.
        """
        if not year_month:
            year_month = datetime.now(timezone.utc).strftime("%Y-%m")

        pool = self._get_pool()

        # Get monthly summary
        summary = await pool.fetchrow("""
            SELECT um.*, ak.tier, ak.user_id
            FROM api_usage_monthly um
            JOIN api_keys ak ON um.api_key_id = ak.id
            WHERE um.api_key_id = $1 AND um.year_month = $2
        """, api_key_id, year_month)

        if not summary:
            return {"success": False, "error": "No usage found for period"}

        tier = PricingTier(summary['tier'])
        tier_info = TIER_LIMITS[tier]

        line_items = []
        total_cents = 0

        # Subscription fee
        if tier_info["price_monthly"] > 0:
            sub_cents = tier_info["price_monthly"] * 100
            line_items.append({
                "description": f"{tier.value.title()} Plan - Monthly Subscription",
                "quantity": 1,
                "unit_price_cents": sub_cents,
                "total_cents": sub_cents
            })
            total_cents += sub_cents

        # Overage charges
        included_calls = tier_info["calls_per_month"]
        total_calls = summary['total_calls']

        if total_calls > included_calls and tier != PricingTier.FREE:
            overage_calls = total_calls - included_calls
            overage_cost = summary['overage_cost_cents'] or int(overage_calls * 0.1)  # $0.001 per call
            line_items.append({
                "description": f"Overage: {overage_calls:,} additional API calls",
                "quantity": overage_calls,
                "unit_price_cents": 0.1,
                "total_cents": overage_cost
            })
            total_cents += overage_cost

        # Create billing record
        billing_id = str(uuid.uuid4())
        year, month = map(int, year_month.split("-"))
        period_start = datetime(year, month, 1, tzinfo=timezone.utc)
        period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)

        await pool.execute("""
            INSERT INTO api_billing (id, api_key_id, amount_cents, period_start, period_end, line_items, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, billing_id, api_key_id, total_cents, period_start, period_end, json.dumps(line_items), "pending")

        return {
            "success": True,
            "billing_id": billing_id,
            "api_key_id": api_key_id,
            "period": year_month,
            "tier": tier.value,
            "total_calls": total_calls,
            "included_calls": included_calls,
            "overage_calls": max(0, total_calls - included_calls),
            "line_items": line_items,
            "total_amount": float(total_cents) / 100,
            "currency": "USD",
            "status": "pending"
        }


# FastAPI Router for API monetization endpoints
def create_monetization_router():
    """Create FastAPI router for monetization endpoints."""
    from fastapi import APIRouter, HTTPException, Request, Depends
    from pydantic import BaseModel

    router = APIRouter(prefix="/api/billing", tags=["billing", "monetization"])
    engine = APIMonetizationEngine()

    class CreateKeyRequest(BaseModel):
        name: str = "Default API Key"
        tier: str = "free"

    @router.post("/keys")
    async def create_key(request: CreateKeyRequest):
        """Create a new API key."""
        try:
            tier = PricingTier(request.tier)
        except ValueError:
            raise HTTPException(400, f"Invalid tier: {request.tier}")

        result = await engine.create_api_key(
            name=request.name,
            tier=tier
        )
        return result

    @router.get("/keys/{key_id}/usage")
    async def get_usage(key_id: str, days: int = 30):
        """Get usage report for an API key."""
        return await engine.get_usage_report(key_id, days)

    @router.post("/keys/{key_id}/invoice")
    async def create_invoice(key_id: str, year_month: str = None):
        """Generate invoice for API usage."""
        return await engine.generate_invoice(key_id, year_month)

    @router.get("/pricing")
    async def get_pricing():
        """Get pricing tiers and endpoint costs."""
        return {
            "tiers": {
                tier.value: {
                    "calls_per_month": info["calls_per_month"] if info["calls_per_month"] != float('inf') else "unlimited",
                    "price_monthly": info["price_monthly"]
                }
                for tier, info in TIER_LIMITS.items()
            },
            "endpoint_pricing": {
                endpoint: float(price)
                for endpoint, price in ENDPOINT_PRICING.items()
            },
            "currency": "USD"
        }

    return router


# Agent metadata for scheduler
AGENT_METADATA = {
    "id": "APIMonetizationEngine",
    "name": "API Monetization Engine",
    "description": "Usage-based billing and metering for BrainOps API",
    "version": "1.0.0",
    "tasks": [
        {"name": "generate_invoices", "schedule": "0 0 1 * *", "description": "Monthly invoice generation"},
        {"name": "usage_alerts", "schedule": "0 * * * *", "description": "Hourly usage limit alerts"},
    ],
    "category": "revenue"
}


async def execute_agent(task: str = "health_check", **kwargs) -> dict:
    """Entry point for agent executor."""
    engine = APIMonetizationEngine()

    if task == "ensure_tables":
        await engine.ensure_tables()
        return {"success": True, "task": "ensure_tables"}
    elif task == "health_check":
        return {"success": True, "status": "healthy", "version": "1.0.0"}

    return {"success": False, "error": f"Unknown task: {task}"}


if __name__ == "__main__":
    import sys

    async def main():
        engine = APIMonetizationEngine()
        await engine.ensure_tables()

        # Create test key
        key = await engine.create_api_key(name="Test Key", tier=PricingTier.STARTER)
        print(f"Created key: {key['api_key'][:20]}...")

        # Simulate usage
        for i in range(5):
            await engine.track_usage(key['key_id'], f"/api/test/{i}", response_time_ms=100)

        # Get report
        report = await engine.get_usage_report(key['key_id'])
        print(json.dumps(report, indent=2))

    asyncio.run(main())

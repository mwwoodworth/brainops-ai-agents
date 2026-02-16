"""
Automated Upsell Engine
=======================
Identifies upsell opportunities from purchase history and triggers outreach.

Logic:
1. Scan recent customers (Gumroad/Stripe).
2. Identify those who haven't upgraded.
3. Use 'OutreachEngine' to draft personalized emails.
4. Queue emails for approval.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import List, Optional

from database.async_connection import get_pool

# We'll use the outreach engine to generate the content
try:
    from outreach_engine import get_outreach_engine

    OUTREACH_AVAILABLE = True
except ImportError:
    OUTREACH_AVAILABLE = False

logger = logging.getLogger(__name__)


class UpsellEngine:
    def __init__(self):
        # Rules map Gumroad product permalink codes to upsell targets
        # Live Gumroad products: HJHMSM($49), VJXCEW($37), XGFKP($29), GSAAVB($97), UPSYKR($149), CAWVO($29)
        self.rules = {
            "prompt_to_kit": {
                "trigger_product": [
                    "XGFKP",
                    "CAWVO",
                ],  # Prompt Pack ($29) / Automation Toolkit ($29)
                "target_product": "HJHMSM",  # MCP Server Starter Kit ($49)
                "delay_days": 3,
                "template": "upsell_starter_kit",
            },
            "kit_to_framework": {
                "trigger_product": ["HJHMSM", "VJXCEW"],  # Starter Kit ($49) / SaaS Scripts ($37)
                "target_product": "GSAAVB",  # AI Orchestration Framework ($97)
                "delay_days": 7,
                "template": "upsell_framework",
            },
            "any_to_bundle": {
                "trigger_product": ["HJHMSM", "VJXCEW", "XGFKP", "GSAAVB", "CAWVO"],
                "target_product": "UPSYKR",  # Command Center UI Kit ($149)
                "delay_days": 5,
                "template": "upsell_premium_bundle",
            },
        }

    async def process_missed_upsells(self, days_back: int = 7, limit: int = 50):
        """
        Scan for customers who purchased a trigger product but not the target product.
        """
        pool = get_pool()
        if not pool:
            logger.error("Database pool unavailable")
            return {"error": "Database unavailable"}

        if not OUTREACH_AVAILABLE:
            logger.error("Outreach engine unavailable")
            return {"error": "Outreach engine unavailable"}

        results = {"processed": 0, "opportunities_found": 0, "drafts_created": 0, "details": []}

        # For each rule, find matching customers
        for rule_name, rule in self.rules.items():
            # Find customers who bought trigger product X days ago
            # Checking unified 'revenue_leads' or specific sales tables?
            # Ideally we use 'gumroad_sales' and 'stripe_events' but 'revenue_leads' should be the source of truth for people.

            # Simplified Logic: Query customers with purchase history
            query = """
                SELECT 
                    rl.id, rl.email, rl.metadata, rl.purchase_history
                FROM revenue_leads rl
                WHERE 
                    -- Has purchased trigger product
                    EXISTS (
                        SELECT 1 FROM jsonb_array_elements(rl.purchase_history) as p 
                        WHERE p->>'product_id' = ANY($1)
                        AND (p->>'date')::timestamp < NOW() - INTERVAL '7 days'
                    )
                    -- Has NOT purchased target product
                    AND NOT EXISTS (
                        SELECT 1 FROM jsonb_array_elements(rl.purchase_history) as p 
                        WHERE p->>'product_id' = $2
                    )
                    -- Has NOT been pitched recently
                    AND (
                        rl.metadata->'last_upsell_pitch' IS NULL 
                        OR (rl.metadata->>'last_upsell_pitch')::timestamp < NOW() - INTERVAL '30 days'
                    )
                LIMIT $3
            """

            # Note: This assumes 'purchase_history' is populated.
            # In a real run, we might need to join with gumroad_sales.
            # For robustness, let's look at the raw sales tables if revenue_leads isn't fully populated.

            # Fallback Query (using gumroad_sales directly)
            # Find emails that bought X but not Y
            fallback_query = """
                SELECT DISTINCT s1.email
                FROM gumroad_sales s1
                LEFT JOIN gumroad_sales s2 ON s1.email = s2.email AND s2.product_code = $2
                WHERE s1.product_code = ANY($1)
                AND s1.sale_timestamp < NOW() - INTERVAL '3 days' -- Using rule delay
                AND s2.email IS NULL
                LIMIT $3
            """

            # Execution
            try:
                # We'll use the fallback logic for now as it's safer
                rows = await pool.fetch(
                    fallback_query, rule["trigger_product"], rule["target_product"], limit
                )

                outreach = get_outreach_engine()

                for row in rows:
                    email = row["email"]

                    # Check if we have a lead for this email
                    lead = await pool.fetchrow(
                        "SELECT id FROM revenue_leads WHERE email = $1", email
                    )
                    lead_id = str(lead["id"]) if lead else None

                    # If lead doesn't exist, create it?
                    # Yes, any paying customer is a lead.
                    if not lead_id:
                        # Skip for now to keep it simple, or log it
                        continue

                    # Generate Draft
                    success, _, draft = await outreach.generate_outreach_draft(
                        lead_id=lead_id, template_name=rule["template"]
                    )

                    if success:
                        results["drafts_created"] += 1
                        results["details"].append(
                            {
                                "email": email,
                                "rule": rule_name,
                                "draft_id": str(draft.id) if draft else "unknown",
                            }
                        )

                        # Update lead metadata to prevent spam
                        await pool.execute(
                            """
                            UPDATE revenue_leads 
                            SET metadata = jsonb_set(COALESCE(metadata, '{}'), '{last_upsell_pitch}', to_jsonb(NOW()))
                            WHERE id = $1
                        """,
                            lead_id,
                        )

            except Exception as e:
                logger.error(f"Error processing rule {rule_name}: {e}")

        return results

    async def get_customer_purchase_history(self, email: str) -> List[dict]:
        """Get consolidated purchase history for a customer"""
        pool = get_pool()
        if not pool:
            return []

        # Gumroad
        gumroad = await pool.fetch(
            """
            SELECT product_name as product, price, sale_timestamp as date, 'gumroad' as source, product_code
            FROM gumroad_sales WHERE email = $1
        """,
            email,
        )

        # Stripe
        stripe = await pool.fetch(
            """
            SELECT 'Stripe Product' as product, amount_cents/100.0 as price, created_at as date, 'stripe' as source, 'stripe_charge' as product_code
            FROM stripe_events WHERE customer_email = $1 AND event_type = 'charge.succeeded'
        """,
            email,
        )

        history = [dict(r) for r in gumroad] + [dict(r) for r in stripe]
        history.sort(key=lambda x: x["date"], reverse=True)
        return history

    async def get_recommended_upsells(self, email: str, current_product: str) -> List[dict]:
        """Recommend upsells based on current holdings"""
        recommendations = []
        for rule_name, rule in self.rules.items():
            if current_product in rule["trigger_product"]:
                recommendations.append(
                    {
                        "product": rule["target_product"],
                        "reason": f"Upgrade from {current_product}",
                        "rule": rule_name,
                    }
                )
        return recommendations


# Singleton
_upsell_engine = None


def get_upsell_engine():
    global _upsell_engine
    if _upsell_engine is None:
        _upsell_engine = UpsellEngine()
    return _upsell_engine


# Module-level convenience functions (used by gumroad_webhook.py and revenue_complete.py)
async def process_purchase_for_upsell(email: str, product_code: str, **kwargs):
    """Process a new purchase to trigger upsell opportunities."""
    engine = get_upsell_engine()
    try:
        recommendations = await engine.get_recommended_upsells(email, product_code)
        if recommendations:
            logger.info(f"Found {len(recommendations)} upsell opportunities for {email}")
        return {"email": email, "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error processing purchase for upsell: {e}")
        return {"error": str(e)}


async def process_missed_upsells(days_back: int = 7, limit: int = 50):
    """Module-level wrapper for UpsellEngine.process_missed_upsells."""
    engine = get_upsell_engine()
    return await engine.process_missed_upsells(days_back=days_back, limit=limit)


async def get_customer_purchase_history(email: str):
    """Module-level wrapper for UpsellEngine.get_customer_purchase_history."""
    engine = get_upsell_engine()
    return await engine.get_customer_purchase_history(email)


async def get_recommended_upsells(email: str, current_product: str):
    """Module-level wrapper for UpsellEngine.get_recommended_upsells."""
    engine = get_upsell_engine()
    return await engine.get_recommended_upsells(email, current_product)

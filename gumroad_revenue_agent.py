"""
Gumroad Revenue Agent
=====================
Fully autonomous Gumroad product management and revenue tracking.
Integrates with existing BrainOps infrastructure for E2E automation.

Capabilities:
- Daily sales sync from Gumroad API
- Revenue analytics and reporting
- Automatic webhook registration
- Product publishing automation
- Conversion tracking
- AI-powered product optimization recommendations

This agent is designed to be scheduled via agent_scheduler.py for autonomous operation.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Configuration
GUMROAD_ACCESS_TOKEN = os.getenv("GUMROAD_ACCESS_TOKEN", "")
BRAINOPS_WEBHOOK_URL = "https://brainops-ai-agents.onrender.com/gumroad/webhook"


class GumroadRevenueAgent:
    """
    Autonomous Gumroad revenue management agent.
    Handles product publishing, sales tracking, and revenue analytics.
    """

    def __init__(self, access_token: str = None):
        self.access_token = access_token or GUMROAD_ACCESS_TOKEN
        self.base_url = "https://api.gumroad.com/v2"
        self.agent_id = "GumroadRevenueAgent"
        self.version = "1.0.0"

    async def execute(self, task: str = "daily_sync", **kwargs) -> dict:
        """
        Main execution entry point for scheduled agent.

        Tasks:
        - daily_sync: Sync sales and update analytics
        - register_webhook: Register webhook with Gumroad
        - publish_products: Publish products from JSON config
        - revenue_report: Generate revenue analytics report
        - optimize_products: AI recommendations for product optimization
        """
        logger.info(f"GumroadRevenueAgent executing task: {task}")

        try:
            if task == "daily_sync":
                return await self.daily_sales_sync()
            elif task == "register_webhook":
                return await self.register_webhook()
            elif task == "publish_products":
                return await self.publish_products_from_config()
            elif task == "revenue_report":
                return await self.generate_revenue_report(days=kwargs.get("days", 30))
            elif task == "optimize_products":
                return await self.generate_optimization_recommendations()
            elif task == "health_check":
                return await self.health_check()
            else:
                return {"success": False, "error": f"Unknown task: {task}"}
        except Exception as e:
            logger.exception(f"GumroadRevenueAgent error on task {task}")
            return {"success": False, "error": str(e)}

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        data: dict = None,
        params: dict = None
    ) -> dict:
        """Make authenticated request to Gumroad API using Bearer token."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.base_url}{endpoint}"

            # Use Bearer token authentication (required by Gumroad API)
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            if params is None:
                params = {}

            if method.upper() == "GET":
                response = await client.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(url, data=data or {}, headers=headers)
            elif method.upper() == "PUT":
                response = await client.put(url, data=data or {}, headers=headers)
            elif method.upper() == "DELETE":
                response = await client.delete(url, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Gumroad API error: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text, "status_code": response.status_code}

    async def health_check(self) -> dict:
        """Verify Gumroad API connectivity."""
        result = await self._api_request("GET", "/user")
        if result.get("success"):
            user = result.get("user", {})
            return {
                "success": True,
                "connected": True,
                "email": user.get("email"),
                "name": user.get("name"),
                "gumroad_id": user.get("user_id"),
                "timestamp": datetime.utcnow().isoformat()
            }
        return {
            "success": False,
            "connected": False,
            "error": result.get("error", "Unknown error"),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def register_webhook(self) -> dict:
        """Register webhook subscription with Gumroad."""
        results = {}

        webhook_events = ["sale", "refund", "cancellation"]

        for event in webhook_events:
            try:
                result = await self._api_request(
                    "PUT",
                    "/resource_subscriptions",
                    data={
                        "resource_name": event,
                        "post_url": BRAINOPS_WEBHOOK_URL
                    }
                )
                results[event] = {
                    "success": result.get("success", False),
                    "subscription": result.get("resource_subscription", {}),
                    "webhook_url": BRAINOPS_WEBHOOK_URL
                }
                logger.info(f"Registered webhook for {event}: {result.get('success')}")
            except Exception as e:
                results[event] = {"success": False, "error": str(e)}
                logger.error(f"Failed to register webhook for {event}: {e}")

        return {
            "success": all(r.get("success") for r in results.values()),
            "webhooks": results,
            "webhook_url": BRAINOPS_WEBHOOK_URL,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def list_products(self) -> list[dict]:
        """Get all products from Gumroad."""
        result = await self._api_request("GET", "/products")
        if result.get("success"):
            return result.get("products", [])
        return []

    async def get_sales(
        self,
        after: datetime = None,
        before: datetime = None,
        product_id: str = None
    ) -> list[dict]:
        """Get sales from Gumroad API."""
        params = {}
        if after:
            params["after"] = after.strftime("%Y-%m-%d")
        if before:
            params["before"] = before.strftime("%Y-%m-%d")
        if product_id:
            params["product_id"] = product_id

        all_sales = []
        page = 1

        while True:
            params["page"] = page
            result = await self._api_request("GET", "/sales", params=params)

            if not result.get("success"):
                break

            sales = result.get("sales", [])
            if not sales:
                break

            all_sales.extend(sales)
            page += 1

            # Safety limit
            if page > 100:
                break

        return all_sales

    async def daily_sales_sync(self) -> dict:
        """
        Sync today's sales from Gumroad to local database.
        Records in gumroad_sales table for revenue tracking.
        """
        try:
            from database.async_connection import get_pool
            pool = get_pool()
        except Exception as e:
            logger.warning(f"Database pool unavailable: {e}")
            pool = None

        # Get sales from last 2 days to catch any missed
        after = datetime.utcnow() - timedelta(days=2)
        sales = await self.get_sales(after=after)

        synced_count = 0
        total_revenue = Decimal("0")

        for sale in sales:
            try:
                sale_id = sale.get("id")
                price_cents = sale.get("price", 0)
                price = Decimal(str(price_cents)) / 100

                if pool:
                    await pool.execute("""
                        INSERT INTO gumroad_sales (
                            sale_id, email, customer_name, product_code,
                            product_name, price, currency, sale_timestamp,
                            metadata, is_test
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (sale_id) DO UPDATE SET
                            updated_at = NOW()
                    """,
                        sale_id,
                        sale.get("email", ""),
                        sale.get("full_name", ""),
                        sale.get("product_permalink", "").upper(),
                        sale.get("product_name", ""),
                        price,
                        sale.get("currency", "USD"),
                        datetime.fromisoformat(sale.get("created_at", datetime.utcnow().isoformat()).replace("Z", "+00:00")),
                        json.dumps(sale),
                        sale.get("test", False)
                    )
                    synced_count += 1
                    total_revenue += price

            except Exception as e:
                logger.error(f"Failed to sync sale {sale.get('id')}: {e}")

        return {
            "success": True,
            "task": "daily_sync",
            "sales_synced": synced_count,
            "total_revenue": float(total_revenue),
            "period_start": after.isoformat(),
            "period_end": datetime.utcnow().isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def generate_revenue_report(self, days: int = 30) -> dict:
        """Generate comprehensive revenue analytics report."""
        after = datetime.utcnow() - timedelta(days=days)
        sales = await self.get_sales(after=after)

        if not sales:
            return {
                "success": True,
                "period_days": days,
                "total_sales": 0,
                "total_revenue": 0,
                "products": {},
                "timestamp": datetime.utcnow().isoformat()
            }

        # Calculate metrics
        total_revenue = sum(s.get("price", 0) for s in sales) / 100
        product_breakdown = {}
        daily_revenue = {}

        for sale in sales:
            # Product breakdown
            product = sale.get("product_name", "Unknown")
            if product not in product_breakdown:
                product_breakdown[product] = {"count": 0, "revenue": 0}
            product_breakdown[product]["count"] += 1
            product_breakdown[product]["revenue"] += sale.get("price", 0) / 100

            # Daily breakdown
            sale_date = sale.get("created_at", "")[:10]
            if sale_date not in daily_revenue:
                daily_revenue[sale_date] = 0
            daily_revenue[sale_date] += sale.get("price", 0) / 100

        # Sort products by revenue
        top_products = sorted(
            product_breakdown.items(),
            key=lambda x: x[1]["revenue"],
            reverse=True
        )[:10]

        return {
            "success": True,
            "period_days": days,
            "total_sales": len(sales),
            "total_revenue": total_revenue,
            "average_order_value": total_revenue / len(sales) if sales else 0,
            "top_products": dict(top_products),
            "daily_revenue": daily_revenue,
            "refund_count": sum(1 for s in sales if s.get("refunded")),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def publish_products_from_config(
        self,
        config_path: str = None
    ) -> dict:
        """
        Publish products from JSON config to Gumroad.
        Creates new products or updates existing ones.
        """
        import json
        from pathlib import Path

        # Determine config path: argument -> env var -> default relative path
        if not config_path:
            config_path = os.getenv("GUMROAD_PRODUCT_CONFIG", "gumroad-products-import.json")

        config_file = Path(config_path)
        if not config_file.is_absolute():
             # relative to current working directory
             config_file = Path(os.getcwd()) / config_path

        if not config_file.exists():
            logger.warning(f"Product config not found at: {config_file}")
            return {"success": False, "error": f"Config not found: {config_file}"}

        try:
            with open(config_file) as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON in config: {e}"}

        products = config.get("products", [])
        results = []

        # Get existing products
        existing = await self.list_products()
        existing_permalinks = {}
        for product in existing:
            for key in ("custom_permalink", "permalink", "url", "short_url"):
                value = product.get(key)
                if value:
                    existing_permalinks[value] = product.get("id")

        for product_spec in products:
            try:
                permalink = product_spec.get("permalink", "")
                description = product_spec.get("description", "") or ""
                description_file = product_spec.get("description_file")
                if description_file:
                    desc_path = Path(description_file)
                    if not desc_path.is_absolute():
                        desc_path = config_file.parent / desc_path
                    if desc_path.exists():
                        description = desc_path.read_text()
                    else:
                        logger.warning(f"Description file missing: {desc_path}")

                # Check if product exists
                if permalink in existing_permalinks:
                    # Update existing
                    product_id = existing_permalinks[permalink]
                    result = await self._api_request(
                        "PUT",
                        f"/products/{product_id}",
                        data={
                            "name": product_spec.get("name"),
                            "price": product_spec.get("price"),
                            "description": description,
                        }
                    )
                    results.append({
                        "sku": product_spec.get("sku"),
                        "action": "updated",
                        "success": result.get("success", False)
                    })
                else:
                    # Create new
                    result = await self._api_request(
                        "POST",
                        "/products",
                        data={
                            "name": product_spec.get("name"),
                            "price": product_spec.get("price"),
                            "url": product_spec.get("permalink"),
                            "description": description,
                        }
                    )
                    results.append({
                        "sku": product_spec.get("sku"),
                        "action": "created",
                        "success": result.get("success", False),
                        "product_id": result.get("product", {}).get("id")
                    })

            except Exception as e:
                results.append({
                    "sku": product_spec.get("sku"),
                    "action": "error",
                    "success": False,
                    "error": str(e)
                })

        return {
            "success": all(r.get("success") for r in results),
            "products_processed": len(results),
            "created": sum(1 for r in results if r.get("action") == "created" and r.get("success")),
            "updated": sum(1 for r in results if r.get("action") == "updated" and r.get("success")),
            "errors": sum(1 for r in results if not r.get("success")),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def generate_optimization_recommendations(self) -> dict:
        """
        AI-powered product optimization recommendations.
        Analyzes sales data to suggest improvements.
        """
        # Get recent sales data
        report = await self.generate_revenue_report(days=30)
        products = await self.list_products()

        recommendations = []

        # Analyze each product
        for product in products:
            product_name = product.get("name", "")
            product_sales = report.get("top_products", {}).get(product_name, {})

            if product_sales.get("count", 0) == 0:
                recommendations.append({
                    "product": product_name,
                    "type": "no_sales",
                    "recommendation": "Consider updating product description, price point, or marketing.",
                    "priority": "high"
                })
            elif product_sales.get("count", 0) < 3:
                recommendations.append({
                    "product": product_name,
                    "type": "low_sales",
                    "recommendation": "Test different pricing strategies or bundle with other products.",
                    "priority": "medium"
                })

        # Price optimization suggestions
        avg_order = report.get("average_order_value", 0)
        if avg_order > 0 and avg_order < 100:
            recommendations.append({
                "type": "pricing",
                "recommendation": f"Average order value is ${avg_order:.2f}. Consider creating higher-tier bundles.",
                "priority": "medium"
            })

        return {
            "success": True,
            "recommendations": recommendations,
            "metrics_summary": {
                "total_sales": report.get("total_sales", 0),
                "total_revenue": report.get("total_revenue", 0),
                "avg_order_value": report.get("average_order_value", 0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# Agent registration for scheduler
AGENT_METADATA = {
    "id": "GumroadRevenueAgent",
    "name": "Gumroad Revenue Agent",
    "description": "Autonomous Gumroad product management and revenue tracking",
    "version": "1.0.0",
    "tasks": [
        {"name": "daily_sync", "schedule": "0 6 * * *", "description": "Daily sales sync"},
        {"name": "revenue_report", "schedule": "0 8 * * 1", "description": "Weekly revenue report"},
        {"name": "health_check", "schedule": "*/30 * * * *", "description": "API health check"},
    ],
    "dependencies": ["gumroad_webhook"],
    "category": "revenue"
}


async def execute_agent(task: str = "daily_sync", **kwargs) -> dict:
    """Entry point for agent executor."""
    agent = GumroadRevenueAgent()
    return await agent.execute(task, **kwargs)


# CLI support
if __name__ == "__main__":
    import sys

    task = sys.argv[1] if len(sys.argv) > 1 else "health_check"

    async def main():
        agent = GumroadRevenueAgent()
        result = await agent.execute(task)
        print(json.dumps(result, indent=2, default=str))

    asyncio.run(main())

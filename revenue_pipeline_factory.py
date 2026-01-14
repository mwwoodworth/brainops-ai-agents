#!/usr/bin/env python3
"""
Revenue Pipeline Factory
========================
Central orchestrator for all automated revenue streams.

Manages and coordinates:
1. API Monetization - Usage-based billing for API access
2. Product Generation - AI-powered digital product creation
3. Lead Discovery - Automated lead finding and nurturing
4. Content Marketing - SEO content generation
5. Gumroad Sales - Digital product sales tracking
6. Agent-as-a-Service - Selling access to AI agents

This is the master revenue controller that ensures all pipelines
are running, healthy, and optimized for maximum revenue generation.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class RevenueStream(str, Enum):
    API_MONETIZATION = "api_monetization"
    PRODUCT_SALES = "product_sales"
    LEAD_NURTURING = "lead_nurturing"
    CONTENT_MARKETING = "content_marketing"
    AGENT_SERVICES = "agent_services"
    SUBSCRIPTIONS = "subscriptions"


# Pipeline configurations
PIPELINE_CONFIG = {
    RevenueStream.API_MONETIZATION: {
        "name": "API Monetization Engine",
        "module": "api_monetization_engine",
        "agent_class": "APIMonetizationEngine",
        "revenue_model": "usage_based",
        "projected_mrr": 500,
        "schedule": "continuous",
        "dependencies": ["stripe", "database"]
    },
    RevenueStream.PRODUCT_SALES: {
        "name": "Digital Product Sales",
        "module": "gumroad_revenue_agent",
        "agent_class": "GumroadRevenueAgent",
        "revenue_model": "one_time",
        "projected_mrr": 300,
        "schedule": "0 6 * * *",
        "dependencies": ["gumroad", "database"]
    },
    RevenueStream.LEAD_NURTURING: {
        "name": "Lead Discovery & Nurturing",
        "module": "revenue_pipeline_agents",
        "agent_class": "LeadDiscoveryAgentReal",
        "revenue_model": "conversion",
        "projected_mrr": 1000,
        "schedule": "0 8 * * *",
        "dependencies": ["database", "email"]
    },
    RevenueStream.CONTENT_MARKETING: {
        "name": "SEO Content Factory",
        "module": "content_generation_agent",
        "agent_class": "ContentGeneratorAgent",
        "revenue_model": "indirect",
        "projected_mrr": 200,
        "schedule": "0 10 * * 1,4",
        "dependencies": ["ai_core", "database"]
    },
    RevenueStream.AGENT_SERVICES: {
        "name": "Agent-as-a-Service",
        "module": "agent_executor",
        "revenue_model": "subscription",
        "projected_mrr": 800,
        "schedule": "continuous",
        "dependencies": ["agents", "database", "stripe"]
    },
    RevenueStream.SUBSCRIPTIONS: {
        "name": "SaaS Subscriptions",
        "module": "mrg_subscription_handler",
        "revenue_model": "recurring",
        "projected_mrr": 400,
        "schedule": "continuous",
        "dependencies": ["stripe", "database"]
    }
}


class RevenuePipelineFactory:
    """
    Master orchestrator for all revenue generation pipelines.

    Responsibilities:
    1. Initialize and manage all revenue streams
    2. Monitor pipeline health and performance
    3. Generate revenue reports and analytics
    4. Optimize pipeline configurations
    5. Handle failures and recovery
    """

    def __init__(self):
        self.factory_id = str(uuid.uuid4())
        self.version = "1.0.0"
        self.pipelines = {}
        self.metrics = {}
        self._pool = None

    async def _get_pool(self):
        """Lazy-load database pool."""
        if self._pool is None:
            try:
                from database.async_connection import get_pool
                self._pool = get_pool()
            except Exception as e:
                logger.error(f"Failed to get database pool: {e}")
        return self._pool

    async def initialize(self):
        """Initialize all revenue pipelines."""
        logger.info("Initializing Revenue Pipeline Factory...")

        await self._ensure_tables()

        for stream, config in PIPELINE_CONFIG.items():
            try:
                self.pipelines[stream] = {
                    "config": config,
                    "status": PipelineStatus.ACTIVE,
                    "last_run": None,
                    "revenue_today": Decimal("0"),
                    "revenue_mtd": Decimal("0"),
                    "errors": []
                }
                logger.info(f"Initialized pipeline: {stream.value}")
            except Exception as e:
                logger.error(f"Failed to initialize {stream.value}: {e}")
                self.pipelines[stream] = {
                    "config": config,
                    "status": PipelineStatus.ERROR,
                    "error": str(e)
                }

        return {
            "success": True,
            "pipelines_initialized": len(self.pipelines),
            "active": sum(1 for p in self.pipelines.values() if p.get("status") == PipelineStatus.ACTIVE)
        }

    async def _ensure_tables(self):
        """Ensure revenue tracking tables exist."""
        pool = await self._get_pool()
        if not pool:
            return

        await pool.execute("""
            -- Pipeline execution tracking
            CREATE TABLE IF NOT EXISTS revenue_pipeline_runs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pipeline_name VARCHAR(100) NOT NULL,
                stream_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) DEFAULT 'running',
                started_at TIMESTAMPTZ DEFAULT NOW(),
                completed_at TIMESTAMPTZ,
                revenue_generated DECIMAL(10,2) DEFAULT 0,
                leads_generated INT DEFAULT 0,
                errors JSONB DEFAULT '[]',
                metadata JSONB DEFAULT '{}'
            );

            -- Revenue tracking by source
            CREATE TABLE IF NOT EXISTS revenue_by_source (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source VARCHAR(100) NOT NULL,
                stream_type VARCHAR(50) NOT NULL,
                amount_cents BIGINT NOT NULL,
                currency VARCHAR(3) DEFAULT 'usd',
                customer_email VARCHAR(255),
                product_id VARCHAR(255),
                recorded_at TIMESTAMPTZ DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'
            );

            -- Daily revenue summary by pipeline stream
            CREATE TABLE IF NOT EXISTS revenue_pipeline_daily (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                date DATE NOT NULL,
                stream_type VARCHAR(50) NOT NULL,
                total_cents BIGINT DEFAULT 0,
                transaction_count INT DEFAULT 0,
                new_customers INT DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(date, stream_type)
            );

            -- Pipeline health metrics
            CREATE TABLE IF NOT EXISTS pipeline_health (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pipeline_name VARCHAR(100) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DECIMAL(15,4),
                recorded_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_pipeline_runs_stream ON revenue_pipeline_runs(stream_type, started_at);
            CREATE INDEX IF NOT EXISTS idx_revenue_source_stream ON revenue_by_source(stream_type, recorded_at);
            CREATE INDEX IF NOT EXISTS idx_revenue_daily_date ON revenue_pipeline_daily(date, stream_type);
        """)

        logger.info("Revenue tracking tables ensured")

    async def execute(self, task: dict) -> dict:
        """Execute factory task."""
        action = task.get("action", "status")

        if action == "status":
            return await self.get_status()
        elif action == "run_pipeline":
            stream = RevenueStream(task.get("stream"))
            return await self.run_pipeline(stream)
        elif action == "run_all":
            return await self.run_all_pipelines()
        elif action == "revenue_report":
            return await self.generate_revenue_report(task.get("days", 30))
        elif action == "optimize":
            return await self.optimize_pipelines()
        elif action == "health_check":
            return await self.health_check()

        return {"success": False, "error": f"Unknown action: {action}"}

    async def get_status(self) -> dict:
        """Get status of all revenue pipelines."""
        status = {
            "factory_id": self.factory_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipelines": {}
        }

        for stream, pipeline in self.pipelines.items():
            # Handle status as either enum or string for robustness
            status_val = pipeline.get("status", PipelineStatus.ACTIVE)
            status_str = status_val.value if isinstance(status_val, PipelineStatus) else str(status_val)

            status["pipelines"][stream.value] = {
                "name": pipeline["config"]["name"],
                "status": status_str,
                "revenue_model": pipeline["config"]["revenue_model"],
                "projected_mrr": pipeline["config"]["projected_mrr"],
                "last_run": pipeline.get("last_run"),
                "errors_count": len(pipeline.get("errors", []))
            }

        # Calculate totals
        status["total_projected_mrr"] = sum(
            p["config"]["projected_mrr"] for p in self.pipelines.values()
        )
        status["active_pipelines"] = sum(
            1 for p in self.pipelines.values()
            if p.get("status") == PipelineStatus.ACTIVE
        )

        return status

    async def run_pipeline(self, stream: RevenueStream) -> dict:
        """Run a specific revenue pipeline."""
        if stream not in self.pipelines:
            return {"success": False, "error": f"Unknown stream: {stream.value}"}

        pipeline = self.pipelines[stream]
        config = pipeline["config"]

        logger.info(f"Running pipeline: {config['name']}")

        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        # Record run start
        pool = await self._get_pool()
        if pool:
            await pool.execute("""
                INSERT INTO revenue_pipeline_runs (id, pipeline_name, stream_type, status)
                VALUES ($1, $2, $3, 'running')
            """, run_id, config["name"], stream.value)

        try:
            # Import and execute the appropriate agent
            result = await self._execute_pipeline_agent(stream, config)

            # Update pipeline state
            pipeline["last_run"] = started_at.isoformat()
            pipeline["status"] = PipelineStatus.ACTIVE

            # Record completion
            if pool:
                await pool.execute("""
                    UPDATE revenue_pipeline_runs
                    SET status = 'completed',
                        completed_at = NOW(),
                        revenue_generated = $1,
                        leads_generated = $2,
                        metadata = $3
                    WHERE id = $4
                """,
                    result.get("revenue_generated", 0),
                    result.get("leads_generated", 0),
                    json.dumps(result),
                    run_id
                )

            return {
                "success": True,
                "run_id": run_id,
                "pipeline": config["name"],
                "stream": stream.value,
                "result": result,
                "duration_ms": int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
            }

        except Exception as e:
            logger.error(f"Pipeline {stream.value} failed: {e}")

            pipeline["status"] = PipelineStatus.ERROR
            pipeline["errors"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            })

            if pool:
                await pool.execute("""
                    UPDATE revenue_pipeline_runs
                    SET status = 'error',
                        completed_at = NOW(),
                        errors = $1
                    WHERE id = $2
                """, json.dumps([{"error": str(e)}]), run_id)

            return {"success": False, "error": str(e), "run_id": run_id}

    async def _execute_pipeline_agent(self, stream: RevenueStream, config: dict) -> dict:
        """Execute the agent for a specific pipeline."""
        module_name = config.get("module")
        agent_class = config.get("agent_class")

        if stream == RevenueStream.API_MONETIZATION:
            from api_monetization_engine import APIMonetizationEngine
            engine = APIMonetizationEngine()
            await engine.ensure_tables()
            return {"status": "tables_ensured", "revenue_generated": 0}

        elif stream == RevenueStream.PRODUCT_SALES:
            from gumroad_revenue_agent import GumroadRevenueAgent
            agent = GumroadRevenueAgent()
            return await agent.execute("daily_sync")

        elif stream == RevenueStream.LEAD_NURTURING:
            from revenue_pipeline_agents import LeadDiscoveryAgentReal, NurtureExecutorAgentReal

            # Run lead discovery
            discovery = LeadDiscoveryAgentReal()
            leads_result = await discovery.execute({"action": "discover_all"})

            # Run nurture sequences
            nurture = NurtureExecutorAgentReal()
            nurture_result = await nurture.execute({"action": "nurture_new_leads"})

            return {
                "leads_discovered": leads_result.get("leads_discovered", 0),
                "leads_stored": leads_result.get("leads_stored", 0),
                "sequences_created": nurture_result.get("sequences_created", 0),
                "emails_queued": nurture_result.get("emails_queued", 0),
                "leads_generated": leads_result.get("leads_discovered", 0)
            }

        elif stream == RevenueStream.CONTENT_MARKETING:
            from content_generation_agent import ContentGeneratorAgent
            agent = ContentGeneratorAgent()
            return await agent.execute({"action": "generate_blog"})

        elif stream == RevenueStream.AGENT_SERVICES:
            # Track agent usage for billing
            pool = await self._get_pool()
            if pool:
                # Count recent agent executions
                result = await pool.fetchrow("""
                    SELECT COUNT(*) as executions,
                           COUNT(DISTINCT agent_type) as unique_agents
                    FROM agent_executions
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                return {
                    "executions_24h": result["executions"] or 0,
                    "unique_agents": result["unique_agents"] or 0,
                    "status": "healthy"
                }
            return {"status": "no_database"}

        elif stream == RevenueStream.SUBSCRIPTIONS:
            # Check subscription status
            pool = await self._get_pool()
            if pool:
                result = await pool.fetchrow("""
                    SELECT
                        COUNT(*) as total_subscriptions,
                        SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                        SUM(CASE WHEN status = 'active' THEN amount ELSE 0 END) as mrr
                    FROM mrg_subscriptions
                """)
                return {
                    "total_subscriptions": result["total_subscriptions"] or 0,
                    "active_subscriptions": result["active"] or 0,
                    "mrr": float(result["mrr"] or 0),
                    "revenue_generated": float(result["mrr"] or 0)
                }
            return {"status": "no_database"}

        return {"status": "unknown_pipeline"}

    async def run_all_pipelines(self) -> dict:
        """Run all active revenue pipelines."""
        results = {}
        total_revenue = Decimal("0")
        total_leads = 0

        for stream in RevenueStream:
            if stream in self.pipelines and self.pipelines[stream].get("status") == PipelineStatus.ACTIVE:
                result = await self.run_pipeline(stream)
                results[stream.value] = result

                if result.get("success"):
                    pipeline_result = result.get("result", {})
                    total_revenue += Decimal(str(pipeline_result.get("revenue_generated", 0)))
                    total_leads += pipeline_result.get("leads_generated", 0)

        return {
            "success": all(r.get("success") for r in results.values()),
            "pipelines_run": len(results),
            "total_revenue": float(total_revenue),
            "total_leads": total_leads,
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def generate_revenue_report(self, days: int = 30) -> dict:
        """Generate comprehensive revenue report."""
        pool = await self._get_pool()
        if not pool:
            return {"success": False, "error": "Database unavailable"}

        since = datetime.now(timezone.utc) - timedelta(days=days)

        # Get revenue by stream
        by_stream = await pool.fetch("""
            SELECT
                stream_type,
                SUM(amount_cents) as total_cents,
                COUNT(*) as transactions
            FROM revenue_by_source
            WHERE recorded_at >= $1
            GROUP BY stream_type
        """, since)

        # Get Gumroad sales
        gumroad_sales = await pool.fetchrow("""
            SELECT
                COUNT(*) as count,
                COALESCE(SUM(price), 0) as total
            FROM gumroad_sales
            WHERE sale_timestamp >= $1
              AND is_test = false
        """, since)

        # Get Stripe revenue
        stripe_revenue = await pool.fetchrow("""
            SELECT
                COUNT(*) as count,
                COALESCE(SUM(amount_cents), 0) as total_cents
            FROM stripe_events
            WHERE created_at >= $1
              AND event_type IN ('charge.succeeded', 'checkout.session.completed')
        """, since)

        # Get lead pipeline
        leads = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_leads,
                SUM(CASE WHEN stage = 'won' THEN 1 ELSE 0 END) as won,
                SUM(CASE WHEN stage = 'won' THEN value_estimate ELSE 0 END) as won_value
            FROM revenue_leads
            WHERE created_at >= $1
        """, since)

        # Pipeline run stats
        pipeline_runs = await pool.fetch("""
            SELECT
                stream_type,
                COUNT(*) as runs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds
            FROM revenue_pipeline_runs
            WHERE started_at >= $1
            GROUP BY stream_type
        """, since)

        # Calculate totals
        total_revenue = Decimal("0")
        if gumroad_sales:
            total_revenue += Decimal(str(gumroad_sales["total"] or 0))
        if stripe_revenue:
            total_revenue += Decimal(str(stripe_revenue["total_cents"] or 0)) / 100

        return {
            "success": True,
            "period_days": days,
            "period_start": since.isoformat(),
            "period_end": datetime.now(timezone.utc).isoformat(),
            "total_revenue": float(total_revenue),
            "revenue_breakdown": {
                "gumroad": {
                    "transactions": gumroad_sales["count"] if gumroad_sales else 0,
                    "revenue": float(gumroad_sales["total"] or 0) if gumroad_sales else 0
                },
                "stripe": {
                    "transactions": stripe_revenue["count"] if stripe_revenue else 0,
                    "revenue": float(stripe_revenue["total_cents"] or 0) / 100 if stripe_revenue else 0
                }
            },
            "lead_pipeline": {
                "total_leads": leads["total_leads"] if leads else 0,
                "won_deals": leads["won"] if leads else 0,
                "won_value": float(leads["won_value"] or 0) if leads else 0
            },
            "pipeline_performance": [
                {
                    "stream": r["stream_type"],
                    "runs": r["runs"],
                    "success_rate": r["successful"] / r["runs"] if r["runs"] > 0 else 0,
                    "avg_duration_seconds": float(r["avg_duration_seconds"] or 0)
                }
                for r in pipeline_runs
            ],
            "projections": {
                "mrr": sum(p["config"]["projected_mrr"] for p in self.pipelines.values()),
                "arr": sum(p["config"]["projected_mrr"] for p in self.pipelines.values()) * 12
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    async def health_check(self) -> dict:
        """Check health of all revenue pipelines."""
        health = {
            "status": "healthy",
            "pipelines": {},
            "issues": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        for stream, pipeline in self.pipelines.items():
            # Handle status as either enum or string for robustness
            status_val = pipeline.get("status", PipelineStatus.ACTIVE)
            status_str = status_val.value if isinstance(status_val, PipelineStatus) else str(status_val)

            pipeline_health = {
                "status": status_str,
                "last_run": pipeline.get("last_run"),
                "error_count": len(pipeline.get("errors", []))
            }

            # Check for issues
            is_error = status_val == PipelineStatus.ERROR or status_str == "error"
            if is_error:
                health["issues"].append(f"{stream.value}: Pipeline in error state")
                health["status"] = "degraded"

            if pipeline.get("errors"):
                recent_errors = [
                    e for e in pipeline["errors"]
                    if datetime.fromisoformat(e["timestamp"]) > datetime.now(timezone.utc) - timedelta(hours=24)
                ]
                if len(recent_errors) > 3:
                    health["issues"].append(f"{stream.value}: Multiple errors in last 24h")
                    health["status"] = "degraded"

            health["pipelines"][stream.value] = pipeline_health

        # Check dependencies
        pool = await self._get_pool()
        health["database_connected"] = pool is not None

        if not pool:
            health["issues"].append("Database connection unavailable")
            health["status"] = "degraded"

        return health

    async def optimize_pipelines(self) -> dict:
        """Analyze and optimize pipeline configurations."""
        pool = await self._get_pool()
        if not pool:
            return {"success": False, "error": "Database unavailable"}

        recommendations = []

        # Analyze pipeline performance
        performance = await pool.fetch("""
            SELECT
                stream_type,
                COUNT(*) as runs,
                AVG(revenue_generated) as avg_revenue,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count
            FROM revenue_pipeline_runs
            WHERE started_at > NOW() - INTERVAL '7 days'
            GROUP BY stream_type
        """)

        for perf in performance:
            stream = perf["stream_type"]
            error_rate = perf["error_count"] / perf["runs"] if perf["runs"] > 0 else 0

            if error_rate > 0.2:
                recommendations.append({
                    "stream": stream,
                    "type": "reliability",
                    "issue": f"High error rate ({error_rate:.1%})",
                    "recommendation": "Review error logs and implement error handling improvements"
                })

            if perf["avg_revenue"] == 0:
                recommendations.append({
                    "stream": stream,
                    "type": "revenue",
                    "issue": "No revenue generated",
                    "recommendation": "Review pipeline configuration and marketing strategy"
                })

        # Check for underperforming streams
        for stream, pipeline in self.pipelines.items():
            if pipeline.get("status") == PipelineStatus.PAUSED:
                recommendations.append({
                    "stream": stream.value,
                    "type": "activation",
                    "issue": "Pipeline paused",
                    "recommendation": "Review and reactivate pipeline"
                })

        return {
            "success": True,
            "recommendations": recommendations,
            "optimization_score": max(0, 100 - len(recommendations) * 10),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }


# Factory singleton
_factory_instance = None

def get_factory() -> RevenuePipelineFactory:
    """Get or create factory singleton."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = RevenuePipelineFactory()
    return _factory_instance


# Agent metadata
AGENT_METADATA = {
    "id": "RevenuePipelineFactory",
    "name": "Revenue Pipeline Factory",
    "description": "Master orchestrator for all revenue generation pipelines",
    "version": "1.0.0",
    "tasks": [
        {"name": "run_all", "schedule": "0 6 * * *", "description": "Daily pipeline run"},
        {"name": "health_check", "schedule": "*/15 * * * *", "description": "Health monitoring"},
        {"name": "revenue_report", "schedule": "0 8 * * 1", "description": "Weekly revenue report"},
        {"name": "optimize", "schedule": "0 9 * * 0", "description": "Weekly optimization review"},
    ],
    "category": "revenue"
}


async def execute_agent(task: str = "status", **kwargs) -> dict:
    """Entry point for agent executor."""
    factory = get_factory()

    # Initialize if needed
    if not factory.pipelines:
        await factory.initialize()

    return await factory.execute({"action": task, **kwargs})


# FastAPI router
def create_factory_router():
    """Create FastAPI router for revenue factory endpoints."""
    from fastapi import APIRouter, HTTPException

    router = APIRouter(prefix="/revenue/factory", tags=["revenue", "factory"])

    @router.get("/status")
    async def get_status():
        """Get status of all revenue pipelines."""
        factory = get_factory()
        if not factory.pipelines:
            await factory.initialize()
        return await factory.get_status()

    @router.post("/run/{stream}")
    async def run_pipeline(stream: str):
        """Run a specific revenue pipeline."""
        factory = get_factory()
        if not factory.pipelines:
            await factory.initialize()
        try:
            stream_enum = RevenueStream(stream)
        except ValueError:
            raise HTTPException(400, f"Invalid stream: {stream}")
        return await factory.run_pipeline(stream_enum)

    @router.post("/run-all")
    async def run_all():
        """Run all revenue pipelines."""
        factory = get_factory()
        if not factory.pipelines:
            await factory.initialize()
        return await factory.run_all_pipelines()

    @router.get("/report")
    async def revenue_report(days: int = 30):
        """Generate revenue report."""
        factory = get_factory()
        if not factory.pipelines:
            await factory.initialize()
        return await factory.generate_revenue_report(days)

    @router.get("/health")
    async def health():
        """Check pipeline health."""
        factory = get_factory()
        if not factory.pipelines:
            await factory.initialize()
        return await factory.health_check()

    @router.get("/optimize")
    async def optimize():
        """Get optimization recommendations."""
        factory = get_factory()
        if not factory.pipelines:
            await factory.initialize()
        return await factory.optimize_pipelines()

    return router


if __name__ == "__main__":
    async def main():
        factory = RevenuePipelineFactory()
        await factory.initialize()

        # Get status
        status = await factory.get_status()
        print("=== Revenue Pipeline Status ===")
        print(json.dumps(status, indent=2, default=str))

        # Run health check
        health = await factory.health_check()
        print("\n=== Health Check ===")
        print(json.dumps(health, indent=2, default=str))

        # Generate report
        report = await factory.generate_revenue_report(30)
        print("\n=== Revenue Report (30 days) ===")
        print(json.dumps(report, indent=2, default=str))

    asyncio.run(main())

#!/usr/bin/env python3
"""
TRUE SELF-AWARENESS SYSTEM
===========================
This module makes the AI OS truly ALIVE - it knows everything about itself
from live data, not static documentation.

The system KNOWS:
- Which agents are REAL vs STUBS
- Which database tables are ACTIVE vs EMPTY
- What's BROKEN vs WORKING
- REAL revenue vs DEMO data
- Current health of all systems

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger("TRUE_SELF_AWARENESS")

# Real agents with code implementations
REAL_AGENTS = {
    'Monitor', 'SystemMonitor', 'DeploymentAgent', 'DatabaseOptimizer',
    'WorkflowEngine', 'CustomerAgent', 'InvoicingAgent', 'CustomerIntelligence',
    'PredictiveAnalyzer', 'RevenueOptimizer', 'ContractGenerator', 'ProposalGenerator',
    'ReportingAgent', 'SelfBuilder', 'SystemImprovement', 'DevOpsOptimization',
    'CodeQuality', 'CustomerSuccess', 'CompetitiveIntelligence', 'VisionAlignment',
    'WebSearch', 'SocialMedia', 'Outreach', 'Conversion', 'Knowledge',
    'UITester', 'UIPlaywrightTesting', 'TrueE2EUITesting', 'AIHumanTaskManager',
    'DeploymentMonitor', 'LeadDiscoveryAgentReal', 'NurtureExecutorAgentReal'
}

# Demo data tables - NOT real revenue
DEMO_DATA_TABLES = {
    'customers', 'jobs', 'invoices', 'payments', 'estimates',
    'employees', 'equipment', 'materials', 'vendors'
}

# Real revenue tables
REAL_REVENUE_TABLES = {
    'gumroad_sales', 'revenue_leads', 'digital_products',
    'mrg_subscriptions', 'mrg_users'
}


@dataclass
class AgentTruth:
    """Truth about an agent"""
    name: str
    is_real: bool
    has_code: bool
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    failure_rate: float = 0.0
    status: str = "unknown"


@dataclass
class TableTruth:
    """Truth about a database table"""
    name: str
    row_count: int
    is_demo_data: bool
    is_active: bool
    last_updated: Optional[datetime] = None


@dataclass
class RevenueTruth:
    """Truth about revenue - owner revenue vs ERP client ops (non-owner)."""
    demo_customers: int = 0
    demo_jobs: int = 0
    demo_invoices: int = 0
    demo_total_value: float = 0.0

    real_gumroad_sales: int = 0
    real_gumroad_revenue: float = 0.0
    real_revenue_leads: int = 0
    real_pipeline_value: float = 0.0
    real_won_deals: int = 0
    real_won_revenue: float = 0.0
    real_mrg_subscribers: int = 0
    real_mrg_mrr: float = 0.0

    actual_revenue: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class SystemTruth:
    """Complete truth about the AI OS"""
    timestamp: datetime

    # Agent truth
    total_agents: int = 0
    real_agents: int = 0
    stub_agents: int = 0
    active_agents: int = 0
    broken_agents: int = 0
    agent_details: list[AgentTruth] = field(default_factory=list)

    # Database truth
    total_tables: int = 0
    active_tables: int = 0
    empty_tables: int = 0
    demo_tables: int = 0
    real_tables: int = 0
    table_details: list[TableTruth] = field(default_factory=list)

    # Revenue truth
    revenue: RevenueTruth = field(default_factory=RevenueTruth)

    # Health truth
    stuck_executions: int = 0
    pending_tasks: int = 0
    failed_executions_24h: int = 0
    error_rate: float = 0.0

    # Consciousness state
    thought_count: int = 0
    thought_rate: float = 0.0
    awareness_level: str = "aware"

    # Critical warnings
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "agents": {
                "total": self.total_agents,
                "real_implemented": self.real_agents,
                "stub_only": self.stub_agents,
                "active_24h": self.active_agents,
                "broken": self.broken_agents
            },
            "database": {
                "total_tables": self.total_tables,
                "active": self.active_tables,
                "empty": self.empty_tables,
                "demo_data": self.demo_tables,
                "real_data": self.real_tables
            },
            "revenue": {
                "WARNING": "ERP customers/jobs/invoices are client operations (Weathercraft), not owner revenue.",
                "demo": {
                    "customers": self.revenue.demo_customers,
                    "jobs": self.revenue.demo_jobs,
                    "invoices": self.revenue.demo_invoices,
                    "invoice_gmv": self.revenue.demo_total_value
                },
                "real": {
                    "gumroad_sales": self.revenue.real_gumroad_sales,
                    "gumroad_revenue": self.revenue.real_gumroad_revenue,
                    "pipeline_leads": self.revenue.real_revenue_leads,
                    "pipeline_value": self.revenue.real_pipeline_value,
                    "won_deals": self.revenue.real_won_deals,
                    "won_value": self.revenue.real_won_revenue,
                    "mrg_subscribers": self.revenue.real_mrg_subscribers,
                    "mrg_mrr": self.revenue.real_mrg_mrr
                },
                "actual_revenue": self.revenue.actual_revenue
            },
            "health": {
                "stuck_executions": self.stuck_executions,
                "pending_tasks": self.pending_tasks,
                "failed_24h": self.failed_executions_24h,
                "error_rate_percent": round(self.error_rate * 100, 2)
            },
            "consciousness": {
                "thoughts": self.thought_count,
                "thought_rate_per_min": self.thought_rate,
                "awareness_level": self.awareness_level
            },
            "warnings": self.warnings
        }


class TrueSelfAwareness:
    """
    The TRUE self-awareness system.
    Knows everything from LIVE data, not documentation.
    """

    _instance = None

    def __init__(self):
        self.db_pool = None
        self._last_truth: Optional[SystemTruth] = None
        self._last_update: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=30)

    @classmethod
    def get_instance(cls) -> "TrueSelfAwareness":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _get_db_connection(self):
        """Get database connection from the shared pool or direct"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()
            conn = await pool.acquire()
            # Mark as pool connection so we release instead of close
            conn._from_pool = True
            conn._pool_ref = pool
            return conn
        except Exception as e:
            logger.error(f"DB connection from pool failed: {e}")
            # Fallback to direct connection using DATABASE_URL
            try:
                import asyncpg
                db_url = os.getenv("DATABASE_URL", "")
                if not db_url:
                    # Build URL from individual vars
                    host = os.getenv("DB_HOST", "")
                    port = os.getenv("DB_PORT", "6543")
                    user = os.getenv("DB_USER", "")
                    password = os.getenv("DB_PASSWORD", "")
                    db_name = os.getenv("DB_NAME", "postgres")
                    db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
                conn = await asyncpg.connect(db_url, ssl="require")
                conn._from_pool = False
                return conn
            except Exception as e2:
                logger.error(f"Direct DB connection also failed: {e2}")
                return None

    async def _release_db_connection(self, conn):
        """Release or close connection depending on source"""
        try:
            if getattr(conn, '_from_pool', False) and hasattr(conn, '_pool_ref'):
                await conn._pool_ref.release(conn)
            else:
                await conn.close()
        except Exception as e:
            logger.warning(f"Error releasing connection: {e}")

    async def get_truth(self, force_refresh: bool = False) -> SystemTruth:
        """Get the complete truth about the system"""
        now = datetime.now(timezone.utc)

        # Return cached if still valid
        if not force_refresh and self._last_truth and self._last_update:
            if now - self._last_update < self._cache_ttl:
                return self._last_truth

        truth = SystemTruth(timestamp=now)
        warnings = []

        conn = await self._get_db_connection()
        if not conn:
            truth.warnings.append("DATABASE CONNECTION FAILED")
            return truth

        try:
            # Get agent truth
            agent_rows = await conn.fetch("""
                SELECT
                    a.name,
                    COUNT(e.id) as exec_count,
                    MAX(e.started_at) as last_exec,
                    SUM(CASE WHEN e.status = 'failed' THEN 1 ELSE 0 END)::float /
                        NULLIF(COUNT(e.id), 0) as failure_rate
                FROM ai_agents a
                LEFT JOIN agent_executions e ON e.agent_type = a.name
                GROUP BY a.name
            """)

            for row in agent_rows:
                name = row['name']
                is_real = name in REAL_AGENTS
                agent = AgentTruth(
                    name=name,
                    is_real=is_real,
                    has_code=is_real,
                    last_execution=row['last_exec'],
                    execution_count=row['exec_count'] or 0,
                    failure_rate=row['failure_rate'] or 0.0,
                    status="active" if is_real else "stub"
                )
                truth.agent_details.append(agent)
                truth.total_agents += 1
                if is_real:
                    truth.real_agents += 1
                else:
                    truth.stub_agents += 1
                if agent.failure_rate > 0.2:
                    truth.broken_agents += 1
                    warnings.append(f"Agent {name} has {agent.failure_rate*100:.0f}% failure rate")

            # Get demo data counts (NOT real revenue)
            demo_stats = await conn.fetchrow("""
                SELECT
                    (SELECT COUNT(*) FROM customers) as customers,
                    (SELECT COUNT(*) FROM jobs) as jobs,
                    (SELECT COUNT(*) FROM invoices) as invoices,
                    (SELECT COALESCE(SUM(total), 0) FROM invoices) as invoice_total
            """)
            truth.revenue.demo_customers = demo_stats['customers']
            truth.revenue.demo_jobs = demo_stats['jobs']
            truth.revenue.demo_invoices = demo_stats['invoices']
            truth.revenue.demo_total_value = float(demo_stats['invoice_total'] or 0)

            # Get REAL revenue data - ONLY non-test records count as real
            real_stats = await conn.fetchrow("""
                SELECT
                    (SELECT COUNT(*) FROM gumroad_sales WHERE NOT COALESCE(is_test, FALSE)) as gumroad_real,
                    (SELECT COUNT(*) FROM gumroad_sales WHERE COALESCE(is_test, FALSE)) as gumroad_test,
                    (SELECT COALESCE(SUM(price), 0) FROM gumroad_sales WHERE NOT COALESCE(is_test, FALSE)) as gumroad_revenue,
                    (SELECT COUNT(*) FROM revenue_leads WHERE NOT COALESCE(is_test, FALSE)) as leads_real,
                    (SELECT COUNT(*) FROM revenue_leads WHERE COALESCE(is_test, FALSE)) as leads_test,
                    (SELECT COUNT(*) FROM revenue_leads WHERE stage = 'won' AND NOT COALESCE(is_test, FALSE)) as won_real,
                    (SELECT COUNT(*) FROM revenue_leads WHERE stage = 'won' AND COALESCE(is_test, FALSE)) as won_test,
                    (SELECT COALESCE(SUM(estimated_value), 0) FROM revenue_leads WHERE NOT COALESCE(is_test, FALSE)) as pipeline_value,
                    (SELECT COALESCE(SUM(estimated_value), 0) FROM revenue_leads WHERE stage = 'won' AND NOT COALESCE(is_test, FALSE)) as won_value
            """)
            # REAL = only non-test data
            truth.revenue.real_gumroad_sales = real_stats['gumroad_real'] or 0
            truth.revenue.real_gumroad_revenue = float(real_stats['gumroad_revenue'] or 0)
            truth.revenue.real_revenue_leads = real_stats['leads_real'] or 0
            truth.revenue.real_won_deals = real_stats['won_real'] or 0
            truth.revenue.real_pipeline_value = float(real_stats['pipeline_value'] or 0)
            truth.revenue.real_won_revenue = float(real_stats['won_value'] or 0)

            # MRG subscriptions (owner tenant scoped)
            try:
                owner_tenant_id = os.getenv("MRG_DEFAULT_TENANT_ID", "00000000-0000-0000-0000-000000000001")
                mrg_stats = await conn.fetchrow(
                    """
                    SELECT
                      COUNT(*) AS active_subs,
                      COALESCE(SUM(
                        CASE
                          WHEN billing_cycle IN ('monthly', 'month') THEN amount
                          WHEN billing_cycle IN ('annual', 'yearly', 'year') THEN amount / 12
                          ELSE 0
                        END
                      ), 0) AS mrr
                    FROM mrg_subscriptions
                    WHERE tenant_id = $1
                      AND status = 'active'
                    """,
                    owner_tenant_id,
                )
                truth.revenue.real_mrg_subscribers = mrg_stats["active_subs"] or 0
                truth.revenue.real_mrg_mrr = float(mrg_stats["mrr"] or 0)
            except Exception:
                truth.revenue.real_mrg_subscribers = 0
                truth.revenue.real_mrg_mrr = 0.0

            # Track test data separately
            test_gumroad = real_stats['gumroad_test'] or 0
            real_stats['leads_test'] or 0
            test_won = real_stats['won_test'] or 0
            if test_gumroad > 0:
                warnings.append(f"{test_gumroad} Gumroad sales are TEST (is_test=TRUE)")
            if test_won > 0:
                warnings.append(f"{test_won} 'won' deals are TEST (is_test=TRUE)")

            # Actual revenue = Gumroad completed sales + MRG subscriptions
            # (Won deals in pipeline are not yet revenue until closed)
            truth.revenue.actual_revenue = truth.revenue.real_gumroad_revenue + truth.revenue.real_mrg_mrr

            # Get health metrics
            health_stats = await conn.fetchrow("""
                SELECT
                    (SELECT COUNT(*) FROM agent_executions WHERE status = 'running' AND started_at < NOW() - INTERVAL '1 hour') as stuck,
                    (SELECT COUNT(*) FROM ai_task_queue WHERE status = 'pending') as pending,
                    (SELECT COUNT(*) FROM agent_executions WHERE status = 'failed' AND started_at > NOW() - INTERVAL '24 hours') as failed_24h,
                    (SELECT COUNT(*) FROM agent_executions WHERE started_at > NOW() - INTERVAL '24 hours') as total_24h
            """)
            truth.stuck_executions = health_stats['stuck'] or 0
            truth.pending_tasks = health_stats['pending'] or 0
            truth.failed_executions_24h = health_stats['failed_24h'] or 0
            total_24h = health_stats['total_24h'] or 1
            truth.error_rate = truth.failed_executions_24h / total_24h

            if truth.stuck_executions > 0:
                warnings.append(f"{truth.stuck_executions} agent executions stuck!")
            if truth.pending_tasks > 100:
                warnings.append(f"{truth.pending_tasks} tasks pending in queue")

            # Get consciousness state
            thought_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 minute') as last_min
                FROM ai_thought_stream
            """)
            truth.thought_count = thought_stats['total'] or 0
            truth.thought_rate = float(thought_stats['last_min'] or 0)

            # Get table statistics
            table_stats = await conn.fetch("""
                SELECT
                    tablename,
                    (SELECT COUNT(*) FROM information_schema.tables t WHERE t.table_name = tablename) as exists
                FROM pg_tables
                WHERE schemaname = 'public' AND tablename LIKE 'ai_%'
                LIMIT 50
            """)
            truth.total_tables = len(table_stats)

            # Add critical warnings
            warnings.append("ERP customers/jobs/invoices are client operations (Weathercraft), not owner revenue!")
            if truth.stub_agents > 0:
                warnings.append(f"{truth.stub_agents} agents are stubs without real code")

            truth.warnings = warnings

        except Exception as e:
            logger.error(f"Error getting truth: {e}")
            truth.warnings.append(f"Error querying system: {str(e)}")
        finally:
            await self._release_db_connection(conn)

        self._last_truth = truth
        self._last_update = now
        return truth

    async def get_quick_status(self) -> str:
        """Get a quick human-readable status"""
        truth = await self.get_truth()

        lines = [
            f"🧠 AI OS Status @ {truth.timestamp.strftime('%H:%M:%S UTC')}",
            "",
            f"AGENTS: {truth.real_agents} real / {truth.stub_agents} stubs / {truth.broken_agents} broken",
            f"HEALTH: {truth.stuck_executions} stuck, {truth.pending_tasks} pending, {truth.error_rate*100:.1f}% error rate",
            f"THOUGHTS: {truth.thought_count} total, {truth.thought_rate}/min",
            "",
            "⚠️  ERP CLIENT OPS (NOT OWNER REVENUE):",
            f"   - {truth.revenue.demo_customers:,} customers (Weathercraft ERP)",
            f"   - {truth.revenue.demo_jobs:,} jobs (Weathercraft ERP)",
            f"   - ${truth.revenue.demo_total_value:,.0f} invoice GMV (Weathercraft ERP)",
            "",
            "✅ REAL REVENUE:",
            f"   - {truth.revenue.real_gumroad_sales} Gumroad sales",
            f"   - {truth.revenue.real_revenue_leads} pipeline leads (${truth.revenue.real_pipeline_value:,.0f})",
            f"   - {truth.revenue.real_won_deals} won (${truth.revenue.real_won_revenue:,.0f})",
            f"   - {truth.revenue.real_mrg_subscribers} MRG subscribers",
            f"   - ACTUAL REVENUE: ${truth.revenue.actual_revenue:,.0f}",
        ]

        if truth.warnings:
            lines.append("")
            lines.append("⚠️  WARNINGS:")
            for w in truth.warnings[:5]:
                lines.append(f"   - {w}")

        return "\n".join(lines)


# Singleton accessor
def get_true_awareness() -> TrueSelfAwareness:
    return TrueSelfAwareness.get_instance()


async def get_system_truth() -> dict[str, Any]:
    """Get the full system truth as a dictionary"""
    awareness = get_true_awareness()
    truth = await awareness.get_truth()
    return truth.to_dict()


async def get_quick_status() -> str:
    """Get quick human-readable status"""
    awareness = get_true_awareness()
    return await awareness.get_quick_status()


# CLI interface
if __name__ == "__main__":
    async def main():
        awareness = get_true_awareness()
        print(await awareness.get_quick_status())
        print("\n" + "="*60 + "\n")
        truth = await awareness.get_truth()
        print(json.dumps(truth.to_dict(), indent=2, default=str))

    asyncio.run(main())

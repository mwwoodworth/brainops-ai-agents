#!/usr/bin/env python3
"""
BrainOps Unified DevOps Control (God Mode)
Aggregates health, revenue, and agent status into a single dashboard.
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# Load Env
secure_env = "/home/matt-woodworth/dev/_secure/BrainOps.env"
if os.path.exists(secure_env):
    load_dotenv(secure_env, override=True)

from config import config
from database.async_connection import init_pool, close_pool, get_pool, PoolConfig

WEATHERCRAFT_TENANT_ID = os.getenv("WEATHERCRAFT_TENANT_ID", "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457")
WEATHERCRAFT_QA_TENANT_ID = os.getenv("WEATHERCRAFT_QA_TENANT_ID", "497509e7-b9f8-4766-922c-94ed19626547")
SYSTEM_TENANT_ID = os.getenv("SYSTEM_TENANT_ID", "00000000-0000-0000-0000-000000000001")
MRG_DEFAULT_TENANT_ID = os.getenv("MRG_DEFAULT_TENANT_ID", SYSTEM_TENANT_ID)

async def check_system_health():
    print("\n🖥️  SYSTEM HEALTH")
    print("-" * 20)
    
    # 1. Database
    try:
        pool_config = PoolConfig(
            host=config.database.host,
            port=config.database.port,
            user=config.database.user,
            password=config.database.password,
            database=config.database.database,
            ssl=config.database.ssl,
            ssl_verify=config.database.ssl_verify
        )
        await init_pool(pool_config)
        pool = get_pool()
        res = await pool.fetchval("SELECT version()")
        print(f"✅ Database: CONNECTED ({res.split()[0]} {res.split()[1]})")
    except Exception as e:
        print(f"❌ Database: FAILED ({e})")
        return

async def check_erp_ledger_integrity():
    print("\n💰 ERP INVOICE LEDGER INTEGRITY (WEATHERCRAFT TENANTS)")
    print("-" * 20)
    pool = get_pool()

    # Ledger Gap
    gap = await pool.fetchval("""
        SELECT count(*) FROM invoices i 
        WHERE i.status = 'paid' 
        AND i.tenant_id = ANY($1::uuid[])
        AND NOT EXISTS (
            SELECT 1 FROM real_revenue_tracking r 
            WHERE r.metadata->>'invoice_id' = i.id::text
        )
    """, [WEATHERCRAFT_TENANT_ID, WEATHERCRAFT_QA_TENANT_ID])
    if gap == 0:
        print("✅ Ledger: PERFECT (0 missing entries)")
    else:
        print(f"❌ Ledger: GAP ({gap} paid invoices missing from ledger)")

    # Dunning
    failed = await pool.fetchval("SELECT count(*) FROM failed_payments WHERE status = 'pending'")
    print(f"⚠️  Active Dunning Cases: {failed}")

async def check_personal_revenue():
    print("\n💵 PERSONAL REVENUE (OWNER ONLY; EXCLUDES ERP INVOICES)")
    print("-" * 20)
    pool = get_pool()

    gumroad = await pool.fetchrow(
        """
        SELECT
          COUNT(*) FILTER (WHERE NOT COALESCE(is_test, FALSE)) AS real_count,
          COALESCE(SUM(price::numeric) FILTER (WHERE NOT COALESCE(is_test, FALSE)), 0) AS real_sum,
          COUNT(*) FILTER (WHERE COALESCE(is_test, FALSE)) AS test_count,
          COALESCE(SUM(price::numeric) FILTER (WHERE COALESCE(is_test, FALSE)), 0) AS test_sum
        FROM gumroad_sales
        """
    )
    gumroad_real = float(gumroad["real_sum"] or 0)
    gumroad_test = float(gumroad["test_sum"] or 0)
    print(
        f"✅ Gumroad: real {int(gumroad['real_count'] or 0)} (${gumroad_real:,.2f})"
        f" | test {int(gumroad['test_count'] or 0)} (${gumroad_test:,.2f})"
    )

    mrg_active_default = await pool.fetchval(
        "SELECT COUNT(*) FROM mrg_subscriptions WHERE status='active' AND tenant_id = $1",
        MRG_DEFAULT_TENANT_ID,
    )
    mrg_active_all = await pool.fetchval(
        "SELECT COUNT(*) FROM mrg_subscriptions WHERE status='active'"
    )
    print(
        f"✅ MRG Subscriptions: {int(mrg_active_default or 0)} active (default tenant)"
        f" | {int(mrg_active_all or 0)} active (all tenants)"
    )

    stripe = await pool.fetchrow(
        """
        SELECT
          COUNT(*) AS total_30d,
          COUNT(*) FILTER (WHERE processed_at IS NOT NULL) AS processed_30d,
          COUNT(*) FILTER (WHERE processed_at IS NULL) AS pending_30d,
          COUNT(*) FILTER (WHERE tenant_id IS NULL) AS missing_tenant_30d,
          MAX(created_at) AS last_event_at
        FROM stripe_webhook_events
        WHERE created_at > NOW() - INTERVAL '30 days'
        """
    )
    print(
        "✅ Stripe webhooks (30d):"
        f" total={int(stripe['total_30d'] or 0)}"
        f" processed={int(stripe['processed_30d'] or 0)}"
        f" pending={int(stripe['pending_30d'] or 0)}"
        f" missing_tenant={int(stripe['missing_tenant_30d'] or 0)}"
        f" last={stripe['last_event_at']}"
    )

async def check_agent_status():
    print("\n🤖 AGENT STATUS")
    print("-" * 20)
    pool = get_pool()

    rows = await pool.fetch(
        """
        SELECT
          agent_name,
          COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour' AND status = 'completed') AS completed_1h,
          COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour' AND status IN ('failed', 'error')) AS failed_1h,
          COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour' AND status = 'running') AS running_1h,

          COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '6 hours' AND status = 'completed') AS completed_6h,
          COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '6 hours' AND status IN ('failed', 'error')) AS failed_6h,
          COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '6 hours' AND status = 'running') AS running_6h,

          COUNT(*) FILTER (WHERE status = 'completed') AS completed_24h,
          COUNT(*) FILTER (WHERE status IN ('failed', 'error')) AS failed_24h,
          COUNT(*) FILTER (WHERE status = 'running') AS running_24h
        FROM ai_agent_executions
        WHERE created_at > NOW() - INTERVAL '24 hours'
        GROUP BY agent_name
        ORDER BY agent_name
        """
    )

    if not rows:
        print("   (No agent activity in last 24h)")
        return

    def rate(completed: int, failed: int) -> float:
        denom = completed + failed
        return (completed / denom) * 100 if denom > 0 else 0.0

    totals = {
        "1h": {"completed": 0, "failed": 0, "running": 0},
        "6h": {"completed": 0, "failed": 0, "running": 0},
        "24h": {"completed": 0, "failed": 0, "running": 0},
    }

    for r in rows:
        totals["1h"]["completed"] += int(r["completed_1h"] or 0)
        totals["1h"]["failed"] += int(r["failed_1h"] or 0)
        totals["1h"]["running"] += int(r["running_1h"] or 0)
        totals["6h"]["completed"] += int(r["completed_6h"] or 0)
        totals["6h"]["failed"] += int(r["failed_6h"] or 0)
        totals["6h"]["running"] += int(r["running_6h"] or 0)
        totals["24h"]["completed"] += int(r["completed_24h"] or 0)
        totals["24h"]["failed"] += int(r["failed_24h"] or 0)
        totals["24h"]["running"] += int(r["running_24h"] or 0)

    print(
        "OVERALL:"
        f"  1h {rate(totals['1h']['completed'], totals['1h']['failed']):5.1f}%"
        f"  6h {rate(totals['6h']['completed'], totals['6h']['failed']):5.1f}%"
        f"  24h {rate(totals['24h']['completed'], totals['24h']['failed']):5.1f}%"
        f"  (running 1h/6h/24h: {totals['1h']['running']}/{totals['6h']['running']}/{totals['24h']['running']})"
    )

    for r in rows:
        agent = r["agent_name"]
        c1, f1 = int(r["completed_1h"] or 0), int(r["failed_1h"] or 0)
        c6, f6 = int(r["completed_6h"] or 0), int(r["failed_6h"] or 0)
        c24, f24 = int(r["completed_24h"] or 0), int(r["failed_24h"] or 0)
        run24 = int(r["running_24h"] or 0)

        r1 = rate(c1, f1)
        icon = "✅" if (c1 + f1) > 0 and r1 >= 95 else "⚠️" if (c1 + f1) > 0 and r1 >= 80 else "❌"
        if (c1 + f1) == 0 and run24 == 0:
            icon = "⚪"

        print(
            f"{icon} {agent.ljust(28)}"
            f"  1h {r1:5.1f}% ({c1}/{c1+f1})"
            f"  6h {rate(c6, f6):5.1f}% ({c6}/{c6+f6})"
            f"  24h {rate(c24, f24):5.1f}% ({c24}/{c24+f24})"
            f"  running(24h)={run24}"
        )

async def main():
    print(f"🚀 BRAINOPS GOD MODE [{datetime.now().isoformat()}]")
    print("===================================================")
    
    try:
        await check_system_health()
        await check_erp_ledger_integrity()
        await check_personal_revenue()
        await check_agent_status()
    finally:
        await close_pool()
    
    print("\n✅ Status Check Complete.")

if __name__ == "__main__":
    asyncio.run(main())

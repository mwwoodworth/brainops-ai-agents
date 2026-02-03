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

async def check_revenue_integrity():
    print("\n💰 REVENUE INTEGRITY")
    print("-" * 20)
    pool = get_pool()
    
    # Ledger Gap
    gap = await pool.fetchval("""
        SELECT count(*) FROM invoices i 
        WHERE i.status = 'paid' 
        AND NOT EXISTS (
            SELECT 1 FROM real_revenue_tracking r 
            WHERE r.metadata->>'invoice_id' = i.id::text
        )
    """)
    if gap == 0:
        print("✅ Ledger: PERFECT (0 missing entries)")
    else:
        print(f"❌ Ledger: GAP ({gap} paid invoices missing from ledger)")

    # Dunning
    failed = await pool.fetchval("SELECT count(*) FROM failed_payments WHERE status = 'pending'")
    print(f"⚠️  Active Dunning Cases: {failed}")

async def check_agent_status():
    print("\n🤖 AGENT STATUS")
    print("-" * 20)
    pool = get_pool()
    
    stats = await pool.fetch("""
        SELECT agent_name, status, count(*) as count 
        FROM ai_agent_executions 
        WHERE created_at > NOW() - INTERVAL '24 hours'
        GROUP BY agent_name, status
    """)
    
    if not stats:
        print("   (No agent activity in last 24h)")
    
    summary = {}
    for r in stats:
        agent = r['agent_name']
        if agent not in summary: summary[agent] = {"success": 0, "failed": 0}
        if r['status'] == 'completed': summary[agent]["success"] += r['count']
        if r['status'] == 'failed': summary[agent]["failed"] += r['count']
        
    for agent, s in summary.items():
        rate = (s['success'] / (s['success'] + s['failed'])) * 100 if (s['success'] + s['failed']) > 0 else 0
        status_icon = "✅" if rate > 90 else "⚠️" if rate > 50 else "❌"
        print(f"{status_icon} {agent.ljust(25)}: {int(rate)}% Success ({s['success']}/{s['success']+s['failed']})")

async def main():
    print(f"🚀 BRAINOPS GOD MODE [{datetime.now().isoformat()}]")
    print("===================================================")
    
    try:
        await check_system_health()
        await check_revenue_integrity()
        await check_agent_status()
    finally:
        await close_pool()
    
    print("\n✅ Status Check Complete.")

if __name__ == "__main__":
    asyncio.run(main())
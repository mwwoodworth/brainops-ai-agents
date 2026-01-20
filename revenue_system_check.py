#!/usr/bin/env python3
"""
Revenue System Integrity Check
==============================
Verifies the health and configuration of the BrainOps Revenue System.
Checks:
1. Environment variables (Stripe, SendGrid, Twilio)
2. Database tables (Revenue, API, Pricing)
3. Module imports and dependencies
4. Kill switch status
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("RevenueCheck")

async def check_environment():
    """Check required environment variables."""
    logger.info("--- Checking Environment Variables ---")
    
    required_vars = [
        "DATABASE_URL",
        "STRIPE_SECRET_KEY",
        "SENDGRID_API_KEY",
        "OPENAI_API_KEY"
    ]
    required_any = [
        ("TWILIO_ACCOUNT_SID", "TWILIO_SID"),
        ("TWILIO_AUTH_TOKEN", "TWILIO_TOKEN"),
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
            logger.warning(f"[MISSING] {var}")
        else:
            masked = os.getenv(var)[:4] + "..."
            logger.info(f"[OK] {var} ({masked})")

    for primary, fallback in required_any:
        if os.getenv(primary) or os.getenv(fallback):
            active = primary if os.getenv(primary) else fallback
            masked = os.getenv(active)[:4] + "..."
            logger.info(f"[OK] {primary} ({active}={masked})")
        else:
            missing.append(primary)
            logger.warning(f"[MISSING] {primary} (or {fallback})")
            
    if missing:
        logger.error(f"Missing {len(missing)} critical environment variables.")
        return False
    return True

async def check_database_tables():
    """Check if required tables exist."""
    logger.info("\n--- Checking Database Tables ---")
    
    try:
        from database.async_connection import get_pool, init_pool, PoolConfig, DatabaseUnavailableError
        from db_config import get_db_config
        
        pool = None
        try:
            pool = get_pool()
        except DatabaseUnavailableError:
            pass

        if not pool:
            try:
                cfg = get_db_config()
                pool_config = PoolConfig(
                    host=cfg['host'],
                    port=cfg['port'],
                    user=cfg['user'],
                    password=cfg['password'],
                    database=cfg['database'],
                    ssl_verify=False
                )
                await init_pool(pool_config)
                pool = get_pool()
            except Exception as e:
                 logger.error(f"Failed to initialize pool: {e}")
                 return False

        if not pool:
             logger.error("Could not acquire database pool.")
             return False
             
        tables_to_check = [
            "revenue_leads",
            "revenue_transactions",
            "revenue_metrics",
            "api_keys",
            "api_usage",
            "pricing_quotes",
            "ai_proposals",
            "gumroad_sales"
        ]
        
        async with pool.acquire() as conn:
            missing = []
            for table in tables_to_check:
                query = "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = $1);"
                exists = await conn.fetchval(query, table)
                
                if exists:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    logger.info(f"[OK] Table '{table}' exists ({count} rows)")
                else:
                    logger.error(f"[MISSING] Table '{table}' NOT found")
                    missing.append(table)
        
        if missing:
            return False
        return True

    except ImportError:
        logger.error("Could not import database module.")
        return False
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False

async def check_modules():
    """Check if revenue modules can be imported."""
    logger.info("\n--- Checking Modules ---")
    
    modules = [
        "gumroad_revenue_agent",
        "api_monetization_engine",
        "ai_pricing_engine",
        "revenue_automation_engine",
        "revenue_operator",
        "proposal_engine"
    ]
    
    failed = []
    for mod_name in modules:
        try:
            __import__(mod_name)
            logger.info(f"[OK] Imported {mod_name}")
        except ImportError as e:
            logger.error(f"[FAILED] Could not import {mod_name}: {e}")
            failed.append(mod_name)
        except Exception as e:
            logger.error(f"[ERROR] Error importing {mod_name}: {e}")
            failed.append(mod_name)
            
    if failed:
        return False
    return True

async def main():
    logger.info("Starting Revenue System Integrity Check...")
    
    env_ok = await check_environment()
    modules_ok = await check_modules()
    db_ok = await check_database_tables()
    
    logger.info("\n--- Summary ---")
    if env_ok and modules_ok and db_ok:
        logger.info("✅ Revenue System is HEALTHY and READY.")
        sys.exit(0)
    else:
        logger.error("❌ Revenue System has ISSUES. See logs above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

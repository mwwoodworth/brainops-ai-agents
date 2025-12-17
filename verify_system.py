
import os
import sys
import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemVerifier")

# DB Credentials from context/files
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
    'password': os.getenv('DB_PASSWORD', 'REDACTED_SUPABASE_DB_PASSWORD'), # Using the one from context
    'port': int(os.getenv('DB_PORT', 5432))
}

async def verify_db():
    logger.info("--- Verifying Database Connection ---")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()
        logger.info(f"✅ Database connected: {version[0]}")
        
        # Check for key tables
        tables = ["ai_agents", "agent_executions", "ai_error_logs", "ai_component_health"]
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                logger.info(f"✅ Table '{table}' exists with {count} rows")
            except Exception as e:
                logger.warning(f"⚠️ Table '{table}' check failed: {e}")
                conn.rollback() # Reset transaction
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

async def verify_ai_core():
    logger.info("\n--- Verifying AI Core ---")
    try:
        from ai_core import RealAICore
        ai = RealAICore()
        logger.info("✅ RealAICore instantiated")
        
        # We might not have API keys in this env, so just checking instantiation and logic
        if not ai.openai_client and not ai.anthropic_client:
             logger.warning("⚠️ No AI Clients initialized (API keys likely missing)")
        else:
             logger.info("✅ AI Clients initialized")
             
    except Exception as e:
        logger.error(f"❌ AI Core verification failed: {e}")

async def verify_scheduler():
    logger.info("\n--- Verifying Agent Scheduler ---")
    try:
        from agent_scheduler import AgentScheduler
        # Pass DB config explicitly to avoid env var issues if not set
        scheduler = AgentScheduler(db_config=DB_CONFIG)
        logger.info("✅ AgentScheduler instantiated")
        status = scheduler.get_status()
        logger.info(f"✅ Scheduler status: {status}")
        scheduler.shutdown()
    except Exception as e:
        logger.error(f"❌ Scheduler verification failed: {e}")

async def verify_self_healing():
    logger.info("\n--- Verifying Self Healing ---")
    try:
        from self_healing_recovery import SelfHealingRecovery
        # We need to monkeypatch the DB config in the class or instance because it reads from env
        # But wait, the class __init__ reads env. Let's set env vars temporarily or mock.
        # Actually, let's just set the env vars for the process if possible, or trust the fallback if defaults match.
        # The file has defaults: postgres.yomagoqdmxszqtdwuhab / aws-0-us-east-2.pooler.supabase.com.
        # But password is redacted in file.
        
        # We'll set the env var for the process
        os.environ['DB_PASSWORD'] = DB_CONFIG['password']
        
        healer = SelfHealingRecovery()
        logger.info("✅ SelfHealingRecovery instantiated")
        report = healer.get_health_report()
        logger.info(f"✅ Health Report generated: {list(report.keys())}")
        
    except Exception as e:
        logger.error(f"❌ Self Healing verification failed: {e}")

async def main():
    db_ok = await verify_db()
    if db_ok:
        await verify_ai_core()
        await verify_scheduler()
        await verify_self_healing()
    else:
        logger.error("Skipping other checks due to DB failure")

if __name__ == "__main__":
    asyncio.run(main())

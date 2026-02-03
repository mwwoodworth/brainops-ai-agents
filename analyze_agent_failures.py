
import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Setup path
sys.path.append(os.getcwd())

# Load Secure Env
secure_env_path = "/home/matt-woodworth/dev/_secure/BrainOps.env"
if os.path.exists(secure_env_path):
    load_dotenv(secure_env_path, override=True)

from config import config
from database.async_connection import init_pool, close_pool, get_pool, PoolConfig

async def analyze_failures():
    print("\n🕵️ AGENT FAILURE FORENSICS 🕵️")
    
    # Init DB
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
    
    try:
        targets = ['DashboardMonitor']
        
        for agent in targets:
            print(f"\n🔍 Analyzing {agent}...")
            
            # Get Top Errors
            errors = await pool.fetch("""
                SELECT error_message, count(*) as count
                FROM ai_agent_executions
                WHERE agent_name = $1 AND status = 'failed'
                GROUP BY error_message
                ORDER BY count DESC
                LIMIT 5
            """)
            
            print("   ⚠️  Top Errors:")
            for e in errors:
                msg = e['error_message'] or "No error message logged"
                print(f"      [{e['count']}] {msg[:100]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        await close_pool()

if __name__ == "__main__":
    asyncio.run(analyze_failures())


import sys
import os
import asyncio
import logging

# Add current directory to path
sys.path.append(os.getcwd())

# Mock configuration to avoid needing full .env for this test if possible,
# or better, load the real .env if we can.
# We will try to import app and run the lifespan startup.

from dotenv import load_dotenv
load_dotenv()

async def test_startup():
    print("--- Starting Startup Diagnosis ---")
    try:
        from app import app
        from config import config
        
        print(f"Config Environment: {config.environment}")
        print(f"DB Host: {config.database.host}")
        
        # Manually trigger lifespan
        async with app.router.lifespan_context(app) as state:
            print("--- Startup Successful ---")
            print(f"State keys: {state.keys()}")
            
    except Exception as e:
        print(f"--- Startup FAILED ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(test_startup())
    except Exception as e:
        print(f"Fatal error: {e}")

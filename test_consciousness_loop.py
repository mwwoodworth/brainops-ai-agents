import asyncio
import os
from consciousness_loop import ConsciousnessLoop

# Set env vars for the module to pick up
os.environ["DB_HOST"] = "aws-0-us-east-2.pooler.supabase.com"
os.environ["DB_NAME"] = "postgres"
os.environ["DB_USER"] = "postgres.yomagoqdmxszqtdwuhab"
os.environ["DB_PASSWORD"] = "Brain0ps2O2S"
os.environ["DB_PORT"] = "5432"

async def test_loop():
    print("Initializing ConsciousnessLoop...")
    loop = ConsciousnessLoop()
    
    # Run the loop in a background task
    task = asyncio.create_task(loop.start())
    
    print("Running loop for 10 seconds...")
    await asyncio.sleep(10)
    
    print("Stopping loop...")
    loop.running = False
    await task
    
    # Verify data in DB
    print("Verifying DB records...")
    # Use pool from consciousness loop instead of direct connection
    if loop.pool:
        async with loop.pool.acquire() as conn:
            thoughts = await conn.fetchval("SELECT COUNT(*) FROM ai_thought_stream")
            vitals = await conn.fetchval("SELECT COUNT(*) FROM ai_vital_signs")
            state = await conn.fetchval("SELECT COUNT(*) FROM ai_consciousness_state")

            print(f"Thoughts recorded: {thoughts}")
            print(f"Vitals recorded: {vitals}")
            print(f"States recorded: {state}")

            if thoughts > 0 and vitals > 0:
                print("SUCCESS: Consciousness is active and recording.")
            else:
                print("FAILURE: No data recorded.")
    else:
        print("SKIP: No database pool available")

if __name__ == "__main__":
    asyncio.run(test_loop())

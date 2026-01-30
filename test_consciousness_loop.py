import asyncio
import os

if __name__ != "__main__":
    # Manual/operational script; not a unit test. Requires live DB credentials.
    import pytest

    pytest.skip("manual consciousness loop runner (not collected as a unit test)", allow_module_level=True)

from consciousness_loop import ConsciousnessLoop

# SECURITY: Load credentials from environment or .env file
# DO NOT hardcode credentials - they must be set in environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_dotenv = None  # dotenv not installed, use existing environment

# Verify required environment variables are set
required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    raise OSError(f"Required environment variables not set: {', '.join(missing)}. "
                          "Set them in .env file or environment.")

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

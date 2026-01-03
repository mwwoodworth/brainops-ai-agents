
import asyncio
import logging
import sys

# Add the directory to sys.path
sys.path.append('/home/matt-woodworth/dev/brainops-ai-agents')

from ai_operating_system import AIOperatingSystem
from ai_tracer import BrainOpsTracer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def bootstrap_schemas():
    logger.info("Bootstrapping BrainOps schemas...")

    # 1. Tracer Schema
    try:
        logger.info("Initializing Tracer schema...")
        BrainOpsTracer()
        # _ensure_schema is called in __init__
        logger.info("Tracer schema initialized.")
    except Exception as e:
        logger.error(f"Failed to init Tracer schema: {e}")

    # 2. AI OS Schema
    try:
        logger.info("Initializing AI OS schema...")
        ai_os = AIOperatingSystem()
        await ai_os._setup_database()
        logger.info("AI OS schema initialized.")
    except Exception as e:
        logger.error(f"Failed to init AI OS schema: {e}")

if __name__ == "__main__":
    asyncio.run(bootstrap_schemas())

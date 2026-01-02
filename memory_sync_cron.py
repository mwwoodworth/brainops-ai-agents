"""
Cron job for memory sync - ensures perfect sync between local and master
Runs every 5 minutes to sync bidirectionally
"""

import asyncio
import logging
import sys

from embedded_memory_system import get_embedded_memory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_sync():
    """Execute full bidirectional sync"""
    try:
        logger.info("🔄 Starting scheduled memory sync...")

        memory_system = await get_embedded_memory()

        # Sync from master (pull latest)
        await memory_system.sync_from_master()

        # Get stats
        stats = memory_system.get_stats()
        logger.info(f"📊 Memory Stats: {stats}")

        logger.info("✅ Sync completed successfully")

    except Exception as e:
        logger.error(f"❌ Sync failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_sync())

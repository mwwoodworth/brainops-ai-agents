#!/usr/bin/env bash
# Initialize the real-time state synchronization system on boot.
# Runs a full system scan and a single propagation cycle so the API
# has fresh state as soon as the service starts.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/state_sync_startup.log"

{
  echo "==> Bootstrapping Real-Time State Sync ($(date -u +"%Y-%m-%dT%H:%M:%SZ"))"

  python3 - <<'PY'
import asyncio
import logging
from datetime import datetime

from config import config
from database.async_connection import init_pool, PoolConfig, using_fallback
from realtime_state_sync import get_state_sync
from change_propagation_daemon import ChangePropagator


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("state_sync_startup")


async def main() -> None:
    db_config = config.database
    pool_config = PoolConfig(
        host=db_config.host,
        port=db_config.port,
        user=db_config.user,
        password=db_config.password,
        database=db_config.database,
        ssl=db_config.ssl,
        ssl_verify=db_config.ssl_verify,
    )
    await init_pool(pool_config)

    sync = get_state_sync()
    results = await sync.full_system_scan()
    logger.info(
        "Full system scan complete with %s components tracked (fallback=%s)",
        len(sync.state.components),
        using_fallback(),
    )

    try:
        propagator = ChangePropagator()
        propagation = await propagator.run_propagation_cycle()
        logger.info(
            "Propagation cycle finished: %s changes (%s successful, %s failed)",
            propagation.get("changes", 0),
            propagation.get("successful", 0),
            propagation.get("failed", 0),
        )
    except Exception as exc:  # Defensive guard so boot continues
        logger.warning("Change propagation skipped: %s", exc)

    logger.info("State sync initialization complete at %s", datetime.utcnow().isoformat())


asyncio.run(main())
PY

  echo "==> State sync initialization finished"
} | tee -a "$LOG_FILE"

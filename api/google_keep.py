"""
Google Keep Sync API Routes
Endpoints for managing the Keep-based Gemini Live bridge.
Auth handled by app.py SECURED_DEPENDENCIES.
"""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/google-keep",
    tags=["google-keep", "gemini-live"]
)


@router.get("/status")
async def keep_status():
    """Get Google Keep sync status."""
    try:
        from google_keep_sync import get_keep_status
        status = await get_keep_status()
        return {"success": True, **status}
    except ImportError:
        return {"success": False, "error": "google_keep_sync module not available"}
    except Exception as e:
        logger.error("Keep status failed: %s", e)
        return {"success": False, "error": str(e)}


@router.post("/sync")
async def keep_sync():
    """Trigger an immediate Keep sync."""
    try:
        from google_keep_sync import run_keep_sync
        result = await run_keep_sync()
        return result
    except ImportError:
        return {"success": False, "error": "google_keep_sync module not available"}
    except Exception as e:
        logger.error("Keep sync failed: %s", e)
        return {"success": False, "error": str(e)}

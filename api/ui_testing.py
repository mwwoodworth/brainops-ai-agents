"""
AI UI Testing API Router
=========================
Endpoints for running and managing AI-powered UI tests.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui-testing", tags=["UI Testing"])

# In-memory storage for test results (will be persisted to DB)
_test_results: Dict[str, Dict[str, Any]] = {}
_running_tests: Dict[str, bool] = {}


class TestRequest(BaseModel):
    url: str
    test_name: Optional[str] = "UI Test"


class ApplicationTestRequest(BaseModel):
    base_url: str
    routes: List[str]
    app_name: str


class ScheduledTestConfig(BaseModel):
    app_name: str  # "mrg" or "erp"
    interval_minutes: int = 60


async def _run_test_background(test_id: str, url: str, test_name: str):
    """Run a UI test in the background"""
    from ai_ui_testing import run_ui_test

    try:
        _running_tests[test_id] = True
        result = await run_ui_test(url, test_name)
        result["test_id"] = test_id
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        _test_results[test_id] = result
    except Exception as e:
        logger.error(f"Background test {test_id} failed: {e}")
        _test_results[test_id] = {
            "test_id": test_id,
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
    finally:
        _running_tests.pop(test_id, None)


async def _run_application_test_background(
    test_id: str,
    base_url: str,
    routes: List[str],
    app_name: str
):
    """Run full application test in background"""
    from ai_ui_testing import AIUITestingEngine

    try:
        _running_tests[test_id] = True
        engine = AIUITestingEngine()
        await engine.initialize()

        try:
            result = await engine.test_application(
                base_url=base_url,
                routes=routes,
                app_name=app_name
            )
            result["test_id"] = test_id
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            _test_results[test_id] = result
        finally:
            await engine.close()

    except Exception as e:
        logger.error(f"Application test {test_id} failed: {e}")
        _test_results[test_id] = {
            "test_id": test_id,
            "application": app_name,
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
    finally:
        _running_tests.pop(test_id, None)


@router.post("/test")
async def run_single_test(
    request: TestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Run a UI test on a single URL.
    Returns immediately with a test_id for polling.
    """
    import hashlib
    test_id = hashlib.md5(
        f"{request.url}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    # Start test in background
    background_tasks.add_task(
        _run_test_background,
        test_id,
        request.url,
        request.test_name or "UI Test"
    )

    return {
        "test_id": test_id,
        "status": "started",
        "url": request.url,
        "message": f"Test started. Poll /ui-testing/result/{test_id} for results."
    }


@router.post("/test/sync")
async def run_single_test_sync(request: TestRequest) -> Dict[str, Any]:
    """
    Run a UI test synchronously and wait for results.
    Use for immediate feedback (may timeout for slow pages).
    """
    from ai_ui_testing import run_ui_test

    try:
        result = await asyncio.wait_for(
            run_ui_test(request.url, request.test_name or "UI Test"),
            timeout=60.0
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail="Test timed out. Use async /test endpoint for long-running tests."
        )


@router.post("/test/application")
async def run_application_test(
    request: ApplicationTestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Run comprehensive UI tests on multiple routes of an application.
    """
    import hashlib
    test_id = hashlib.md5(
        f"{request.app_name}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    background_tasks.add_task(
        _run_application_test_background,
        test_id,
        request.base_url,
        request.routes,
        request.app_name
    )

    return {
        "test_id": test_id,
        "status": "started",
        "application": request.app_name,
        "routes_count": len(request.routes),
        "message": f"Application test started. Poll /ui-testing/result/{test_id}"
    }


@router.post("/test/mrg")
async def run_mrg_test(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Run comprehensive UI tests on MyRoofGenius.
    Tests all public routes with AI vision analysis.
    """
    from ai_ui_testing import MRG_ROUTES
    import hashlib

    test_id = hashlib.md5(
        f"mrg_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    background_tasks.add_task(
        _run_application_test_background,
        test_id,
        "https://myroofgenius.com",
        MRG_ROUTES,
        "MyRoofGenius"
    )

    return {
        "test_id": test_id,
        "status": "started",
        "application": "MyRoofGenius",
        "base_url": "https://myroofgenius.com",
        "routes_count": len(MRG_ROUTES),
        "message": f"MyRoofGenius test started. Poll /ui-testing/result/{test_id}"
    }


@router.post("/test/erp")
async def run_erp_test(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Run comprehensive UI tests on Weathercraft ERP.
    Tests all public routes with AI vision analysis.
    """
    from ai_ui_testing import ERP_ROUTES
    import hashlib

    test_id = hashlib.md5(
        f"erp_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    background_tasks.add_task(
        _run_application_test_background,
        test_id,
        "https://weathercraft-erp.vercel.app",
        ERP_ROUTES,
        "Weathercraft ERP"
    )

    return {
        "test_id": test_id,
        "status": "started",
        "application": "Weathercraft ERP",
        "base_url": "https://weathercraft-erp.vercel.app",
        "routes_count": len(ERP_ROUTES),
        "message": f"ERP test started. Poll /ui-testing/result/{test_id}"
    }


@router.get("/result/{test_id}")
async def get_test_result(test_id: str) -> Dict[str, Any]:
    """
    Get the result of a UI test by ID.
    """
    if test_id in _running_tests:
        return {
            "test_id": test_id,
            "status": "running",
            "message": "Test is still in progress"
        }

    if test_id in _test_results:
        return _test_results[test_id]

    raise HTTPException(
        status_code=404,
        detail=f"Test {test_id} not found"
    )


@router.get("/results")
async def get_all_results(
    limit: int = Query(default=50, le=200),
    app_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get all recent test results.
    """
    results = list(_test_results.values())

    if app_name:
        results = [r for r in results if r.get("application") == app_name]

    # Sort by completion time
    results.sort(
        key=lambda x: x.get("completed_at", ""),
        reverse=True
    )

    return {
        "total": len(results),
        "results": results[:limit],
        "running_tests": list(_running_tests.keys())
    }


@router.get("/status")
async def get_testing_status() -> Dict[str, Any]:
    """
    Get the current status of the UI testing system.
    """
    return {
        "status": "operational",
        "running_tests": len(_running_tests),
        "completed_tests": len(_test_results),
        "running_test_ids": list(_running_tests.keys()),
        "capabilities": [
            "Single URL testing with AI vision",
            "Full application testing",
            "Accessibility analysis (axe-core)",
            "Performance metrics collection",
            "Visual issue detection",
            "Usability scoring"
        ],
        "supported_applications": [
            {
                "name": "MyRoofGenius",
                "endpoint": "/ui-testing/test/mrg",
                "base_url": "https://myroofgenius.com"
            },
            {
                "name": "Weathercraft ERP",
                "endpoint": "/ui-testing/test/erp",
                "base_url": "https://weathercraft-erp.vercel.app"
            }
        ]
    }


@router.delete("/results")
async def clear_results() -> Dict[str, str]:
    """
    Clear all stored test results.
    """
    _test_results.clear()
    return {"message": "All test results cleared"}

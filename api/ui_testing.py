"""
AI UI Testing API Router
=========================
Endpoints for running and managing AI-powered UI tests.
Includes database persistence, scheduled testing, and health monitoring.
"""

import logging
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui-testing", tags=["UI Testing"])

# In-memory cache with database backup
_test_results: Dict[str, Dict[str, Any]] = {}
_running_tests: Dict[str, bool] = {}
_scheduled_tests: Dict[str, Dict[str, Any]] = {}

# Database configuration - MUST come from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD"),  # No default - must be set
    "port": int(os.getenv("DB_PORT", "5432"))
}


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
    enabled: bool = True


async def _ensure_tables():
    """Ensure UI testing tables exist (lazy initialization)"""
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ui_test_results (
                id SERIAL PRIMARY KEY,
                test_id VARCHAR(64) UNIQUE NOT NULL,
                application VARCHAR(128),
                url TEXT,
                status VARCHAR(32) NOT NULL,
                severity VARCHAR(32),
                message TEXT,
                ai_analysis JSONB,
                performance_metrics JSONB,
                accessibility_issues JSONB,
                suggestions JSONB,
                routes_tested INTEGER DEFAULT 0,
                issues_found INTEGER DEFAULT 0,
                started_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ui_test_app ON ui_test_results(application);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ui_test_completed ON ui_test_results(completed_at DESC);
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ui_test_schedules (
                id SERIAL PRIMARY KEY,
                app_name VARCHAR(128) UNIQUE NOT NULL,
                interval_minutes INTEGER DEFAULT 60,
                enabled BOOLEAN DEFAULT true,
                last_run_at TIMESTAMP WITH TIME ZONE,
                next_run_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        logger.info("UI testing tables ensured")
    except Exception as e:
        logger.warning(f"Could not ensure UI testing tables: {e}")


async def _persist_result(test_id: str, result: Dict[str, Any]):
    """Persist test result to database"""
    try:
        import psycopg2
        await _ensure_tables()

        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO ui_test_results (
                test_id, application, url, status, severity, message,
                ai_analysis, performance_metrics, accessibility_issues,
                suggestions, routes_tested, issues_found, completed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (test_id) DO UPDATE SET
                status = EXCLUDED.status,
                message = EXCLUDED.message,
                ai_analysis = EXCLUDED.ai_analysis,
                performance_metrics = EXCLUDED.performance_metrics,
                accessibility_issues = EXCLUDED.accessibility_issues,
                suggestions = EXCLUDED.suggestions,
                issues_found = EXCLUDED.issues_found,
                completed_at = EXCLUDED.completed_at
        """, (
            test_id,
            result.get("application"),
            result.get("url"),
            result.get("status"),
            result.get("severity"),
            result.get("message"),
            json.dumps(result.get("ai_analysis", {})),
            json.dumps(result.get("performance_metrics", {})),
            json.dumps(result.get("accessibility_issues", [])),
            json.dumps(result.get("suggestions", [])),
            result.get("routes_tested", 0),
            result.get("issues_found", 0),
            result.get("completed_at")
        ))

        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Persisted UI test result: {test_id}")
    except Exception as e:
        logger.error(f"Failed to persist test result: {e}")


async def _run_test_background(test_id: str, url: str, test_name: str):
    """Run a UI test in the background"""
    from ai_ui_testing import run_ui_test

    started_at = datetime.now(timezone.utc).isoformat()

    try:
        _running_tests[test_id] = True
        result = await run_ui_test(url, test_name)
        result["test_id"] = test_id
        result["url"] = url
        result["started_at"] = started_at
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        _test_results[test_id] = result

        # Persist to database
        await _persist_result(test_id, result)

    except Exception as e:
        logger.error(f"Background test {test_id} failed: {e}")
        result = {
            "test_id": test_id,
            "url": url,
            "status": "error",
            "error": str(e),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        _test_results[test_id] = result
        await _persist_result(test_id, result)
    finally:
        _running_tests.pop(test_id, None)


async def _run_application_test_background(
    test_id: str,
    base_url: str,
    routes: List[str],
    app_name: str,
    max_timeout_seconds: int = 300  # 5 minute max timeout for entire test
):
    """Run full application test in background with timeout protection"""
    from ai_ui_testing import AIUITestingEngine

    started_at = datetime.now(timezone.utc).isoformat()
    engine = None

    try:
        _running_tests[test_id] = True

        # Wrap entire test in timeout
        async def run_test():
            nonlocal engine
            engine = AIUITestingEngine()
            await engine.initialize(timeout_seconds=30)

            result = await engine.test_application(
                base_url=base_url,
                routes=routes,
                app_name=app_name
            )
            return result

        try:
            result = await asyncio.wait_for(run_test(), timeout=max_timeout_seconds)
            result["test_id"] = test_id
            result["started_at"] = started_at
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            result["routes_tested"] = len(routes)

            # Count issues
            issues_count = 0
            for route_result in result.get("route_results", []):
                if route_result.get("status") == "failed":
                    issues_count += 1
            result["issues_found"] = issues_count

            _test_results[test_id] = result
            await _persist_result(test_id, result)

        except asyncio.TimeoutError:
            logger.error(f"Application test {test_id} timed out after {max_timeout_seconds}s")
            result = {
                "test_id": test_id,
                "application": app_name,
                "status": "timeout",
                "error": f"Test timed out after {max_timeout_seconds} seconds",
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            _test_results[test_id] = result
            await _persist_result(test_id, result)

    except Exception as e:
        logger.error(f"Application test {test_id} failed: {e}")
        result = {
            "test_id": test_id,
            "application": app_name,
            "status": "error",
            "error": str(e),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        _test_results[test_id] = result
        await _persist_result(test_id, result)
    finally:
        _running_tests.pop(test_id, None)
        if engine:
            try:
                await engine.close()
            except Exception:
                pass


@router.post("/test")
async def run_single_test(
    request: TestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Run a UI test on a single URL.
    Returns immediately with a test_id for polling.
    """
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
            timeout=120.0  # Extended timeout
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail="Test timed out after 120s. Use async /test endpoint for long-running tests."
        )


@router.post("/test/application")
async def run_application_test(
    request: ApplicationTestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Run comprehensive UI tests on multiple routes of an application.
    """
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
    Checks in-memory cache first, then database.
    """
    # Check running tests
    if test_id in _running_tests:
        return {
            "test_id": test_id,
            "status": "running",
            "message": "Test is still in progress"
        }

    # Check in-memory cache
    if test_id in _test_results:
        return _test_results[test_id]

    # Check database
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT test_id, application, url, status, severity, message,
                   ai_analysis, performance_metrics, accessibility_issues,
                   suggestions, routes_tested, issues_found, completed_at
            FROM ui_test_results
            WHERE test_id = %s
        """, (test_id,))

        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            result = {
                "test_id": row[0],
                "application": row[1],
                "url": row[2],
                "status": row[3],
                "severity": row[4],
                "message": row[5],
                "ai_analysis": row[6] if row[6] else {},
                "performance_metrics": row[7] if row[7] else {},
                "accessibility_issues": row[8] if row[8] else [],
                "suggestions": row[9] if row[9] else [],
                "routes_tested": row[10],
                "issues_found": row[11],
                "completed_at": row[12].isoformat() if row[12] else None
            }
            # Cache it
            _test_results[test_id] = result
            return result
    except Exception as e:
        logger.warning(f"Database lookup failed: {e}")

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
    Get all recent test results from database.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        if app_name:
            cursor.execute("""
                SELECT test_id, application, url, status, severity, message,
                       routes_tested, issues_found, completed_at
                FROM ui_test_results
                WHERE application = %s
                ORDER BY completed_at DESC
                LIMIT %s
            """, (app_name, limit))
        else:
            cursor.execute("""
                SELECT test_id, application, url, status, severity, message,
                       routes_tested, issues_found, completed_at
                FROM ui_test_results
                ORDER BY completed_at DESC
                LIMIT %s
            """, (limit,))

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "test_id": row[0],
                "application": row[1],
                "url": row[2],
                "status": row[3],
                "severity": row[4],
                "message": row[5],
                "routes_tested": row[6],
                "issues_found": row[7],
                "completed_at": row[8].isoformat() if row[8] else None
            })

        return {
            "total": len(results),
            "results": results,
            "running_tests": list(_running_tests.keys())
        }
    except Exception as e:
        logger.warning(f"Database query failed, using in-memory: {e}")
        # Fallback to in-memory
        results = list(_test_results.values())
        if app_name:
            results = [r for r in results if r.get("application") == app_name]
        results.sort(key=lambda x: x.get("completed_at", ""), reverse=True)

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
    # Get total tests from database
    total_tests = len(_test_results)
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ui_test_results")
        total_tests = cursor.fetchone()[0]
        cursor.close()
        conn.close()
    except Exception:
        pass

    return {
        "status": "operational",
        "running_tests": len(_running_tests),
        "total_tests_run": total_tests,
        "running_test_ids": list(_running_tests.keys()),
        "capabilities": [
            "Single URL testing with AI vision",
            "Full application testing",
            "Accessibility analysis (axe-core)",
            "Performance metrics collection",
            "Visual issue detection",
            "Usability scoring",
            "Database persistence",
            "Scheduled testing"
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


@router.post("/schedule")
async def create_schedule(config: ScheduledTestConfig) -> Dict[str, Any]:
    """
    Create or update a scheduled test configuration.
    """
    try:
        import psycopg2
        await _ensure_tables()

        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        next_run = datetime.now(timezone.utc) + timedelta(minutes=config.interval_minutes)

        cursor.execute("""
            INSERT INTO ui_test_schedules (app_name, interval_minutes, enabled, next_run_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (app_name) DO UPDATE SET
                interval_minutes = EXCLUDED.interval_minutes,
                enabled = EXCLUDED.enabled,
                next_run_at = EXCLUDED.next_run_at,
                updated_at = NOW()
            RETURNING id
        """, (config.app_name, config.interval_minutes, config.enabled, next_run))

        schedule_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()

        _scheduled_tests[config.app_name] = {
            "interval_minutes": config.interval_minutes,
            "enabled": config.enabled,
            "next_run_at": next_run.isoformat()
        }

        return {
            "success": True,
            "schedule_id": schedule_id,
            "app_name": config.app_name,
            "interval_minutes": config.interval_minutes,
            "enabled": config.enabled,
            "next_run_at": next_run.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to create schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedules")
async def get_schedules() -> Dict[str, Any]:
    """
    Get all scheduled test configurations.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT app_name, interval_minutes, enabled, last_run_at, next_run_at
            FROM ui_test_schedules
            ORDER BY app_name
        """)

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        schedules = []
        for row in rows:
            schedules.append({
                "app_name": row[0],
                "interval_minutes": row[1],
                "enabled": row[2],
                "last_run_at": row[3].isoformat() if row[3] else None,
                "next_run_at": row[4].isoformat() if row[4] else None
            })

        return {"schedules": schedules}
    except Exception as e:
        logger.warning(f"Failed to get schedules: {e}")
        return {"schedules": list(_scheduled_tests.values())}


@router.delete("/results")
async def clear_results(keep_db: bool = False) -> Dict[str, str]:
    """
    Clear all stored test results.
    By default also clears database. Set keep_db=true to only clear cache.
    """
    _test_results.clear()

    if not keep_db:
        try:
            import psycopg2
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE ui_test_results")
            conn.commit()
            cursor.close()
            conn.close()
            return {"message": "All test results cleared (cache and database)"}
        except Exception as e:
            logger.warning(f"Failed to clear database: {e}")
            return {"message": "Cache cleared, database clear failed"}

    return {"message": "In-memory cache cleared"}


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check for the UI testing system.
    """
    db_status = "unknown"
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"

    playwright_status = "unknown"
    browser_installed = False
    try:
        from playwright.async_api import async_playwright
        playwright_status = "module_available"
        # Check if browser is actually installed
        import os
        chromium_path = os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")
        import glob
        browser_paths = glob.glob(chromium_path)
        if browser_paths:
            playwright_status = "browser_installed"
            browser_installed = True
        else:
            playwright_status = "browser_missing"
    except ImportError:
        playwright_status = "not_installed"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "playwright": playwright_status,
        "browser_installed": browser_installed,
        "running_tests": len(_running_tests),
        "running_test_ids": list(_running_tests.keys()),
        "cached_results": len(_test_results),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.post("/cancel/{test_id}")
async def cancel_test(test_id: str) -> Dict[str, Any]:
    """
    Cancel a running test and clean up its state.
    """
    if test_id in _running_tests:
        del _running_tests[test_id]
        _test_results[test_id] = {
            "test_id": test_id,
            "status": "cancelled",
            "message": "Test was cancelled by user",
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        return {"status": "cancelled", "test_id": test_id}
    elif test_id in _test_results:
        return {"status": "already_completed", "test_id": test_id}
    else:
        return {"status": "not_found", "test_id": test_id}


@router.post("/cancel-all")
async def cancel_all_tests() -> Dict[str, Any]:
    """
    Cancel all running tests.
    """
    cancelled = list(_running_tests.keys())
    for test_id in cancelled:
        del _running_tests[test_id]
        _test_results[test_id] = {
            "test_id": test_id,
            "status": "cancelled",
            "message": "Test was cancelled (bulk cancel)",
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
    return {"status": "cancelled", "count": len(cancelled), "test_ids": cancelled}


@router.get("/diagnostics")
async def get_diagnostics() -> Dict[str, Any]:
    """
    Detailed diagnostics for UI testing system.
    """
    import os
    import glob

    # Check Playwright
    playwright_info = {"installed": False, "browser_paths": []}
    try:
        from playwright.async_api import async_playwright
        playwright_info["installed"] = True

        # Check browser installation
        home = os.path.expanduser("~")
        browser_paths = glob.glob(f"{home}/.cache/ms-playwright/*/")
        playwright_info["browser_paths"] = browser_paths
        playwright_info["browser_count"] = len(browser_paths)
    except ImportError:
        playwright_info["error"] = "playwright module not installed"

    # Test browser launch capability
    browser_launch_test = {"status": "not_tested"}
    try:
        from ai_ui_testing import AIUITestingEngine
        engine = AIUITestingEngine()
        await asyncio.wait_for(engine.initialize(timeout_seconds=15), timeout=20)
        if engine._browser:
            browser_launch_test = {"status": "success", "playwright_available": True}
            await engine.close()
        else:
            browser_launch_test = {"status": "failed", "playwright_available": False, "reason": "Browser did not launch"}
    except asyncio.TimeoutError:
        browser_launch_test = {"status": "timeout", "reason": "Browser launch timed out"}
    except Exception as e:
        browser_launch_test = {"status": "error", "reason": str(e)}

    return {
        "playwright": playwright_info,
        "browser_launch_test": browser_launch_test,
        "running_tests": list(_running_tests.keys()),
        "cached_results_count": len(_test_results),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

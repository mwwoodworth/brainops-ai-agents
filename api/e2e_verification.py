"""
E2E Verification API Router
============================
API endpoints for comprehensive end-to-end system verification.
Ensures 100% operational status across ALL BrainOps systems.
"""

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/e2e", tags=["E2E Verification"])

# Background verification task
_background_verification_running = False
_last_background_result = None


def _extract_api_key(request: Request) -> str | None:
    api_key = (
        request.headers.get("X-API-Key")
        or request.headers.get("x-api-key")
        or request.headers.get("X-Api-Key")
    )
    if api_key:
        return api_key

    authorization = (
        request.headers.get("Authorization") or request.headers.get("authorization") or ""
    )
    if authorization.startswith("ApiKey "):
        return authorization[len("ApiKey ") :].strip() or None
    if authorization.startswith("Bearer "):
        return authorization[len("Bearer ") :].strip() or None
    return None


@router.get("/verify")
async def run_verification(
    request: Request,
    quick: bool = Query(False, description="Run quick health check only (critical endpoints)"),
    skip_erp: bool = Query(False, description="Exclude ERP checks (non-ERP verification scope)"),
):
    """
    Run E2E verification of ALL systems.

    Returns whether the system is 100% operational.
    Any failure in critical endpoints means the system is NOT 100% operational.

    Use ?quick=true for a faster check of critical endpoints only.
    """
    try:
        from e2e_system_verification import run_full_e2e_verification, run_quick_health_check

        api_key_override = _extract_api_key(request)

        if quick:
            result = await run_quick_health_check(
                api_key_override=api_key_override, skip_erp=skip_erp
            )
            return {
                "verification_type": "quick",
                "is_100_percent_operational": result["is_healthy"],
                "skip_erp": skip_erp,
                **result,
            }
        else:
            result = await run_full_e2e_verification(
                api_key_override=api_key_override, skip_erp=skip_erp
            )
            return {"verification_type": "full", **result}
    except Exception as e:
        logger.error(f"E2E verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}") from e


@router.post("/verify")
async def run_verification_post(
    request: Request,
    quick: bool = Query(False, description="Run quick health check only (critical endpoints)"),
    skip_erp: bool = Query(False, description="Exclude ERP checks (non-ERP verification scope)"),
):
    return await run_verification(request, quick=quick, skip_erp=skip_erp)


@router.get("/status")
async def get_verification_status():
    """Get the status of the E2E verification system"""
    try:
        from e2e_system_verification import e2e_verification

        return {
            "system": "e2e_verification",
            "status": "operational",
            "total_tests_configured": len(e2e_verification.tests),
            "categories": {
                "core_api": sum(
                    1 for t in e2e_verification.tests if t.category.value == "core_api"
                ),
                "bleeding_edge": sum(
                    1 for t in e2e_verification.tests if t.category.value == "bleeding_edge"
                ),
                "frontend": sum(
                    1 for t in e2e_verification.tests if t.category.value == "frontend"
                ),
                "mcp": sum(1 for t in e2e_verification.tests if t.category.value == "mcp"),
            },
            "critical_tests": sum(1 for t in e2e_verification.tests if t.critical),
            "last_verification": e2e_verification.get_report_summary(),
            "capabilities": [
                "full_e2e_verification",
                "quick_health_check",
                "category_breakdown",
                "actionable_recommendations",
                "100_percent_operational_check",
            ],
        }
    except Exception as e:
        logger.error(f"Could not get verification status: {e}")
        return {"system": "e2e_verification", "status": "error", "error": str(e)}


@router.get("/last-report")
async def get_last_report():
    """Get the last verification report"""
    try:
        from e2e_system_verification import get_last_verification_report

        report = await get_last_verification_report()
        if not report:
            return {
                "message": "No verification report available. Run /e2e/verify first.",
                "has_report": False,
            }

        return {"has_report": True, **report}
    except Exception as e:
        logger.error(f"Could not get last report: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health-matrix")
async def get_health_matrix():
    """
    Get a comprehensive health matrix of all systems.
    This is a quick reference view without running full verification.
    """
    try:
        from e2e_system_verification import e2e_verification

        # Group tests by category
        matrix = {}
        for test in e2e_verification.tests:
            cat = test.category.value
            if cat not in matrix:
                matrix[cat] = {"endpoints": [], "critical_count": 0, "total_count": 0}
            matrix[cat]["endpoints"].append(
                {
                    "name": test.name,
                    "url": test.url,
                    "critical": test.critical,
                    "timeout": test.timeout_seconds,
                }
            )
            matrix[cat]["total_count"] += 1
            if test.critical:
                matrix[cat]["critical_count"] += 1

        return {
            "health_matrix": matrix,
            "total_endpoints": len(e2e_verification.tests),
            "total_critical": sum(1 for t in e2e_verification.tests if t.critical),
            "message": "Use /e2e/verify to run actual verification",
        }
    except Exception as e:
        logger.error(f"Could not get health matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/verify/background")
async def start_background_verification(
    request: Request,
    background_tasks: BackgroundTasks,
    skip_erp: bool = Query(False, description="Exclude ERP checks (non-ERP verification scope)"),
):
    """Start E2E verification in the background"""
    global _background_verification_running, _last_background_result

    if _background_verification_running:
        return {
            "status": "already_running",
            "message": "Background verification is already in progress",
        }

    api_key_override = _extract_api_key(request)

    async def run_bg_verification():
        global _background_verification_running, _last_background_result
        try:
            from e2e_system_verification import run_full_e2e_verification

            _background_verification_running = True
            _last_background_result = await run_full_e2e_verification(
                api_key_override=api_key_override,
                skip_erp=skip_erp,
            )
        finally:
            _background_verification_running = False

    # Schedule background task
    background_tasks.add_task(lambda: asyncio.run(run_bg_verification()))

    return {
        "status": "started",
        "message": "Background verification started. Check /e2e/verify/background/status for results.",
    }


@router.get("/verify/background/status")
async def get_background_verification_status():
    """Get status of background verification"""
    global _background_verification_running, _last_background_result

    if _background_verification_running:
        return {"status": "running", "message": "Verification in progress..."}

    if _last_background_result:
        return {"status": "completed", "result": _last_background_result}

    return {
        "status": "not_started",
        "message": "No background verification has been run. Use POST /e2e/verify/background to start.",
    }


@router.get("/systems")
async def list_verified_systems():
    """List all systems being verified"""
    try:
        from e2e_system_verification import e2e_verification

        systems = {}
        for test in e2e_verification.tests:
            # Extract base URL
            url_parts = test.url.split("/")
            base_url = "/".join(url_parts[:3])

            if base_url not in systems:
                systems[base_url] = {"base_url": base_url, "endpoints": [], "critical_endpoints": 0}

            systems[base_url]["endpoints"].append(test.name)
            if test.critical:
                systems[base_url]["critical_endpoints"] += 1

        return {
            "systems": list(systems.values()),
            "total_systems": len(systems),
            "total_endpoints": len(e2e_verification.tests),
        }
    except Exception as e:
        logger.error(f"Could not list systems: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/debug/api-key")
async def debug_api_key():
    """Debug endpoint to check which API key the E2E module is using"""
    import os

    from e2e_system_verification import (
        API_KEY,
        BRAINOPS_API_URL,
        _api_keys_list,
        _compute_e2e_internal_sig,
    )

    api_keys_env = os.getenv("API_KEYS", "NOT SET")
    brainops_key_env = os.getenv("BRAINOPS_API_KEY", "NOT SET")

    # Compute the HMAC sig the verifier would use, for comparison
    sig = _compute_e2e_internal_sig(API_KEY) if API_KEY else "NO_KEY"

    return {
        "e2e_api_key_used": API_KEY[:10] + "..." if len(API_KEY) > 10 else API_KEY,
        "api_key_full_length": len(API_KEY),
        "api_keys_from_env": api_keys_env[:15] + "..." if len(api_keys_env) > 15 else api_keys_env,
        "api_keys_parsed_count": len(_api_keys_list),
        "brainops_api_key_env": brainops_key_env[:10] + "..."
        if len(brainops_key_env) > 10
        else brainops_key_env,
        "source": "API_KEYS" if _api_keys_list else "BRAINOPS_API_KEY or default",
        "brainops_api_url": BRAINOPS_API_URL,
        "e2e_internal_sig_first_12": sig[:12] if sig else "NONE",
        "has_compute_sig_func": True,
    }


# =============================================================================
# UI Testing with Playwright (True Browser-Based Testing)
# =============================================================================


@router.get("/ui/test/{app_name}")
async def run_ui_test(app_name: str, background_tasks: BackgroundTasks):
    """
    Run Playwright-based UI tests for a specific application.

    Available apps: weathercraft-erp, myroofgenius, brainops-command-center

    Tests what USERS actually see in their browser:
    - Page load success
    - Critical element presence
    - Performance metrics
    - Accessibility checks
    - Responsive design
    - API integration monitoring
    """
    valid_apps = ["weathercraft-erp", "myroofgenius", "brainops-command-center"]
    if app_name not in valid_apps:
        raise HTTPException(status_code=400, detail=f"Invalid app. Valid options: {valid_apps}")

    try:
        from ui_tester_agent import PLAYWRIGHT_AVAILABLE, UITesterAgent

        if not PLAYWRIGHT_AVAILABLE:
            return {
                "status": "error",
                "error": "Playwright not installed. Browser testing unavailable.",
                "recommendation": "Use /e2e/verify for API-level testing instead",
            }

        tester = UITesterAgent()
        results = await tester.run_full_test_suite(app_name)

        return {
            "test_type": "ui_browser",
            "application": app_name,
            "is_operational": results["summary"]["failed"] == 0
            and results["summary"]["errors"] == 0,
            "results": results,
        }
    except Exception as e:
        logger.error(f"UI test failed for {app_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/ui/test-all")
async def run_all_ui_tests():
    """
    Run UI tests for ALL configured applications.

    Tests: weathercraft-erp, myroofgenius, brainops-command-center
    """
    try:
        from ui_tester_agent import PLAYWRIGHT_AVAILABLE, UITesterAgent

        if not PLAYWRIGHT_AVAILABLE:
            return {
                "status": "error",
                "error": "Playwright not available",
                "recommendation": "Install with: pip install playwright && playwright install chromium",
            }

        tester = UITesterAgent()
        all_results = {}
        overall_pass = True

        for app_name in tester.test_urls.keys():
            try:
                results = await tester.run_full_test_suite(app_name)
                all_results[app_name] = results
                if results["summary"]["failed"] > 0 or results["summary"]["errors"] > 0:
                    overall_pass = False
            except Exception as e:
                all_results[app_name] = {"error": str(e)}
                overall_pass = False

        return {
            "test_type": "ui_browser_full",
            "all_apps_pass": overall_pass,
            "results": all_results,
            "summary": {
                "total_apps": len(all_results),
                "apps_passed": sum(
                    1 for r in all_results.values() if r.get("summary", {}).get("failed", 1) == 0
                ),
                "apps_failed": sum(
                    1
                    for r in all_results.values()
                    if r.get("summary", {}).get("failed", 0) > 0 or "error" in r
                ),
            },
        }
    except Exception as e:
        logger.error(f"All UI tests failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/ui/status")
async def get_ui_test_status():
    """Check if UI testing (Playwright) is available"""
    try:
        from ui_tester_agent import PLAYWRIGHT_AVAILABLE, UITesterAgent

        if not PLAYWRIGHT_AVAILABLE:
            return {
                "ui_testing_available": False,
                "reason": "Playwright not installed",
                "install_command": "pip install playwright && playwright install chromium",
            }

        tester = UITesterAgent()
        return {
            "ui_testing_available": True,
            "configured_apps": list(tester.test_urls.keys()),
            "test_types": [
                "page_load",
                "element_presence",
                "interaction",
                "responsive_design",
                "performance",
                "accessibility",
                "api_integration",
            ],
            "endpoints": {"test_single": "/e2e/ui/test/{app_name}", "test_all": "/e2e/ui/test-all"},
        }
    except Exception as e:
        logger.error(f"UI status check failed: {e}")
        return {"ui_testing_available": False, "error": str(e)}


@router.get("/ui/visual-ai/{app_name}")
async def run_visual_ai_analysis(app_name: str):
    """
    Run Gemini Vision-powered visual analysis on a specific app.

    This uses AI to analyze:
    - Visual design quality
    - UX best practices
    - Accessibility issues
    - Brand consistency
    - Content quality
    - Performance indicators

    Returns detailed scores, issues, and recommendations.
    """
    valid_apps = ["weathercraft-erp", "myroofgenius", "brainops-command-center"]
    if app_name not in valid_apps:
        raise HTTPException(status_code=400, detail=f"Invalid app. Valid options: {valid_apps}")

    try:
        from ui_tester_agent import GEMINI_VISION_AVAILABLE, PLAYWRIGHT_AVAILABLE, UITesterAgent

        if not PLAYWRIGHT_AVAILABLE:
            return {"status": "error", "error": "Playwright not available"}

        if not GEMINI_VISION_AVAILABLE:
            return {
                "status": "error",
                "error": "Gemini Vision not available (GOOGLE_API_KEY not set)",
                "recommendation": "Set GOOGLE_API_KEY environment variable",
            }

        tester = UITesterAgent()
        await tester.setup_browser(headless=True)

        app_url = tester.test_urls[app_name]["base_url"]
        result = await tester.test_visual_ai_analysis(app_url, app_name)

        await tester.teardown_browser()

        return {
            "test_type": "visual_ai_analysis",
            "application": app_name,
            "powered_by": "gemini-1.5-pro-002",
            "results": result,
        }
    except Exception as e:
        logger.error(f"Visual AI analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/ui/visual-ai-all")
async def run_all_visual_ai_analysis():
    """
    Run Gemini Vision analysis on ALL configured applications.
    Returns comprehensive AI insights for the entire frontend ecosystem.
    """
    try:
        from ui_tester_agent import GEMINI_VISION_AVAILABLE, PLAYWRIGHT_AVAILABLE, UITesterAgent

        if not PLAYWRIGHT_AVAILABLE or not GEMINI_VISION_AVAILABLE:
            return {
                "status": "error",
                "playwright_available": PLAYWRIGHT_AVAILABLE,
                "gemini_vision_available": GEMINI_VISION_AVAILABLE,
            }

        tester = UITesterAgent()
        await tester.setup_browser(headless=True)

        all_results = {}
        overall_scores = []

        for app_name, config in tester.test_urls.items():
            try:
                result = await tester.test_visual_ai_analysis(config["base_url"], app_name)
                all_results[app_name] = result
                if result.get("overall_score"):
                    overall_scores.append(result["overall_score"])
            except Exception as e:
                all_results[app_name] = {"error": str(e)}

        await tester.teardown_browser()

        return {
            "test_type": "visual_ai_analysis_full",
            "powered_by": "gemini-1.5-pro-002",
            "results": all_results,
            "ecosystem_health": {
                "average_score": sum(overall_scores) / len(overall_scores) if overall_scores else 0,
                "apps_analyzed": len(all_results),
                "apps_passing": sum(1 for r in all_results.values() if r.get("status") == "pass"),
            },
        }
    except Exception as e:
        logger.error(f"All visual AI analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/complete-verification")
async def complete_verification(
    skip_erp: bool = Query(False, description="Exclude ERP checks (non-ERP verification scope)")
):
    """
    Run COMPLETE verification: API + UI tests combined.

    This is the ultimate 100% operational check:
    1. API endpoint verification (all backend services)
    2. UI browser testing (what users actually see)
    """
    results = {
        "verification_type": "complete",
        "api_verification": None,
        "ui_verification": None,
        "is_100_percent_operational": False,
    }

    # Run API verification
    try:
        from e2e_system_verification import run_full_e2e_verification

        results["api_verification"] = await run_full_e2e_verification(skip_erp=skip_erp)
    except Exception as e:
        results["api_verification"] = {"error": str(e), "is_100_percent_operational": False}

    # Run UI verification
    try:
        from ui_tester_agent import PLAYWRIGHT_AVAILABLE, UITesterAgent

        if PLAYWRIGHT_AVAILABLE:
            tester = UITesterAgent()
            ui_results = {}
            ui_pass = True

            for app_name in tester.test_urls.keys():
                if skip_erp and app_name == "weathercraft-erp":
                    continue
                try:
                    app_results = await tester.run_full_test_suite(app_name)
                    ui_results[app_name] = app_results["summary"]
                    if app_results["summary"]["failed"] > 0 or app_results["summary"]["errors"] > 0:
                        ui_pass = False
                except Exception as e:
                    ui_results[app_name] = {"error": str(e)}
                    ui_pass = False

            results["ui_verification"] = {
                "available": True,
                "all_pass": ui_pass,
                "apps": ui_results,
            }
        else:
            results["ui_verification"] = {"available": False, "reason": "Playwright not installed"}
    except Exception as e:
        results["ui_verification"] = {"error": str(e), "available": False}

    # Determine overall status
    api_ok = (
        results["api_verification"].get("is_100_percent_operational", False)
        if isinstance(results["api_verification"], dict)
        else False
    )
    ui_ok = (
        results["ui_verification"].get("all_pass", False)
        if isinstance(results["ui_verification"], dict)
        else False
    )

    results["is_100_percent_operational"] = api_ok and ui_ok
    results["summary"] = {
        "api_ok": api_ok,
        "ui_ok": ui_ok,
        "recommendation": "System fully operational"
        if results["is_100_percent_operational"]
        else "Check failed tests for issues",
    }

    return results

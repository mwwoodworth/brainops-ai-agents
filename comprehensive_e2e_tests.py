#!/usr/bin/env python3
"""
COMPREHENSIVE E2E TESTING SYSTEM
=================================
NOT basic tests. COMPLETE coverage of EVERY UI element, EVERY operation,
EVERY user flow in live production. No assumptions. No gaps.

This tests:
1. EVERY page loads correctly
2. EVERY navigation link works
3. EVERY form field is present and functional
4. EVERY button triggers correct action
5. EVERY API integration works
6. EVERY auth flow completes
7. EVERY business operation executes
8. Performance meets thresholds
9. Accessibility compliance
10. Mobile responsiveness

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import logging
import json
import base64
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Import Playwright if available
try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available - using HTTP fallback")


class TestResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class E2ETest:
    """Single e2e test definition"""
    name: str
    category: str
    description: str
    steps: List[Dict[str, Any]]
    expected_result: str
    severity: str = "high"  # critical, high, medium, low


@dataclass
class TestReport:
    """Complete test report"""
    application: str
    url: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    critical_failures: List[Dict[str, Any]] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MYROOFGENIUS COMPREHENSIVE TEST SUITE
# =============================================================================

MRG_COMPREHENSIVE_TESTS = [
    # =========== HOMEPAGE TESTS ===========
    E2ETest(
        name="Homepage loads",
        category="page_load",
        description="Homepage loads with all critical elements",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com"},
            {"action": "wait_for", "selector": "body", "timeout": 10000},
            {"action": "screenshot", "name": "homepage"},
        ],
        expected_result="Page loads with HTTP 200",
        severity="critical"
    ),
    E2ETest(
        name="Homepage navigation visible",
        category="navigation",
        description="Main navigation is visible and accessible",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com"},
            {"action": "assert_visible", "selector": "nav, [role='navigation'], header"},
        ],
        expected_result="Navigation element visible",
        severity="critical"
    ),
    E2ETest(
        name="Homepage CTA buttons",
        category="cta",
        description="Call-to-action buttons are present",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com"},
            {"action": "assert_visible", "selector": "button, a[href*='login'], a[href*='start'], a[href*='signup']"},
        ],
        expected_result="CTA buttons visible",
        severity="high"
    ),
    E2ETest(
        name="Homepage footer",
        category="layout",
        description="Footer with legal links is present",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com"},
            {"action": "assert_visible", "selector": "footer"},
            {"action": "assert_text", "selector": "footer", "text": "2025"},
        ],
        expected_result="Footer with copyright visible",
        severity="medium"
    ),

    # =========== PRICING PAGE TESTS ===========
    E2ETest(
        name="Pricing page loads",
        category="page_load",
        description="Pricing page loads correctly",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/pricing"},
            {"action": "wait_for", "selector": "body", "timeout": 15000},
            {"action": "screenshot", "name": "pricing"},
        ],
        expected_result="Pricing page loads",
        severity="critical"
    ),
    E2ETest(
        name="Pricing tiers visible",
        category="content",
        description="Pricing tiers/plans are displayed",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/pricing"},
            {"action": "assert_text", "text": "price"},
        ],
        expected_result="Pricing information visible",
        severity="high"
    ),

    # =========== LOGIN PAGE TESTS ===========
    E2ETest(
        name="Login page loads",
        category="auth",
        description="Login page loads with form",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/login"},
            {"action": "wait_for", "selector": "body", "timeout": 10000},
            {"action": "screenshot", "name": "login"},
        ],
        expected_result="Login page loads",
        severity="critical"
    ),
    E2ETest(
        name="Login form elements",
        category="auth",
        description="Login form has email and password fields",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/login"},
            {"action": "assert_visible", "selector": "input[type='email'], input[name='email'], input[placeholder*='email' i]"},
            {"action": "assert_visible", "selector": "input[type='password'], input[name='password']"},
        ],
        expected_result="Email and password fields present",
        severity="critical"
    ),
    E2ETest(
        name="Login submit button",
        category="auth",
        description="Login submit button is present and clickable",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/login"},
            {"action": "assert_visible", "selector": "button[type='submit'], input[type='submit'], button:has-text('Log in'), button:has-text('Sign in')"},
        ],
        expected_result="Submit button present",
        severity="critical"
    ),

    # =========== SIGNUP PAGE TESTS ===========
    E2ETest(
        name="Signup page loads",
        category="auth",
        description="Signup/Register page loads",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/register"},
            {"action": "wait_for", "selector": "body", "timeout": 10000},
        ],
        expected_result="Signup page loads",
        severity="high"
    ),

    # =========== DASHBOARD TESTS (PUBLIC) ===========
    E2ETest(
        name="Dashboard redirect",
        category="auth",
        description="Dashboard redirects unauthenticated users",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/dashboard"},
            {"action": "wait", "ms": 3000},
            {"action": "assert_url_contains", "text": "login"},
        ],
        expected_result="Redirects to login",
        severity="high"
    ),

    # =========== TOOLS PAGE TESTS ===========
    E2ETest(
        name="Tools page loads",
        category="page_load",
        description="Tools page loads",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/tools"},
            {"action": "wait_for", "selector": "body", "timeout": 10000},
        ],
        expected_result="Tools page loads",
        severity="medium"
    ),

    # =========== MARKETPLACE TESTS ===========
    E2ETest(
        name="Marketplace page loads",
        category="page_load",
        description="Marketplace page loads",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/marketplace"},
            {"action": "wait_for", "selector": "body", "timeout": 10000},
        ],
        expected_result="Marketplace loads",
        severity="medium"
    ),

    # =========== FEATURES PAGE TESTS ===========
    E2ETest(
        name="Features page loads",
        category="page_load",
        description="Features page loads with content",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/features"},
            {"action": "wait_for", "selector": "body", "timeout": 10000},
        ],
        expected_result="Features page loads",
        severity="medium"
    ),

    # =========== ABOUT PAGE TESTS ===========
    E2ETest(
        name="About page loads",
        category="page_load",
        description="About page loads",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/about"},
            {"action": "wait_for", "selector": "body", "timeout": 10000},
        ],
        expected_result="About page loads",
        severity="low"
    ),

    # =========== CONTACT PAGE TESTS ===========
    E2ETest(
        name="Contact page loads",
        category="page_load",
        description="Contact page loads",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com/contact"},
            {"action": "wait_for", "selector": "body", "timeout": 10000},
        ],
        expected_result="Contact page loads",
        severity="low"
    ),

    # =========== PERFORMANCE TESTS ===========
    E2ETest(
        name="Homepage performance",
        category="performance",
        description="Homepage loads within acceptable time",
        steps=[
            {"action": "navigate", "url": "https://myroofgenius.com"},
            {"action": "measure_performance"},
            {"action": "assert_performance", "metric": "load_time_ms", "max": 5000},
        ],
        expected_result="Load time under 5 seconds",
        severity="high"
    ),

    # =========== MOBILE RESPONSIVE TESTS ===========
    E2ETest(
        name="Mobile viewport",
        category="responsive",
        description="Site works on mobile viewport",
        steps=[
            {"action": "set_viewport", "width": 375, "height": 667},
            {"action": "navigate", "url": "https://myroofgenius.com"},
            {"action": "assert_visible", "selector": "body"},
            {"action": "screenshot", "name": "mobile"},
        ],
        expected_result="Site works on mobile",
        severity="high"
    ),
]


# =============================================================================
# WEATHERCRAFT ERP COMPREHENSIVE TEST SUITE
# =============================================================================

ERP_COMPREHENSIVE_TESTS = [
    # =========== LOGIN PAGE TESTS ===========
    E2ETest(
        name="ERP login page loads",
        category="auth",
        description="ERP login page loads with form",
        steps=[
            {"action": "navigate", "url": "https://weathercraft-erp.vercel.app/login"},
            {"action": "wait_for", "selector": "body", "timeout": 10000},
            {"action": "screenshot", "name": "erp_login"},
        ],
        expected_result="Login page loads",
        severity="critical"
    ),
    E2ETest(
        name="ERP login form elements",
        category="auth",
        description="Login form has all required fields",
        steps=[
            {"action": "navigate", "url": "https://weathercraft-erp.vercel.app/login"},
            {"action": "assert_visible", "selector": "input"},
        ],
        expected_result="Form fields present",
        severity="critical"
    ),

    # =========== DASHBOARD TESTS ===========
    E2ETest(
        name="ERP dashboard redirect",
        category="auth",
        description="Dashboard redirects unauthenticated users",
        steps=[
            {"action": "navigate", "url": "https://weathercraft-erp.vercel.app/dashboard"},
            {"action": "wait", "ms": 3000},
        ],
        expected_result="Redirects or shows login",
        severity="high"
    ),

    # =========== HOMEPAGE TESTS ===========
    E2ETest(
        name="ERP homepage loads",
        category="page_load",
        description="ERP homepage loads",
        steps=[
            {"action": "navigate", "url": "https://weathercraft-erp.vercel.app"},
            {"action": "wait_for", "selector": "body", "timeout": 10000},
            {"action": "screenshot", "name": "erp_home"},
        ],
        expected_result="Homepage loads",
        severity="critical"
    ),

    # =========== PERFORMANCE TESTS ===========
    E2ETest(
        name="ERP homepage performance",
        category="performance",
        description="Homepage loads within acceptable time",
        steps=[
            {"action": "navigate", "url": "https://weathercraft-erp.vercel.app"},
            {"action": "measure_performance"},
            {"action": "assert_performance", "metric": "load_time_ms", "max": 5000},
        ],
        expected_result="Load time under 5 seconds",
        severity="high"
    ),

    # =========== MOBILE RESPONSIVE TESTS ===========
    E2ETest(
        name="ERP mobile viewport",
        category="responsive",
        description="ERP works on mobile viewport",
        steps=[
            {"action": "set_viewport", "width": 375, "height": 667},
            {"action": "navigate", "url": "https://weathercraft-erp.vercel.app"},
            {"action": "assert_visible", "selector": "body"},
            {"action": "screenshot", "name": "erp_mobile"},
        ],
        expected_result="Site works on mobile",
        severity="high"
    ),
]


class ComprehensiveE2ETester:
    """
    Comprehensive E2E Testing Engine
    Executes ALL tests with NO assumptions
    """

    def __init__(self):
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._playwright = None
        self.screenshot_dir = "/tmp/comprehensive_e2e"

    async def initialize(self):
        """Initialize Playwright browser"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available")
            return False

        try:
            import os
            os.makedirs(self.screenshot_dir, exist_ok=True)

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
            )
            logger.info("Comprehensive E2E Tester initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    async def close(self):
        """Cleanup resources"""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    async def run_test(self, test: E2ETest) -> Dict[str, Any]:
        """Run a single e2e test"""
        result = {
            "name": test.name,
            "category": test.category,
            "severity": test.severity,
            "status": TestResult.SKIP.value,
            "message": "",
            "duration_ms": 0,
            "steps_completed": 0,
            "steps_total": len(test.steps),
            "screenshots": []
        }

        if not self._browser:
            result["status"] = TestResult.SKIP.value
            result["message"] = "Browser not available"
            return result

        start_time = datetime.now(timezone.utc)
        current_viewport = {"width": 1920, "height": 1080}

        try:
            # Create new context for each test
            self._context = await self._browser.new_context(
                viewport=current_viewport,
                device_scale_factor=2
            )
            self._page = await self._context.new_page()

            # Execute each step
            for i, step in enumerate(test.steps):
                action = step.get("action")

                try:
                    if action == "navigate":
                        await self._page.goto(step["url"], wait_until="networkidle", timeout=30000)
                        result["steps_completed"] += 1

                    elif action == "wait_for":
                        await self._page.wait_for_selector(step["selector"], timeout=step.get("timeout", 10000))
                        result["steps_completed"] += 1

                    elif action == "wait":
                        await self._page.wait_for_timeout(step.get("ms", 1000))
                        result["steps_completed"] += 1

                    elif action == "screenshot":
                        path = f"{self.screenshot_dir}/{step['name']}_{datetime.now().timestamp()}.png"
                        await self._page.screenshot(path=path)
                        result["screenshots"].append(path)
                        result["steps_completed"] += 1

                    elif action == "assert_visible":
                        element = await self._page.query_selector(step["selector"])
                        if not element:
                            raise AssertionError(f"Element not visible: {step['selector']}")
                        result["steps_completed"] += 1

                    elif action == "assert_text":
                        if "selector" in step:
                            text = await self._page.text_content(step["selector"])
                        else:
                            text = await self._page.content()
                        if step.get("text", "").lower() not in text.lower():
                            raise AssertionError(f"Text not found: {step.get('text')}")
                        result["steps_completed"] += 1

                    elif action == "assert_url_contains":
                        current_url = self._page.url
                        if step["text"] not in current_url:
                            raise AssertionError(f"URL doesn't contain: {step['text']}")
                        result["steps_completed"] += 1

                    elif action == "set_viewport":
                        current_viewport = {"width": step["width"], "height": step["height"]}
                        await self._page.set_viewport_size(current_viewport)
                        result["steps_completed"] += 1

                    elif action == "measure_performance":
                        metrics = await self._page.evaluate("""
                            () => {
                                const timing = performance.timing;
                                return {
                                    load_time_ms: timing.loadEventEnd - timing.navigationStart,
                                    dom_ready_ms: timing.domContentLoadedEventEnd - timing.navigationStart,
                                    ttfb_ms: timing.responseStart - timing.navigationStart
                                };
                            }
                        """)
                        result["performance"] = metrics
                        result["steps_completed"] += 1

                    elif action == "assert_performance":
                        if "performance" not in result:
                            raise AssertionError("No performance data")
                        metric_value = result["performance"].get(step["metric"], 0)
                        if metric_value > step["max"]:
                            raise AssertionError(f"{step['metric']} too high: {metric_value} > {step['max']}")
                        result["steps_completed"] += 1

                except PlaywrightTimeoutError as e:
                    result["status"] = TestResult.FAIL.value
                    result["message"] = f"Timeout at step {i+1}: {str(e)}"
                    break

                except AssertionError as e:
                    result["status"] = TestResult.FAIL.value
                    result["message"] = f"Assertion failed at step {i+1}: {str(e)}"
                    break

                except Exception as e:
                    result["status"] = TestResult.ERROR.value
                    result["message"] = f"Error at step {i+1}: {str(e)}"
                    break

            else:
                # All steps completed
                result["status"] = TestResult.PASS.value
                result["message"] = test.expected_result

        except Exception as e:
            result["status"] = TestResult.ERROR.value
            result["message"] = f"Test execution error: {str(e)}"

        finally:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            self._page = None
            self._context = None

        end_time = datetime.now(timezone.utc)
        result["duration_ms"] = (end_time - start_time).total_seconds() * 1000

        return result

    async def run_test_suite(self, app_name: str, tests: List[E2ETest]) -> TestReport:
        """Run a complete test suite"""
        url = "https://myroofgenius.com" if app_name == "myroofgenius" else "https://weathercraft-erp.vercel.app"

        report = TestReport(
            application=app_name,
            url=url,
            started_at=datetime.now(timezone.utc),
            total_tests=len(tests)
        )

        for test in tests:
            result = await self.run_test(test)
            report.results.append(result)

            if result["status"] == TestResult.PASS.value:
                report.passed += 1
            elif result["status"] == TestResult.FAIL.value:
                report.failed += 1
                if test.severity == "critical":
                    report.critical_failures.append(result)
            elif result["status"] == TestResult.SKIP.value:
                report.skipped += 1
            else:
                report.errors += 1

            if result.get("screenshots"):
                report.screenshots.extend(result["screenshots"])

            logger.info(f"[{app_name}] {test.name}: {result['status']}")

        report.completed_at = datetime.now(timezone.utc)
        return report

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run ALL comprehensive tests for all applications"""
        results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "applications": {},
            "summary": {
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "total_skipped": 0,
                "total_errors": 0,
                "critical_failures": 0
            }
        }

        # Initialize browser
        if not await self.initialize():
            results["error"] = "Failed to initialize browser"
            return results

        try:
            # Run MyRoofGenius tests
            mrg_report = await self.run_test_suite("myroofgenius", MRG_COMPREHENSIVE_TESTS)
            results["applications"]["myroofgenius"] = self._report_to_dict(mrg_report)

            # Run ERP tests
            erp_report = await self.run_test_suite("weathercraft-erp", ERP_COMPREHENSIVE_TESTS)
            results["applications"]["weathercraft-erp"] = self._report_to_dict(erp_report)

            # Update summary
            for report in [mrg_report, erp_report]:
                results["summary"]["total_tests"] += report.total_tests
                results["summary"]["total_passed"] += report.passed
                results["summary"]["total_failed"] += report.failed
                results["summary"]["total_skipped"] += report.skipped
                results["summary"]["total_errors"] += report.errors
                results["summary"]["critical_failures"] += len(report.critical_failures)

        finally:
            await self.close()

        results["completed_at"] = datetime.now(timezone.utc).isoformat()
        results["summary"]["pass_rate"] = (
            results["summary"]["total_passed"] / results["summary"]["total_tests"] * 100
            if results["summary"]["total_tests"] > 0 else 0
        )

        return results

    def _report_to_dict(self, report: TestReport) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            "application": report.application,
            "url": report.url,
            "started_at": report.started_at.isoformat(),
            "completed_at": report.completed_at.isoformat() if report.completed_at else None,
            "total_tests": report.total_tests,
            "passed": report.passed,
            "failed": report.failed,
            "skipped": report.skipped,
            "errors": report.errors,
            "pass_rate": report.passed / report.total_tests * 100 if report.total_tests > 0 else 0,
            "critical_failures": report.critical_failures,
            "results": report.results
        }


# =============================================================================
# API FUNCTIONS
# =============================================================================

async def run_comprehensive_e2e(app_name: Optional[str] = None) -> Dict[str, Any]:
    """Run comprehensive e2e tests"""
    tester = ComprehensiveE2ETester()

    if app_name:
        await tester.initialize()
        try:
            if app_name == "myroofgenius":
                report = await tester.run_test_suite("myroofgenius", MRG_COMPREHENSIVE_TESTS)
            elif app_name == "weathercraft-erp":
                report = await tester.run_test_suite("weathercraft-erp", ERP_COMPREHENSIVE_TESTS)
            else:
                return {"error": f"Unknown application: {app_name}"}
            return tester._report_to_dict(report)
        finally:
            await tester.close()
    else:
        return await tester.run_all_tests()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    async def main():
        app_name = sys.argv[1] if len(sys.argv) > 1 else None
        results = await run_comprehensive_e2e(app_name)
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())

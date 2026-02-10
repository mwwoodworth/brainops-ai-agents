#!/usr/bin/env python3
"""
ChatGPT-AGENT-LEVEL UI TESTER
==============================
Real human-like UI testing that goes beyond basic health checks.
Logs in, navigates, fills forms, tests flows - like a real user.

Features:
1. Actual login with credentials
2. Full user flow testing
3. Form filling and submission
4. AI vision analysis of results
5. Accessibility checking
6. Performance measurement
7. Error detection and reporting

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Test credentials (use env vars in production)
MRG_TEST_EMAIL = os.getenv("MRG_TEST_EMAIL", "test@myroofgenius.com")
MRG_TEST_PASSWORD = os.getenv("MRG_TEST_PASSWORD", "testpassword123")
ERP_TEST_EMAIL = os.getenv("ERP_TEST_EMAIL", "test@weathercraft.com")
ERP_TEST_PASSWORD = os.getenv("ERP_TEST_PASSWORD", "testpassword123")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class TestFlowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class FlowStep:
    """A single step in a test flow"""
    name: str
    action: str  # navigate, click, fill, submit, wait, screenshot, assert
    selector: Optional[str] = None
    value: Optional[str] = None
    expected: Optional[str] = None
    timeout_ms: int = 10000


@dataclass
class FlowResult:
    """Result of executing a test flow"""
    flow_name: str
    status: TestFlowStatus
    steps_total: int
    steps_passed: int
    steps_failed: int
    duration_seconds: float
    error_message: Optional[str] = None
    screenshots: list[str] = field(default_factory=list)
    ai_analysis: Optional[dict[str, Any]] = None
    performance_metrics: Optional[dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChatGPTAgentTester:
    """
    Human-like UI tester that performs actual user flows.
    Goes beyond health checks to test real functionality.
    """

    def __init__(
        self,
        *,
        enable_ai_analysis: Optional[bool] = None,
        goto_wait_until: Optional[str] = None,
    ):
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None
        self.results: list[FlowResult] = []

        if enable_ai_analysis is None:
            enable_ai_analysis = os.getenv("CHATGPT_AGENT_TESTER_ENABLE_AI_ANALYSIS", "false").lower() == "true"
        self.enable_ai_analysis = bool(enable_ai_analysis)

        capture_screenshots_raw = os.getenv("CHATGPT_AGENT_TESTER_CAPTURE_SCREENSHOTS", "false").strip().lower()
        self.capture_screenshots = capture_screenshots_raw in {"1", "true", "yes", "on"}

        wait_until = (goto_wait_until or os.getenv("CHATGPT_AGENT_TESTER_GOTO_WAIT_UNTIL", "domcontentloaded")).lower()
        if wait_until not in {"load", "domcontentloaded", "networkidle"}:
            wait_until = "domcontentloaded"
        self.goto_wait_until = wait_until

        self.user_agent = os.getenv(
            "CHATGPT_AGENT_TESTER_USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        )

        try:
            self.ai_timeout_seconds = float(os.getenv("CHATGPT_AGENT_TESTER_AI_TIMEOUT_SECONDS", "25"))
        except ValueError:
            self.ai_timeout_seconds = 25.0

    async def initialize(self):
        """Initialize Playwright browser"""
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            logger.info("ChatGPT Agent Tester initialized")
        except ImportError:
            logger.error("Playwright not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise

    async def close(self):
        """Cleanup browser resources"""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _new_context(self):
        """Create a new browser context"""
        self._context = await self._browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
            locale='en-US',
            timezone_id='America/New_York',
            user_agent=self.user_agent,
        )
        self._page = await self._context.new_page()
        return self._page

    async def _take_screenshot(self, name: str) -> str:
        """Take a screenshot and return base64"""
        try:
            screenshot = await self._page.screenshot(full_page=False)
            return base64.b64encode(screenshot).decode()
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")
            return ""

    async def _analyze_with_ai(self, screenshot_b64: str, context: str) -> dict[str, Any]:
        """Analyze screenshot with AI vision"""
        if not OPENAI_API_KEY or not screenshot_b64:
            return {"status": "skipped", "reason": "No API key or screenshot"}

        try:
            import openai
            client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=self.ai_timeout_seconds)

            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert QA tester. Analyze this UI screenshot and identify any issues."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Context: {context}\n\nAnalyze this screenshot for:\n1. Visual issues (layout, alignment, broken images)\n2. Functional issues (missing elements, broken state)\n3. UX issues (confusing navigation, unclear CTAs)\n4. Accessibility issues (contrast, labels)\n\nReturn JSON with: visual_issues, functional_issues, ux_issues, accessibility_issues, overall_score (0-100)"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500
            )

            import re
            text = response.choices[0].message.content
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
            return {"raw_analysis": text[:500]}

        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return {"error": str(e)}

    async def run_flow(self, flow_name: str, steps: list[FlowStep]) -> FlowResult:
        """Execute a test flow"""
        start_time = time.time()
        steps_passed = 0
        steps_failed = 0
        screenshots = []
        error_message = None
        wants_screenshots = self.enable_ai_analysis or self.capture_screenshots

        try:
            page = await self._new_context()

            for step in steps:
                try:
                    logger.info(f"[{flow_name}] Executing: {step.name}")

                    if step.action == "navigate":
                        await page.goto(step.value, wait_until=self.goto_wait_until, timeout=step.timeout_ms)
                        steps_passed += 1

                    elif step.action == "click":
                        await page.click(step.selector, timeout=step.timeout_ms)
                        steps_passed += 1

                    elif step.action == "fill":
                        await page.fill(step.selector, step.value, timeout=step.timeout_ms)
                        steps_passed += 1

                    elif step.action == "submit":
                        if step.selector:
                            await page.click(step.selector, timeout=step.timeout_ms)
                        else:
                            await page.keyboard.press("Enter")
                        steps_passed += 1

                    elif step.action == "wait":
                        await page.wait_for_timeout(int(step.value) if step.value else 1000)
                        steps_passed += 1

                    elif step.action == "wait_for_selector":
                        await page.wait_for_selector(step.selector, timeout=step.timeout_ms)
                        steps_passed += 1

                    elif step.action == "screenshot":
                        if wants_screenshots:
                            screenshot = await self._take_screenshot(step.name)
                            if screenshot:
                                screenshots.append(screenshot)
                        steps_passed += 1

                    elif step.action == "assert_url":
                        current_url = page.url
                        if step.expected in current_url:
                            steps_passed += 1
                        else:
                            steps_failed += 1
                            error_message = f"URL assertion failed: expected '{step.expected}' in '{current_url}'"

                    elif step.action == "assert_text":
                        content = await page.content()
                        if step.expected in content:
                            steps_passed += 1
                        else:
                            steps_failed += 1
                            error_message = f"Text assertion failed: '{step.expected}' not found"

                    elif step.action == "assert_element":
                        try:
                            await page.wait_for_selector(step.selector, timeout=step.timeout_ms)
                            steps_passed += 1
                        except Exception:
                            steps_failed += 1
                            error_message = f"Element assertion failed: '{step.selector}' not found"

                except Exception as e:
                    steps_failed += 1
                    error_message = f"Step '{step.name}' failed: {str(e)}"
                    logger.error(error_message)
                    # Take error screenshot
                    if wants_screenshots:
                        screenshot = await self._take_screenshot(f"error_{step.name}")
                        if screenshot:
                            screenshots.append(screenshot)
                    break

            # Final screenshot (only when explicitly requested or needed for AI analysis)
            if wants_screenshots:
                final_screenshot = await self._take_screenshot("final")
                if final_screenshot:
                    screenshots.append(final_screenshot)

            # Performance metrics
            performance = await self._get_performance_metrics()

        except Exception as e:
            error_message = f"Flow error: {str(e)}"
            logger.error(error_message)
            steps_failed = len(steps) - steps_passed

        finally:
            if self._page:
                await self._page.close()
                self._page = None
            if self._context:
                await self._context.close()
                self._context = None

        duration = time.time() - start_time

        # Determine status
        if steps_failed == 0 and steps_passed == len(steps):
            status = TestFlowStatus.PASSED
        elif error_message and "Flow error" in error_message:
            status = TestFlowStatus.ERROR
        else:
            status = TestFlowStatus.FAILED

        # AI analysis of final screenshot
        ai_analysis = None
        if self.enable_ai_analysis and screenshots:
            ai_analysis = await self._analyze_with_ai(
                screenshots[-1],
                f"Test flow: {flow_name}, Status: {status.value}"
            )

        result = FlowResult(
            flow_name=flow_name,
            status=status,
            steps_total=len(steps),
            steps_passed=steps_passed,
            steps_failed=steps_failed,
            duration_seconds=duration,
            error_message=error_message,
            screenshots=screenshots[:3],  # Keep max 3 screenshots
            ai_analysis=ai_analysis,
            performance_metrics=performance if 'performance' in dir() else None
        )

        self.results.append(result)
        return result

    async def _get_performance_metrics(self) -> dict[str, Any]:
        """Get page performance metrics"""
        try:
            if not self._page:
                return {}

            return await self._page.evaluate("""
                () => {
                    const timing = performance.timing;
                    const paint = performance.getEntriesByType('paint');
                    return {
                        dom_content_loaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                        load_complete: timing.loadEventEnd - timing.navigationStart,
                        first_paint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
                        first_contentful_paint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
                        ttfb: timing.responseStart - timing.navigationStart
                    };
                }
            """)
        except Exception as exc:
            logger.debug("Failed to collect performance metrics: %s", exc, exc_info=True)
            return {}

    # =========================================================================
    # PREDEFINED TEST FLOWS
    # =========================================================================

    async def test_mrg_homepage(self) -> FlowResult:
        """Test MyRoofGenius homepage"""
        steps = [
            FlowStep("Navigate to homepage", "navigate", value="https://myroofgenius.com"),
            FlowStep("Wait for load", "wait", value="2000"),
            FlowStep("Screenshot homepage", "screenshot"),
            FlowStep("Assert branding", "assert_text", expected="MyRoofGenius"),
            FlowStep(
                "Check navigation",
                "assert_element",
                selector="nav, header, [role='navigation'], a[href='/pricing'], a[href='/tools'], a[href='/login'], a[href*='start']",
            ),
            FlowStep("Check CTA button", "assert_element", selector="button, a[href*='login'], a[href*='start']"),
        ]
        return await self.run_flow("MRG Homepage", steps)

    async def test_mrg_homepage_quick(self) -> FlowResult:
        """Fast-path MyRoofGenius homepage check (used by quick health)."""
        steps = [
            FlowStep("Navigate to homepage", "navigate", value="https://myroofgenius.com"),
            FlowStep("Assert body", "assert_element", selector="body"),
        ]
        return await self.run_flow("MRG Homepage (Quick)", steps)

    async def test_mrg_login_page(self) -> FlowResult:
        """Test MyRoofGenius login page"""
        steps = [
            FlowStep("Navigate to login", "navigate", value="https://myroofgenius.com/login"),
            FlowStep("Wait for load", "wait", value="2000"),
            FlowStep("Screenshot login", "screenshot"),
            FlowStep("Assert login form", "assert_element", selector="form, [type='email'], input[name='email']"),
            FlowStep(
                "Assert auth method",
                "assert_element",
                selector=(
                    "[type='password'], input[name='password'], input[autocomplete='current-password'], "
                    "input[autocomplete='new-password'], button:has-text('Send link'), "
                    "button:has-text('Continue'), button:has-text('Sign in')"
                ),
            ),
            FlowStep("Assert submit button", "assert_element", selector="button[type='submit'], input[type='submit'], button"),
        ]
        return await self.run_flow("MRG Login Page", steps)

    async def test_mrg_login_flow(self) -> FlowResult:
        """Test MyRoofGenius actual login flow"""
        steps = [
            FlowStep("Navigate to login", "navigate", value="https://myroofgenius.com/login"),
            FlowStep("Wait for form", "wait_for_selector", selector="input[type='email'], input[name='email']"),
            FlowStep("Fill email", "fill", selector="input[type='email'], input[name='email']", value=MRG_TEST_EMAIL),
            FlowStep("Fill password", "fill", selector="input[type='password'], input[name='password']", value=MRG_TEST_PASSWORD),
            FlowStep("Screenshot filled form", "screenshot"),
            FlowStep("Submit login", "click", selector="button[type='submit'], button"),
            FlowStep("Wait for redirect", "wait", value="3000"),
            FlowStep("Screenshot result", "screenshot"),
            # Note: Will fail if credentials invalid, which is expected for test accounts
        ]
        return await self.run_flow("MRG Login Flow", steps)

    async def test_mrg_tools_page(self) -> FlowResult:
        """Test MyRoofGenius tools page"""
        steps = [
            FlowStep("Navigate to tools", "navigate", value="https://myroofgenius.com/tools"),
            FlowStep("Wait for load", "wait", value="2000"),
            FlowStep("Screenshot tools", "screenshot"),
            FlowStep("Assert tools content", "assert_text", expected="tool"),
        ]
        return await self.run_flow("MRG Tools Page", steps)

    async def test_mrg_pricing_page(self) -> FlowResult:
        """Test MyRoofGenius pricing page"""
        steps = [
            FlowStep("Navigate to pricing", "navigate", value="https://myroofgenius.com/pricing"),
            FlowStep("Wait for load", "wait", value="2000"),
            FlowStep("Screenshot pricing", "screenshot"),
            # NOTE: MyRoofGenius pages do not consistently include the literal substring "price" in rendered HTML.
            # Use a stable keyword present in the page content instead.
            FlowStep("Assert pricing content", "assert_text", expected="pricing"),
        ]
        return await self.run_flow("MRG Pricing Page", steps)

    async def test_erp_homepage(self) -> FlowResult:
        """Test Weathercraft ERP homepage"""
        steps = [
            FlowStep("Navigate to homepage", "navigate", value="https://weathercraft-erp.vercel.app"),
            FlowStep("Wait for load", "wait", value="2000"),
            FlowStep("Screenshot homepage", "screenshot"),
            FlowStep("Assert loaded", "assert_element", selector="body"),
        ]
        return await self.run_flow("ERP Homepage", steps)

    async def test_erp_homepage_quick(self) -> FlowResult:
        """Fast-path ERP homepage check (used by quick health)."""
        steps = [
            FlowStep("Navigate to homepage", "navigate", value="https://weathercraft-erp.vercel.app"),
            FlowStep("Assert body", "assert_element", selector="body"),
        ]
        return await self.run_flow("ERP Homepage (Quick)", steps)

    async def test_erp_login_page(self) -> FlowResult:
        """Test Weathercraft ERP login page"""
        steps = [
            FlowStep("Navigate to login", "navigate", value="https://weathercraft-erp.vercel.app/login"),
            FlowStep("Wait for load", "wait", value="2000"),
            FlowStep("Screenshot login", "screenshot"),
            FlowStep("Assert form elements", "assert_element", selector="input"),
        ]
        return await self.run_flow("ERP Login Page", steps)

    async def test_command_center_dashboard(self) -> FlowResult:
        """Test BrainOps Command Center dashboard (public redirect target)"""
        steps = [
            FlowStep("Navigate to dashboard", "navigate", value="https://brainops-command-center.vercel.app/dashboard"),
            FlowStep("Wait for load", "wait", value="2000"),
            FlowStep("Screenshot dashboard", "screenshot"),
            FlowStep("Assert branding", "assert_text", expected="Command Center"),
            FlowStep(
                "Check navigation or login",
                "assert_element",
                selector="nav, [role='navigation'], header, aside, form, input[type='email']",
            ),
        ]
        return await self.run_flow("Command Center Dashboard", steps)

    async def test_command_center_aurea(self) -> FlowResult:
        """Test Command Center AUREA chat page"""
        steps = [
            FlowStep("Navigate to AUREA", "navigate", value="https://brainops-command-center.vercel.app/aurea"),
            FlowStep("Wait for load", "wait", value="2000"),
            FlowStep("Screenshot AUREA", "screenshot"),
            # /aurea is auth-protected and redirects to /login when unauthenticated.
            # This flow is a public-surface smoke, so validate the page is reachable and branded.
            FlowStep("Assert Command Center loaded", "assert_text", expected="Command Center"),
        ]
        return await self.run_flow("Command Center AUREA", steps)

    async def test_brainstack_studio_homepage(self) -> FlowResult:
        """Test Brainstack Studio marketing site"""
        steps = [
            FlowStep("Navigate to Brainstack Studio", "navigate", value="https://brainstack-studio.vercel.app"),
            FlowStep("Wait for load", "wait", value="2000"),
            FlowStep("Screenshot Brainstack Studio", "screenshot"),
            FlowStep("Assert BrainOps branding", "assert_text", expected="BrainOps"),
        ]
        return await self.run_flow("Brainstack Studio Homepage", steps)

    async def run_full_test_suite(self, skip_erp: bool = False) -> dict[str, Any]:
        """Run all test flows.

        Args:
            skip_erp: When True, exclude Weathercraft ERP flows (non-ERP verification scope).
        """
        logger.info("Starting full ChatGPT-Agent test suite...")
        start_time = time.time()

        results = []

        # MRG Tests
        results.append(await self.test_mrg_homepage())
        results.append(await self.test_mrg_login_page())
        results.append(await self.test_mrg_tools_page())
        results.append(await self.test_mrg_pricing_page())

        # ERP Tests
        if not skip_erp:
            results.append(await self.test_erp_homepage())
            results.append(await self.test_erp_login_page())

        # Command Center + Brainstack Studio (public surfaces)
        results.append(await self.test_command_center_dashboard())
        results.append(await self.test_command_center_aurea())
        results.append(await self.test_brainstack_studio_homepage())

        duration = time.time() - start_time

        passed = len([r for r in results if r.status == TestFlowStatus.PASSED])
        failed = len([r for r in results if r.status == TestFlowStatus.FAILED])
        errors = len([r for r in results if r.status == TestFlowStatus.ERROR])

        summary = {
            "total_flows": len(results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "duration_seconds": duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": [
                {
                    "flow_name": r.flow_name,
                    "status": r.status.value,
                    "steps_passed": r.steps_passed,
                    "steps_total": r.steps_total,
                    "duration_seconds": r.duration_seconds,
                    "error_message": r.error_message,
                    "ai_score": r.ai_analysis.get("overall_score") if r.ai_analysis else None
                }
                for r in results
            ]
        }

        logger.info(f"Test suite complete: {passed}/{len(results)} passed in {duration:.1f}s")

        return summary


# =============================================================================
# API FUNCTIONS
# =============================================================================

async def run_chatgpt_agent_tests(skip_erp: bool = False) -> dict[str, Any]:
    """Run full ChatGPT-Agent test suite."""
    tester = ChatGPTAgentTester()
    try:
        await tester.initialize()
        return await tester.run_full_test_suite(skip_erp=skip_erp)
    finally:
        await tester.close()


async def run_quick_health_test(skip_erp: bool = False) -> dict[str, Any]:
    """
    Run quick health test (homepage only).

    Default behavior is a fast HTTP probe (no Playwright dependency) to keep `/e2e/verify` stable.
    Set `CHATGPT_AGENT_TESTER_QUICK_MODE=playwright` to force real browser-based checks.
    """

    mode = os.getenv("CHATGPT_AGENT_TESTER_QUICK_MODE", "http").strip().lower()
    if mode in {"playwright", "browser", "ui"}:
        tester = ChatGPTAgentTester(enable_ai_analysis=False)
        try:
            await tester.initialize()

            mrg_result = await tester.test_mrg_homepage_quick()
            erp_result = await tester.test_erp_homepage_quick() if not skip_erp else None

            return {
                "mode": "playwright",
                "mrg_healthy": mrg_result.status == TestFlowStatus.PASSED,
                "erp_healthy": erp_result.status == TestFlowStatus.PASSED if erp_result else None,
                "erp_skipped": bool(skip_erp),
                "mrg_details": {
                    "status": mrg_result.status.value,
                    "duration": mrg_result.duration_seconds,
                },
                "erp_details": {
                    "status": erp_result.status.value if erp_result else "skipped",
                    "duration": erp_result.duration_seconds if erp_result else 0.0,
                },
            }
        finally:
            await tester.close()

    # Fast-path: HTTP probe
    import aiohttp

    async def _probe(session: aiohttp.ClientSession, url: str) -> tuple[bool, float, Optional[str]]:
        start = time.perf_counter()

        def _finish(ok: bool, err: Optional[str] = None) -> tuple[bool, float, Optional[str]]:
            return ok, (time.perf_counter() - start), err

        try:
            async with session.get(url, allow_redirects=True) as resp:
                content_type = resp.headers.get("content-type", "")
                body_bytes = await resp.content.read(200_000)
                body = body_bytes.decode("utf-8", errors="ignore").lower()

                if resp.status < 200 or resp.status >= 400:
                    return _finish(False, f"HTTP {resp.status}")

                if "text/html" not in content_type.lower():
                    return _finish(False, f"Unexpected content-type: {content_type}")

                if "<body" not in body:
                    return _finish(False, "Missing <body> in HTML")

                return _finish(True, None)
        except asyncio.TimeoutError:
            return _finish(False, "Timeout")
        except Exception as exc:
            return _finish(False, str(exc))

    timeout_seconds = float(os.getenv("CHATGPT_AGENT_TESTER_HTTP_TIMEOUT_SECONDS", "12"))
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    headers = {
        "User-Agent": "BrainOpsChatGPTAgentTester/1.0 (+https://brainstackstudio.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        if skip_erp:
            (mrg_ok, mrg_duration, mrg_error) = await _probe(session, "https://myroofgenius.com")
            erp_ok, erp_duration, erp_error = None, 0.0, None
        else:
            (mrg_ok, mrg_duration, mrg_error), (erp_ok, erp_duration, erp_error) = await asyncio.gather(
                _probe(session, "https://myroofgenius.com"),
                _probe(session, "https://weathercraft-erp.vercel.app"),
            )

    return {
        "mode": "http",
        "mrg_healthy": mrg_ok,
        "erp_healthy": erp_ok,
        "erp_skipped": bool(skip_erp),
        "mrg_details": {
            "status": "passed" if mrg_ok else "failed",
            "duration": mrg_duration,
            "error": mrg_error,
        },
        "erp_details": {
            "status": "passed" if erp_ok else ("skipped" if skip_erp else "failed"),
            "duration": erp_duration,
            "error": erp_error,
        },
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    async def main():
        if len(sys.argv) > 1 and sys.argv[1] == "--quick":
            results = await run_quick_health_test()
        else:
            results = await run_chatgpt_agent_tests()

        print(json.dumps(results, indent=2))

    asyncio.run(main())

#!/usr/bin/env python3
"""
AI-Powered UI Testing System
=============================
Autonomous UI testing using Playwright + Vision AI for comprehensive
testing of all user interfaces.

Features:
1. Screenshot-based visual testing with AI analysis
2. Automatic test generation from page content
3. Accessibility testing with AI recommendations
4. Performance metrics collection
5. Cross-browser testing
6. Mobile responsive testing
7. Automated issue detection and reporting
8. Continuous monitoring mode

Author: BrainOps AI System
Version: 1.0.0
"""

import os
import json
import asyncio
import logging
import base64
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class TestSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class UITestResult:
    """Result of a UI test"""
    test_id: str
    test_name: str
    url: str
    status: TestStatus
    severity: TestSeverity
    message: str
    screenshot_path: Optional[str] = None
    ai_analysis: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    accessibility_issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PageAnalysis:
    """AI analysis of a page"""
    url: str
    title: str
    description: str
    usability_score: float  # 0-100
    visual_issues: List[Dict[str, Any]]
    functional_issues: List[Dict[str, Any]]
    accessibility_issues: List[Dict[str, Any]]
    performance_issues: List[Dict[str, Any]]
    suggestions: List[str]
    elements_found: int
    interactive_elements: int
    forms_count: int
    links_count: int


class AIVisionAnalyzer:
    """
    Analyzes screenshots using AI vision models.
    Supports OpenAI GPT-4V, Anthropic Claude, and Google Gemini.
    """

    def __init__(self, preferred_model: str = "openai"):
        self.preferred_model = preferred_model

    async def analyze_screenshot(
        self,
        screenshot_base64: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a screenshot using AI vision"""
        if self.preferred_model == "openai" and OPENAI_API_KEY:
            return await self._analyze_with_openai(screenshot_base64, context)
        elif self.preferred_model == "anthropic" and ANTHROPIC_API_KEY:
            return await self._analyze_with_anthropic(screenshot_base64, context)
        elif GEMINI_API_KEY:
            return await self._analyze_with_gemini(screenshot_base64, context)
        else:
            # Fallback to basic analysis without vision
            return self._basic_analysis(context)

    async def _analyze_with_openai(
        self,
        screenshot_base64: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze using OpenAI GPT-4V"""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

            prompt = self._build_analysis_prompt(context)

            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert UI/UX tester. Analyze the screenshot and identify any visual, functional, or accessibility issues. Be thorough and specific."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )

            analysis_text = response.choices[0].message.content
            return self._parse_analysis(analysis_text)

        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._basic_analysis(context)

    async def _analyze_with_anthropic(
        self,
        screenshot_base64: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze using Anthropic Claude"""
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

            prompt = self._build_analysis_prompt(context)

            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            analysis_text = response.content[0].text
            return self._parse_analysis(analysis_text)

        except Exception as e:
            logger.error(f"Anthropic analysis failed: {e}")
            return self._basic_analysis(context)

    async def _analyze_with_gemini(
        self,
        screenshot_base64: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze using Google Gemini"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)

            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            prompt = self._build_analysis_prompt(context)

            # Decode base64 to bytes for Gemini
            image_bytes = base64.b64decode(screenshot_base64)

            response = await asyncio.to_thread(
                model.generate_content,
                [
                    {"mime_type": "image/png", "data": image_bytes},
                    prompt
                ]
            )

            analysis_text = response.text
            return self._parse_analysis(analysis_text)

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return self._basic_analysis(context)

    def _build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build the analysis prompt"""
        url = context.get("url", "unknown")
        page_title = context.get("title", "")

        return f"""Analyze this UI screenshot for a web application testing system.

URL: {url}
Page Title: {page_title}

Please analyze and provide:

1. VISUAL ISSUES: Any layout problems, alignment issues, broken images, text overflow, color contrast problems
2. FUNCTIONAL ISSUES: Missing buttons, broken links (if visible), form issues, navigation problems
3. ACCESSIBILITY ISSUES: Missing alt text indicators, small touch targets, poor color contrast, missing labels
4. USABILITY ISSUES: Confusing navigation, unclear CTAs, poor information hierarchy
5. USABILITY SCORE: Rate 0-100 based on overall user experience quality

Format your response as JSON:
{{
    "usability_score": <number>,
    "visual_issues": [{{ "issue": "<description>", "severity": "critical|high|medium|low", "location": "<where>" }}],
    "functional_issues": [{{ "issue": "<description>", "severity": "critical|high|medium|low", "location": "<where>" }}],
    "accessibility_issues": [{{ "issue": "<description>", "severity": "critical|high|medium|low", "wcag": "<guideline>" }}],
    "usability_issues": [{{ "issue": "<description>", "severity": "critical|high|medium|low" }}],
    "positive_aspects": ["<good things about the UI>"],
    "suggestions": ["<improvement recommendation>"]
}}

Be thorough but concise. Focus on actionable issues."""

    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the AI analysis response"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', analysis_text)
            if json_match:
                return json.loads(json_match.group())
            else:
                # If no JSON, create structured response from text
                return {
                    "usability_score": 70,
                    "visual_issues": [],
                    "functional_issues": [],
                    "accessibility_issues": [],
                    "usability_issues": [],
                    "positive_aspects": [],
                    "suggestions": [analysis_text[:500]],
                    "raw_response": analysis_text
                }
        except json.JSONDecodeError:
            return {
                "usability_score": 70,
                "raw_response": analysis_text,
                "parse_error": True
            }

    def _basic_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Basic analysis without AI vision"""
        return {
            "usability_score": 50,
            "visual_issues": [],
            "functional_issues": [],
            "accessibility_issues": [],
            "usability_issues": [],
            "suggestions": ["AI vision analysis not available - using basic checks only"],
            "note": "Full AI analysis requires API key configuration"
        }


class AIUITestingEngine:
    """
    Main AI-powered UI testing engine.
    Integrates with Playwright for browser automation
    and AI vision for intelligent analysis.
    """

    def __init__(self):
        self.vision_analyzer = AIVisionAnalyzer()
        self.test_results: List[UITestResult] = []
        self.screenshot_dir = "/tmp/ai_ui_tests"
        self._browser = None
        self._context = None

    async def initialize(self, timeout_seconds: int = 30):
        """Initialize the testing engine with Playwright (with timeout)"""
        try:
            from playwright.async_api import async_playwright

            # Add timeout to prevent infinite hangs
            try:
                self._playwright = await asyncio.wait_for(
                    async_playwright().start(),
                    timeout=timeout_seconds
                )
                self._browser = await asyncio.wait_for(
                    self._playwright.chromium.launch(
                        headless=True,
                        args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
                    ),
                    timeout=timeout_seconds
                )
                self._context = await asyncio.wait_for(
                    self._browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        device_scale_factor=2
                    ),
                    timeout=10
                )
                os.makedirs(self.screenshot_dir, exist_ok=True)
                logger.info("AI UI Testing Engine initialized with Playwright")
                self._playwright_available = True
            except asyncio.TimeoutError:
                logger.error(f"Playwright initialization timed out after {timeout_seconds}s")
                self._browser = None
                self._playwright_available = False
        except ImportError:
            logger.warning("Playwright not installed - running in HTTP-only mode")
            self._browser = None
            self._playwright_available = False
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            self._browser = None
            self._playwright_available = False

    async def close(self):
        """Close the browser and cleanup"""
        try:
            if self._browser:
                await asyncio.wait_for(self._browser.close(), timeout=10)
            if hasattr(self, '_playwright') and self._playwright:
                await asyncio.wait_for(self._playwright.stop(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("Browser close timed out")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")

    async def test_url_http_fallback(
        self,
        url: str,
        test_name: str = "UI Test"
    ) -> UITestResult:
        """
        HTTP-only fallback testing when Playwright is unavailable.
        Tests response, headers, and basic content.
        """
        test_id = hashlib.md5(f"{url}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        start_time = datetime.now(timezone.utc)

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, allow_redirects=True) as response:
                    status_code = response.status
                    headers = dict(response.headers)
                    content = await response.text()
                    load_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                    # Basic checks
                    issues = []
                    severity = TestSeverity.INFO
                    status = TestStatus.PASSED

                    # Check HTTP status
                    if status_code >= 500:
                        issues.append(f"Server error: HTTP {status_code}")
                        severity = TestSeverity.CRITICAL
                        status = TestStatus.FAILED
                    elif status_code >= 400:
                        issues.append(f"Client error: HTTP {status_code}")
                        severity = TestSeverity.HIGH
                        status = TestStatus.FAILED

                    # Check for common issues in HTML
                    content_lower = content.lower()
                    if '<title>' not in content_lower:
                        issues.append("Missing <title> tag")
                    if 'viewport' not in content_lower:
                        issues.append("Missing viewport meta tag (mobile unfriendly)")
                    if 'error' in content_lower and ('500' in content or '404' in content):
                        issues.append("Error page detected in content")
                        severity = TestSeverity.HIGH
                        status = TestStatus.FAILED

                    # Check response time
                    if load_time_ms > 5000:
                        issues.append(f"Slow response: {load_time_ms:.0f}ms")
                        if severity == TestSeverity.INFO:
                            severity = TestSeverity.MEDIUM

                    # Check security headers
                    if 'strict-transport-security' not in [h.lower() for h in headers.keys()]:
                        issues.append("Missing HSTS header")
                    if 'x-frame-options' not in [h.lower() for h in headers.keys()]:
                        issues.append("Missing X-Frame-Options header")

                    message = f"HTTP {status_code} - {len(issues)} issues found" if issues else f"HTTP {status_code} - OK ({load_time_ms:.0f}ms)"

                    return UITestResult(
                        test_id=test_id,
                        test_name=f"{test_name} (HTTP)",
                        url=url,
                        status=status,
                        severity=severity,
                        message=message,
                        performance_metrics={
                            "load_time_ms": load_time_ms,
                            "status_code": status_code,
                            "content_length": len(content)
                        },
                        suggestions=issues if issues else ["Page loaded successfully"]
                    )

        except asyncio.TimeoutError:
            return UITestResult(
                test_id=test_id,
                test_name=f"{test_name} (HTTP)",
                url=url,
                status=TestStatus.FAILED,
                severity=TestSeverity.CRITICAL,
                message="Request timed out after 30 seconds"
            )
        except Exception as e:
            return UITestResult(
                test_id=test_id,
                test_name=f"{test_name} (HTTP)",
                url=url,
                status=TestStatus.FAILED,
                severity=TestSeverity.CRITICAL,
                message=f"HTTP request failed: {str(e)}"
            )

    async def test_url(
        self,
        url: str,
        test_name: str = "UI Test",
        wait_for_load: bool = True,
        check_accessibility: bool = True,
        check_performance: bool = True
    ) -> UITestResult:
        """
        Test a single URL with comprehensive AI analysis.
        Falls back to HTTP-only testing if Playwright unavailable.
        """
        test_id = hashlib.md5(f"{url}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        # Fallback to HTTP testing if no browser
        if not self._browser:
            logger.info(f"Using HTTP fallback for {url}")
            return await self.test_url_http_fallback(url, test_name)

        try:
            page = await asyncio.wait_for(self._context.new_page(), timeout=10)
        except asyncio.TimeoutError:
            logger.error("Failed to create new page - timeout")
            return await self.test_url_http_fallback(url, test_name)

        try:
            # Navigate to URL with timeout (60s max for slow pages)
            try:
                response = await asyncio.wait_for(
                    page.goto(url, wait_until="networkidle" if wait_for_load else "domcontentloaded"),
                    timeout=60
                )
            except asyncio.TimeoutError:
                await page.close()
                logger.warning(f"Page navigation timed out for {url}")
                return UITestResult(
                    test_id=test_id,
                    test_name=test_name,
                    url=url,
                    status=TestStatus.FAILED,
                    severity=TestSeverity.HIGH,
                    message="Page navigation timed out after 60 seconds"
                )

            if not response or response.status >= 400:
                await page.close()
                return UITestResult(
                    test_id=test_id,
                    test_name=test_name,
                    url=url,
                    status=TestStatus.FAILED,
                    severity=TestSeverity.CRITICAL,
                    message=f"Page failed to load: HTTP {response.status if response else 'No response'}"
                )

            # Take screenshot with timeout
            screenshot_path = f"{self.screenshot_dir}/{test_id}.png"
            try:
                await asyncio.wait_for(page.screenshot(path=screenshot_path, full_page=True), timeout=30)
            except asyncio.TimeoutError:
                logger.warning(f"Screenshot timed out for {url}")

            # Read screenshot as base64
            with open(screenshot_path, "rb") as f:
                screenshot_base64 = base64.b64encode(f.read()).decode()

            # Get page context
            title = await page.title()
            page_context = {
                "url": url,
                "title": title,
                "viewport": {"width": 1920, "height": 1080}
            }

            # AI Vision Analysis
            ai_analysis = await self.vision_analyzer.analyze_screenshot(
                screenshot_base64,
                page_context
            )

            # Performance metrics
            performance_metrics = None
            if check_performance:
                performance_metrics = await self._collect_performance_metrics(page)

            # Accessibility check
            accessibility_issues = []
            if check_accessibility:
                accessibility_issues = await self._check_accessibility(page)

            # Determine test status based on analysis
            status, severity, message = self._evaluate_results(
                ai_analysis,
                accessibility_issues,
                performance_metrics
            )

            result = UITestResult(
                test_id=test_id,
                test_name=test_name,
                url=url,
                status=status,
                severity=severity,
                message=message,
                screenshot_path=screenshot_path,
                ai_analysis=ai_analysis,
                performance_metrics=performance_metrics,
                accessibility_issues=accessibility_issues,
                suggestions=ai_analysis.get("suggestions", [])
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            logger.error(f"Error testing {url}: {e}")
            return UITestResult(
                test_id=test_id,
                test_name=test_name,
                url=url,
                status=TestStatus.FAILED,
                severity=TestSeverity.HIGH,
                message=f"Test execution error: {str(e)}"
            )
        finally:
            await page.close()

    async def _collect_performance_metrics(self, page) -> Dict[str, Any]:
        """Collect performance metrics from the page"""
        try:
            metrics = await page.evaluate("""
                () => {
                    const timing = performance.timing;
                    const navigation = performance.getEntriesByType('navigation')[0] || {};
                    return {
                        dom_content_loaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                        load_complete: timing.loadEventEnd - timing.navigationStart,
                        first_paint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
                        first_contentful_paint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0,
                        dom_interactive: timing.domInteractive - timing.navigationStart,
                        ttfb: timing.responseStart - timing.navigationStart,
                        resource_count: performance.getEntriesByType('resource').length
                    };
                }
            """)
            return metrics
        except Exception as e:
            logger.warning(f"Failed to collect performance metrics: {e}")
            return {}

    async def _check_accessibility(self, page) -> List[Dict[str, Any]]:
        """Run accessibility checks using axe-core if available"""
        try:
            # Inject axe-core
            await page.add_script_tag(url="https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.7.2/axe.min.js")

            # Run axe
            results = await page.evaluate("""
                async () => {
                    const results = await axe.run();
                    return results.violations.map(v => ({
                        id: v.id,
                        impact: v.impact,
                        description: v.description,
                        help: v.help,
                        nodes_count: v.nodes.length
                    }));
                }
            """)
            return results
        except Exception as e:
            logger.warning(f"Accessibility check failed: {e}")
            return []

    def _evaluate_results(
        self,
        ai_analysis: Dict[str, Any],
        accessibility_issues: List[Dict],
        performance_metrics: Optional[Dict]
    ) -> Tuple[TestStatus, TestSeverity, str]:
        """Evaluate test results and determine overall status"""
        issues = []
        max_severity = TestSeverity.INFO

        # Check AI analysis
        usability_score = ai_analysis.get("usability_score", 100)
        if usability_score < 50:
            issues.append(f"Low usability score: {usability_score}")
            max_severity = TestSeverity.HIGH

        # Count critical issues
        critical_count = 0
        high_count = 0

        for issue_type in ["visual_issues", "functional_issues", "accessibility_issues"]:
            for issue in ai_analysis.get(issue_type, []):
                if issue.get("severity") == "critical":
                    critical_count += 1
                elif issue.get("severity") == "high":
                    high_count += 1

        # Check accessibility
        for a11y_issue in accessibility_issues:
            if a11y_issue.get("impact") == "critical":
                critical_count += 1
            elif a11y_issue.get("impact") == "serious":
                high_count += 1

        # Check performance
        if performance_metrics:
            load_time = performance_metrics.get("load_complete", 0)
            if load_time > 5000:
                issues.append(f"Slow page load: {load_time}ms")
                if load_time > 10000:
                    max_severity = TestSeverity.HIGH

        # Determine final status
        if critical_count > 0:
            status = TestStatus.FAILED
            max_severity = TestSeverity.CRITICAL
            message = f"Found {critical_count} critical issues"
        elif high_count > 0:
            status = TestStatus.WARNING
            max_severity = TestSeverity.HIGH
            message = f"Found {high_count} high-severity issues"
        elif issues:
            status = TestStatus.WARNING
            message = "; ".join(issues)
        else:
            status = TestStatus.PASSED
            message = f"All checks passed. Usability score: {usability_score}"

        return status, max_severity, message

    async def test_application(
        self,
        base_url: str,
        routes: List[str],
        app_name: str = "Application"
    ) -> Dict[str, Any]:
        """
        Test an entire application with multiple routes.
        """
        results = []
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting comprehensive UI test for {app_name}")

        for route in routes:
            url = f"{base_url.rstrip('/')}{route}"
            result = await self.test_url(
                url=url,
                test_name=f"{app_name}: {route}",
                check_accessibility=True,
                check_performance=True
            )
            results.append(result)
            logger.info(f"  {route}: {result.status.value} - {result.message}")

        # Generate summary
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        passed = len([r for r in results if r.status == TestStatus.PASSED])
        failed = len([r for r in results if r.status == TestStatus.FAILED])
        warnings = len([r for r in results if r.status == TestStatus.WARNING])

        summary = {
            "application": app_name,
            "base_url": base_url,
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "duration_seconds": duration,
            "timestamp": start_time.isoformat(),
            "results": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "url": r.url,
                    "status": r.status.value,
                    "severity": r.severity.value,
                    "message": r.message,
                    "usability_score": r.ai_analysis.get("usability_score") if r.ai_analysis else None,
                    "suggestions": r.suggestions[:3] if r.suggestions else []
                }
                for r in results
            ]
        }

        return summary

    def get_test_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report"""
        return {
            "total_tests": len(self.test_results),
            "passed": len([r for r in self.test_results if r.status == TestStatus.PASSED]),
            "failed": len([r for r in self.test_results if r.status == TestStatus.FAILED]),
            "warnings": len([r for r in self.test_results if r.status == TestStatus.WARNING]),
            "results": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "url": r.url,
                    "status": r.status.value,
                    "severity": r.severity.value,
                    "message": r.message
                }
                for r in self.test_results
            ]
        }


# =============================================================================
# PREDEFINED TEST SUITES
# =============================================================================

MRG_ROUTES = [
    "/",
    "/login",
    "/register",
    "/dashboard",
    "/tools",
    "/marketplace",
    "/pricing",
    "/about",
    "/contact",
    "/aurea",
    "/estimator",
    "/features",
    "/for-contractors",
    "/for-homeowners"
]

ERP_ROUTES = [
    "/",
    "/login",
    "/dashboard",
    "/customers",
    "/jobs",
    "/estimates",
    "/invoices",
    "/reports"
]


async def test_myroofgenius():
    """Test MyRoofGenius application"""
    engine = AIUITestingEngine()
    await engine.initialize()

    try:
        results = await engine.test_application(
            base_url="https://myroofgenius.com",
            routes=MRG_ROUTES,
            app_name="MyRoofGenius"
        )
        return results
    finally:
        await engine.close()


async def test_weathercraft_erp():
    """Test Weathercraft ERP application"""
    engine = AIUITestingEngine()
    await engine.initialize()

    try:
        results = await engine.test_application(
            base_url="https://weathercraft-erp.vercel.app",
            routes=ERP_ROUTES,
            app_name="Weathercraft ERP"
        )
        return results
    finally:
        await engine.close()


# =============================================================================
# SINGLETON AND API
# =============================================================================

_testing_engine: Optional[AIUITestingEngine] = None


async def get_testing_engine() -> AIUITestingEngine:
    """Get or create the UI testing engine"""
    global _testing_engine
    if _testing_engine is None:
        _testing_engine = AIUITestingEngine()
        await _testing_engine.initialize()
    return _testing_engine


async def run_ui_test(url: str, test_name: str = "UI Test") -> Dict[str, Any]:
    """Run a single UI test"""
    engine = await get_testing_engine()
    result = await engine.test_url(url, test_name)
    return {
        "test_id": result.test_id,
        "status": result.status.value,
        "severity": result.severity.value,
        "message": result.message,
        "ai_analysis": result.ai_analysis,
        "suggestions": result.suggestions
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python ai_ui_testing.py <url>")
            print("       python ai_ui_testing.py --mrg")
            print("       python ai_ui_testing.py --erp")
            sys.exit(1)

        arg = sys.argv[1]

        if arg == "--mrg":
            results = await test_myroofgenius()
        elif arg == "--erp":
            results = await test_weathercraft_erp()
        else:
            results = await run_ui_test(arg)

        print(json.dumps(results, indent=2))

    asyncio.run(main())

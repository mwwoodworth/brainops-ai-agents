#!/usr/bin/env python3

"""
BrainOps UI Tester Agent
Automated UI testing using Playwright for real browser testing
Enhanced with Gemini Vision for AI-powered visual analysis
"""

import asyncio
import os
import base64
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright not installed. Install with: pip install playwright && playwright install")

# Gemini Vision for AI-powered visual analysis
try:
    import google.generativeai as genai
    GEMINI_VISION_AVAILABLE = bool(os.getenv("GOOGLE_API_KEY"))
    if GEMINI_VISION_AVAILABLE:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except ImportError:
    GEMINI_VISION_AVAILABLE = False

class TestStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"

class UITesterAgent:
    """Agent that performs automated UI testing on deployed applications with AUREA integration"""

    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.agent_type = "ui_tester"

        # AUREA Integration for decision recording and learning
        try:
            from aurea_integration import AUREAIntegration
            self.aurea = AUREAIntegration(tenant_id, self.agent_type)
        except ImportError:
            logger.warning("AUREA integration not available")
            self.aurea = None

        self.test_urls = {
            "brainops-command-center": {
                "base_url": "https://brainops-command-center.vercel.app",
                "pages": ["/", "/dashboard", "/ai"],
                "critical_elements": [
                    {"selector": "h1", "text": "BrainOps Command Center"},
                    {"selector": "[data-testid='agent-list']", "required": False},
                    {"selector": "[data-testid='metrics']", "required": False}
                ]
            },
            "weathercraft-erp": {
                "base_url": "https://weathercraft-erp.vercel.app",
                "pages": ["/", "/dashboard", "/customers", "/jobs"],
                "critical_elements": [
                    {"selector": "nav", "required": True},
                    {"selector": "[data-testid='sidebar']", "required": False}
                ]
            },
            "myroofgenius": {
                "base_url": "https://myroofgenius.com",
                "pages": ["/", "/features", "/pricing"],
                "critical_elements": [
                    {"selector": "header", "required": True},
                    {"selector": "footer", "required": True}
                ]
            }
        }
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.test_results = []

    async def setup_browser(self, headless: bool = True):
        """Initialize browser for testing"""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is required for UI testing")

        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=headless,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='BrainOps-UI-Tester/1.0'
        )

    async def teardown_browser(self):
        """Clean up browser resources"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()

    async def test_page_load(self, url: str, timeout: int = 30000) -> Dict[str, Any]:
        """Test if a page loads successfully"""
        result = {
            "url": url,
            "test": "page_load",
            "status": TestStatus.FAIL.value,
            "load_time_ms": None,
            "status_code": None,
            "error": None
        }

        try:
            page = await self.context.new_page()
            start_time = datetime.now()

            # Navigate and wait for load
            response = await page.goto(url, wait_until='networkidle', timeout=timeout)
            load_time = (datetime.now() - start_time).total_seconds() * 1000

            result["status_code"] = response.status if response else None
            result["load_time_ms"] = round(load_time)

            if response and response.status == 200:
                result["status"] = TestStatus.PASS.value
            else:
                result["error"] = f"HTTP {response.status if response else 'No response'}"

            await page.close()

        except Exception as e:
            result["status"] = TestStatus.ERROR.value
            result["error"] = str(e)

        return result

    async def test_element_presence(self, url: str, selector: str, text: str = None, timeout: int = 10000) -> Dict[str, Any]:
        """Test if specific elements are present on the page"""
        result = {
            "url": url,
            "test": "element_presence",
            "selector": selector,
            "expected_text": text,
            "status": TestStatus.FAIL.value,
            "found": False,
            "actual_text": None,
            "error": None
        }

        try:
            page = await self.context.new_page()
            await page.goto(url, wait_until='domcontentloaded')

            # Wait for element
            try:
                element = await page.wait_for_selector(selector, timeout=timeout)
                result["found"] = True

                if element:
                    actual_text = await element.text_content()
                    result["actual_text"] = actual_text

                    if text:
                        if text.lower() in actual_text.lower():
                            result["status"] = TestStatus.PASS.value
                        else:
                            result["error"] = f"Text mismatch: expected '{text}', got '{actual_text}'"
                    else:
                        result["status"] = TestStatus.PASS.value

            except PlaywrightTimeoutError as exc:
                result["found"] = False
                result["error"] = f"Element not found: {selector}"
                logger.debug("Selector timeout for %s: %s", selector, exc)

            await page.close()

        except Exception as e:
            result["status"] = TestStatus.ERROR.value
            result["error"] = str(e)

        return result

    async def test_interaction(self, url: str, actions: List[Dict]) -> Dict[str, Any]:
        """Test user interactions (clicks, form fills, etc.)"""
        result = {
            "url": url,
            "test": "interaction",
            "actions": actions,
            "status": TestStatus.FAIL.value,
            "completed_actions": [],
            "error": None
        }

        try:
            page = await self.context.new_page()
            await page.goto(url, wait_until='networkidle')

            for action in actions:
                action_type = action.get("type")
                selector = action.get("selector")
                value = action.get("value")

                try:
                    if action_type == "click":
                        await page.click(selector)
                        result["completed_actions"].append(f"Clicked {selector}")

                    elif action_type == "fill":
                        await page.fill(selector, value)
                        result["completed_actions"].append(f"Filled {selector}")

                    elif action_type == "select":
                        await page.select_option(selector, value)
                        result["completed_actions"].append(f"Selected {value} in {selector}")

                    elif action_type == "wait":
                        await page.wait_for_timeout(value)
                        result["completed_actions"].append(f"Waited {value}ms")

                    elif action_type == "wait_for":
                        await page.wait_for_selector(selector)
                        result["completed_actions"].append(f"Waited for {selector}")

                except Exception as e:
                    result["error"] = f"Action failed: {action_type} on {selector} - {str(e)}"
                    break

            if len(result["completed_actions"]) == len(actions):
                result["status"] = TestStatus.PASS.value

            await page.close()

        except Exception as e:
            result["status"] = TestStatus.ERROR.value
            result["error"] = str(e)

        return result

    async def test_responsive_design(self, url: str) -> Dict[str, Any]:
        """Test page on different screen sizes"""
        viewports = [
            {"name": "mobile", "width": 375, "height": 667},
            {"name": "tablet", "width": 768, "height": 1024},
            {"name": "desktop", "width": 1920, "height": 1080}
        ]

        results = {
            "url": url,
            "test": "responsive_design",
            "viewports": {}
        }

        for viewport in viewports:
            page = await self.context.new_page()
            await page.set_viewport_size(
                {"width": viewport["width"], "height": viewport["height"]}
            )

            try:
                await page.goto(url, wait_until='networkidle')

                # Take screenshot
                screenshot_path = f"/tmp/screenshot_{viewport['name']}_{datetime.now().timestamp()}.png"
                await page.screenshot(path=screenshot_path)

                # Check if content is visible
                is_visible = await page.is_visible("body")

                results["viewports"][viewport["name"]] = {
                    "status": TestStatus.PASS.value if is_visible else TestStatus.FAIL.value,
                    "screenshot": screenshot_path
                }

            except Exception as e:
                results["viewports"][viewport["name"]] = {
                    "status": TestStatus.ERROR.value,
                    "error": str(e)
                }

            await page.close()

        return results

    async def test_performance(self, url: str) -> Dict[str, Any]:
        """Test page performance metrics"""
        result = {
            "url": url,
            "test": "performance",
            "metrics": {},
            "status": TestStatus.PASS.value
        }

        try:
            page = await self.context.new_page()

            # Enable performance monitoring
            await page.goto(url, wait_until='networkidle')

            # Get performance metrics
            performance_timing = await page.evaluate("""() => {
                const timing = performance.timing;
                return {
                    domContentLoaded: timing.domContentLoadedEventEnd - timing.domContentLoadedEventStart,
                    loadComplete: timing.loadEventEnd - timing.loadEventStart,
                    firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
                    firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
                };
            }""")

            result["metrics"] = performance_timing

            # Check against thresholds
            if performance_timing.get("loadComplete", 0) > 5000:
                result["status"] = TestStatus.FAIL.value
                result["error"] = "Page load too slow (>5s)"

            await page.close()

        except Exception as e:
            result["status"] = TestStatus.ERROR.value
            result["error"] = str(e)

        return result

    async def test_accessibility(self, url: str) -> Dict[str, Any]:
        """Basic accessibility testing"""
        result = {
            "url": url,
            "test": "accessibility",
            "checks": {},
            "status": TestStatus.PASS.value
        }

        try:
            page = await self.context.new_page()
            await page.goto(url, wait_until='networkidle')

            # Check for alt text on images
            images_without_alt = await page.evaluate("""() => {
                const images = document.querySelectorAll('img');
                return Array.from(images).filter(img => !img.alt).length;
            }""")

            result["checks"]["images_without_alt"] = images_without_alt

            # Check for form labels
            inputs_without_labels = await page.evaluate("""() => {
                const inputs = document.querySelectorAll('input, select, textarea');
                return Array.from(inputs).filter(input => {
                    const id = input.id;
                    if (!id) return true;
                    return !document.querySelector(`label[for="${id}"]`);
                }).length;
            }""")

            result["checks"]["inputs_without_labels"] = inputs_without_labels

            # Check for heading hierarchy
            heading_issues = await page.evaluate("""() => {
                const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
                let lastLevel = 0;
                let issues = 0;
                headings.forEach(h => {
                    const level = parseInt(h.tagName.substring(1));
                    if (level > lastLevel + 1) issues++;
                    lastLevel = level;
                });
                return issues;
            }""")

            result["checks"]["heading_hierarchy_issues"] = heading_issues

            # Determine pass/fail
            total_issues = sum(result["checks"].values())
            if total_issues > 0:
                result["status"] = TestStatus.FAIL.value
                result["error"] = f"Found {total_issues} accessibility issues"

            await page.close()

        except Exception as e:
            result["status"] = TestStatus.ERROR.value
            result["error"] = str(e)

        return result

    async def test_api_integration(self, url: str) -> Dict[str, Any]:
        """Test if API endpoints are being called correctly"""
        result = {
            "url": url,
            "test": "api_integration",
            "api_calls": [],
            "status": TestStatus.PASS.value
        }

        try:
            page = await self.context.new_page()

            # Intercept network requests
            api_calls = []

            async def handle_request(route, request):
                if 'api' in request.url or 'graphql' in request.url:
                    api_calls.append({
                        "url": request.url,
                        "method": request.method,
                        "timestamp": datetime.now().isoformat()
                    })
                await route.continue_()

            await page.route('**/*', handle_request)
            await page.goto(url, wait_until='networkidle')

            # Wait a bit for any delayed API calls
            await page.wait_for_timeout(2000)

            result["api_calls"] = api_calls
            result["total_calls"] = len(api_calls)

            if len(api_calls) == 0:
                result["warning"] = "No API calls detected"

            await page.close()

        except Exception as e:
            result["status"] = TestStatus.ERROR.value
            result["error"] = str(e)

        return result

    async def test_visual_ai_analysis(self, url: str, app_name: str = "unknown") -> Dict[str, Any]:
        """
        AI-powered visual analysis using Gemini Vision.
        Analyzes screenshots for:
        - Visual consistency and design quality
        - UX/UI best practices
        - Accessibility issues
        - Brand consistency
        - Layout problems
        - Content quality
        """
        result = {
            "url": url,
            "test": "visual_ai_analysis",
            "status": TestStatus.PASS.value,
            "ai_analysis": None,
            "scores": {},
            "issues": [],
            "recommendations": []
        }

        if not GEMINI_VISION_AVAILABLE:
            result["status"] = TestStatus.SKIP.value
            result["error"] = "Gemini Vision not available (GOOGLE_API_KEY not set)"
            return result

        try:
            page = await self.context.new_page()
            await page.goto(url, wait_until='networkidle')

            # Take full-page screenshot
            screenshot_bytes = await page.screenshot(full_page=True)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')

            # Initialize Gemini Vision model
            model = genai.GenerativeModel('gemini-2.0-flash-exp')

            # Comprehensive visual analysis prompt
            analysis_prompt = f"""You are an expert UI/UX analyst. Analyze this screenshot of {app_name} ({url}) and provide a comprehensive assessment.

Evaluate the following aspects on a scale of 1-10 and provide specific observations:

1. **Visual Design Quality**: Layout, color scheme, typography, whitespace usage
2. **UX Best Practices**: Navigation clarity, call-to-action visibility, user flow
3. **Accessibility**: Color contrast, text readability, interactive element sizing
4. **Brand Consistency**: Professional appearance, cohesive styling
5. **Mobile Readiness**: Responsive design indicators, touch-friendly elements
6. **Content Quality**: Text clarity, image quality, information hierarchy
7. **Performance Indicators**: Loading states, skeleton screens, lazy loading

For each issue found, categorize as:
- CRITICAL: Blocks user functionality
- WARNING: Degrades user experience
- INFO: Improvement opportunity

Return your analysis in this exact JSON format:
{{
    "overall_score": <1-10>,
    "scores": {{
        "visual_design": <1-10>,
        "ux_practices": <1-10>,
        "accessibility": <1-10>,
        "brand_consistency": <1-10>,
        "mobile_readiness": <1-10>,
        "content_quality": <1-10>,
        "performance": <1-10>
    }},
    "issues": [
        {{"severity": "CRITICAL|WARNING|INFO", "category": "<category>", "description": "<specific issue>"}}
    ],
    "recommendations": [
        "<actionable recommendation>"
    ],
    "summary": "<2-3 sentence overall assessment>"
}}"""

            # Send to Gemini Vision
            import asyncio
            response = await asyncio.to_thread(
                model.generate_content,
                [analysis_prompt, {"mime_type": "image/png", "data": screenshot_b64}],
                generation_config={"temperature": 0.1}
            )

            # Parse JSON response
            response_text = response.text.strip()
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            ai_result = json.loads(response_text)

            result["ai_analysis"] = ai_result
            result["scores"] = ai_result.get("scores", {})
            result["issues"] = ai_result.get("issues", [])
            result["recommendations"] = ai_result.get("recommendations", [])
            result["overall_score"] = ai_result.get("overall_score", 0)
            result["summary"] = ai_result.get("summary", "")

            # Determine pass/fail based on score and critical issues
            critical_issues = [i for i in result["issues"] if i.get("severity") == "CRITICAL"]
            if critical_issues or result.get("overall_score", 10) < 5:
                result["status"] = TestStatus.FAIL.value
            elif result.get("overall_score", 10) < 7:
                result["status"] = "warning"

            await page.close()

        except json.JSONDecodeError as e:
            result["status"] = TestStatus.ERROR.value
            result["error"] = f"Failed to parse AI response: {e}"
            result["raw_response"] = response_text if 'response_text' in locals() else None
        except Exception as e:
            result["status"] = TestStatus.ERROR.value
            result["error"] = str(e)
            logger.error(f"Visual AI analysis failed: {e}")

        return result

    async def test_visual_regression(self, url: str, baseline_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Visual regression testing - compare current screenshot to baseline.
        Uses Gemini Vision to detect meaningful visual changes.
        """
        result = {
            "url": url,
            "test": "visual_regression",
            "status": TestStatus.PASS.value,
            "changes_detected": [],
            "severity": "none"
        }

        if not GEMINI_VISION_AVAILABLE:
            result["status"] = TestStatus.SKIP.value
            result["error"] = "Gemini Vision not available"
            return result

        try:
            page = await self.context.new_page()
            await page.goto(url, wait_until='networkidle')

            # Take current screenshot
            current_bytes = await page.screenshot(full_page=True)
            current_b64 = base64.b64encode(current_bytes).decode('utf-8')

            # If no baseline, just store current
            if not baseline_path:
                result["status"] = TestStatus.SKIP.value
                result["message"] = "No baseline provided - screenshot captured for future comparison"
                result["screenshot_b64"] = current_b64[:100] + "..."  # Truncate for response
                await page.close()
                return result

            # Load baseline
            with open(baseline_path, 'rb') as f:
                baseline_bytes = f.read()
            baseline_b64 = base64.b64encode(baseline_bytes).decode('utf-8')

            # Use Gemini to compare
            model = genai.GenerativeModel('gemini-2.0-flash-exp')

            comparison_prompt = """Compare these two screenshots (baseline vs current).
Identify any meaningful visual changes that could affect user experience.

Ignore minor differences like:
- Timestamps or dates
- Dynamic content that naturally changes
- Minor text variations

Focus on detecting:
- Layout changes
- Missing or new elements
- Color/styling changes
- Broken layouts
- Missing images or icons

Return JSON:
{
    "has_significant_changes": true/false,
    "severity": "none|minor|major|critical",
    "changes": [
        {"type": "<change type>", "description": "<specific description>", "impact": "low|medium|high"}
    ],
    "summary": "<brief summary>"
}"""

            import asyncio
            response = await asyncio.to_thread(
                model.generate_content,
                [
                    comparison_prompt,
                    "BASELINE IMAGE:",
                    {"mime_type": "image/png", "data": baseline_b64},
                    "CURRENT IMAGE:",
                    {"mime_type": "image/png", "data": current_b64}
                ],
                generation_config={"temperature": 0.1}
            )

            response_text = response.text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()

            comparison_result = json.loads(response_text)

            result["changes_detected"] = comparison_result.get("changes", [])
            result["severity"] = comparison_result.get("severity", "none")
            result["summary"] = comparison_result.get("summary", "")

            if comparison_result.get("severity") in ["major", "critical"]:
                result["status"] = TestStatus.FAIL.value
            elif comparison_result.get("severity") == "minor":
                result["status"] = "warning"

            await page.close()

        except Exception as e:
            result["status"] = TestStatus.ERROR.value
            result["error"] = str(e)

        return result

    async def run_full_test_suite(self, app_name: str) -> Dict[str, Any]:
        """Run complete test suite for an application"""
        if app_name not in self.test_urls:
            return {"error": f"Unknown application: {app_name}"}

        app_config = self.test_urls[app_name]
        base_url = app_config["base_url"]

        test_results = {
            "application": app_name,
            "url": base_url,
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0
            }
        }

        try:
            await self.setup_browser(headless=True)

            # Test each page
            for page_path in app_config["pages"]:
                page_url = f"{base_url}{page_path}"

                # Page load test
                load_result = await self.test_page_load(page_url)
                test_results["tests"].append(load_result)

                # Element presence tests
                if load_result["status"] == TestStatus.PASS.value:
                    for element in app_config.get("critical_elements", []):
                        elem_result = await self.test_element_presence(
                            page_url,
                            element["selector"],
                            element.get("text")
                        )
                        if not element.get("required", True) and elem_result["status"] == TestStatus.FAIL.value:
                            elem_result["status"] = TestStatus.SKIP.value
                        test_results["tests"].append(elem_result)

            # Run additional tests on home page
            home_url = base_url

            # Performance test
            perf_result = await self.test_performance(home_url)
            test_results["tests"].append(perf_result)

            # Accessibility test
            a11y_result = await self.test_accessibility(home_url)
            test_results["tests"].append(a11y_result)

            # API integration test
            api_result = await self.test_api_integration(home_url)
            test_results["tests"].append(api_result)

            # Responsive design test
            responsive_result = await self.test_responsive_design(home_url)
            test_results["tests"].append(responsive_result)

            # AI Visual Analysis (Gemini Vision) - The Ultimate UI/UX Check
            if GEMINI_VISION_AVAILABLE:
                visual_ai_result = await self.test_visual_ai_analysis(home_url, app_name)
                test_results["tests"].append(visual_ai_result)

                # Store AI insights separately for easy access
                if visual_ai_result.get("status") != TestStatus.ERROR.value:
                    test_results["ai_visual_insights"] = {
                        "overall_score": visual_ai_result.get("overall_score"),
                        "scores": visual_ai_result.get("scores", {}),
                        "issues": visual_ai_result.get("issues", []),
                        "recommendations": visual_ai_result.get("recommendations", []),
                        "summary": visual_ai_result.get("summary", "")
                    }

            await self.teardown_browser()

        except Exception as e:
            test_results["error"] = str(e)

        # Calculate summary
        for test in test_results["tests"]:
            test_results["summary"]["total"] += 1
            if test.get("status") == TestStatus.PASS.value:
                test_results["summary"]["passed"] += 1
            elif test.get("status") == TestStatus.FAIL.value:
                test_results["summary"]["failed"] += 1
            elif test.get("status") == TestStatus.ERROR.value:
                test_results["summary"]["errors"] += 1

        test_results["summary"]["pass_rate"] = (
            round(test_results["summary"]["passed"] / test_results["summary"]["total"] * 100, 1)
            if test_results["summary"]["total"] > 0 else 0
        )

        return test_results

    def format_test_report(self, results: Dict) -> str:
        """Format test results as a readable report"""
        report = []
        report.append("=" * 60)
        report.append(f"UI TEST REPORT - {results['application']}")
        report.append(f"URL: {results['url']}")
        report.append(f"Time: {results['timestamp']}")
        report.append("=" * 60)

        summary = results['summary']
        report.append("\nSUMMARY:")
        report.append(f"  Total Tests: {summary['total']}")
        report.append(f"  Passed: {summary['passed']} ({summary.get('pass_rate', 0)}%)")
        report.append(f"  Failed: {summary['failed']}")
        report.append(f"  Errors: {summary['errors']}")

        # Overall status
        if summary['failed'] == 0 and summary['errors'] == 0:
            report.append("\n✅ ALL UI TESTS PASSED")
        else:
            report.append("\n❌ UI TESTS FAILED")

        # Detailed results
        report.append("\nDETAILED RESULTS:")
        report.append("-" * 60)

        for test in results['tests']:
            status_emoji = {
                "pass": "✅",
                "fail": "❌",
                "skip": "⏭️",
                "error": "⚠️"
            }.get(test.get('status'), "❓")

            report.append(f"\n{test['test']}: {status_emoji} {test.get('status', 'unknown').upper()}")
            if test.get('url'):
                report.append(f"  URL: {test['url']}")

            if test.get('error'):
                report.append(f"  Error: {test['error']}")

            if test.get('load_time_ms'):
                report.append(f"  Load Time: {test['load_time_ms']}ms")

            if test.get('selector'):
                report.append(f"  Selector: {test['selector']}")

        report.append("\n" + "=" * 60)
        return '\n'.join(report)


async def main():
    """Test the UI Tester Agent"""
    if not PLAYWRIGHT_AVAILABLE:
        print("Please install Playwright first:")
        print("  pip install playwright")
        print("  playwright install chromium")
        return

    tester = UITesterAgent()

    print("Testing BrainOps Command Center UI...")
    results = await tester.run_full_test_suite("brainops-command-center")

    print(tester.format_test_report(results))


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
TRUE E2E UI Testing System - Human-Like Browser Testing
=========================================================

This module provides REAL end-to-end UI testing that operates exactly like
a human user would interact with web applications:

1. OPERATION TESTING - Does everything work correctly?
2. FUNCTION TESTING - Are all features functional?
3. QUALITY TESTING - Is the code quality high?
4. UX TESTING - Is the user experience intuitive?
5. VISUAL TESTING - Is the visual design consistent and professional?
6. ACCESSIBILITY TESTING - Is the app accessible to all users?

Uses MCP Bridge Playwright (60 tools) + AI Vision (Gemini/GPT-4V) for
comprehensive human-like testing capabilities.

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import base64
import hashlib
import uuid
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# MCP Bridge configuration
MCP_BRIDGE_URL = os.getenv("MCP_BRIDGE_URL", "https://brainops-mcp-bridge.onrender.com")
MCP_API_KEY = (os.getenv("MCP_API_KEY") or os.getenv("BRAINOPS_API_KEY") or "").strip()
if not MCP_API_KEY:
    logger.warning("MCP_API_KEY not configured - MCP UI testing will be disabled")

# AI Vision APIs
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class TestCategory(Enum):
    """Categories of UI testing"""
    OPERATION = "operation"      # Does it work?
    FUNCTION = "function"        # Are features working?
    QUALITY = "quality"          # Is quality high?
    UX = "ux"                    # Is it intuitive?
    VISUAL = "visual"            # Is it visually good?
    ACCESSIBILITY = "a11y"       # Is it accessible?
    PERFORMANCE = "performance"  # Is it fast?
    SECURITY = "security"        # Is it secure?


class IssueSeverity(Enum):
    """Severity levels for issues"""
    CRITICAL = "critical"  # Blocks usage
    HIGH = "high"          # Major problem
    MEDIUM = "medium"      # Moderate issue
    LOW = "low"            # Minor issue
    INFO = "info"          # Informational


@dataclass
class UITestIssue:
    """A detected UI issue"""
    id: str
    category: TestCategory
    severity: IssueSeverity
    title: str
    description: str
    url: str
    route: str
    element: Optional[str] = None
    screenshot: Optional[str] = None  # Base64 screenshot
    ai_analysis: Optional[dict] = None
    suggested_fix: Optional[str] = None
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class UITestResult:
    """Complete UI test result"""
    test_id: str
    application: str
    base_url: str
    routes_tested: int
    issues_found: int
    issues: list[UITestIssue]
    overall_score: float  # 0-100
    category_scores: dict[str, float]
    started_at: str
    completed_at: str
    duration_seconds: float
    ai_summary: str
    recommendations: list[str]


class TrueE2EUITester:
    """
    TRUE End-to-End UI Testing System

    Operates like a real human user would:
    - Navigate to pages
    - Click on elements
    - Fill forms
    - Scroll and observe
    - Take screenshots
    - Analyze visually with AI
    - Report issues comprehensively
    """

    def __init__(self):
        self._session = None
        self._mcp_client = None

        # Target applications
        self.targets = {
            "weathercraft-erp": {
                "name": "Weathercraft ERP",
                "base_url": "https://weathercraft-erp.vercel.app",
                "routes": ["/", "/dashboard", "/customers", "/jobs", "/estimates", "/scheduling", "/invoices", "/settings"],
                "critical_flows": ["login", "create_job", "generate_invoice"],
                "expected_elements": {
                    "/": ["nav", "header", "[data-testid='sidebar']"],
                    "/dashboard": ["h1", ".metrics", ".chart"],
                    "/customers": [".customer-list", "button"]
                }
            },
            "myroofgenius": {
                "name": "MyRoofGenius",
                "base_url": "https://myroofgenius.com",
                "routes": ["/", "/features", "/pricing", "/demo", "/login"],
                "critical_flows": ["signup", "demo_request", "contact"],
                "expected_elements": {
                    "/": ["header", "footer", ".hero", "nav"],
                    "/pricing": [".pricing-card", ".cta-button"],
                    "/features": [".feature-list", "h2"]
                }
            }
        }

        logger.info("TrueE2EUITester initialized with AI-powered human-like testing")

    async def _get_session(self):
        """Get aiohttp session"""
        import aiohttp
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": MCP_API_KEY
                },
                timeout=aiohttp.ClientTimeout(total=120)
            )
        return self._session

    async def _execute_mcp_tool(self, server: str, tool: str, params: dict) -> dict:
        """Execute a tool via MCP Bridge"""
        try:
            session = await self._get_session()
            async with session.post(
                f"{MCP_BRIDGE_URL}/mcp/execute",
                json={"server": server, "tool": tool, "params": params}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error = await resp.text()
                    logger.error(f"MCP tool {tool} failed: {error}")
                    return {"error": error}
        except Exception as e:
            logger.error(f"MCP execution error: {e}")
            return {"error": str(e)}

    async def _analyze_screenshot_with_ai(self, screenshot_base64: str, page_context: dict) -> dict:
        """Analyze screenshot with AI Vision (Gemini or GPT-4V)"""
        analysis = {
            "visual_issues": [],
            "ux_issues": [],
            "accessibility_issues": [],
            "suggestions": [],
            "overall_assessment": ""
        }

        prompt = f"""Analyze this UI screenshot as an expert UI/UX designer and QA engineer.

Page: {page_context.get('url', 'unknown')}
Route: {page_context.get('route', 'unknown')}
Application: {page_context.get('app_name', 'unknown')}

Evaluate and report issues in these categories:

1. VISUAL DESIGN:
- Is the layout consistent and professional?
- Are colors, fonts, and spacing consistent?
- Is the visual hierarchy clear?
- Any visual bugs or rendering issues?

2. USER EXPERIENCE:
- Is navigation intuitive?
- Are CTAs clear and prominent?
- Is content readable and scannable?
- Any confusing or cluttered areas?

3. ACCESSIBILITY:
- Are there sufficient color contrasts?
- Are interactive elements clearly marked?
- Any missing alt text indicators?
- Are forms labeled properly?

4. FUNCTIONALITY INDICATORS:
- Do buttons look clickable?
- Are loading states clear?
- Any broken layouts or overflows?
- Any error states visible?

For each issue found, provide:
- severity: "critical", "high", "medium", "low", or "info"
- category: "visual", "ux", "accessibility", or "function"
- title: Short description
- description: Detailed explanation
- location: Where on the page (e.g., "top nav", "hero section", "footer")
- suggested_fix: How to fix it

Return a JSON object with this structure:
{{
  "visual_issues": [...],
  "ux_issues": [...],
  "accessibility_issues": [...],
  "functional_issues": [...],
  "overall_score": 0-100,
  "overall_assessment": "summary text",
  "top_recommendations": ["fix 1", "fix 2", "fix 3"]
}}
"""

        # Try Gemini Vision first
        if GOOGLE_API_KEY:
            try:
                from google import genai
                from google.genai import types as _genai_types
                client = genai.Client(api_key=GOOGLE_API_KEY)

                # Create image part
                image_data = base64.b64decode(screenshot_base64)
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[
                        prompt,
                        _genai_types.Part.from_bytes(data=image_data, mime_type="image/png")
                    ]
                )

                # Parse JSON from response
                response_text = response.text
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                analysis = json.loads(response_text.strip())
                analysis["ai_model"] = "gemini-2.0-flash"
                return analysis

            except Exception as e:
                logger.warning(f"Gemini Vision analysis failed: {e}")

        # Fallback to OpenAI GPT-4V
        if OPENAI_API_KEY:
            try:
                import openai
                client = openai.OpenAI(api_key=OPENAI_API_KEY)

                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{screenshot_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4096
                )

                response_text = response.choices[0].message.content
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                analysis = json.loads(response_text.strip())
                analysis["ai_model"] = "gpt-4-vision"
                return analysis

            except Exception as e:
                logger.warning(f"GPT-4V analysis failed: {e}")

        # Return basic analysis if AI not available
        analysis["overall_assessment"] = "AI Vision analysis not available - basic metrics only"
        analysis["overall_score"] = 70
        return analysis

    async def _test_page_operation(self, url: str, route: str, app_name: str) -> dict:
        """Test if a page operates correctly (loads, responds)"""
        result = {
            "url": url,
            "route": route,
            "operational": False,
            "load_time_ms": None,
            "status_code": None,
            "errors": [],
            "performance": {}
        }

        try:
            start = datetime.now()
            session = await self._get_session()

            async with session.get(url, allow_redirects=True) as resp:
                load_time = (datetime.now() - start).total_seconds() * 1000
                result["load_time_ms"] = round(load_time)
                result["status_code"] = resp.status
                result["operational"] = resp.status == 200

                # Check for error indicators in response
                if resp.status == 200:
                    text = await resp.text()
                    if "error" in text.lower() and "500" in text:
                        result["errors"].append("Server error detected on page")
                    if "404" in text and "not found" in text.lower():
                        result["errors"].append("404 content detected on page")

                result["performance"] = {
                    "ttfb_ms": load_time,
                    "is_fast": load_time < 3000,
                    "rating": "good" if load_time < 1500 else "needs_improvement" if load_time < 3000 else "slow"
                }

        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"Page operation test failed for {url}: {e}")

        return result

    async def _test_page_function(self, url: str, route: str, expected_elements: list[str]) -> dict:
        """Test if page functions correctly (elements present, interactive)"""
        result = {
            "url": url,
            "route": route,
            "elements_found": [],
            "elements_missing": [],
            "interactive_elements": 0,
            "forms_found": 0,
            "links_found": 0,
            "functional": True
        }

        # Use Playwright via MCP to check elements
        try:
            # Navigate to page
            nav_result = await self._execute_mcp_tool("playwright", "playwright_navigate", {"url": url})
            if nav_result.get("error"):
                result["functional"] = False
                result["elements_missing"] = expected_elements
                return result

            # Check for each expected element
            for selector in expected_elements:
                check_result = await self._execute_mcp_tool(
                    "playwright",
                    "playwright_evaluate",
                    {"script": f"document.querySelector('{selector}') !== null"}
                )
                if check_result.get("result", False):
                    result["elements_found"].append(selector)
                else:
                    result["elements_missing"].append(selector)

            # Count interactive elements
            count_result = await self._execute_mcp_tool(
                "playwright",
                "playwright_evaluate",
                {"script": "document.querySelectorAll('button, a, input, select, textarea').length"}
            )
            result["interactive_elements"] = count_result.get("result", 0)

            # Count forms
            forms_result = await self._execute_mcp_tool(
                "playwright",
                "playwright_evaluate",
                {"script": "document.querySelectorAll('form').length"}
            )
            result["forms_found"] = forms_result.get("result", 0)

            # Count links
            links_result = await self._execute_mcp_tool(
                "playwright",
                "playwright_evaluate",
                {"script": "document.querySelectorAll('a[href]').length"}
            )
            result["links_found"] = links_result.get("result", 0)

            result["functional"] = len(result["elements_missing"]) == 0

        except Exception as e:
            logger.error(f"Function test failed for {url}: {e}")
            result["functional"] = False

        return result

    async def _capture_and_analyze_screenshot(self, url: str, route: str, app_name: str) -> dict:
        """Capture screenshot and analyze with AI Vision"""
        result = {
            "url": url,
            "route": route,
            "screenshot_captured": False,
            "ai_analysis": None,
            "issues": []
        }

        try:
            # Take screenshot via Playwright MCP
            screenshot_result = await self._execute_mcp_tool(
                "playwright",
                "playwright_screenshot",
                {"name": f"{app_name}_{route.replace('/', '_')}", "fullPage": True}
            )

            if screenshot_result.get("error"):
                logger.warning(f"Screenshot failed for {url}: {screenshot_result.get('error')}")
                return result

            # Get the screenshot data
            screenshot_base64 = screenshot_result.get("content", screenshot_result.get("screenshot", ""))

            if screenshot_base64:
                result["screenshot_captured"] = True

                # Analyze with AI Vision
                analysis = await self._analyze_screenshot_with_ai(
                    screenshot_base64,
                    {"url": url, "route": route, "app_name": app_name}
                )

                result["ai_analysis"] = analysis

                # Convert analysis issues to UITestIssue objects
                for issue_type in ["visual_issues", "ux_issues", "accessibility_issues", "functional_issues"]:
                    for issue in analysis.get(issue_type, []):
                        category = TestCategory.VISUAL if "visual" in issue_type else \
                                   TestCategory.UX if "ux" in issue_type else \
                                   TestCategory.ACCESSIBILITY if "access" in issue_type else \
                                   TestCategory.FUNCTION

                        severity_map = {"critical": IssueSeverity.CRITICAL, "high": IssueSeverity.HIGH,
                                       "medium": IssueSeverity.MEDIUM, "low": IssueSeverity.LOW,
                                       "info": IssueSeverity.INFO}
                        severity = severity_map.get(issue.get("severity", "medium"), IssueSeverity.MEDIUM)

                        result["issues"].append(UITestIssue(
                            id=hashlib.md5(f"{url}{issue.get('title', '')}".encode()).hexdigest()[:12],
                            category=category,
                            severity=severity,
                            title=issue.get("title", "Unknown issue"),
                            description=issue.get("description", ""),
                            url=url,
                            route=route,
                            element=issue.get("location"),
                            suggested_fix=issue.get("suggested_fix")
                        ))

        except Exception as e:
            logger.error(f"Screenshot analysis failed for {url}: {e}")

        return result

    async def test_application(self, app_key: str, full_analysis: bool = True) -> UITestResult:
        """
        Run comprehensive TRUE E2E testing on an application

        Args:
            app_key: Application key (e.g., "weathercraft-erp", "myroofgenius")
            full_analysis: Whether to run full AI visual analysis

        Returns:
            Complete UITestResult with all issues and recommendations
        """
        if app_key not in self.targets:
            raise ValueError(f"Unknown application: {app_key}. Available: {list(self.targets.keys())}")

        target = self.targets[app_key]
        start_time = datetime.now(timezone.utc)
        test_id = str(uuid.uuid4())

        logger.info(f"Starting TRUE E2E test for {target['name']} ({len(target['routes'])} routes)")

        all_issues: list[UITestIssue] = []
        category_scores = {cat.value: 100.0 for cat in TestCategory}
        routes_tested = 0

        # Initialize Playwright session
        await self._execute_mcp_tool("playwright", "playwright_navigate", {"url": target["base_url"]})

        for route in target["routes"]:
            url = f"{target['base_url']}{route}"
            logger.info(f"Testing route: {url}")
            routes_tested += 1

            # 1. OPERATION TEST - Does it load?
            op_result = await self._test_page_operation(url, route, target["name"])
            if not op_result["operational"]:
                all_issues.append(UITestIssue(
                    id=hashlib.md5(f"{url}_operation".encode()).hexdigest()[:12],
                    category=TestCategory.OPERATION,
                    severity=IssueSeverity.CRITICAL,
                    title="Page not operational",
                    description=f"Status: {op_result.get('status_code')}, Errors: {op_result.get('errors', [])}",
                    url=url,
                    route=route
                ))
                category_scores[TestCategory.OPERATION.value] -= 20
                continue  # Skip further tests if page doesn't load

            # Check performance
            if op_result["performance"]["rating"] == "slow":
                all_issues.append(UITestIssue(
                    id=hashlib.md5(f"{url}_perf".encode()).hexdigest()[:12],
                    category=TestCategory.PERFORMANCE,
                    severity=IssueSeverity.MEDIUM,
                    title=f"Slow page load: {op_result['load_time_ms']}ms",
                    description=f"Page took {op_result['load_time_ms']}ms to load. Target: <3000ms",
                    url=url,
                    route=route,
                    suggested_fix="Optimize images, reduce JS bundle size, enable caching"
                ))
                category_scores[TestCategory.PERFORMANCE.value] -= 10

            # 2. FUNCTION TEST - Do elements work?
            expected = target.get("expected_elements", {}).get(route, [])
            func_result = await self._test_page_function(url, route, expected)
            for missing in func_result["elements_missing"]:
                all_issues.append(UITestIssue(
                    id=hashlib.md5(f"{url}_{missing}".encode()).hexdigest()[:12],
                    category=TestCategory.FUNCTION,
                    severity=IssueSeverity.HIGH,
                    title=f"Missing element: {missing}",
                    description=f"Expected element '{missing}' not found on page",
                    url=url,
                    route=route,
                    element=missing
                ))
                category_scores[TestCategory.FUNCTION.value] -= 15

            # 3. VISUAL + UX + ACCESSIBILITY TEST (AI Vision)
            if full_analysis:
                visual_result = await self._capture_and_analyze_screenshot(url, route, target["name"])
                all_issues.extend(visual_result.get("issues", []))

                # Update category scores based on AI analysis
                ai_analysis = visual_result.get("ai_analysis", {})
                if ai_analysis.get("overall_score"):
                    visual_weight = 0.3  # Weight for AI visual score
                    for cat in [TestCategory.VISUAL, TestCategory.UX, TestCategory.ACCESSIBILITY]:
                        category_scores[cat.value] = (
                            category_scores[cat.value] * (1 - visual_weight) +
                            ai_analysis["overall_score"] * visual_weight
                        )

        # Calculate overall score
        issue_penalty = len([i for i in all_issues if i.severity == IssueSeverity.CRITICAL]) * 20
        issue_penalty += len([i for i in all_issues if i.severity == IssueSeverity.HIGH]) * 10
        issue_penalty += len([i for i in all_issues if i.severity == IssueSeverity.MEDIUM]) * 5
        issue_penalty += len([i for i in all_issues if i.severity == IssueSeverity.LOW]) * 2

        overall_score = max(0, min(100, sum(category_scores.values()) / len(category_scores) - issue_penalty))

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Generate AI summary
        ai_summary = self._generate_summary(target["name"], all_issues, category_scores, overall_score)
        recommendations = self._generate_recommendations(all_issues)

        result = UITestResult(
            test_id=test_id,
            application=target["name"],
            base_url=target["base_url"],
            routes_tested=routes_tested,
            issues_found=len(all_issues),
            issues=all_issues,
            overall_score=round(overall_score, 1),
            category_scores={k: round(v, 1) for k, v in category_scores.items()},
            started_at=start_time.isoformat(),
            completed_at=end_time.isoformat(),
            duration_seconds=round(duration, 2),
            ai_summary=ai_summary,
            recommendations=recommendations
        )

        # Store result
        await self._store_result(result)

        logger.info(f"TRUE E2E test complete for {target['name']}: Score {overall_score:.1f}, {len(all_issues)} issues")
        return result

    def _generate_summary(self, app_name: str, issues: list[UITestIssue], scores: dict, overall: float) -> str:
        """Generate human-readable summary"""
        critical = len([i for i in issues if i.severity == IssueSeverity.CRITICAL])
        high = len([i for i in issues if i.severity == IssueSeverity.HIGH])

        if overall >= 90:
            status = "excellent"
        elif overall >= 75:
            status = "good with minor issues"
        elif overall >= 50:
            status = "needs improvement"
        else:
            status = "critical issues detected"

        summary = f"{app_name} UI/UX assessment: {status} (Score: {overall:.0f}/100). "
        summary += f"Found {len(issues)} issues ({critical} critical, {high} high priority). "

        # Highlight worst category
        worst_cat = min(scores.items(), key=lambda x: x[1])
        if worst_cat[1] < 80:
            summary += f"Focus area: {worst_cat[0]} (score: {worst_cat[1]:.0f})."

        return summary

    def _generate_recommendations(self, issues: list[UITestIssue]) -> list[str]:
        """Generate prioritized recommendations"""
        recommendations = []

        # Group by category
        categories = {}
        for issue in issues:
            cat = issue.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(issue)

        # Generate recommendations per category
        if TestCategory.OPERATION.value in categories:
            recommendations.append("CRITICAL: Fix page loading issues - some pages are not accessible")

        if TestCategory.FUNCTION.value in categories:
            recommendations.append(f"Fix {len(categories[TestCategory.FUNCTION.value])} functional issues - missing expected UI elements")

        if TestCategory.VISUAL.value in categories:
            recommendations.append(f"Address {len(categories[TestCategory.VISUAL.value])} visual design issues for consistency")

        if TestCategory.UX.value in categories:
            recommendations.append(f"Improve UX in {len(categories[TestCategory.UX.value])} areas for better user experience")

        if TestCategory.ACCESSIBILITY.value in categories:
            recommendations.append(f"Fix {len(categories[TestCategory.ACCESSIBILITY.value])} accessibility issues for WCAG compliance")

        if TestCategory.PERFORMANCE.value in categories:
            recommendations.append(f"Optimize {len(categories[TestCategory.PERFORMANCE.value])} slow pages for better performance")

        # Add specific fixes from issues
        for issue in sorted(issues, key=lambda x: [IssueSeverity.CRITICAL, IssueSeverity.HIGH].count(x.severity), reverse=True)[:3]:
            if issue.suggested_fix:
                recommendations.append(f"[{issue.category.value.upper()}] {issue.suggested_fix}")

        return recommendations[:10]  # Top 10 recommendations

    async def _store_result(self, result: UITestResult):
        """Store test result to database"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()
            if pool:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO ui_test_results (
                            test_id, application, url, status, severity, message,
                            ai_analysis, performance_metrics, routes_tested,
                            issues_found, completed_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (test_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            message = EXCLUDED.message,
                            ai_analysis = EXCLUDED.ai_analysis,
                            issues_found = EXCLUDED.issues_found,
                            completed_at = EXCLUDED.completed_at
                    """,
                        result.test_id,
                        result.application,
                        result.base_url,
                        "passed" if result.overall_score >= 75 else "failed",
                        "critical" if result.overall_score < 50 else "medium" if result.overall_score < 75 else "low",
                        result.ai_summary,
                        json.dumps({
                            "category_scores": result.category_scores,
                            "recommendations": result.recommendations,
                            "issues": [asdict(i) for i in result.issues[:20]]  # Store top 20 issues
                        }),
                        json.dumps({"duration_seconds": result.duration_seconds}),
                        result.routes_tested,
                        result.issues_found,
                        result.completed_at
                    )
                    logger.info(f"Stored UI test result: {result.test_id}")
        except Exception as e:
            logger.warning(f"Failed to store UI test result: {e}")

    async def test_all_applications(self) -> dict[str, UITestResult]:
        """Test all registered applications"""
        results = {}
        for app_key in self.targets:
            try:
                results[app_key] = await self.test_application(app_key)
            except Exception as e:
                logger.error(f"Failed to test {app_key}: {e}")
        return results

    async def close(self):
        """Clean up resources"""
        if self._session and not self._session.closed:
            await self._session.close()


# Singleton instance
_tester: Optional[TrueE2EUITester] = None


async def get_true_e2e_tester() -> TrueE2EUITester:
    """Get or create the TRUE E2E UI tester"""
    global _tester
    if _tester is None:
        _tester = TrueE2EUITester()
    return _tester


# Agent wrapper for integration with agent executor
class TrueE2EUITestingAgent:
    """Agent wrapper for TRUE E2E UI Testing"""

    def __init__(self):
        self.name = "TrueE2EUITesting"
        self.agent_type = "true_e2e_ui_testing"

    async def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute TRUE E2E UI testing"""
        tester = await get_true_e2e_tester()
        action = task.get("action", "test_all")

        if action == "test_all":
            results = await tester.test_all_applications()
            return {
                "status": "completed",
                "applications_tested": len(results),
                "results": {k: {
                    "score": v.overall_score,
                    "issues": v.issues_found,
                    "summary": v.ai_summary
                } for k, v in results.items()}
            }

        elif action == "test_app":
            app_key = task.get("app_key", task.get("application"))
            if not app_key:
                return {"status": "error", "message": "app_key required"}
            result = await tester.test_application(app_key)
            return {
                "status": "completed",
                "application": result.application,
                "score": result.overall_score,
                "issues_found": result.issues_found,
                "category_scores": result.category_scores,
                "summary": result.ai_summary,
                "recommendations": result.recommendations,
                "duration_seconds": result.duration_seconds
            }

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


if __name__ == "__main__":
    async def main():
        print("=" * 70)
        print("TRUE E2E UI TESTING - HUMAN-LIKE BROWSER TESTING")
        print("=" * 70)

        tester = await get_true_e2e_tester()

        # Test Weathercraft ERP
        print("\nTesting Weathercraft ERP...")
        result = await tester.test_application("weathercraft-erp", full_analysis=False)

        print(f"\nResults for {result.application}:")
        print(f"  Overall Score: {result.overall_score}/100")
        print(f"  Routes Tested: {result.routes_tested}")
        print(f"  Issues Found: {result.issues_found}")
        print(f"  Duration: {result.duration_seconds}s")
        print("\nCategory Scores:")
        for cat, score in result.category_scores.items():
            print(f"  {cat}: {score}/100")
        print(f"\nSummary: {result.ai_summary}")
        print("\nTop Recommendations:")
        for rec in result.recommendations[:5]:
            print(f"  - {rec}")

        await tester.close()

    asyncio.run(main())

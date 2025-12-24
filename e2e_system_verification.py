"""
E2E System Verification Engine
==============================
Comprehensive end-to-end verification of ALL BrainOps systems.
Iterates through every endpoint, validates responses, and ensures 100% operation.

This is the SINGLE SOURCE OF TRUTH for system health.
No partial operations allowed - everything must be 100% or flagged for immediate action.
"""

import asyncio
import aiohttp
import json
import time
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

# Production configuration
BRAINOPS_API_URL = os.getenv("BRAINOPS_API_URL", "https://brainops-ai-agents.onrender.com")
BRAINOPS_BACKEND_URL = os.getenv("BRAINOPS_BACKEND_URL", "https://brainops-backend-prod.onrender.com")
ERP_URL = os.getenv("ERP_URL", "https://weathercraft-erp.vercel.app")
MRG_URL = os.getenv("MRG_URL", "https://myroofgenius.com")
MCP_BRIDGE_URL = os.getenv("MCP_BRIDGE_URL", "https://brainops-mcp-bridge.onrender.com")
API_KEY = os.getenv("BRAINOPS_API_KEY", "brainops_prod_key_2025")


class VerificationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    DEGRADED = "degraded"
    TIMEOUT = "timeout"
    UNREACHABLE = "unreachable"
    INVALID_RESPONSE = "invalid_response"


class SystemCategory(Enum):
    CORE_API = "core_api"
    BLEEDING_EDGE = "bleeding_edge"
    FRONTEND = "frontend"
    DATABASE = "database"
    EXTERNAL = "external"
    MCP = "mcp"


@dataclass
class EndpointTest:
    """Definition of an endpoint test"""
    name: str
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    expected_status: int = 200
    expected_fields: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    category: SystemCategory = SystemCategory.CORE_API
    critical: bool = True  # If True, failure means system is NOT 100% operational
    validation_func: Optional[str] = None  # Name of custom validation function


@dataclass
class TestResult:
    """Result of a single endpoint test"""
    test_name: str
    endpoint: str
    status: VerificationStatus
    response_time_ms: float
    status_code: Optional[int] = None
    response_body: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    missing_fields: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SystemVerificationReport:
    """Complete verification report for all systems"""
    report_id: str
    started_at: str
    completed_at: str
    duration_seconds: float
    total_tests: int
    passed: int
    failed: int
    degraded: int
    overall_status: VerificationStatus
    pass_rate: float
    results_by_category: Dict[str, Dict[str, Any]]
    failed_tests: List[TestResult]
    all_results: List[TestResult]
    recommendations: List[str]
    is_100_percent_operational: bool


class E2ESystemVerification:
    """
    End-to-End System Verification Engine

    Verifies 100% operation of ALL BrainOps systems including:
    - Core API endpoints
    - Bleeding-edge systems (Digital Twin, Market Intel, Orchestrator, Self-Healing)
    - Frontend applications (ERP, MyRoofGenius)
    - Database connectivity
    - MCP Bridge and tools
    - External integrations
    """

    def __init__(self):
        self.tests: List[EndpointTest] = []
        self.results: List[TestResult] = []
        self.last_report: Optional[SystemVerificationReport] = None
        self._initialize_tests()

    def _initialize_tests(self):
        """Initialize all endpoint tests"""
        headers = {"X-API-Key": API_KEY}

        # ============================================
        # CORE API ENDPOINTS
        # ============================================
        self.tests.extend([
            EndpointTest(
                name="Health Check",
                url=f"{BRAINOPS_API_URL}/health",
                headers=headers,
                expected_fields=["status", "version", "database", "capabilities"],
                category=SystemCategory.CORE_API,
                critical=True
            ),
            EndpointTest(
                name="Root Endpoint",
                url=f"{BRAINOPS_API_URL}/",
                headers=headers,
                expected_fields=["service", "version", "status"],
                category=SystemCategory.CORE_API,
                critical=True
            ),
            EndpointTest(
                name="Observability Metrics",
                url=f"{BRAINOPS_API_URL}/observability/metrics",
                headers=headers,
                expected_fields=["requests", "cache", "database"],
                category=SystemCategory.CORE_API,
                critical=True
            ),
            EndpointTest(
                name="Agents List",
                url=f"{BRAINOPS_API_URL}/agents",
                headers=headers,
                expected_fields=["agents"],
                category=SystemCategory.CORE_API,
                critical=True
            ),
            EndpointTest(
                name="AI Providers Status",
                url=f"{BRAINOPS_API_URL}/ai/providers/status",
                headers=headers,
                expected_fields=[],  # Structure varies
                category=SystemCategory.CORE_API,
                critical=False
            ),
            EndpointTest(
                name="Systems Usage",
                url=f"{BRAINOPS_API_URL}/systems/usage",
                headers=headers,
                expected_fields=["active_systems"],
                category=SystemCategory.CORE_API,
                critical=True
            ),
            EndpointTest(
                name="Brain Critical Keys",
                url=f"{BRAINOPS_API_URL}/brain/critical",
                headers=headers,
                expected_fields=[],  # Returns array
                category=SystemCategory.CORE_API,
                critical=True
            ),
            EndpointTest(
                name="Brain Memory",
                url=f"{BRAINOPS_API_URL}/brain/critical",
                headers=headers,
                expected_fields=[],
                category=SystemCategory.CORE_API,
                critical=False
            ),
        ])

        # ============================================
        # BLEEDING EDGE - DIGITAL TWIN
        # ============================================
        self.tests.extend([
            EndpointTest(
                name="Digital Twin - Status",
                url=f"{BRAINOPS_API_URL}/digital-twin/status",
                headers=headers,
                expected_fields=["system", "status", "capabilities"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Digital Twin - Dashboard",
                url=f"{BRAINOPS_API_URL}/digital-twin/dashboard",
                headers=headers,
                expected_fields=["summary"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Digital Twin - List Twins",
                url=f"{BRAINOPS_API_URL}/digital-twin/twins",
                headers=headers,
                expected_fields=["twins", "total"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
        ])

        # ============================================
        # BLEEDING EDGE - MARKET INTELLIGENCE
        # ============================================
        self.tests.extend([
            EndpointTest(
                name="Market Intelligence - Status",
                url=f"{BRAINOPS_API_URL}/market-intelligence/status",
                headers=headers,
                expected_fields=["system", "status", "capabilities"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Market Intelligence - Dashboard",
                url=f"{BRAINOPS_API_URL}/market-intelligence/dashboard",
                headers=headers,
                expected_fields=["overview"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Market Intelligence - Trends",
                url=f"{BRAINOPS_API_URL}/market-intelligence/trends",
                headers=headers,
                expected_fields=["trends"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Market Intelligence - Signals",
                url=f"{BRAINOPS_API_URL}/market-intelligence/signals",
                headers=headers,
                expected_fields=["signals"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
        ])

        # ============================================
        # BLEEDING EDGE - SYSTEM ORCHESTRATOR
        # ============================================
        self.tests.extend([
            EndpointTest(
                name="Orchestrator - Status",
                url=f"{BRAINOPS_API_URL}/orchestrator/status",
                headers=headers,
                expected_fields=["system", "status", "capabilities", "scale_capacity"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Orchestrator - Dashboard",
                url=f"{BRAINOPS_API_URL}/orchestrator/dashboard",
                headers=headers,
                expected_fields=["overview", "scale_capacity"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Orchestrator - Systems List",
                url=f"{BRAINOPS_API_URL}/orchestrator/systems",
                headers=headers,
                expected_fields=["systems", "total"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Orchestrator - Groups",
                url=f"{BRAINOPS_API_URL}/orchestrator/groups",
                headers=headers,
                expected_fields=["groups"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
        ])

        # ============================================
        # BLEEDING EDGE - SELF-HEALING
        # ============================================
        self.tests.extend([
            EndpointTest(
                name="Self-Healing - Status",
                url=f"{BRAINOPS_API_URL}/self-healing/status",
                headers=headers,
                expected_fields=["system", "status", "capabilities", "recovery_improvement"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Self-Healing - Dashboard",
                url=f"{BRAINOPS_API_URL}/self-healing/dashboard",
                headers=headers,
                expected_fields=["overview", "autonomy_tiers"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Self-Healing - Active Incidents",
                url=f"{BRAINOPS_API_URL}/self-healing/incidents/active",
                headers=headers,
                expected_fields=["active_incidents", "total"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
            EndpointTest(
                name="Self-Healing - Metrics",
                url=f"{BRAINOPS_API_URL}/self-healing/metrics",
                headers=headers,
                expected_fields=["metrics", "performance"],
                category=SystemCategory.BLEEDING_EDGE,
                critical=True
            ),
        ])

        # ============================================
        # FRONTEND APPLICATIONS
        # ============================================
        self.tests.extend([
            EndpointTest(
                name="ERP - Homepage",
                url=ERP_URL,
                expected_status=200,
                expected_fields=[],
                category=SystemCategory.FRONTEND,
                critical=True,
                timeout_seconds=15.0
            ),
            EndpointTest(
                name="ERP - Login Page",
                url=f"{ERP_URL}/login",
                expected_status=200,
                expected_fields=[],
                category=SystemCategory.FRONTEND,
                critical=True,
                timeout_seconds=15.0
            ),
            EndpointTest(
                name="MyRoofGenius - Homepage",
                url=MRG_URL,
                expected_status=200,
                expected_fields=[],
                category=SystemCategory.FRONTEND,
                critical=True,
                timeout_seconds=15.0
            ),
        ])

        # ============================================
        # BACKEND API
        # ============================================
        self.tests.extend([
            EndpointTest(
                name="Backend - Health",
                url=f"{BRAINOPS_BACKEND_URL}/health",
                expected_status=200,
                expected_fields=[],
                category=SystemCategory.CORE_API,
                critical=True
            ),
        ])

        # ============================================
        # MCP BRIDGE
        # ============================================
        self.tests.extend([
            EndpointTest(
                name="MCP Bridge - Health",
                url=f"{MCP_BRIDGE_URL}/health",
                expected_status=200,
                expected_fields=[],
                category=SystemCategory.MCP,
                critical=True
            ),
            EndpointTest(
                name="MCP Bridge - Status",
                url=f"{MCP_BRIDGE_URL}/status",
                expected_status=200,
                expected_fields=[],
                category=SystemCategory.MCP,
                critical=False
            ),
        ])

    async def _run_single_test(self, test: EndpointTest, session: aiohttp.ClientSession) -> TestResult:
        """Execute a single endpoint test"""
        start_time = time.perf_counter()

        try:
            async with session.request(
                method=test.method,
                url=test.url,
                headers=test.headers,
                json=test.body if test.body else None,
                timeout=aiohttp.ClientTimeout(total=test.timeout_seconds)
            ) as response:
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Try to parse JSON response
                response_body = None
                try:
                    if response.content_type and 'json' in response.content_type:
                        response_body = await response.json()
                except:
                    pass

                # Check status code
                if response.status != test.expected_status:
                    return TestResult(
                        test_name=test.name,
                        endpoint=test.url,
                        status=VerificationStatus.FAILED,
                        response_time_ms=response_time_ms,
                        status_code=response.status,
                        response_body=response_body,
                        error_message=f"Expected status {test.expected_status}, got {response.status}"
                    )

                # Check expected fields
                missing_fields = []
                if test.expected_fields and response_body:
                    for field in test.expected_fields:
                        if field not in response_body:
                            missing_fields.append(field)

                if missing_fields:
                    return TestResult(
                        test_name=test.name,
                        endpoint=test.url,
                        status=VerificationStatus.INVALID_RESPONSE,
                        response_time_ms=response_time_ms,
                        status_code=response.status,
                        response_body=response_body,
                        missing_fields=missing_fields,
                        error_message=f"Missing required fields: {missing_fields}"
                    )

                # Check for error in response body
                if response_body and isinstance(response_body, dict):
                    if response_body.get("status") == "error" or response_body.get("error"):
                        return TestResult(
                            test_name=test.name,
                            endpoint=test.url,
                            status=VerificationStatus.FAILED,
                            response_time_ms=response_time_ms,
                            status_code=response.status,
                            response_body=response_body,
                            error_message=response_body.get("error") or response_body.get("message") or "Error in response"
                        )

                # Check for degraded performance (>20 second response - Render free tier is slow)
                if response_time_ms > 20000:
                    return TestResult(
                        test_name=test.name,
                        endpoint=test.url,
                        status=VerificationStatus.DEGRADED,
                        response_time_ms=response_time_ms,
                        status_code=response.status,
                        response_body=response_body,
                        error_message=f"Slow response: {response_time_ms:.0f}ms"
                    )

                # All checks passed
                return TestResult(
                    test_name=test.name,
                    endpoint=test.url,
                    status=VerificationStatus.PASSED,
                    response_time_ms=response_time_ms,
                    status_code=response.status,
                    response_body=response_body
                )

        except asyncio.TimeoutError:
            return TestResult(
                test_name=test.name,
                endpoint=test.url,
                status=VerificationStatus.TIMEOUT,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
                error_message=f"Timeout after {test.timeout_seconds}s"
            )
        except aiohttp.ClientError as e:
            return TestResult(
                test_name=test.name,
                endpoint=test.url,
                status=VerificationStatus.UNREACHABLE,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
        except Exception as e:
            return TestResult(
                test_name=test.name,
                endpoint=test.url,
                status=VerificationStatus.FAILED,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
                error_message=f"Unexpected error: {str(e)}"
            )

    async def run_full_verification(self) -> SystemVerificationReport:
        """Run complete E2E verification of all systems"""
        started_at = datetime.utcnow()
        report_id = hashlib.md5(started_at.isoformat().encode()).hexdigest()[:12]

        logger.info(f"Starting E2E verification (ID: {report_id}) - {len(self.tests)} tests")

        # Run all tests concurrently
        async with aiohttp.ClientSession() as session:
            tasks = [self._run_single_test(test, session) for test in self.tests]
            self.results = await asyncio.gather(*tasks)

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        # Analyze results
        passed = sum(1 for r in self.results if r.status == VerificationStatus.PASSED)
        failed = sum(1 for r in self.results if r.status in [
            VerificationStatus.FAILED,
            VerificationStatus.TIMEOUT,
            VerificationStatus.UNREACHABLE,
            VerificationStatus.INVALID_RESPONSE
        ])
        degraded = sum(1 for r in self.results if r.status == VerificationStatus.DEGRADED)

        # Check critical tests
        critical_tests = [t for t in self.tests if t.critical]
        critical_results = [r for r, t in zip(self.results, self.tests) if t.critical]
        critical_failures = [r for r in critical_results if r.status != VerificationStatus.PASSED]

        # Determine overall status
        if failed > 0:
            overall_status = VerificationStatus.FAILED
        elif degraded > 0:
            overall_status = VerificationStatus.DEGRADED
        else:
            overall_status = VerificationStatus.PASSED

        # Calculate pass rate
        total = len(self.results)
        pass_rate = (passed / total * 100) if total > 0 else 0

        # Group results by category
        results_by_category = {}
        for category in SystemCategory:
            cat_tests = [(r, t) for r, t in zip(self.results, self.tests) if t.category == category]
            if cat_tests:
                cat_passed = sum(1 for r, t in cat_tests if r.status == VerificationStatus.PASSED)
                cat_failed = sum(1 for r, t in cat_tests if r.status != VerificationStatus.PASSED)
                results_by_category[category.value] = {
                    "total": len(cat_tests),
                    "passed": cat_passed,
                    "failed": cat_failed,
                    "pass_rate": cat_passed / len(cat_tests) * 100 if cat_tests else 0,
                    "status": "passed" if cat_failed == 0 else "failed"
                }

        # Get failed tests
        failed_tests = [r for r in self.results if r.status != VerificationStatus.PASSED]

        # Generate recommendations
        recommendations = self._generate_recommendations(failed_tests, results_by_category)

        # Is system 100% operational?
        is_100_percent = len(critical_failures) == 0 and failed == 0

        report = SystemVerificationReport(
            report_id=report_id,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=duration,
            total_tests=total,
            passed=passed,
            failed=failed,
            degraded=degraded,
            overall_status=overall_status,
            pass_rate=pass_rate,
            results_by_category=results_by_category,
            failed_tests=failed_tests,
            all_results=self.results,
            recommendations=recommendations,
            is_100_percent_operational=is_100_percent
        )

        self.last_report = report

        # Log summary
        if is_100_percent:
            logger.info(f"E2E Verification PASSED: 100% operational ({passed}/{total} tests)")
        else:
            logger.error(f"E2E Verification FAILED: {failed} failures, {degraded} degraded ({pass_rate:.1f}% pass rate)")
            for ft in failed_tests[:5]:
                logger.error(f"  - {ft.test_name}: {ft.error_message}")

        return report

    def _generate_recommendations(self, failed_tests: List[TestResult], by_category: Dict) -> List[str]:
        """Generate actionable recommendations based on failures"""
        recommendations = []

        # Check for category-wide failures
        for cat, stats in by_category.items():
            if stats["pass_rate"] == 0:
                recommendations.append(f"CRITICAL: {cat} is completely down - immediate investigation required")
            elif stats["pass_rate"] < 50:
                recommendations.append(f"WARNING: {cat} has major issues ({stats['pass_rate']:.0f}% pass rate)")

        # Check for specific failure patterns
        timeout_count = sum(1 for t in failed_tests if t.status == VerificationStatus.TIMEOUT)
        if timeout_count > 3:
            recommendations.append(f"PERFORMANCE: {timeout_count} endpoints timing out - check server resources")

        unreachable_count = sum(1 for t in failed_tests if t.status == VerificationStatus.UNREACHABLE)
        if unreachable_count > 0:
            recommendations.append(f"NETWORK: {unreachable_count} endpoints unreachable - check DNS and connectivity")

        invalid_count = sum(1 for t in failed_tests if t.status == VerificationStatus.INVALID_RESPONSE)
        if invalid_count > 0:
            recommendations.append(f"API: {invalid_count} endpoints returning invalid responses - check API contracts")

        if not recommendations and failed_tests:
            recommendations.append("Review individual test failures for specific issues")

        if not failed_tests:
            recommendations.append("All systems operational - no action required")

        return recommendations

    async def run_quick_health_check(self) -> Dict[str, Any]:
        """Quick health check of critical endpoints only"""
        critical_tests = [t for t in self.tests if t.critical]

        async with aiohttp.ClientSession() as session:
            tasks = [self._run_single_test(test, session) for test in critical_tests]
            results = await asyncio.gather(*tasks)

        passed = sum(1 for r in results if r.status == VerificationStatus.PASSED)
        failed = [r for r in results if r.status != VerificationStatus.PASSED]

        return {
            "is_healthy": len(failed) == 0,
            "critical_tests": len(critical_tests),
            "passed": passed,
            "failed": len(failed),
            "failed_endpoints": [{"name": f.test_name, "error": f.error_message} for f in failed],
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_report_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of last verification report"""
        if not self.last_report:
            return None

        return {
            "report_id": self.last_report.report_id,
            "completed_at": self.last_report.completed_at,
            "is_100_percent_operational": self.last_report.is_100_percent_operational,
            "pass_rate": self.last_report.pass_rate,
            "total_tests": self.last_report.total_tests,
            "passed": self.last_report.passed,
            "failed": self.last_report.failed,
            "degraded": self.last_report.degraded,
            "results_by_category": self.last_report.results_by_category,
            "recommendations": self.last_report.recommendations
        }


# Singleton instance
e2e_verification = E2ESystemVerification()


# ============================================
# API FUNCTIONS
# ============================================

async def run_full_e2e_verification() -> Dict[str, Any]:
    """Run complete E2E verification and return report"""
    report = await e2e_verification.run_full_verification()

    # Convert to dict, handling dataclass conversion
    failed_tests_dicts = []
    for ft in report.failed_tests:
        ft_dict = asdict(ft)
        # Remove large response bodies for summary
        if ft_dict.get("response_body") and len(str(ft_dict["response_body"])) > 500:
            ft_dict["response_body"] = {"truncated": True, "message": "Response too large"}
        failed_tests_dicts.append(ft_dict)

    return {
        "report_id": report.report_id,
        "started_at": report.started_at,
        "completed_at": report.completed_at,
        "duration_seconds": report.duration_seconds,
        "is_100_percent_operational": report.is_100_percent_operational,
        "overall_status": report.overall_status.value,
        "pass_rate": report.pass_rate,
        "summary": {
            "total_tests": report.total_tests,
            "passed": report.passed,
            "failed": report.failed,
            "degraded": report.degraded
        },
        "results_by_category": report.results_by_category,
        "failed_tests": failed_tests_dicts,
        "recommendations": report.recommendations
    }


async def run_quick_health_check() -> Dict[str, Any]:
    """Run quick health check of critical systems"""
    return await e2e_verification.run_quick_health_check()


async def get_last_verification_report() -> Optional[Dict[str, Any]]:
    """Get the last verification report summary"""
    return e2e_verification.get_report_summary()

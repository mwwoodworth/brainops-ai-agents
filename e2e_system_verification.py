"""
E2E System Verification Engine
==============================
Comprehensive end-to-end verification of ALL BrainOps systems.
Iterates through every endpoint, validates responses, and ensures 100% operation.

This is the SINGLE SOURCE OF TRUTH for system health.
No partial operations allowed - everything must be 100% or flagged for immediate action.
"""

import asyncio
import hashlib
import hmac as hmac_mod
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Production configuration
BRAINOPS_API_URL = os.getenv("BRAINOPS_API_URL", "https://brainops-ai-agents.onrender.com")
# Localhost URL for self-calls: avoids external round-trip and rate-limit contention.
_SELF_CALL_PORT = os.getenv("PORT", "10000")
_SELF_CALL_BASE = f"http://localhost:{_SELF_CALL_PORT}"
BRAINOPS_BACKEND_URL = os.getenv(
    "BRAINOPS_BACKEND_URL", "https://brainops-backend-prod.onrender.com"
)
ERP_URL = os.getenv("ERP_URL", "https://weathercraft-erp.vercel.app")
MRG_URL = os.getenv("MRG_URL", "https://myroofgenius.com")
MCP_BRIDGE_URL = os.getenv("MCP_BRIDGE_URL", "https://brainops-mcp-bridge.onrender.com")
MCP_API_KEY = (os.getenv("MCP_API_KEY") or "").strip()
# Guest checkout smoke tests create a Stripe session but do not charge.
E2E_TEST_EMAIL = (os.getenv("E2E_TEST_EMAIL") or "ops-e2e-checkout@example.com").strip()
# Use the same API key source as the server validates against
_api_keys_str = os.getenv("API_KEYS", "")
_api_keys_list = [k.strip() for k in _api_keys_str.split(",") if k.strip()] if _api_keys_str else []
API_KEY = _api_keys_list[0] if _api_keys_list else os.getenv("BRAINOPS_API_KEY", "")
logger.info(
    "E2E verification API key configured: %s (source=%s)",
    bool(API_KEY),
    "API_KEYS" if _api_keys_list else "BRAINOPS_API_KEY",
)


def _compute_e2e_internal_sig(api_key: str) -> str:
    """Compute HMAC-SHA256 signature for internal E2E rate-limit exemption.

    The server's _rate_limit_key() validates this to give internal E2E
    requests a unique rate-limit bucket, preventing the verifier from
    429-ing itself when probing many endpoints in rapid succession.
    """
    return hmac_mod.new(
        api_key.encode("utf-8"), b"brainops-e2e-internal", hashlib.sha256
    ).hexdigest()


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
    headers: dict[str, str] = field(default_factory=dict)
    body: Optional[dict[str, Any]] = None
    expected_status: int = 200
    acceptable_statuses: list[int] = field(default_factory=list)
    expected_fields: list[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    max_response_time_ms: Optional[float] = None
    category: SystemCategory = SystemCategory.CORE_API
    critical: bool = True  # If True, failure means system is NOT 100% operational
    validation_func: Optional[str] = None  # Name of custom validation function
    required_env_vars: list[str] = field(
        default_factory=list
    )  # If any are missing/empty, fail fast with a clear error


@dataclass
class TestResult:
    """Result of a single endpoint test"""

    test_name: str
    endpoint: str
    status: VerificationStatus
    response_time_ms: float
    status_code: Optional[int] = None
    response_body: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    missing_fields: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
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
    results_by_category: dict[str, dict[str, Any]]
    failed_tests: list[TestResult]
    all_results: list[TestResult]
    recommendations: list[str]
    is_100_percent_operational: bool
    skip_erp: bool = False


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
        self.tests: list[EndpointTest] = []
        self.results: list[TestResult] = []
        self.last_report: Optional[SystemVerificationReport] = None
        self._initialize_tests()

    @staticmethod
    def _apply_api_key_override(
        tests: list[EndpointTest], api_key_override: Optional[str]
    ) -> list[EndpointTest]:
        """Return a per-run test list with X-API-Key overridden when provided.

        This avoids coupling outbound cross-system checks (e.g., Command Center unified health)
        to a particular env var ordering (API_KEYS vs BRAINOPS_API_KEY) and prevents global
        mutation of the singleton test registry.
        """
        if not api_key_override:
            return tests

        updated: list[EndpointTest] = []
        for test in tests:
            merged_headers = dict(test.headers or {})
            # Only override tests that already use the default BrainOps API key.
            # Do not override MCP Bridge calls (MCP_API_KEY) or negative auth tests
            # that intentionally omit headers.
            existing = merged_headers.get("X-API-Key")
            if existing is not None and (not existing or existing == API_KEY):
                merged_headers["X-API-Key"] = api_key_override
                merged_headers.pop("x-api-key", None)
                updated.append(replace(test, headers=merged_headers))
            else:
                updated.append(test)
        return updated

    @staticmethod
    def _apply_scope(tests: list[EndpointTest], skip_erp: bool) -> list[EndpointTest]:
        """Filter/adjust tests based on verification scope."""
        if not skip_erp:
            return tests

        updated: list[EndpointTest] = []
        for test in tests:
            is_erp = (
                test.name.startswith("ERP -")
                or test.url == ERP_URL
                or test.url.startswith(ERP_URL + "/")
                or "weathercraft-erp" in test.url
            )
            if is_erp:
                continue

            if test.name.startswith("ChatGPT Agent UI -"):
                # Run the UI smoke in non-ERP mode so the payload does not require ERP.
                url = test.url
                if "skip_erp=" not in url:
                    url = f"{url}{'&' if '?' in url else '?'}skip_erp=true"
                updated.append(
                    replace(
                        test,
                        url=url,
                        expected_fields=["mrg_healthy", "erp_skipped"],
                        validation_func="validate_chatgpt_agent_quick_non_erp",
                    )
                )
                continue

            updated.append(test)

        return updated

    def _initialize_tests(self):
        """Initialize all endpoint tests"""
        internal_health_key = (os.getenv("INTERNAL_HEALTH_KEY") or "").strip()
        headers = {"X-API-Key": API_KEY}
        if internal_health_key:
            headers["x-health-key"] = internal_health_key
        mcp_headers = {"X-API-Key": MCP_API_KEY}
        erp_health_headers: dict[str, str] = {}
        health_check_secret = (
            os.getenv("HEALTH_CHECK_SECRET")
            or os.getenv("WEATHERCRAFT_HEALTH_CHECK_SECRET")
            or ""
        ).strip()
        if health_check_secret:
            erp_health_headers["x-health-check-secret"] = health_check_secret
        elif internal_health_key:
            erp_health_headers["x-health-key"] = internal_health_key

        # ============================================
        # CORE API ENDPOINTS
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="Health Check",
                    url=f"{BRAINOPS_API_URL}/health",
                    headers=headers,
                    expected_fields=["status", "version", "database"],
                    category=SystemCategory.CORE_API,
                    critical=True,
                ),
                EndpointTest(
                    name="Root Endpoint",
                    url=f"{BRAINOPS_API_URL}/",
                    headers=headers,
                    expected_fields=["service", "version", "status"],
                    category=SystemCategory.CORE_API,
                    critical=True,
                ),
                EndpointTest(
                    name="Observability Metrics",
                    url=f"{BRAINOPS_API_URL}/observability/metrics",
                    headers=headers,
                    expected_fields=["requests", "cache", "database"],
                    category=SystemCategory.CORE_API,
                    critical=True,
                ),
                EndpointTest(
                    name="Agents List",
                    url=f"{BRAINOPS_API_URL}/agents",
                    headers=headers,
                    expected_fields=["agents"],
                    category=SystemCategory.CORE_API,
                    critical=True,
                ),
                EndpointTest(
                    name="Observability Dashboard",
                    url=f"{BRAINOPS_API_URL}/api/v1/observability/dashboard",
                    headers=headers,
                    expected_fields=["overall_status", "agents", "database", "mcp", "memory"],
                    category=SystemCategory.CORE_API,
                    critical=True,
                ),
                EndpointTest(
                    name="AI Providers Status",
                    url=f"{BRAINOPS_API_URL}/ai/providers/status",
                    headers=headers,
                    expected_fields=[],  # Structure varies
                    category=SystemCategory.CORE_API,
                    critical=False,
                ),
                EndpointTest(
                    name="Systems Usage",
                    url=f"{BRAINOPS_API_URL}/systems/usage",
                    headers=headers,
                    expected_fields=["active_systems"],
                    category=SystemCategory.CORE_API,
                    critical=True,
                ),
                EndpointTest(
                    name="Brain Critical Keys",
                    url=f"{BRAINOPS_API_URL}/brain/critical",
                    headers=headers,
                    expected_fields=["status", "critical_items"],
                    category=SystemCategory.CORE_API,
                    critical=True,
                ),
                EndpointTest(
                    name="Brain Memory",
                    url=f"{BRAINOPS_API_URL}/brain/status",
                    headers=headers,
                    expected_fields=["status", "timestamp"],
                    category=SystemCategory.CORE_API,
                    critical=False,
                ),
            ]
        )

        # ============================================
        # BLEEDING EDGE - DIGITAL TWIN
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="Digital Twin - Status",
                    url=f"{BRAINOPS_API_URL}/digital-twin/status",
                    headers=headers,
                    expected_fields=["system", "status", "capabilities"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
                EndpointTest(
                    name="Digital Twin - Dashboard",
                    url=f"{BRAINOPS_API_URL}/digital-twin/dashboard",
                    headers=headers,
                    expected_fields=["summary"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
                EndpointTest(
                    name="Digital Twin - List Twins",
                    url=f"{BRAINOPS_API_URL}/digital-twin/twins",
                    headers=headers,
                    expected_fields=["twins", "total"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
            ]
        )

        # ============================================
        # BLEEDING EDGE - MARKET INTELLIGENCE
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="Market Intelligence - Status",
                    url=f"{BRAINOPS_API_URL}/market-intelligence/status",
                    headers=headers,
                    expected_fields=["system", "status", "capabilities"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                    timeout_seconds=35.0,
                    max_response_time_ms=32000.0,
                ),
                EndpointTest(
                    name="Market Intelligence - Dashboard",
                    url=f"{BRAINOPS_API_URL}/market-intelligence/dashboard",
                    headers=headers,
                    expected_fields=["overview"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                    timeout_seconds=35.0,
                    max_response_time_ms=32000.0,
                ),
                EndpointTest(
                    name="Market Intelligence - Trends",
                    url=f"{BRAINOPS_API_URL}/market-intelligence/trends",
                    headers=headers,
                    expected_fields=["trends"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                    timeout_seconds=35.0,
                    max_response_time_ms=32000.0,
                ),
                EndpointTest(
                    name="Market Intelligence - Signals",
                    url=f"{BRAINOPS_API_URL}/market-intelligence/signals",
                    headers=headers,
                    expected_fields=["signals"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                    timeout_seconds=35.0,
                    max_response_time_ms=32000.0,
                ),
            ]
        )

        # ============================================
        # BLEEDING EDGE - ROOFING LABOR ML
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="Roofing Labor ML - Status",
                    url=f"{BRAINOPS_API_URL}/roofing/labor-ml/status",
                    headers=headers,
                    expected_fields=["system", "status", "samples"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
            ]
        )

        # ============================================
        # BLEEDING EDGE - SYSTEM ORCHESTRATOR
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="Orchestrator - Status",
                    url=f"{BRAINOPS_API_URL}/orchestrator/status",
                    headers=headers,
                    expected_fields=["system", "status", "capabilities", "scale_capacity"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
                EndpointTest(
                    name="Orchestrator - Dashboard",
                    url=f"{BRAINOPS_API_URL}/orchestrator/dashboard",
                    headers=headers,
                    expected_fields=["overview", "scale_capacity"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
                EndpointTest(
                    name="Orchestrator - Systems List",
                    url=f"{BRAINOPS_API_URL}/orchestrator/systems",
                    headers=headers,
                    expected_fields=["systems", "total"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
                EndpointTest(
                    name="Orchestrator - Groups",
                    url=f"{BRAINOPS_API_URL}/orchestrator/groups",
                    headers=headers,
                    expected_fields=["groups"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
            ]
        )

        # ============================================
        # BLEEDING EDGE - SELF-HEALING
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="Self-Healing - Status",
                    url=f"{BRAINOPS_API_URL}/api/v1/self-healing/status",
                    headers=headers,
                    expected_fields=["system", "status", "capabilities", "recovery_improvement"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
                EndpointTest(
                    name="Self-Healing - Dashboard",
                    url=f"{BRAINOPS_API_URL}/api/v1/self-healing/dashboard",
                    headers=headers,
                    expected_fields=["overview", "autonomy_tiers"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
                EndpointTest(
                    name="Self-Healing - Active Incidents",
                    url=f"{BRAINOPS_API_URL}/api/v1/self-healing/incidents/active",
                    headers=headers,
                    expected_fields=["active_incidents", "total"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
                EndpointTest(
                    name="Self-Healing - Metrics",
                    url=f"{BRAINOPS_API_URL}/api/v1/self-healing/metrics",
                    headers=headers,
                    expected_fields=["metrics", "performance"],
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                ),
            ]
        )

        # ============================================
        # FRONTEND APPLICATIONS
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="ERP - Homepage",
                    url=ERP_URL,
                    expected_status=200,
                    expected_fields=[],
                    category=SystemCategory.FRONTEND,
                    critical=True,
                    timeout_seconds=15.0,
                ),
                EndpointTest(
                    name="ERP - Login Page",
                    url=f"{ERP_URL}/login",
                    expected_status=200,
                    expected_fields=[],
                    category=SystemCategory.FRONTEND,
                    critical=True,
                    timeout_seconds=15.0,
                ),
                EndpointTest(
                    name="ERP - API Health",
                    url=f"{ERP_URL}/api/health",
                    headers=erp_health_headers or None,
                    expected_status=200,
                    acceptable_statuses=[401],
                    expected_fields=[],
                    validation_func="validate_erp_api_health",
                    category=SystemCategory.FRONTEND,
                    critical=True,
                    timeout_seconds=15.0,
                ),
                EndpointTest(
                    name="MyRoofGenius - Homepage",
                    url=MRG_URL,
                    expected_status=200,
                    expected_fields=[],
                    category=SystemCategory.FRONTEND,
                    critical=True,
                    timeout_seconds=15.0,
                ),
                EndpointTest(
                    name="MyRoofGenius - API Health",
                    url=f"{MRG_URL}/api/health",
                    expected_status=200,
                    expected_fields=["status", "services"],
                    category=SystemCategory.FRONTEND,
                    critical=True,
                    timeout_seconds=20.0,
                ),
                EndpointTest(
                    name="Command Center - Unified Health",
                    url="https://brainops-command-center.vercel.app/api/unified-health",
                    headers=headers,
                    expected_status=200,
                    expected_fields=["overall", "services"],
                    category=SystemCategory.FRONTEND,
                    critical=True,
                    timeout_seconds=20.0,
                ),
                EndpointTest(
                    name="Command Center - Unified Health (No Auth)",
                    url="https://brainops-command-center.vercel.app/api/unified-health",
                    expected_status=401,
                    expected_fields=[],
                    category=SystemCategory.FRONTEND,
                    critical=True,
                    timeout_seconds=20.0,
                ),
                EndpointTest(
                    name="Brainstack Studio - Homepage",
                    url="https://brainstack-studio.vercel.app/",
                    expected_status=200,
                    expected_fields=[],
                    category=SystemCategory.FRONTEND,
                    critical=False,
                    timeout_seconds=15.0,
                ),
            ]
        )

        # ============================================
        # BACKEND API
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="Backend - Health",
                    url=f"{BRAINOPS_BACKEND_URL}/health",
                    expected_status=200,
                    expected_fields=[],
                    category=SystemCategory.CORE_API,
                    critical=True,
                ),
                EndpointTest(
                    name="Backend - API v1 Health",
                    url=f"{BRAINOPS_BACKEND_URL}/api/v1/health",
                    headers=headers,
                    expected_status=200,
                    expected_fields=["status", "version", "database", "cns"],
                    category=SystemCategory.CORE_API,
                    critical=True,
                    timeout_seconds=25.0,
                ),
                EndpointTest(
                    name="Backend - CNS Status",
                    url=f"{BRAINOPS_BACKEND_URL}/api/v1/cns/status",
                    headers=headers,
                    expected_status=200,
                    expected_fields=["status", "initialized", "memory_count", "task_count"],
                    validation_func="validate_backend_cns_status",
                    category=SystemCategory.CORE_API,
                    critical=True,
                    timeout_seconds=25.0,
                ),
                EndpointTest(
                    name="Backend - CNS Status (No Auth)",
                    url=f"{BRAINOPS_BACKEND_URL}/api/v1/cns/status",
                    expected_status=401,
                    expected_fields=[],
                    category=SystemCategory.CORE_API,
                    critical=True,
                    timeout_seconds=25.0,
                ),
                EndpointTest(
                    name="Backend - Revenue Dashboard",
                    url=f"{BRAINOPS_BACKEND_URL}/api/v1/revenue/dashboard",
                    headers=headers,
                    expected_status=200,
                    expected_fields=["mrr", "arr", "total_revenue"],
                    category=SystemCategory.CORE_API,
                    critical=False,
                    timeout_seconds=25.0,
                ),
            ]
        )

        # ============================================
        # MCP BRIDGE
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="MCP Bridge - Health",
                    url=f"{MCP_BRIDGE_URL}/health",
                    expected_status=200,
                    expected_fields=["status", "mcpServers", "totalTools"],
                    category=SystemCategory.MCP,
                    critical=True,
                ),
                EndpointTest(
                    name="MCP Bridge - Servers (Auth)",
                    url=f"{MCP_BRIDGE_URL}/mcp/servers",
                    headers=mcp_headers,
                    expected_status=200,
                    expected_fields=["servers", "statistics"],
                    validation_func="validate_mcp_servers",
                    category=SystemCategory.MCP,
                    critical=True,
                    required_env_vars=["MCP_API_KEY"],
                    timeout_seconds=25.0,
                ),
                EndpointTest(
                    name="MCP Bridge - Tools Inventory (Auth)",
                    url=f"{MCP_BRIDGE_URL}/mcp/tools",
                    headers=mcp_headers,
                    expected_status=200,
                    expected_fields=["totalServers", "totalTools", "servers"],
                    validation_func="validate_mcp_tools_inventory",
                    category=SystemCategory.MCP,
                    critical=True,
                    required_env_vars=["MCP_API_KEY"],
                    timeout_seconds=25.0,
                ),
                EndpointTest(
                    name="MCP Bridge - Servers (No Auth)",
                    url=f"{MCP_BRIDGE_URL}/mcp/servers",
                    expected_status=401,
                    expected_fields=[],
                    category=SystemCategory.MCP,
                    critical=True,
                    timeout_seconds=25.0,
                ),
            ]
        )

        # ============================================
        # REVENUE / STRIPE (MRG)
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="MRG - Stripe Prices",
                    url=f"{MRG_URL}/api/stripe/prices",
                    expected_status=200,
                    expected_fields=["prices"],
                    category=SystemCategory.EXTERNAL,
                    critical=False,
                    timeout_seconds=25.0,
                ),
                EndpointTest(
                    name="MRG - Stripe Checkout Session (Guest)",
                    url=f"{MRG_URL}/api/stripe/checkout-session-v2",
                    method="POST",
                    expected_status=200,
                    expected_fields=["sessionId", "url"],
                    body={
                        "productKey": "ai-roof-inspector-basic",
                        "customerEmail": E2E_TEST_EMAIL,
                    },
                    category=SystemCategory.EXTERNAL,
                    critical=False,
                    timeout_seconds=35.0,
                ),
                EndpointTest(
                    name="MRG - Stripe Checkout Session (No Email)",
                    url=f"{MRG_URL}/api/stripe/checkout-session-v2",
                    method="POST",
                    expected_status=401,
                    expected_fields=["error"],
                    body={
                        "productKey": "ai-roof-inspector-basic",
                    },
                    category=SystemCategory.EXTERNAL,
                    critical=False,
                    timeout_seconds=35.0,
                ),
            ]
        )

        # ============================================
        # BLEEDING EDGE - UI AGENT TESTS (REAL BROWSER)
        # ============================================
        self.tests.extend(
            [
                EndpointTest(
                    name="ChatGPT Agent UI - Quick",
                    url=f"{BRAINOPS_API_URL}/api/v1/always-know/chatgpt-agent-test",
                    method="POST",
                    headers=headers,
                    expected_status=200,
                    expected_fields=["mrg_healthy", "erp_healthy"],
                    validation_func="validate_chatgpt_agent_quick",
                    category=SystemCategory.BLEEDING_EDGE,
                    critical=True,
                    timeout_seconds=180.0,
                    max_response_time_ms=60000.0,
                ),
            ]
        )

    def _run_validation(self, test: EndpointTest, response_body: Any) -> list[str]:
        if not test.validation_func:
            return []

        validator = getattr(self, test.validation_func, None)
        if not callable(validator):
            return [f"Unknown validation_func: {test.validation_func}"]

        try:
            errors = validator(test, response_body)
            if not errors:
                return []
            if isinstance(errors, list):
                return [str(e) for e in errors]
            return [str(errors)]
        except Exception as exc:
            return [f"Validation '{test.validation_func}' raised: {exc}"]

    @staticmethod
    def validate_chatgpt_agent_quick(_test: EndpointTest, response_body: Any) -> list[str]:
        """Validate the ChatGPT-Agent quick UI test result payload."""
        if not isinstance(response_body, dict):
            return ["Response body is not a JSON object"]

        errors: list[str] = []
        if response_body.get("mrg_healthy") is not True:
            errors.append("MyRoofGenius UI quick test failed")
        if response_body.get("erp_healthy") is not True:
            errors.append("Weathercraft ERP UI quick test failed")

        return errors

    @staticmethod
    def validate_chatgpt_agent_quick_non_erp(_test: EndpointTest, response_body: Any) -> list[str]:
        """Validate the ChatGPT-Agent quick UI test result payload in non-ERP scope."""
        if not isinstance(response_body, dict):
            return ["Response body is not a JSON object"]

        errors: list[str] = []
        if response_body.get("mrg_healthy") is not True:
            errors.append("MyRoofGenius UI quick test failed")

        # If ERP is skipped, do not fail the system for ERP UI state.
        if response_body.get("erp_skipped") is not True:
            errors.append("Expected erp_skipped=true for non-ERP verification scope")

        return errors

    @staticmethod
    def validate_backend_api_v1_health(_test: EndpointTest, response_body: Any) -> list[str]:
        if not isinstance(response_body, dict):
            return ["Response body is not a JSON object"]

        errors: list[str] = []
        cns = response_body.get("cns")
        if cns != "operational":
            errors.append(f"Backend /api/v1/health cns != operational (got {cns!r})")
        return errors

    @staticmethod
    def validate_backend_cns_status(_test: EndpointTest, response_body: Any) -> list[str]:
        if not isinstance(response_body, dict):
            return ["Response body is not a JSON object"]

        errors: list[str] = []
        if response_body.get("status") != "operational":
            errors.append(
                f"Backend CNS status != operational (got {response_body.get('status')!r})"
            )
        if response_body.get("initialized") is not True:
            errors.append("Backend CNS initialized != true")
        return errors

    @staticmethod
    def validate_erp_api_health(_test: EndpointTest, response_body: Any) -> list[str]:
        if not isinstance(response_body, dict):
            return ["Response body is not a JSON object"]

        status = str(response_body.get("status", "")).lower()
        errors: list[str] = []

        if status in {"ok", "healthy"}:
            services = response_body.get("services")
            if not isinstance(services, dict):
                errors.append("Expected services object for healthy ERP /api/health response")
            return errors

        if status == "unauthorized":
            if not response_body.get("error"):
                errors.append("Unauthorized ERP /api/health response missing error field")
            return errors

        errors.append(f"Unexpected ERP /api/health status payload: {response_body.get('status')!r}")
        return errors

    @staticmethod
    def validate_mcp_tools_inventory(_test: EndpointTest, response_body: Any) -> list[str]:
        if not isinstance(response_body, dict):
            return ["Response body is not a JSON object"]

        errors: list[str] = []
        total_servers = response_body.get("totalServers")
        total_tools = response_body.get("totalTools")
        servers = response_body.get("servers")

        if not isinstance(total_servers, int) or total_servers <= 0:
            errors.append(f"Invalid totalServers: {total_servers!r}")
        if not isinstance(total_tools, int) or total_tools <= 0:
            errors.append(f"Invalid totalTools: {total_tools!r}")
        if not isinstance(servers, dict) or len(servers.keys()) <= 0:
            errors.append("Invalid servers map (empty or not an object)")
        elif (
            isinstance(total_servers, int)
            and total_servers > 0
            and len(servers.keys()) != total_servers
        ):
            errors.append(
                f"servers map length != totalServers ({len(servers.keys())} != {total_servers})"
            )

        return errors

    @staticmethod
    def validate_mcp_servers(_test: EndpointTest, response_body: Any) -> list[str]:
        if not isinstance(response_body, dict):
            return ["Response body is not a JSON object"]

        errors: list[str] = []
        servers = response_body.get("servers")
        stats = response_body.get("statistics")

        if not isinstance(servers, dict) or len(servers.keys()) <= 0:
            errors.append("Missing/empty 'servers' map")
        if not isinstance(stats, dict):
            errors.append("Missing/invalid 'statistics'")
            return errors

        total_servers = stats.get("total_servers")
        total_tools = stats.get("total_tools")
        if not isinstance(total_servers, int) or total_servers <= 0:
            errors.append(f"Invalid statistics.total_servers: {total_servers!r}")
        if not isinstance(total_tools, int) or total_tools <= 0:
            errors.append(f"Invalid statistics.total_tools: {total_tools!r}")
        if isinstance(servers, dict) and isinstance(total_servers, int) and total_servers > 0:
            if len(servers.keys()) != total_servers:
                errors.append(
                    f"servers map length != statistics.total_servers ({len(servers.keys())} != {total_servers})"
                )

        return errors

    async def _run_single_test(
        self, test: EndpointTest, session: aiohttp.ClientSession
    ) -> TestResult:
        """Execute a single endpoint test"""
        start_time = time.perf_counter()

        # Fail-fast for missing required env vars so reports are actionable.
        if test.required_env_vars:
            missing = [k for k in test.required_env_vars if not (os.getenv(k) or "").strip()]
            if missing:
                return TestResult(
                    test_name=test.name,
                    endpoint=test.url,
                    status=VerificationStatus.FAILED,
                    response_time_ms=0.0,
                    error_message=f"Missing required env var(s): {', '.join(missing)}",
                )

        # Self-call optimisation: when probing *this* service, rewrite the URL
        # to localhost so the request stays in-process.  This avoids external
        # round-trips and any edge-level rate limiting.  We also inject the
        # X-Internal-E2E HMAC header so the application-level rate limiter
        # gives each probe its own bucket (defence in depth).
        #
        # IMPORTANT: The HMAC must be computed with the ACTUAL X-API-Key that
        # will appear in the request headers — _apply_api_key_override may
        # have replaced the key from the original test definition with the
        # caller's key, so we must sign with whatever key the server will see.
        headers = dict(test.headers) if test.headers else {}
        url = test.url
        is_self_call = bool(API_KEY) and url.startswith(BRAINOPS_API_URL)
        if is_self_call:
            url = _SELF_CALL_BASE + url[len(BRAINOPS_API_URL) :]
            signing_key = headers.get("X-API-Key") or API_KEY
            headers["X-Internal-E2E"] = _compute_e2e_internal_sig(signing_key)

        try:
            async with session.request(
                method=test.method,
                url=url,
                headers=headers,
                json=test.body if test.body else None,
                timeout=aiohttp.ClientTimeout(total=test.timeout_seconds),
            ) as response:
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Try to parse JSON response
                response_body = None
                try:
                    if response.content_type and "json" in response.content_type:
                        response_body = await response.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "Failed to parse JSON response from %s: %s",
                        test.url,
                        exc,
                        exc_info=True,
                    )

                # Check status code
                allowed_statuses = {test.expected_status, *test.acceptable_statuses}
                if response.status not in allowed_statuses:
                    return TestResult(
                        test_name=test.name,
                        endpoint=test.url,
                        status=VerificationStatus.FAILED,
                        response_time_ms=response_time_ms,
                        status_code=response.status,
                        response_body=response_body,
                        error_message=(
                            f"Expected status {test.expected_status}"
                            + (
                                f" (acceptable: {sorted(test.acceptable_statuses)})"
                                if test.acceptable_statuses
                                else ""
                            )
                            + f", got {response.status}"
                        ),
                    )

                # Check expected fields (treat empty objects as valid payloads, and fail if
                # an endpoint that claims JSON fields does not return a JSON object).
                missing_fields: list[str] = []
                if test.expected_fields:
                    if not isinstance(response_body, dict):
                        return TestResult(
                            test_name=test.name,
                            endpoint=test.url,
                            status=VerificationStatus.INVALID_RESPONSE,
                            response_time_ms=response_time_ms,
                            status_code=response.status,
                            response_body=response_body
                            if isinstance(response_body, dict)
                            else None,
                            error_message="Expected JSON object response",
                        )

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
                        error_message=f"Missing required fields: {missing_fields}",
                    )

                # Check for app-level error payloads, but only on tests expecting success.
                # Negative auth tests intentionally return {error: "..."} with 401/403.
                if isinstance(response_body, dict) and 200 <= test.expected_status < 300:
                    if response_body.get("status") == "error":
                        return TestResult(
                            test_name=test.name,
                            endpoint=test.url,
                            status=VerificationStatus.FAILED,
                            response_time_ms=response_time_ms,
                            status_code=response.status,
                            response_body=response_body,
                            error_message=response_body.get("message")
                            or response_body.get("error")
                            or "Error in response",
                        )
                    err_val = response_body.get("error")
                    if err_val:
                        return TestResult(
                            test_name=test.name,
                            endpoint=test.url,
                            status=VerificationStatus.FAILED,
                            response_time_ms=response_time_ms,
                            status_code=response.status,
                            response_body=response_body,
                            error_message=str(err_val),
                        )

                max_response_time_ms: Optional[float] = test.max_response_time_ms
                if max_response_time_ms is None:
                    env_max_ms = os.getenv("E2E_MAX_RESPONSE_TIME_MS", "").strip()
                    if env_max_ms:
                        try:
                            max_response_time_ms = float(env_max_ms)
                        except ValueError:
                            max_response_time_ms = None

                # Check for degraded performance (Render free tier can be slow; keep defaults conservative)
                if response_time_ms > (
                    max_response_time_ms if max_response_time_ms is not None else 20000.0
                ):
                    return TestResult(
                        test_name=test.name,
                        endpoint=test.url,
                        status=VerificationStatus.DEGRADED,
                        response_time_ms=response_time_ms,
                        status_code=response.status,
                        response_body=response_body,
                        error_message=f"Slow response: {response_time_ms:.0f}ms",
                    )

                validation_errors = []
                if response_body is not None and test.validation_func:
                    validation_errors = self._run_validation(test, response_body)

                if validation_errors:
                    return TestResult(
                        test_name=test.name,
                        endpoint=test.url,
                        status=VerificationStatus.FAILED,
                        response_time_ms=response_time_ms,
                        status_code=response.status,
                        response_body=response_body if isinstance(response_body, dict) else None,
                        validation_errors=validation_errors,
                        error_message="; ".join(validation_errors),
                    )

                # All checks passed
                return TestResult(
                    test_name=test.name,
                    endpoint=test.url,
                    status=VerificationStatus.PASSED,
                    response_time_ms=response_time_ms,
                    status_code=response.status,
                    response_body=response_body,
                )

        except asyncio.TimeoutError:
            return TestResult(
                test_name=test.name,
                endpoint=test.url,
                status=VerificationStatus.TIMEOUT,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
                error_message=f"Timeout after {test.timeout_seconds}s",
            )
        except aiohttp.ClientError as e:
            return TestResult(
                test_name=test.name,
                endpoint=test.url,
                status=VerificationStatus.UNREACHABLE,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e),
            )
        except Exception as e:
            return TestResult(
                test_name=test.name,
                endpoint=test.url,
                status=VerificationStatus.FAILED,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
                error_message=f"Unexpected error: {str(e)}",
            )

    async def run_full_verification(
        self,
        api_key_override: Optional[str] = None,
        skip_erp: bool = False,
    ) -> SystemVerificationReport:
        """Run complete E2E verification of all systems"""
        started_at = datetime.utcnow()
        report_id = hashlib.md5(started_at.isoformat().encode()).hexdigest()[:12]

        tests = self._apply_scope(
            self._apply_api_key_override(self.tests, api_key_override), skip_erp=skip_erp
        )

        logger.info(f"Starting E2E verification (ID: {report_id}) - {len(tests)} tests")

        # Run tests with bounded concurrency to avoid self-DOS'ing the service under test.
        # Additionally, run heavyweight browser-based UI tests serially after the main batch.
        # Default to 4 to align with the production DB pool max_size (Supabase session mode).
        # Higher values can self-DOS the service under test during `/e2e/verify`.
        max_concurrency = max(1, int(os.getenv("E2E_MAX_CONCURRENCY", "4")))
        semaphore = asyncio.Semaphore(max_concurrency)

        def _should_run_serial(test: EndpointTest) -> bool:
            return (
                "/api/v1/always-know/chatgpt-agent-test" in test.url
                or test.name.startswith("ChatGPT Agent UI")
                or "/market-intelligence/" in test.url
            )

        serial_indices: list[int] = []
        parallel_indices: list[int] = []
        for idx, test in enumerate(tests):
            (serial_indices if _should_run_serial(test) else parallel_indices).append(idx)

        logger.info(
            "E2E execution plan: %s parallel (max_concurrency=%s), %s serial",
            len(parallel_indices),
            max_concurrency,
            len(serial_indices),
        )

        async with aiohttp.ClientSession() as session:
            results_by_index: dict[int, TestResult] = {}

            async def _run_index(i: int) -> tuple[int, TestResult]:
                async with semaphore:
                    return i, await self._run_single_test(tests[i], session)

            if parallel_indices:
                pairs = await asyncio.gather(*[_run_index(i) for i in parallel_indices])
                results_by_index.update(dict(pairs))

            for i in serial_indices:
                results_by_index[i] = await self._run_single_test(tests[i], session)

            self.results = [results_by_index[i] for i in range(len(tests))]

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        # Analyze results
        passed = sum(1 for r in self.results if r.status == VerificationStatus.PASSED)
        failed = sum(
            1
            for r in self.results
            if r.status
            in [
                VerificationStatus.FAILED,
                VerificationStatus.TIMEOUT,
                VerificationStatus.UNREACHABLE,
                VerificationStatus.INVALID_RESPONSE,
            ]
        )
        degraded = sum(1 for r in self.results if r.status == VerificationStatus.DEGRADED)

        # Check critical tests
        critical_results = [r for r, t in zip(self.results, tests) if t.critical]
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
            cat_tests = [(r, t) for r, t in zip(self.results, tests) if t.category == category]
            if cat_tests:
                cat_passed = sum(1 for r, t in cat_tests if r.status == VerificationStatus.PASSED)
                cat_failed = sum(1 for r, t in cat_tests if r.status != VerificationStatus.PASSED)
                results_by_category[category.value] = {
                    "total": len(cat_tests),
                    "passed": cat_passed,
                    "failed": cat_failed,
                    "pass_rate": cat_passed / len(cat_tests) * 100 if cat_tests else 0,
                    "status": "passed" if cat_failed == 0 else "failed",
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
            is_100_percent_operational=is_100_percent,
            skip_erp=skip_erp,
        )

        self.last_report = report

        # Log summary
        if is_100_percent:
            logger.info(f"E2E Verification PASSED: 100% operational ({passed}/{total} tests)")
        else:
            logger.error(
                f"E2E Verification FAILED: {failed} failures, {degraded} degraded ({pass_rate:.1f}% pass rate)"
            )
            for ft in failed_tests[:5]:
                logger.error(f"  - {ft.test_name}: {ft.error_message}")

        return report

    def _generate_recommendations(
        self, failed_tests: list[TestResult], by_category: dict
    ) -> list[str]:
        """Generate actionable recommendations based on failures"""
        recommendations = []

        # Check for category-wide failures
        for cat, stats in by_category.items():
            if stats["pass_rate"] == 0:
                recommendations.append(
                    f"CRITICAL: {cat} is completely down - immediate investigation required"
                )
            elif stats["pass_rate"] < 50:
                recommendations.append(
                    f"WARNING: {cat} has major issues ({stats['pass_rate']:.0f}% pass rate)"
                )

        # Check for specific failure patterns
        timeout_count = sum(1 for t in failed_tests if t.status == VerificationStatus.TIMEOUT)
        if timeout_count > 3:
            recommendations.append(
                f"PERFORMANCE: {timeout_count} endpoints timing out - check server resources"
            )

        unreachable_count = sum(
            1 for t in failed_tests if t.status == VerificationStatus.UNREACHABLE
        )
        if unreachable_count > 0:
            recommendations.append(
                f"NETWORK: {unreachable_count} endpoints unreachable - check DNS and connectivity"
            )

        invalid_count = sum(
            1 for t in failed_tests if t.status == VerificationStatus.INVALID_RESPONSE
        )
        if invalid_count > 0:
            recommendations.append(
                f"API: {invalid_count} endpoints returning invalid responses - check API contracts"
            )

        if not recommendations and failed_tests:
            recommendations.append("Review individual test failures for specific issues")

        if not failed_tests:
            recommendations.append("All systems operational - no action required")

        return recommendations

    async def run_quick_health_check(
        self,
        api_key_override: Optional[str] = None,
        skip_erp: bool = False,
    ) -> dict[str, Any]:
        """Quick health check of critical endpoints only"""
        critical_tests = self._apply_scope(
            self._apply_api_key_override(
                [t for t in self.tests if t.critical],
                api_key_override,
            ),
            skip_erp=skip_erp,
        )

        # Same safeguards as full verification: limit concurrency + run UI browser tests serially.
        max_concurrency = max(1, int(os.getenv("E2E_MAX_CONCURRENCY", "4")))
        semaphore = asyncio.Semaphore(max_concurrency)

        def _should_run_serial(test: EndpointTest) -> bool:
            return (
                "/api/v1/always-know/chatgpt-agent-test" in test.url
                or test.name.startswith("ChatGPT Agent UI")
                or "/market-intelligence/" in test.url
            )

        async with aiohttp.ClientSession() as session:
            results_by_index: dict[int, TestResult] = {}
            parallel_indices = [
                i for i, t in enumerate(critical_tests) if not _should_run_serial(t)
            ]
            serial_indices = [i for i, t in enumerate(critical_tests) if _should_run_serial(t)]

            async def _run_index(i: int) -> tuple[int, TestResult]:
                async with semaphore:
                    return i, await self._run_single_test(critical_tests[i], session)

            if parallel_indices:
                pairs = await asyncio.gather(*[_run_index(i) for i in parallel_indices])
                results_by_index.update(dict(pairs))

            for i in serial_indices:
                results_by_index[i] = await self._run_single_test(critical_tests[i], session)

            results = [results_by_index[i] for i in range(len(critical_tests))]

        passed = sum(1 for r in results if r.status == VerificationStatus.PASSED)
        failed = [r for r in results if r.status != VerificationStatus.PASSED]

        return {
            "is_healthy": len(failed) == 0,
            "critical_tests": len(critical_tests),
            "passed": passed,
            "failed": len(failed),
            "failed_endpoints": [{"name": f.test_name, "error": f.error_message} for f in failed],
            "skip_erp": skip_erp,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_report_summary(self) -> Optional[dict[str, Any]]:
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
            "recommendations": self.last_report.recommendations,
            "skip_erp": getattr(self.last_report, "skip_erp", False),
        }


# Singleton instance
e2e_verification = E2ESystemVerification()


# ============================================
# API FUNCTIONS
# ============================================


async def run_full_e2e_verification(
    api_key_override: Optional[str] = None,
    skip_erp: bool = False,
) -> dict[str, Any]:
    """Run complete E2E verification and return report"""
    report = await e2e_verification.run_full_verification(
        api_key_override=api_key_override, skip_erp=skip_erp
    )

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
        "skip_erp": getattr(report, "skip_erp", False),
        "overall_status": report.overall_status.value,
        "pass_rate": report.pass_rate,
        "summary": {
            "total_tests": report.total_tests,
            "passed": report.passed,
            "failed": report.failed,
            "degraded": report.degraded,
        },
        "results_by_category": report.results_by_category,
        "failed_tests": failed_tests_dicts,
        "recommendations": report.recommendations,
    }


async def run_quick_health_check(
    api_key_override: Optional[str] = None,
    skip_erp: bool = False,
) -> dict[str, Any]:
    """Run quick health check of critical systems"""
    return await e2e_verification.run_quick_health_check(
        api_key_override=api_key_override, skip_erp=skip_erp
    )


async def get_last_verification_report() -> Optional[dict[str, Any]]:
    """Get the last verification report summary"""
    return e2e_verification.get_report_summary()

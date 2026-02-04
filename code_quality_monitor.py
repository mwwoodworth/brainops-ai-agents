"""
CODE QUALITY & INTEGRATION MONITOR
===================================

This system provides DEEP monitoring beyond service health:
- Runtime error detection from application logs
- Broken integration detection (API failures, missing endpoints)
- Code quality issues (TypeErrors, AttributeErrors, ImportErrors)
- Database connection and query issues
- Build and deployment failures

Integrates with the Neural Core to provide TRUE awareness.

Created: 2026-01-27
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for detected issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IssueType(Enum):
    """Types of issues the monitor can detect"""
    RUNTIME_ERROR = "runtime_error"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    ATTRIBUTE_ERROR = "attribute_error"
    CONNECTION_ERROR = "connection_error"
    API_INTEGRATION = "api_integration"
    DATABASE_ERROR = "database_error"
    BUILD_FAILURE = "build_failure"
    MISSING_ENDPOINT = "missing_endpoint"
    AUTHENTICATION_ERROR = "authentication_error"
    CONFIGURATION_ERROR = "configuration_error"
    TIMEOUT_ERROR = "timeout_error"
    DEPENDENCY_ERROR = "dependency_error"


@dataclass
class DetectedIssue:
    """A detected code quality or integration issue"""
    id: str
    issue_type: IssueType
    severity: IssueSeverity
    source: str  # Which system/file/endpoint
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    auto_fixable: bool = False
    fix_suggestion: Optional[str] = None
    resolved: bool = False


class CodeQualityMonitor:
    """
    DEEP CODE QUALITY AND INTEGRATION MONITOR

    This goes beyond service health to detect actual code issues.
    """

    # Known error patterns to detect
    ERROR_PATTERNS = {
        r"ImportError|ModuleNotFoundError": {
            "type": IssueType.IMPORT_ERROR,
            "severity": IssueSeverity.ERROR,
            "fix_suggestion": "Install missing module or check import path"
        },
        r"TypeError": {
            "type": IssueType.TYPE_ERROR,
            "severity": IssueSeverity.ERROR,
            "fix_suggestion": "Check argument types and function signatures"
        },
        r"AttributeError": {
            "type": IssueType.ATTRIBUTE_ERROR,
            "severity": IssueSeverity.ERROR,
            "fix_suggestion": "Check object has required attribute or method"
        },
        r"ConnectionError|ConnectionRefused": {
            "type": IssueType.CONNECTION_ERROR,
            "severity": IssueSeverity.CRITICAL,
            "fix_suggestion": "Check service is running and network connectivity"
        },
        r"TimeoutError|timeout": {
            "type": IssueType.TIMEOUT_ERROR,
            "severity": IssueSeverity.WARNING,
            "fix_suggestion": "Increase timeout or optimize slow operations"
        },
        r"DatabaseError|OperationalError|psycopg": {
            "type": IssueType.DATABASE_ERROR,
            "severity": IssueSeverity.CRITICAL,
            "fix_suggestion": "Check database connection and query syntax"
        },
        r"401|Unauthorized|AuthenticationError": {
            "type": IssueType.AUTHENTICATION_ERROR,
            "severity": IssueSeverity.ERROR,
            "fix_suggestion": "Check API keys and authentication tokens"
        },
        r"404|Not Found|Missing endpoint": {
            "type": IssueType.MISSING_ENDPOINT,
            "severity": IssueSeverity.ERROR,
            "fix_suggestion": "Check endpoint path and API version"
        },
        r"KeyError|ConfigurationError|Missing env": {
            "type": IssueType.CONFIGURATION_ERROR,
            "severity": IssueSeverity.ERROR,
            "fix_suggestion": "Check environment variables and config files"
        }
    }

    # Integration endpoints to test
    INTEGRATION_TESTS = {
        "brainops_backend": {
            "url": "https://brainops-backend-prod.onrender.com",
            "endpoints": [
                {"path": "/health", "method": "GET", "expected_status": 200},
                {"path": "/api/v1/tenant", "method": "GET", "expected_status": [200, 401]},
            ]
        },
        "brainops_ai_agents": {
            "url": "https://brainops-ai-agents.onrender.com",
            "endpoints": [
                {"path": "/health", "method": "GET", "expected_status": 200},
                {"path": "/brain/status", "method": "GET", "expected_status": 200},
                {"path": "/agents", "method": "GET", "expected_status": 200},
                {"path": "/bleeding-edge/status", "method": "GET", "expected_status": 200},
            ]
        },
        "mcp_bridge": {
            "url": "https://brainops-mcp-bridge.onrender.com",
            "endpoints": [
                {"path": "/health", "method": "GET", "expected_status": 200},
            ]
        }
    }

    def __init__(self):
        self.detected_issues: List[DetectedIssue] = []
        self.issue_count = 0
        self.last_scan: Optional[datetime] = None
        self.api_key = ((os.getenv("BRAINOPS_API_KEY") or os.getenv("API_KEYS") or "").split(",")[0]).strip()
        self.render_api_key = os.getenv("RENDER_API_KEY", "")

    async def full_scan(self) -> Dict[str, Any]:
        """
        COMPREHENSIVE SCAN

        Run all monitors and return a complete health picture.
        """
        self.last_scan = datetime.now(timezone.utc)
        results = {
            "scan_timestamp": self.last_scan.isoformat(),
            "integrations": await self.check_integrations(),
            "runtime_errors": await self.check_runtime_errors(),
            "deployments": await self.check_deployment_status(),
            "database": await self.check_database_health(),
            "total_issues": len(self.detected_issues),
            "critical_issues": len([i for i in self.detected_issues if i.severity == IssueSeverity.CRITICAL]),
            "auto_fixable": len([i for i in self.detected_issues if i.auto_fixable])
        }
        return results

    async def check_integrations(self) -> Dict[str, Any]:
        """Test all integration endpoints"""
        results = {}

        async with httpx.AsyncClient() as client:
            for service_name, config in self.INTEGRATION_TESTS.items():
                service_results = {
                    "healthy": True,
                    "endpoints_tested": 0,
                    "endpoints_failed": 0,
                    "issues": []
                }

                for endpoint in config["endpoints"]:
                    service_results["endpoints_tested"] += 1

                    try:
                        url = f"{config['url']}{endpoint['path']}"
                        headers = {"X-API-Key": self.api_key} if self.api_key else {}

                        response = await client.request(
                            endpoint["method"],
                            url,
                            headers=headers,
                            timeout=10.0
                        )

                        expected = endpoint["expected_status"]
                        if isinstance(expected, list):
                            is_expected = response.status_code in expected
                        else:
                            is_expected = response.status_code == expected

                        if not is_expected:
                            service_results["endpoints_failed"] += 1
                            service_results["healthy"] = False

                            issue = DetectedIssue(
                                id=f"issue_{self.issue_count}_{int(datetime.now().timestamp())}",
                                issue_type=IssueType.API_INTEGRATION,
                                severity=IssueSeverity.ERROR if response.status_code >= 500 else IssueSeverity.WARNING,
                                source=f"{service_name}:{endpoint['path']}",
                                message=f"Unexpected status {response.status_code} (expected {expected})",
                                details={
                                    "url": url,
                                    "method": endpoint["method"],
                                    "status_code": response.status_code,
                                    "expected": expected
                                },
                                auto_fixable=response.status_code >= 500,
                                fix_suggestion="Restart service" if response.status_code >= 500 else "Check endpoint implementation"
                            )
                            self.detected_issues.append(issue)
                            self.issue_count += 1
                            service_results["issues"].append(issue.message)

                    except httpx.TimeoutException:
                        service_results["endpoints_failed"] += 1
                        service_results["healthy"] = False
                        issue = DetectedIssue(
                            id=f"issue_{self.issue_count}_{int(datetime.now().timestamp())}",
                            issue_type=IssueType.TIMEOUT_ERROR,
                            severity=IssueSeverity.CRITICAL,
                            source=f"{service_name}:{endpoint['path']}",
                            message=f"Timeout connecting to {endpoint['path']}",
                            auto_fixable=True,
                            fix_suggestion="Restart service or increase timeout"
                        )
                        self.detected_issues.append(issue)
                        self.issue_count += 1
                        service_results["issues"].append(issue.message)

                    except Exception as e:
                        service_results["endpoints_failed"] += 1
                        service_results["healthy"] = False
                        issue = DetectedIssue(
                            id=f"issue_{self.issue_count}_{int(datetime.now().timestamp())}",
                            issue_type=IssueType.CONNECTION_ERROR,
                            severity=IssueSeverity.CRITICAL,
                            source=f"{service_name}:{endpoint['path']}",
                            message=str(e),
                            auto_fixable=True,
                            fix_suggestion="Check network connectivity and restart service"
                        )
                        self.detected_issues.append(issue)
                        self.issue_count += 1
                        service_results["issues"].append(str(e))

                results[service_name] = service_results

        return results

    async def check_runtime_errors(self) -> Dict[str, Any]:
        """Check for runtime errors in recent logs via Render API"""
        if not self.render_api_key:
            return {"error": "No Render API key configured"}

        services = [
            ("srv-d413iu75r7bs738btc10", "AI Agents"),
            ("srv-d1tfs4idbo4c73di6k00", "Backend"),
            ("srv-d4rhvg63jp1c73918770", "MCP Bridge")
        ]

        results = {
            "services_checked": 0,
            "errors_found": 0,
            "warnings_found": 0,
            "issues": []
        }

        async with httpx.AsyncClient() as client:
            for service_id, service_name in services:
                try:
                    # Get recent logs from Render (last 100 lines)
                    response = await client.get(
                        f"https://api.render.com/v1/services/{service_id}/logs",
                        headers={"Authorization": f"Bearer {self.render_api_key}"},
                        params={"limit": 100},
                        timeout=30.0
                    )

                    results["services_checked"] += 1

                    if response.status_code == 200:
                        logs = response.json()
                        for log_entry in logs:
                            log_text = log_entry.get("message", "")

                            # Check for error patterns
                            for pattern, config in self.ERROR_PATTERNS.items():
                                if re.search(pattern, log_text, re.IGNORECASE):
                                    severity = config["severity"]
                                    if severity == IssueSeverity.ERROR:
                                        results["errors_found"] += 1
                                    elif severity == IssueSeverity.WARNING:
                                        results["warnings_found"] += 1

                                    issue = DetectedIssue(
                                        id=f"issue_{self.issue_count}_{int(datetime.now().timestamp())}",
                                        issue_type=config["type"],
                                        severity=severity,
                                        source=service_name,
                                        message=log_text[:200],
                                        details={"log_entry": log_entry},
                                        fix_suggestion=config["fix_suggestion"]
                                    )
                                    self.detected_issues.append(issue)
                                    self.issue_count += 1
                                    results["issues"].append({
                                        "service": service_name,
                                        "type": config["type"].value,
                                        "message": log_text[:100]
                                    })
                                    break  # Only match first pattern per log line

                except Exception as e:
                    logger.warning(f"Failed to check logs for {service_name}: {e}")

        return results

    async def check_deployment_status(self) -> Dict[str, Any]:
        """Check recent deployment status"""
        if not self.render_api_key:
            return {"error": "No Render API key configured"}

        services = [
            ("srv-d413iu75r7bs738btc10", "AI Agents"),
            ("srv-d1tfs4idbo4c73di6k00", "Backend"),
            ("srv-d4rhvg63jp1c73918770", "MCP Bridge")
        ]

        results = {
            "services_checked": 0,
            "failed_deployments": 0,
            "recent_deployments": []
        }

        async with httpx.AsyncClient() as client:
            for service_id, service_name in services:
                try:
                    response = await client.get(
                        f"https://api.render.com/v1/services/{service_id}/deploys",
                        headers={"Authorization": f"Bearer {self.render_api_key}"},
                        params={"limit": 3},
                        timeout=30.0
                    )

                    results["services_checked"] += 1

                    if response.status_code == 200:
                        deploys = response.json()
                        for deploy in deploys:
                            deploy_data = deploy.get("deploy", deploy)
                            status = deploy_data.get("status", "unknown")

                            results["recent_deployments"].append({
                                "service": service_name,
                                "status": status,
                                "created_at": deploy_data.get("createdAt"),
                                "finished_at": deploy_data.get("finishedAt")
                            })

                            if status in ["build_failed", "update_failed", "canceled"]:
                                results["failed_deployments"] += 1

                                issue = DetectedIssue(
                                    id=f"issue_{self.issue_count}_{int(datetime.now().timestamp())}",
                                    issue_type=IssueType.BUILD_FAILURE,
                                    severity=IssueSeverity.CRITICAL,
                                    source=service_name,
                                    message=f"Deployment {status}",
                                    details=deploy_data,
                                    auto_fixable=True,
                                    fix_suggestion="Review build logs and fix errors, then redeploy"
                                )
                                self.detected_issues.append(issue)
                                self.issue_count += 1

                except Exception as e:
                    logger.warning(f"Failed to check deployments for {service_name}: {e}")

        return results

    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and health"""
        results = {
            "connected": False,
            "query_successful": False,
            "issues": []
        }

        try:
            from database.async_connection import get_pool

            pool = get_pool()
            if pool:
                results["connected"] = True

                # Try a simple query
                try:
                    test_result = await pool.fetchval("SELECT 1")
                    results["query_successful"] = test_result == 1
                except Exception as e:
                    issue = DetectedIssue(
                        id=f"issue_{self.issue_count}_{int(datetime.now().timestamp())}",
                        issue_type=IssueType.DATABASE_ERROR,
                        severity=IssueSeverity.CRITICAL,
                        source="database",
                        message=f"Query failed: {e}",
                        auto_fixable=False,
                        fix_suggestion="Check database connectivity and credentials"
                    )
                    self.detected_issues.append(issue)
                    self.issue_count += 1
                    results["issues"].append(str(e))

        except ImportError:
            results["issues"].append("Database module not available")
        except Exception as e:
            results["issues"].append(str(e))

        return results

    def get_all_issues(self, unresolved_only: bool = True) -> List[Dict[str, Any]]:
        """Get all detected issues"""
        issues = self.detected_issues
        if unresolved_only:
            issues = [i for i in issues if not i.resolved]

        return [
            {
                "id": i.id,
                "type": i.issue_type.value,
                "severity": i.severity.value,
                "source": i.source,
                "message": i.message,
                "auto_fixable": i.auto_fixable,
                "fix_suggestion": i.fix_suggestion,
                "timestamp": i.timestamp.isoformat(),
                "resolved": i.resolved
            }
            for i in issues
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of code quality status"""
        unresolved = [i for i in self.detected_issues if not i.resolved]

        by_severity = {}
        for issue in unresolved:
            sev = issue.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        by_type = {}
        for issue in unresolved:
            t = issue.issue_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "total_issues": len(self.detected_issues),
            "unresolved_issues": len(unresolved),
            "auto_fixable": len([i for i in unresolved if i.auto_fixable]),
            "by_severity": by_severity,
            "by_type": by_type,
            "critical_count": by_severity.get("critical", 0),
            "error_count": by_severity.get("error", 0)
        }


# Global instance
_code_quality_monitor: Optional[CodeQualityMonitor] = None


def get_code_quality_monitor() -> CodeQualityMonitor:
    """Get the global code quality monitor instance"""
    global _code_quality_monitor
    if _code_quality_monitor is None:
        _code_quality_monitor = CodeQualityMonitor()
    return _code_quality_monitor


async def run_full_scan() -> Dict[str, Any]:
    """Run a full code quality scan"""
    monitor = get_code_quality_monitor()
    return await monitor.full_scan()

#!/usr/bin/env python3
"""
Playwright UI/UX Testing System for BrainOps AI OS.

Production-ready UI testing for live frontends with:
- Playwright browser automation
- AI vision analysis of screenshots
- Automatic issue reporting
- Database persistence
- Agent-compatible interface
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ai_ui_testing import (
    AIUITestingEngine,
    ERP_ROUTES,
    MRG_ROUTES,
    TestSeverity,
    TestStatus,
    UITestResult,
)
from database.async_connection import PoolConfig, get_pool, init_pool, using_fallback

logger = logging.getLogger(__name__)

DEFAULT_BASE_URLS = {
    "weathercraft-erp": "https://weathercraft-erp.vercel.app",
    "myroofgenius": "https://myroofgenius.com",
}


@dataclass
class UITestTarget:
    """Represents a live application target to test."""

    name: str
    base_url: str
    routes: List[str]
    tags: List[str] = field(default_factory=list)


@dataclass
class UITestRunConfig:
    """Runtime configuration for Playwright UI testing."""

    max_concurrency: int = 3
    route_timeout_seconds: int = 180
    playwright_init_timeout_seconds: int = 30
    wait_for_load: bool = True
    check_accessibility: bool = True
    check_performance: bool = True
    vision_provider: Optional[str] = None

    @classmethod
    def from_env(cls) -> "UITestRunConfig":
        return cls(
            max_concurrency=_env_int("UI_TEST_MAX_CONCURRENCY", 3),
            route_timeout_seconds=_env_int("UI_TEST_ROUTE_TIMEOUT_SECONDS", 180),
            playwright_init_timeout_seconds=_env_int("UI_TEST_PLAYWRIGHT_INIT_TIMEOUT", 30),
            wait_for_load=_env_bool("UI_TEST_WAIT_FOR_LOAD", True),
            check_accessibility=_env_bool("UI_TEST_CHECK_ACCESSIBILITY", True),
            check_performance=_env_bool("UI_TEST_CHECK_PERFORMANCE", True),
            vision_provider=os.getenv("UI_TEST_VISION_PROVIDER"),
        )


@dataclass
class UIIssue:
    """Normalized UI issue extracted from AI vision + checks."""

    severity: str
    category: str
    description: str
    url: str
    route: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "url": self.url,
            "route": self.route,
            "location": self.location,
            "source": self.source,
        }


class UIPlaywrightTestStore:
    """Database persistence for UI test runs and issues."""

    def __init__(self) -> None:
        self._initialized = False
        self._memory_runs: Dict[str, Dict[str, Any]] = {}
        self._memory_issues: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._ensure_pool()
        if using_fallback():
            logger.warning("Database fallback active; UI test results will be cached in memory.")
            self._initialized = True
            return
        await self._ensure_schema()
        self._initialized = True

    async def persist_run(self, run_record: Dict[str, Any]) -> None:
        await self.initialize()
        if using_fallback():
            self._memory_runs[run_record["test_id"]] = run_record
            return
        try:
            pool = get_pool()
            await pool.execute(
                """
                INSERT INTO ui_test_results (
                    test_id, application, url, status, severity, message,
                    ai_analysis, performance_metrics, accessibility_issues, suggestions,
                    routes_tested, issues_found, started_at, completed_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7::jsonb, $8::jsonb, $9::jsonb, $10::jsonb,
                    $11, $12, $13, $14
                )
                ON CONFLICT (test_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    severity = EXCLUDED.severity,
                    message = EXCLUDED.message,
                    ai_analysis = EXCLUDED.ai_analysis,
                    performance_metrics = EXCLUDED.performance_metrics,
                    accessibility_issues = EXCLUDED.accessibility_issues,
                    suggestions = EXCLUDED.suggestions,
                    routes_tested = EXCLUDED.routes_tested,
                    issues_found = EXCLUDED.issues_found,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at
                """,
                run_record["test_id"],
                run_record.get("application"),
                run_record.get("url"),
                run_record.get("status"),
                run_record.get("severity"),
                run_record.get("message"),
                json.dumps(run_record.get("ai_analysis", {}), default=str),
                json.dumps(run_record.get("performance_metrics", {}), default=str),
                json.dumps(run_record.get("accessibility_issues", []), default=str),
                json.dumps(run_record.get("suggestions", []), default=str),
                run_record.get("routes_tested", 0),
                run_record.get("issues_found", 0),
                run_record.get("started_at"),
                run_record.get("completed_at"),
            )
        except Exception as exc:
            logger.error("Failed to persist UI test run: %s", exc)
            self._memory_runs[run_record["test_id"]] = run_record

    async def persist_issues(self, issues: Iterable[UIIssue], test_id: str, application: str) -> None:
        await self.initialize()
        issues_list = [issue.as_dict() if isinstance(issue, UIIssue) else issue for issue in issues]
        if using_fallback():
            for issue in issues_list:
                self._memory_issues.append({**issue, "test_id": test_id, "application": application})
            return
        if not issues_list:
            return
        try:
            pool = get_pool()
            values = [
                (
                    test_id,
                    application,
                    issue.get("url"),
                    issue.get("severity"),
                    issue.get("category"),
                    issue.get("description"),
                    issue.get("location"),
                )
                for issue in issues_list
            ]
            await pool.executemany(
                """
                INSERT INTO ui_test_issues (
                    test_id, application, url, severity, category, description, location
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                values,
            )
        except Exception as exc:
            logger.error("Failed to persist UI issues: %s", exc)
            for issue in issues_list:
                self._memory_issues.append({**issue, "test_id": test_id, "application": application})

    async def _ensure_pool(self) -> None:
        try:
            get_pool()
            return
        except RuntimeError:
            pass

        db_password = os.getenv("DB_PASSWORD") or os.getenv("SUPABASE_DB_PASSWORD")
        if not db_password and os.getenv("ENVIRONMENT") == "production":
            raise RuntimeError("DB_PASSWORD must be set for production UI testing.")

        pool_config = PoolConfig(
            host=os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
            port=int(os.getenv("DB_PORT", "5432")),
            user=os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
            password=db_password or "",
            database=os.getenv("DB_NAME", "postgres"),
            min_size=_env_int("UI_TEST_DB_MIN_POOL", 1),
            max_size=_env_int("UI_TEST_DB_MAX_POOL", 4),
            ssl=_env_bool("DB_SSL", True),
            ssl_verify=_env_bool("DB_SSL_VERIFY", True),
        )
        await init_pool(pool_config)

    async def _ensure_schema(self) -> None:
        pool = get_pool()
        await pool.execute(
            """
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
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        await pool.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ui_test_app
            ON ui_test_results(application);
            """
        )
        await pool.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ui_test_completed
            ON ui_test_results(completed_at DESC);
            """
        )
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS ui_test_issues (
                id SERIAL PRIMARY KEY,
                test_id VARCHAR(64) NOT NULL,
                application VARCHAR(128),
                url TEXT,
                severity VARCHAR(32),
                category VARCHAR(64),
                description TEXT,
                location TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        await pool.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ui_test_issues_test
            ON ui_test_issues(test_id);
            """
        )
        await pool.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ui_test_issues_severity
            ON ui_test_issues(severity);
            """
        )


class UIPlaywrightTestRunner:
    """Orchestrates UI testing with Playwright + AI vision."""

    def __init__(
        self,
        targets: Optional[Dict[str, UITestTarget]] = None,
        run_config: Optional[UITestRunConfig] = None,
        store: Optional[UIPlaywrightTestStore] = None,
    ) -> None:
        self.targets = targets or _default_targets()
        self.run_config = run_config or UITestRunConfig.from_env()
        self.store = store or UIPlaywrightTestStore()
        self._engine: Optional[AIUITestingEngine] = None

    async def initialize(self) -> None:
        if self._engine:
            return
        await self.store.initialize()
        self._engine = AIUITestingEngine()
        await self._engine.initialize(timeout_seconds=self.run_config.playwright_init_timeout_seconds)
        if self.run_config.vision_provider:
            self._engine.vision_analyzer.preferred_model = self.run_config.vision_provider

    async def close(self) -> None:
        if self._engine:
            await self._engine.close()
            self._engine = None

    async def run_targets(
        self,
        target_names: Optional[List[str]] = None,
        triggered_by: str = "manual",
        store_results: bool = True,
    ) -> Dict[str, Any]:
        await self.initialize()
        selected = self._resolve_targets(target_names)
        results: Dict[str, Any] = {"targets": [], "status": "completed"}

        for target in selected:
            try:
                target_result = await self.run_target(
                    target,
                    triggered_by=triggered_by,
                    store_results=store_results,
                )
                results["targets"].append(target_result)
            except Exception as exc:
                logger.error("UI testing failed for %s: %s", target.name, exc)
                results["targets"].append(
                    {
                        "application": target.name,
                        "base_url": target.base_url,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                results["status"] = "partial_failure"

        await self.close()
        return results

    async def run_target(
        self,
        target: UITestTarget,
        triggered_by: str = "manual",
        store_results: bool = True,
    ) -> Dict[str, Any]:
        await self.initialize()
        if not self._engine:
            raise RuntimeError("Playwright UI testing engine failed to initialize.")

        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc)
        logger.info("Starting UI test run %s for %s", run_id, target.name)

        route_results = await self._run_routes(target)
        completed_at = datetime.now(timezone.utc)

        report, issues = self._build_report(
            run_id,
            target,
            route_results,
            started_at,
            completed_at,
            triggered_by,
        )

        if store_results:
            await self.store.persist_run(report["record"])
            await self.store.persist_issues(issues, run_id, target.name)

        return report["response"]

    async def run_single_url(
        self,
        url: str,
        test_name: str = "UI Test",
        triggered_by: str = "manual",
        store_results: bool = True,
    ) -> Dict[str, Any]:
        await self.initialize()
        if not self._engine:
            raise RuntimeError("Playwright UI testing engine failed to initialize.")

        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc)
        result = await self._run_route(test_name, url, route="custom", timeout=self.run_config.route_timeout_seconds)
        completed_at = datetime.now(timezone.utc)

        target = UITestTarget(name=test_name, base_url=url, routes=["custom"])
        report, issues = self._build_report(
            run_id,
            target,
            [result],
            started_at,
            completed_at,
            triggered_by,
        )

        if store_results:
            await self.store.persist_run(report["record"])
            await self.store.persist_issues(issues, run_id, test_name)

        return report["response"]

    def list_targets(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": target.name,
                "base_url": target.base_url,
                "routes": target.routes,
                "tags": target.tags,
            }
            for target in self.targets.values()
        ]

    def _resolve_targets(self, target_names: Optional[List[str]]) -> List[UITestTarget]:
        if not target_names:
            return list(self.targets.values())
        resolved = []
        seen = set()
        for name in target_names:
            normalized = str(name).strip().lower()
            for key, target in self.targets.items():
                if normalized in {key.lower(), target.name.lower(), target.base_url.lower()}:
                    if key not in seen:
                        resolved.append(target)
                        seen.add(key)
        if not resolved:
            raise ValueError(f"No matching UI targets for {target_names}")
        return resolved

    async def _run_routes(self, target: UITestTarget) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, self.run_config.max_concurrency))
        tasks = []
        for route in target.routes:
            route_path = route if str(route).startswith("/") else f"/{route}"
            url = f"{target.base_url.rstrip('/')}{route_path}"
            tasks.append(
                asyncio.create_task(
                    self._run_route(target.name, url, route, semaphore, self.run_config.route_timeout_seconds)
                )
            )
        return await asyncio.gather(*tasks)

    async def _run_route(
        self,
        target_name: str,
        url: str,
        route: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self._engine:
            raise RuntimeError("Playwright UI testing engine not initialized.")

        async def _execute() -> Tuple[UITestResult, float]:
            start = datetime.now(timezone.utc)
            result = await self._engine.test_url(
                url=url,
                test_name=f"{target_name}: {route}",
                wait_for_load=self.run_config.wait_for_load,
                check_accessibility=self.run_config.check_accessibility,
                check_performance=self.run_config.check_performance,
            )
            duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            return result, duration_ms

        if semaphore:
            async with semaphore:
                return await self._run_with_timeout(_execute, url, route, timeout)
        return await self._run_with_timeout(_execute, url, route, timeout)

    async def _run_with_timeout(
        self,
        coroutine_factory,
        url: str,
        route: str,
        timeout: Optional[int],
    ) -> Dict[str, Any]:
        try:
            if timeout:
                result, duration_ms = await asyncio.wait_for(coroutine_factory(), timeout=timeout)
            else:
                result, duration_ms = await coroutine_factory()
            return _serialize_result(result, url, route, duration_ms)
        except asyncio.TimeoutError:
            return _timeout_result(url, route, timeout or 0)
        except Exception as exc:
            return _error_result(url, route, str(exc))

    def _build_report(
        self,
        run_id: str,
        target: UITestTarget,
        route_results: List[Dict[str, Any]],
        started_at: datetime,
        completed_at: datetime,
        triggered_by: str,
    ) -> Tuple[Dict[str, Any], List[UIIssue]]:
        issues = _extract_issues(route_results)
        issue_payloads = [issue.as_dict() for issue in issues]
        issue_summary = _summarize_issues(issues)
        status, severity, message = _summarize_status(route_results, issue_summary)
        performance_summary = _summarize_performance(route_results)
        accessibility_summary = _summarize_accessibility(route_results)
        suggestions = _summarize_suggestions(route_results)

        record = {
            "test_id": run_id,
            "application": target.name,
            "url": target.base_url,
            "status": status,
            "severity": severity,
            "message": message,
            "routes_tested": len(route_results),
            "issues_found": issue_summary["total"],
            "started_at": started_at,
            "completed_at": completed_at,
            "ai_analysis": {
                "target": {
                    "name": target.name,
                    "base_url": target.base_url,
                    "routes": target.routes,
                },
                "route_results": route_results,
                "issue_summary": issue_summary,
                "issues": issue_payloads[:200],
                "triggered_by": triggered_by,
                "playwright_available": bool(getattr(self._engine, "_playwright_available", False)),
                "vision_provider": self.run_config.vision_provider,
                "vision_enabled": _vision_enabled(),
            },
            "performance_metrics": performance_summary,
            "accessibility_issues": accessibility_summary,
            "suggestions": suggestions,
        }

        response = {
            **record,
            "issues": issue_payloads,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        }

        return {"record": record, "response": response}, issues


class PlaywrightUITestingAgent:
    """Agent wrapper for Playwright UI/UX testing."""

    def __init__(self) -> None:
        self.name = "UIPlaywrightTesting"
        self.agent_type = "ui_testing_playwright"
        self._runner = UIPlaywrightTestRunner()

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        action = (task.get("action") or task.get("type") or "run_all").lower()
        triggered_by = task.get("triggered_by", "agent")
        store_results = task.get("store_results", True)

        if action in {"list_targets", "targets"}:
            return {"status": "ok", "targets": self._runner.list_targets()}

        if action in {"run_url", "single_url"} and task.get("url"):
            result = await self._runner.run_single_url(
                url=task["url"],
                test_name=task.get("test_name", "UI Test"),
                triggered_by=triggered_by,
                store_results=store_results,
            )
            return {"status": "completed", "result": result}

        if action in {"run_target", "target"} and task.get("target"):
            target_names = [task["target"]] if isinstance(task["target"], str) else task["target"]
            result = await self._runner.run_targets(
                target_names=target_names,
                triggered_by=triggered_by,
                store_results=store_results,
            )
            return {"status": "completed", "result": result}

        result = await self._runner.run_targets(
            target_names=[task["targets"]] if isinstance(task.get("targets"), str) else task.get("targets"),
            triggered_by=triggered_by,
            store_results=store_results,
        )
        return {"status": "completed", "result": result}


def _default_targets() -> Dict[str, UITestTarget]:
    return {
        "weathercraft-erp": UITestTarget(
            name="Weathercraft ERP",
            base_url=DEFAULT_BASE_URLS["weathercraft-erp"],
            routes=ERP_ROUTES,
            tags=["erp", "internal", "playwright"],
        ),
        "myroofgenius": UITestTarget(
            name="MyRoofGenius",
            base_url=DEFAULT_BASE_URLS["myroofgenius"],
            routes=MRG_ROUTES,
            tags=["marketing", "consumer", "playwright"],
        ),
    }


def _serialize_result(result: UITestResult, url: str, route: str, duration_ms: float) -> Dict[str, Any]:
    ai_analysis = result.ai_analysis or {}
    return {
        "route": route,
        "url": url,
        "status": result.status.value if isinstance(result.status, TestStatus) else str(result.status),
        "severity": result.severity.value if isinstance(result.severity, TestSeverity) else str(result.severity),
        "message": result.message,
        "duration_ms": round(duration_ms, 2),
        "screenshot_path": result.screenshot_path,
        "ai_analysis": ai_analysis,
        "performance_metrics": result.performance_metrics or {},
        "accessibility_issues": result.accessibility_issues or [],
        "suggestions": result.suggestions or [],
    }


def _timeout_result(url: str, route: str, timeout: int) -> Dict[str, Any]:
    return {
        "route": route,
        "url": url,
        "status": "failed",
        "severity": "high",
        "message": f"Route timed out after {timeout}s",
        "duration_ms": timeout * 1000,
        "ai_analysis": {},
        "performance_metrics": {},
        "accessibility_issues": [],
        "suggestions": ["Investigate slow page load or blocking requests."],
    }


def _error_result(url: str, route: str, error: str) -> Dict[str, Any]:
    return {
        "route": route,
        "url": url,
        "status": "error",
        "severity": "critical",
        "message": f"Route test error: {error}",
        "duration_ms": 0,
        "ai_analysis": {},
        "performance_metrics": {},
        "accessibility_issues": [],
        "suggestions": ["Check logs for Playwright or network errors."],
    }


def _extract_issues(route_results: List[Dict[str, Any]]) -> List[UIIssue]:
    issues: List[UIIssue] = []

    for result in route_results:
        url = result.get("url", "")
        route = result.get("route")
        ai_analysis = result.get("ai_analysis", {}) or {}

        for category, issue_key in (
            ("visual", "visual_issues"),
            ("functional", "functional_issues"),
            ("accessibility", "accessibility_issues"),
            ("usability", "usability_issues"),
        ):
            for issue in ai_analysis.get(issue_key, []) or []:
                issues.append(
                    UIIssue(
                        severity=_normalize_severity(issue.get("severity")),
                        category=category,
                        description=issue.get("issue") or issue.get("description") or "Unspecified issue",
                        url=url,
                        route=route,
                        location=issue.get("location") or issue.get("wcag"),
                        source="ai_vision",
                    )
                )

        for a11y_issue in result.get("accessibility_issues", []) or []:
            issues.append(
                UIIssue(
                    severity=_normalize_severity(a11y_issue.get("impact")),
                    category="accessibility",
                    description=a11y_issue.get("description") or a11y_issue.get("help") or "Accessibility issue",
                    url=url,
                    route=route,
                    location=a11y_issue.get("id"),
                    source="axe",
                )
            )

        perf = result.get("performance_metrics") or {}
        load_complete = perf.get("load_complete")
        if load_complete and load_complete > 5000:
            issues.append(
                UIIssue(
                    severity="medium" if load_complete < 10000 else "high",
                    category="performance",
                    description=f"Slow load complete: {load_complete}ms",
                    url=url,
                    route=route,
                    source="performance",
                )
            )

        if result.get("status") in {"failed", "error"}:
            issues.append(
                UIIssue(
                    severity="critical",
                    category="execution",
                    description=result.get("message", "UI test failure"),
                    url=url,
                    route=route,
                    source="runner",
                )
            )

    return issues


def _summarize_status(route_results: List[Dict[str, Any]], issue_summary: Dict[str, Any]) -> Tuple[str, str, str]:
    statuses = [result.get("status") for result in route_results]
    failed = sum(1 for status in statuses if status in {"failed", "error"})
    warnings = sum(1 for status in statuses if status == "warning")
    if failed:
        status = "failed"
    elif warnings:
        status = "warning"
    else:
        status = "passed"

    severity = issue_summary["max_severity"] or ("high" if failed else "low")
    message = f"{failed} failed, {warnings} warnings, {issue_summary['total']} issues"
    return status, severity, message


def _summarize_performance(route_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    load_times = [
        result.get("performance_metrics", {}).get("load_complete")
        for result in route_results
        if result.get("performance_metrics", {}).get("load_complete")
    ]
    if not load_times:
        return {}
    return {
        "max_load_complete_ms": max(load_times),
        "avg_load_complete_ms": sum(load_times) / len(load_times),
        "routes_with_metrics": len(load_times),
    }


def _summarize_accessibility(route_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aggregated: List[Dict[str, Any]] = []
    for result in route_results:
        aggregated.extend(result.get("accessibility_issues", []) or [])
    return aggregated[:200]


def _summarize_suggestions(route_results: List[Dict[str, Any]]) -> List[str]:
    suggestions: List[str] = []
    for result in route_results:
        suggestions.extend(result.get("suggestions", []) or [])
    return list(dict.fromkeys(suggestions))[:25]


def _summarize_issues(issues: List[UIIssue]) -> Dict[str, Any]:
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    max_severity = None
    for issue in issues:
        severity = issue.severity or "info"
        severity_counts.setdefault(severity, 0)
        severity_counts[severity] += 1
        if max_severity is None or _severity_rank(severity) > _severity_rank(max_severity):
            max_severity = severity
    return {
        "total": len(issues),
        "by_severity": severity_counts,
        "max_severity": max_severity,
    }


def _normalize_severity(raw: Optional[str]) -> str:
    if not raw:
        return "low"
    raw_lower = str(raw).lower()
    mapping = {
        "critical": "critical",
        "high": "high",
        "serious": "high",
        "medium": "medium",
        "moderate": "medium",
        "low": "low",
        "minor": "low",
        "info": "info",
    }
    return mapping.get(raw_lower, "low")


def _severity_rank(severity: str) -> int:
    return {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}.get(severity, 0)


def _vision_enabled() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


async def run_default_ui_tests() -> Dict[str, Any]:
    """Convenience entrypoint to run the default UI test suite."""
    runner = UIPlaywrightTestRunner()
    return await runner.run_targets()


if __name__ == "__main__":
    import sys

    async def _main() -> None:
        runner = UIPlaywrightTestRunner()
        if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
            result = await runner.run_single_url(sys.argv[1])
        else:
            result = await runner.run_targets()
        print(json.dumps(result, indent=2, default=str))

    asyncio.run(_main())

"""
Shared system status helpers extracted from app.py.

These functions are used by multiple route domains (health, observability, metrics)
and need to live in a shared location to avoid duplication.
"""
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def collect_active_systems(app_state: Any) -> list[str]:
    """Return a list of systems that are initialized and active.

    Accepts ``app.state`` so callers don't need to import the FastAPI app.
    """
    active: list[str] = []

    # fmt: off
    _checks: list[tuple[str, str, str]] = [
        ("aurea",               "AUREA_AVAILABLE",               "AUREA Orchestrator"),
        ("healer",              "SELF_HEALING_AVAILABLE",         "Self-Healing Recovery"),
        ("memory",              "MEMORY_AVAILABLE",               "Memory Manager"),
        ("embedded_memory",     "EMBEDDED_MEMORY_AVAILABLE",      "Embedded Memory (RAG)"),
        ("training",            "TRAINING_AVAILABLE",             "Training Pipeline"),
        ("learning",            "LEARNING_AVAILABLE",             "Learning System"),
        ("scheduler",           "SCHEDULER_AVAILABLE",            "Agent Scheduler"),
        ("nerve_center",        "NERVE_CENTER_AVAILABLE",         "NerveCenter (Operational Coordinator)"),
        ("operational_monitor", "OPERATIONAL_MONITOR_AVAILABLE",  "Operational Monitor"),
    ]
    # fmt: on

    # Lazy import to avoid circular dependency with app.py
    import app as _app_mod

    for state_attr, flag_name, label in _checks:
        flag = getattr(_app_mod, flag_name, False)
        if flag and getattr(app_state, state_attr, None):
            active.append(label)

    # AI Core uses a module-level reference instead of app.state
    if getattr(_app_mod, "AI_AVAILABLE", False) and getattr(_app_mod, "ai_core", None):
        active.append("AI Core")

    _agent_checks: list[tuple[str, str, str]] = [
        ("system_improvement", "SYSTEM_IMPROVEMENT_AVAILABLE", "System Improvement Agent"),
        ("devops_agent", "DEVOPS_AGENT_AVAILABLE", "DevOps Optimization Agent"),
        ("code_quality", "CODE_QUALITY_AVAILABLE", "Code Quality Agent"),
        ("customer_success", "CUSTOMER_SUCCESS_AVAILABLE", "Customer Success Agent"),
        ("competitive_intel", "COMPETITIVE_INTEL_AVAILABLE", "Competitive Intelligence Agent"),
        ("vision_alignment", "VISION_ALIGNMENT_AVAILABLE", "Vision Alignment Agent"),
    ]
    for state_attr, flag_name, label in _agent_checks:
        flag = getattr(_app_mod, flag_name, False)
        if flag and getattr(app_state, state_attr, None):
            active.append(label)

    if getattr(_app_mod, "RECONCILER_AVAILABLE", False) and getattr(app_state, "reconciler", None):
        active.append("Self-Healing Reconciler")
    if getattr(_app_mod, "BLEEDING_EDGE_AVAILABLE", False):
        active.append(
            "Bleeding Edge AI (OODA, Hallucination, Memory, Dependability, Consciousness, Circuit Breaker)"
        )
    if getattr(_app_mod, "AUTONOMOUS_RESOLVER_AVAILABLE", False):
        active.append("Autonomous Issue Resolver (Detects AND FIXES AI OS Issues)")
    if getattr(_app_mod, "MEMORY_ENFORCEMENT_AVAILABLE", False):
        active.append("Memory Enforcement (RBA/WBA, Verification, Audit)")
    if getattr(_app_mod, "MEMORY_HYGIENE_AVAILABLE", False):
        active.append("Memory Hygiene (Deduplication, Conflicts, Decay)")
    if getattr(_app_mod, "WORKFLOWS_AVAILABLE", False):
        active.append("Advanced Workflow Engine (LangGraph, OODA, HITL, Checkpoints)")

    return active


def scheduler_snapshot(app_state: Any) -> dict[str, Any]:
    """Return scheduler status with safe defaults."""
    import app as _app_mod

    scheduler = getattr(app_state, "scheduler", None)
    if not (getattr(_app_mod, "SCHEDULER_AVAILABLE", False) and scheduler):
        return {"enabled": False, "message": "Scheduler not available"}

    apscheduler_jobs = scheduler.scheduler.get_jobs()
    return {
        "enabled": True,
        "running": scheduler.scheduler.running,
        "registered_jobs_count": len(scheduler.registered_jobs),
        "apscheduler_jobs_count": len(apscheduler_jobs),
        "next_jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in apscheduler_jobs[:5]
        ],
    }


def aurea_status(app_state: Any) -> dict[str, Any]:
    """Return AUREA orchestrator status."""
    import app as _app_mod

    aurea = getattr(app_state, "aurea", None)
    if not (getattr(_app_mod, "AUREA_AVAILABLE", False) and aurea):
        return {"available": False, "running": False}
    try:
        return {**aurea.get_status(), "available": True}
    except Exception as exc:
        logger.error("Failed to read AUREA status: %s", exc)
        return {"available": True, "running": False, "error": str(exc)}


def self_healing_status(app_state: Any) -> dict[str, Any]:
    """Return self-healing system status."""
    import app as _app_mod

    healer = getattr(app_state, "healer", None)
    if not (getattr(_app_mod, "SELF_HEALING_AVAILABLE", False) and healer):
        return {"available": False}

    try:
        circuit_breakers = getattr(healer, "circuit_breakers", None) or {}
        breaker_total = len(circuit_breakers) if isinstance(circuit_breakers, dict) else 0
        rules = getattr(healer, "healing_rules", None) or []
        active_rules = len(rules) if hasattr(rules, "__len__") else None
        return {
            "available": True,
            "report_available": hasattr(healer, "get_health_report"),
            "circuit_breakers_total": breaker_total,
            "active_healing_rules": active_rules,
        }
    except Exception as exc:
        logger.error("Failed to read self-healing status: %s", exc)
        return {"available": True, "error": str(exc)}


async def memory_stats_snapshot(pool: Any) -> dict[str, Any]:
    """Get a fast snapshot of memory/learning health."""
    try:
        existing_tables = await pool.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('ai_persistent_memory', 'memory_entries', 'memories')
        """
        )
        if not existing_tables:
            return {"status": "not_configured"}

        table_names = [t["table_name"] for t in existing_tables]
        preferred = next(
            (t for t in ("ai_persistent_memory", "memory_entries", "memories") if t in table_names),
            table_names[0],
        )
        stats = await pool.fetchrow(f"SELECT COUNT(*) AS total FROM {preferred}")
        return {
            "status": "operational",
            "table": preferred,
            "total_records": stats["total"] if stats else 0,
        }
    except Exception as exc:
        logger.error("Failed to fetch memory stats: %s", exc)
        return {"status": "error", "error": str(exc)}


async def get_agent_usage(pool: Any) -> dict[str, Any]:
    """Fetch recent agent usage, trying both legacy and new table names."""
    queries: list[tuple[str, str, str, str]] = [
        ("ai_agents", "ai_agent_executions", "e.agent_name = a.name", "e.created_at"),
        ("agents", "ai_agent_executions", "e.agent_name = a.name", "e.created_at"),
    ]
    errors: list[str] = []

    for agents_table, executions_table, join_cond, time_col in queries:
        try:
            rows = await pool.fetch(
                f"""
                SELECT
                    a.id::text AS id,
                    a.name,
                    COALESCE(a.category, 'other') AS category,
                    COALESCE(a.enabled, true) AS enabled,
                    COUNT(e.id) AS executions_last_30d,
                    MAX({time_col}) AS last_execution,
                    AVG(e.execution_time_ms) FILTER (WHERE e.execution_time_ms IS NOT NULL) AS avg_duration_ms
                FROM {agents_table} a
                LEFT JOIN {executions_table} e
                    ON {join_cond}
                    AND {time_col} >= NOW() - INTERVAL '30 days'
                GROUP BY a.id, a.name, a.category, a.enabled
                ORDER BY executions_last_30d DESC, last_execution DESC NULLS LAST
                LIMIT 20
            """
            )
            usage = []
            for row in rows:
                data = row if isinstance(row, dict) else dict(row)
                usage.append(
                    {
                        "id": str(data.get("id")),
                        "name": data.get("name"),
                        "category": data.get("category"),
                        "enabled": bool(data.get("enabled", True)),
                        "executions_last_30d": int(data.get("executions_last_30d") or 0),
                        "last_execution": data.get("last_execution").isoformat()
                        if data.get("last_execution")
                        else None,
                        "avg_duration_ms": float(data.get("avg_duration_ms") or 0),
                    }
                )
            return {"agents": usage, "table": agents_table, "executions_table": executions_table}
        except Exception as exc:
            errors.append(f"{agents_table}/{executions_table}: {exc}")
            continue

    return {"agents": [], "warning": "No agent usage data available", "errors": errors[:2]}


async def get_schedule_usage(pool: Any) -> dict[str, Any]:
    """Fetch scheduler schedule rows with resiliency."""
    schedules: list[dict[str, Any]] = []
    try:
        rows = await pool.fetch(
            """
            SELECT
                s.id::text AS id,
                s.agent_id::text AS agent_id,
                s.enabled,
                s.frequency_minutes,
                s.created_at,
                COALESCE(a.name, s.agent_id::text) AS agent_name
            FROM public.agent_schedules s
            LEFT JOIN ai_agents a ON a.id = s.agent_id
            ORDER BY s.enabled DESC, s.created_at DESC NULLS LAST
            LIMIT 50
        """
        )
        for row in rows:
            data = row if isinstance(row, dict) else dict(row)
            schedules.append(
                {
                    "id": data.get("id"),
                    "agent_id": data.get("agent_id"),
                    "agent_name": data.get("agent_name"),
                    "enabled": bool(data.get("enabled", True)),
                    "frequency_minutes": data.get("frequency_minutes"),
                    "created_at": data.get("created_at").isoformat()
                    if data.get("created_at")
                    else None,
                }
            )
        return {"schedules": schedules, "table": "public.agent_schedules"}
    except Exception as exc:
        logger.error("Failed to load schedule usage: %s", exc)
        return {"schedules": schedules, "error": str(exc)}

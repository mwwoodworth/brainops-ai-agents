"""
Task Adapter — Translates between TaskMate schema and cc_tasks (canonical store)

Maps:
  taskmate_tasks.task_id     → cc_tasks.metadata->>'task_id'
  taskmate_tasks.priority    → cc_tasks.priority  (P0-P5 ↔ critical/high/medium/low)
  taskmate_tasks.status      → cc_tasks.status    (open/closed ↔ pending/completed)
  taskmate_tasks.owner       → cc_tasks.assigned_to
  taskmate_tasks.blocked_by  → cc_tasks.blocking_reason
  taskmate_tasks.evidence    → cc_tasks.metadata->>'evidence'
  taskmate_tasks.closed_at   → cc_tasks.completed_date
"""

import json
from typing import Optional


# --- Priority mapping ---

_TM_TO_CC_PRIORITY = {
    "P0": "critical",
    "P1": "critical",
    "P2": "high",
    "P3": "medium",
    "P4": "low",
    "P5": "low",
}

_CC_TO_TM_PRIORITY = {
    "critical": "P0",
    "high": "P2",
    "medium": "P3",
    "low": "P4",
}


def to_cc_priority(tm_priority: str) -> str:
    return _TM_TO_CC_PRIORITY.get(tm_priority, "medium")


def to_tm_priority(cc_priority: str) -> str:
    return _CC_TO_TM_PRIORITY.get(cc_priority, "P3")


# --- Status mapping ---

_TM_TO_CC_STATUS = {
    "open": "pending",
    "closed": "completed",
}

_CC_TO_TM_STATUS = {
    "pending": "open",
    "completed": "closed",
}


def to_cc_status(tm_status: str) -> str:
    return _TM_TO_CC_STATUS.get(tm_status, tm_status)


def to_tm_status(cc_status: str) -> str:
    return _CC_TO_TM_STATUS.get(cc_status, cc_status)


# --- Row conversion ---


def cc_row_to_taskmate(row: dict) -> dict:
    """Convert a cc_tasks row dict to taskmate response shape."""
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    task_id = metadata.get("task_id") or metadata.get("source_id") or str(row.get("id", ""))

    return {
        "id": row.get("id"),
        "task_id": task_id,
        "title": row.get("title"),
        "description": row.get("description"),
        "priority": to_tm_priority(row.get("priority", "medium")),
        "status": to_tm_status(row.get("status", "pending")),
        "owner": row.get("assigned_to"),
        "blocked_by": row.get("blocking_reason"),
        "evidence": metadata.get("evidence"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "closed_at": row.get("completed_date"),
    }


def build_cc_metadata(
    task_id: str,
    evidence: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    """Build metadata JSONB for cc_tasks from taskmate fields."""
    meta = {"task_id": task_id, "source": "taskmate_api"}
    if evidence:
        meta["evidence"] = evidence
    if extra:
        meta.update(extra)
    return json.dumps(meta)

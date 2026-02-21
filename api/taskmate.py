"""
TaskMate — Cross-Model Task Manager API
P1-TASKMATE-001 | Created 2026-02-13
Updated 2026-02-20: Unified to cc_tasks canonical store

Endpoints:
  GET    /taskmate/tasks           List tasks (filter: status, priority, owner)
  POST   /taskmate/tasks           Create task
  PATCH  /taskmate/tasks/{task_id} Update task
  GET    /taskmate/tasks/{task_id} Get task with comments
  POST   /taskmate/tasks/{task_id}/comments  Add comment
  GET    /taskmate/summary         Dashboard counts
  DELETE /taskmate/tasks/{task_id} Delete task (soft-delete)

Backend: cc_tasks (canonical store). API contract unchanged.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel

from database.async_connection import get_pool, using_fallback
from database.verify_tables import verify_tables_async
from api.task_adapter import (
    cc_row_to_taskmate,
    to_cc_priority,
    to_cc_status,
    to_tm_priority,
    to_tm_status,
    build_cc_metadata,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/taskmate", tags=["TaskMate"])

# cc_tasks is the canonical store; taskmate_comments kept for comment support
REQUIRED_TABLES = ["cc_tasks", "taskmate_comments"]

DEFAULT_TENANT_ID = "a17d1f59-7baf-4350-b0c1-1ea6ae2fbd2a"


# --- Pydantic Models ---


class TaskCreate(BaseModel):
    task_id: str
    title: str
    description: Optional[str] = None
    priority: str = "P2"
    status: str = "open"
    owner: Optional[str] = None
    blocked_by: Optional[str] = None
    evidence: Optional[str] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    owner: Optional[str] = None
    blocked_by: Optional[str] = None
    evidence: Optional[str] = None


class CommentCreate(BaseModel):
    author: str
    body: str


# --- Helpers ---


async def _check_tables():
    if using_fallback():
        raise HTTPException(503, "Database unavailable")
    pool = get_pool()
    ok = await verify_tables_async(REQUIRED_TABLES, pool, module_name="taskmate")
    if not ok:
        raise HTTPException(503, "TaskMate tables not available. Check cc_tasks.")
    return pool


def _task_id_where(param_idx: int) -> str:
    """WHERE clause to find a task by its taskmate task_id.

    Checks metadata->>'task_id' first (new tasks), then metadata->>'source_id'
    (migrated tasks), then falls back to id::text (cc_tasks native).
    """
    return (
        f"(metadata->>'task_id' = ${param_idx} "
        f"OR metadata->>'source_id' = ${param_idx} "
        f"OR id::text = ${param_idx})"
    )


# --- Endpoints ---


@router.get("/tasks")
async def list_tasks(
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    owner: Optional[str] = Query(None),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()

    conditions = ["deleted_at IS NULL"]
    params: list = []
    idx = 1

    if status:
        cc_status = to_cc_status(status)
        conditions.append(f"status = ${idx}")
        params.append(cc_status)
        idx += 1
    if priority:
        cc_priority = to_cc_priority(priority)
        conditions.append(f"priority = ${idx}")
        params.append(cc_priority)
        idx += 1
    if owner:
        conditions.append(f"assigned_to = ${idx}")
        params.append(owner)
        idx += 1

    where = " AND ".join(conditions)
    sql = f"""
        SELECT id, title, description, status, priority,
               assigned_to, blocking_reason, completed_date,
               metadata, created_at, updated_at
        FROM cc_tasks
        WHERE {where}
        ORDER BY
            CASE priority
                WHEN 'critical' THEN 0
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                ELSE 3
            END,
            created_at DESC
        LIMIT 100
    """

    rows = await pool.fetch(sql, *params)
    return {
        "tasks": [cc_row_to_taskmate(dict(r)) for r in rows],
        "count": len(rows),
    }


@router.post("/tasks", status_code=201)
async def create_task(
    task: TaskCreate,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()

    # Check for duplicate task_id
    existing = await pool.fetchval(
        "SELECT 1 FROM cc_tasks WHERE metadata->>'task_id' = $1 AND deleted_at IS NULL",
        task.task_id,
    )
    if existing:
        raise HTTPException(409, f"Task {task.task_id} already exists")

    cc_priority = to_cc_priority(task.priority)
    cc_status = to_cc_status(task.status)
    metadata = build_cc_metadata(
        task_id=task.task_id,
        evidence=task.evidence,
    )

    sql = """
        INSERT INTO cc_tasks
            (title, description, status, priority, assigned_to,
             blocking_reason, metadata, created_by, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, NOW(), NOW())
        RETURNING id, title, status, priority, metadata, created_at
    """
    row = await pool.fetchrow(
        sql,
        task.title,
        task.description,
        cc_status,
        cc_priority,
        task.owner,
        task.blocked_by,
        metadata,
        "taskmate_api",
    )

    tm_status = to_tm_status(row["status"])
    meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
    return {
        "id": row["id"],
        "task_id": meta.get("task_id", str(row["id"])),
        "status": tm_status,
    }


@router.patch("/tasks/{task_id}")
async def update_task(
    task_id: str,
    update: TaskUpdate,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()

    sets = ["updated_at = NOW()"]
    params: list = []
    idx = 1

    if update.title is not None:
        sets.append(f"title = ${idx}")
        params.append(update.title)
        idx += 1

    if update.description is not None:
        sets.append(f"description = ${idx}")
        params.append(update.description)
        idx += 1

    if update.priority is not None:
        sets.append(f"priority = ${idx}")
        params.append(to_cc_priority(update.priority))
        idx += 1

    if update.status is not None:
        cc_status = to_cc_status(update.status)
        sets.append(f"status = ${idx}")
        params.append(cc_status)
        idx += 1
        if update.status == "closed":
            sets.append(f"completed_date = ${idx}")
            params.append(datetime.now(timezone.utc))
            idx += 1

    if update.owner is not None:
        sets.append(f"assigned_to = ${idx}")
        params.append(update.owner)
        idx += 1

    if update.blocked_by is not None:
        sets.append(f"blocking_reason = ${idx}")
        params.append(update.blocked_by)
        idx += 1

    if update.evidence is not None:
        sets.append(
            f"metadata = jsonb_set(COALESCE(metadata, '{{}}'::jsonb), '{{evidence}}', ${idx}::jsonb)"
        )
        params.append(json.dumps(update.evidence))
        idx += 1

    if len(sets) <= 1:
        raise HTTPException(400, "No fields to update")

    params.append(task_id)
    where = _task_id_where(idx)

    sql = f"""
        UPDATE cc_tasks
        SET {', '.join(sets)}
        WHERE {where} AND deleted_at IS NULL
        RETURNING id, title, status, priority, metadata, updated_at
    """

    row = await pool.fetchrow(sql, *params)
    if not row:
        raise HTTPException(404, f"Task {task_id} not found")

    result = cc_row_to_taskmate(dict(row))
    return {
        "id": result["id"],
        "task_id": result["task_id"],
        "status": result["status"],
        "updated_at": row["updated_at"],
    }


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()

    # Soft delete (cc_tasks convention)
    where = _task_id_where(1)
    row = await pool.fetchrow(
        f"""
        UPDATE cc_tasks
        SET deleted_at = NOW(), updated_at = NOW()
        WHERE {where} AND deleted_at IS NULL
        RETURNING id, metadata
        """,
        task_id,
    )
    if not row:
        raise HTTPException(404, f"Task {task_id} not found")

    meta = (
        row["metadata"]
        if isinstance(row["metadata"], dict)
        else json.loads(row["metadata"] or "{}")
    )
    resolved_task_id = meta.get("task_id") or meta.get("source_id") or str(row["id"])
    return {"deleted": True, "id": row["id"], "task_id": resolved_task_id}


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()

    where = _task_id_where(1)
    task = await pool.fetchrow(
        f"""
        SELECT id, title, description, status, priority,
               assigned_to, blocking_reason, completed_date,
               metadata, created_at, updated_at
        FROM cc_tasks
        WHERE {where} AND deleted_at IS NULL
        """,
        task_id,
    )
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")

    result = cc_row_to_taskmate(dict(task))

    # Fetch comments (still from taskmate_comments, linked by task_id)
    comments = await pool.fetch(
        "SELECT id, author, body, created_at FROM taskmate_comments WHERE task_id = $1 ORDER BY created_at",
        task_id,
    )

    return {
        "task": result,
        "comments": [dict(c) for c in comments],
    }


@router.post("/tasks/{task_id}/comments", status_code=201)
async def add_comment(
    task_id: str,
    comment: CommentCreate,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()
    tenant_id = DEFAULT_TENANT_ID

    # Verify task exists in cc_tasks
    where = _task_id_where(1)
    exists = await pool.fetchval(
        f"SELECT 1 FROM cc_tasks WHERE {where} AND deleted_at IS NULL",
        task_id,
    )
    if not exists:
        raise HTTPException(404, f"Task {task_id} not found")

    row = await pool.fetchrow(
        """INSERT INTO taskmate_comments (task_id, author, body, tenant_id)
           VALUES ($1, $2, $3, $4::uuid) RETURNING id, created_at""",
        task_id,
        comment.author,
        comment.body,
        tenant_id,
    )

    return {"id": row["id"], "task_id": task_id, "created_at": row["created_at"]}


@router.get("/summary")
async def task_summary(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()

    rows = await pool.fetch(
        """SELECT status, priority, count(*) as cnt
           FROM cc_tasks WHERE deleted_at IS NULL
           GROUP BY status, priority
           ORDER BY status, priority""",
    )

    by_status: dict = {}
    by_priority: dict = {}
    total = 0
    for r in rows:
        cnt = r["cnt"]
        total += cnt
        # Report in taskmate terms
        tm_status = to_tm_status(r["status"])
        tm_priority = to_tm_priority(r["priority"])
        by_status[tm_status] = by_status.get(tm_status, 0) + cnt
        by_priority[tm_priority] = by_priority.get(tm_priority, 0) + cnt

    return {
        "total": total,
        "by_status": by_status,
        "by_priority": by_priority,
        "details": [
            {
                "status": to_tm_status(r["status"]),
                "priority": to_tm_priority(r["priority"]),
                "cnt": r["cnt"],
            }
            for r in rows
        ],
    }

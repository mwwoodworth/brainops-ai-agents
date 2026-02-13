"""
TaskMate — Cross-Model Task Manager API
P1-TASKMATE-001 | Created 2026-02-13

Endpoints:
  GET    /taskmate/tasks           List tasks (filter: status, priority, owner)
  POST   /taskmate/tasks           Create task
  PATCH  /taskmate/tasks/{task_id} Update task
  GET    /taskmate/tasks/{task_id} Get task with comments
  POST   /taskmate/tasks/{task_id}/comments  Add comment
  GET    /taskmate/summary         Dashboard counts
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel

from database.async_connection import get_pool, using_fallback
from database.verify_tables import verify_tables_async

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/taskmate", tags=["TaskMate"])

REQUIRED_TABLES = ["taskmate_tasks", "taskmate_comments"]

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


async def _get_tenant_id(x_tenant_id: Optional[str]) -> str:
    return x_tenant_id or DEFAULT_TENANT_ID


async def _check_tables():
    if using_fallback():
        raise HTTPException(503, "Database unavailable")
    pool = get_pool()
    ok = await verify_tables_async(REQUIRED_TABLES, pool, module_name="taskmate")
    if not ok:
        raise HTTPException(503, "TaskMate tables not created. Run migration.")
    return pool


# --- Endpoints ---


@router.get("/tasks")
async def list_tasks(
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    owner: Optional[str] = Query(None),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()
    tenant_id = await _get_tenant_id(x_tenant_id)

    conditions = ["tenant_id = $1"]
    params = [tenant_id]
    idx = 2

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1
    if priority:
        conditions.append(f"priority = ${idx}")
        params.append(priority)
        idx += 1
    if owner:
        conditions.append(f"owner = ${idx}")
        params.append(owner)
        idx += 1

    where = " AND ".join(conditions)
    sql = f"""
        SELECT id, task_id, title, priority, status, owner, blocked_by,
               evidence, created_at, updated_at, closed_at
        FROM taskmate_tasks
        WHERE {where}
        ORDER BY
            CASE priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 ELSE 3 END,
            created_at DESC
        LIMIT 100
    """

    rows = await pool.fetch(sql, *params)
    return {
        "tasks": [dict(r) for r in rows],
        "count": len(rows),
    }


@router.post("/tasks", status_code=201)
async def create_task(
    task: TaskCreate,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()
    tenant_id = await _get_tenant_id(x_tenant_id)

    sql = """
        INSERT INTO taskmate_tasks
            (task_id, title, description, priority, status, owner, blocked_by, evidence, tenant_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::uuid)
        ON CONFLICT (task_id) DO NOTHING
        RETURNING id, task_id, status
    """
    row = await pool.fetchrow(
        sql,
        task.task_id,
        task.title,
        task.description,
        task.priority,
        task.status,
        task.owner,
        task.blocked_by,
        task.evidence,
        tenant_id,
    )
    if not row:
        raise HTTPException(409, f"Task {task.task_id} already exists")

    return {"id": row["id"], "task_id": row["task_id"], "status": row["status"]}


@router.patch("/tasks/{task_id}")
async def update_task(
    task_id: str,
    update: TaskUpdate,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()
    tenant_id = await _get_tenant_id(x_tenant_id)

    sets = ["updated_at = NOW()"]
    params = []
    idx = 1

    for field in ["title", "description", "priority", "status", "owner", "blocked_by", "evidence"]:
        val = getattr(update, field, None)
        if val is not None:
            sets.append(f"{field} = ${idx}")
            params.append(val)
            idx += 1

    if update.status == "closed":
        sets.append(f"closed_at = ${idx}")
        params.append(datetime.now(timezone.utc))
        idx += 1

    if len(sets) <= 1:
        raise HTTPException(400, "No fields to update")

    params.append(task_id)
    params.append(tenant_id)

    sql = f"""
        UPDATE taskmate_tasks
        SET {', '.join(sets)}
        WHERE task_id = ${idx} AND tenant_id = ${idx + 1}::uuid
        RETURNING id, task_id, status, updated_at
    """

    row = await pool.fetchrow(sql, *params)
    if not row:
        raise HTTPException(404, f"Task {task_id} not found")

    return dict(row)


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()
    tenant_id = await _get_tenant_id(x_tenant_id)

    task = await pool.fetchrow(
        "SELECT * FROM taskmate_tasks WHERE task_id = $1 AND tenant_id = $2::uuid",
        task_id,
        tenant_id,
    )
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")

    comments = await pool.fetch(
        "SELECT id, author, body, created_at FROM taskmate_comments WHERE task_id = $1 AND tenant_id = $2::uuid ORDER BY created_at",
        task_id,
        tenant_id,
    )

    return {
        "task": dict(task),
        "comments": [dict(c) for c in comments],
    }


@router.post("/tasks/{task_id}/comments", status_code=201)
async def add_comment(
    task_id: str,
    comment: CommentCreate,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    pool = await _check_tables()
    tenant_id = await _get_tenant_id(x_tenant_id)

    # Verify task exists
    exists = await pool.fetchval(
        "SELECT 1 FROM taskmate_tasks WHERE task_id = $1 AND tenant_id = $2::uuid",
        task_id,
        tenant_id,
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
    tenant_id = await _get_tenant_id(x_tenant_id)

    rows = await pool.fetch(
        """SELECT status, priority, count(*) as cnt
           FROM taskmate_tasks WHERE tenant_id = $1::uuid
           GROUP BY status, priority
           ORDER BY status, priority""",
        tenant_id,
    )

    by_status = {}
    by_priority = {}
    total = 0
    for r in rows:
        cnt = r["cnt"]
        total += cnt
        by_status[r["status"]] = by_status.get(r["status"], 0) + cnt
        by_priority[r["priority"]] = by_priority.get(r["priority"], 0) + cnt

    return {
        "total": total,
        "by_status": by_status,
        "by_priority": by_priority,
        "details": [dict(r) for r in rows],
    }

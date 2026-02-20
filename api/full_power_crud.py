"""
Full-Power CRUD API
===================
Unified CRUD and control-plane endpoints for:
- Leads
- Campaigns
- Tasks
- Agent Executions
- Memories
- Workflows
- Alerts
- Brain Logs

This module intentionally centralizes high-coverage operational endpoints with:
- Strong request/response validation
- Cursor-based pagination for large lists
- Structured error responses
- Light in-process rate limiting for write-heavy endpoints
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from campaign_manager import CAMPAIGNS, campaign_to_dict, get_campaign
from database.async_connection import DatabaseUnavailableError, get_pool, get_tenant_pool
from utils.embedding_provider import generate_embedding_async
from api.task_adapter import (
    cc_row_to_taskmate,
    to_cc_priority,
    to_cc_status,
    build_cc_metadata,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Full Power API"])

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457")
ENABLE_FULL_POWER_API = (os.getenv("ENABLE_FULL_POWER_API", "true") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _cursor_encode(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, default=str).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def _cursor_decode(cursor: Optional[str]) -> Optional[dict[str, Any]]:
    if not cursor:
        return None
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("utf-8"))
        decoded = json.loads(raw.decode("utf-8"))
        return decoded if isinstance(decoded, dict) else None
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid cursor")


class APIError(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=_iso_now)


class CursorPageInfo(BaseModel):
    next_cursor: Optional[str] = None
    has_more: bool = False
    returned: int


# ---------------------------------------------------------------------------
# In-process write rate limiter (lightweight, non-distributed)
# ---------------------------------------------------------------------------

_RATE_BUCKETS: dict[str, list[float]] = {}


def write_rate_limit(limit: int = 120, window_seconds: int = 60):
    async def _check(request: Request):
        identity = (
            request.headers.get("X-API-Key")
            or request.headers.get("X-Tenant-ID")
            or (request.client.host if request.client else "unknown")
        )
        key = f"write:{identity}"
        now = time.monotonic()
        bucket = _RATE_BUCKETS.setdefault(key, [])
        bucket[:] = [ts for ts in bucket if now - ts < window_seconds]
        if len(bucket) >= limit:
            raise HTTPException(status_code=429, detail="Write rate limit exceeded")
        bucket.append(now)

    return _check


WRITE_LIMIT = write_rate_limit()


async def require_feature_enabled() -> None:
    if not ENABLE_FULL_POWER_API:
        raise HTTPException(status_code=404, detail="Full Power API disabled")


async def get_tenant_id(x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")) -> str:
    return x_tenant_id or DEFAULT_TENANT_ID


def _ensure_uuid(value: str, field_name: str = "id") -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {field_name}") from exc


# ---------------------------------------------------------------------------
# Leads CRUD
# ---------------------------------------------------------------------------


class LeadCreate(BaseModel):
    company_name: str
    contact_name: Optional[str] = None
    email: str
    phone: Optional[str] = None
    website: Optional[str] = None
    stage: str = "new"
    source: str = "api_v2"
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    value_estimate: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LeadUpdate(BaseModel):
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    stage: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    value_estimate: Optional[float] = Field(default=None, ge=0.0)
    metadata: Optional[dict[str, Any]] = None


class LeadResponse(BaseModel):
    id: str
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    stage: Optional[str] = None
    score: Optional[float] = None
    value_estimate: Optional[float] = None
    source: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class LeadBulkUpdateItem(BaseModel):
    id: str
    patch: LeadUpdate


class LeadBulkRequest(BaseModel):
    create: list[LeadCreate] = Field(default_factory=list)
    update: list[LeadBulkUpdateItem] = Field(default_factory=list)
    archive_ids: list[str] = Field(default_factory=list)


class LeadBulkResult(BaseModel):
    created: int = 0
    updated: int = 0
    archived: int = 0
    errors: list[str] = Field(default_factory=list)


@router.post(
    "/leads",
    response_model=LeadResponse,
    tags=["Leads"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def create_lead_v2(
    payload: LeadCreate, tenant_id: str = Depends(get_tenant_id)
) -> LeadResponse:
    """Create a lead (with tenant marker stored in metadata)."""
    pool = get_pool()
    lead_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    metadata = dict(payload.metadata)
    metadata.setdefault("tenant_id", tenant_id)

    try:
        row = await pool.fetchrow(
            """
            INSERT INTO revenue_leads (
                id, company_name, contact_name, email, phone, website,
                stage, score, value_estimate, source, metadata, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8, $9, $10, $11::jsonb, $12, $12
            )
            RETURNING *
            """,
            lead_id,
            payload.company_name,
            payload.contact_name,
            payload.email,
            payload.phone,
            payload.website,
            payload.stage,
            payload.score,
            payload.value_estimate,
            payload.source,
            json.dumps(metadata),
            now,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lead create failed: {exc}") from exc

    return _lead_row_to_response(row)


@router.get(
    "/leads/{lead_id}",
    response_model=LeadResponse,
    tags=["Leads"],
    dependencies=[Depends(require_feature_enabled)],
)
async def get_lead_v2(lead_id: str, tenant_id: str = Depends(get_tenant_id)) -> LeadResponse:
    """Get one lead by id."""
    pool = get_pool()
    lead_uuid = _ensure_uuid(lead_id, "lead_id")

    row = await pool.fetchrow("SELECT * FROM revenue_leads WHERE id = $1", lead_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Lead not found")

    metadata = _safe_json(row.get("metadata"))
    if metadata.get("tenant_id") not in (None, tenant_id):
        raise HTTPException(status_code=404, detail="Lead not found")

    return _lead_row_to_response(row)


@router.get(
    "/leads",
    tags=["Leads"],
    dependencies=[Depends(require_feature_enabled)],
)
async def list_leads_v2(
    tenant_id: str = Depends(get_tenant_id),
    stage: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    min_score: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    query: Optional[str] = Query(default=None, description="company/contact/email contains"),
    include_archived: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
    sort_by: Literal["created_at", "updated_at"] = Query(default="created_at"),
    sort_order: Literal["asc", "desc"] = Query(default="desc"),
) -> dict[str, Any]:
    """List leads with filtering, sorting, and cursor pagination."""
    pool = get_pool()
    where: list[str] = ["1=1"]
    params: list[Any] = []

    if stage:
        params.append(stage)
        where.append(f"stage = ${len(params)}")

    if source:
        params.append(source)
        where.append(f"source = ${len(params)}")

    if min_score is not None:
        params.append(min_score)
        where.append(f"score >= ${len(params)}")

    if query:
        params.append(f"%{query}%")
        where.append(
            f"(company_name ILIKE ${len(params)} OR contact_name ILIKE ${len(params)} OR email ILIKE ${len(params)})"
        )

    if not include_archived:
        where.append("COALESCE((metadata->>'archived')::boolean, false) = false")

    cursor_data = _cursor_decode(cursor)
    if cursor_data:
        cursor_ts_raw = cursor_data.get("ts")
        cursor_id_raw = cursor_data.get("id")
        if cursor_ts_raw and cursor_id_raw:
            cursor_ts = datetime.fromisoformat(cursor_ts_raw)
            cursor_id = _ensure_uuid(cursor_id_raw, "cursor.id")
            params.extend([cursor_ts, cursor_id])
            comparator = "<" if sort_order == "desc" else ">"
            where.append(f"({sort_by}, id) {comparator} (${len(params)-1}, ${len(params)})")

    where_sql = " AND ".join(where)
    order_sql = f"ORDER BY {sort_by} {sort_order.upper()}, id {sort_order.upper()}"
    params.append(limit + 1)

    rows = await pool.fetch(
        f"""
        SELECT *
        FROM revenue_leads
        WHERE {where_sql}
        {order_sql}
        LIMIT ${len(params)}
        """,
        *params,
    )

    filtered_rows: list[Any] = []
    for row in rows:
        metadata = _safe_json(row.get("metadata"))
        if metadata.get("tenant_id") not in (None, tenant_id):
            continue
        filtered_rows.append(row)

    has_more = len(filtered_rows) > limit
    page_rows = filtered_rows[:limit]
    next_cursor = None
    if has_more and page_rows:
        tail = page_rows[-1]
        next_cursor = _cursor_encode(
            {
                "ts": _row_datetime_iso(tail.get(sort_by)),
                "id": str(tail["id"]),
            }
        )

    return {
        "items": [_lead_row_to_response(r).model_dump() for r in page_rows],
        "page": CursorPageInfo(
            next_cursor=next_cursor, has_more=has_more, returned=len(page_rows)
        ).model_dump(),
    }


@router.patch(
    "/leads/{lead_id}",
    response_model=LeadResponse,
    tags=["Leads"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def update_lead_v2(
    lead_id: str,
    patch: LeadUpdate,
    tenant_id: str = Depends(get_tenant_id),
) -> LeadResponse:
    """Update mutable lead fields."""
    pool = get_pool()
    lead_uuid = _ensure_uuid(lead_id, "lead_id")

    current = await pool.fetchrow("SELECT * FROM revenue_leads WHERE id = $1", lead_uuid)
    if not current:
        raise HTTPException(status_code=404, detail="Lead not found")

    current_meta = _safe_json(current.get("metadata"))
    if current_meta.get("tenant_id") not in (None, tenant_id):
        raise HTTPException(status_code=404, detail="Lead not found")

    updates = _model_dump(patch)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    sets: list[str] = []
    args: list[Any] = []
    for key in [
        "company_name",
        "contact_name",
        "email",
        "phone",
        "website",
        "stage",
        "score",
        "value_estimate",
    ]:
        if key in updates:
            args.append(updates[key])
            sets.append(f"{key} = ${len(args)}")

    if "metadata" in updates and updates["metadata"] is not None:
        merged_meta = {**current_meta, **updates["metadata"], "tenant_id": tenant_id}
        args.append(json.dumps(merged_meta))
        sets.append(f"metadata = ${len(args)}::jsonb")

    args.append(datetime.now(timezone.utc))
    sets.append(f"updated_at = ${len(args)}")

    args.append(lead_uuid)
    row = await pool.fetchrow(
        f"""
        UPDATE revenue_leads
        SET {', '.join(sets)}
        WHERE id = ${len(args)}
        RETURNING *
        """,
        *args,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Lead not found")
    return _lead_row_to_response(row)


@router.delete(
    "/leads/{lead_id}",
    tags=["Leads"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def archive_lead_v2(lead_id: str, tenant_id: str = Depends(get_tenant_id)) -> dict[str, Any]:
    """Soft-delete lead (archive flag in metadata)."""
    pool = get_pool()
    lead_uuid = _ensure_uuid(lead_id, "lead_id")
    row = await pool.fetchrow("SELECT metadata FROM revenue_leads WHERE id = $1", lead_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Lead not found")

    metadata = _safe_json(row.get("metadata"))
    if metadata.get("tenant_id") not in (None, tenant_id):
        raise HTTPException(status_code=404, detail="Lead not found")

    metadata.update(
        {
            "archived": True,
            "archived_at": _iso_now(),
            "tenant_id": tenant_id,
        }
    )
    await pool.execute(
        """
        UPDATE revenue_leads
        SET metadata = $1::jsonb,
            stage = 'lost',
            updated_at = $2
        WHERE id = $3
        """,
        json.dumps(metadata),
        datetime.now(timezone.utc),
        lead_uuid,
    )
    return {"success": True, "id": lead_id, "archived": True}


@router.post(
    "/leads/bulk",
    response_model=LeadBulkResult,
    tags=["Leads"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def bulk_leads_v2(
    payload: LeadBulkRequest, tenant_id: str = Depends(get_tenant_id)
) -> LeadBulkResult:
    """Bulk create/update/archive leads."""
    result = LeadBulkResult()

    for item in payload.create:
        try:
            await create_lead_v2(item, tenant_id=tenant_id)
            result.created += 1
        except Exception as exc:
            result.errors.append(f"create:{item.email}:{exc}")

    for item in payload.update:
        try:
            await update_lead_v2(item.id, item.patch, tenant_id=tenant_id)
            result.updated += 1
        except Exception as exc:
            result.errors.append(f"update:{item.id}:{exc}")

    for lead_id in payload.archive_ids:
        try:
            await archive_lead_v2(lead_id, tenant_id=tenant_id)
            result.archived += 1
        except Exception as exc:
            result.errors.append(f"archive:{lead_id}:{exc}")

    return result


# ---------------------------------------------------------------------------
# Campaign CRUD (in-memory dynamic layer + built-in campaign templates)
# ---------------------------------------------------------------------------


class CampaignTemplateIn(BaseModel):
    step: int = Field(ge=1)
    delay_days: int = Field(ge=0)
    subject: str
    body_html: str


class CampaignCreate(BaseModel):
    id: Optional[str] = None
    name: str
    campaign_type: str = "custom"
    brand: str = "BrainOps"
    target_audience: str = "general"
    is_active: bool = True
    daily_outreach_limit: int = Field(default=100, ge=1, le=5000)
    templates: list[CampaignTemplateIn] = Field(default_factory=list)


class CampaignPatch(BaseModel):
    name: Optional[str] = None
    campaign_type: Optional[str] = None
    brand: Optional[str] = None
    target_audience: Optional[str] = None
    is_active: Optional[bool] = None
    daily_outreach_limit: Optional[int] = Field(default=None, ge=1, le=5000)


_CUSTOM_CAMPAIGNS: dict[str, dict[str, Any]] = {}


@router.get(
    "/campaigns",
    tags=["Campaigns"],
    dependencies=[Depends(require_feature_enabled)],
)
async def list_campaigns_v2(active_only: bool = True) -> dict[str, Any]:
    """List built-in and dynamic campaigns."""
    items = [campaign_to_dict(c) for c in CAMPAIGNS.values() if (c.is_active or not active_only)]
    for campaign in _CUSTOM_CAMPAIGNS.values():
        if active_only and not campaign.get("is_active", True):
            continue
        items.append(dict(campaign))
    return {"items": items, "total": len(items)}


@router.post(
    "/campaigns",
    tags=["Campaigns"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def create_campaign_v2(payload: CampaignCreate) -> dict[str, Any]:
    """Create a dynamic campaign configuration."""
    campaign_id = payload.id or f"custom_{uuid.uuid4().hex[:12]}"
    if campaign_id in CAMPAIGNS or campaign_id in _CUSTOM_CAMPAIGNS:
        raise HTTPException(status_code=409, detail="Campaign id already exists")

    item = {
        "id": campaign_id,
        "name": payload.name,
        "campaign_type": payload.campaign_type,
        "brand": payload.brand,
        "target_audience": payload.target_audience,
        "is_active": payload.is_active,
        "daily_outreach_limit": payload.daily_outreach_limit,
        "template_count": len(payload.templates),
        "templates": [t.model_dump() for t in payload.templates],
        "created_at": _iso_now(),
        "updated_at": _iso_now(),
    }
    _CUSTOM_CAMPAIGNS[campaign_id] = item
    return item


@router.get(
    "/campaigns/{campaign_id}",
    tags=["Campaigns"],
    dependencies=[Depends(require_feature_enabled)],
)
async def get_campaign_v2(campaign_id: str) -> dict[str, Any]:
    """Get campaign details."""
    builtin = get_campaign(campaign_id)
    if builtin:
        data = campaign_to_dict(builtin)
        data["source"] = "builtin"
        return data
    custom = _CUSTOM_CAMPAIGNS.get(campaign_id)
    if custom:
        data = dict(custom)
        data["source"] = "dynamic"
        return data
    raise HTTPException(status_code=404, detail="Campaign not found")


@router.patch(
    "/campaigns/{campaign_id}",
    tags=["Campaigns"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def update_campaign_v2(campaign_id: str, patch: CampaignPatch) -> dict[str, Any]:
    """Update campaign metadata."""
    updates = _model_dump(patch)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    if campaign_id in _CUSTOM_CAMPAIGNS:
        _CUSTOM_CAMPAIGNS[campaign_id].update(updates)
        _CUSTOM_CAMPAIGNS[campaign_id]["updated_at"] = _iso_now()
        return dict(_CUSTOM_CAMPAIGNS[campaign_id])

    builtin = get_campaign(campaign_id)
    if not builtin:
        raise HTTPException(status_code=404, detail="Campaign not found")

    for key, value in updates.items():
        if hasattr(builtin, key):
            setattr(builtin, key, value)

    return campaign_to_dict(builtin)


@router.post(
    "/campaigns/{campaign_id}/pause",
    tags=["Campaigns"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def pause_campaign_v2(campaign_id: str) -> dict[str, Any]:
    """Pause campaign."""
    return await _set_campaign_active(campaign_id, active=False)


@router.post(
    "/campaigns/{campaign_id}/resume",
    tags=["Campaigns"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def resume_campaign_v2(campaign_id: str) -> dict[str, Any]:
    """Resume campaign."""
    return await _set_campaign_active(campaign_id, active=True)


@router.post(
    "/campaigns/{campaign_id}/archive",
    tags=["Campaigns"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def archive_campaign_v2(campaign_id: str) -> dict[str, Any]:
    """Archive campaign configuration."""
    if campaign_id in _CUSTOM_CAMPAIGNS:
        _CUSTOM_CAMPAIGNS[campaign_id]["is_active"] = False
        _CUSTOM_CAMPAIGNS[campaign_id]["archived"] = True
        _CUSTOM_CAMPAIGNS[campaign_id]["updated_at"] = _iso_now()
        return {"success": True, "campaign_id": campaign_id, "archived": True}

    builtin = get_campaign(campaign_id)
    if not builtin:
        raise HTTPException(status_code=404, detail="Campaign not found")
    builtin.is_active = False
    return {"success": True, "campaign_id": campaign_id, "archived": True}


@router.get(
    "/campaigns/{campaign_id}/analytics",
    tags=["Campaigns"],
    dependencies=[Depends(require_feature_enabled)],
)
async def campaign_analytics_v2(campaign_id: str) -> dict[str, Any]:
    """Campaign analytics summary."""
    from prospect_discovery import get_discovery_engine

    if campaign_id not in CAMPAIGNS and campaign_id not in _CUSTOM_CAMPAIGNS:
        raise HTTPException(status_code=404, detail="Campaign not found")

    try:
        engine = get_discovery_engine()
        stats = await engine.get_campaign_stats(campaign_id)
        return {"campaign_id": campaign_id, "analytics": stats, "timestamp": _iso_now()}
    except Exception:
        return {
            "campaign_id": campaign_id,
            "analytics": {
                "total_leads": 0,
                "contacted": 0,
                "replied": 0,
                "conversion_rate": 0,
            },
            "timestamp": _iso_now(),
        }


async def _set_campaign_active(campaign_id: str, active: bool) -> dict[str, Any]:
    if campaign_id in _CUSTOM_CAMPAIGNS:
        _CUSTOM_CAMPAIGNS[campaign_id]["is_active"] = active
        _CUSTOM_CAMPAIGNS[campaign_id]["updated_at"] = _iso_now()
        return {"success": True, "campaign_id": campaign_id, "is_active": active}

    builtin = get_campaign(campaign_id)
    if not builtin:
        raise HTTPException(status_code=404, detail="Campaign not found")
    builtin.is_active = active
    return {"success": True, "campaign_id": campaign_id, "is_active": active}


# ---------------------------------------------------------------------------
# Tasks CRUD + assignment/completion/cancel/dependencies
# ---------------------------------------------------------------------------


class TaskCreateV2(BaseModel):
    task_id: str
    title: str
    description: Optional[str] = None
    priority: str = "P2"
    status: str = "open"
    owner: Optional[str] = None
    blocked_by: Optional[str] = None
    evidence: Optional[str] = None


class TaskPatchV2(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    owner: Optional[str] = None
    blocked_by: Optional[str] = None
    evidence: Optional[str] = None


class TaskAssignRequest(BaseModel):
    owner: str


class TaskDependencyRequest(BaseModel):
    blocked_by: list[str] = Field(default_factory=list)


@router.post(
    "/tasks",
    tags=["Tasks"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def create_task_v2(
    payload: TaskCreateV2, tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Create task in cc_tasks (canonical store)."""
    pool = get_pool()
    existing = await pool.fetchval(
        "SELECT 1 FROM cc_tasks WHERE metadata->>'task_id' = $1 AND deleted_at IS NULL",
        payload.task_id,
    )
    if existing:
        raise HTTPException(status_code=409, detail="Task already exists")

    metadata = build_cc_metadata(task_id=payload.task_id, evidence=payload.evidence)
    row = await pool.fetchrow(
        """
        INSERT INTO cc_tasks
            (title, description, status, priority, assigned_to,
             blocking_reason, metadata, created_by, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, NOW(), NOW())
        RETURNING *
        """,
        payload.title,
        payload.description,
        to_cc_status(payload.status),
        to_cc_priority(payload.priority),
        payload.owner,
        payload.blocked_by,
        metadata,
        "v2_api",
    )
    return cc_row_to_taskmate(dict(row))


def _cc_task_id_where(param_idx: int) -> str:
    """WHERE clause to find task by taskmate task_id in cc_tasks."""
    return (
        f"(metadata->>'task_id' = ${param_idx} "
        f"OR metadata->>'source_id' = ${param_idx} "
        f"OR id::text = ${param_idx})"
    )


@router.get(
    "/tasks",
    tags=["Tasks"],
    dependencies=[Depends(require_feature_enabled)],
)
async def list_tasks_v2(
    tenant_id: str = Depends(get_tenant_id),
    status: Optional[str] = None,
    owner: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    """List tasks from cc_tasks with cursor pagination."""
    pool = get_pool()
    where = ["deleted_at IS NULL"]
    params: list[Any] = []

    if status:
        params.append(to_cc_status(status))
        where.append(f"status = ${len(params)}")
    if owner:
        params.append(owner)
        where.append(f"assigned_to = ${len(params)}")
    if priority:
        params.append(to_cc_priority(priority))
        where.append(f"priority = ${len(params)}")

    cursor_data = _cursor_decode(cursor)
    if cursor_data and cursor_data.get("created_at") and cursor_data.get("id"):
        params.append(datetime.fromisoformat(cursor_data["created_at"]))
        params.append(cursor_data["id"])
        where.append(f"(created_at, id::text) < (${len(params)-1}, ${len(params)})")

    params.append(limit + 1)
    rows = await pool.fetch(
        f"""
        SELECT *
        FROM cc_tasks
        WHERE {' AND '.join(where)}
        ORDER BY created_at DESC, id DESC
        LIMIT ${len(params)}
        """,
        *params,
    )

    has_more = len(rows) > limit
    page_rows = rows[:limit]
    next_cursor = None
    if has_more and page_rows:
        tail = dict(page_rows[-1])
        next_cursor = _cursor_encode(
            {
                "created_at": _row_datetime_iso(tail.get("created_at")),
                "id": str(tail.get("id", "")),
            }
        )

    return {
        "items": [cc_row_to_taskmate(dict(r)) for r in page_rows],
        "page": CursorPageInfo(
            next_cursor=next_cursor, has_more=has_more, returned=len(page_rows)
        ).model_dump(),
    }


@router.get(
    "/tasks/{task_id}",
    tags=["Tasks"],
    dependencies=[Depends(require_feature_enabled)],
)
async def get_task_v2(task_id: str, tenant_id: str = Depends(get_tenant_id)) -> dict[str, Any]:
    """Read one task and comments from cc_tasks."""
    pool = get_pool()
    where = _cc_task_id_where(1)
    task = await pool.fetchrow(
        f"SELECT * FROM cc_tasks WHERE {where} AND deleted_at IS NULL",
        task_id,
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    comments = await pool.fetch(
        "SELECT id, author, body, created_at FROM taskmate_comments WHERE task_id = $1 ORDER BY created_at",
        task_id,
    )
    return {"task": cc_row_to_taskmate(dict(task)), "comments": [dict(c) for c in comments]}


@router.patch(
    "/tasks/{task_id}",
    tags=["Tasks"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def update_task_v2(
    task_id: str,
    patch: TaskPatchV2,
    tenant_id: str = Depends(get_tenant_id),
) -> dict[str, Any]:
    """Update task fields in cc_tasks."""
    pool = get_pool()
    updates = _model_dump(patch)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Map taskmate field names → cc_tasks column names
    _field_map = {
        "title": "title",
        "description": "description",
        "priority": "priority",
        "status": "status",
        "owner": "assigned_to",
        "blocked_by": "blocking_reason",
    }

    sets: list[str] = ["updated_at = NOW()"]
    args: list[Any] = []
    for key, value in updates.items():
        if key == "evidence":
            args.append(json.dumps(value))
            sets.append(
                f"metadata = jsonb_set(COALESCE(metadata, '{{}}'::jsonb), '{{evidence}}', ${len(args)}::jsonb)"
            )
        elif key == "priority":
            args.append(to_cc_priority(value))
            sets.append(f"priority = ${len(args)}")
        elif key == "status":
            cc_status = to_cc_status(value)
            args.append(cc_status)
            sets.append(f"status = ${len(args)}")
            if value == "closed":
                args.append(datetime.now(timezone.utc))
                sets.append(f"completed_date = ${len(args)}")
        elif key in _field_map:
            args.append(value)
            sets.append(f"{_field_map[key]} = ${len(args)}")

    args.append(task_id)
    where = _cc_task_id_where(len(args))
    row = await pool.fetchrow(
        f"""
        UPDATE cc_tasks
        SET {', '.join(sets)}
        WHERE {where} AND deleted_at IS NULL
        RETURNING *
        """,
        *args,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return cc_row_to_taskmate(dict(row))


@router.post(
    "/tasks/{task_id}/assign",
    tags=["Tasks"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def assign_task_v2(
    task_id: str,
    request: TaskAssignRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> dict[str, Any]:
    """Assign task owner."""
    return await update_task_v2(task_id, TaskPatchV2(owner=request.owner), tenant_id)


@router.post(
    "/tasks/{task_id}/complete",
    tags=["Tasks"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def complete_task_v2(task_id: str, tenant_id: str = Depends(get_tenant_id)) -> dict[str, Any]:
    """Complete task."""
    return await update_task_v2(task_id, TaskPatchV2(status="closed"), tenant_id)


@router.post(
    "/tasks/{task_id}/cancel",
    tags=["Tasks"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def cancel_task_v2(task_id: str, tenant_id: str = Depends(get_tenant_id)) -> dict[str, Any]:
    """Cancel task."""
    return await update_task_v2(task_id, TaskPatchV2(status="cancelled"), tenant_id)


@router.post(
    "/tasks/{task_id}/dependencies",
    tags=["Tasks"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def set_task_dependencies_v2(
    task_id: str,
    request: TaskDependencyRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> dict[str, Any]:
    """Set task dependencies (blocked_by)."""
    blocked_by = ",".join(request.blocked_by) if request.blocked_by else None
    return await update_task_v2(task_id, TaskPatchV2(blocked_by=blocked_by), tenant_id)


# ---------------------------------------------------------------------------
# Agent Executions (read/list/retry/cancel)
# ---------------------------------------------------------------------------


@router.get(
    "/agent-executions",
    tags=["Agent Executions"],
    dependencies=[Depends(require_feature_enabled)],
)
async def list_agent_executions_v2(
    status: Optional[str] = None,
    agent_type: Optional[str] = None,
    started_after: Optional[datetime] = None,
    started_before: Optional[datetime] = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List agent executions with filters."""
    pool = get_pool()
    where = ["1=1"]
    params: list[Any] = []

    if status:
        params.append(status)
        where.append(f"status = ${len(params)}")
    if agent_type:
        params.append(agent_type)
        where.append(f"agent_type = ${len(params)}")
    if started_after:
        params.append(started_after)
        where.append(f"created_at >= ${len(params)}")
    if started_before:
        params.append(started_before)
        where.append(f"created_at <= ${len(params)}")

    params.append(limit)
    rows = await pool.fetch(
        f"""
        SELECT id, task_execution_id, agent_type, status, error_message, created_at, completed_at
        FROM agent_executions
        WHERE {' AND '.join(where)}
        ORDER BY created_at DESC
        LIMIT ${len(params)}
        """,
        *params,
    )
    return {"items": [dict(r) for r in rows], "total": len(rows)}


@router.get(
    "/agent-executions/{execution_id}",
    tags=["Agent Executions"],
    dependencies=[Depends(require_feature_enabled)],
)
async def get_agent_execution_v2(execution_id: str) -> dict[str, Any]:
    """Read one execution record."""
    pool = get_pool()
    execution_uuid = _ensure_uuid(execution_id, "execution_id")
    row = await pool.fetchrow("SELECT * FROM agent_executions WHERE id = $1", execution_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Execution not found")
    return dict(row)


@router.post(
    "/agent-executions/{execution_id}/retry",
    tags=["Agent Executions"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def retry_agent_execution_v2(
    execution_id: str, tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Queue execution retry request."""
    pool = get_pool()
    execution_uuid = _ensure_uuid(execution_id, "execution_id")
    row = await pool.fetchrow("SELECT * FROM agent_executions WHERE id = $1", execution_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Execution not found")

    tenant_pool = get_tenant_pool(tenant_id)
    queue_row = await tenant_pool.fetchrow(
        """
        INSERT INTO ai_task_queue (tenant_id, task_type, payload, priority, status, created_at, updated_at)
        VALUES ($1::uuid, 'retry_execution', $2::jsonb, 95, 'pending', NOW(), NOW())
        RETURNING id, status, created_at
        """,
        tenant_id,
        json.dumps(
            {
                "execution_id": execution_id,
                "agent_type": row.get("agent_type"),
                "task_execution_id": str(row.get("task_execution_id") or ""),
            }
        ),
    )
    return {
        "success": True,
        "execution_id": execution_id,
        "queue_id": str(queue_row["id"]),
        "status": queue_row["status"],
    }


@router.post(
    "/agent-executions/{execution_id}/cancel",
    tags=["Agent Executions"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def cancel_agent_execution_v2(execution_id: str) -> dict[str, Any]:
    """Cancel an in-flight execution record."""
    pool = get_pool()
    execution_uuid = _ensure_uuid(execution_id, "execution_id")
    row = await pool.fetchrow(
        """
        UPDATE agent_executions
        SET status = 'cancelled',
            completed_at = COALESCE(completed_at, NOW())
        WHERE id = $1
        RETURNING id, status, completed_at
        """,
        execution_uuid,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Execution not found")
    return {"success": True, "execution_id": str(row["id"]), "status": row["status"]}


# ---------------------------------------------------------------------------
# Memories CRUD
# ---------------------------------------------------------------------------


class MemoryCreateV2(BaseModel):
    memory_type: str = "semantic"
    content: dict[str, Any] | str
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    source_system: str = "full_power_api"
    source_agent: str = "api_v2"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryMetadataPatchV2(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/memories",
    tags=["Memories"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def create_memory_v2(
    payload: MemoryCreateV2, tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Create memory entry."""
    pool = get_tenant_pool(tenant_id)
    content_json = (
        payload.content if isinstance(payload.content, dict) else {"text": payload.content}
    )
    content_text = (
        payload.content if isinstance(payload.content, str) else json.dumps(payload.content)
    )

    embedding = await generate_embedding_async(content_text, log=logger)
    embedding_str = f"[{','.join(map(str, embedding))}]" if embedding else None

    row = await pool.fetchrow(
        """
        INSERT INTO unified_ai_memory (
            tenant_id, memory_type, content, importance_score, tags,
            source_system, source_agent, created_by, metadata, search_text, embedding
        ) VALUES (
            $1::uuid, $2, $3::jsonb, $4, $5,
            $6, $7, $8, $9::jsonb, $10, $11::vector
        )
        RETURNING id::text, created_at
        """,
        tenant_id,
        payload.memory_type,
        json.dumps(content_json),
        payload.importance_score,
        payload.tags,
        payload.source_system,
        payload.source_agent,
        "api_v2",
        json.dumps(payload.metadata),
        content_text,
        embedding_str,
    )
    return {
        "id": row["id"],
        "created_at": _row_datetime_iso(row.get("created_at")),
        "has_embedding": embedding is not None,
    }


@router.get(
    "/memories/{memory_id}",
    tags=["Memories"],
    dependencies=[Depends(require_feature_enabled)],
)
async def get_memory_v2(memory_id: str, tenant_id: str = Depends(get_tenant_id)) -> dict[str, Any]:
    """Read one memory."""
    pool = get_tenant_pool(tenant_id)
    memory_uuid = _ensure_uuid(memory_id, "memory_id")
    row = await pool.fetchrow(
        "SELECT * FROM unified_ai_memory WHERE id = $1 AND tenant_id = $2::uuid",
        memory_uuid,
        tenant_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")
    return dict(row)


@router.get(
    "/memories/search",
    tags=["Memories"],
    dependencies=[Depends(require_feature_enabled)],
)
async def search_memories_v2(
    tenant_id: str = Depends(get_tenant_id),
    q: str = Query(..., min_length=1),
    mode: Literal["vector", "keyword", "hybrid"] = Query(default="hybrid"),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Search memory by vector and/or keyword."""
    pool = get_tenant_pool(tenant_id)

    items: list[dict[str, Any]] = []
    if mode in {"vector", "hybrid"}:
        embedding = await generate_embedding_async(q, log=logger)
        if embedding:
            embedding_str = f"[{','.join(map(str, embedding))}]"
            rows = await pool.fetch(
                """
                SELECT id::text, memory_type, content, metadata, created_at,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM unified_ai_memory
                WHERE tenant_id = $2::uuid
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_str,
                tenant_id,
                limit,
            )
            items.extend(dict(r) for r in rows)

    if mode in {"keyword", "hybrid"}:
        rows = await pool.fetch(
            """
            SELECT id::text, memory_type, content, metadata, created_at, NULL::float AS similarity
            FROM unified_ai_memory
            WHERE tenant_id = $1::uuid
              AND (
                search_text ILIKE $2
                OR content::text ILIKE $2
              )
            ORDER BY created_at DESC
            LIMIT $3
            """,
            tenant_id,
            f"%{q}%",
            limit,
        )
        seen = {item.get("id") for item in items}
        for row in rows:
            data = dict(row)
            if data.get("id") not in seen:
                items.append(data)

    return {"items": items[:limit], "total": len(items[:limit]), "mode": mode}


@router.patch(
    "/memories/{memory_id}/metadata",
    tags=["Memories"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def patch_memory_metadata_v2(
    memory_id: str,
    payload: MemoryMetadataPatchV2,
    tenant_id: str = Depends(get_tenant_id),
) -> dict[str, Any]:
    """Merge memory metadata."""
    pool = get_tenant_pool(tenant_id)
    memory_uuid = _ensure_uuid(memory_id, "memory_id")
    row = await pool.fetchrow(
        """
        UPDATE unified_ai_memory
        SET metadata = COALESCE(metadata, '{}'::jsonb) || $1::jsonb,
            updated_at = NOW()
        WHERE id = $2 AND tenant_id = $3::uuid
        RETURNING id::text, metadata, updated_at
        """,
        json.dumps(payload.metadata),
        memory_uuid,
        tenant_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")
    return dict(row)


@router.post(
    "/memories/{memory_id}/archive",
    tags=["Memories"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def archive_memory_v2(
    memory_id: str, tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Archive memory by setting expires_at and metadata flag."""
    pool = get_tenant_pool(tenant_id)
    memory_uuid = _ensure_uuid(memory_id, "memory_id")
    row = await pool.fetchrow(
        """
        UPDATE unified_ai_memory
        SET expires_at = NOW(),
            metadata = COALESCE(metadata, '{}'::jsonb) || '{"archived": true}'::jsonb,
            updated_at = NOW()
        WHERE id = $1 AND tenant_id = $2::uuid
        RETURNING id::text, expires_at
        """,
        memory_uuid,
        tenant_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        "success": True,
        "id": row["id"],
        "archived_at": _row_datetime_iso(row.get("expires_at")),
    }


# ---------------------------------------------------------------------------
# Workflow CRUD + execute/pause/resume/clone
# ---------------------------------------------------------------------------


class WorkflowCreateV2(BaseModel):
    name: str
    description: Optional[str] = None
    trigger_conditions: dict[str, Any] = Field(default_factory=dict)
    steps: list[str] = Field(default_factory=list)
    decision_points: list[str] = Field(default_factory=list)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    failure_handlers: list[str] = Field(default_factory=list)


class WorkflowPatchV2(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    trigger_conditions: Optional[dict[str, Any]] = None
    steps: Optional[list[str]] = None
    decision_points: Optional[list[str]] = None
    success_criteria: Optional[dict[str, Any]] = None
    failure_handlers: Optional[list[str]] = None
    is_active: Optional[bool] = None


class WorkflowExecuteRequest(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/workflows",
    tags=["Workflows"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def create_workflow_v2(payload: WorkflowCreateV2) -> dict[str, Any]:
    """Create workflow definition."""
    pool = get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO ai_workflows (
            name, description, trigger_conditions, steps,
            decision_points, success_criteria, failure_handlers,
            performance_metrics, is_active, created_at, updated_at
        ) VALUES (
            $1, $2, $3::jsonb, $4::text[],
            $5::text[], $6::jsonb, $7::text[],
            '{}'::jsonb, true, NOW(), NOW()
        )
        RETURNING *
        """,
        payload.name,
        payload.description,
        json.dumps(payload.trigger_conditions),
        payload.steps,
        payload.decision_points,
        json.dumps(payload.success_criteria),
        payload.failure_handlers,
    )
    return dict(row)


@router.get(
    "/workflows",
    tags=["Workflows"],
    dependencies=[Depends(require_feature_enabled)],
)
async def list_workflows_v2(
    active_only: bool = True, limit: int = Query(default=100, ge=1, le=500)
) -> dict[str, Any]:
    """List workflows."""
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT * FROM ai_workflows
        WHERE ($1::boolean = false OR is_active = true)
        ORDER BY updated_at DESC
        LIMIT $2
        """,
        active_only,
        limit,
    )
    return {"items": [dict(r) for r in rows], "total": len(rows)}


@router.get(
    "/workflows/{workflow_id}",
    tags=["Workflows"],
    dependencies=[Depends(require_feature_enabled)],
)
async def get_workflow_v2(workflow_id: str) -> dict[str, Any]:
    """Read workflow definition."""
    pool = get_pool()
    workflow_uuid = _ensure_uuid(workflow_id, "workflow_id")
    row = await pool.fetchrow("SELECT * FROM ai_workflows WHERE id = $1", workflow_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return dict(row)


@router.patch(
    "/workflows/{workflow_id}",
    tags=["Workflows"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def update_workflow_v2(workflow_id: str, patch: WorkflowPatchV2) -> dict[str, Any]:
    """Update workflow definition."""
    updates = _model_dump(patch)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    sets = ["updated_at = NOW()"]
    args: list[Any] = []
    for key, value in updates.items():
        if key in {"trigger_conditions", "success_criteria"}:
            args.append(json.dumps(value))
            sets.append(f"{key} = ${len(args)}::jsonb")
        elif key in {"steps", "decision_points", "failure_handlers"}:
            args.append(value)
            sets.append(f"{key} = ${len(args)}::text[]")
        else:
            args.append(value)
            sets.append(f"{key} = ${len(args)}")

    args.append(_ensure_uuid(workflow_id, "workflow_id"))
    row = await get_pool().fetchrow(
        f"""
        UPDATE ai_workflows
        SET {', '.join(sets)}
        WHERE id = ${len(args)}
        RETURNING *
        """,
        *args,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return dict(row)


@router.post(
    "/workflows/{workflow_id}/execute",
    tags=["Workflows"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def execute_workflow_v2(workflow_id: str, request: WorkflowExecuteRequest) -> dict[str, Any]:
    """Execute workflow (creates execution row)."""
    pool = get_pool()
    workflow_uuid = _ensure_uuid(workflow_id, "workflow_id")
    workflow = await pool.fetchrow("SELECT * FROM ai_workflows WHERE id = $1", workflow_uuid)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    execution = await pool.fetchrow(
        """
        INSERT INTO ai_workflow_executions (workflow_name, status, input, started_at, created_at)
        VALUES ($1, 'running', $2::jsonb, NOW(), NOW())
        RETURNING *
        """,
        workflow["name"],
        json.dumps(request.input),
    )
    return {"execution": dict(execution), "workflow": dict(workflow)}


@router.post(
    "/workflows/executions/{execution_id}/pause",
    tags=["Workflows"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def pause_workflow_execution_v2(execution_id: str) -> dict[str, Any]:
    """Pause workflow execution."""
    pool = get_pool()
    execution_uuid = _ensure_uuid(execution_id, "execution_id")
    row = await pool.fetchrow(
        """
        UPDATE ai_workflow_executions
        SET status = 'paused', completed_at = NULL
        WHERE id = $1
        RETURNING id::text, status
        """,
        execution_uuid,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Execution not found")
    return dict(row)


@router.post(
    "/workflows/executions/{execution_id}/resume",
    tags=["Workflows"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def resume_workflow_execution_v2(execution_id: str) -> dict[str, Any]:
    """Resume workflow execution."""
    pool = get_pool()
    execution_uuid = _ensure_uuid(execution_id, "execution_id")
    row = await pool.fetchrow(
        """
        UPDATE ai_workflow_executions
        SET status = 'running', started_at = COALESCE(started_at, NOW())
        WHERE id = $1
        RETURNING id::text, status
        """,
        execution_uuid,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Execution not found")
    return dict(row)


@router.post(
    "/workflows/{workflow_id}/clone",
    tags=["Workflows"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def clone_workflow_v2(
    workflow_id: str, suffix: str = Query(default="clone")
) -> dict[str, Any]:
    """Clone workflow definition."""
    pool = get_pool()
    workflow_uuid = _ensure_uuid(workflow_id, "workflow_id")
    original = await pool.fetchrow("SELECT * FROM ai_workflows WHERE id = $1", workflow_uuid)
    if not original:
        raise HTTPException(status_code=404, detail="Workflow not found")

    clone = await pool.fetchrow(
        """
        INSERT INTO ai_workflows (
            name, description, trigger_conditions, steps, decision_points,
            success_criteria, failure_handlers, performance_metrics,
            is_active, execution_count, success_rate, avg_execution_time,
            created_at, updated_at
        ) VALUES (
            $1, $2, $3::jsonb, $4::text[], $5::text[],
            $6::jsonb, $7::text[], $8::jsonb,
            false, 0, 0, 0,
            NOW(), NOW()
        )
        RETURNING *
        """,
        f"{original['name']} ({suffix})",
        original.get("description"),
        json.dumps(original.get("trigger_conditions") or {}),
        list(original.get("steps") or []),
        list(original.get("decision_points") or []),
        json.dumps(original.get("success_criteria") or {}),
        list(original.get("failure_handlers") or []),
        json.dumps(original.get("performance_metrics") or {}),
    )
    return dict(clone)


# ---------------------------------------------------------------------------
# Alerts CRUD (tenant-scoped using ai_task_queue)
# ---------------------------------------------------------------------------


class AlertCreateV2(BaseModel):
    title: str
    message: str
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    module: str = "general"
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/alerts",
    tags=["Alerts"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def create_alert_v2(
    payload: AlertCreateV2, tenant_id: str = Depends(get_tenant_id)
) -> dict[str, Any]:
    """Create alert."""
    pool = get_tenant_pool(tenant_id)
    alert_payload = {
        "title": payload.title,
        "message": payload.message,
        "severity": payload.severity,
        "module": payload.module,
        "state": "open",
        "metadata": payload.metadata,
        "created_at": _iso_now(),
    }
    row = await pool.fetchrow(
        """
        INSERT INTO ai_task_queue (tenant_id, task_type, payload, priority, status, created_at, updated_at)
        VALUES ($1::uuid, 'alert', $2::jsonb, $3, 'open', NOW(), NOW())
        RETURNING id::text, status, payload, created_at
        """,
        tenant_id,
        json.dumps(alert_payload),
        _severity_priority(payload.severity),
    )
    return dict(row)


@router.get(
    "/alerts",
    tags=["Alerts"],
    dependencies=[Depends(require_feature_enabled)],
)
async def list_alerts_v2(
    tenant_id: str = Depends(get_tenant_id),
    status: Optional[str] = None,
    severity: Optional[str] = None,
    module: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List alerts."""
    pool = get_tenant_pool(tenant_id)
    where = ["tenant_id = $1::uuid", "task_type = 'alert'"]
    params: list[Any] = [tenant_id]

    if status:
        params.append(status)
        where.append(f"status = ${len(params)}")
    if severity:
        params.append(severity)
        where.append(f"payload->>'severity' = ${len(params)}")
    if module:
        params.append(module)
        where.append(f"payload->>'module' = ${len(params)}")

    params.append(limit)
    rows = await pool.fetch(
        f"""
        SELECT id::text, status, payload, priority, created_at, updated_at, completed_at
        FROM ai_task_queue
        WHERE {' AND '.join(where)}
        ORDER BY created_at DESC
        LIMIT ${len(params)}
        """,
        *params,
    )
    return {"items": [dict(r) for r in rows], "total": len(rows)}


@router.get(
    "/alerts/{alert_id}",
    tags=["Alerts"],
    dependencies=[Depends(require_feature_enabled)],
)
async def get_alert_v2(alert_id: str, tenant_id: str = Depends(get_tenant_id)) -> dict[str, Any]:
    """Read alert."""
    pool = get_tenant_pool(tenant_id)
    alert_uuid = _ensure_uuid(alert_id, "alert_id")
    row = await pool.fetchrow(
        """
        SELECT id::text, status, payload, priority, created_at, updated_at, completed_at
        FROM ai_task_queue
        WHERE id = $1 AND tenant_id = $2::uuid AND task_type = 'alert'
        """,
        alert_uuid,
        tenant_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    return dict(row)


@router.post(
    "/alerts/{alert_id}/acknowledge",
    tags=["Alerts"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def acknowledge_alert_v2(
    alert_id: str,
    tenant_id: str = Depends(get_tenant_id),
    actor: str = Query(default="api_v2"),
) -> dict[str, Any]:
    """Acknowledge alert."""
    return await _update_alert_state(alert_id, tenant_id, "acknowledged", actor=actor)


@router.post(
    "/alerts/{alert_id}/resolve",
    tags=["Alerts"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def resolve_alert_v2(
    alert_id: str,
    tenant_id: str = Depends(get_tenant_id),
    actor: str = Query(default="api_v2"),
) -> dict[str, Any]:
    """Resolve alert."""
    return await _update_alert_state(alert_id, tenant_id, "resolved", actor=actor)


@router.post(
    "/alerts/{alert_id}/escalate",
    tags=["Alerts"],
    dependencies=[Depends(require_feature_enabled), Depends(WRITE_LIMIT)],
)
async def escalate_alert_v2(
    alert_id: str,
    tenant_id: str = Depends(get_tenant_id),
    actor: str = Query(default="api_v2"),
) -> dict[str, Any]:
    """Escalate alert severity and state."""
    pool = get_tenant_pool(tenant_id)
    alert_uuid = _ensure_uuid(alert_id, "alert_id")
    row = await pool.fetchrow(
        """
        UPDATE ai_task_queue
        SET payload = COALESCE(payload, '{}'::jsonb)
            || '{"severity":"critical","state":"escalated"}'::jsonb
            || jsonb_build_object('escalated_by', $1, 'escalated_at', NOW()::text),
            status = 'escalated',
            priority = 100,
            updated_at = NOW()
        WHERE id = $2
          AND tenant_id = $3::uuid
          AND task_type = 'alert'
        RETURNING id::text, status, payload, priority, updated_at
        """,
        actor,
        alert_uuid,
        tenant_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    return dict(row)


async def _update_alert_state(
    alert_id: str, tenant_id: str, state: str, actor: str
) -> dict[str, Any]:
    pool = get_tenant_pool(tenant_id)
    alert_uuid = _ensure_uuid(alert_id, "alert_id")
    row = await pool.fetchrow(
        """
        UPDATE ai_task_queue
        SET payload = COALESCE(payload, '{}'::jsonb)
            || jsonb_build_object('state', $1, 'updated_by', $2, 'updated_at', NOW()::text),
            status = $1,
            completed_at = CASE WHEN $1 = 'resolved' THEN NOW() ELSE completed_at END,
            updated_at = NOW()
        WHERE id = $3
          AND tenant_id = $4::uuid
          AND task_type = 'alert'
        RETURNING id::text, status, payload, updated_at, completed_at
        """,
        state,
        actor,
        alert_uuid,
        tenant_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    return dict(row)


# ---------------------------------------------------------------------------
# Brain Logs (read/search/filter severity/module/time-range)
# ---------------------------------------------------------------------------


@router.get(
    "/brain-logs",
    tags=["Brain Logs"],
    dependencies=[Depends(require_feature_enabled)],
)
async def list_brain_logs_v2(
    severity: Optional[str] = Query(default=None),
    module: Optional[str] = Query(default=None),
    contains: Optional[str] = Query(default=None),
    from_ts: Optional[datetime] = Query(default=None),
    to_ts: Optional[datetime] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """Read/search logs with severity/module/time filtering."""
    pool = get_pool()
    source_table = await _detect_log_table(pool)
    if not source_table:
        return {"items": [], "total": 0, "source": None}

    start = from_ts or datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end = to_ts or datetime.now(timezone.utc)

    rows = await pool.fetch(
        f"""
        SELECT to_jsonb(t) AS row_data
        FROM {source_table} t
        WHERE created_at >= $1
          AND created_at <= $2
        ORDER BY created_at DESC
        LIMIT $3
        """,
        start,
        end,
        limit,
    )

    items: list[dict[str, Any]] = []
    for row in rows:
        data = row.get("row_data") or {}
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = {}

        item = {
            "timestamp": data.get("created_at") or data.get("timestamp"),
            "severity": (
                data.get("severity")
                or data.get("level")
                or (data.get("data") or {}).get("level")
                or "info"
            ),
            "module": data.get("module")
            or data.get("system")
            or data.get("logger")
            or data.get("action"),
            "message": data.get("message")
            or (data.get("data") or {}).get("message")
            or json.dumps(data),
            "raw": data,
        }

        if severity and str(item["severity"]).lower() != severity.lower():
            continue
        if module and module.lower() not in str(item["module"] or "").lower():
            continue
        if contains and contains.lower() not in str(item["message"] or "").lower():
            continue
        items.append(item)

    return {
        "items": items,
        "total": len(items),
        "source": source_table,
        "filters": {
            "severity": severity,
            "module": module,
            "contains": contains,
            "from_ts": start.isoformat(),
            "to_ts": end.isoformat(),
        },
    }


async def _detect_log_table(pool) -> Optional[str]:
    tables = ["unified_brain_logs", "agent_execution_logs"]
    row = await pool.fetchrow(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = ANY($1::text[])
        ORDER BY CASE table_name WHEN 'unified_brain_logs' THEN 0 ELSE 1 END
        LIMIT 1
        """,
        tables,
    )
    return row["table_name"] if row else None


def _safe_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            data = json.loads(value)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _row_datetime_iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def _lead_row_to_response(row: Any) -> LeadResponse:
    data = dict(row)
    metadata = _safe_json(data.get("metadata"))
    return LeadResponse(
        id=str(data.get("id")),
        company_name=data.get("company_name"),
        contact_name=data.get("contact_name"),
        email=data.get("email"),
        phone=data.get("phone"),
        website=data.get("website"),
        stage=data.get("stage"),
        score=float(data.get("score") or 0.0) if data.get("score") is not None else None,
        value_estimate=float(data.get("value_estimate") or 0.0)
        if data.get("value_estimate") is not None
        else None,
        source=data.get("source"),
        metadata=metadata,
        created_at=_row_datetime_iso(data.get("created_at")),
        updated_at=_row_datetime_iso(data.get("updated_at")),
    )


def _severity_priority(severity: str) -> int:
    mapping = {"critical": 100, "high": 80, "medium": 50, "low": 20}
    return mapping.get((severity or "").lower(), 50)

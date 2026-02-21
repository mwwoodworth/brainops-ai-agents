"""Tenant context helpers — extracted from app.py (Wave 2C).

Provides tenant UUID resolution and tenant-scoped database queries.
"""

import os
import uuid
from typing import Any, Optional

from starlette.requests import HTTPConnection

from config import config


def normalize_tenant_uuid(candidate: Any) -> Optional[str]:
    """Return canonical UUID string or None when input is missing/invalid."""
    if candidate is None:
        return None
    try:
        raw = str(candidate).strip()
    except Exception:
        return None

    if not raw or raw.lower() in {"null", "none", "undefined"}:
        return None

    try:
        return str(uuid.UUID(raw))
    except (ValueError, TypeError, AttributeError):
        return None


def resolve_tenant_uuid_from_request(request: Optional[HTTPConnection]) -> Optional[str]:
    """Resolve tenant UUID from request context, then fall back to configured default."""
    candidates: list[Any] = []
    if request is not None:
        try:
            candidates.append(request.headers.get(config.tenant.header_name))
        except Exception:
            pass

        state = getattr(request, "state", None)
        if state is not None:
            candidates.append(getattr(state, "tenant_id", None))
            user = getattr(state, "user", None)
            if isinstance(user, dict):
                candidates.append(user.get("tenant_id"))

    candidates.extend([config.tenant.default_tenant_id, os.getenv("DEFAULT_TENANT_ID")])

    for candidate in candidates:
        normalized = normalize_tenant_uuid(candidate)
        if normalized:
            return normalized
    return None


async def fetchval_with_tenant_context(
    pool: Any,
    query: str,
    *args: Any,
    tenant_uuid: Optional[str] = None,
):
    """
    Execute fetchval in a transaction with explicit tenant context.
    Prevents stale/invalid session tenant settings (e.g. empty string UUID).
    """
    raw_pool = getattr(pool, "pool", None) or getattr(pool, "_pool", None)
    if raw_pool is None:
        return await pool.fetchval(query, *args)

    async with raw_pool.acquire(timeout=10.0) as conn:
        async with conn.transaction():
            if tenant_uuid:
                await conn.execute(
                    "SELECT set_config('app.current_tenant_id', $1, true)", tenant_uuid
                )
            else:
                await conn.execute("RESET app.current_tenant_id")
            return await conn.fetchval(query, *args)

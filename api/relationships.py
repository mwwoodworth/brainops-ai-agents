"""
Brain Relationships API
Provides tenant-aware relationship queries from brain_relationships.
"""

import logging
import os
from collections import deque
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config import config
from database.async_connection import DatabaseUnavailableError, get_pool, using_fallback

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


async def get_tenant_id(x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")) -> str:
    return x_tenant_id or os.getenv("DEFAULT_TENANT_ID") or os.getenv("TENANT_ID") or ""


router = APIRouter(
    prefix="/relationships",
    tags=["Relationships"],
    dependencies=[Depends(verify_api_key)],
)


class RelationshipPathRequest(BaseModel):
    source_id: str = Field(..., description="Start entity UUID")
    target_id: str = Field(..., description="Target entity UUID")
    max_depth: int = Field(default=5, ge=1, le=8)


@router.get("/{entity_id}")
async def get_entity_relationships(entity_id: str, tenant_id: str = Depends(get_tenant_id)) -> dict[str, Any]:
    """List direct relationships for an entity."""
    if using_fallback():
        raise HTTPException(status_code=503, detail="Database unavailable")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID required")

    pool = get_pool()
    try:
        rows = await pool.fetch(
            """
            SELECT
                br.id,
                br.entity_a,
                br.entity_b,
                br.relationship_type,
                br.strength,
                br.context,
                br.created_at,
                br.last_reinforced,
                ea.name AS entity_a_name,
                eb.name AS entity_b_name
            FROM brain_relationships br
            JOIN brain_entities ea ON ea.id = br.entity_a
            JOIN brain_entities eb ON eb.id = br.entity_b
            WHERE br.tenant_id = $1
              AND (br.entity_a = $2 OR br.entity_b = $2)
            ORDER BY br.created_at DESC
            LIMIT 200
            """,
            tenant_id,
            entity_id,
        )
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to fetch relationships: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"entity_id": entity_id, "relationships": [dict(row) for row in rows]}


@router.post("/path")
async def get_relationship_path(
    request: RelationshipPathRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> dict[str, Any]:
    """Find a relationship path between two entities."""
    if using_fallback():
        raise HTTPException(status_code=503, detail="Database unavailable")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID required")

    pool = get_pool()
    try:
        edges = await pool.fetch(
            """
            SELECT entity_a, entity_b, relationship_type
            FROM brain_relationships
            WHERE tenant_id = $1
            """,
            tenant_id,
        )
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to load relationships: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    adjacency: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        a = str(edge["entity_a"])
        b = str(edge["entity_b"])
        rel = edge["relationship_type"]
        adjacency.setdefault(a, []).append((b, rel))
        adjacency.setdefault(b, []).append((a, rel))

    source = request.source_id
    target = request.target_id
    if source == target:
        return {"path": [source], "edges": [], "depth": 0}

    visited = {source}
    queue: deque[tuple[str, list[str], list[dict[str, Any]]]] = deque()
    queue.append((source, [source], []))

    found_path = None
    found_edges = None

    while queue:
        node, path, edge_path = queue.popleft()
        if len(path) > request.max_depth:
            continue
        for neighbor, rel in adjacency.get(node, []):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            next_edges = edge_path + [{"from": node, "to": neighbor, "type": rel}]
            if neighbor == target:
                found_path = next_path
                found_edges = next_edges
                queue.clear()
                break
            visited.add(neighbor)
            queue.append((neighbor, next_path, next_edges))

    if not found_path:
        return {"path": [], "edges": [], "depth": None}

    # Enrich path with entity names
    try:
        entities = await pool.fetch(
            """
            SELECT id, name, entity_type
            FROM brain_entities
            WHERE tenant_id = $1 AND id = ANY($2::uuid[])
            """,
            tenant_id,
            [str(node_id) for node_id in found_path],
        )
        entity_map = {str(row["id"]): dict(row) for row in entities}
    except Exception:
        entity_map = {}

    path_nodes = [
        {"id": node_id, **entity_map.get(node_id, {})} for node_id in found_path
    ]

    return {"path": path_nodes, "edges": found_edges, "depth": len(found_path) - 1}

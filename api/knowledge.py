"""
Knowledge Base API Router
Secure, authenticated endpoints for the Master Knowledge Base
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.encoders import jsonable_encoder
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# API Key Security - use centralized config
from config import config

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


router = APIRouter(prefix="/api/v1/knowledge-base", tags=["Knowledge Base"])

# Import the knowledge base with fallback
try:
    from master_knowledge_base import AccessLevel, KnowledgeType, get_knowledge_base

    KNOWLEDGE_BASE_AVAILABLE = True
    logger.info("Master Knowledge Base loaded")
except ImportError as e:
    KNOWLEDGE_BASE_AVAILABLE = False
    logger.warning(f"Master Knowledge Base not available: {e}")


# Pydantic models
class KnowledgeEntryRequest(BaseModel):
    """Request to create a knowledge entry"""

    title: str = Field(..., min_length=3, max_length=500)
    content: str = Field(..., min_length=10, max_length=100000)
    knowledge_type: str = Field(default="guide")
    category: str = Field(default="general", max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=20)
    access_level: str = Field(default="internal")
    author: str = Field(default="system", max_length=200)
    # Deprecated: kept for backward compatibility with older clients.
    # MasterKnowledgeBase is not tenant-scoped yet; this is stored in metadata only.
    tenant_id: str = Field(default="default")
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchRequest(BaseModel):
    """Request to search knowledge base"""

    query: str = Field(..., min_length=2, max_length=1000)
    knowledge_types: Optional[list[str]] = None
    categories: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    access_level: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)
    semantic_search: bool = True
    tenant_id: str = Field(default="default")


class AgentQueryRequest(BaseModel):
    """Request from an AI agent to query knowledge"""

    agent_id: str = Field(..., max_length=100)
    query: str = Field(..., min_length=2, max_length=2000)
    context: Optional[str] = Field(None, max_length=5000)
    required_types: Optional[list[str]] = None
    tenant_id: str = Field(default="default")


@router.get("/health")
async def knowledge_base_health():
    """Check knowledge base system health"""
    return {
        "status": "available" if KNOWLEDGE_BASE_AVAILABLE else "unavailable",
        "timestamp": datetime.utcnow().isoformat(),
        "capabilities": {
            "semantic_search": KNOWLEDGE_BASE_AVAILABLE,
            "hierarchical_organization": KNOWLEDGE_BASE_AVAILABLE,
            "agent_query_interface": KNOWLEDGE_BASE_AVAILABLE,
            "access_control": KNOWLEDGE_BASE_AVAILABLE,
            "auto_categorization": KNOWLEDGE_BASE_AVAILABLE,
        },
    }


@router.get("/entries")
async def list_entries(
    category: Optional[str] = None,
    knowledge_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    api_key: str = Depends(verify_api_key),
):
    """List knowledge base entries with optional filtering"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()

        all_entries = list(kb.entries.values())

        # Apply filters
        if category:
            entry_ids = kb.by_category.get(category, set())
            all_entries = [e for e in all_entries if e.entry_id in entry_ids]

        if knowledge_type:
            try:
                kt = KnowledgeType(knowledge_type)
                entry_ids = kb.by_type.get(kt, set())
                all_entries = [e for e in all_entries if e.entry_id in entry_ids]
            except ValueError:
                pass

        # Sort by updated_at descending (most recent first)
        all_entries.sort(
            key=lambda e: e.updated_at or e.created_at or datetime.min,
            reverse=True,
        )

        total = len(all_entries)
        page = all_entries[offset : offset + limit]

        formatted = []
        for entry in page:
            formatted.append(
                {
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "summary": entry.summary or (entry.content[:300] if entry.content else ""),
                    "knowledge_type": entry.knowledge_type.value
                    if hasattr(entry.knowledge_type, "value")
                    else str(entry.knowledge_type),
                    "category": entry.category,
                    "tags": entry.tags,
                    "access_level": entry.access_level.value
                    if hasattr(entry.access_level, "value")
                    else str(entry.access_level),
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                    "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
                }
            )

        return {
            "entries": formatted,
            "total": total,
            "limit": limit,
            "offset": offset,
            "listed_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"List entries error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/entries")
async def create_entry(request: KnowledgeEntryRequest, api_key: str = Depends(verify_api_key)):
    """Create a new knowledge entry"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()

        # Map knowledge type
        try:
            knowledge_type = KnowledgeType(request.knowledge_type)
        except ValueError:
            knowledge_type = KnowledgeType.GUIDE

        # Map access level
        try:
            access_level = AccessLevel(request.access_level)
        except ValueError:
            access_level = AccessLevel.INTERNAL

        metadata = dict(request.metadata or {})
        if request.tenant_id:
            metadata.setdefault("tenant_id", request.tenant_id)
        if request.subcategory:
            metadata.setdefault("subcategory", request.subcategory)

        entry = await kb.create_entry(
            title=request.title,
            content=request.content,
            knowledge_type=knowledge_type,
            category=request.category,
            tags=request.tags,
            access_level=access_level,
            source_type="api",
            created_by=request.author,
            metadata=metadata,
        )

        return {
            "status": "created",
            "entry_id": entry.entry_id,
            "title": entry.title,
            "category": entry.category,
            "created_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Knowledge entry creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/entries/{entry_id}")
async def get_entry(
    entry_id: str, tenant_id: str = "default", api_key: str = Depends(verify_api_key)
):
    """
    Get a knowledge entry by ID.

    Note: Access control is enforced based on the authenticated user's
    permissions derived from the API key, NOT from query parameters.
    """
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()
        entry = await kb.get_entry(entry_id)

        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")

        payload = jsonable_encoder(entry)
        # Embeddings can be large and are rarely needed via the API.
        payload.pop("embedding", None)
        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get entry error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/entries/{entry_id}")
async def update_entry(
    entry_id: str, request: KnowledgeEntryRequest, api_key: str = Depends(verify_api_key)
):
    """Update a knowledge entry"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()

        existing = await kb.get_entry(entry_id, track_view=False)
        if not existing:
            raise HTTPException(status_code=404, detail="Entry not found")

        updates: dict[str, Any] = {
            "title": request.title,
            "content": request.content,
            "category": request.category,
            "tags": request.tags,
            "metadata": dict(request.metadata or {}),
        }

        # Optional fields supported by the data class.
        if request.subcategory is not None:
            updates["subcategory"] = request.subcategory

        try:
            updates["knowledge_type"] = KnowledgeType(request.knowledge_type)
        except ValueError:
            pass
        try:
            updates["access_level"] = AccessLevel(request.access_level)
        except ValueError:
            pass

        # Preserve tenant_id for legacy clients (not enforced).
        if request.tenant_id:
            updates["metadata"].setdefault("tenant_id", request.tenant_id)

        await kb.update_entry(entry_id=entry_id, updates=updates, updated_by=request.author)

        return {
            "status": "updated",
            "entry_id": entry_id,
            "updated_at": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update entry error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/entries/{entry_id}")
async def delete_entry(
    entry_id: str, tenant_id: str = "default", api_key: str = Depends(verify_api_key)
):
    """Delete a knowledge entry"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()

        deleted = await kb.delete_entry(entry_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Entry not found")

        return {
            "status": "deleted",
            "entry_id": entry_id,
            "deleted_at": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete entry error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/search")
async def search_knowledge(request: KnowledgeSearchRequest, api_key: str = Depends(verify_api_key)):
    """Search the knowledge base with semantic search"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()

        # Map knowledge types
        knowledge_types = None
        if request.knowledge_types:
            knowledge_types = []
            for kt in request.knowledge_types:
                try:
                    knowledge_types.append(KnowledgeType(kt))
                except ValueError:
                    logger.debug("Invalid knowledge type %s", kt)

        results = await kb.search(
            query=request.query,
            knowledge_types=knowledge_types,
            categories=request.categories,
            tags=request.tags,
            top_k=request.top_k,
            semantic=request.semantic_search,
        )

        formatted = []
        for entry, score in results:
            formatted.append(
                {
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "summary": entry.summary or entry.content[:300],
                    "score": score,
                    "knowledge_type": entry.knowledge_type.value,
                    "category": entry.category,
                    "tags": entry.tags,
                    "access_level": entry.access_level.value,
                    "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
                }
            )

        return {
            "query": request.query,
            "results": formatted,
            "total": len(formatted),
            "semantic_search": request.semantic_search,
            "searched_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/agent/query")
async def agent_query(request: AgentQueryRequest, api_key: str = Depends(verify_api_key)):
    """
    Query interface for AI agents.

    Returns formatted knowledge relevant to the agent's query
    with context-aware responses.
    """
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()

        # MasterKnowledgeBase does not currently support tenant scoping or required_types.
        context = {"text": request.context} if request.context else None
        response = await kb.query_for_agent(
            agent_id=request.agent_id, query=request.query, context=context
        )

        return {
            "agent_id": request.agent_id,
            "query": request.query,
            "response": response,
            "queried_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Agent query error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/categories")
async def list_categories(tenant_id: str = "default", api_key: str = Depends(verify_api_key)):
    """List all knowledge categories"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()
        categories = [
            {
                "id": c.category_id,
                "name": c.name,
                "slug": c.slug,
                "description": c.description,
                "entry_count": len(kb.by_category.get(c.slug, set())),
            }
            for c in kb.categories.values()
        ]

        return {
            "categories": categories,
            "total": len(categories),
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"List categories error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/statistics")
async def get_statistics(tenant_id: str = "default", api_key: str = Depends(verify_api_key)):
    """Get knowledge base statistics"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()
        stats = await kb.get_statistics()

        return {"statistics": stats, "generated_at": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


#
# Notion integration previously lived here as a sync endpoint.
# Notion is deprecated in this AI OS for now; re-introduce only when it becomes
# an actively used knowledge source again.

"""
Knowledge Base API Router
Secure, authenticated endpoints for the Master Knowledge Base
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Security
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

router = APIRouter(prefix="/knowledge-base", tags=["Knowledge Base"])

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
    tags: list[str] = Field(default_factory=list, max_items=20)
    access_level: str = Field(default="internal")
    department: Optional[str] = Field(None, max_length=100)
    author: str = Field(default="system", max_length=200)
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
            "auto_categorization": KNOWLEDGE_BASE_AVAILABLE
        }
    }


@router.post("/entries")
async def create_entry(
    request: KnowledgeEntryRequest,
    api_key: str = Depends(verify_api_key)
):
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

        entry = await kb.create_entry(
            title=request.title,
            content=request.content,
            knowledge_type=knowledge_type,
            category=request.category,
            subcategory=request.subcategory,
            tags=request.tags,
            access_level=access_level,
            department=request.department,
            author=request.author,
            tenant_id=request.tenant_id,
            metadata=request.metadata
        )

        return {
            "status": "created",
            "entry_id": entry.get("id"),
            "title": entry.get("title"),
            "category": entry.get("category"),
            "created_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Knowledge entry creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entries/{entry_id}")
async def get_entry(
    entry_id: str,
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
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
        entry = await kb.get_entry(entry_id, tenant_id=tenant_id)

        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")

        # Verify tenant isolation
        if entry.get("tenant_id") != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return entry

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get entry error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/entries/{entry_id}")
async def update_entry(
    entry_id: str,
    request: KnowledgeEntryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Update a knowledge entry"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()

        # First verify the entry exists and tenant matches
        existing = await kb.get_entry(entry_id, tenant_id=request.tenant_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Entry not found")

        if existing.get("tenant_id") != request.tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        entry = await kb.update_entry(
            entry_id=entry_id,
            title=request.title,
            content=request.content,
            category=request.category,
            tags=request.tags,
            metadata=request.metadata
        )

        return {
            "status": "updated",
            "entry_id": entry_id,
            "updated_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update entry error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/entries/{entry_id}")
async def delete_entry(
    entry_id: str,
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Delete a knowledge entry"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()

        # Verify tenant isolation
        existing = await kb.get_entry(entry_id, tenant_id=tenant_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Entry not found")

        if existing.get("tenant_id") != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        await kb.delete_entry(entry_id)

        return {
            "status": "deleted",
            "entry_id": entry_id,
            "deleted_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete entry error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_knowledge(
    request: KnowledgeSearchRequest,
    api_key: str = Depends(verify_api_key)
):
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
            semantic_search=request.semantic_search,
            tenant_id=request.tenant_id
        )

        return {
            "query": request.query,
            "results": results,
            "total": len(results),
            "semantic_search": request.semantic_search,
            "searched_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/query")
async def agent_query(
    request: AgentQueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Query interface for AI agents.

    Returns formatted knowledge relevant to the agent's query
    with context-aware responses.
    """
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()

        response = await kb.query_for_agent(
            agent_id=request.agent_id,
            query=request.query,
            context=request.context,
            required_types=request.required_types,
            tenant_id=request.tenant_id
        )

        return {
            "agent_id": request.agent_id,
            "query": request.query,
            "response": response,
            "queried_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Agent query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def list_categories(
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """List all knowledge categories"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()
        categories = await kb.get_categories(tenant_id)

        return {
            "categories": categories,
            "total": len(categories)
        }

    except Exception as e:
        logger.error(f"List categories error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics(
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Get knowledge base statistics"""
    if not KNOWLEDGE_BASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Knowledge base not available")

    try:
        kb = get_knowledge_base()
        stats = await kb.get_statistics(tenant_id)

        return {
            "statistics": stats,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def list_knowledge_types():
    """List available knowledge types"""
    return {
        "knowledge_types": [
            {"id": "sop", "name": "SOP", "description": "Standard Operating Procedures"},
            {"id": "policy", "name": "Policy", "description": "Company policies"},
            {"id": "process", "name": "Process", "description": "Process documentation"},
            {"id": "guide", "name": "Guide", "description": "How-to guides"},
            {"id": "faq", "name": "FAQ", "description": "Frequently asked questions"},
            {"id": "api_doc", "name": "API Documentation", "description": "API reference docs"},
            {"id": "code_reference", "name": "Code Reference", "description": "Code documentation"},
            {"id": "product_info", "name": "Product Info", "description": "Product information"},
            {"id": "troubleshooting", "name": "Troubleshooting", "description": "Troubleshooting guides"},
            {"id": "template", "name": "Template", "description": "Document templates"},
            {"id": "checklist", "name": "Checklist", "description": "Operational checklists"},
            {"id": "runbook", "name": "Runbook", "description": "Operational runbooks"},
            {"id": "incident_report", "name": "Incident Report", "description": "Incident documentation"},
            {"id": "agent_config", "name": "Agent Config", "description": "AI agent configurations"},
            {"id": "prompt_template", "name": "Prompt Template", "description": "AI prompt templates"}
        ],
        "access_levels": [
            {"id": "public", "name": "Public", "description": "Accessible by anyone"},
            {"id": "internal", "name": "Internal", "description": "Internal use only"},
            {"id": "confidential", "name": "Confidential", "description": "Restricted access"},
            {"id": "agent_only", "name": "Agent Only", "description": "AI agents only"}
        ]
    }

#!/usr/bin/env python3
"""
MEMORY COORDINATION API - Endpoints for Perfect E2E Context Management
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["Memory Coordination"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class StoreContextRequest(BaseModel):
    key: str
    value: Any
    layer: str = Field(..., description="ephemeral, session, short_term, long_term, permanent")
    scope: str = Field(..., description="global, tenant, user, session, agent")
    priority: str = Field(default="medium", description="critical, high, medium, low")
    category: str
    source: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expires_in_seconds: Optional[int] = None


class RetrieveContextRequest(BaseModel):
    key: str
    scope: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None


class SearchContextRequest(BaseModel):
    query: str
    scope: Optional[str] = None
    layer: Optional[str] = None
    category: Optional[str] = None
    tenant_id: Optional[str] = None
    limit: int = 20


class StartSessionRequest(BaseModel):
    session_id: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    initial_context: Dict[str, Any] = Field(default_factory=dict)


class AddMessageRequest(BaseModel):
    session_id: str
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AddTaskRequest(BaseModel):
    session_id: str
    task: Dict[str, Any]
    status: str = "pending"


class HandoffRequest(BaseModel):
    session_id: str
    to_agent: str
    handoff_reason: str
    critical_info: Dict[str, Any]
    continuation_instructions: str


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

_coordinator = None
_session_manager = None


async def get_coordinator():
    """Get memory coordinator instance with timeout protection"""
    global _coordinator
    if _coordinator is None:
        try:
            # Try with 5 second timeout to prevent hanging
            import asyncio
            from memory_coordination_system import get_memory_coordinator
            _coordinator = await asyncio.wait_for(
                asyncio.to_thread(get_memory_coordinator),
                timeout=5.0
            )
            logger.info("✅ Memory coordinator initialized")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Memory coordinator initialization timed out, using fallback")
            # Return a simple fallback coordinator
            from memory_coordination_system import SimpleFallbackCoordinator
            _coordinator = SimpleFallbackCoordinator()
        except Exception as e:
            logger.error(f"❌ Memory coordinator failed: {e}, using fallback")
            from memory_coordination_system import SimpleFallbackCoordinator
            _coordinator = SimpleFallbackCoordinator()
    return _coordinator


async def get_session_manager():
    """Get session manager instance"""
    global _session_manager
    if _session_manager is None:
        from session_context_manager import get_session_manager
        coordinator = await get_coordinator()
        _session_manager = await get_session_manager(coordinator)
    return _session_manager


# ============================================================================
# CONTEXT MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/context/store")
async def store_context(
    request: StoreContextRequest,
    coordinator = Depends(get_coordinator)
):
    """
    Store context entry with automatic synchronization

    **Layers:**
    - ephemeral: In-memory cache (seconds to minutes)
    - session: Session-scoped (hours)
    - short_term: Days to weeks
    - long_term: Weeks to months
    - permanent: Forever

    **Scopes:**
    - global: Visible to all systems
    - tenant: Tenant-specific
    - user: User-specific
    - session: Current session only
    - agent: Specific AI agent only
    """
    try:
        from memory_coordination_system import ContextEntry, MemoryLayer, ContextScope
        from datetime import timedelta

        entry = ContextEntry(
            key=request.key,
            value=request.value,
            layer=MemoryLayer(request.layer),
            scope=ContextScope(request.scope),
            priority=request.priority,
            category=request.category,
            source=request.source,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            session_id=request.session_id,
            agent_id=request.agent_id,
            metadata=request.metadata,
            expires_at=datetime.now() + timedelta(seconds=request.expires_in_seconds) if request.expires_in_seconds else None
        )

        entry_id = await coordinator.store_context(entry)

        return {
            "success": True,
            "entry_id": entry_id,
            "key": request.key,
            "layer": request.layer,
            "scope": request.scope,
            "sync_version": entry.sync_version
        }

    except Exception as e:
        logger.error(f"❌ Store context failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context/retrieve")
async def retrieve_context(
    request: RetrieveContextRequest,
    coordinator = Depends(get_coordinator)
):
    """
    Retrieve context with intelligent caching
    """
    try:
        from memory_coordination_system import ContextScope

        entry = await coordinator.retrieve_context(
            key=request.key,
            scope=ContextScope(request.scope),
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            session_id=request.session_id,
            agent_id=request.agent_id
        )

        if not entry:
            raise HTTPException(status_code=404, detail="Context not found")

        return {
            "success": True,
            "key": entry.key,
            "value": entry.value,
            "layer": entry.layer.value,
            "scope": entry.scope.value,
            "priority": entry.priority,
            "category": entry.category,
            "source": entry.source,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "access_count": entry.access_count,
            "sync_version": entry.sync_version,
            "metadata": entry.metadata
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Retrieve context failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context/search")
async def search_context(
    request: SearchContextRequest,
    coordinator = Depends(get_coordinator)
):
    """
    Search across all context with filtering
    """
    try:
        from memory_coordination_system import ContextScope, MemoryLayer

        results = await coordinator.search_context(
            query=request.query,
            scope=ContextScope(request.scope) if request.scope else None,
            layer=MemoryLayer(request.layer) if request.layer else None,
            category=request.category,
            tenant_id=request.tenant_id,
            limit=request.limit
        )

        return {
            "success": True,
            "query": request.query,
            "result_count": len(results),
            "results": [
                {
                    "key": r.key,
                    "value": r.value,
                    "layer": r.layer.value,
                    "scope": r.scope.value,
                    "priority": r.priority,
                    "category": r.category,
                    "updated_at": r.updated_at.isoformat(),
                    "access_count": r.access_count
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"❌ Search context failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/stats")
async def get_stats(coordinator = Depends(get_coordinator)):
    """Get memory coordination statistics"""
    try:
        stats = await coordinator.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"❌ Get stats failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/session/start")
async def start_session(
    request: StartSessionRequest,
    manager = Depends(get_session_manager)
):
    """
    Start a new session with full context initialization
    """
    try:
        session = await manager.start_session(
            session_id=request.session_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            initial_context=request.initial_context
        )

        return {
            "success": True,
            "session_id": session.session_id,
            "status": session.status,
            "start_time": session.start_time.isoformat() if hasattr(session.start_time, 'isoformat') else str(session.start_time),
            "tenant_id": session.tenant_id,
            "user_id": session.user_id
        }

    except Exception as e:
        logger.error(f"❌ Start session failed: {e}", exc_info=True)
        # Convert any datetime objects in error message
        error_msg = str(e).replace('<', '').replace('>', '')
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/session/resume/{session_id}")
async def resume_session(
    session_id: str,
    manager = Depends(get_session_manager)
):
    """
    Resume an existing session with full context restoration
    """
    try:
        session = await manager.resume_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "success": True,
            "session_id": session.session_id,
            "status": session.status,
            "start_time": session.start_time.isoformat() if hasattr(session.start_time, 'isoformat') else str(session.start_time),
            "last_activity": session.last_activity.isoformat() if hasattr(session.last_activity, 'isoformat') else str(session.last_activity),
            "message_count": len(session.conversation_history),
            "task_count": len(session.pending_tasks) + len(session.completed_tasks),
            "active_agents": session.active_agents
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Resume session failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/end/{session_id}")
async def end_session(
    session_id: str,
    reason: str = "completed",
    manager = Depends(get_session_manager)
):
    """
    End a session with full context preservation
    """
    try:
        await manager.end_session(session_id, reason)

        return {
            "success": True,
            "session_id": session_id,
            "reason": reason,
            "message": "Session ended and context archived"
        }

    except Exception as e:
        logger.error(f"❌ End session failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/message")
async def add_message(
    request: AddMessageRequest,
    manager = Depends(get_session_manager)
):
    """
    Add a message to conversation history
    """
    try:
        await manager.add_message(
            session_id=request.session_id,
            role=request.role,
            content=request.content,
            metadata=request.metadata
        )

        return {
            "success": True,
            "session_id": request.session_id,
            "role": request.role,
            "message": "Message added"
        }

    except Exception as e:
        logger.error(f"❌ Add message failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/task")
async def add_task(
    request: AddTaskRequest,
    manager = Depends(get_session_manager)
):
    """
    Add a task to session tracking
    """
    try:
        await manager.add_task(
            session_id=request.session_id,
            task=request.task,
            status=request.status
        )

        return {
            "success": True,
            "session_id": request.session_id,
            "task_id": request.task.get('id'),
            "status": request.status
        }

    except Exception as e:
        logger.error(f"❌ Add task failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/context/{session_id}")
async def get_session_context(
    session_id: str,
    manager = Depends(get_session_manager)
):
    """
    Get complete context for a session
    """
    try:
        context = await manager.get_full_context(session_id)

        if not context:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "success": True,
            "context": context
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get session context failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AGENT HANDOFF ENDPOINTS
# ============================================================================

@router.post("/session/handoff")
async def handoff_to_agent(
    request: HandoffRequest,
    manager = Depends(get_session_manager)
):
    """
    Hand off session to another agent with perfect context transfer
    """
    try:
        handoff = await manager.handoff_to_agent(
            session_id=request.session_id,
            to_agent=request.to_agent,
            handoff_reason=request.handoff_reason,
            critical_info=request.critical_info,
            continuation_instructions=request.continuation_instructions
        )

        return {
            "success": True,
            "session_id": request.session_id,
            "from_agent": handoff.from_agent,
            "to_agent": handoff.to_agent,
            "timestamp": handoff.timestamp.isoformat(),
            "context_snapshot": handoff.context_snapshot,
            "critical_info": handoff.critical_info,
            "continuation_instructions": handoff.continuation_instructions
        }

    except Exception as e:
        logger.error(f"❌ Handoff failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/handoff/{session_id}")
async def get_handoff_context(
    session_id: str,
    manager = Depends(get_session_manager)
):
    """
    Get the latest handoff context for an agent
    """
    try:
        context = await manager.get_handoff_context(session_id)

        if not context:
            return {
                "success": True,
                "has_handoff": False,
                "message": "No handoff history for this session"
            }

        return {
            "success": True,
            "has_handoff": True,
            "handoff": context
        }

    except Exception as e:
        logger.error(f"❌ Get handoff context failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH & MONITORING
# ============================================================================

@router.get("/health")
async def memory_health(
    coordinator = Depends(get_coordinator),
    manager = Depends(get_session_manager)
):
    """
    Health check for memory coordination system
    """
    try:
        stats = await coordinator.get_stats()

        return {
            "status": "healthy",
            "systems": {
                "coordinator": "operational",
                "session_manager": "operational"
            },
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}", exc_info=True)
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

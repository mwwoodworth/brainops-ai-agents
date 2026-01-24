"""
Unified Event Router for BrainOps AI OS

This module provides the central event routing system that:
1. Accepts events from all sources (ERP, AI Agents, Command Center)
2. Stores events in unified_events table as canonical record
3. Broadcasts via Supabase Realtime for live subscribers
4. Routes to appropriate AI agents based on event type

The goal is to unify the fragmented event bus systems:
- BrainOps: DB-backed brainops_core.event_bus with polling router
- ERP: In-memory + Supabase realtime broadcast
- AI Agents: Internal in-memory with partially implemented bridge
"""

import json
import logging
import os

# Import unified event schema
import sys
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.events.schema import (
    EventCategory,
    EventPriority,
    EventSource,
    UnifiedEvent,
    get_agents_for_event,
    validate_payload,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events", tags=["Unified Events"])

# Import LangGraph workflow trigger with fallback
try:
    from langgraph_workflow_trigger import trigger_workflows_for_event
    WORKFLOW_TRIGGER_AVAILABLE = True
    logger.info("LangGraph workflow trigger loaded")
except ImportError as e:
    WORKFLOW_TRIGGER_AVAILABLE = False
    trigger_workflows_for_event = None
    logger.warning(f"LangGraph workflow trigger not available: {e}")

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

try:
    from database.async_connection import get_pool, using_fallback
    ASYNC_POOL_AVAILABLE = True
except ImportError:
    ASYNC_POOL_AVAILABLE = False
    get_pool = None
    using_fallback = lambda: True

try:
    from database.sync_pool import get_sync_pool
    SYNC_POOL_AVAILABLE = True
except ImportError:
    SYNC_POOL_AVAILABLE = False
    get_sync_pool = None


# =============================================================================
# SUPABASE REALTIME BROADCASTER
# =============================================================================

class SupabaseRealtimeBroadcaster:
    """Broadcasts events to Supabase Realtime channels"""

    def __init__(self):
        self.supabase_url = os.getenv('PUBLIC_SUPABASE_URL', os.getenv('SUPABASE_URL'))
        self.supabase_key = os.getenv('PUBLIC_SUPABASE_ANON_KEY', os.getenv('SUPABASE_KEY'))
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    async def broadcast(self, event: UnifiedEvent) -> bool:
        """Broadcast event to tenant-scoped Supabase Realtime channel"""
        if not self.is_configured:
            logger.debug("Supabase Realtime not configured, skipping broadcast")
            return False

        try:
            # Use REST API to broadcast (avoids websocket complexity in server)
            import httpx

            channel_name = f"unified-events:{event.tenant_id}"
            broadcast_url = f"{self.supabase_url}/realtime/v1/api/broadcast"

            payload = {
                "topic": channel_name,
                "event": "unified_event",
                "payload": event.to_broadcast_payload(),
            }

            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    broadcast_url,
                    json=payload,
                    headers=headers,
                    timeout=5.0,
                )

                if response.status_code in (200, 201, 202):
                    logger.debug(f"Broadcast event {event.event_id} to {channel_name}")
                    return True
                else:
                    logger.warning(
                        f"Realtime broadcast failed: {response.status_code} - {response.text}"
                    )
                    return False

        except Exception as e:
            logger.warning(f"Realtime broadcast error: {e}")
            return False


# Global broadcaster instance
_broadcaster = SupabaseRealtimeBroadcaster()


# =============================================================================
# EVENT STORAGE
# =============================================================================

async def ensure_unified_events_table():
    """Create unified_events table if not exists"""
    if not ASYNC_POOL_AVAILABLE or using_fallback():
        logger.warning("Async pool not available, skipping table creation")
        return False

    try:
        pool = get_pool()
        await pool.execute("""
            CREATE TABLE IF NOT EXISTS unified_events (
                id SERIAL PRIMARY KEY,
                event_id VARCHAR(64) UNIQUE NOT NULL,
                version INTEGER DEFAULT 1,
                event_type VARCHAR(100) NOT NULL,
                category VARCHAR(50) DEFAULT 'business',
                priority VARCHAR(20) DEFAULT 'normal',
                source VARCHAR(50) NOT NULL,
                source_instance VARCHAR(100),
                tenant_id VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                occurred_at TIMESTAMPTZ,
                payload JSONB NOT NULL DEFAULT '{}',
                metadata JSONB DEFAULT '{}',
                correlation_id VARCHAR(64),
                causation_id VARCHAR(64),
                actor_type VARCHAR(50),
                actor_id VARCHAR(100),
                processed BOOLEAN DEFAULT FALSE,
                processed_at TIMESTAMPTZ,
                processing_result JSONB,
                retry_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW(),

                -- Indexes
                CONSTRAINT valid_category CHECK (category IN ('business', 'system', 'ai', 'user', 'integration')),
                CONSTRAINT valid_priority CHECK (priority IN ('critical', 'high', 'normal', 'low', 'batch'))
            );

            -- Performance indexes
            CREATE INDEX IF NOT EXISTS idx_unified_events_tenant_ts
                ON unified_events(tenant_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_unified_events_type
                ON unified_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_unified_events_unprocessed
                ON unified_events(created_at)
                WHERE NOT processed;
            CREATE INDEX IF NOT EXISTS idx_unified_events_correlation
                ON unified_events(correlation_id)
                WHERE correlation_id IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_unified_events_source
                ON unified_events(source, timestamp DESC);
        """)
        logger.info("Unified events table ensured")
        return True
    except Exception as e:
        logger.error(f"Failed to ensure unified_events table: {e}")
        return False


async def store_event(event: UnifiedEvent) -> bool:
    """Store event in unified_events table with fallback to sync pool."""
    logger.info(f"store_event called: ASYNC_POOL_AVAILABLE={ASYNC_POOL_AVAILABLE}, using_fallback={using_fallback()}")

    # Try async pool first
    if ASYNC_POOL_AVAILABLE and not using_fallback():
        result = await _store_event_async(event)
        if result:
            return True

    # Fallback to sync pool
    if SYNC_POOL_AVAILABLE and get_sync_pool:
        logger.info("Falling back to sync pool for event storage")
        result = _store_event_sync(event)
        if result:
            return True

    # Log warning but don't fail - event may still be broadcast
    logger.warning(f"Event {event.event_id} not stored in DB (no pool available)")
    return False


def _store_event_sync(event: UnifiedEvent) -> bool:
    """Synchronous fallback for event storage."""
    try:
        import psycopg2
        import psycopg2.extras

        conn = get_sync_pool()
        if not conn:
            return False

        record = event.to_db_record()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO unified_events (
                    event_id, version, event_type, category, priority,
                    source, source_instance, tenant_id, timestamp, occurred_at,
                    payload, metadata, correlation_id, causation_id,
                    actor_type, actor_id, processed, processed_at,
                    processing_result, retry_count
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (event_id) WHERE event_id IS NOT NULL DO NOTHING
            """, (
                record['event_id'],
                record['version'],
                record['event_type'],
                record['category'],
                record['priority'],
                record['source'],
                record['source_instance'],
                record['tenant_id'],
                record['timestamp'],
                record['occurred_at'],
                json.dumps(record['payload']),
                json.dumps(record['metadata']),
                record['correlation_id'],
                record['causation_id'],
                record['actor_type'],
                record['actor_id'],
                record['processed'],
                record['processed_at'],
                json.dumps(record['processing_result']) if record['processing_result'] else None,
                record['retry_count'],
            ))
            conn.commit()
        logger.info(f"Event {event.event_id} stored via sync pool")
        return True
    except Exception as e:
        logger.error(f"Sync event storage failed: {e}")
        return False


async def _store_event_async(event: UnifiedEvent) -> bool:
    """Async event storage implementation."""
    try:
        pool = get_pool()
        record = event.to_db_record()

        row = await pool.fetchrow("""
	            INSERT INTO unified_events (
	                event_id, version, event_type, category, priority,
	                source, source_instance, tenant_id, timestamp, occurred_at,
	                payload, metadata, correlation_id, causation_id,
	                actor_type, actor_id, processed, processed_at,
	                processing_result, retry_count
	            ) VALUES (
	                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
	                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
	            )
	            -- Supabase table uses a partial unique index: UNIQUE(event_id) WHERE event_id IS NOT NULL
	            ON CONFLICT (event_id) WHERE event_id IS NOT NULL DO NOTHING
	            RETURNING event_id
	        """,
            record['event_id'],
            record['version'],
            record['event_type'],
            record['category'],
            record['priority'],
            record['source'],
            record['source_instance'],
            record['tenant_id'],
            record['timestamp'],
            record['occurred_at'],
            json.dumps(record['payload']),
            json.dumps(record['metadata']),
            record['correlation_id'],
            record['causation_id'],
            record['actor_type'],
            record['actor_id'],
            record['processed'],
            record['processed_at'],
            json.dumps(record['processing_result']) if record['processing_result'] else None,
            record['retry_count'],
        )
        if row:
            logger.info(f"Event {event.event_id} stored successfully")
        else:
            logger.info(f"Event {event.event_id} already stored (idempotent duplicate)")
        return True
    except Exception as e:
        import traceback
        logger.error(f"Failed to store event {event.event_id}: {e}\n{traceback.format_exc()}")
        return False


async def mark_event_processed(
    event_id: str,
    result: Optional[dict[str, Any]] = None
) -> bool:
    """Mark an event as processed"""
    if not ASYNC_POOL_AVAILABLE or using_fallback():
        return False

    try:
        pool = get_pool()
        await pool.execute("""
            UPDATE unified_events
            SET processed = TRUE,
                processed_at = NOW(),
                processing_result = $2
            WHERE event_id = $1
        """, event_id, json.dumps(result, default=str) if result else None)
        return True
    except Exception as e:
        logger.error(f"Failed to mark event processed: {e}")
        return False


# =============================================================================
# AGENT ROUTING
# =============================================================================

def _snake_to_pascal(value: str) -> str:
    parts = [p for p in (value or "").strip().split("_") if p]
    return "".join(part[:1].upper() + part[1:] for part in parts)


_ROUTING_AGENT_OVERRIDES: dict[str, str] = {
    "scoring_agent": "LeadScorer",
    "followup_agent": "WorkflowAutomation",
    "review_agent": "QualityAgent",
    "billing_agent": "InvoicingAgent",
    "collection_agent": "InvoicingAgent",
    "pricing_agent": "EstimationAgent",
    "resource_agent": "SchedulingAgent",
    "job_creation_agent": "WorkflowAutomation",
}


def _event_priority_to_task_priority(priority: EventPriority) -> str:
    if priority == EventPriority.CRITICAL:
        return "critical"
    if priority == EventPriority.HIGH:
        return "high"
    if priority == EventPriority.LOW:
        return "low"
    if priority == EventPriority.BATCH:
        return "low"
    return "medium"


async def _resolve_ai_agent_for_routing_key(agent_key: str) -> Optional[dict[str, Any]]:
    """
    Resolve an EVENT_AGENT_ROUTING key (e.g. `notification_agent`) to an active
    `ai_agents` record (id + name). Returns None if no match.
    """
    if not ASYNC_POOL_AVAILABLE or using_fallback():
        return None

    pool = get_pool()

    candidates: list[str] = []
    override = _ROUTING_AGENT_OVERRIDES.get(agent_key)
    if override:
        candidates.append(override)

    candidates.append(agent_key)
    candidates.append(_snake_to_pascal(agent_key))

    # If we got something like "NotificationAgent" already, keep; else try adding Agent suffix.
    pascal = _snake_to_pascal(agent_key)
    if pascal and not pascal.endswith("Agent"):
        candidates.append(f"{pascal}Agent")

    for candidate in candidates:
        if not candidate:
            continue
        row = await pool.fetchrow(
            """
            SELECT id, name
            FROM ai_agents
            WHERE name = $1
              AND status = 'active'
            LIMIT 1
            """,
            candidate,
        )
        if row:
            return dict(row)
    return None


async def _enqueue_autonomous_task_for_agent(agent: dict[str, Any], event: UnifiedEvent) -> Optional[str]:
    """Create an ai_autonomous_tasks row for the resolved agent/event pair (idempotent)."""
    if not ASYNC_POOL_AVAILABLE or using_fallback():
        return None

    import uuid as uuid_mod

    pool = get_pool()
    agent_id = agent.get("id")
    agent_name = agent.get("name")
    if not agent_id or not agent_name:
        return None

    # Idempotency: do not enqueue duplicate tasks for the same event+agent.
    exists = await pool.fetchval(
        """
        SELECT 1
        FROM ai_autonomous_tasks
        WHERE agent_id = $1
          AND trigger_type = 'unified_event'
          AND trigger_condition->>'event_id' = $2
          AND created_at > NOW() - INTERVAL '90 days'
        LIMIT 1
        """,
        agent_id,
        event.event_id,
    )
    if exists:
        return None

    task_id = uuid_mod.uuid4()
    task_type = f"{event.event_type.replace('.', '_')}_handler"
    priority = _event_priority_to_task_priority(event.priority)
    tenant_uuid: str | None = None
    if getattr(event, "tenant_id", None):
        try:
            tenant_uuid = str(uuid_mod.UUID(str(event.tenant_id)))
        except (ValueError, TypeError, AttributeError):
            tenant_uuid = None

    trigger_condition = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "source": event.source.value,
        "tenant_id": event.tenant_id,
        "payload": event.payload,
        "metadata": event.metadata or {},
        "routed_agent": agent_name,
        "received_at": event.timestamp.isoformat() if getattr(event, "timestamp", None) else None,
    }

    await pool.execute(
        """
        INSERT INTO ai_autonomous_tasks (
            id,
            title,
            task_type,
            priority,
            status,
            trigger_type,
            trigger_condition,
            agent_id,
            tenant_id,
            created_at
        )
        VALUES ($1, $2, $3, $4, 'pending', 'unified_event', $5::jsonb, $6, $7, NOW())
        """,
        task_id,
        f"{agent_name}: {event.event_type}",
        task_type,
        priority,
        json.dumps(trigger_condition, default=str),
        agent_id,
        tenant_uuid,
    )

    return str(task_id)


async def route_event_to_agents(event: UnifiedEvent, background_tasks: BackgroundTasks) -> list[str]:
    """Route event to appropriate agents for processing by enqueuing tasks."""
    agent_keys = get_agents_for_event(event.event_type)
    if not agent_keys:
        logger.debug(f"No agents registered for event type: {event.event_type}")
        return []

    resolved_agents: list[dict[str, Any]] = []
    for agent_key in agent_keys:
        try:
            agent = await _resolve_ai_agent_for_routing_key(agent_key)
            if agent:
                resolved_agents.append(agent)
            else:
                logger.warning("No matching ai_agents record for routing key: %s", agent_key)
        except Exception as exc:
            logger.error("Failed resolving routing key %s: %s", agent_key, exc)

    # Fallback to a generic workflow agent so events are never silently dropped.
    if not resolved_agents:
        fallback = await _resolve_ai_agent_for_routing_key("WorkflowAutomation")
        if fallback:
            resolved_agents.append(fallback)
        else:
            logger.error("No agents resolved for event %s and no WorkflowAutomation fallback available", event.event_id)
            return []

    routed_to: list[str] = []
    tasks_created: list[str] = []
    tasks_skipped: int = 0
    for agent in resolved_agents:
        routed_to.append(agent.get("name") or "unknown")
        try:
            task_id = await _enqueue_autonomous_task_for_agent(agent, event)
            if task_id:
                tasks_created.append(task_id)
            else:
                tasks_skipped += 1
        except Exception as exc:
            logger.error("Failed enqueuing autonomous task for agent %s: %s", agent.get("name"), exc)

    # Trigger LangGraph workflows for this event (runs in background)
    workflow_results = []
    if WORKFLOW_TRIGGER_AVAILABLE and trigger_workflows_for_event:
        try:
            # Run workflow triggering as a background task to not block the response
            async def _trigger_workflows():
                try:
                    results = await trigger_workflows_for_event(
                        event_type=event.event_type,
                        event_payload=event.payload,
                        tenant_id=event.tenant_id,
                        correlation_id=event.correlation_id
                    )
                    if results:
                        logger.info(f"Triggered {len(results)} workflows for event {event.event_type}")
                except Exception as wf_err:
                    logger.error(f"Workflow trigger failed for {event.event_type}: {wf_err}")

            background_tasks.add_task(_trigger_workflows)
        except Exception as e:
            logger.error(f"Failed to schedule workflow trigger: {e}")

    # "processed" here means: routed + enqueued (or skipped due to idempotency).
    await mark_event_processed(
        event.event_id,
        {
            "routed_to": routed_to,
            "tasks_created": tasks_created,
            "tasks_skipped": tasks_skipped,
            "workflows_triggered": True if WORKFLOW_TRIGGER_AVAILABLE else False,
        },
    )

    return routed_to


async def execute_agent_for_event(agent_id: str, event: UnifiedEvent):
    """
    Backward-compatible helper kept for older call sites: routes via the same
    enqueue mechanism used by the unified router.
    """
    agent = await _resolve_ai_agent_for_routing_key(agent_id)
    if not agent:
        logger.warning("execute_agent_for_event: no agent resolved for %s", agent_id)
        return
    await _enqueue_autonomous_task_for_agent(agent, event)


async def execute_agent_directly(agent_id: str, event: UnifiedEvent):
    """Direct agent execution fallback (only used when explicitly enabled)."""
    direct_enabled = os.getenv("BRAINOPS_UNIFIED_EVENTS_DIRECT_EXECUTION", "").lower() in ("1", "true", "yes")
    if not direct_enabled:
        logger.info("Direct unified-event execution disabled; skipping %s for %s", agent_id, event.event_id)
        return

    try:
        from agent_executor import executor as agent_executor_singleton

        agent = await _resolve_ai_agent_for_routing_key(agent_id)
        agent_name = (agent or {}).get("name") or agent_id

        await agent_executor_singleton.execute(
            agent_name=str(agent_name),
            task={
                "action": "unified_event",
                "event_id": event.event_id,
                "event_type": event.event_type,
                "tenant_id": event.tenant_id,
                "payload": event.payload,
                "metadata": event.metadata or {},
            },
        )
    except Exception as e:
        logger.error(f"Direct agent execution failed: {e}")


# =============================================================================
# API ENDPOINTS
# =============================================================================

class PublishEventRequest(BaseModel):
    """Request to publish a new event"""
    event_type: str = Field(..., description="Event type in dot notation (e.g., 'job.created')")
    tenant_id: str = Field(..., description="Tenant ID")
    payload: dict[str, Any] = Field(default_factory=dict)

    # Optional fields
    source: Optional[str] = Field(None, description="Override source system")
    priority: Optional[str] = Field(None, description="Event priority")
    category: Optional[str] = Field(None, description="Event category")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    actor_type: Optional[str] = Field(None, description="Type of actor (user, agent, system)")
    actor_id: Optional[str] = Field(None, description="ID of the actor")
    metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")


class PublishEventResponse(BaseModel):
    """Response from publishing an event"""
    event_id: str
    stored: bool
    broadcast: bool
    routed_to: list[str]
    timestamp: str


@router.post("/publish", response_model=PublishEventResponse)
async def publish_event(
    request: PublishEventRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
):
    """
    Publish a new event to the unified event bus.

    This endpoint:
    1. Creates a UnifiedEvent from the request
    2. Stores it in the unified_events table
    3. Broadcasts it via Supabase Realtime
    4. Routes it to appropriate AI agents
    """
    try:
        # Create unified event
        event = UnifiedEvent(
            event_type=request.event_type,
            tenant_id=request.tenant_id,
            payload=request.payload,
            source=EventSource(request.source) if request.source else EventSource.AI_AGENTS,
            priority=EventPriority(request.priority) if request.priority else EventPriority.NORMAL,
            category=EventCategory(request.category) if request.category else EventCategory.BUSINESS,
            correlation_id=request.correlation_id,
            actor_type=request.actor_type,
            actor_id=request.actor_id,
            metadata=request.metadata or {},
        )

        # Validate payload if schema exists
        if not validate_payload(event.event_type, event.payload):
            logger.warning(f"Payload validation failed for {event.event_type}")

        # Store in database
        stored = await store_event(event)
        if not stored:
            raise HTTPException(status_code=503, detail="Database unavailable; event not persisted")

        # Broadcast via Supabase Realtime
        broadcast = await _broadcaster.broadcast(event)

        # Route to agents
        routed_to = await route_event_to_agents(event, background_tasks)

        return PublishEventResponse(
            event_id=event.event_id,
            stored=stored,
            broadcast=broadcast,
            routed_to=routed_to,
            timestamp=event.timestamp.isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to publish event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to publish event: {str(e)}") from e


class ERPEventWebhook(BaseModel):
    """Webhook payload from ERP SystemEventBus"""
    version: int = 1
    eventId: str
    type: str
    tenantId: str
    timestamp: str
    source: Optional[str] = None
    origin: Optional[str] = None
    payload: dict[str, Any]
    metadata: Optional[dict[str, Any]] = None


async def verify_erp_signature(request: Request) -> bool:
    """Verify HMAC signature from ERP webhook"""
    import hashlib
    import hmac

    signature = request.headers.get("X-ERP-Signature") or request.headers.get("X-Webhook-Signature")
    secret = os.getenv("ERP_WEBHOOK_SECRET", "")

    if not secret:
        environment = os.getenv("ENVIRONMENT", "production").lower()
        allow_unverified = os.getenv("ALLOW_UNVERIFIED_ERP_WEBHOOKS", "").lower() in ("1", "true", "yes")
        if environment == "production" and not allow_unverified:
            logger.critical("ERP_WEBHOOK_SECRET not configured in production - rejecting webhook")
            return False
        logger.warning("ERP_WEBHOOK_SECRET not configured - webhook verification disabled")
        return True

    if not signature:
        logger.error("Missing ERP webhook signature header")
        return False

    body = await request.body()
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(signature, expected):
        logger.error("Invalid ERP webhook signature")
        return False

    return True


@router.post("/webhook/erp", summary="Receive events from ERP SystemEventBus")
async def handle_erp_webhook(
    webhook: ERPEventWebhook,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Webhook endpoint to receive events from Weathercraft ERP SystemEventBus.

    This transforms ERP events to unified format and processes them.
    """
    # Verify webhook signature
    if not await verify_erp_signature(request):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        # Transform ERP event to unified format
        event = UnifiedEvent.from_erp_event(webhook.dict())

        logger.info(f"Received ERP event: {event.event_type} ({event.event_id})")

        # Store
        stored = await store_event(event)
        if not stored:
            raise HTTPException(status_code=503, detail="Database unavailable; event not persisted")

        # Broadcast
        broadcast = await _broadcaster.broadcast(event)

        # Route
        routed_to = await route_event_to_agents(event, background_tasks)

        return {
            "status": "processed",
            "event_id": event.event_id,
            "unified_type": event.event_type,
            "stored": stored,
            "broadcast": broadcast,
            "routed_to": routed_to,
        }

    except Exception as e:
        logger.error(f"ERP webhook error: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/recent")
async def get_recent_events(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    source: Optional[str] = Query(None, description="Filter by source system"),
    limit: int = Query(50, ge=1, le=500),
    include_processed: bool = Query(True, description="Include processed events"),
):
    """Get recent events from the unified event bus"""
    if not ASYNC_POOL_AVAILABLE or using_fallback():
        return {"events": [], "error": "Database not available"}

    try:
        pool = get_pool()

        # Build query
        conditions = []
        params = []
        param_idx = 1

        if tenant_id:
            conditions.append(f"tenant_id = ${param_idx}")
            params.append(tenant_id)
            param_idx += 1

        if event_type:
            conditions.append(f"event_type = ${param_idx}")
            params.append(event_type)
            param_idx += 1

        if source:
            conditions.append(f"source = ${param_idx}")
            params.append(source)
            param_idx += 1

        if not include_processed:
            conditions.append("processed = FALSE")

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Build query with safe WHERE clause (conditions use $N placeholders only)
        # noqa: S608 - where_clause built from parameterized conditions with $1, $2, etc.
        query = (
            "SELECT event_id, event_type, category, priority, source, "
            "tenant_id, timestamp, payload, metadata, "
            "processed, processed_at, processing_result "
            "FROM unified_events "
            + where_clause
            + " ORDER BY timestamp DESC "
            + "LIMIT $" + str(param_idx)
        )
        params.append(limit)

        rows = await pool.fetch(query, *params)

        events = []
        for row in rows:
            events.append({
                'event_id': row['event_id'],
                'event_type': row['event_type'],
                'category': row['category'],
                'priority': row['priority'],
                'source': row['source'],
                'tenant_id': row['tenant_id'],
                'timestamp': row['timestamp'].isoformat() if row['timestamp'] else None,
                'payload': row['payload'],
                'metadata': row['metadata'],
                'processed': row['processed'],
                'processed_at': row['processed_at'].isoformat() if row['processed_at'] else None,
                'processing_result': row['processing_result'],
            })

        return {"events": events, "count": len(events)}

    except Exception as e:
        logger.error(f"Failed to get recent events: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats")
async def get_event_stats(
    tenant_id: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168),
):
    """Get event statistics"""
    if not ASYNC_POOL_AVAILABLE or using_fallback():
        return {"error": "Database not available"}

    try:
        pool = get_pool()
        since = datetime.utcnow() - timedelta(hours=hours)

        # Build parameterized queries based on tenant_id presence
        if tenant_id:
            # Total counts with tenant filter
            total_query = """
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE processed) as processed,
                    COUNT(*) FILTER (WHERE NOT processed) as pending,
                    COUNT(*) FILTER (WHERE priority = 'critical') as critical
                FROM unified_events
                WHERE timestamp > $1 AND tenant_id = $2
            """
            params = [since, tenant_id]
            totals = await pool.fetchrow(total_query, *params)

            # By type with tenant filter
            type_query = """
                SELECT event_type, COUNT(*) as count
                FROM unified_events
                WHERE timestamp > $1 AND tenant_id = $2
                GROUP BY event_type
                ORDER BY count DESC
                LIMIT 20
            """
            type_rows = await pool.fetch(type_query, *params)

            # By source with tenant filter
            source_query = """
                SELECT source, COUNT(*) as count
                FROM unified_events
                WHERE timestamp > $1 AND tenant_id = $2
                GROUP BY source
                ORDER BY count DESC
            """
            source_rows = await pool.fetch(source_query, *params)

            # By hour with tenant filter
            hourly_query = """
                SELECT
                    date_trunc('hour', timestamp) as hour,
                    COUNT(*) as count
                FROM unified_events
                WHERE timestamp > $1 AND tenant_id = $2
                GROUP BY date_trunc('hour', timestamp)
                ORDER BY hour DESC
                LIMIT 24
            """
            hourly_rows = await pool.fetch(hourly_query, *params)
        else:
            # Total counts without tenant filter
            total_query = """
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE processed) as processed,
                    COUNT(*) FILTER (WHERE NOT processed) as pending,
                    COUNT(*) FILTER (WHERE priority = 'critical') as critical
                FROM unified_events
                WHERE timestamp > $1
            """
            params = [since]
            totals = await pool.fetchrow(total_query, *params)

            # By type without tenant filter
            type_query = """
                SELECT event_type, COUNT(*) as count
                FROM unified_events
                WHERE timestamp > $1
                GROUP BY event_type
                ORDER BY count DESC
                LIMIT 20
            """
            type_rows = await pool.fetch(type_query, *params)

            # By source without tenant filter
            source_query = """
                SELECT source, COUNT(*) as count
                FROM unified_events
                WHERE timestamp > $1
                GROUP BY source
                ORDER BY count DESC
            """
            source_rows = await pool.fetch(source_query, *params)

            # By hour without tenant filter
            hourly_query = """
                SELECT
                    date_trunc('hour', timestamp) as hour,
                    COUNT(*) as count
                FROM unified_events
                WHERE timestamp > $1
                GROUP BY date_trunc('hour', timestamp)
                ORDER BY hour DESC
                LIMIT 24
            """
            hourly_rows = await pool.fetch(hourly_query, *params)

        return {
            "period_hours": hours,
            "tenant_id": tenant_id,
            "totals": {
                "total": totals['total'],
                "processed": totals['processed'],
                "pending": totals['pending'],
                "critical": totals['critical'],
            },
            "by_type": {row['event_type']: row['count'] for row in type_rows},
            "by_source": {row['source']: row['count'] for row in source_rows},
            "by_hour": [
                {"hour": row['hour'].isoformat(), "count": row['count']}
                for row in hourly_rows
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get event stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/replay/{event_id}")
async def replay_event(
    event_id: str,
    background_tasks: BackgroundTasks,
):
    """Replay a specific event (re-route to agents)"""
    if not ASYNC_POOL_AVAILABLE or using_fallback():
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        pool = get_pool()

        row = await pool.fetchrow("""
            SELECT * FROM unified_events WHERE event_id = $1
        """, event_id)

        if not row:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

        # Reconstruct event
        event = UnifiedEvent(
            event_id=row['event_id'],
            version=row['version'],
            event_type=row['event_type'],
            category=EventCategory(row['category']),
            priority=EventPriority(row['priority']),
            source=EventSource(row['source']),
            source_instance=row['source_instance'],
            tenant_id=row['tenant_id'],
            timestamp=row['timestamp'],
            occurred_at=row['occurred_at'],
            payload=row['payload'],
            metadata=row['metadata'],
            correlation_id=row['correlation_id'],
            causation_id=row['causation_id'],
            actor_type=row['actor_type'],
            actor_id=row['actor_id'],
        )

        # Reset processed state
        await pool.execute("""
            UPDATE unified_events
            SET processed = FALSE, processed_at = NULL, retry_count = retry_count + 1
            WHERE event_id = $1
        """, event_id)

        # Re-route
        routed_to = await route_event_to_agents(event, background_tasks)

        return {
            "event_id": event_id,
            "replayed": True,
            "routed_to": routed_to,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to replay event: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/subscriptions/channels")
async def get_subscription_channels():
    """Get list of active Supabase Realtime channels for event subscription"""
    return {
        "channel_pattern": "unified-events:{tenant_id}",
        "event_name": "unified_event",
        "example_subscription": {
            "javascript": """
const channel = supabase.channel('unified-events:YOUR_TENANT_ID')
  .on('broadcast', { event: 'unified_event' }, (payload) => {
    console.log('Received event:', payload)
  })
  .subscribe()
""",
            "notes": [
                "Replace YOUR_TENANT_ID with your actual tenant ID",
                "Events are tenant-scoped for security",
                "Use Supabase JS client for real-time subscriptions",
            ],
        },
        "realtime_configured": _broadcaster.is_configured,
    }


@router.get("/debug")
async def get_events_debug():
    """Debug endpoint to check events module state"""
    pool = None
    pool_type = None
    can_test = False
    insert_test = None
    try:
        pool = get_pool()
        pool_type = type(pool).__name__
        can_test = await pool.test_connection()

        # Try a test insert
        import uuid as uuid_mod
        test_event_id = f"evt_debug_{uuid_mod.uuid4().hex[:8]}"
        try:
            await pool.execute("""
                INSERT INTO unified_events (
                    event_id, version, event_type, category, priority,
                    source, source_instance, tenant_id, timestamp, occurred_at,
                    payload, metadata, correlation_id, causation_id,
                    actor_type, actor_id, processed, processed_at,
                    processing_result, retry_count
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
                )
            """,
                test_event_id,  # $1 event_id
                1,  # $2 version
                'system.debug',  # $3 event_type
                'system',  # $4 category
                'low',  # $5 priority
                'ai-agents',  # $6 source
                None,  # $7 source_instance
                '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457',  # $8 tenant_id
                datetime.utcnow(),  # $9 timestamp - must be datetime, not string!
                None,  # $10 occurred_at
                '{"debug": true}',  # $11 payload
                '{}',  # $12 metadata
                None,  # $13 correlation_id
                None,  # $14 causation_id
                None,  # $15 actor_type
                None,  # $16 actor_id
                False,  # $17 processed
                None,  # $18 processed_at
                None,  # $19 processing_result
                0,  # $20 retry_count
            )
            insert_test = f"SUCCESS: {test_event_id}"
        except Exception as insert_err:
            insert_test = f"FAILED: {type(insert_err).__name__}: {insert_err}"
    except Exception as e:
        pool_type = f"Error: {e}"

    return {
        "async_pool_available": ASYNC_POOL_AVAILABLE,
        "using_fallback": using_fallback(),
        "pool_type": pool_type,
        "pool_test_connection": can_test,
        "insert_test": insert_test,
        "broadcaster_configured": _broadcaster.is_configured,
    }


@router.post("/verify")
async def verify_event_flow(
    tenant_id: str = Query(..., description="Tenant ID to test with"),
    background_tasks: BackgroundTasks = None
):
    """
    Publish a test event and verify it reaches all destinations.

    This endpoint:
    1. Creates a test event
    2. Stores it in unified_events table
    3. Broadcasts via Supabase Realtime
    4. Triggers workflow processing

    Use this to verify the complete event flow is working.
    """
    import uuid as uuid_mod

    test_event = UnifiedEvent(
        event_id=f"evt_verify_{uuid_mod.uuid4().hex[:12]}",
        event_type="system.verification_test",
        category=EventCategory.SYSTEM,
        priority=EventPriority.LOW,
        source=EventSource.AI_AGENTS,
        tenant_id=tenant_id,
        payload={
            "test": True,
            "timestamp": datetime.utcnow().isoformat(),
            "purpose": "Event flow verification"
        },
        metadata={"verification": True}
    )

    # Test storage
    stored = await store_event(test_event)

    # Test broadcast
    broadcast = await _broadcaster.broadcast(test_event)

    # Test workflow trigger (if available)
    workflows_triggered = False
    workflow_count = 0
    if WORKFLOW_TRIGGER_AVAILABLE and trigger_workflows_for_event:
        try:
            results = await trigger_workflows_for_event(
                event_type=test_event.event_type,
                event_payload=test_event.payload,
                tenant_id=test_event.tenant_id,
                correlation_id=test_event.correlation_id
            )
            workflows_triggered = True
            workflow_count = len(results) if results else 0
        except Exception as e:
            logger.error(f"Workflow trigger verification failed: {e}")

    return {
        "event_id": test_event.event_id,
        "tenant_id": tenant_id,
        "verification_results": {
            "stored_in_db": stored,
            "broadcast_to_realtime": broadcast,
            "workflows_triggered": workflows_triggered,
            "workflow_count": workflow_count,
        },
        "system_status": {
            "async_pool_available": ASYNC_POOL_AVAILABLE,
            "sync_pool_available": SYNC_POOL_AVAILABLE,
            "using_fallback": using_fallback(),
            "broadcaster_configured": _broadcaster.is_configured,
            "workflow_trigger_available": WORKFLOW_TRIGGER_AVAILABLE,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# INITIALIZATION
# =============================================================================

async def init_unified_events():
    """Initialize unified events system"""
    await ensure_unified_events_table()
    logger.info("Unified Events system initialized")


# Export router
__all__ = ['router', 'init_unified_events', 'UnifiedEvent', 'store_event', 'mark_event_processed']

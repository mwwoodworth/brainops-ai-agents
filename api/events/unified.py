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
    """Store event in unified_events table"""
    logger.info(f"store_event called: ASYNC_POOL_AVAILABLE={ASYNC_POOL_AVAILABLE}, using_fallback={using_fallback()}")
    if not ASYNC_POOL_AVAILABLE:
        logger.warning("ASYNC_POOL_AVAILABLE is False, event not stored in DB")
        return False
    if using_fallback():
        logger.warning("Database pool is using fallback, event not stored in DB")
        return False

    try:
        pool = get_pool()
        record = event.to_db_record()

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
            ON CONFLICT (event_id) DO UPDATE SET
                retry_count = unified_events.retry_count + 1,
                processed = FALSE,
                processed_at = NULL
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
        logger.info(f"Event {event.event_id} stored successfully")
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
        """, event_id, json.dumps(result) if result else None)
        return True
    except Exception as e:
        logger.error(f"Failed to mark event processed: {e}")
        return False


# =============================================================================
# AGENT ROUTING
# =============================================================================

async def route_event_to_agents(event: UnifiedEvent, background_tasks: BackgroundTasks) -> list[str]:
    """Route event to appropriate agents for processing"""
    agents = get_agents_for_event(event.event_type)

    if not agents:
        logger.debug(f"No agents registered for event type: {event.event_type}")
        return []

    routed = []
    for agent_id in agents:
        try:
            # Add agent execution to background tasks
            background_tasks.add_task(
                execute_agent_for_event,
                agent_id,
                event
            )
            routed.append(agent_id)
            logger.info(f"Routed event {event.event_id} to agent {agent_id}")
        except Exception as e:
            logger.error(f"Failed to route to agent {agent_id}: {e}")

    return routed


async def execute_agent_for_event(agent_id: str, event: UnifiedEvent):
    """Execute an agent with the given event"""
    try:
        # Import agent execution machinery
        # This integrates with existing agent infrastructure
        logger.info(f"Executing agent {agent_id} for event {event.event_id}")

        # Try to use the intelligent task orchestrator
        try:
            from intelligent_task_orchestrator import get_task_orchestrator
            orchestrator = get_task_orchestrator()

            await orchestrator.submit_task(
                title=f"Process {event.event_type} event",
                task_type=agent_id.replace('_agent', ''),
                payload={
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'tenant_id': event.tenant_id,
                    'payload': event.payload,
                },
                priority=80 if event.priority == EventPriority.CRITICAL else
                        70 if event.priority == EventPriority.HIGH else
                        50 if event.priority == EventPriority.NORMAL else 30,
            )
            logger.info(f"Submitted task to orchestrator for {agent_id}")
        except Exception as e:
            logger.warning(f"Task orchestrator not available: {e}")

            # Fallback: Direct agent execution for known agents
            await execute_agent_directly(agent_id, event)

    except Exception as e:
        logger.error(f"Agent execution failed for {agent_id}: {e}")


async def execute_agent_directly(agent_id: str, event: UnifiedEvent):
    """Direct agent execution fallback"""
    try:
        # Map agent IDs to actual implementations
        agent_map = {
            'customer_success_agent': 'customer_success_agent.CustomerSuccessAgent',
            'revenue_agent': 'revenue_generation_system.RevenueGenerationSystem',
            'followup_agent': 'intelligent_followup_system.IntelligentFollowupSystem',
            'scheduling_agent': 'scheduling_agent.SchedulingAgent',
        }

        if agent_id in agent_map:
            module_path, class_name = agent_map[agent_id].rsplit('.', 1)
            # Dynamic import would go here
            logger.info(f"Would execute {agent_map[agent_id]} for {event.event_type}")
        else:
            logger.debug(f"No direct implementation for agent: {agent_id}")

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
        logger.warning("ERP_WEBHOOK_SECRET not configured - webhook verification disabled")
        return True  # Allow in dev mode if no secret configured

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
        # Return 200 to prevent ERP from retrying indefinitely
        return {
            "status": "error",
            "error": str(e),
            "event_id": webhook.eventId,
        }


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
    try:
        pool = get_pool()
        pool_type = type(pool).__name__
        can_test = await pool.test_connection()
    except Exception as e:
        pool_type = f"Error: {e}"

    return {
        "async_pool_available": ASYNC_POOL_AVAILABLE,
        "using_fallback": using_fallback(),
        "pool_type": pool_type,
        "pool_test_connection": can_test,
        "broadcaster_configured": _broadcaster.is_configured,
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

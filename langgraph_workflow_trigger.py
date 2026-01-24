#!/usr/bin/env python3
"""
LangGraph Workflow Trigger Service
Connects unified events to LangGraph workflow execution
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

# Singleton instance
_workflow_trigger: Optional['LangGraphWorkflowTrigger'] = None

# Database configuration
def _get_database_url() -> str:
    """Get database URL from environment."""
    return os.getenv('DATABASE_URL', '')


class LangGraphWorkflowTrigger:
    """
    Triggers LangGraph workflows based on unified events.

    This service bridges the gap between the unified event system
    and the LangGraph workflow execution engine.
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self._orchestrator = None

    async def initialize(self) -> bool:
        """Initialize database connection and orchestrator."""
        if self._initialized:
            return True

        try:
            database_url = _get_database_url()
            if not database_url:
                logger.error("DATABASE_URL not configured for workflow trigger")
                return False

            self._pool = await asyncpg.create_pool(
                database_url,
                min_size=1,
                max_size=3,
                command_timeout=30,
                ssl='require'
            )

            # Lazy import to avoid circular dependencies
            try:
                from langgraph_orchestrator import LangGraphOrchestrator
                self._orchestrator = LangGraphOrchestrator()
                logger.info("LangGraphOrchestrator loaded for workflow execution")
            except ImportError as e:
                logger.warning(f"LangGraphOrchestrator not available: {e}")
                self._orchestrator = None

            # Ensure workflow_event_triggers table exists
            await self._ensure_triggers_table()

            self._initialized = True
            logger.info("LangGraphWorkflowTrigger initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LangGraphWorkflowTrigger: {e}")
            return False

    async def _ensure_triggers_table(self):
        """Create workflow_event_triggers table if it doesn't exist."""
        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_event_triggers (
                    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    workflow_id UUID NOT NULL,
                    workflow_name VARCHAR(255),
                    priority INTEGER DEFAULT 50,
                    conditions JSONB DEFAULT '{}',
                    enabled BOOLEAN DEFAULT true,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_event_triggers_event_type
                    ON workflow_event_triggers(event_type) WHERE enabled = true;

                CREATE INDEX IF NOT EXISTS idx_workflow_event_triggers_workflow
                    ON workflow_event_triggers(workflow_id);
            """)
            logger.info("workflow_event_triggers table ready")

    async def trigger_for_event(self, event_type: str, event_payload: dict,
                                 tenant_id: str, correlation_id: Optional[str] = None) -> list[dict]:
        """
        Find and execute matching workflows for an event.

        Args:
            event_type: The type of event (e.g., 'job.created', 'estimate.accepted')
            event_payload: The event payload data
            tenant_id: The tenant ID
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of execution results
        """
        if not await self.initialize():
            logger.warning("Workflow trigger not initialized, skipping")
            return []

        results = []

        try:
            # Get matching workflows for this event type
            workflows = await self._get_workflows_for_event(event_type, tenant_id)

            if not workflows:
                logger.debug(f"No workflows registered for event type: {event_type}")
                return []

            logger.info(f"Found {len(workflows)} workflows to trigger for event: {event_type}")

            for workflow in workflows:
                try:
                    result = await self._execute_workflow(
                        workflow=workflow,
                        event_type=event_type,
                        event_payload=event_payload,
                        tenant_id=tenant_id,
                        correlation_id=correlation_id
                    )
                    results.append(result)

                    # Update workflow stats
                    await self._update_workflow_stats(
                        workflow_id=workflow['workflow_id'],
                        success=result.get('success', False),
                        execution_time_ms=result.get('execution_time_ms', 0)
                    )

                except Exception as e:
                    logger.error(f"Failed to execute workflow {workflow.get('workflow_name')}: {e}")
                    results.append({
                        'workflow_id': str(workflow['workflow_id']),
                        'workflow_name': workflow.get('workflow_name'),
                        'success': False,
                        'error': str(e)
                    })

        except Exception as e:
            logger.error(f"Error triggering workflows for event {event_type}: {e}")

        return results

    async def _get_workflows_for_event(self, event_type: str, tenant_id: str) -> list[dict]:
        """Get workflows that should be triggered for an event type."""
        if not self._pool:
            return []

        async with self._pool.acquire() as conn:
            # First check if there are explicit triggers
            rows = await conn.fetch("""
                SELECT
                    wt.id as trigger_id,
                    wt.workflow_id,
                    wt.workflow_name,
                    wt.priority,
                    wt.conditions,
                    w.name as actual_workflow_name,
                    w.graph_definition,
                    w.config
                FROM workflow_event_triggers wt
                LEFT JOIN langgraph_workflows w ON wt.workflow_id = w.id
                WHERE wt.event_type = $1
                  AND wt.enabled = true
                ORDER BY wt.priority DESC
            """, event_type)

            if rows:
                return [dict(r) for r in rows]

            # Fallback: Find workflows by name pattern matching
            # This allows workflows named like 'job_created_workflow' to match 'job.created' events
            event_base = event_type.replace('.', '_').lower()
            rows = await conn.fetch("""
                SELECT
                    id as workflow_id,
                    name as workflow_name,
                    graph_definition,
                    config,
                    50 as priority
                FROM langgraph_workflows
                WHERE (
                    LOWER(name) LIKE $1
                    OR LOWER(name) LIKE $2
                )
                AND status = 'active'
                AND (tenant_id = $3 OR tenant_id = '00000000-0000-0000-0000-000000000001'::uuid)
                ORDER BY execution_count DESC
                LIMIT 3
            """, f"%{event_base}%", f"%{event_type.split('.')[0]}%", tenant_id)

            return [dict(r) for r in rows]

    async def _execute_workflow(self, workflow: dict, event_type: str,
                                 event_payload: dict, tenant_id: str,
                                 correlation_id: Optional[str] = None) -> dict:
        """Execute a single workflow."""
        start_time = datetime.now(timezone.utc)

        workflow_id = str(workflow.get('workflow_id', ''))
        workflow_name = workflow.get('workflow_name') or workflow.get('actual_workflow_name', 'unknown')

        try:
            if self._orchestrator:
                # Use LangGraph orchestrator for real execution
                prompt = f"""
                Process this event and take appropriate action:

                Event Type: {event_type}
                Event Data: {json.dumps(event_payload, default=str)}
                Tenant: {tenant_id}
                Workflow: {workflow_name}

                Based on the workflow definition and event data, determine and execute
                the appropriate actions. Respond with a summary of what was done.
                """

                result = await self._orchestrator.execute(
                    agent_name=workflow_name,
                    prompt=prompt,
                    tenant_id=tenant_id,
                    context={
                        'event_type': event_type,
                        'event_payload': event_payload,
                        'workflow_id': workflow_id,
                        'correlation_id': correlation_id
                    }
                )

                execution_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

                return {
                    'workflow_id': workflow_id,
                    'workflow_name': workflow_name,
                    'success': True,
                    'execution_time_ms': execution_time_ms,
                    'result': result
                }
            else:
                # Fallback: Just log and record the trigger
                logger.info(f"Workflow {workflow_name} triggered for {event_type} (orchestrator not available)")

                execution_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

                return {
                    'workflow_id': workflow_id,
                    'workflow_name': workflow_name,
                    'success': True,
                    'execution_time_ms': execution_time_ms,
                    'note': 'Trigger recorded, orchestrator not available'
                }

        except Exception as e:
            execution_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            logger.error(f"Workflow execution failed for {workflow_name}: {e}")

            return {
                'workflow_id': workflow_id,
                'workflow_name': workflow_name,
                'success': False,
                'execution_time_ms': execution_time_ms,
                'error': str(e)
            }

    async def _update_workflow_stats(self, workflow_id: str, success: bool,
                                      execution_time_ms: int) -> None:
        """Update execution_count and success_rate in langgraph_workflows."""
        if not self._pool or not workflow_id:
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    UPDATE langgraph_workflows
                    SET
                        execution_count = COALESCE(execution_count, 0) + 1,
                        success_rate = (
                            (COALESCE(success_rate, 0) * COALESCE(execution_count, 0)) +
                            CASE WHEN $1 THEN 1.0 ELSE 0.0 END
                        ) / (COALESCE(execution_count, 0) + 1),
                        avg_execution_time_ms = (
                            (COALESCE(avg_execution_time_ms, 0) * COALESCE(execution_count, 0)) + $2
                        ) / (COALESCE(execution_count, 0) + 1),
                        updated_at = NOW()
                    WHERE id = $3::uuid
                """, success, execution_time_ms, workflow_id)

                logger.debug(f"Updated workflow stats for {workflow_id}: success={success}, time={execution_time_ms}ms")

        except Exception as e:
            logger.error(f"Failed to update workflow stats: {e}")

    async def register_trigger(self, event_type: str, workflow_id: str,
                                workflow_name: str = None, priority: int = 50,
                                conditions: dict = None) -> Optional[str]:
        """
        Register a new workflow trigger.

        Args:
            event_type: The event type to trigger on
            workflow_id: The workflow to execute
            workflow_name: Optional display name
            priority: Execution priority (higher = first)
            conditions: Optional JSON conditions for filtering

        Returns:
            The trigger ID if successful
        """
        if not await self.initialize():
            return None

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO workflow_event_triggers
                    (event_type, workflow_id, workflow_name, priority, conditions)
                    VALUES ($1, $2::uuid, $3, $4, $5)
                    RETURNING id
                """, event_type, workflow_id, workflow_name, priority,
                    json.dumps(conditions or {}))

                trigger_id = str(row['id']) if row else None
                logger.info(f"Registered workflow trigger: {event_type} -> {workflow_name}")
                return trigger_id

        except Exception as e:
            logger.error(f"Failed to register workflow trigger: {e}")
            return None

    async def close(self):
        """Close database connections."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False


def get_workflow_trigger() -> LangGraphWorkflowTrigger:
    """Get singleton workflow trigger instance."""
    global _workflow_trigger
    if _workflow_trigger is None:
        _workflow_trigger = LangGraphWorkflowTrigger()
    return _workflow_trigger


async def trigger_workflows_for_event(event_type: str, event_payload: dict,
                                       tenant_id: str, correlation_id: str = None) -> list[dict]:
    """
    Convenience function to trigger workflows for an event.

    This is the main entry point called from unified.py route_event_to_agents.
    """
    trigger = get_workflow_trigger()
    return await trigger.trigger_for_event(
        event_type=event_type,
        event_payload=event_payload,
        tenant_id=tenant_id,
        correlation_id=correlation_id
    )

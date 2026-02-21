"""
Scheduler-related database queries.

Extracted from inline SQL in app.py scheduler routes during Phase 2 Wave 2B.
Provides a clean interface for email queue stats, agent schedule CRUD,
and active agent lookups.
"""

import logging

from database.async_connection import get_pool

logger = logging.getLogger(__name__)


async def fetch_email_queue_counts() -> dict:
    """Return email queue status counts from ai_email_queue."""
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE status = 'queued') as queued,
            COUNT(*) FILTER (WHERE status = 'processing') as processing,
            COUNT(*) FILTER (WHERE status = 'sent') as sent,
            COUNT(*) FILTER (WHERE status = 'failed') as failed,
            COUNT(*) FILTER (WHERE status = 'skipped') as skipped,
            COUNT(*) as total
        FROM ai_email_queue
        """
    )
    return dict(row) if row else {}


async def fetch_active_agents(pool) -> list:
    """Return all active agents from ai_agents."""
    return await pool.fetch(
        "SELECT id, name, type, category FROM ai_agents WHERE status = 'active'"
    )


async def fetch_scheduled_agent_ids(pool) -> set[str]:
    """Return set of agent IDs that have an enabled schedule."""
    rows = await pool.fetch("SELECT agent_id FROM agent_schedules WHERE enabled = true")
    return {str(row["agent_id"]) for row in rows}


async def insert_agent_schedule(pool, agent_id: str, frequency_minutes: int) -> None:
    """Insert a new agent schedule row."""
    await pool.execute(
        """
        INSERT INTO agent_schedules (id, agent_id, frequency_minutes, enabled, created_at)
        VALUES (gen_random_uuid(), $1, $2, true, NOW())
        """,
        agent_id,
        frequency_minutes,
    )


async def upsert_agent_schedule(
    pool,
    agent_id: str,
    frequency_minutes: int,
    enabled: bool,
    schedule_id: str,
) -> dict | None:
    """
    Create or update an agent schedule.

    Returns dict with keys: action, schedule_id, agent_name.
    Returns None if the agent doesn't exist.
    """
    # Verify agent exists
    agent = await pool.fetchrow("SELECT id, name FROM agents WHERE id = $1", agent_id)
    if not agent:
        return None

    # Check if schedule already exists
    existing = await pool.fetchrow("SELECT id FROM agent_schedules WHERE agent_id = $1", agent_id)

    if existing:
        await pool.execute(
            """
            UPDATE agent_schedules
            SET frequency_minutes = $1, enabled = $2, updated_at = NOW()
            WHERE agent_id = $3
            """,
            frequency_minutes,
            enabled,
            agent_id,
        )
        return {
            "action": "updated",
            "schedule_id": str(existing["id"]),
            "agent_name": agent["name"],
        }
    else:
        await pool.execute(
            """
            INSERT INTO agent_schedules (id, agent_id, frequency_minutes, enabled, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            """,
            schedule_id,
            agent_id,
            frequency_minutes,
            enabled,
        )
        return {
            "action": "created",
            "schedule_id": schedule_id,
            "agent_name": agent["name"],
        }

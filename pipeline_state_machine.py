#!/usr/bin/env python3
"""
Pipeline State Machine
======================
THE canonical source of truth for lead/deal progression.

All state transitions are recorded in an append-only ledger (revenue_actions).
Truth is computed from ledger facts, not derived guesses.

Part of Revenue Perfection Session - Total Completion Protocol.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)


class PipelineState(str, Enum):
    """
    Canonical pipeline states for lead progression.
    Each state has clear entry/exit criteria.
    """
    # Discovery & Classification
    NEW_REAL = "new_real"              # Verified real lead, not test/demo

    # Enrichment
    ENRICHED = "enriched"              # Has company info, decision maker, pain points

    # Outreach Preparation
    CONTACT_READY = "contact_ready"    # All info available for outreach

    # Outreach Execution
    OUTREACH_PENDING_APPROVAL = "outreach_pending_approval"  # Draft needs human approval
    OUTREACH_SENT = "outreach_sent"    # First outreach delivered

    # Response Handling
    REPLIED = "replied"                # Prospect responded
    MEETING_BOOKED = "meeting_booked"  # Call/meeting scheduled

    # Proposal Stage
    PROPOSAL_DRAFTED = "proposal_drafted"        # Proposal created
    PROPOSAL_APPROVED = "proposal_approved"      # Proposal approved by human
    PROPOSAL_SENT = "proposal_sent"              # Proposal delivered to prospect

    # Closing
    WON_INVOICE_PENDING = "won_invoice_pending"  # Deal won, invoice not sent
    INVOICED = "invoiced"              # Invoice sent
    PAID = "paid"                      # Payment received - REAL REVENUE

    # Terminal States
    LOST = "lost"                      # Deal lost


# Valid state transitions
VALID_TRANSITIONS = {
    PipelineState.NEW_REAL: [PipelineState.ENRICHED, PipelineState.LOST],
    PipelineState.ENRICHED: [PipelineState.CONTACT_READY, PipelineState.LOST],
    PipelineState.CONTACT_READY: [PipelineState.OUTREACH_PENDING_APPROVAL, PipelineState.LOST],
    PipelineState.OUTREACH_PENDING_APPROVAL: [PipelineState.OUTREACH_SENT, PipelineState.CONTACT_READY, PipelineState.LOST],
    PipelineState.OUTREACH_SENT: [PipelineState.REPLIED, PipelineState.OUTREACH_PENDING_APPROVAL, PipelineState.LOST],  # Can retry
    PipelineState.REPLIED: [PipelineState.MEETING_BOOKED, PipelineState.PROPOSAL_DRAFTED, PipelineState.LOST],
    PipelineState.MEETING_BOOKED: [PipelineState.PROPOSAL_DRAFTED, PipelineState.LOST],
    PipelineState.PROPOSAL_DRAFTED: [PipelineState.PROPOSAL_APPROVED, PipelineState.LOST],
    PipelineState.PROPOSAL_APPROVED: [PipelineState.PROPOSAL_SENT, PipelineState.PROPOSAL_DRAFTED, PipelineState.LOST],
    PipelineState.PROPOSAL_SENT: [PipelineState.WON_INVOICE_PENDING, PipelineState.PROPOSAL_DRAFTED, PipelineState.LOST],
    PipelineState.WON_INVOICE_PENDING: [PipelineState.INVOICED, PipelineState.LOST],
    PipelineState.INVOICED: [PipelineState.PAID, PipelineState.LOST],
    PipelineState.PAID: [],  # Terminal success
    PipelineState.LOST: [],  # Terminal failure
}


# Map legacy stages to new states
LEGACY_STAGE_MAP = {
    "new": PipelineState.NEW_REAL,
    "contacted": PipelineState.OUTREACH_SENT,
    "qualified": PipelineState.CONTACT_READY,
    "proposal_sent": PipelineState.PROPOSAL_SENT,
    "negotiating": PipelineState.MEETING_BOOKED,
    "won": PipelineState.PAID,
    "lost": PipelineState.LOST,
}


DEFAULT_TRANSITION_TRIGGERS: dict[PipelineState, set[str]] = {
    PipelineState.ENRICHED: {"enrichment_completed", "legacy_migration"},
    PipelineState.CONTACT_READY: {"lead_qualified", "batch_advancement", "legacy_migration"},
    PipelineState.OUTREACH_PENDING_APPROVAL: {"outreach_draft_submitted", "retry_outreach"},
    PipelineState.OUTREACH_SENT: {"outreach_sent", "resend_outreach"},
    PipelineState.REPLIED: {"reply_received", "meeting_request"},
    PipelineState.MEETING_BOOKED: {"meeting_scheduled", "discovery_call_booked"},
    PipelineState.PROPOSAL_DRAFTED: {"proposal_drafted", "proposal_revision"},
    PipelineState.PROPOSAL_APPROVED: {"proposal_approved"},
    PipelineState.PROPOSAL_SENT: {"proposal_sent"},
    PipelineState.WON_INVOICE_PENDING: {"deal_won"},
    PipelineState.INVOICED: {"invoice_created", "invoice_sent"},
    PipelineState.PAID: {"payment_received_manual", "payment_received_stripe", "payment_received_stripe_invoice"},
    PipelineState.LOST: {"deal_lost", "disqualified"},
}


@dataclass
class StateTransition:
    """Represents a state transition event."""
    id: str
    lead_id: str
    from_state: Optional[str]
    to_state: str
    trigger: str  # What caused this transition
    actor: str  # Who/what made this transition (human/agent)
    metadata: dict[str, Any]
    created_at: datetime


class PipelineStateMachine:
    """
    Manages lead state transitions with full audit trail.

    All state changes are recorded in revenue_actions table.
    Truth is computed from ledger, not from current state alone.
    """

    def __init__(self):
        self._pool = None

    def _get_pool(self):
        """Get database pool."""
        try:
            from database.async_connection import get_pool
            return get_pool()
        except Exception as e:
            logger.error(f"Failed to get database pool: {e}")
            return None

    def _validate_trigger(self, to_state: PipelineState, trigger: str, force: bool) -> tuple[bool, str]:
        """Validate trigger semantics for state transition integrity."""
        if force:
            return True, "Forced transition"

        strict = (os.getenv("PIPELINE_STRICT_TRIGGER_VALIDATION", "true") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not strict:
            return True, "Trigger validation disabled"

        allowed = DEFAULT_TRANSITION_TRIGGERS.get(to_state)
        if not allowed:
            return True, "No trigger constraints for target state"
        if trigger in allowed:
            return True, "Trigger accepted"
        return False, (
            f"Trigger '{trigger}' is not valid for state '{to_state.value}'. "
            f"Allowed: {sorted(allowed)}"
        )

    async def _record_audit_event(
        self,
        lead_id: str,
        action_data: dict[str, Any],
        success: bool,
        actor: str,
        error: Optional[str] = None,
    ) -> None:
        """Write additional audit event for transition attempts (success and failure)."""
        pool = self._get_pool()
        if not pool:
            return
        try:
            payload = dict(action_data)
            if error:
                payload["error"] = error
            await pool.execute(
                """
                INSERT INTO revenue_actions (
                    id, lead_id, action_type, action_data, result, success, created_at, executed_by
                ) VALUES ($1, $2, 'state_transition_audit', $3, $4, $5, $6, $7)
                """,
                uuid.uuid4(),
                uuid.UUID(lead_id) if isinstance(lead_id, str) else lead_id,
                payload,
                {"status": "ok" if success else "failed"},
                success,
                datetime.now(timezone.utc),
                actor,
            )
        except Exception as exc:
            logger.debug("State transition audit logging skipped: %s", exc)

    async def get_lead_state(self, lead_id: str) -> Optional[str]:
        """Get current state of a lead from ledger."""
        pool = self._get_pool()
        if not pool:
            return None

        # Get most recent state from ledger
        result = await pool.fetchrow("""
            SELECT action_data->>'to_state' as current_state
            FROM revenue_actions
            WHERE lead_id = $1
            AND action_type = 'state_transition'
            ORDER BY created_at DESC
            LIMIT 1
        """, uuid.UUID(lead_id) if isinstance(lead_id, str) else lead_id)

        if result:
            return result['current_state']

        # Fallback to legacy stage from revenue_leads
        legacy = await pool.fetchrow("""
            SELECT stage FROM revenue_leads WHERE id = $1
        """, uuid.UUID(lead_id) if isinstance(lead_id, str) else lead_id)

        if legacy:
            return LEGACY_STAGE_MAP.get(legacy['stage'], PipelineState.NEW_REAL).value

        return None

    async def can_transition(self, lead_id: str, to_state: PipelineState) -> tuple[bool, str]:
        """Check if a state transition is valid."""
        current = await self.get_lead_state(lead_id)
        if not current:
            return False, "Lead not found"

        if current == to_state.value:
            return True, f"Lead already in state {to_state.value}"

        try:
            current_state = PipelineState(current)
        except ValueError:
            # Legacy state, allow transition
            return True, "Legacy state migration allowed"

        if current_state in {PipelineState.PAID, PipelineState.LOST}:
            return False, f"Cannot transition from terminal state {current_state.value}"

        valid_next = VALID_TRANSITIONS.get(current_state, [])
        if to_state in valid_next:
            return True, "Transition allowed"

        return False, f"Cannot transition from {current} to {to_state.value}"

    async def transition(
        self,
        lead_id: str,
        to_state: PipelineState,
        trigger: str,
        actor: str = "system",
        metadata: Optional[dict] = None,
        force: bool = False
    ) -> tuple[bool, str, Optional[StateTransition]]:
        """
        Execute a state transition with full audit trail.

        Args:
            lead_id: UUID of the lead
            to_state: Target state
            trigger: What caused this transition
            actor: Who/what made this transition
            metadata: Additional data to record
            force: Skip validation (for migrations)

        Returns:
            (success, message, transition_record)
        """
        pool = self._get_pool()
        if not pool:
            return False, "Database not available", None

        # Validate transition
        if not force:
            can_do, reason = await self.can_transition(lead_id, to_state)
            if not can_do:
                await self._record_audit_event(
                    lead_id=lead_id,
                    action_data={
                        "from_state": await self.get_lead_state(lead_id),
                        "to_state": to_state.value,
                        "trigger": trigger,
                        "metadata": metadata or {},
                    },
                    success=False,
                    actor=actor,
                    error=reason,
                )
                return False, reason, None

        current_state = await self.get_lead_state(lead_id)
        if current_state == to_state.value:
            # Idempotent: no-op transition still gets audited for traceability.
            await self._record_audit_event(
                lead_id=lead_id,
                action_data={
                    "from_state": current_state,
                    "to_state": to_state.value,
                    "trigger": trigger,
                    "metadata": metadata or {},
                    "idempotent": True,
                },
                success=True,
                actor=actor,
            )
            transition = StateTransition(
                id=str(uuid.uuid4()),
                lead_id=lead_id,
                from_state=current_state,
                to_state=to_state.value,
                trigger=trigger,
                actor=actor,
                metadata=metadata or {},
                created_at=datetime.now(timezone.utc),
            )
            return True, f"Already in {to_state.value}", transition

        trigger_ok, trigger_msg = self._validate_trigger(to_state, trigger, force=force)
        if not trigger_ok:
            await self._record_audit_event(
                lead_id=lead_id,
                action_data={
                    "from_state": current_state,
                    "to_state": to_state.value,
                    "trigger": trigger,
                    "metadata": metadata or {},
                },
                success=False,
                actor=actor,
                error=trigger_msg,
            )
            return False, trigger_msg, None

        transition_count = await pool.fetchval(
            """
            SELECT COUNT(*)
            FROM revenue_actions
            WHERE lead_id = $1
              AND action_type = 'state_transition'
            """,
            uuid.UUID(lead_id) if isinstance(lead_id, str) else lead_id,
        ) or 0
        transition_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Record in append-only ledger
        action_data = {
            "from_state": current_state,
            "to_state": to_state.value,
            "trigger": trigger,
            "sequence": int(transition_count) + 1,
            "metadata": metadata or {}
        }

        try:
            # Insert into revenue_actions (append-only ledger)
            await pool.execute("""
                INSERT INTO revenue_actions (id, lead_id, action_type, action_data, result, success, created_at, executed_by)
                VALUES ($1, $2, 'state_transition', $3, $4, true, $5, $6)
            """,
                uuid.UUID(transition_id),
                uuid.UUID(lead_id) if isinstance(lead_id, str) else lead_id,
                action_data,
                {"new_state": to_state.value},
                now,
                actor
            )

            # Update denormalized state in revenue_leads for backward compatibility
            # Map new state back to legacy stage for queries
            legacy_stage = self._to_legacy_stage(to_state)
            await pool.execute("""
                UPDATE revenue_leads
                SET stage = $1, updated_at = $2
                WHERE id = $3
            """, legacy_stage, now, uuid.UUID(lead_id) if isinstance(lead_id, str) else lead_id)

            logger.info(
                "Lead %s... transitioned: %s -> %s by %s",
                str(lead_id)[:8],
                current_state,
                to_state.value,
                actor,
            )
            await self._record_audit_event(
                lead_id=lead_id,
                action_data=action_data,
                success=True,
                actor=actor,
            )

            # Phase 2: Revenue reinforcement
            # Best-effort: when a lead reaches late-stage conversion states, ask the optimizer to recompile.
            try:
                if to_state in {PipelineState.WON_INVOICE_PENDING, PipelineState.INVOICED, PipelineState.PAID}:
                    from optimization.revenue_prompt_compile_queue import enqueue_revenue_prompt_compile_task

                    tenant_hint = None
                    if isinstance(metadata, dict):
                        tenant_hint = metadata.get("tenant_id")

                    priority = 95 if to_state == PipelineState.PAID else 85
                    await enqueue_revenue_prompt_compile_task(
                        pool=pool,
                        tenant_id=str(tenant_hint) if tenant_hint else None,
                        lead_id=str(lead_id),
                        reason=f"pipeline_state_transition:{to_state.value}",
                        priority=priority,
                        force=True,
                    )
            except Exception as exc:
                logger.debug("Revenue prompt compile enqueue skipped: %s", exc)

            transition = StateTransition(
                id=transition_id,
                lead_id=lead_id,
                from_state=current_state,
                to_state=to_state.value,
                trigger=trigger,
                actor=actor,
                metadata=metadata or {},
                created_at=now
            )

            return True, f"Transitioned to {to_state.value}", transition

        except Exception as e:
            logger.error(f"Transition failed: {e}")
            await self._record_audit_event(
                lead_id=lead_id,
                action_data=action_data,
                success=False,
                actor=actor,
                error=str(e),
            )
            return False, str(e), None

    def _to_legacy_stage(self, state: PipelineState) -> str:
        """Map new state to legacy stage for backward compatibility."""
        mapping = {
            PipelineState.NEW_REAL: "new",
            PipelineState.ENRICHED: "new",
            PipelineState.CONTACT_READY: "qualified",
            PipelineState.OUTREACH_PENDING_APPROVAL: "qualified",
            PipelineState.OUTREACH_SENT: "contacted",
            PipelineState.REPLIED: "contacted",
            PipelineState.MEETING_BOOKED: "negotiating",
            PipelineState.PROPOSAL_DRAFTED: "qualified",
            PipelineState.PROPOSAL_APPROVED: "qualified",
            PipelineState.PROPOSAL_SENT: "proposal_sent",
            PipelineState.WON_INVOICE_PENDING: "won",
            PipelineState.INVOICED: "won",
            PipelineState.PAID: "won",
            PipelineState.LOST: "lost",
        }
        return mapping.get(state, "new")

    async def get_lead_timeline(self, lead_id: str) -> list[dict]:
        """Get full state transition history for a lead."""
        pool = self._get_pool()
        if not pool:
            return []

        rows = await pool.fetch("""
            SELECT id, action_type, action_data, result, success, created_at, executed_by
            FROM revenue_actions
            WHERE lead_id = $1
            ORDER BY created_at ASC
        """, uuid.UUID(lead_id) if isinstance(lead_id, str) else lead_id)

        return [
            {
                "id": str(r["id"]),
                "action_type": r["action_type"],
                "action_data": r["action_data"],
                "result": r["result"],
                "success": r["success"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "executed_by": r["executed_by"]
            }
            for r in rows
        ]

    async def get_pipeline_stats(self, real_only: bool = True) -> dict:
        """Get pipeline statistics from ledger facts."""
        pool = self._get_pool()
        if not pool:
            return {}

        # Test filter for real leads
        test_filter = ""
        if real_only:
            test_filter = """
                AND rl.email NOT ILIKE '%test%'
                AND rl.email NOT ILIKE '%example%'
                AND rl.email NOT ILIKE '%demo%'
                AND rl.email NOT ILIKE '%sample%'
                AND rl.email NOT ILIKE '%fake%'
            """

        # Get current state distribution from ledger
        query = f"""
            WITH latest_states AS (
                SELECT DISTINCT ON (ra.lead_id)
                    ra.lead_id,
                    ra.action_data->>'to_state' as current_state,
                    ra.created_at
                FROM revenue_actions ra
                JOIN revenue_leads rl ON ra.lead_id = rl.id
                WHERE ra.action_type = 'state_transition'
                {test_filter}
                ORDER BY ra.lead_id, ra.created_at DESC
            )
            SELECT current_state, COUNT(*) as count
            FROM latest_states
            GROUP BY current_state
        """

        rows = await pool.fetch(query)
        state_counts = {r["current_state"]: r["count"] for r in rows}

        # Get real revenue from PAID state
        paid_query = f"""
            SELECT COALESCE(SUM(rl.value_estimate), 0) as paid_revenue
            FROM revenue_actions ra
            JOIN revenue_leads rl ON ra.lead_id = rl.id
            WHERE ra.action_type = 'state_transition'
            AND ra.action_data->>'to_state' = 'paid'
            {test_filter}
        """
        paid_row = await pool.fetchrow(paid_query)

        return {
            "state_distribution": state_counts,
            "paid_revenue": float(paid_row["paid_revenue"] or 0) if paid_row else 0,
            "total_in_pipeline": sum(state_counts.values()),
            "ledger_backed": True
        }

    async def migrate_legacy_leads(self) -> dict:
        """Migrate leads from legacy stage to new state machine."""
        pool = self._get_pool()
        if not pool:
            return {"error": "Database not available"}

        # Get leads without state machine entries
        leads = await pool.fetch("""
            SELECT rl.id, rl.stage, rl.email
            FROM revenue_leads rl
            LEFT JOIN revenue_actions ra ON rl.id = ra.lead_id AND ra.action_type = 'state_transition'
            WHERE ra.id IS NULL
            AND rl.email NOT ILIKE '%test%'
            AND rl.email NOT ILIKE '%example%'
            AND rl.email NOT ILIKE '%demo%'
        """)

        migrated = 0
        for lead in leads:
            new_state = LEGACY_STAGE_MAP.get(lead["stage"], PipelineState.NEW_REAL)
            success, msg, _ = await self.transition(
                str(lead["id"]),
                new_state,
                trigger="legacy_migration",
                actor="system:migration",
                metadata={"legacy_stage": lead["stage"]},
                force=True
            )
            if success:
                migrated += 1

        return {
            "total_legacy_leads": len(leads),
            "migrated": migrated,
            "status": "complete"
        }


# Singleton instance
_state_machine: Optional[PipelineStateMachine] = None


def get_state_machine() -> PipelineStateMachine:
    """Get singleton state machine instance."""
    global _state_machine
    if _state_machine is None:
        _state_machine = PipelineStateMachine()
    return _state_machine

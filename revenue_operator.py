#!/usr/bin/env python3
"""
Revenue Operator
================
AI-driven revenue operations that generate real work items.

Features:
- Next-best-action generation for top leads
- Proposal drafting for qualified leads
- Outreach recommendations
- Hard alarms for stalled pipelines
- Kill switch integration

Part of Revenue Perfection Session.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
import json

logger = logging.getLogger(__name__)


def _parse_metadata(metadata: Any) -> dict:
    """Parse metadata field which may be a string or dict."""
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


class RevenueOperator:
    """
    AI Revenue Operator - generates real revenue-driving work.

    All operations are approval-gated and respect controls.
    Reads from ground truth (ledger-backed facts).
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

    async def check_kill_switches(self) -> tuple[bool, str]:
        """Check if any kill switches are active."""
        pool = self._get_pool()
        if not pool:
            return True, "Database unavailable - assuming kill switch active"

        controls = await pool.fetch("""
            SELECT key, value FROM unified_brain
            WHERE key LIKE 'control_all_%' AND value = '"disabled"'::jsonb
        """)

        active_switches = [c["key"].replace("control_", "") for c in controls]
        if active_switches:
            return True, f"Kill switches active: {', '.join(active_switches)}"

        return False, "No kill switches active"

    async def get_next_best_actions(self, limit: int = 10) -> dict[str, Any]:
        """
        Generate next-best-action plan for top leads.

        Returns prioritized list of actions.
        """
        # Check kill switches
        killed, reason = await self.check_kill_switches()
        if killed:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "blocked",
                "reason": reason,
                "actions": []
            }

        pool = self._get_pool()
        if not pool:
            return {"error": "Database not available"}

        # Get top REAL leads from ledger
        test_filter = """
            rl.email NOT ILIKE '%test%'
            AND rl.email NOT ILIKE '%example%'
            AND rl.email NOT ILIKE '%demo%'
        """

        leads = await pool.fetch(f"""
            SELECT
                rl.id,
                rl.company_name,
                rl.stage,
                rl.score,
                rl.value_estimate,
                EXTRACT(DAY FROM (NOW() - rl.updated_at)) as days_stale,
                rl.metadata
            FROM revenue_leads rl
            WHERE {test_filter}
            AND rl.stage NOT IN ('won', 'lost')
            ORDER BY rl.score DESC, rl.updated_at ASC
            LIMIT $1
        """, limit)

        actions = []
        for lead in leads:
            lead_id = str(lead["id"])
            company = lead["company_name"]
            stage = lead["stage"]
            days_stale = lead["days_stale"] or 0
            metadata = _parse_metadata(lead["metadata"])
            has_enrichment = bool(metadata.get("enrichment"))

            # Determine next best action based on state
            action = self._determine_action(
                lead_id=lead_id,
                company=company,
                stage=stage,
                days_stale=days_stale,
                has_enrichment=has_enrichment
            )
            actions.append(action)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "total_actions": len(actions),
            "actions": actions
        }

    def _determine_action(
        self,
        lead_id: str,
        company: str,
        stage: str,
        days_stale: float,
        has_enrichment: bool
    ) -> dict:
        """Determine the next best action for a lead."""
        if stage == "new":
            if not has_enrichment:
                return {
                    "lead_id": lead_id[:8] + "...",
                    "company": company,
                    "current_stage": stage,
                    "action": "ENRICH",
                    "api_call": f"POST /outreach/leads/{lead_id}/enrich",
                    "priority": "high",
                    "reason": "Lead needs enrichment before outreach"
                }
            else:
                return {
                    "lead_id": lead_id[:8] + "...",
                    "company": company,
                    "current_stage": stage,
                    "action": "DRAFT_OUTREACH",
                    "api_call": f"POST /outreach/leads/{lead_id}/draft",
                    "priority": "high",
                    "reason": "Enriched lead ready for initial outreach"
                }

        elif stage == "contacted":
            if days_stale > 7:
                return {
                    "lead_id": lead_id[:8] + "...",
                    "company": company,
                    "current_stage": stage,
                    "action": "FOLLOWUP",
                    "api_call": f"POST /outreach/leads/{lead_id}/draft?sequence_step=2",
                    "priority": "critical",
                    "reason": f"Lead stale for {int(days_stale)} days - needs follow-up"
                }
            else:
                return {
                    "lead_id": lead_id[:8] + "...",
                    "company": company,
                    "current_stage": stage,
                    "action": "WAIT",
                    "priority": "low",
                    "reason": "Recently contacted - waiting for response"
                }

        elif stage == "qualified":
            return {
                "lead_id": lead_id[:8] + "...",
                "company": company,
                "current_stage": stage,
                "action": "DRAFT_PROPOSAL",
                "api_call": f"POST /proposals/draft (lead_id={lead_id[:8]}...)",
                "priority": "high",
                "reason": "Qualified lead ready for proposal"
            }

        elif stage == "proposal_sent":
            if days_stale > 3:
                return {
                    "lead_id": lead_id[:8] + "...",
                    "company": company,
                    "current_stage": stage,
                    "action": "PROPOSAL_FOLLOWUP",
                    "api_call": f"POST /outreach/leads/{lead_id}/draft?sequence_step=3",
                    "priority": "high",
                    "reason": f"Proposal sent {int(days_stale)} days ago - follow up"
                }
            else:
                return {
                    "lead_id": lead_id[:8] + "...",
                    "company": company,
                    "current_stage": stage,
                    "action": "WAIT",
                    "priority": "medium",
                    "reason": "Waiting for proposal response"
                }

        elif stage == "negotiating":
            return {
                "lead_id": lead_id[:8] + "...",
                "company": company,
                "current_stage": stage,
                "action": "CLOSE_DEAL",
                "api_call": "Manual: Schedule call to close",
                "priority": "critical",
                "reason": "Lead in negotiation - push to close"
            }

        return {
            "lead_id": lead_id[:8] + "...",
            "company": company,
            "current_stage": stage,
            "action": "REVIEW",
            "priority": "low",
            "reason": f"Unknown stage: {stage}"
        }

    async def generate_alarms(self) -> dict[str, Any]:
        """
        Generate hard alarms for revenue pipeline issues.

        Alarms trigger when:
        - real_leads > 0 AND real_revenue == 0 AND outreach_sent == 0 -> critical
        - outreach_sent > 0 AND reply_rate == 0 over threshold -> message/offer mismatch
        """
        pool = self._get_pool()
        if not pool:
            return {"error": "Database not available"}

        alarms = []

        # Get real lead count
        real_leads = await pool.fetchrow("""
            SELECT COUNT(*) as count FROM revenue_leads
            WHERE email NOT ILIKE '%test%'
            AND email NOT ILIKE '%example%'
            AND email NOT ILIKE '%demo%'
        """)
        real_lead_count = real_leads["count"] if real_leads else 0

        # Get real revenue
        real_revenue = await pool.fetchrow("""
            SELECT COALESCE(SUM(amount), 0) as total
            FROM real_revenue_tracking
            WHERE COALESCE(is_verified, false) = true
              AND COALESCE(is_demo, false) = false
        """)
        revenue_total = float(real_revenue["total"] or 0) if real_revenue else 0

        # Get outreach sent count
        outreach_sent = await pool.fetchrow("""
            SELECT COUNT(*) as count FROM revenue_actions
            WHERE action_type = 'outreach_draft'
            AND action_data->>'status' = 'sent'
        """)
        outreach_count = outreach_sent["count"] if outreach_sent else 0

        # Get reply count
        replies = await pool.fetchrow("""
            SELECT COUNT(*) as count
            FROM lead_activities
            WHERE event_type = 'reply_received'
        """)
        reply_count = replies["count"] if replies else 0

        # Generate alarms
        if real_lead_count > 0 and revenue_total == 0 and outreach_count == 0:
            alarms.append({
                "severity": "critical",
                "code": "NO_OUTREACH",
                "message": f"{real_lead_count} real leads, $0 revenue, 0 outreach sent",
                "recommendation": "Execute batch outreach immediately",
                "api_action": "POST /outreach/batch/enrich-all then POST /outreach/batch/draft-outreach"
            })
        elif real_lead_count > 0 and revenue_total == 0:
            alarms.append({
                "severity": "critical",
                "code": "NO_REVENUE",
                "message": f"{real_lead_count} real leads, {outreach_count} outreach sent, $0 revenue",
                "recommendation": "Focus on conversion - proposals and follow-ups"
            })

        if outreach_count > 10 and reply_count == 0:
            alarms.append({
                "severity": "warning",
                "code": "MESSAGE_MISMATCH",
                "message": f"{outreach_count} outreach sent, 0 replies - possible message/offer mismatch",
                "recommendation": "Review outreach templates and value proposition"
            })

        # Check for stale leads
        stale = await pool.fetchrow("""
            SELECT COUNT(*) as count FROM revenue_leads
            WHERE stage NOT IN ('won', 'lost')
            AND updated_at < NOW() - INTERVAL '14 days'
            AND email NOT ILIKE '%test%'
            AND email NOT ILIKE '%example%'
        """)
        stale_count = stale["count"] if stale else 0

        if stale_count > 5:
            alarms.append({
                "severity": "warning",
                "code": "STALE_LEADS",
                "message": f"{stale_count} leads stale for 14+ days",
                "recommendation": "Execute follow-up campaign"
            })

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_alarms": len(alarms),
            "metrics": {
                "real_leads": real_lead_count,
                "real_revenue": revenue_total,
                "outreach_sent": outreach_count,
                "replies": reply_count,
                "stale_leads": stale_count
            },
            "alarms": alarms
        }

    async def auto_draft_proposals(self, limit: int = 5) -> dict[str, Any]:
        """
        Auto-draft proposals for qualified leads.

        Creates drafts that require human approval.
        """
        killed, reason = await self.check_kill_switches()
        if killed:
            return {"status": "blocked", "reason": reason}

        pool = self._get_pool()
        if not pool:
            return {"error": "Database not available"}

        from proposal_engine import get_proposal_engine

        # Get qualified leads without proposals
        leads = await pool.fetch("""
            SELECT rl.id, rl.company_name, rl.industry
            FROM revenue_leads rl
            LEFT JOIN ai_proposals p ON rl.id = p.lead_id
            WHERE rl.stage IN ('qualified', 'negotiating')
            AND p.id IS NULL
            AND rl.email NOT ILIKE '%test%'
            AND rl.email NOT ILIKE '%example%'
            LIMIT $1
        """, limit)

        engine = get_proposal_engine()
        proposals_created = []

        for lead in leads:
            # Select offer based on industry
            offer_id = "brainops_automation"  # Default
            if lead.get("industry") == "roofing":
                offer_id = "mrg_pro"

            success, msg, proposal = await engine.draft_proposal(
                str(lead["id"]),
                offer_id
            )

            if success and proposal:
                proposals_created.append({
                    "lead_id": str(lead["id"])[:8] + "...",
                    "company": lead["company_name"],
                    "proposal_id": proposal.id[:8] + "...",
                    "offer": proposal.offer_name
                })

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "proposals_created": len(proposals_created),
            "proposals": proposals_created,
            "note": "All proposals require human approval before sending"
        }


# Singleton instance
_revenue_operator: Optional[RevenueOperator] = None


def get_revenue_operator() -> RevenueOperator:
    """Get singleton revenue operator instance."""
    global _revenue_operator
    if _revenue_operator is None:
        _revenue_operator = RevenueOperator()
    return _revenue_operator

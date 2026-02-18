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
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
import json

logger = logging.getLogger(__name__)

def _env_flag(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


ENABLE_REVENUE_LIFECYCLE_AUTOMATION = _env_flag("ENABLE_REVENUE_LIFECYCLE_AUTOMATION", "true")
ENABLE_ADVANCED_LEAD_SCORING = _env_flag("ENABLE_REVENUE_ADVANCED_SCORING", "true")

# New enhancements (default OFF for safe rollout)
ENABLE_REVENUE_STAGE_AUTOMATION = _env_flag("ENABLE_REVENUE_STAGE_AUTOMATION", "false")
ENABLE_REVENUE_FOLLOWUP_ESCALATION = _env_flag("ENABLE_REVENUE_FOLLOWUP_ESCALATION", "false")
ENABLE_REVENUE_WIN_LOSS_LEARNING = _env_flag("ENABLE_REVENUE_WIN_LOSS_LEARNING", "false")
ENABLE_REVENUE_FORECASTING = _env_flag("ENABLE_REVENUE_FORECASTING", "false")
ENABLE_REVENUE_CHURN_PREDICTION = _env_flag("ENABLE_REVENUE_CHURN_PREDICTION", "false")


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
        self._default_tenant_id = (
            os.getenv("DEFAULT_TENANT_ID")
            or os.getenv("TENANT_ID")
            or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
        )

    def _get_pool(self):
        """Get database pool."""
        try:
            from database.async_connection import get_pool
            return get_pool()
        except Exception as e:
            logger.error(f"Failed to get database pool: {e}")
            return None

    def _get_tenant_pool(self, tenant_id: Optional[str] = None):
        """Get tenant-scoped pool for new lifecycle intelligence operations."""
        resolved_tenant = (tenant_id or "").strip() or self._default_tenant_id
        try:
            from database.async_connection import get_tenant_pool

            return get_tenant_pool(resolved_tenant)
        except Exception as exc:
            logger.error("Failed to get tenant scoped pool for %s: %s", resolved_tenant, exc)
            return None

    @staticmethod
    def _stage_probability(stage: str) -> float:
        """Map pipeline stage to default close probability."""
        stage_norm = (stage or "").strip().lower()
        mapping = {
            "new": 0.10,
            "enriched": 0.16,
            "contacted": 0.20,
            "outreach_sent": 0.25,
            "replied": 0.35,
            "qualified": 0.50,
            "meeting_booked": 0.58,
            "proposal_drafted": 0.65,
            "proposal_sent": 0.72,
            "proposal_approved": 0.80,
            "negotiating": 0.86,
            "won_invoice_pending": 0.95,
            "invoiced": 0.98,
            "paid": 1.0,
            "won": 1.0,
            "lost": 0.0,
        }
        return mapping.get(stage_norm, 0.20)

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
        stage_norm = (stage or "").strip().lower()

        if stage_norm in {"new", "new_real"}:
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

        elif stage_norm in {"enriched", "contact_ready"}:
            return {
                "lead_id": lead_id[:8] + "...",
                "company": company,
                "current_stage": stage,
                "action": "DRAFT_OUTREACH",
                "api_call": f"POST /outreach/leads/{lead_id}/draft",
                "priority": "high",
                "reason": "Lead is contact-ready and should enter outreach"
            }

        elif stage_norm in {"contacted", "outreach_sent"}:
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

        elif stage_norm in {"replied"}:
            return {
                "lead_id": lead_id[:8] + "...",
                "company": company,
                "current_stage": stage,
                "action": "BOOK_MEETING",
                "api_call": "Manual: Book discovery or closing call",
                "priority": "high",
                "reason": "Lead replied and should be progressed to meeting"
            }

        elif stage_norm in {"qualified", "meeting_booked", "proposal_drafted"}:
            return {
                "lead_id": lead_id[:8] + "...",
                "company": company,
                "current_stage": stage,
                "action": "DRAFT_PROPOSAL",
                "api_call": f"POST /proposals/draft (lead_id={lead_id[:8]}...)",
                "priority": "high",
                "reason": "Qualified lead ready for proposal"
            }

        elif stage_norm in {"proposal_sent", "proposal_approved"}:
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

        elif stage_norm in {"negotiating", "won_invoice_pending"}:
            return {
                "lead_id": lead_id[:8] + "...",
                "company": company,
                "current_stage": stage,
                "action": "CLOSE_DEAL",
                "api_call": "Manual: Schedule call to close",
                "priority": "critical",
                "reason": "Lead in negotiation - push to close"
            }

        elif stage_norm in {"invoiced"}:
            return {
                "lead_id": lead_id[:8] + "...",
                "company": company,
                "current_stage": stage,
                "action": "CAPTURE_PAYMENT",
                "api_call": "POST /payments/invoices/retry-outstanding",
                "priority": "critical",
                "reason": "Invoice sent; actively collect payment"
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

    async def _score_recent_leads(self, limit: int = 25) -> dict[str, Any]:
        """Run advanced scoring for active leads and write back score hints."""
        if not ENABLE_ADVANCED_LEAD_SCORING:
            return {"status": "skipped", "reason": "ENABLE_REVENUE_ADVANCED_SCORING=false"}

        pool = self._get_pool()
        if not pool:
            return {"status": "error", "error": "Database not available"}

        try:
            from advanced_lead_scoring import AdvancedLeadScoringEngine

            scorer = AdvancedLeadScoringEngine()
            leads = await pool.fetch(
                """
                SELECT id
                FROM revenue_leads
                WHERE stage NOT IN ('won', 'lost')
                ORDER BY updated_at DESC NULLS LAST, created_at DESC
                LIMIT $1
                """,
                max(1, limit),
            )
            scored = 0
            failed = 0
            for lead in leads:
                lead_id = str(lead["id"])
                try:
                    result = await scorer.calculate_multi_factor_score(lead_id)
                    await pool.execute(
                        """
                        UPDATE revenue_leads
                        SET score = $1,
                            updated_at = $2,
                            metadata = COALESCE(metadata, '{}'::jsonb) || $3::jsonb
                        WHERE id = $4
                        """,
                        float(result.composite_score) / 100.0,
                        datetime.now(timezone.utc),
                        json.dumps(
                            {
                                "advanced_scoring": {
                                    "composite": float(result.composite_score),
                                    "tier": result.tier.value,
                                    "probability_conversion_30d": float(result.probability_conversion_30d),
                                    "next_best_action": result.next_best_action,
                                    "scored_at": datetime.now(timezone.utc).isoformat(),
                                }
                            }
                        ),
                        lead["id"],
                    )
                    scored += 1
                except Exception:
                    failed += 1

            return {"status": "completed", "scored": scored, "failed": failed}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def _create_and_send_invoices(self, limit: int = 10) -> dict[str, Any]:
        """Create and send invoices for ready proposals."""
        pool = self._get_pool()
        if not pool:
            return {"status": "error", "error": "Database not available"}

        from payment_capture import get_payment_capture

        pc = get_payment_capture()
        proposals = await pool.fetch(
            """
            SELECT p.id
            FROM ai_proposals p
            LEFT JOIN ai_invoices i ON i.proposal_id = p.id
            WHERE p.status IN ('approved', 'sent')
              AND i.id IS NULL
            ORDER BY p.updated_at DESC NULLS LAST, p.created_at DESC
            LIMIT $1
            """,
            max(1, limit),
        )

        created = 0
        sent = 0
        errors = 0
        for proposal in proposals:
            success, _msg, invoice = await pc.create_invoice(str(proposal["id"]))
            if not success or not invoice:
                errors += 1
                continue
            created += 1
            sent_ok, _ = await pc.send_invoice(invoice.id)
            if sent_ok:
                sent += 1

        return {
            "status": "completed",
            "created": created,
            "sent": sent,
            "errors": errors,
        }

    async def _advance_pipeline_stages(
        self,
        *,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Progress leads through the full revenue lifecycle based on score + engagement.

        discover -> qualify -> score -> nurture -> propose -> negotiate -> close -> invoice -> collect
        """
        if not ENABLE_REVENUE_STAGE_AUTOMATION:
            return {"status": "skipped", "reason": "ENABLE_REVENUE_STAGE_AUTOMATION=false"}

        pool = self._get_tenant_pool(tenant_id)
        if not pool:
            return {"status": "error", "error": "Tenant pool unavailable"}

        try:
            leads = await pool.fetch(
                """
                WITH engagement AS (
                    SELECT
                        lead_id,
                        COUNT(*) FILTER (WHERE event_type = 'reply_received') AS replies,
                        COUNT(*) FILTER (WHERE event_type = 'meeting_booked') AS meetings,
                        COUNT(*) FILTER (WHERE event_type = 'email_open') AS opens,
                        COUNT(*) FILTER (WHERE event_type = 'email_click') AS clicks,
                        MAX(created_at) AS last_event_at
                    FROM lead_activities
                    GROUP BY lead_id
                )
                SELECT
                    rl.id,
                    rl.stage,
                    rl.score,
                    rl.value_estimate,
                    rl.metadata,
                    COALESCE(eng.replies, 0) AS replies,
                    COALESCE(eng.meetings, 0) AS meetings,
                    COALESCE(eng.opens, 0) AS opens,
                    COALESCE(eng.clicks, 0) AS clicks,
                    eng.last_event_at
                FROM revenue_leads rl
                LEFT JOIN engagement eng ON eng.lead_id = rl.id
                WHERE rl.stage NOT IN ('won', 'lost', 'paid')
                ORDER BY rl.updated_at ASC NULLS LAST, rl.created_at ASC
                LIMIT $1
                """,
                max(1, limit),
            )
        except Exception as exc:
            logger.warning("Stage automation query fallback due to join error: %s", exc)
            leads = await pool.fetch(
                """
                SELECT id, stage, score, value_estimate, metadata
                FROM revenue_leads
                WHERE stage NOT IN ('won', 'lost', 'paid')
                ORDER BY updated_at ASC NULLS LAST, created_at ASC
                LIMIT $1
                """,
                max(1, limit),
            )

        transitions: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for lead in leads:
            current_stage = str(lead.get("stage") or "").strip().lower()
            score = float(lead.get("score") or 0.0)
            replies = int(lead.get("replies") or 0)
            meetings = int(lead.get("meetings") or 0)
            opens = int(lead.get("opens") or 0)
            clicks = int(lead.get("clicks") or 0)
            metadata = _parse_metadata(lead.get("metadata"))
            next_stage: Optional[str] = None
            reason: Optional[str] = None

            if current_stage in {"new", "enriched", "contact_ready"} and score >= 0.55:
                next_stage, reason = "qualified", "score_threshold_met"
            elif current_stage in {"qualified", "meeting_booked"} and (replies > 0 or meetings > 0):
                next_stage, reason = "proposal_drafted", "engagement_signaled_buying_intent"
            elif current_stage in {"proposal_drafted", "proposal_sent"} and (meetings > 0 or replies >= 2):
                next_stage, reason = "negotiating", "active_negotiation_signals"
            elif current_stage == "negotiating":
                won_signal = bool(metadata.get("deal_won")) or bool(metadata.get("proposal_accepted"))
                if won_signal or meetings >= 2:
                    next_stage, reason = "won_invoice_pending", "close_signal_detected"
            elif current_stage == "invoiced":
                paid_signal = bool(metadata.get("payment_received"))
                if paid_signal:
                    next_stage, reason = "paid", "payment_signal_detected"
            elif current_stage in {"contacted", "outreach_sent"} and (replies > 0 or clicks >= 2 or opens >= 3):
                next_stage, reason = "replied", "engagement_progression"

            if not next_stage or next_stage == current_stage:
                continue

            transition_meta = {
                "lifecycle_transition": {
                    "from": current_stage,
                    "to": next_stage,
                    "reason": reason,
                    "at": now.isoformat(),
                }
            }
            await pool.execute(
                """
                UPDATE revenue_leads
                SET stage = $1,
                    updated_at = $2,
                    metadata = COALESCE(metadata, '{}'::jsonb) || $3::jsonb
                WHERE id = $4
                """,
                next_stage,
                now,
                json.dumps(transition_meta),
                lead["id"],
            )
            await pool.execute(
                """
                INSERT INTO revenue_actions (
                    id, lead_id, action_type, action_data, success, created_at, executed_by
                ) VALUES ($1, $2, 'lifecycle_stage_advance', $3, true, $4, 'system:revenue_operator')
                """,
                uuid.uuid4(),
                lead["id"],
                {
                    "from": current_stage,
                    "to": next_stage,
                    "reason": reason,
                    "score": score,
                    "engagement": {"opens": opens, "clicks": clicks, "replies": replies, "meetings": meetings},
                },
                now,
            )
            transitions.append(
                {
                    "lead_id": str(lead["id"]),
                    "from_stage": current_stage,
                    "to_stage": next_stage,
                    "reason": reason,
                }
            )

        return {
            "status": "completed",
            "processed": len(leads),
            "transitions": len(transitions),
            "items": transitions[:25],
        }

    async def _schedule_followups_with_escalation(
        self,
        *,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Schedule automated follow-ups with escalating urgency."""
        if not ENABLE_REVENUE_FOLLOWUP_ESCALATION:
            return {"status": "skipped", "reason": "ENABLE_REVENUE_FOLLOWUP_ESCALATION=false"}

        pool = self._get_tenant_pool(tenant_id)
        if not pool:
            return {"status": "error", "error": "Tenant pool unavailable"}

        rows = await pool.fetch(
            """
            SELECT
                rl.id,
                rl.stage,
                rl.email,
                rl.company_name,
                rl.metadata,
                COALESCE(rl.last_contact, rl.updated_at, rl.created_at) AS last_touch_at,
                EXTRACT(DAY FROM (NOW() - COALESCE(rl.last_contact, rl.updated_at, rl.created_at))) AS days_since_touch
            FROM revenue_leads rl
            WHERE rl.stage IN ('contacted', 'outreach_sent', 'qualified', 'proposal_sent', 'negotiating', 'invoiced')
              AND rl.email IS NOT NULL
            ORDER BY days_since_touch DESC NULLS LAST
            LIMIT $1
            """,
            max(1, limit),
        )

        stage_thresholds = {
            "contacted": 3,
            "outreach_sent": 3,
            "qualified": 2,
            "proposal_sent": 2,
            "negotiating": 1,
            "invoiced": 1,
        }
        scheduled = 0
        escalated = 0
        now = datetime.now(timezone.utc)
        items: list[dict[str, Any]] = []

        for row in rows:
            stage = str(row.get("stage") or "").strip().lower()
            days_since_touch = float(row.get("days_since_touch") or 0.0)
            threshold = float(stage_thresholds.get(stage, 3))
            if days_since_touch < threshold:
                continue

            escalation_level = min(3, int(days_since_touch // threshold))
            escalation_level = max(1, escalation_level)
            scheduled_for = now + timedelta(hours=4 if escalation_level >= 3 else 18)
            metadata = _parse_metadata(row.get("metadata"))
            target_id = str(row["id"])

            # Avoid duplicate queued followups for same lead/stage/escalation in last day.
            existing = await pool.fetchval(
                """
                SELECT id
                FROM ai_scheduled_outreach
                WHERE target_id = $1
                  AND status IN ('scheduled', 'queued')
                  AND message_template = $2
                  AND created_at > NOW() - INTERVAL '1 day'
                LIMIT 1
                """,
                target_id,
                f"lifecycle_followup_{stage}",
            )
            if existing:
                continue

            followup_payload = {
                "lead_id": target_id,
                "company_name": row.get("company_name"),
                "stage": stage,
                "escalation_level": escalation_level,
                "days_since_touch": days_since_touch,
            }
            await pool.execute(
                """
                INSERT INTO ai_scheduled_outreach (
                    target_id, channel, message_template, personalization,
                    scheduled_for, status, metadata, created_at
                )
                VALUES ($1, 'email', $2, $3, $4, 'scheduled', $5, $6)
                """,
                target_id,
                f"lifecycle_followup_{stage}",
                json.dumps(followup_payload),
                scheduled_for,
                json.dumps(
                    {
                        "source": "revenue_operator_followup",
                        "escalation_level": escalation_level,
                        "stage": stage,
                        "last_touch_at": str(row.get("last_touch_at")) if row.get("last_touch_at") else None,
                    }
                ),
                now,
            )

            await pool.execute(
                """
                UPDATE revenue_leads
                SET metadata = COALESCE(metadata, '{}'::jsonb) || $1::jsonb,
                    updated_at = $2
                WHERE id = $3
                """,
                json.dumps(
                    {
                        "followup": {
                            "last_scheduled_at": now.isoformat(),
                            "scheduled_for": scheduled_for.isoformat(),
                            "escalation_level": escalation_level,
                        }
                    }
                ),
                now,
                row["id"],
            )
            await pool.execute(
                """
                INSERT INTO revenue_actions (
                    id, lead_id, action_type, action_data, success, created_at, executed_by
                ) VALUES ($1, $2, 'followup_escalation_scheduled', $3, true, $4, 'system:revenue_operator')
                """,
                uuid.uuid4(),
                row["id"],
                {
                    "stage": stage,
                    "days_since_touch": days_since_touch,
                    "escalation_level": escalation_level,
                    "scheduled_for": scheduled_for.isoformat(),
                },
                now,
            )
            scheduled += 1
            if escalation_level >= 2:
                escalated += 1
            items.append(
                {
                    "lead_id": target_id,
                    "stage": stage,
                    "escalation_level": escalation_level,
                    "scheduled_for": scheduled_for.isoformat(),
                }
            )

        return {
            "status": "completed",
            "evaluated": len(rows),
            "scheduled": scheduled,
            "escalated": escalated,
            "items": items[:25],
        }

    async def _run_win_loss_analysis(
        self,
        *,
        tenant_id: Optional[str] = None,
        lookback_days: int = 90,
    ) -> dict[str, Any]:
        """Generate win/loss analysis and persist a learning snapshot."""
        if not ENABLE_REVENUE_WIN_LOSS_LEARNING:
            return {"status": "skipped", "reason": "ENABLE_REVENUE_WIN_LOSS_LEARNING=false"}

        pool = self._get_tenant_pool(tenant_id)
        if not pool:
            return {"status": "error", "error": "Tenant pool unavailable"}

        rows = await pool.fetch(
            """
            SELECT
                COALESCE(NULLIF(rl.stage, ''), 'unknown') AS stage,
                COALESCE(NULLIF(rl.source, ''), 'unknown') AS source,
                COALESCE(NULLIF(rl.industry, ''), 'unknown') AS industry,
                rl.value_estimate,
                rl.metadata
            FROM revenue_leads rl
            WHERE rl.updated_at > NOW() - ($1::text || ' days')::interval
              AND rl.stage IN ('won', 'paid', 'lost', 'invoiced', 'won_invoice_pending')
            """,
            max(7, lookback_days),
        )

        total = len(rows)
        wins = 0
        losses = 0
        value_won = 0.0
        value_lost = 0.0
        source_stats: dict[str, dict[str, Any]] = {}
        loss_reasons: dict[str, int] = {}

        for row in rows:
            stage = str(row.get("stage") or "").lower()
            source = str(row.get("source") or "unknown")
            is_win = stage in {"won", "paid", "invoiced", "won_invoice_pending"}
            value = float(row.get("value_estimate") or 0.0)
            metadata = _parse_metadata(row.get("metadata"))

            stats = source_stats.setdefault(source, {"wins": 0, "losses": 0, "total_value": 0.0})
            stats["total_value"] += value
            if is_win:
                wins += 1
                value_won += value
                stats["wins"] += 1
            else:
                losses += 1
                value_lost += value
                stats["losses"] += 1
                reason = str(metadata.get("loss_reason") or metadata.get("closed_lost_reason") or "unspecified")
                loss_reasons[reason] = int(loss_reasons.get(reason, 0)) + 1

        win_rate = (wins / total) if total else 0.0
        top_loss_reasons = sorted(loss_reasons.items(), key=lambda item: item[1], reverse=True)[:5]
        payload = {
            "lookback_days": lookback_days,
            "total_closed": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "won_value": value_won,
            "lost_value": value_lost,
            "source_performance": source_stats,
            "top_loss_reasons": top_loss_reasons,
            "learned_at": datetime.now(timezone.utc).isoformat(),
        }

        anchor_lead_id = rows[0].get("id") if rows else None
        if anchor_lead_id:
            await pool.execute(
                """
                INSERT INTO revenue_actions (
                    id, lead_id, action_type, action_data, success, created_at, executed_by
                ) VALUES ($1, $2, 'win_loss_analysis', $3, true, $4, 'system:revenue_operator')
                """,
                uuid.uuid4(),
                anchor_lead_id,
                payload,
                datetime.now(timezone.utc),
            )
        return {"status": "completed", **payload}

    async def _build_revenue_forecast(
        self,
        *,
        tenant_id: Optional[str] = None,
        months: int = 3,
    ) -> dict[str, Any]:
        """Project near-term revenue using stage-weighted pipeline probabilities."""
        if not ENABLE_REVENUE_FORECASTING:
            return {"status": "skipped", "reason": "ENABLE_REVENUE_FORECASTING=false"}

        pool = self._get_tenant_pool(tenant_id)
        if not pool:
            return {"status": "error", "error": "Tenant pool unavailable"}

        rows = await pool.fetch(
            """
            SELECT id, stage, score, value_estimate, metadata
            FROM revenue_leads
            WHERE stage NOT IN ('lost', 'paid')
              AND value_estimate IS NOT NULL
            ORDER BY updated_at DESC NULLS LAST
            LIMIT 500
            """
        )
        now = datetime.now(timezone.utc)
        monthly: dict[str, dict[str, Any]] = {}
        total_expected = 0.0
        for row in rows:
            stage = str(row.get("stage") or "").lower()
            score = float(row.get("score") or 0.0)
            stage_prob = self._stage_probability(stage)
            probability = max(0.05, min(1.0, (stage_prob * 0.7) + (score * 0.3)))
            value = float(row.get("value_estimate") or 0.0)
            expected = value * probability
            metadata = _parse_metadata(row.get("metadata"))
            close_raw = metadata.get("expected_close_date")
            if close_raw:
                try:
                    close_at = datetime.fromisoformat(str(close_raw).replace("Z", "+00:00"))
                except Exception:
                    close_at = None
            else:
                close_at = None
            if close_at is None:
                stage_days = {
                    "new": 35,
                    "enriched": 30,
                    "contacted": 24,
                    "replied": 20,
                    "qualified": 15,
                    "proposal_drafted": 10,
                    "proposal_sent": 7,
                    "negotiating": 5,
                    "won_invoice_pending": 2,
                    "invoiced": 1,
                }
                close_at = now + timedelta(days=stage_days.get(stage, 21))
            bucket = close_at.strftime("%Y-%m")
            slot = monthly.setdefault(bucket, {"expected_revenue": 0.0, "pipeline_value": 0.0, "lead_count": 0})
            slot["expected_revenue"] += expected
            slot["pipeline_value"] += value
            slot["lead_count"] += 1
            total_expected += expected

        sorted_buckets = sorted(monthly.items(), key=lambda item: item[0])[: max(1, months)]
        forecast = [
            {
                "month": bucket,
                "expected_revenue": round(data["expected_revenue"], 2),
                "pipeline_value": round(data["pipeline_value"], 2),
                "lead_count": data["lead_count"],
                "confidence": min(
                    0.95,
                    0.55 + (0.06 * min(data["lead_count"], 5)),
                ),
            }
            for bucket, data in sorted_buckets
        ]
        snapshot = {
            "generated_at": now.isoformat(),
            "months": months,
            "total_expected_revenue": round(total_expected, 2),
            "forecast": forecast,
        }
        anchor_lead_id = rows[0].get("id") if rows else None
        if anchor_lead_id:
            await pool.execute(
                """
                INSERT INTO revenue_actions (
                    id, lead_id, action_type, action_data, success, created_at, executed_by
                ) VALUES ($1, $2, 'pipeline_forecast_snapshot', $3, true, $4, 'system:revenue_operator')
                """,
                uuid.uuid4(),
                anchor_lead_id,
                snapshot,
                now,
            )
        return {"status": "completed", **snapshot}

    async def _predict_pipeline_churn(
        self,
        *,
        tenant_id: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Predict churn risk from engagement decay patterns within the active pipeline."""
        if not ENABLE_REVENUE_CHURN_PREDICTION:
            return {"status": "skipped", "reason": "ENABLE_REVENUE_CHURN_PREDICTION=false"}

        pool = self._get_tenant_pool(tenant_id)
        if not pool:
            return {"status": "error", "error": "Tenant pool unavailable"}

        rows = await pool.fetch(
            """
            WITH engagement AS (
                SELECT
                    lead_id,
                    COUNT(*) FILTER (WHERE event_type = 'email_open') AS opens_30d,
                    COUNT(*) FILTER (WHERE event_type = 'email_click') AS clicks_30d,
                    COUNT(*) FILTER (WHERE event_type = 'reply_received') AS replies_30d,
                    MAX(created_at) AS last_event_at
                FROM lead_activities
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY lead_id
            )
            SELECT
                rl.id,
                rl.company_name,
                rl.stage,
                rl.score,
                rl.value_estimate,
                COALESCE(eng.opens_30d, 0) AS opens_30d,
                COALESCE(eng.clicks_30d, 0) AS clicks_30d,
                COALESCE(eng.replies_30d, 0) AS replies_30d,
                eng.last_event_at,
                EXTRACT(DAY FROM (NOW() - COALESCE(eng.last_event_at, rl.updated_at, rl.created_at))) AS days_inactive
            FROM revenue_leads rl
            LEFT JOIN engagement eng ON eng.lead_id = rl.id
            WHERE rl.stage IN ('contacted', 'outreach_sent', 'qualified', 'proposal_sent', 'negotiating', 'invoiced')
            ORDER BY days_inactive DESC NULLS LAST
            LIMIT $1
            """,
            max(10, limit),
        )

        at_risk: list[dict[str, Any]] = []
        for row in rows:
            days_inactive = float(row.get("days_inactive") or 0.0)
            opens = int(row.get("opens_30d") or 0)
            clicks = int(row.get("clicks_30d") or 0)
            replies = int(row.get("replies_30d") or 0)
            engagement_score = min(1.0, (opens * 0.05) + (clicks * 0.2) + (replies * 0.4))
            base_risk = min(1.0, days_inactive / 30.0)
            churn_probability = min(0.99, max(0.01, (base_risk * 0.75) + ((1.0 - engagement_score) * 0.25)))
            if churn_probability < 0.55:
                continue

            recommendation = "executive_followup" if churn_probability >= 0.80 else "nurture_reengagement"
            at_risk.append(
                {
                    "lead_id": str(row["id"]),
                    "company_name": row.get("company_name"),
                    "stage": row.get("stage"),
                    "value_estimate": float(row.get("value_estimate") or 0.0),
                    "days_inactive": days_inactive,
                    "engagement_score": round(engagement_score, 3),
                    "churn_probability": round(churn_probability, 3),
                    "recommendation": recommendation,
                }
            )

        at_risk.sort(key=lambda item: item["churn_probability"], reverse=True)
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "at_risk_count": len(at_risk),
            "high_risk_count": len([x for x in at_risk if x["churn_probability"] >= 0.8]),
            "top_at_risk": at_risk[:20],
        }
        anchor_lead_id = rows[0].get("id") if rows else None
        if anchor_lead_id:
            await pool.execute(
                """
                INSERT INTO revenue_actions (
                    id, lead_id, action_type, action_data, success, created_at, executed_by
                ) VALUES ($1, $2, 'pipeline_churn_prediction', $3, true, $4, 'system:revenue_operator')
                """,
                uuid.uuid4(),
                anchor_lead_id,
                snapshot,
                datetime.now(timezone.utc),
            )
        return {"status": "completed", **snapshot}

    async def run_full_lifecycle(
        self,
        limit: int = 25,
        auto_send_outreach: bool = False,
        tenant_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute end-to-end lifecycle orchestration:
        discovery -> scoring -> nurture -> proposal -> close support -> invoice -> payment retries.
        """
        if not ENABLE_REVENUE_LIFECYCLE_AUTOMATION:
            return {
                "status": "disabled",
                "reason": "ENABLE_REVENUE_LIFECYCLE_AUTOMATION=false",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        killed, reason = await self.check_kill_switches()
        if killed:
            return {
                "status": "blocked",
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        results: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "lifecycle": [
                "discover",
                "qualify",
                "score",
                "nurture",
                "propose",
                "negotiate",
                "close",
                "invoice",
                "collect",
            ],
        }

        # 1) Discovery
        try:
            from revenue_pipeline_agents import LeadDiscoveryAgentReal

            discovery_agent = LeadDiscoveryAgentReal()
            results["discovery"] = await discovery_agent.execute({"action": "discover_all"})
        except Exception as exc:
            results["discovery"] = {"status": "error", "error": str(exc)}

        # 2) Scoring
        results["scoring"] = await self._score_recent_leads(limit=limit)

        # 2b) Qualification + stage progression
        results["qualification"] = await self._advance_pipeline_stages(
            tenant_id=tenant_id,
            limit=max(10, min(limit * 4, 200)),
        )

        # 3) Nurture
        try:
            from revenue_pipeline_agents import NurtureExecutorAgentReal

            nurture_agent = NurtureExecutorAgentReal()
            results["nurture"] = await nurture_agent.execute({"action": "nurture_new_leads"})
        except Exception as exc:
            results["nurture"] = {"status": "error", "error": str(exc)}

        # 4) Outreach acceleration (optional)
        if auto_send_outreach:
            try:
                from outreach_engine import get_outreach_engine
                from database.async_connection import get_pool

                engine = get_outreach_engine()
                pool = get_pool()
                leads = await pool.fetch(
                    """
                    SELECT rl.id
                    FROM revenue_leads rl
                    WHERE rl.stage IN ('new', 'contacted', 'qualified')
                    ORDER BY rl.score DESC NULLS LAST, rl.updated_at ASC
                    LIMIT $1
                    """,
                    max(1, min(limit, 50)),
                )
                drafted = 0
                for lead in leads:
                    ok, _, _ = await engine.generate_outreach_draft(str(lead["id"]), 1)
                    if ok:
                        drafted += 1
                results["outreach"] = {"status": "completed", "drafted": drafted}
            except Exception as exc:
                results["outreach"] = {"status": "error", "error": str(exc)}
        else:
            results["outreach"] = {"status": "skipped", "reason": "auto_send_outreach=false"}

        # 5) Proposal drafting
        results["proposals"] = await self.auto_draft_proposals(limit=max(1, min(limit, 20)))

        # 5b) Negotiation + close progression pass
        results["deal_progression"] = await self._advance_pipeline_stages(
            tenant_id=tenant_id,
            limit=max(10, min(limit * 4, 200)),
        )

        # 6) Invoicing
        results["invoices"] = await self._create_and_send_invoices(limit=max(1, min(limit, 20)))

        # 7) Payment retries
        try:
            from payment_capture import get_payment_capture

            pc = get_payment_capture()
            results["payments"] = await pc.retry_outstanding_payments(max_invoices=max(1, min(limit, 50)))
        except Exception as exc:
            results["payments"] = {"status": "error", "error": str(exc)}

        # 8) Follow-up escalation
        results["followups"] = await self._schedule_followups_with_escalation(
            tenant_id=tenant_id,
            limit=max(10, min(limit * 4, 200)),
        )

        # Optional intelligence layers (all default OFF)
        results["win_loss_analysis"] = await self._run_win_loss_analysis(tenant_id=tenant_id)
        results["forecast"] = await self._build_revenue_forecast(
            tenant_id=tenant_id,
            months=max(1, min(6, int(os.getenv("REVENUE_FORECAST_MONTHS", "3")))),
        )
        results["churn_prediction"] = await self._predict_pipeline_churn(
            tenant_id=tenant_id,
            limit=max(20, min(limit * 5, 250)),
        )

        results["status"] = "completed"
        return results


# Singleton instance
_revenue_operator: Optional[RevenueOperator] = None


def get_revenue_operator() -> RevenueOperator:
    """Get singleton revenue operator instance."""
    global _revenue_operator
    if _revenue_operator is None:
        _revenue_operator = RevenueOperator()
    return _revenue_operator

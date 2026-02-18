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
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
import json

logger = logging.getLogger(__name__)

ENABLE_REVENUE_LIFECYCLE_AUTOMATION = (
    os.getenv("ENABLE_REVENUE_LIFECYCLE_AUTOMATION", "true").strip().lower() in {"1", "true", "yes", "on"}
)
ENABLE_ADVANCED_LEAD_SCORING = (
    os.getenv("ENABLE_REVENUE_ADVANCED_SCORING", "true").strip().lower() in {"1", "true", "yes", "on"}
)


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

    async def run_full_lifecycle(
        self,
        limit: int = 25,
        auto_send_outreach: bool = False,
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

        results: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "running"}

        # 1) Discovery
        try:
            from revenue_pipeline_agents import LeadDiscoveryAgentReal

            discovery_agent = LeadDiscoveryAgentReal()
            results["discovery"] = await discovery_agent.execute({"action": "discover_all"})
        except Exception as exc:
            results["discovery"] = {"status": "error", "error": str(exc)}

        # 2) Scoring
        results["scoring"] = await self._score_recent_leads(limit=limit)

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

        # 6) Invoicing
        results["invoices"] = await self._create_and_send_invoices(limit=max(1, min(limit, 20)))

        # 7) Payment retries
        try:
            from payment_capture import get_payment_capture

            pc = get_payment_capture()
            results["payments"] = await pc.retry_outstanding_payments(max_invoices=max(1, min(limit, 50)))
        except Exception as exc:
            results["payments"] = {"status": "error", "error": str(exc)}

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

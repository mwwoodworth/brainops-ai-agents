from __future__ import annotations

import json
import logging
import math
import os
import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _safe_uuid(value: Any) -> Optional[uuid.UUID]:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except Exception:
        return None


def _is_test_email(email: str | None) -> bool:
    if not email:
        return True
    lowered = email.strip().lower()
    if any(token in lowered for token in ("@example.", "@test.", "@demo.", "@invalid.")):
        return True
    return any(lowered.endswith(suffix) for suffix in (".test", ".example", ".invalid"))


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return 0.0


def _normalize_revenue(amount: float, cap: float) -> float:
    """Log-scaled normalization into [0, 1]."""
    amount = max(0.0, float(amount or 0.0))
    cap = max(1.0, float(cap or 1.0))
    return max(0.0, min(1.0, math.log1p(amount) / math.log1p(cap)))


def _state_reward(state: str) -> float:
    # Keep simple and monotonic: higher pipeline states get higher partial credit.
    state_norm = (state or "").strip().lower()
    mapping = {
        "paid": 1.0,
        "invoiced": 0.85,
        "won_invoice_pending": 0.75,
        # Legacy stages (denormalized into revenue_leads.stage)
        "won": 0.8,
        "proposal_sent": 0.55,
        "negotiating": 0.35,
        "proposal_approved": 0.5,
        "proposal_drafted": 0.4,
        "meeting_booked": 0.3,
        "replied": 0.25,
        "outreach_sent": 0.12,
        "contacted": 0.12,
        "outreach_pending_approval": 0.08,
        "contact_ready": 0.06,
        "qualified": 0.06,
        "enriched": 0.04,
        "new_real": 0.02,
        "new": 0.02,
        "lost": 0.0,
    }
    return float(mapping.get(state_norm, 0.0))


def _training_source_allowlist() -> set[str] | None:
    """
    Restrict which ai_email_queue emails are used for training.

    This prevents transactional/internal emails (invoices, partner notifications, ops alerts, etc.)
    from polluting the outreach optimizer training set.

    Env:
      - DSPY_REVENUE_TRAINING_SOURCE_ALLOWLIST="outreach_engine,nurture_sequence,campaign_outreach"
      - DSPY_REVENUE_TRAINING_SOURCE_ALLOWLIST="*"  (allow all sources)
    """
    raw = os.getenv("DSPY_REVENUE_TRAINING_SOURCE_ALLOWLIST")
    if raw is None:
        # Safe defaults: only sources that represent actual outreach/nurture messaging.
        return {"outreach_engine", "nurture_sequence", "campaign_outreach"}
    raw = raw.strip()
    if not raw or raw == "*":
        return None
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


async def build_revenue_training_samples(
    *,
    pool: Any | None = None,
    lookback_days: int = 180,
    limit: int = 250,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build DSPy training samples from REAL outcomes:
    - Sent emails from ai_email_queue (lead_id in metadata preferred)
    - Conversion/revenue signal from real_revenue_tracking (metadata.lead_id)
    - Pipeline state from revenue_actions ledger (state_transition to_state)

    Returns (training_samples, stats).
    """
    if pool is None:
        from database.async_connection import get_pool

        pool = get_pool()

    lookback_days = max(1, int(lookback_days))
    limit = max(1, min(2000, int(limit)))

    # Pull recent sent emails. Prefer those with a lead_id in metadata to avoid ambiguous joins.
    rows = await pool.fetch(
        """
        SELECT id, recipient, subject, body, sent_at, metadata
        FROM ai_email_queue
        WHERE status = 'sent'
          AND sent_at IS NOT NULL
          AND sent_at > NOW() - ($1 * INTERVAL '1 day')
        ORDER BY sent_at DESC
        LIMIT $2
        """,
        lookback_days,
        limit,
    )

    email_rows: list[dict[str, Any]] = []
    lead_ids: list[uuid.UUID] = []
    lead_id_strs: list[str] = []
    lead_ids_by_email: dict[str, str] = {}
    allowed_sources = _training_source_allowlist()
    skipped_by_source: dict[str, int] = {}

    for row in rows:
        item = dict(row)
        metadata = _parse_json(item.get("metadata"))
        item["metadata"] = metadata

        source = str(metadata.get("source") or "").strip().lower()
        if allowed_sources is not None:
            if not source or source not in allowed_sources:
                key = source or "<missing>"
                skipped_by_source[key] = skipped_by_source.get(key, 0) + 1
                continue

        recipient = str(item.get("recipient") or "").strip()
        if not recipient or _is_test_email(recipient):
            continue
        if str(metadata.get("is_test") or "").strip().lower() in {"true", "1", "yes"}:
            continue
        if str(metadata.get("is_demo") or "").strip().lower() in {"true", "1", "yes"}:
            continue

        lead_id = _safe_uuid(metadata.get("lead_id"))
        if lead_id:
            lead_ids.append(lead_id)
            lead_id_strs.append(str(lead_id))
            item["lead_id"] = lead_id
        else:
            item["lead_id"] = None
            lead_ids_by_email[recipient.lower()] = ""

        email_rows.append(item)

    if not email_rows:
        return (
            [],
            {
                "emails_considered": 0,
                "samples_built": 0,
                "reason": "no_sent_emails",
                "training_source_allowlist": sorted(list(allowed_sources)) if allowed_sources is not None else "*",
                "skipped_by_source": skipped_by_source,
            },
        )

    # Best-effort: resolve missing lead_id by recipient email (batch).
    unresolved_emails = [email for email, lead in lead_ids_by_email.items() if not lead]
    if unresolved_emails:
        lead_email_rows = await pool.fetch(
            """
            SELECT id, email
            FROM revenue_leads
            WHERE LOWER(email) = ANY($1::text[])
            """,
            unresolved_emails,
        )
        for lr in lead_email_rows:
            if lr.get("email") and lr.get("id"):
                lead_ids_by_email[str(lr["email"]).strip().lower()] = str(lr["id"])

        for item in email_rows:
            if item.get("lead_id") is not None:
                continue
            recipient = str(item.get("recipient") or "").strip().lower()
            lead_id_str = lead_ids_by_email.get(recipient) or ""
            lead_id = _safe_uuid(lead_id_str)
            if lead_id:
                lead_ids.append(lead_id)
                lead_id_strs.append(str(lead_id))
                item["lead_id"] = lead_id

    # De-dup lead ids.
    lead_ids = list({lid for lid in lead_ids if lid})
    lead_id_strs = list({s for s in lead_id_strs if s})

    # Load lead context.
    leads_by_id: dict[str, dict[str, Any]] = {}
    if lead_ids:
        lead_rows = await pool.fetch(
            """
            SELECT
                id, company_name, contact_name, email, phone, website,
                industry, source, stage, status,
                value_estimate, estimated_value, expected_revenue,
                metadata, created_at, updated_at,
                is_test, is_demo
            FROM revenue_leads
            WHERE id = ANY($1::uuid[])
            """,
            lead_ids,
        )
        for lr in lead_rows:
            leads_by_id[str(lr["id"])] = dict(lr)

    # Latest pipeline state from ledger (if available).
    state_by_lead: dict[str, str] = {}
    if lead_ids:
        state_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (lead_id)
                lead_id,
                action_data->>'to_state' AS to_state,
                created_at
            FROM revenue_actions
            WHERE action_type = 'state_transition'
              AND lead_id = ANY($1::uuid[])
            ORDER BY lead_id, created_at DESC
            """,
            lead_ids,
        )
        for sr in state_rows:
            lead_id_str = str(sr.get("lead_id") or "")
            to_state = str(sr.get("to_state") or "").strip()
            if lead_id_str and to_state:
                state_by_lead[lead_id_str] = to_state

    # Revenue outcomes from real_revenue_tracking (metadata.lead_id).
    revenue_by_lead: dict[str, float] = {}
    if lead_id_strs:
        # metadata->>'lead_id' is stored as text; use text matching to avoid cast failures.
        rev_rows = await pool.fetch(
            """
            SELECT (metadata->>'lead_id') AS lead_id, COALESCE(SUM(amount), 0) AS total_amount
            FROM real_revenue_tracking
            WHERE metadata ? 'lead_id'
              AND (metadata->>'lead_id') = ANY($1::text[])
              AND COALESCE(is_demo, false) = false
            GROUP BY (metadata->>'lead_id')
            """,
            lead_id_strs,
        )
        for rr in rev_rows:
            lead_id_str = str(rr.get("lead_id") or "").strip()
            if not lead_id_str:
                continue
            revenue_by_lead[lead_id_str] = _to_float(rr.get("total_amount"))

    # Distribute revenue across a lead's sent emails so we don't over-credit high-volume sequences.
    email_count_by_lead: dict[str, int] = {}
    for item in email_rows:
        lid = item.get("lead_id")
        if not lid:
            continue
        s = str(lid)
        email_count_by_lead[s] = email_count_by_lead.get(s, 0) + 1

    cap = _to_float(os.getenv("DSPY_REWARD_CAP_AMOUNT") or 5000)

    training_samples: list[dict[str, Any]] = []
    stats = {
        "emails_considered": len(email_rows),
        "samples_built": 0,
        "unique_leads": len(leads_by_id),
        "total_revenue": float(sum(revenue_by_lead.values()) if revenue_by_lead else 0.0),
        "positive_samples": 0,
        "negative_samples": 0,
        "training_source_allowlist": sorted(list(allowed_sources)) if allowed_sources is not None else "*",
        "skipped_by_source": skipped_by_source,
    }

    for item in email_rows:
        lead_uuid = item.get("lead_id")
        if not lead_uuid:
            continue

        lead_id_str = str(lead_uuid)
        lead = leads_by_id.get(lead_id_str)
        if not lead:
            # Lead id not present in revenue_leads (email could be unrelated transactional/internal mail).
            continue
        if lead.get("is_test") or lead.get("is_demo"):
            continue

        total_rev = float(revenue_by_lead.get(lead_id_str, 0.0))
        n_emails = max(1, int(email_count_by_lead.get(lead_id_str, 1)))
        per_email_rev = total_rev / n_emails

        state = (
            state_by_lead.get(lead_id_str)
            or str(lead.get("stage") or "").strip().lower()
            or str(lead.get("status") or "").strip().lower()
        )

        reward_score = _normalize_revenue(per_email_rev, cap) if per_email_rev > 0 else _state_reward(state)
        reward_score = max(0.0, min(1.0, float(reward_score)))

        subject = str(item.get("subject") or "").strip()
        body = str(item.get("body") or "").strip()
        if not body:
            continue

        draft_email = "\n".join(
            [
                f"Subject: {subject}" if subject else "Subject:",
                "",
                body,
            ]
        ).strip()

        meta = _parse_json(item.get("metadata"))
        sequence_step = meta.get("sequence_step") or meta.get("touchpoint_day")

        leads_context = json.dumps(
            {
                "lead_id": lead_id_str,
                "company_name": lead.get("company_name"),
                "contact_name": lead.get("contact_name"),
                "email": lead.get("email"),
                "industry": lead.get("industry"),
                "source": lead.get("source"),
                "metadata": _parse_json(lead.get("metadata")),
            },
            default=str,
        )

        revenue_metrics = json.dumps(
            {
                "pipeline_state": state,
                "total_revenue_for_lead": total_rev,
                "revenue_credit_for_email": per_email_rev,
                "value_estimate": _to_float(lead.get("value_estimate")),
                "estimated_value": _to_float(lead.get("estimated_value")),
                "expected_revenue": _to_float(lead.get("expected_revenue")),
                "email_source": meta.get("source"),
                "sequence_step": sequence_step,
                "sent_at": item.get("sent_at").isoformat() if item.get("sent_at") else None,
            },
            default=str,
        )

        sample = {
            "leads": leads_context,
            "revenue_metrics": revenue_metrics,
            "email": draft_email,
            # We treat historical sent emails as our "successful patterns" label; reward_score weights them.
            "optimized_email": draft_email,
            "revenue_generated": per_email_rev,
            "reward_score": reward_score,
            "outcome": state,
        }
        training_samples.append(sample)

        stats["samples_built"] += 1
        if reward_score > 0:
            stats["positive_samples"] += 1
        else:
            stats["negative_samples"] += 1

    return training_samples, stats

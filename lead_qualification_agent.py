"""
Lead Qualification Agent

Qualifies and scores incoming leads for the ERP lead pipeline.

Design goals:
- No stubs: always produces a score using deterministic heuristics.
- AI-enhanced when configured: optionally uses an LLM to refine scoring and extract fields.
- Tenant-safe: updates the lead only when tenant_id matches.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from database.async_connection import get_pool

logger = logging.getLogger(__name__)


try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional dependency/runtime import
    AsyncOpenAI = None  # type: ignore[assignment]


FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _email_domain(email: str) -> str:
    email = _clean_text(email).lower()
    if "@" not in email:
        return ""
    return email.split("@", 1)[1].strip()


def _clamp_int(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, int(value)))


@dataclass(frozen=True)
class LeadQualificationResult:
    score: int
    grade: str
    priority: str
    project_type: str
    summary: str
    signals: list[str]
    ai_used: bool


class LeadQualificationAgent:
    """AI-assisted (optional) lead qualification + scoring."""

    def __init__(self, tenant_id: str):
        if not tenant_id:
            raise ValueError("tenant_id is required for LeadQualificationAgent")
        self.tenant_id = tenant_id

        self._enable_ai = os.getenv("ENABLE_LEAD_QUALIFICATION_AI", "true").lower() in {"1", "true", "yes"}
        self._openai_model = os.getenv("LEAD_QUALIFICATION_MODEL", "gpt-4o-mini")
        self._openai_client: Optional["AsyncOpenAI"] = None

        if self._enable_ai and AsyncOpenAI is not None and os.getenv("OPENAI_API_KEY"):
            try:
                self._openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as exc:  # pragma: no cover
                logger.warning("LeadQualificationAgent: failed to init OpenAI client: %s", exc)
                self._openai_client = None

    async def qualify_lead(self, lead_id: str, event_payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Qualify a lead by ID.

        - Reads the canonical lead record from `public.leads`.
        - Produces a score and updates `lead_score`, `score`, `priority`, `lead_grade`,
          `project_type`, and `ai_analysis`.
        - Adds a structured blob into `metadata.lead_qualification`.
        """
        now = datetime.now(timezone.utc)

        lead_uuid = self._coerce_uuid(lead_id)
        if not lead_uuid:
            return {"status": "error", "error": "invalid_lead_id", "lead_id": lead_id}

        pool = get_pool()
        lead = await pool.fetchrow(
            """
            select
              id,
              tenant_id,
              name,
              email,
              phone,
              company,
              company_name,
              source,
              urgency,
              project_type,
              address,
              property_address,
              message,
              description,
              metadata
            from public.leads
            where id = $1::uuid and tenant_id = $2::uuid
            """,
            str(lead_uuid),
            self.tenant_id,
        )

        if not lead:
            return {"status": "not_found", "lead_id": str(lead_uuid), "tenant_id": self.tenant_id}

        lead_dict: dict[str, Any] = dict(lead)
        existing_project_type = _clean_text(lead_dict.get("project_type"))

        heuristic = self._heuristic_qualify(lead_dict)
        final_result = heuristic

        ai_detail: dict[str, Any] = {}
        if self._openai_client is not None:
            try:
                ai_detail = await self._ai_refine(lead_dict, event_payload or {})
                final_result = self._merge_ai(heuristic, ai_detail)
            except Exception as exc:
                logger.warning("LeadQualificationAgent: AI refine failed (fallback to heuristics): %s", exc)

        # Preserve a user-selected project_type when already set.
        project_type = existing_project_type or final_result.project_type

        metadata_patch = {
            "lead_qualification": {
                "qualified_at": now.isoformat(),
                "score": final_result.score,
                "grade": final_result.grade,
                "priority": final_result.priority,
                "project_type": project_type,
                "signals": final_result.signals,
                "ai_used": final_result.ai_used,
                "ai_detail": ai_detail,
            }
        }

        await pool.execute(
            """
            update public.leads
            set
              lead_score = $1,
              score = $1,
              lead_grade = $2,
              priority = $3,
              project_type = $4,
              ai_analysis = $5,
              ai_enriched = true,
              score_updated_at = now(),
              metadata = coalesce(metadata, '{}'::jsonb) || $6::jsonb
            where id = $7::uuid and tenant_id = $8::uuid
            """,
            final_result.score,
            final_result.grade,
            final_result.priority,
            project_type,
            final_result.summary,
            json.dumps(metadata_patch),
            str(lead_uuid),
            self.tenant_id,
        )

        return {
            "status": "qualified",
            "lead_id": str(lead_uuid),
            "tenant_id": self.tenant_id,
            "score": final_result.score,
            "grade": final_result.grade,
            "priority": final_result.priority,
            "project_type": project_type,
            "ai_used": final_result.ai_used,
        }

    def _coerce_uuid(self, value: str) -> Optional[UUID]:
        try:
            return UUID(str(value))
        except Exception:
            return None

    def _heuristic_qualify(self, lead: dict[str, Any]) -> LeadQualificationResult:
        message = _clean_text(lead.get("message") or lead.get("description"))
        urgency = _clean_text(lead.get("urgency")).lower()
        source = _clean_text(lead.get("source")).lower()

        has_phone = bool(_clean_text(lead.get("phone")))
        has_address = bool(_clean_text(lead.get("address") or lead.get("property_address")))
        has_company = bool(_clean_text(lead.get("company") or lead.get("company_name")))

        signals: list[str] = []

        score = 50
        if has_phone:
            score += 10
            signals.append("phone_present")
        if has_address:
            score += 15
            signals.append("address_present")
        if len(message) >= 80:
            score += 10
            signals.append("detailed_message")
        if has_company:
            score += 5
            signals.append("company_present")

        if urgency in {"high", "urgent", "emergency"}:
            score += 10
            signals.append("high_urgency")

        # Source-based adjustments (soft signals only).
        if source in {"referral", "existing_customer"}:
            score += 5
            signals.append(f"source_{source}")

        domain = _email_domain(_clean_text(lead.get("email")))
        if domain and domain not in FREE_EMAIL_DOMAINS:
            score += 3
            signals.append("non_free_email_domain")

        score = _clamp_int(score)

        if score >= 85:
            grade = "A"
            priority = "high"
        elif score >= 70:
            grade = "B"
            priority = "medium"
        elif score >= 55:
            grade = "C"
            priority = "medium"
        else:
            grade = "D"
            priority = "low"

        project_type = _clean_text(lead.get("project_type")) or "unknown"

        summary = (
            f"Heuristic lead score={score} grade={grade} priority={priority}. "
            f"Signals: {', '.join(signals) if signals else 'none'}."
        )

        return LeadQualificationResult(
            score=score,
            grade=grade,
            priority=priority,
            project_type=project_type,
            summary=summary,
            signals=signals,
            ai_used=False,
        )

    async def _ai_refine(self, lead: dict[str, Any], event_payload: dict[str, Any]) -> dict[str, Any]:
        assert self._openai_client is not None

        # Redact obviously sensitive fields from prompt where possible.
        prompt_lead = {
            "name": _clean_text(lead.get("name")),
            "email_domain": _email_domain(_clean_text(lead.get("email"))),
            "has_phone": bool(_clean_text(lead.get("phone"))),
            "company": _clean_text(lead.get("company") or lead.get("company_name")),
            "source": _clean_text(lead.get("source")),
            "urgency": _clean_text(lead.get("urgency")),
            "project_type": _clean_text(lead.get("project_type")),
            "address_present": bool(_clean_text(lead.get("address") or lead.get("property_address"))),
            "message": _clean_text(lead.get("message") or lead.get("description")),
        }

        prompt_event = {
            "type_hints": sorted({str(k) for k in (event_payload or {}).keys()})[:50],
        }

        system = (
            "You are an enterprise lead qualification system for a roofing/contracting ERP. "
            "Return STRICT JSON only. Do not include markdown."
        )
        user = f"""Qualify the lead.\n\nLead:\n{json.dumps(prompt_lead, ensure_ascii=False)}\n\nEvent payload hints:\n{json.dumps(prompt_event, ensure_ascii=False)}\n\nReturn JSON with:\n{{\n  \"score\": 0-100,\n  \"grade\": \"A\"|\"B\"|\"C\"|\"D\",\n  \"priority\": \"low\"|\"medium\"|\"high\",\n  \"project_type\": string,\n  \"signals\": [string],\n  \"summary\": string\n}}\n\nRules:\n- Score must be integer 0-100.\n- Prefer project_type values like: \"repair\", \"replacement\", \"inspection\", \"gutter\", \"insurance\", \"commercial\", \"residential\", or \"unknown\".\n- If unsure, be conservative.\n"""

        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=500,
        )

        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract the first JSON object.
            match = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
        return data if isinstance(data, dict) else {}

    def _merge_ai(self, heuristic: LeadQualificationResult, ai: dict[str, Any]) -> LeadQualificationResult:
        try:
            score = _clamp_int(int(ai.get("score", heuristic.score)))
        except Exception:
            score = heuristic.score

        grade = _clean_text(ai.get("grade")) or heuristic.grade
        if grade not in {"A", "B", "C", "D"}:
            grade = heuristic.grade

        priority = _clean_text(ai.get("priority")) or heuristic.priority
        if priority not in {"low", "medium", "high"}:
            priority = heuristic.priority

        project_type = _clean_text(ai.get("project_type")) or heuristic.project_type or "unknown"

        signals_raw = ai.get("signals")
        signals: list[str]
        if isinstance(signals_raw, list):
            signals = [str(x) for x in signals_raw if str(x).strip()]
        else:
            signals = list(heuristic.signals)

        summary = _clean_text(ai.get("summary")) or heuristic.summary

        return LeadQualificationResult(
            score=score,
            grade=grade,
            priority=priority,
            project_type=project_type,
            summary=summary,
            signals=signals,
            ai_used=True,
        )


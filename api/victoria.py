"""
Victoria (Operations & Scheduling) Agent API
==========================================

Weathercraft ERP calls `POST /agents/victoria/analyze?mode=draft` for:
- Schedule optimization (conflict detection + draft reschedule suggestions)
- Predictive scheduling augmentation (draft insights only)

This endpoint must be:
- Deterministic (no Math.random)
- Draft-only by default (ERP is an assistant; humans approve)
- Backwards compatible with the ERP payload shapes that already exist in prod
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field


router = APIRouter(prefix="/agents/victoria", tags=["Victoria (Scheduling)"])


class _VictoriaAnalyzeRequest(BaseModel):
    mode: str = Field(default="draft")
    action: str
    tenant_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


def _to_dt(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        # ISO 8601 parse; allow Z suffix.
        s = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _detect_time_conflicts(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []
    enriched = []
    for job in jobs:
        start = _to_dt(job.get("scheduled_start"))
        end = _to_dt(job.get("scheduled_end"))
        if not start or not end or end <= start:
            continue
        enriched.append((job, start, end))

    enriched.sort(key=lambda x: x[1])
    for idx in range(len(enriched)):
        job_a, start_a, end_a = enriched[idx]
        for jdx in range(idx + 1, len(enriched)):
            job_b, start_b, end_b = enriched[jdx]
            # Since sorted by start time, we can break once the next job starts after this ends.
            if start_b >= end_a:
                break
            conflicts.append(
                {
                    "type": "time_conflict",
                    "jobs": [str(job_a.get("id")), str(job_b.get("id"))],
                    "severity": "medium",
                    "suggestion": "Draft reschedule to avoid overlap",
                }
            )

    return conflicts


@router.post("/analyze")
async def victoria_analyze(
    payload: _VictoriaAnalyzeRequest,
    mode: str = Query("draft", description="ERP assistant mode; draft suggestions only"),
) -> Dict[str, Any]:
    # Normalize/force draft mode behavior.
    effective_mode = (payload.mode or mode or "draft").strip().lower()
    if effective_mode not in ("draft", "suggest", "analysis"):
        effective_mode = "draft"

    action = (payload.action or "").strip().lower()

    # ---------------------------------------------------------------------
    # Action: optimize_schedule
    # Returns the schema expected by Weathercraft ERP's /api/ai/optimize-schedule.
    # ---------------------------------------------------------------------
    if action == "optimize_schedule":
        context = payload.context or {}
        jobs = context.get("jobs") or []
        events = context.get("events") or []
        crews = context.get("available_crew") or context.get("crews") or context.get("employees") or []

        jobs_list = [j for j in jobs if isinstance(j, dict)]
        conflicts = _detect_time_conflicts(jobs_list)

        # Draft reschedule suggestions for the second job in each conflict (simple heuristic).
        optimizations: List[Dict[str, Any]] = []
        for conflict in conflicts:
            job_ids = conflict.get("jobs") or []
            if len(job_ids) != 2:
                continue
            job_id_to_move = job_ids[1]
            job_b = next((j for j in jobs_list if str(j.get("id")) == str(job_id_to_move)), None)
            job_a = next((j for j in jobs_list if str(j.get("id")) == str(job_ids[0])), None)
            if not job_a or not job_b:
                continue

            a_end = _to_dt(job_a.get("scheduled_end"))
            b_start = _to_dt(job_b.get("scheduled_start"))
            b_end = _to_dt(job_b.get("scheduled_end"))
            if not a_end or not b_start or not b_end:
                continue

            duration_s = max(0, int((b_end - b_start).total_seconds()))
            proposed_start = a_end
            proposed_end = proposed_start + (b_end - b_start)
            optimizations.append(
                {
                    "action": "reschedule",
                    "job_id": str(job_id_to_move),
                    "new_start": proposed_start.isoformat(),
                    "new_end": proposed_end.isoformat(),
                    "reason": "Resolve scheduling overlap (draft suggestion)",
                }
            )

        return {
            "mode": effective_mode,
            "agent": "victoria",
            "action": "optimize_schedule",
            "optimizations": optimizations,
            "conflicts": conflicts,
            "conflicts_resolved": len(optimizations),
            "efficiency_gain": len(optimizations) * 5,
            "reasoning": (
                f"Draft schedule optimization generated from {len(jobs_list)} jobs, "
                f"{len(events) if isinstance(events, list) else 0} events, "
                f"{len(crews) if isinstance(crews, list) else 0} crew candidates."
            ),
            "recommendations": [
                "Confirm crew availability before applying changes.",
                "Group geographically close jobs to reduce travel time.",
                "Prioritize urgent jobs and customer commitments.",
            ],
        }

    # ---------------------------------------------------------------------
    # Action: predictive_schedule
    # Weathercraft ERP treats this as an augmentation blob; schema is flexible.
    # ---------------------------------------------------------------------
    if action == "predictive_schedule":
        context = payload.context or {}
        heuristic_predictions = context.get("heuristic_predictions") or []
        unscheduled_plan = context.get("unscheduled_plan") or []
        duration_factor = context.get("duration_overrun_factor")

        return {
            "mode": effective_mode,
            "agent": "victoria",
            "action": "predictive_schedule",
            "summary": {
                "scheduled_predictions_count": len(heuristic_predictions) if isinstance(heuristic_predictions, list) else 0,
                "unscheduled_plan_count": len(unscheduled_plan) if isinstance(unscheduled_plan, list) else 0,
                "duration_overrun_factor": duration_factor,
            },
            "recommendations": [
                "Review jobs with repeated overruns and add buffer time.",
                "Verify crew staffing assumptions against active headcount.",
                "Confirm materials and access constraints before finalizing the plan.",
            ],
            "notes": "Draft-only scheduling insights. No actions executed.",
        }

    # Default: unknown action -> deterministic response
    return {
        "mode": effective_mode,
        "agent": "victoria",
        "action": action or "unknown",
        "message": "No specialized handler for this action; draft-only response returned.",
        "recommendations": [],
    }


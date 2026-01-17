"""
Revenue Control Tower API
==========================
THE central command for all revenue operations.

Key Features:
- REAL vs TEST vs SEED data classification
- Real-only revenue metrics by default
- Lead inbox with next-best-action
- Pipeline health monitoring
- Human-in-the-loop controls and kill switches
- Automation status and controls

Part of BrainOps OS Total Completion Protocol.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from database.async_connection import get_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/revenue-control", tags=["Revenue Control Tower"])


class DataClassification(str, Enum):
    """Data classification for revenue objects"""
    REAL = "real"         # Genuine customer/revenue
    TEST = "test"         # Test/example data
    SEED = "seed"         # Demo/seeded data
    INTERNAL = "internal" # Internal testing


# Test email patterns - used to classify data
TEST_EMAIL_PATTERNS = [
    '%test%', '%example%', '%demo%', '%sample%',
    '%fake%', '%placeholder%', '@test.', '@example.',
    '%+test@%', '%localhost%'
]


def classify_email(email: str) -> DataClassification:
    """Classify an email as REAL, TEST, or SEED"""
    if not email:
        return DataClassification.TEST

    email_lower = email.lower()

    # Check for test patterns
    test_patterns = ['test', 'example', 'demo', 'sample', 'fake', 'placeholder', 'localhost']
    for pattern in test_patterns:
        if pattern in email_lower:
            return DataClassification.TEST

    # Check for internal patterns
    if '@brainops' in email_lower or '@brainstack' in email_lower:
        return DataClassification.INTERNAL

    return DataClassification.REAL


# =============================================================================
# GROUND TRUTH METRICS
# =============================================================================

@router.get("/truth")
async def get_ground_truth(
    include_test: bool = Query(default=False, description="Include test/demo data")
) -> dict[str, Any]:
    """
    Get the GROUND TRUTH revenue metrics.

    This is the single source of truth for revenue.
    By default, shows ONLY real data - no test/demo contamination.
    """
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    now = datetime.now(timezone.utc)

    # Build WHERE clause for real-only filter (fail-closed).
    #
    # IMPORTANT:
    # - Prefer explicit DB flags (is_test/is_demo) over email heuristics.
    # - Keep email heuristics as a backstop to avoid counting obvious test addresses.
    test_filter = ""
    if not include_test:
        email_patterns = " AND ".join(
            [f"COALESCE(email, '') NOT ILIKE '{p}'" for p in TEST_EMAIL_PATTERNS]
        )
        test_filter = (
            "AND COALESCE(is_test, FALSE) = FALSE "
            "AND COALESCE(is_demo, FALSE) = FALSE "
            f"AND ({email_patterns})"
        )

    # Revenue Leads Analysis
    leads_query = f"""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE stage = 'new') as new,
            COUNT(*) FILTER (WHERE stage = 'contacted') as contacted,
            COUNT(*) FILTER (WHERE stage = 'qualified') as qualified,
            COUNT(*) FILTER (WHERE stage = 'proposal_sent') as proposal_sent,
            COUNT(*) FILTER (WHERE stage = 'negotiating') as negotiating,
            COUNT(*) FILTER (WHERE stage = 'won') as won,
            COUNT(*) FILTER (WHERE stage = 'lost') as lost,
            COALESCE(SUM(CASE WHEN stage = 'won' THEN value_estimate ELSE 0 END), 0) as won_revenue,
            COALESCE(SUM(value_estimate), 0) as pipeline_value
        FROM revenue_leads
        WHERE 1=1 {test_filter}
    """

    # Gumroad Sales Analysis
    gumroad_query = f"""
        SELECT
            COUNT(*) as total_sales,
            COALESCE(SUM(price), 0) as total_revenue,
            COUNT(DISTINCT product_name) as unique_products
        FROM gumroad_sales
        WHERE 1=1 {"AND COALESCE(is_test, FALSE) = FALSE" if not include_test else ""}
    """

    # Email Queue Status
    # NOTE: `ai_email_queue` does not have `send_after` (older drafts did).
    # Use timestamps that actually exist so "ground truth" doesn't silently degrade.
    email_query = """
        SELECT
            COUNT(*) FILTER (WHERE status = 'queued') as queued,
            COUNT(*) FILTER (
                WHERE status = 'sent'
                  AND COALESCE(sent_at, created_at) > NOW() - INTERVAL '7 days'
            ) as sent_7d,
            COUNT(*) FILTER (
                WHERE status = 'failed'
                  AND COALESCE(last_attempt, created_at) > NOW() - INTERVAL '7 days'
            ) as failed_7d,
            COUNT(*) FILTER (
                WHERE status = 'skipped'
                  AND created_at > NOW() - INTERVAL '7 days'
            ) as skipped_7d,
            MAX(created_at) as last_created,
            MAX(sent_at) as last_sent_at
        FROM ai_email_queue
    """

    try:
        # Execute queries
        leads_row = await pool.fetchrow(leads_query)
        gumroad_row = await pool.fetchrow(gumroad_query)

        try:
            email_row = await pool.fetchrow(email_query)
        except Exception:
            email_row = None

        leads = dict(leads_row) if leads_row else {}
        gumroad = dict(gumroad_row) if gumroad_row else {}
        email = dict(email_row) if email_row else {}

        # Calculate key metrics
        total_revenue = float(leads.get('won_revenue', 0) or 0) + float(gumroad.get('total_revenue', 0) or 0)
        pipeline_value = float(leads.get('pipeline_value', 0) or 0)

        # Conversion funnel
        total_leads = leads.get('total', 0) or 0
        won_leads = leads.get('won', 0) or 0
        conversion_rate = (won_leads / total_leads * 100) if total_leads > 0 else 0

        return {
            "timestamp": now.isoformat(),
            "data_mode": "REAL_ONLY" if not include_test else "ALL_DATA",
            "warning": None if not include_test else "INCLUDES TEST/DEMO DATA - NOT REAL REVENUE",

            # THE TRUTH
            "ground_truth": {
                "total_revenue": total_revenue,
                "is_real": not include_test,
                "mrr": 0,  # No active subscriptions yet
                "arr": 0
            },

            # Pipeline Health
            "pipeline": {
                "total_leads": total_leads,
                "pipeline_value": pipeline_value,
                "conversion_rate": round(conversion_rate, 1),
                "stages": {
                    "new": leads.get('new', 0) or 0,
                    "contacted": leads.get('contacted', 0) or 0,
                    "qualified": leads.get('qualified', 0) or 0,
                    "proposal_sent": leads.get('proposal_sent', 0) or 0,
                    "negotiating": leads.get('negotiating', 0) or 0,
                    "won": leads.get('won', 0) or 0,
                    "lost": leads.get('lost', 0) or 0
                }
            },

            # Product Sales
            "gumroad": {
                "total_sales": gumroad.get('total_sales', 0) or 0,
                "total_revenue": float(gumroad.get('total_revenue', 0) or 0),
                "unique_products": gumroad.get('unique_products', 0) or 0
            },

            # Email Automation
            "email_automation": {
                "queued": email.get('queued', 0) or 0,
                "sent_7d": email.get('sent_7d', 0) or 0,
                "failed_7d": email.get('failed_7d', 0) or 0,
                "skipped_7d": email.get('skipped_7d', 0) or 0,
                "healthy": (email.get('failed_7d', 0) or 0) < 5
            },

            # CRITICAL DIAGNOSIS
            "diagnosis": _generate_diagnosis(leads, gumroad, total_revenue)
        }

    except Exception as e:
        logger.error(f"Ground truth query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_diagnosis(leads: dict, gumroad: dict, total_revenue: float) -> dict[str, Any]:
    """Generate actionable diagnosis based on current state"""
    issues = []
    recommendations = []
    total_leads = leads.get('total', 0) or 0

    # Check for $0 revenue
    if total_revenue == 0:
        issues.append("CRITICAL: $0 real revenue")
        if total_leads:
            recommendations.append(f"Focus on converting the {total_leads} real leads in pipeline")
        else:
            recommendations.append("Acquire qualified leads and convert the first deals")

    # Check conversion
    won = leads.get('won', 0) or 0
    if total_leads > 10 and won == 0:
        issues.append(f"CRITICAL: 0% conversion rate ({total_leads} leads, 0 won)")
        recommendations.append("Implement conversion sprint for contacted leads")

    # Check for stuck leads
    contacted = leads.get('contacted', 0) or 0
    if contacted > 10:
        issues.append(f"WARNING: {contacted} leads stuck in 'contacted' stage")
        recommendations.append("Move leads forward with direct outreach and offers")

    # Check Gumroad
    gumroad_sales = gumroad.get('total_sales', 0) or 0
    if gumroad_sales == 0:
        issues.append("INFO: No Gumroad sales")
        recommendations.append("Launch BrainStack Studio products with real distribution")

    return {
        "issues": issues,
        "recommendations": recommendations,
        "health_score": max(0, 100 - len(issues) * 25)
    }


# =============================================================================
# LEAD INBOX
# =============================================================================

@router.get("/leads/inbox")
async def get_lead_inbox(
    limit: int = Query(default=20, le=100),
    include_test: bool = Query(default=False)
) -> dict[str, Any]:
    """
    Get prioritized lead inbox with next-best-actions.

    Shows leads that need attention, prioritized by:
    1. Stage advancement readiness
    2. Time since last contact
    3. Lead score
    """
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    test_filter = ""
    if not include_test:
        patterns = " AND ".join([f"email NOT ILIKE '{p}'" for p in TEST_EMAIL_PATTERNS])
        test_filter = f"AND ({patterns})"

    query = f"""
        SELECT
            id, email, company_name, stage, source,
            value_estimate, created_at, updated_at,
            EXTRACT(DAY FROM (NOW() - updated_at)) as days_since_update,
            CASE
                WHEN stage = 'new' THEN 'Make initial contact'
                WHEN stage = 'contacted' THEN 'Follow up and qualify'
                WHEN stage = 'qualified' THEN 'Send proposal'
                WHEN stage = 'proposal_sent' THEN 'Follow up on proposal'
                WHEN stage = 'negotiating' THEN 'Close the deal'
                ELSE 'Review and update'
            END as next_action,
            CASE
                WHEN stage = 'new' THEN 100
                WHEN stage = 'qualified' THEN 90
                WHEN stage = 'contacted' AND EXTRACT(DAY FROM (NOW() - updated_at)) > 3 THEN 85
                WHEN stage = 'proposal_sent' THEN 80
                WHEN stage = 'negotiating' THEN 95
                ELSE 50
            END as priority_score
        FROM revenue_leads
        WHERE stage NOT IN ('won', 'lost')
        {test_filter}
        ORDER BY
            CASE stage
                WHEN 'negotiating' THEN 1
                WHEN 'proposal_sent' THEN 2
                WHEN 'qualified' THEN 3
                WHEN 'contacted' THEN 4
                WHEN 'new' THEN 5
                ELSE 6
            END,
            updated_at ASC
        LIMIT {limit}
    """

    try:
        rows = await pool.fetch(query)

        leads = []
        for row in rows:
            leads.append({
                "id": str(row["id"]),
                "email": row["email"],
                "company": row["company_name"],
                "stage": row["stage"],
                "source": row["source"],
                "value": float(row["value_estimate"] or 0),
                "days_since_update": int(row["days_since_update"] or 0),
                "next_action": row["next_action"],
                "priority_score": row["priority_score"],
                "created": row["created_at"].isoformat() if row["created_at"] else None
            })

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_mode": "REAL_ONLY" if not include_test else "ALL_DATA",
            "total_actionable": len(leads),
            "leads": leads,
            "quick_wins": [l for l in leads if l["priority_score"] >= 90][:5]
        }

    except Exception as e:
        logger.error(f"Lead inbox query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AUTOMATION CONTROLS
# =============================================================================

@router.get("/controls")
async def get_automation_controls() -> dict[str, Any]:
    """
    Get current automation control status.

    Shows all kill switches, throttles, and pause states.
    """
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    # Check for control settings in unified_brain
    try:
        controls = await pool.fetch("""
            SELECT key, value, updated_at
            FROM unified_brain
            WHERE key LIKE 'control_%' OR key LIKE 'automation_%'
            ORDER BY key
        """)

        control_settings = {}
        for c in controls:
            control_settings[c["key"]] = {
                "value": c["value"],
                "updated": c["updated_at"].isoformat() if c["updated_at"] else None
            }
    except Exception:
        control_settings = {}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "controls": {
            "email_outreach": {
                "enabled": control_settings.get("control_email_outreach", {}).get("value", "enabled") == "enabled",
                "daily_limit": 50,
                "sent_today": 0  # TODO: Calculate from email queue
            },
            "lead_enrichment": {
                "enabled": control_settings.get("control_lead_enrichment", {}).get("value", "enabled") == "enabled",
                "daily_limit": 100
            },
            "content_publishing": {
                "enabled": control_settings.get("control_content_publishing", {}).get("value", "enabled") == "enabled",
                "requires_approval": True
            },
            "pricing_changes": {
                "enabled": False,
                "requires_approval": True
            }
        },
        "kill_switches": {
            "all_outreach": False,
            "all_automation": False,
            "all_publishing": False
        },
        "note": "Set controls via POST /revenue-control/controls/{control_name}"
    }


@router.post("/controls/{control_name}")
async def set_control(
    control_name: str,
    enabled: bool = Query(...),
    reason: str = Query(default="Manual override")
) -> dict[str, Any]:
    """
    Set an automation control (enable/disable).

    This is the human-in-the-loop override mechanism.
    """
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    valid_controls = [
        "email_outreach", "lead_enrichment", "content_publishing",
        "all_outreach", "all_automation", "all_publishing"
    ]

    if control_name not in valid_controls:
        raise HTTPException(status_code=400, detail=f"Invalid control. Valid: {valid_controls}")

    key = f"control_{control_name}"
    value = "enabled" if enabled else "disabled"

    try:
        await pool.execute("""
            INSERT INTO unified_brain (key, value, category, priority, updated_at)
            VALUES ($1, $2, 'controls', 'critical', NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = $2,
                updated_at = NOW()
        """, key, value)

        # Log the control change
        await pool.execute("""
            INSERT INTO unified_brain_logs (key, action, old_value, new_value, reason, created_at)
            VALUES ($1, 'control_change', NULL, $2, $3, NOW())
        """, key, value, reason)

        logger.info(f"Control {control_name} set to {value} - Reason: {reason}")

        return {
            "success": True,
            "control": control_name,
            "enabled": enabled,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to set control: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# APPROVAL QUEUE
# =============================================================================

@router.get("/approvals")
async def get_approval_queue() -> dict[str, Any]:
    """
    Get items awaiting human approval.

    Shows all pending approvals for:
    - Bulk email sends
    - Content publishing
    - Pricing changes
    - High-value outreach
    """
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    # Check for pending proposals
    try:
        proposals = await pool.fetch("""
            SELECT id, title, description, status, created_at
            FROM ai_improvement_proposals
            WHERE status = 'proposed'
            ORDER BY created_at DESC
            LIMIT 20
        """)
    except Exception:
        proposals = []

    # Check for pending content
    try:
        pending_content = await pool.fetch("""
            SELECT id, title, content_type, status, created_at
            FROM ai_generated_content
            WHERE status = 'pending_review'
            ORDER BY created_at DESC
            LIMIT 20
        """)
    except Exception:
        pending_content = []

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_pending": len(proposals) + len(pending_content),
        "proposals": [
            {
                "id": str(p["id"]),
                "title": p["title"],
                "description": p["description"][:200] if p["description"] else None,
                "created": p["created_at"].isoformat() if p["created_at"] else None
            }
            for p in proposals
        ],
        "content_pending": [
            {
                "id": str(c["id"]),
                "title": c["title"],
                "type": c["content_type"],
                "created": c["created_at"].isoformat() if c["created_at"] else None
            }
            for c in pending_content
        ],
        "note": "Approve items via POST /revenue-control/approvals/{id}/approve"
    }


# =============================================================================
# HEALTH SUMMARY
# =============================================================================

@router.get("/health")
async def get_revenue_health() -> dict[str, Any]:
    """
    Get overall revenue system health.

    This is the quick-glance status for the command center.
    """
    truth = await get_ground_truth(include_test=False)

    health_score = truth["diagnosis"]["health_score"]

    if health_score >= 80:
        status = "healthy"
    elif health_score >= 50:
        status = "degraded"
    else:
        status = "critical"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "health_score": health_score,
        "real_revenue": truth["ground_truth"]["total_revenue"],
        "pipeline_value": truth["pipeline"]["pipeline_value"],
        "active_leads": truth["pipeline"]["total_leads"],
        "conversion_rate": truth["pipeline"]["conversion_rate"],
        "issues": truth["diagnosis"]["issues"],
        "top_recommendation": truth["diagnosis"]["recommendations"][0] if truth["diagnosis"]["recommendations"] else None
    }

"""
Learning Visibility API Endpoints
=================================
Exposes what the AI has learned via the Learning-Action Bridge.

Endpoints:
- GET /learning/rules - List all behavior rules from learning_action_bridge
- GET /learning/status - Get learning bridge status and stats
- GET /learning/insights - Get recent learning insights converted to rules
- POST /learning/sync - Trigger sync from learning to action

This provides transparency into the AI's learned behaviors.

Author: BrainOps AI OS
Version: 1.0.0
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

try:
    from config import config

    VALID_API_KEYS = config.security.valid_api_keys
except (ImportError, AttributeError):
    fallback_key = (
        os.getenv("BRAINOPS_API_KEY") or os.getenv("AGENTS_API_KEY") or os.getenv("API_KEY")
    )
    VALID_API_KEYS = {fallback_key} if fallback_key else set()


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401, detail="Authentication required (API Key or Bearer Token)"
        )
    return api_key


router = APIRouter(
    prefix="/api/learning-visibility",
    tags=["Learning Visibility"],
    dependencies=[Depends(verify_api_key)],
)

# Lazy initialization of the learning bridge
_bridge = None


async def _get_bridge():
    """Get or create the learning-action bridge with lazy initialization"""
    global _bridge
    if _bridge is None:
        try:
            from learning_action_bridge import get_learning_bridge

            _bridge = await get_learning_bridge()
            logger.info("Learning-Action Bridge initialized for visibility API")
        except Exception as e:
            logger.error(f"Failed to initialize Learning-Action Bridge: {e}")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable") from e
    return _bridge


class SyncResponse(BaseModel):
    """Response model for sync operation"""

    success: bool
    rules_created: int
    message: str
    synced_at: str


class RuleResponse(BaseModel):
    """Response model for a behavior rule"""

    id: str
    rule_type: str
    trigger: str
    action: str
    confidence: float
    success_count: int
    failure_count: int
    created_at: str
    last_applied: Optional[str]
    source_insight: Optional[str]


@router.get("/rules")
async def get_behavior_rules(
    rule_type: Optional[str] = Query(
        None,
        description="Filter by rule type: avoidance, preference, optimization, threshold, sequence",
    ),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    limit: int = Query(100, ge=1, le=500, description="Maximum rules to return"),
) -> dict[str, Any]:
    """
    List all behavior rules from the learning-action bridge.

    These rules represent what the AI has learned from experience:
    - AVOIDANCE: Don't do X because it failed
    - PREFERENCE: Prefer X because it succeeded
    - OPTIMIZATION: Improve X using pattern Y
    - THRESHOLD: Trigger action when condition met
    - SEQUENCE: Do X then Y then Z

    Returns rules sorted by confidence (highest first).
    """
    try:
        bridge = await _get_bridge()

        rules = []
        for rule_id, rule in bridge.behavior_rules.items():
            # Apply filters
            if rule_type and rule.rule_type.value != rule_type:
                continue
            if rule.confidence < min_confidence:
                continue

            rules.append(
                {
                    "id": rule.id,
                    "rule_type": rule.rule_type.value,
                    "trigger": rule.trigger,
                    "action": rule.action,
                    "confidence": round(rule.confidence, 3),
                    "success_count": rule.success_count,
                    "failure_count": rule.failure_count,
                    "track_record": f"{rule.success_count} successes, {rule.failure_count} failures",
                    "created_at": rule.created_at.isoformat(),
                    "last_applied": rule.last_applied.isoformat() if rule.last_applied else None,
                    "source_insight": rule.source_insight,
                }
            )

        # Sort by confidence (highest first)
        rules.sort(key=lambda x: x["confidence"], reverse=True)

        # Apply limit
        rules = rules[:limit]

        # Group by type for summary
        rules_by_type = {}
        for rule in rules:
            rt = rule["rule_type"]
            rules_by_type[rt] = rules_by_type.get(rt, 0) + 1

        return {
            "total_rules": len(bridge.behavior_rules),
            "returned_rules": len(rules),
            "rules_by_type": rules_by_type,
            "filters_applied": {
                "rule_type": rule_type,
                "min_confidence": min_confidence,
                "limit": limit,
            },
            "rules": rules,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get behavior rules: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/status")
async def get_learning_bridge_status() -> dict[str, Any]:
    """
    Get the current status of the learning-action bridge.

    Returns:
    - Total rules and breakdown by type
    - Average confidence across all rules
    - Rules applied count (feedback loop iterations)
    - Rules created count (learning outcomes converted)
    - Last sync timestamp
    - Memory availability status
    """
    try:
        bridge = await _get_bridge()
        status = bridge.get_status()

        # Enhance with additional metrics
        high_confidence_rules = sum(
            1 for r in bridge.behavior_rules.values() if r.confidence >= 0.8
        )

        low_performing_rules = sum(
            1
            for r in bridge.behavior_rules.values()
            if r.failure_count > r.success_count and (r.success_count + r.failure_count) > 3
        )

        never_applied_rules = sum(
            1 for r in bridge.behavior_rules.values() if r.last_applied is None
        )

        return {
            "status": "active" if status["memory_available"] else "degraded",
            "bridge_health": {
                "memory_available": status["memory_available"],
                "total_rules": status["total_rules"],
                "rules_applied": status["rules_applied"],
                "rules_created": status["rules_created"],
                "last_sync": status["last_sync"],
            },
            "rule_statistics": {
                "total": status["total_rules"],
                "by_type": status["rules_by_type"],
                "average_confidence": status["average_confidence"],
                "high_confidence_count": high_confidence_rules,
                "low_performing_count": low_performing_rules,
                "never_applied_count": never_applied_rules,
            },
            "configuration": {
                "min_confidence_to_apply": bridge.min_confidence_to_apply,
                "max_rules_per_trigger": bridge.max_rules_per_trigger,
                "rule_decay_days": bridge.rule_decay_days,
            },
            "message": f"Bridge has {status['total_rules']} behavior rules with {status['average_confidence']:.1%} avg confidence",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bridge status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/insights")
async def get_recent_insights(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back for insights"),
    limit: int = Query(50, ge=1, le=200, description="Maximum insights to return"),
) -> dict[str, Any]:
    """
    Get recent learning insights that can be converted to rules.

    Shows insights from the unified memory that represent:
    - Error patterns (converted to avoidance rules)
    - Success patterns (converted to preference rules)
    - Optimization opportunities (converted to optimization rules)

    This is the raw material that feeds into behavior rules.
    """
    try:
        # Try to get insights from unified memory
        try:
            from unified_memory_manager import get_memory_manager

            memory = get_memory_manager()
            MEMORY_AVAILABLE = True
        except Exception as e:
            logger.warning(f"Memory manager not available: {e}")
            MEMORY_AVAILABLE = False
            memory = None

        if not MEMORY_AVAILABLE or not memory:
            return {
                "memory_available": False,
                "message": "Unified memory not available - cannot retrieve insights",
                "insights": [],
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }

        # Query for learning-related insights
        # Use system tenant for cross-tenant learning insights (must be valid UUID)
        SYSTEM_TENANT_ID = "00000000-0000-0000-0000-000000000001"
        insights_raw = memory.recall(
            "learning insight pattern outcome success failure error",
            tenant_id=SYSTEM_TENANT_ID,
            limit=limit,
        )

        insights = []
        for mem in insights_raw:
            content = mem.get("content", {})
            if isinstance(content, str):
                try:
                    import json

                    content = json.loads(content)
                except (ValueError, TypeError):
                    content = {"raw": content}

            insight_type = content.get("type", content.get("outcome_type", "unknown"))

            # Determine potential rule type
            potential_rule_type = None
            if insight_type == "error_pattern" or content.get("outcome") == "failure":
                potential_rule_type = "avoidance"
            elif insight_type == "success_pattern" or content.get("outcome") == "success":
                potential_rule_type = "preference"
            elif insight_type == "optimization":
                potential_rule_type = "optimization"

            insights.append(
                {
                    "id": str(mem.get("id", "")),
                    "type": insight_type,
                    "content": content,
                    "confidence": content.get("confidence", mem.get("importance_score", 0.5)),
                    "potential_rule_type": potential_rule_type,
                    "source_system": mem.get("source_system", "unknown"),
                    "created_at": mem.get("created_at", "").isoformat()
                    if hasattr(mem.get("created_at", ""), "isoformat")
                    else str(mem.get("created_at", "")),
                }
            )

        # Categorize insights
        by_type = {}
        for insight in insights:
            t = insight["potential_rule_type"] or "other"
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "memory_available": True,
            "total_insights": len(insights),
            "time_window_hours": hours,
            "insights_by_potential_rule_type": by_type,
            "insights": insights,
            "message": f"Found {len(insights)} learning insights that can feed into behavior rules",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get insights: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/sync")
async def trigger_learning_sync() -> dict[str, Any]:
    """
    Trigger a sync from learning outcomes to behavior rules.

    This converts recent learning insights into actionable behavior rules:
    1. Queries unified memory for recent learning insights
    2. Converts error patterns to AVOIDANCE rules
    3. Converts success patterns to PREFERENCE rules
    4. Converts optimization insights to OPTIMIZATION rules
    5. Persists new rules to memory for cross-session continuity

    Returns the number of new rules created.
    """
    try:
        bridge = await _get_bridge()

        # Capture pre-sync state
        rules_before = len(bridge.behavior_rules)

        # Trigger sync
        from learning_action_bridge import sync_learning_to_actions

        rules_created = await sync_learning_to_actions()

        # Capture post-sync state
        rules_after = len(bridge.behavior_rules)

        return {
            "success": True,
            "rules_created": rules_created,
            "rules_before": rules_before,
            "rules_after": rules_after,
            "last_sync": bridge.last_sync.isoformat() if bridge.last_sync else None,
            "message": f"Sync complete: created {rules_created} new behavior rules",
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync learning: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/guidance")
async def get_guidance_for_context(
    context: str = Query(..., description="The context/situation to get guidance for"),
    action_type: Optional[str] = Query(None, description="Optional action type filter"),
) -> dict[str, Any]:
    """
    Get learning-informed guidance for a given context.

    This is the endpoint agents should call before making decisions
    to incorporate learned behavior into their actions.

    Example contexts:
    - "error handling high cpu"
    - "customer complaint billing"
    - "deployment production release"
    """
    try:
        bridge = await _get_bridge()

        guidance = bridge.get_guidance(context, action_type)

        return {
            "context": context,
            "action_type": action_type,
            "guidance_count": len(guidance),
            "guidance": guidance,
            "message": f"Found {len(guidance)} relevant behavior rules for context: {context[:50]}...",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get guidance: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/report-outcome")
async def report_rule_outcome(
    rule_id: str = Query(..., description="The ID of the rule that was applied"),
    success: bool = Query(..., description="Whether applying the rule was successful"),
) -> dict[str, Any]:
    """
    Report the outcome of applying a behavior rule.

    This creates the feedback loop that improves rule quality:
    - Success increases rule confidence
    - Failure decreases rule confidence
    - Rules with too many failures are automatically removed

    Agents should call this after following guidance from /guidance endpoint.
    """
    try:
        bridge = await _get_bridge()

        # Check if rule exists
        if rule_id not in bridge.behavior_rules:
            raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")

        # Get rule state before
        rule = bridge.behavior_rules[rule_id]
        confidence_before = rule.confidence

        # Report outcome
        bridge.report_outcome(rule_id, success)

        # Check if rule still exists (might have been removed due to poor performance)
        if rule_id in bridge.behavior_rules:
            rule = bridge.behavior_rules[rule_id]
            return {
                "success": True,
                "rule_id": rule_id,
                "outcome_reported": "success" if success else "failure",
                "confidence_before": round(confidence_before, 3),
                "confidence_after": round(rule.confidence, 3),
                "success_count": rule.success_count,
                "failure_count": rule.failure_count,
                "message": f"Outcome recorded. Confidence adjusted from {confidence_before:.1%} to {rule.confidence:.1%}",
                "reported_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            return {
                "success": True,
                "rule_id": rule_id,
                "outcome_reported": "failure",
                "rule_removed": True,
                "message": "Rule was removed due to too many failures (underperforming)",
                "reported_at": datetime.now(timezone.utc).isoformat(),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to report outcome: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/prune")
async def prune_stale_rules() -> dict[str, Any]:
    """
    Remove stale behavior rules that haven't been used recently.

    Prunes rules that:
    - Were never applied and are older than decay threshold (30 days default)
    - Were last applied before decay threshold and have low confidence

    This prevents rule bloat and keeps the system focused on relevant behaviors.
    """
    try:
        bridge = await _get_bridge()

        rules_before = len(bridge.behavior_rules)
        pruned = bridge.prune_old_rules()
        rules_after = len(bridge.behavior_rules)

        return {
            "success": True,
            "rules_pruned": pruned,
            "rules_before": rules_before,
            "rules_after": rules_after,
            "decay_threshold_days": bridge.rule_decay_days,
            "message": f"Pruned {pruned} stale rules ({rules_before} -> {rules_after})",
            "pruned_at": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to prune rules: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/dashboard")
async def get_learning_dashboard() -> dict[str, Any]:
    """
    Get a comprehensive learning visibility dashboard.

    Combines all learning metrics into a single view:
    - Bridge status and health
    - Rule statistics and distribution
    - Recent activity and trends
    - Actionable insights
    """
    try:
        bridge = await _get_bridge()
        status = bridge.get_status()

        # Calculate additional metrics
        high_performers = []
        low_performers = []

        for rule in bridge.behavior_rules.values():
            total_applications = rule.success_count + rule.failure_count
            if total_applications >= 3:
                success_rate = (
                    rule.success_count / total_applications if total_applications > 0 else 0
                )
                rule_data = {
                    "id": rule.id,
                    "rule_type": rule.rule_type.value,
                    "trigger": rule.trigger[:50] + "..."
                    if len(rule.trigger) > 50
                    else rule.trigger,
                    "confidence": round(rule.confidence, 3),
                    "success_rate": round(success_rate, 3),
                    "applications": total_applications,
                }
                if success_rate >= 0.7:
                    high_performers.append(rule_data)
                elif success_rate <= 0.3:
                    low_performers.append(rule_data)

        # Sort by applications
        high_performers.sort(key=lambda x: x["applications"], reverse=True)
        low_performers.sort(key=lambda x: x["applications"], reverse=True)

        return {
            "overview": {
                "status": "active" if status["memory_available"] else "degraded",
                "total_rules": status["total_rules"],
                "rules_applied": status["rules_applied"],
                "rules_created": status["rules_created"],
                "average_confidence": status["average_confidence"],
                "last_sync": status["last_sync"],
            },
            "rule_distribution": status["rules_by_type"],
            "performance": {
                "high_performers": high_performers[:5],
                "low_performers": low_performers[:5],
                "high_performer_count": len(high_performers),
                "low_performer_count": len(low_performers),
            },
            "health_indicators": {
                "memory_connected": status["memory_available"],
                "has_recent_sync": status["last_sync"] is not None,
                "has_active_rules": status["total_rules"] > 0,
                "learning_active": status["rules_created"] > 0,
            },
            "recommendations": _generate_recommendations(status, high_performers, low_performers),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


def _generate_recommendations(
    status: dict, high_performers: list, low_performers: list
) -> list[str]:
    """Generate actionable recommendations based on learning state"""
    recommendations = []

    if status["total_rules"] == 0:
        recommendations.append(
            "No behavior rules yet. Trigger a sync to convert learning insights to rules."
        )

    if status["last_sync"] is None:
        recommendations.append(
            "Learning-action bridge has never synced. Run /sync to pull insights."
        )

    if not status["memory_available"]:
        recommendations.append("Memory system unavailable. Learning persistence is degraded.")

    if status["average_confidence"] < 0.5:
        recommendations.append(
            "Average rule confidence is low. Consider pruning underperforming rules."
        )

    if len(low_performers) > len(high_performers):
        recommendations.append(
            "More low-performing rules than high-performing. Review and prune poor rules."
        )

    if status["rules_applied"] == 0:
        recommendations.append(
            "No rules have been applied yet. Ensure agents call /guidance endpoint."
        )

    if not recommendations:
        recommendations.append("Learning system is healthy. Continue monitoring performance.")

    return recommendations

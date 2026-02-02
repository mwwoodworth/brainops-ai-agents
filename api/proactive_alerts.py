"""
Proactive Alerts API Router
============================
Generates proactive AI recommendations by analyzing:
- Agent execution patterns and suggesting optimizations
- Memory for revenue/growth opportunities
- System behavior anomalies
- Actionable recommendations with priorities

Created: 2026-02-02
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============================================================================
# API Key Security (consistent with other routers)
# =============================================================================
try:
    from config import config
    VALID_API_KEYS = config.security.valid_api_keys
except (ImportError, AttributeError):
    fallback_key = os.getenv("BRAINOPS_API_KEY") or os.getenv("AGENTS_API_KEY") or os.getenv("API_KEY")
    VALID_API_KEYS = {fallback_key} if fallback_key else set()

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


router = APIRouter(
    prefix="/proactive",
    tags=["Proactive Alerts"],
    dependencies=[Depends(verify_api_key)]
)


# =============================================================================
# Data Models
# =============================================================================
class AlertPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertType(str, Enum):
    OPPORTUNITY = "opportunity"
    OPTIMIZATION = "optimization"
    ANOMALY = "anomaly"
    RECOMMENDATION = "recommendation"
    WARNING = "warning"


@dataclass
class ProactiveAlert:
    """Represents a single proactive alert or recommendation"""
    alert_type: AlertType
    title: str
    recommendation: str
    action: str
    priority: AlertPriority
    source: str
    confidence: float = 0.8
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.alert_type.value,
            "title": self.title,
            "recommendation": self.recommendation,
            "action": self.action,
            "priority": self.priority.value,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class AnalysisRequest(BaseModel):
    """Request model for proactive analysis"""
    focus_areas: Optional[list[str]] = Field(
        default=None,
        description="Specific areas to analyze: agents, memory, revenue, system"
    )
    time_window_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Time window for analysis (1-168 hours)"
    )
    include_ai_insights: bool = Field(
        default=True,
        description="Whether to include AI-powered insights"
    )


# =============================================================================
# Lazy-loaded dependencies
# =============================================================================
_pool = None
_memory_manager = None
_ai_intelligence = None
_analyzer = None


async def _get_pool():
    """Lazy load database pool"""
    global _pool
    if _pool is None:
        try:
            from database.async_connection import get_pool
            _pool = get_pool()
        except Exception as e:
            logger.warning(f"Database pool not available: {e}")
            _pool = None
    return _pool


def _get_memory_manager():
    """Lazy load memory manager"""
    global _memory_manager
    if _memory_manager is None:
        try:
            from unified_memory_manager import get_memory_manager
            _memory_manager = get_memory_manager()
        except Exception as e:
            logger.warning(f"Memory manager not available: {e}")
            _memory_manager = None
    return _memory_manager


def _get_ai_intelligence():
    """Lazy load AI intelligence engine"""
    global _ai_intelligence
    if _ai_intelligence is None:
        try:
            from ai_intelligence import TrueAIIntelligence
            _ai_intelligence = TrueAIIntelligence()
        except Exception as e:
            logger.warning(f"AI Intelligence not available: {e}")
            _ai_intelligence = None
    return _ai_intelligence


# =============================================================================
# Core Analysis Engine
# =============================================================================
class ProactiveAnalyzer:
    """
    Proactive Analysis Engine
    Analyzes system patterns and generates actionable recommendations
    """

    def __init__(self):
        self._last_analysis: Optional[datetime] = None
        self._cached_alerts: list[ProactiveAlert] = []
        self._cache_ttl_seconds = 300  # 5 minute cache

    async def analyze_agent_patterns(self, time_window_hours: int = 24) -> list[ProactiveAlert]:
        """Analyze agent execution patterns and suggest optimizations"""
        alerts = []
        pool = await _get_pool()

        if not pool:
            return alerts

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

            # Query agent execution patterns
            query = """
                SELECT
                    agent_name,
                    COUNT(*) as execution_count,
                    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failure_count,
                    MAX(started_at) as last_run
                FROM ai_agent_executions
                WHERE started_at >= $1
                GROUP BY agent_name
                ORDER BY execution_count DESC
            """
            rows = await pool.fetch(query, cutoff)

            for row in rows:
                agent_name = row['agent_name']
                execution_count = row['execution_count'] or 0
                success_count = row['success_count'] or 0
                failure_count = row['failure_count'] or 0
                last_run = row['last_run']
                avg_duration = row['avg_duration_seconds'] or 0

                # Check for agents that haven't run recently
                if last_run:
                    hours_since_run = (datetime.now(timezone.utc) - last_run.replace(tzinfo=timezone.utc)).total_seconds() / 3600
                    if hours_since_run > 72:  # 3 days
                        alerts.append(ProactiveAlert(
                            alert_type=AlertType.OPPORTUNITY,
                            title=f"{agent_name} hasn't run in {int(hours_since_run)} hours",
                            recommendation="Running this agent regularly may improve system performance",
                            action="Schedule agent run",
                            priority=AlertPriority.MEDIUM,
                            source="agent_patterns",
                            metadata={"agent_name": agent_name, "hours_since_run": hours_since_run}
                        ))

                # Check for high failure rates
                if execution_count > 5:
                    failure_rate = failure_count / execution_count
                    if failure_rate > 0.3:  # >30% failure rate
                        alerts.append(ProactiveAlert(
                            alert_type=AlertType.WARNING,
                            title=f"{agent_name} has {int(failure_rate*100)}% failure rate",
                            recommendation="Investigate agent failures and fix root cause",
                            action="Review agent logs and configuration",
                            priority=AlertPriority.HIGH if failure_rate > 0.5 else AlertPriority.MEDIUM,
                            source="agent_patterns",
                            confidence=0.9,
                            metadata={
                                "agent_name": agent_name,
                                "failure_rate": failure_rate,
                                "failure_count": failure_count,
                                "total_executions": execution_count
                            }
                        ))

                # Check for slow agents
                if avg_duration > 300 and execution_count > 3:  # > 5 minutes avg
                    alerts.append(ProactiveAlert(
                        alert_type=AlertType.OPTIMIZATION,
                        title=f"{agent_name} averaging {int(avg_duration)}s per execution",
                        recommendation="Consider optimizing agent performance or splitting into smaller tasks",
                        action="Profile and optimize agent",
                        priority=AlertPriority.LOW,
                        source="agent_patterns",
                        metadata={"agent_name": agent_name, "avg_duration_seconds": avg_duration}
                    ))

            # Check for underutilized revenue agents
            revenue_agents = ['RevenueOptimizer', 'LeadQualifier', 'OutreachAgent', 'ConversionOptimizer']
            agent_names_found = {r['agent_name'] for r in rows}

            for rev_agent in revenue_agents:
                if rev_agent not in agent_names_found:
                    alerts.append(ProactiveAlert(
                        alert_type=AlertType.OPPORTUNITY,
                        title=f"{rev_agent} not running",
                        recommendation=f"Running {rev_agent} weekly correlates with higher conversions",
                        action="Schedule agent run",
                        priority=AlertPriority.MEDIUM,
                        source="agent_patterns",
                        metadata={"agent_name": rev_agent, "agent_type": "revenue"}
                    ))

        except Exception as e:
            logger.error(f"Error analyzing agent patterns: {e}")

        return alerts

    async def analyze_memory_opportunities(self, time_window_hours: int = 24) -> list[ProactiveAlert]:
        """Analyze memory for revenue/growth opportunities"""
        alerts = []
        pool = await _get_pool()

        if not pool:
            return alerts

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

            # Check for unprocessed revenue leads
            try:
                lead_query = """
                    SELECT
                        COUNT(*) as total_leads,
                        COUNT(*) FILTER (WHERE status = 'new') as new_leads,
                        COUNT(*) FILTER (WHERE status = 'qualified') as qualified_leads,
                        COUNT(*) FILTER (WHERE status = 'contacted') as contacted_leads,
                        SUM(estimated_value) FILTER (WHERE status IN ('new', 'qualified')) as pipeline_value
                    FROM revenue_leads
                    WHERE created_at >= $1
                """
                lead_row = await pool.fetchrow(lead_query, cutoff)

                if lead_row:
                    new_leads = lead_row['new_leads'] or 0
                    qualified_leads = lead_row['qualified_leads'] or 0
                    pipeline_value = float(lead_row['pipeline_value'] or 0)

                    if new_leads > 5:
                        alerts.append(ProactiveAlert(
                            alert_type=AlertType.OPPORTUNITY,
                            title=f"{new_leads} unprocessed new leads",
                            recommendation="Qualify and contact these leads to increase conversion rate",
                            action="Run lead qualification workflow",
                            priority=AlertPriority.HIGH,
                            source="memory_opportunities",
                            confidence=0.95,
                            metadata={"new_leads": new_leads, "pipeline_value": pipeline_value}
                        ))

                    if qualified_leads > 3 and pipeline_value > 1000:
                        alerts.append(ProactiveAlert(
                            alert_type=AlertType.OPPORTUNITY,
                            title=f"${pipeline_value:,.0f} in qualified pipeline needs outreach",
                            recommendation="Send proposals to qualified leads to close deals",
                            action="Generate and send proposals",
                            priority=AlertPriority.HIGH,
                            source="memory_opportunities",
                            metadata={"qualified_leads": qualified_leads, "pipeline_value": pipeline_value}
                        ))

            except Exception as e:
                logger.debug(f"Revenue leads table query failed (may not exist): {e}")

            # Check for customer insights in memory
            try:
                memory_query = """
                    SELECT
                        COUNT(*) as total_memories,
                        COUNT(*) FILTER (WHERE importance_score >= 0.8) as high_importance,
                        COUNT(*) FILTER (WHERE tags @> ARRAY['customer']) as customer_insights,
                        COUNT(*) FILTER (WHERE tags @> ARRAY['revenue']) as revenue_insights
                    FROM unified_ai_memory
                    WHERE created_at >= $1
                    AND archived = FALSE
                """
                mem_row = await pool.fetchrow(memory_query, cutoff)

                if mem_row:
                    high_importance = mem_row['high_importance'] or 0
                    customer_insights = mem_row['customer_insights'] or 0
                    revenue_insights = mem_row['revenue_insights'] or 0

                    if high_importance > 10:
                        alerts.append(ProactiveAlert(
                            alert_type=AlertType.RECOMMENDATION,
                            title=f"{high_importance} high-importance memories to review",
                            recommendation="Review recent high-importance memories for actionable insights",
                            action="Review memory insights",
                            priority=AlertPriority.MEDIUM,
                            source="memory_opportunities",
                            metadata={
                                "high_importance_count": high_importance,
                                "customer_insights": customer_insights,
                                "revenue_insights": revenue_insights
                            }
                        ))

            except Exception as e:
                logger.debug(f"Memory query failed: {e}")

            # Check for Gumroad sales opportunities
            try:
                gumroad_query = """
                    SELECT
                        COUNT(*) as total_sales,
                        SUM(price) as total_revenue,
                        COUNT(DISTINCT email) as unique_customers
                    FROM gumroad_sales
                    WHERE created_at >= $1
                """
                gumroad_row = await pool.fetchrow(gumroad_query, cutoff)

                if gumroad_row and gumroad_row['total_sales']:
                    total_sales = gumroad_row['total_sales']
                    total_revenue = float(gumroad_row['total_revenue'] or 0)
                    unique_customers = gumroad_row['unique_customers'] or 0

                    if unique_customers > 0:
                        alerts.append(ProactiveAlert(
                            alert_type=AlertType.RECOMMENDATION,
                            title=f"{unique_customers} Gumroad customers - upsell opportunity",
                            recommendation="Send targeted upsell campaigns to recent purchasers",
                            action="Create upsell email sequence",
                            priority=AlertPriority.MEDIUM,
                            source="memory_opportunities",
                            metadata={
                                "total_sales": total_sales,
                                "total_revenue": total_revenue,
                                "unique_customers": unique_customers
                            }
                        ))

            except Exception as e:
                logger.debug(f"Gumroad sales query failed (may not exist): {e}")

        except Exception as e:
            logger.error(f"Error analyzing memory opportunities: {e}")

        return alerts

    async def analyze_system_anomalies(self, time_window_hours: int = 24) -> list[ProactiveAlert]:
        """Look for anomalies in system behavior"""
        alerts = []
        pool = await _get_pool()

        if not pool:
            return alerts

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

            # Check for error spikes in logs
            try:
                error_query = """
                    SELECT
                        system,
                        action,
                        COUNT(*) as error_count
                    FROM unified_brain_logs
                    WHERE created_at >= $1
                    AND (data->>'error' IS NOT NULL OR data->>'status' = 'error')
                    GROUP BY system, action
                    HAVING COUNT(*) > 5
                    ORDER BY error_count DESC
                    LIMIT 10
                """
                error_rows = await pool.fetch(error_query, cutoff)

                for row in error_rows:
                    system = row['system']
                    action = row['action']
                    error_count = row['error_count']

                    alerts.append(ProactiveAlert(
                        alert_type=AlertType.ANOMALY,
                        title=f"{error_count} errors in {system}/{action}",
                        recommendation=f"Investigate recurring errors in {system} - {action} action",
                        action="Review error logs and fix root cause",
                        priority=AlertPriority.HIGH if error_count > 20 else AlertPriority.MEDIUM,
                        source="system_anomalies",
                        confidence=0.85,
                        metadata={"system": system, "action": action, "error_count": error_count}
                    ))

            except Exception as e:
                logger.debug(f"Error log query failed: {e}")

            # Check for performance degradation
            try:
                perf_query = """
                    SELECT
                        endpoint,
                        AVG(response_time_ms) as avg_response_time,
                        COUNT(*) as request_count,
                        MAX(response_time_ms) as max_response_time
                    FROM api_request_logs
                    WHERE created_at >= $1
                    GROUP BY endpoint
                    HAVING AVG(response_time_ms) > 1000
                    ORDER BY avg_response_time DESC
                    LIMIT 5
                """
                perf_rows = await pool.fetch(perf_query, cutoff)

                for row in perf_rows:
                    endpoint = row['endpoint']
                    avg_response = row['avg_response_time'] or 0

                    alerts.append(ProactiveAlert(
                        alert_type=AlertType.OPTIMIZATION,
                        title=f"Slow endpoint: {endpoint} ({int(avg_response)}ms avg)",
                        recommendation="Optimize this endpoint for better user experience",
                        action="Profile and optimize endpoint",
                        priority=AlertPriority.MEDIUM,
                        source="system_anomalies",
                        metadata={"endpoint": endpoint, "avg_response_ms": avg_response}
                    ))

            except Exception as e:
                logger.debug(f"Performance query failed (table may not exist): {e}")

            # Check database connection pool health
            try:
                if hasattr(pool, '_pool') and pool._pool:
                    pool_size = pool._pool.get_size()
                    free_size = pool._pool.get_idle_size()

                    if pool_size > 0:
                        utilization = (pool_size - free_size) / pool_size
                        if utilization > 0.8:
                            alerts.append(ProactiveAlert(
                                alert_type=AlertType.WARNING,
                                title=f"Database pool {int(utilization*100)}% utilized",
                                recommendation="Consider increasing pool size or optimizing queries",
                                action="Review connection pool settings",
                                priority=AlertPriority.HIGH,
                                source="system_anomalies",
                                metadata={
                                    "pool_size": pool_size,
                                    "free_connections": free_size,
                                    "utilization": utilization
                                }
                            ))
            except Exception as e:
                logger.debug(f"Pool health check failed: {e}")

        except Exception as e:
            logger.error(f"Error analyzing system anomalies: {e}")

        return alerts

    async def get_ai_insights(
        self,
        alerts: list[ProactiveAlert],
        time_window_hours: int = 24
    ) -> list[ProactiveAlert]:
        """Use AI to generate additional insights from the alerts"""
        ai_intelligence = _get_ai_intelligence()

        if not ai_intelligence or not alerts:
            return []

        ai_alerts = []

        try:
            # Summarize current alerts for AI analysis
            alert_summary = "\n".join([
                f"- {a.alert_type.value}: {a.title} (Priority: {a.priority.value})"
                for a in alerts[:10]  # Limit to top 10 for context
            ])

            context = {
                "total_alerts": len(alerts),
                "alert_types": list(set(a.alert_type.value for a in alerts)),
                "high_priority_count": sum(1 for a in alerts if a.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]),
                "time_window_hours": time_window_hours
            }

            # Get AI analysis
            from ai_intelligence import AnalysisDepth
            analysis = await ai_intelligence.analyze_issue(
                f"""Analyze these BrainOps system alerts and suggest strategic recommendations:

{alert_summary}

What patterns do you see? What should be the top priority actions?
What revenue or growth opportunities might be hidden in these alerts?""",
                context=context,
                depth=AnalysisDepth.QUICK
            )

            if analysis and analysis.fix_strategies:
                for strategy in analysis.fix_strategies[:3]:  # Top 3 AI recommendations
                    ai_alerts.append(ProactiveAlert(
                        alert_type=AlertType.RECOMMENDATION,
                        title=strategy.get('description', 'AI-generated recommendation'),
                        recommendation=strategy.get('action', analysis.reasoning[:200]),
                        action=strategy.get('action', 'Review and act'),
                        priority=AlertPriority.MEDIUM,
                        source="ai_intelligence",
                        confidence=strategy.get('confidence', 0.7),
                        metadata={"ai_model": analysis.model_used, "reasoning": analysis.reasoning[:500]}
                    ))

        except Exception as e:
            logger.warning(f"AI insights generation failed: {e}")

        return ai_alerts

    async def run_full_analysis(
        self,
        focus_areas: Optional[list[str]] = None,
        time_window_hours: int = 24,
        include_ai_insights: bool = True
    ) -> dict[str, Any]:
        """Run complete proactive analysis"""

        # Check cache
        now = datetime.now(timezone.utc)
        if (
            self._last_analysis
            and (now - self._last_analysis).total_seconds() < self._cache_ttl_seconds
            and self._cached_alerts
        ):
            # Return cached results
            return {
                "alerts": [a.to_dict() for a in self._cached_alerts],
                "opportunities": [
                    a.to_dict() for a in self._cached_alerts
                    if a.alert_type == AlertType.OPPORTUNITY
                ],
                "analyzed_at": self._last_analysis.isoformat(),
                "cached": True,
                "total_alerts": len(self._cached_alerts),
                "by_priority": self._count_by_priority(self._cached_alerts)
            }

        # Determine which analyses to run
        areas = focus_areas or ['agents', 'memory', 'system']
        all_alerts: list[ProactiveAlert] = []

        # Run analyses in parallel
        tasks = []
        if 'agents' in areas:
            tasks.append(self.analyze_agent_patterns(time_window_hours))
        if 'memory' in areas or 'revenue' in areas:
            tasks.append(self.analyze_memory_opportunities(time_window_hours))
        if 'system' in areas:
            tasks.append(self.analyze_system_anomalies(time_window_hours))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_alerts.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Analysis task failed: {result}")

        # Get AI insights if requested
        if include_ai_insights and all_alerts:
            ai_alerts = await self.get_ai_insights(all_alerts, time_window_hours)
            all_alerts.extend(ai_alerts)

        # Sort by priority
        priority_order = {
            AlertPriority.CRITICAL: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3
        }
        all_alerts.sort(key=lambda a: (priority_order.get(a.priority, 4), -a.confidence))

        # Cache results
        self._cached_alerts = all_alerts
        self._last_analysis = now

        return {
            "alerts": [a.to_dict() for a in all_alerts],
            "opportunities": [
                a.to_dict() for a in all_alerts
                if a.alert_type == AlertType.OPPORTUNITY
            ],
            "analyzed_at": now.isoformat(),
            "cached": False,
            "total_alerts": len(all_alerts),
            "by_priority": self._count_by_priority(all_alerts),
            "by_type": self._count_by_type(all_alerts),
            "focus_areas": areas,
            "time_window_hours": time_window_hours
        }

    def _count_by_priority(self, alerts: list[ProactiveAlert]) -> dict[str, int]:
        """Count alerts by priority"""
        counts = {p.value: 0 for p in AlertPriority}
        for alert in alerts:
            counts[alert.priority.value] += 1
        return counts

    def _count_by_type(self, alerts: list[ProactiveAlert]) -> dict[str, int]:
        """Count alerts by type"""
        counts = {t.value: 0 for t in AlertType}
        for alert in alerts:
            counts[alert.alert_type.value] += 1
        return counts


def _get_analyzer() -> ProactiveAnalyzer:
    """Get or create the singleton analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = ProactiveAnalyzer()
    return _analyzer


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/status")
async def get_proactive_status():
    """Get proactive alerts system status"""
    pool = await _get_pool()
    memory_manager = _get_memory_manager()
    ai_intelligence = _get_ai_intelligence()

    return {
        "system": "proactive_alerts",
        "status": "operational",
        "database_connected": pool is not None,
        "memory_manager_available": memory_manager is not None,
        "ai_intelligence_available": ai_intelligence is not None,
        "capabilities": [
            "agent_pattern_analysis",
            "memory_opportunity_detection",
            "system_anomaly_detection",
            "ai_powered_insights",
            "revenue_opportunity_identification"
        ]
    }


@router.get("/alerts")
async def get_proactive_alerts(
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    priority: Optional[str] = Query(None, description="Filter by priority: critical, high, medium, low"),
    alert_type: Optional[str] = Query(None, description="Filter by type: opportunity, optimization, anomaly, recommendation, warning"),
    limit: int = Query(50, ge=1, le=200, description="Maximum alerts to return"),
    api_key: str = Depends(verify_api_key)
):
    """
    Get current proactive recommendations

    Analyzes:
    - Agent execution patterns and suggests optimizations
    - Memory for revenue and growth opportunities
    - System behavior for anomalies
    """
    analyzer = _get_analyzer()

    try:
        result = await analyzer.run_full_analysis(
            time_window_hours=time_window_hours,
            include_ai_insights=False  # Faster response without AI
        )

        alerts = result.get("alerts", [])

        # Apply filters
        if priority:
            alerts = [a for a in alerts if a.get("priority") == priority]
        if alert_type:
            alerts = [a for a in alerts if a.get("type") == alert_type]

        # Apply limit
        alerts = alerts[:limit]

        return {
            "alerts": alerts,
            "total": len(alerts),
            "analyzed_at": result.get("analyzed_at"),
            "cached": result.get("cached", False),
            "by_priority": result.get("by_priority", {}),
            "filters_applied": {
                "priority": priority,
                "type": alert_type,
                "time_window_hours": time_window_hours
            }
        }

    except Exception as e:
        logger.error(f"Error getting proactive alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/opportunities")
async def get_revenue_opportunities(
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    limit: int = Query(20, ge=1, le=100, description="Maximum opportunities to return"),
    api_key: str = Depends(verify_api_key)
):
    """
    Get revenue and growth opportunities

    Specifically focuses on:
    - Unprocessed leads and pipeline value
    - Customer upsell opportunities
    - Agent optimizations that could improve revenue
    """
    analyzer = _get_analyzer()

    try:
        result = await analyzer.run_full_analysis(
            focus_areas=['memory', 'revenue', 'agents'],
            time_window_hours=time_window_hours,
            include_ai_insights=True  # AI helps find revenue opportunities
        )

        # Filter to opportunities only
        opportunities = result.get("opportunities", [])[:limit]

        # Calculate potential value if available
        total_potential_value = sum(
            opp.get("metadata", {}).get("pipeline_value", 0)
            for opp in opportunities
        )

        return {
            "opportunities": opportunities,
            "total": len(opportunities),
            "total_potential_value": total_potential_value,
            "analyzed_at": result.get("analyzed_at"),
            "recommendation": "Review and act on high-priority opportunities to maximize revenue"
        }

    except Exception as e:
        logger.error(f"Error getting opportunities: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.post("/analyze")
async def run_proactive_analysis(
    request: AnalysisRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Run proactive analysis now

    Triggers immediate analysis of:
    - Agent execution patterns
    - Memory for opportunities
    - System anomalies
    - AI-powered insights (optional)

    Returns comprehensive recommendations.
    """
    analyzer = _get_analyzer()

    # Invalidate cache to force fresh analysis
    analyzer._last_analysis = None
    analyzer._cached_alerts = []

    try:
        result = await analyzer.run_full_analysis(
            focus_areas=request.focus_areas,
            time_window_hours=request.time_window_hours,
            include_ai_insights=request.include_ai_insights
        )

        return {
            "success": True,
            "alerts": result.get("alerts", []),
            "opportunities": result.get("opportunities", []),
            "analyzed_at": result.get("analyzed_at"),
            "total_alerts": result.get("total_alerts", 0),
            "by_priority": result.get("by_priority", {}),
            "by_type": result.get("by_type", {}),
            "analysis_config": {
                "focus_areas": request.focus_areas or ['agents', 'memory', 'system'],
                "time_window_hours": request.time_window_hours,
                "ai_insights_included": request.include_ai_insights
            }
        }

    except Exception as e:
        logger.error(f"Error running proactive analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/dashboard")
async def get_proactive_dashboard(
    api_key: str = Depends(verify_api_key)
):
    """
    Get complete proactive alerts dashboard

    Returns a comprehensive view of all alerts, opportunities,
    and system recommendations with summary statistics.
    """
    analyzer = _get_analyzer()

    try:
        result = await analyzer.run_full_analysis(
            time_window_hours=24,
            include_ai_insights=True
        )

        alerts = result.get("alerts", [])
        opportunities = result.get("opportunities", [])

        # Get top actions
        top_actions = [
            {
                "action": a.get("action"),
                "reason": a.get("title"),
                "priority": a.get("priority"),
                "type": a.get("type")
            }
            for a in alerts[:5]
        ]

        return {
            "summary": {
                "total_alerts": len(alerts),
                "critical_alerts": sum(1 for a in alerts if a.get("priority") == "critical"),
                "high_priority_alerts": sum(1 for a in alerts if a.get("priority") == "high"),
                "opportunities": len(opportunities),
                "analyzed_at": result.get("analyzed_at")
            },
            "top_actions": top_actions,
            "by_priority": result.get("by_priority", {}),
            "by_type": result.get("by_type", {}),
            "alerts": alerts,
            "opportunities": opportunities,
            "system_health": {
                "database_connected": await _get_pool() is not None,
                "memory_manager_available": _get_memory_manager() is not None,
                "ai_intelligence_available": _get_ai_intelligence() is not None
            }
        }

    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

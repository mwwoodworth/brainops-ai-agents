#!/usr/bin/env python3
"""
AI Cost Optimization Engine - Task 21
Intelligent resource management and cost reduction system
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - validate required environment variables
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "5432"))
    }

class ResourceType(Enum):
    """Types of resources to optimize"""
    COMPUTE = "compute"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    DATABASE = "database"
    BANDWIDTH = "bandwidth"
    AI_TOKENS = "ai_tokens"
    THIRD_PARTY = "third_party"

class OptimizationStrategy(Enum):
    """Cost optimization strategies"""
    CACHING = "caching"
    BATCHING = "batching"
    SCHEDULING = "scheduling"
    TIERING = "tiering"
    COMPRESSION = "compression"
    DEDUPLICATION = "deduplication"
    AUTOSCALING = "autoscaling"
    SPOT_INSTANCES = "spot_instances"

class CostLevel(Enum):
    """Cost severity levels"""
    CRITICAL = "critical"  # >$100/day
    HIGH = "high"         # $50-100/day
    MEDIUM = "medium"     # $20-50/day
    LOW = "low"          # $5-20/day
    MINIMAL = "minimal"   # <$5/day

class ResourceMonitor:
    """Monitor resource usage and costs"""

    def __init__(self):
        self.usage_cache = {}
        self.cost_thresholds = {
            ResourceType.AI_TOKENS: 1000,  # $1 per 1000 tokens
            ResourceType.API_CALLS: 100,   # $0.01 per call
            ResourceType.COMPUTE: 50,      # $50/day for compute
            ResourceType.STORAGE: 10,      # $10/month per TB
            ResourceType.DATABASE: 20,     # $20/day for queries
            ResourceType.BANDWIDTH: 5      # $5/GB
        }

    async def track_usage(
        self,
        resource_type: ResourceType,
        amount: float,
        metadata: dict
    ) -> dict:
        """Track resource usage"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            usage_id = str(uuid.uuid4())

            # Calculate estimated cost
            cost = self._calculate_cost(resource_type, amount)

            # Store usage record
            cursor.execute("""
                INSERT INTO ai_resource_usage (
                    id, resource_type, amount, cost,
                    metadata, timestamp
                ) VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                usage_id, resource_type.value, amount,
                cost, json.dumps(metadata)
            ))

            conn.commit()
            cursor.close()
            conn.close()

            return {
                "usage_id": usage_id,
                "resource_type": resource_type.value,
                "amount": amount,
                "cost": cost,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error tracking usage: {e}")
            raise

    def _calculate_cost(
        self,
        resource_type: ResourceType,
        amount: float
    ) -> float:
        """Calculate estimated cost based on resource type"""
        if resource_type == ResourceType.AI_TOKENS:
            # OpenAI pricing: ~$0.002 per 1K tokens
            return (amount / 1000) * 0.002
        elif resource_type == ResourceType.API_CALLS:
            # API gateway: ~$3.50 per million calls
            return (amount / 1000000) * 3.50
        elif resource_type == ResourceType.COMPUTE:
            # Render compute: ~$7/month for basic
            return amount * 0.23  # Daily cost
        elif resource_type == ResourceType.STORAGE:
            # Supabase storage: ~$0.021 per GB/month
            return (amount * 0.021) / 30  # Daily cost
        elif resource_type == ResourceType.DATABASE:
            # Database operations
            return amount * 0.00001  # Per query estimate
        elif resource_type == ResourceType.BANDWIDTH:
            # Bandwidth: ~$0.09 per GB
            return amount * 0.09
        else:
            return 0.0

    async def get_usage_summary(
        self,
        days: int = 7
    ) -> dict:
        """Get usage summary for period"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get usage by resource type
            cursor.execute("""
                SELECT
                    resource_type,
                    SUM(amount) as total_amount,
                    SUM(cost) as total_cost,
                    COUNT(*) as usage_count,
                    AVG(cost) as avg_cost
                FROM ai_resource_usage
                WHERE timestamp > NOW() - INTERVAL '%s days'
                GROUP BY resource_type
                ORDER BY total_cost DESC
            """, (days,))

            usage_by_type = cursor.fetchall()

            # Get daily trends
            cursor.execute("""
                SELECT
                    DATE(timestamp) as date,
                    SUM(cost) as daily_cost,
                    COUNT(*) as operations
                FROM ai_resource_usage
                WHERE timestamp > NOW() - INTERVAL '%s days'
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (days,))

            daily_trends = cursor.fetchall()

            # Get top consumers
            cursor.execute("""
                SELECT
                    metadata->>'service' as service,
                    COUNT(*) as operations,
                    SUM(cost) as total_cost
                FROM ai_resource_usage
                WHERE timestamp > NOW() - INTERVAL '%s days'
                  AND metadata->>'service' IS NOT NULL
                GROUP BY metadata->>'service'
                ORDER BY total_cost DESC
                LIMIT 10
            """, (days,))

            top_consumers = cursor.fetchall()

            cursor.close()
            conn.close()

            return {
                "period_days": days,
                "usage_by_type": [dict(row) for row in usage_by_type],
                "daily_trends": [dict(row) for row in daily_trends],
                "top_consumers": [dict(row) for row in top_consumers],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting usage summary: {e}")
            raise

class CostOptimizer:
    """Optimize costs across all resources"""

    def __init__(self):
        self.strategies = {}
        self.optimization_cache = {}

    async def analyze_costs(
        self,
        usage_data: dict
    ) -> list[dict]:
        """Analyze costs and identify optimization opportunities"""
        opportunities = []

        for usage in usage_data.get('usage_by_type', []):
            resource_type = usage['resource_type']
            total_cost = float(usage.get('total_cost', 0))

            # Check if optimization needed
            if total_cost > 10:  # More than $10
                opportunity = await self._identify_opportunity(
                    resource_type,
                    usage
                )
                if opportunity:
                    opportunities.append(opportunity)

        # Sort by potential savings
        opportunities.sort(
            key=lambda x: x.get('potential_savings', 0),
            reverse=True
        )

        return opportunities

    async def _identify_opportunity(
        self,
        resource_type: str,
        usage: dict
    ) -> Optional[dict]:
        """Identify optimization opportunity for resource"""
        if resource_type == ResourceType.AI_TOKENS.value:
            return self._optimize_ai_tokens(usage)
        elif resource_type == ResourceType.API_CALLS.value:
            return self._optimize_api_calls(usage)
        elif resource_type == ResourceType.DATABASE.value:
            return self._optimize_database(usage)
        elif resource_type == ResourceType.COMPUTE.value:
            return self._optimize_compute(usage)
        else:
            return None

    def _optimize_ai_tokens(self, usage: dict) -> dict:
        """Optimize AI token usage"""
        total_cost = float(usage.get('total_cost', 0))
        potential_savings = total_cost * 0.3  # Can save ~30% with optimization

        return {
            "resource_type": ResourceType.AI_TOKENS.value,
            "current_cost": total_cost,
            "potential_savings": potential_savings,
            "strategy": OptimizationStrategy.CACHING.value,
            "recommendations": [
                "Implement response caching for common queries",
                "Use embeddings cache for similar requests",
                "Batch multiple requests together",
                "Use smaller models for simple tasks",
                "Implement prompt optimization"
            ],
            "implementation_effort": "medium",
            "payback_period_days": 7
        }

    def _optimize_api_calls(self, usage: dict) -> dict:
        """Optimize API call costs"""
        total_cost = float(usage.get('total_cost', 0))
        usage_count = usage.get('usage_count', 0)

        # Check if batching would help
        if usage_count > 1000:
            potential_savings = total_cost * 0.4
            strategy = OptimizationStrategy.BATCHING.value
        else:
            potential_savings = total_cost * 0.2
            strategy = OptimizationStrategy.CACHING.value

        return {
            "resource_type": ResourceType.API_CALLS.value,
            "current_cost": total_cost,
            "potential_savings": potential_savings,
            "strategy": strategy,
            "recommendations": [
                "Batch API calls to reduce overhead",
                "Implement request deduplication",
                "Cache API responses locally",
                "Use webhooks instead of polling",
                "Optimize retry logic"
            ],
            "implementation_effort": "low",
            "payback_period_days": 3
        }

    def _optimize_database(self, usage: dict) -> dict:
        """Optimize database costs"""
        total_cost = float(usage.get('total_cost', 0))

        return {
            "resource_type": ResourceType.DATABASE.value,
            "current_cost": total_cost,
            "potential_savings": total_cost * 0.5,
            "strategy": OptimizationStrategy.CACHING.value,
            "recommendations": [
                "Implement query result caching",
                "Add database connection pooling",
                "Optimize slow queries with indexes",
                "Use read replicas for analytics",
                "Archive old data to cold storage"
            ],
            "implementation_effort": "medium",
            "payback_period_days": 10
        }

    def _optimize_compute(self, usage: dict) -> dict:
        """Optimize compute costs"""
        total_cost = float(usage.get('total_cost', 0))

        return {
            "resource_type": ResourceType.COMPUTE.value,
            "current_cost": total_cost,
            "potential_savings": total_cost * 0.25,
            "strategy": OptimizationStrategy.AUTOSCALING.value,
            "recommendations": [
                "Implement auto-scaling policies",
                "Use spot instances for batch jobs",
                "Schedule heavy workloads during off-peak",
                "Optimize container resource limits",
                "Implement function-level caching"
            ],
            "implementation_effort": "high",
            "payback_period_days": 14
        }

    async def apply_optimization(
        self,
        optimization_id: str,
        strategy: OptimizationStrategy,
        parameters: dict
    ) -> dict:
        """Apply optimization strategy"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            # Record optimization
            cursor.execute("""
                INSERT INTO ai_cost_optimizations (
                    id, strategy, parameters, status,
                    estimated_savings, applied_at
                ) VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                optimization_id,
                strategy.value,
                json.dumps(parameters),
                'active',
                parameters.get('potential_savings', 0)
            ))

            conn.commit()

            # Apply specific strategy
            if strategy == OptimizationStrategy.CACHING:
                result = await self._apply_caching(parameters)
            elif strategy == OptimizationStrategy.BATCHING:
                result = await self._apply_batching(parameters)
            elif strategy == OptimizationStrategy.SCHEDULING:
                result = await self._apply_scheduling(parameters)
            else:
                result = {"status": "strategy_not_implemented"}

            cursor.close()
            conn.close()

            return {
                "optimization_id": optimization_id,
                "strategy": strategy.value,
                "status": "applied",
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error applying optimization: {e}")
            raise

    async def _apply_caching(self, parameters: dict) -> dict:
        """Apply caching optimization"""
        cache_ttl = parameters.get('cache_ttl', 3600)
        cache_size = parameters.get('cache_size', 1000)

        # Configure caching parameters
        return {
            "cache_enabled": True,
            "cache_ttl": cache_ttl,
            "cache_size": cache_size,
            "expected_hit_rate": 0.7
        }

    async def _apply_batching(self, parameters: dict) -> dict:
        """Apply batching optimization"""
        batch_size = parameters.get('batch_size', 100)
        batch_timeout = parameters.get('batch_timeout', 1000)

        return {
            "batching_enabled": True,
            "batch_size": batch_size,
            "batch_timeout_ms": batch_timeout
        }

    async def _apply_scheduling(self, parameters: dict) -> dict:
        """Apply scheduling optimization"""
        off_peak_hours = parameters.get('off_peak_hours', [0, 6])

        return {
            "scheduling_enabled": True,
            "off_peak_start": off_peak_hours[0],
            "off_peak_end": off_peak_hours[1],
            "cost_reduction": 0.3
        }

class BudgetManager:
    """Manage budgets and spending limits"""

    def __init__(self):
        self.budgets = {}
        self.alerts_sent = {}

    async def set_budget(
        self,
        service: str,
        monthly_limit: float,
        alert_threshold: float = 0.8
    ) -> dict:
        """Set budget for a service"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            budget_id = str(uuid.uuid4())

            cursor.execute("""
                INSERT INTO ai_budgets (
                    id, service, monthly_limit,
                    alert_threshold, status, created_at
                ) VALUES (%s, %s, %s, %s, 'active', NOW())
                ON CONFLICT (service) DO UPDATE SET
                    monthly_limit = EXCLUDED.monthly_limit,
                    alert_threshold = EXCLUDED.alert_threshold,
                    updated_at = NOW()
            """, (
                budget_id, service, monthly_limit, alert_threshold
            ))

            conn.commit()
            cursor.close()
            conn.close()

            return {
                "budget_id": budget_id,
                "service": service,
                "monthly_limit": monthly_limit,
                "alert_threshold": alert_threshold,
                "status": "active"
            }

        except Exception as e:
            logger.error(f"Error setting budget: {e}")
            raise

    async def check_budget_status(
        self,
        service: Optional[str] = None
    ) -> list[dict]:
        """Check budget status for services"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get budget status
            if service:
                cursor.execute("""
                    WITH current_spending AS (
                        SELECT
                            metadata->>'service' as service,
                            SUM(cost) as month_to_date
                        FROM ai_resource_usage
                        WHERE DATE_TRUNC('month', timestamp) = DATE_TRUNC('month', NOW())
                          AND metadata->>'service' = %s
                        GROUP BY metadata->>'service'
                    )
                    SELECT
                        b.service,
                        b.monthly_limit,
                        COALESCE(cs.month_to_date, 0) as spent,
                        b.monthly_limit - COALESCE(cs.month_to_date, 0) as remaining,
                        (COALESCE(cs.month_to_date, 0) / b.monthly_limit) * 100 as percent_used,
                        b.alert_threshold,
                        CASE
                            WHEN COALESCE(cs.month_to_date, 0) >= b.monthly_limit THEN 'exceeded'
                            WHEN COALESCE(cs.month_to_date, 0) >= b.monthly_limit * b.alert_threshold THEN 'warning'
                            ELSE 'ok'
                        END as status
                    FROM ai_budgets b
                    LEFT JOIN current_spending cs ON b.service = cs.service
                    WHERE b.service = %s AND b.status = 'active'
                """, (service, service))
            else:
                cursor.execute("""
                    WITH current_spending AS (
                        SELECT
                            metadata->>'service' as service,
                            SUM(cost) as month_to_date
                        FROM ai_resource_usage
                        WHERE DATE_TRUNC('month', timestamp) = DATE_TRUNC('month', NOW())
                          AND metadata->>'service' IS NOT NULL
                        GROUP BY metadata->>'service'
                    )
                    SELECT
                        b.service,
                        b.monthly_limit,
                        COALESCE(cs.month_to_date, 0) as spent,
                        b.monthly_limit - COALESCE(cs.month_to_date, 0) as remaining,
                        (COALESCE(cs.month_to_date, 0) / b.monthly_limit) * 100 as percent_used,
                        b.alert_threshold,
                        CASE
                            WHEN COALESCE(cs.month_to_date, 0) >= b.monthly_limit THEN 'exceeded'
                            WHEN COALESCE(cs.month_to_date, 0) >= b.monthly_limit * b.alert_threshold THEN 'warning'
                            ELSE 'ok'
                        END as status
                    FROM ai_budgets b
                    LEFT JOIN current_spending cs ON b.service = cs.service
                    WHERE b.status = 'active'
                    ORDER BY percent_used DESC
                """)

            budget_status = cursor.fetchall()

            cursor.close()
            conn.close()

            # Check for alerts
            results = []
            for budget in budget_status:
                budget_dict = dict(budget)

                # Send alert if needed
                if budget_dict['status'] in ['warning', 'exceeded']:
                    await self._send_budget_alert(budget_dict)

                results.append(budget_dict)

            return results

        except Exception as e:
            logger.error(f"Error checking budget: {e}")
            raise

    async def _send_budget_alert(self, budget: dict):
        """Send budget alert"""
        service = budget['service']
        alert_key = f"{service}_{datetime.now().strftime('%Y-%m')}"

        # Only send one alert per month per service
        if alert_key not in self.alerts_sent:
            logger.warning(
                f"Budget alert for {service}: "
                f"${budget['spent']:.2f} of ${budget['monthly_limit']:.2f} spent "
                f"({budget['percent_used']:.1f}%)"
            )
            self.alerts_sent[alert_key] = True

class CostRecommendationEngine:
    """Generate cost-saving recommendations"""

    def __init__(self):
        self.recommendations = []

    async def generate_recommendations(
        self,
        usage_data: dict,
        optimization_history: list[dict]
    ) -> list[dict]:
        """Generate cost-saving recommendations"""
        recommendations = []

        # Analyze usage patterns
        for usage in usage_data.get('usage_by_type', []):
            if float(usage.get('total_cost', 0)) > 5:
                rec = self._analyze_resource(usage)
                if rec:
                    recommendations.append(rec)

        # Check for duplicate operations
        duplicate_rec = await self._check_duplicates(usage_data)
        if duplicate_rec:
            recommendations.append(duplicate_rec)

        # Check for idle resources
        idle_rec = await self._check_idle_resources()
        if idle_rec:
            recommendations.append(idle_rec)

        # Sort by impact
        recommendations.sort(
            key=lambda x: x.get('impact_score', 0),
            reverse=True
        )

        return recommendations[:10]  # Top 10 recommendations

    def _analyze_resource(self, usage: dict) -> Optional[dict]:
        """Analyze resource for optimization"""
        resource_type = usage['resource_type']
        total_cost = float(usage.get('total_cost', 0))

        if resource_type == ResourceType.AI_TOKENS.value:
            if total_cost > 50:
                return {
                    "title": "Reduce AI Token Usage",
                    "description": "High AI token consumption detected",
                    "impact_score": 8,
                    "estimated_savings": total_cost * 0.3,
                    "actions": [
                        "Implement semantic caching",
                        "Use smaller models for simple tasks",
                        "Optimize prompt templates"
                    ],
                    "priority": "high"
                }

        return None

    async def _check_duplicates(self, usage_data: dict) -> Optional[dict]:
        """Check for duplicate operations"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            # Check for duplicate API calls
            cursor.execute("""
                SELECT
                    metadata->>'endpoint' as endpoint,
                    COUNT(*) as count
                FROM ai_resource_usage
                WHERE resource_type = 'api_calls'
                  AND timestamp > NOW() - INTERVAL '24 hours'
                  AND metadata->>'endpoint' IS NOT NULL
                GROUP BY metadata->>'endpoint'
                HAVING COUNT(*) > 100
                ORDER BY count DESC
                LIMIT 1
            """)

            duplicate = cursor.fetchone()

            cursor.close()
            conn.close()

            if duplicate and duplicate[1] > 100:
                return {
                    "title": "Eliminate Duplicate API Calls",
                    "description": f"Endpoint '{duplicate[0]}' called {duplicate[1]} times",
                    "impact_score": 7,
                    "estimated_savings": duplicate[1] * 0.0001,  # Estimate
                    "actions": [
                        "Implement request deduplication",
                        "Add response caching",
                        "Batch similar requests"
                    ],
                    "priority": "medium"
                }

        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")

        return None

    async def _check_idle_resources(self) -> Optional[dict]:
        """Check for idle resources"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            # Check for idle agents
            cursor.execute("""
                SELECT
                    COUNT(*) as idle_count
                FROM ai_agents
                WHERE status = 'active'
                  AND updated_at < NOW() - INTERVAL '7 days'
            """)

            idle_count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            if idle_count > 0:
                return {
                    "title": "Deactivate Idle Resources",
                    "description": f"{idle_count} agents haven't run in 7+ days",
                    "impact_score": 5,
                    "estimated_savings": idle_count * 5,  # $5 per idle agent
                    "actions": [
                        "Review and deactivate unused agents",
                        "Implement auto-shutdown for idle resources",
                        "Consolidate similar agents"
                    ],
                    "priority": "low"
                }

        except Exception as e:
            logger.error(f"Error checking idle resources: {e}")

        return None

class AICostOptimizationEngine:
    """Main cost optimization engine"""

    def __init__(self):
        self.monitor = ResourceMonitor()
        self.optimizer = CostOptimizer()
        self.budget_manager = BudgetManager()
        self.recommendation_engine = CostRecommendationEngine()

    async def optimize(self) -> dict:
        """Run full cost optimization cycle"""
        try:
            # Get current usage
            usage_summary = await self.monitor.get_usage_summary(days=30)

            # Analyze costs
            opportunities = await self.optimizer.analyze_costs(usage_summary)

            # Check budgets
            budget_status = await self.budget_manager.check_budget_status()

            # Generate recommendations
            recommendations = await self.recommendation_engine.generate_recommendations(
                usage_summary,
                []  # optimization history
            )

            # Calculate total potential savings
            total_current_cost = sum(
                float(u.get('total_cost', 0))
                for u in usage_summary.get('usage_by_type', [])
            )

            total_potential_savings = sum(
                o.get('potential_savings', 0)
                for o in opportunities
            )

            return {
                "current_monthly_cost": total_current_cost,
                "potential_monthly_savings": total_potential_savings,
                "savings_percentage": (total_potential_savings / max(total_current_cost, 1)) * 100,
                "optimization_opportunities": opportunities[:5],
                "recommendations": recommendations[:5],
                "budget_status": budget_status,
                "usage_summary": usage_summary,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
            raise

    async def track_resource(
        self,
        resource_type: ResourceType,
        amount: float,
        service: str,
        metadata: Optional[dict] = None
    ) -> dict:
        """Track resource usage"""
        full_metadata = metadata or {}
        full_metadata['service'] = service

        return await self.monitor.track_usage(
            resource_type,
            amount,
            full_metadata
        )

    async def set_budget(
        self,
        service: str,
        monthly_limit: float
    ) -> dict:
        """Set budget for service"""
        return await self.budget_manager.set_budget(
            service,
            monthly_limit
        )

    async def apply_optimization(
        self,
        optimization: dict
    ) -> dict:
        """Apply an optimization"""
        return await self.optimizer.apply_optimization(
            str(uuid.uuid4()),
            OptimizationStrategy[optimization['strategy'].upper()],
            optimization
        )

# Singleton instance
_engine_instance = None

def get_cost_optimization_engine():
    """Get or create cost optimization engine instance"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AICostOptimizationEngine()
    return _engine_instance

# Export main components
__all__ = [
    'AICostOptimizationEngine',
    'get_cost_optimization_engine',
    'ResourceType',
    'OptimizationStrategy',
    'CostLevel'
]

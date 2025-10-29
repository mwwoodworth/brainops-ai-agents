#!/usr/bin/env python3
"""
DevOps Optimization Agent - Infrastructure & Deployment Intelligence
Monitors deployments, optimizes infrastructure, auto-scales resources, prevents issues
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
    'password': os.getenv('DB_PASSWORD', 'REDACTED_SUPABASE_DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 6543))
}


class DeploymentStatus(Enum):
    """Deployment status types"""
    SUCCESS = "success"
    FAILED = "failed"
    DEGRADED = "degraded"
    ROLLING_BACK = "rolling_back"
    IN_PROGRESS = "in_progress"


class InfrastructureMetric(Enum):
    """Infrastructure metrics to monitor"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    UPTIME = "uptime"


@dataclass
class DeploymentEvent:
    """Deployment event tracking"""
    id: str
    service_name: str
    version: str
    status: DeploymentStatus
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    deployed_by: str
    commit_hash: str
    rollback_version: Optional[str]
    error_message: Optional[str]
    metrics: Dict[str, Any]


@dataclass
class InfrastructureAlert:
    """Infrastructure health alert"""
    id: str
    service_name: str
    metric_type: InfrastructureMetric
    current_value: float
    threshold_value: float
    severity: str  # critical, warning, info
    message: str
    detected_at: datetime
    auto_remediation: Optional[str]
    resolved: bool


@dataclass
class OptimizationRecommendation:
    """Infrastructure optimization suggestion"""
    id: str
    category: str  # scaling, cost, performance, reliability
    title: str
    description: str
    current_state: str
    recommended_state: str
    estimated_savings: Optional[float]
    estimated_performance_gain: Optional[str]
    implementation_steps: List[str]
    risk_level: str  # low, medium, high
    auto_implementable: bool
    created_at: datetime


class DevOpsOptimizationAgent:
    """Agent that optimizes infrastructure and deployments"""

    def __init__(self):
        self.db_config = DB_CONFIG
        self.conn = None
        self.deployment_history = []
        self.active_alerts = []
        self.recommendations = []
        self._init_database()
        logger.info("✅ DevOps Optimization Agent initialized")

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Create deployment events table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_deployment_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    service_name VARCHAR(255) NOT NULL,
                    version VARCHAR(100) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMP,
                    duration_seconds FLOAT,
                    deployed_by VARCHAR(255),
                    commit_hash VARCHAR(100),
                    rollback_version VARCHAR(100),
                    error_message TEXT,
                    metrics JSONB DEFAULT '{}'::jsonb,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create infrastructure alerts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_infrastructure_alerts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    service_name VARCHAR(255) NOT NULL,
                    metric_type VARCHAR(50) NOT NULL,
                    current_value FLOAT NOT NULL,
                    threshold_value FLOAT NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    message TEXT,
                    detected_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    auto_remediation TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create optimization recommendations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_devops_optimizations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    category VARCHAR(50) NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    current_state TEXT,
                    recommended_state TEXT,
                    estimated_savings FLOAT,
                    estimated_performance_gain TEXT,
                    implementation_steps JSONB DEFAULT '[]'::jsonb,
                    risk_level VARCHAR(20),
                    auto_implementable BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    implemented_at TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'proposed',
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create infrastructure metrics tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_infrastructure_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    service_name VARCHAR(255) NOT NULL,
                    metric_type VARCHAR(50) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create indexes for performance
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_deployment_events_service ON ai_deployment_events(service_name, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_infrastructure_alerts_unresolved ON ai_infrastructure_alerts(service_name, resolved) WHERE resolved = FALSE;
                CREATE INDEX IF NOT EXISTS idx_devops_optimizations_status ON ai_devops_optimizations(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_infrastructure_metrics_service_time ON ai_infrastructure_metrics(service_name, metric_type, timestamp DESC);
            """)

            conn.commit()
            logger.info("✅ DevOps Optimization Agent database tables ready")

        except Exception as e:
            logger.warning(f"Database initialization skipped: {e}. Operating without persistence")
        finally:
            if conn:
                conn.close()

    def monitor_deployment_health(self) -> List[DeploymentEvent]:
        """Monitor recent deployments for issues"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check recent deployments (last 24 hours)
            cur.execute("""
                SELECT * FROM ai_deployment_events
                WHERE started_at > NOW() - INTERVAL '24 hours'
                ORDER BY started_at DESC
                LIMIT 50
            """)

            recent_deployments = cur.fetchall()
            conn.close()

            # Analyze deployment patterns
            if recent_deployments:
                total = len(recent_deployments)
                failed = sum(1 for d in recent_deployments if d['status'] == 'failed')
                failure_rate = (failed / total) * 100

                if failure_rate > 20:
                    logger.warning(f"⚠️ High deployment failure rate: {failure_rate:.1f}% ({failed}/{total})")

            return recent_deployments

        except Exception as e:
            logger.warning(f"Deployment monitoring failed: {e}")
            return []

    def detect_infrastructure_issues(self) -> List[InfrastructureAlert]:
        """Detect infrastructure performance issues"""
        alerts = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for slow response times (agent executions)
            cur.execute("""
                SELECT
                    AVG(duration_ms) as avg_response_time,
                    MAX(duration_ms) as max_response_time,
                    COUNT(*) as total_requests
                FROM agent_executions
                WHERE completed_at > NOW() - INTERVAL '1 hour'
                AND status = 'completed'
            """)

            response_data = cur.fetchone()

            if response_data and response_data['avg_response_time']:
                avg_ms = float(response_data['avg_response_time'])

                # Alert if average response time > 3 seconds
                if avg_ms > 3000:
                    alerts.append({
                        'service_name': 'Agent Execution Engine',
                        'metric_type': 'response_time',
                        'current_value': avg_ms,
                        'threshold_value': 3000.0,
                        'severity': 'warning' if avg_ms < 5000 else 'critical',
                        'message': f'Average response time {avg_ms/1000:.2f}s exceeds 3s target',
                        'auto_remediation': 'Consider implementing caching or optimizing database queries'
                    })

            # Check for high error rates
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_count,
                    COUNT(*) as total_count,
                    COUNT(*) FILTER (WHERE status = 'failed') * 100.0 / NULLIF(COUNT(*), 0) as error_rate
                FROM agent_executions
                WHERE completed_at > NOW() - INTERVAL '1 hour'
            """)

            error_data = cur.fetchone()

            if error_data and error_data['error_rate'] and error_data['error_rate'] > 10:
                alerts.append({
                    'service_name': 'Agent Execution Engine',
                    'metric_type': 'error_rate',
                    'current_value': float(error_data['error_rate']),
                    'threshold_value': 10.0,
                    'severity': 'critical',
                    'message': f'Error rate {error_data["error_rate"]:.1f}% exceeds 10% threshold',
                    'auto_remediation': 'Investigate error logs and implement circuit breakers'
                })

            conn.close()

            # Persist alerts to database
            if alerts:
                self._persist_alerts(alerts)

        except Exception as e:
            logger.warning(f"Infrastructure issue detection failed: {e}")

        return alerts

    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate infrastructure optimization recommendations"""
        recommendations = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Recommendation 1: Identify unused database indexes
            cur.execute("""
                SELECT schemaname, tablename, indexname
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0
                AND schemaname = 'public'
                LIMIT 10
            """)

            unused_indexes = cur.fetchall()
            if unused_indexes:
                recommendations.append({
                    'category': 'cost',
                    'title': 'Remove Unused Database Indexes',
                    'description': f'Found {len(unused_indexes)} indexes that are never used',
                    'current_state': f'{len(unused_indexes)} unused indexes consuming disk space',
                    'recommended_state': 'Remove unused indexes to save storage and improve write performance',
                    'estimated_savings': len(unused_indexes) * 0.5,  # Rough estimate
                    'estimated_performance_gain': '5-10% faster writes',
                    'implementation_steps': [f"DROP INDEX IF EXISTS {idx['indexname']}" for idx in unused_indexes[:5]],
                    'risk_level': 'low',
                    'auto_implementable': False
                })

            # Recommendation 2: Database connection pooling
            cur.execute("""
                SELECT COUNT(*) as connection_count
                FROM pg_stat_activity
                WHERE state = 'active'
            """)

            connection_data = cur.fetchone()
            if connection_data and connection_data['connection_count'] > 20:
                recommendations.append({
                    'category': 'performance',
                    'title': 'Optimize Database Connection Pooling',
                    'description': f'{connection_data["connection_count"]} active connections detected',
                    'current_state': f'{connection_data["connection_count"]} active database connections',
                    'recommended_state': 'Implement connection pooling with max 10 connections',
                    'estimated_savings': 15.0,
                    'estimated_performance_gain': '30% reduction in connection overhead',
                    'implementation_steps': [
                        'Install pgbouncer or use Supabase pooler',
                        'Configure max_connections = 10',
                        'Set pool_mode = transaction',
                        'Update connection strings to use pooler'
                    ],
                    'risk_level': 'medium',
                    'auto_implementable': False
                })

            # Recommendation 3: Cache frequently accessed data
            cur.execute("""
                SELECT table_name, seq_scan, idx_scan
                FROM information_schema.tables t
                JOIN pg_stat_user_tables s ON t.table_name = s.relname
                WHERE t.table_schema = 'public'
                AND s.seq_scan > 1000
                ORDER BY s.seq_scan DESC
                LIMIT 5
            """)

            high_scan_tables = cur.fetchall()
            if high_scan_tables:
                recommendations.append({
                    'category': 'performance',
                    'title': 'Implement Caching for Frequently Accessed Tables',
                    'description': f'{len(high_scan_tables)} tables with >1000 sequential scans',
                    'current_state': 'No caching layer, all queries hit database',
                    'recommended_state': 'Redis cache for hot data with 5-minute TTL',
                    'estimated_savings': 25.0,
                    'estimated_performance_gain': '70% reduction in database load',
                    'implementation_steps': [
                        'Set up Redis instance',
                        'Implement cache-aside pattern',
                        f'Cache tables: {", ".join([t["table_name"] for t in high_scan_tables[:3]])}',
                        'Set TTL = 300s for cached data'
                    ],
                    'risk_level': 'medium',
                    'auto_implementable': False
                })

            conn.close()

            # Persist recommendations
            if recommendations:
                self._persist_recommendations(recommendations)

        except Exception as e:
            logger.warning(f"Optimization recommendation generation failed: {e}")

        return recommendations

    def _persist_alerts(self, alerts: List[Dict]):
        """Persist alerts to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            for alert in alerts:
                cur.execute("""
                    INSERT INTO ai_infrastructure_alerts
                    (service_name, metric_type, current_value, threshold_value, severity, message, auto_remediation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    alert['service_name'],
                    alert['metric_type'],
                    alert['current_value'],
                    alert['threshold_value'],
                    alert['severity'],
                    alert['message'],
                    alert.get('auto_remediation')
                ))

            conn.commit()
            conn.close()
            logger.info(f"✅ Persisted {len(alerts)} infrastructure alerts")

        except Exception as e:
            logger.warning(f"Failed to persist alerts: {e}")

    def _persist_recommendations(self, recommendations: List[Dict]):
        """Persist recommendations to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            for rec in recommendations:
                cur.execute("""
                    INSERT INTO ai_devops_optimizations
                    (category, title, description, current_state, recommended_state,
                     estimated_savings, estimated_performance_gain, implementation_steps,
                     risk_level, auto_implementable)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    rec['category'],
                    rec['title'],
                    rec['description'],
                    rec['current_state'],
                    rec['recommended_state'],
                    rec.get('estimated_savings'),
                    rec.get('estimated_performance_gain'),
                    Json(rec['implementation_steps']),
                    rec['risk_level'],
                    rec['auto_implementable']
                ))

            conn.commit()
            conn.close()
            logger.info(f"✅ Persisted {len(recommendations)} DevOps recommendations")

        except Exception as e:
            logger.warning(f"Failed to persist recommendations: {e}")

    async def continuous_optimization_loop(self, interval_hours: int = 2):
        """Main loop that continuously monitors and optimizes infrastructure"""
        logger.info(f"🔄 Starting DevOps optimization loop (every {interval_hours}h)")

        while True:
            try:
                logger.info("🔍 Analyzing infrastructure and deployments...")

                # Monitor deployments
                deployments = self.monitor_deployment_health()
                logger.info(f"📊 Analyzed {len(deployments)} recent deployments")

                # Detect infrastructure issues
                alerts = self.detect_infrastructure_issues()
                if alerts:
                    logger.warning(f"⚠️ Detected {len(alerts)} infrastructure alerts")
                    for alert in alerts:
                        logger.warning(f"   - {alert['severity'].upper()}: {alert['message']}")

                # Generate optimization recommendations
                recommendations = self.generate_optimization_recommendations()
                if recommendations:
                    logger.info(f"💡 Generated {len(recommendations)} optimization recommendations")
                    for rec in recommendations:
                        logger.info(f"   - {rec['category'].upper()}: {rec['title']}")

            except Exception as e:
                logger.error(f"❌ DevOps optimization loop error: {e}")

            # Wait before next analysis
            await asyncio.sleep(interval_hours * 3600)


if __name__ == "__main__":
    agent = DevOpsOptimizationAgent()
    asyncio.run(agent.continuous_optimization_loop(interval_hours=2))

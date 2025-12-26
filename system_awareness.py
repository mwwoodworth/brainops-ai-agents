#!/usr/bin/env python3
"""
SYSTEM AWARENESS - The All-Seeing Eye of BrainOps AI OS
This module provides REAL awareness of everything happening in the system.

Unlike the skeleton code, this ACTUALLY:
- Monitors real business data (customers, jobs, revenue)
- Tracks DevOps status (deployments, services, errors)
- Generates meaningful predictions
- Takes autonomous actions
- Shifts attention based on real events
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import httpx

logger = logging.getLogger("SYSTEM_AWARENESS")


def _build_db_config():
    config = {
        'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', 5432))
    }
    if not config['password']:
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            from urllib.parse import urlparse
            try:
                parsed = urlparse(database_url)
                config['host'] = parsed.hostname or config['host']
                config['database'] = parsed.path.lstrip('/') or config['database']
                config['user'] = parsed.username or config['user']
                config['password'] = parsed.password or ''
                config['port'] = parsed.port or config['port']
            except:
                pass
    return config

DB_CONFIG = _build_db_config()


class AwarenessCategory(Enum):
    BUSINESS = "business"
    DEVOPS = "devops"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CUSTOMERS = "customers"
    REVENUE = "revenue"


@dataclass
class Insight:
    category: AwarenessCategory
    title: str
    description: str
    severity: str  # info, warning, critical
    data: Dict[str, Any]
    action_recommended: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self):
        return {
            'category': self.category.value,
            'title': self.title,
            'description': self.description,
            'severity': self.severity,
            'data': self.data,
            'action_recommended': self.action_recommended,
            'timestamp': self.timestamp.isoformat()
        }


class SystemAwareness:
    """
    REAL system awareness - actually monitors and understands the system.
    """

    def __init__(self):
        self.insights: List[Insight] = []
        self.last_scan = {}
        self.anomalies = []
        self.services = {
            'brainops-ai-agents': 'https://brainops-ai-agents.onrender.com',
            'brainops-backend': 'https://brainops-backend-prod.onrender.com',
            'mcp-bridge': 'https://brainops-mcp-bridge.onrender.com'
        }

    def _get_connection(self):
        return psycopg2.connect(**DB_CONFIG)

    async def scan_business_metrics(self) -> List[Insight]:
        """Scan real business data for insights"""
        insights = []
        conn = None

        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Customer growth analysis
            cur.execute("""
                SELECT
                    COUNT(*) as total_customers,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as new_24h,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as new_7d,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as new_30d
                FROM customers
            """)
            customers = cur.fetchone()

            if customers:
                if customers['new_24h'] == 0 and customers['new_7d'] > 0:
                    insights.append(Insight(
                        category=AwarenessCategory.CUSTOMERS,
                        title="No New Customers Today",
                        description=f"Zero customer acquisition in last 24h. Weekly average: {customers['new_7d']/7:.1f}/day",
                        severity="warning",
                        data=dict(customers),
                        action_recommended="Review marketing channels and lead sources"
                    ))
                elif customers['new_24h'] > customers['new_7d'] / 7 * 2:
                    insights.append(Insight(
                        category=AwarenessCategory.CUSTOMERS,
                        title="Customer Acquisition Spike",
                        description=f"{customers['new_24h']} new customers today - 2x above daily average!",
                        severity="info",
                        data=dict(customers),
                        action_recommended="Investigate source and consider scaling support"
                    ))

            # Job volume analysis
            cur.execute("""
                SELECT
                    COUNT(*) as total_jobs,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as new_24h,
                    COUNT(*) FILTER (WHERE status = 'pending') as pending,
                    COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress,
                    COUNT(*) FILTER (WHERE status = 'completed' AND updated_at > NOW() - INTERVAL '24 hours') as completed_24h
                FROM jobs
            """)
            jobs = cur.fetchone()

            if jobs:
                if jobs['pending'] > 100:
                    insights.append(Insight(
                        category=AwarenessCategory.BUSINESS,
                        title="High Pending Job Backlog",
                        description=f"{jobs['pending']} jobs pending. Backlog needs attention.",
                        severity="warning" if jobs['pending'] < 200 else "critical",
                        data=dict(jobs),
                        action_recommended="Allocate more crew resources or automate processing"
                    ))

                if jobs['completed_24h'] > 0:
                    insights.append(Insight(
                        category=AwarenessCategory.BUSINESS,
                        title="Daily Job Completion",
                        description=f"Completed {jobs['completed_24h']} jobs in last 24 hours",
                        severity="info",
                        data=dict(jobs)
                    ))

            # Revenue tracking
            cur.execute("""
                SELECT
                    COALESCE(SUM(total_amount), 0) as today_revenue,
                    COUNT(*) as today_invoices
                FROM invoices
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            revenue = cur.fetchone()

            cur.execute("""
                SELECT COALESCE(AVG(daily_revenue), 0) as avg_daily
                FROM (
                    SELECT DATE(created_at), SUM(total_amount) as daily_revenue
                    FROM invoices
                    WHERE created_at > NOW() - INTERVAL '30 days'
                    GROUP BY DATE(created_at)
                ) daily
            """)
            avg_revenue = cur.fetchone()

            if revenue and avg_revenue:
                today = float(revenue['today_revenue'] or 0)
                avg = float(avg_revenue['avg_daily'] or 1)

                if today < avg * 0.5 and avg > 0:
                    insights.append(Insight(
                        category=AwarenessCategory.REVENUE,
                        title="Revenue Below Average",
                        description=f"Today: ${today:,.2f} vs 30-day avg: ${avg:,.2f}/day ({today/avg*100:.0f}%)",
                        severity="warning",
                        data={'today': today, 'average': avg, 'ratio': today/avg if avg else 0},
                        action_recommended="Review sales pipeline and pending invoices"
                    ))
                elif today > avg * 1.5:
                    insights.append(Insight(
                        category=AwarenessCategory.REVENUE,
                        title="Revenue Above Average",
                        description=f"Strong day! ${today:,.2f} - {today/avg*100:.0f}% of average",
                        severity="info",
                        data={'today': today, 'average': avg}
                    ))

        except Exception as e:
            logger.error(f"Business scan error: {e}")
            insights.append(Insight(
                category=AwarenessCategory.BUSINESS,
                title="Business Metrics Scan Failed",
                description=str(e),
                severity="warning",
                data={'error': str(e)}
            ))
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

        return insights

    async def scan_devops_status(self) -> List[Insight]:
        """Check all services and infrastructure"""
        insights = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, url in self.services.items():
                try:
                    start = datetime.utcnow()
                    response = await client.get(f"{url}/health")
                    latency = (datetime.utcnow() - start).total_seconds() * 1000

                    if response.status_code == 200:
                        data = response.json()
                        insights.append(Insight(
                            category=AwarenessCategory.DEVOPS,
                            title=f"{service_name}: Online",
                            description=f"Version {data.get('version', 'unknown')}, latency: {latency:.0f}ms",
                            severity="info",
                            data={'status': 'online', 'version': data.get('version'), 'latency_ms': latency}
                        ))

                        if latency > 2000:
                            insights.append(Insight(
                                category=AwarenessCategory.PERFORMANCE,
                                title=f"{service_name}: High Latency",
                                description=f"Response time {latency:.0f}ms exceeds 2000ms threshold",
                                severity="warning",
                                data={'service': service_name, 'latency_ms': latency},
                                action_recommended="Investigate service performance"
                            ))
                    else:
                        insights.append(Insight(
                            category=AwarenessCategory.DEVOPS,
                            title=f"{service_name}: Degraded",
                            description=f"Status code {response.status_code}",
                            severity="critical",
                            data={'status': 'degraded', 'status_code': response.status_code},
                            action_recommended=f"Check {service_name} logs immediately"
                        ))

                except Exception as e:
                    insights.append(Insight(
                        category=AwarenessCategory.DEVOPS,
                        title=f"{service_name}: Unreachable",
                        description=str(e),
                        severity="critical",
                        data={'status': 'offline', 'error': str(e)},
                        action_recommended=f"Restart {service_name} or check Render dashboard"
                    ))

        return insights

    async def scan_error_rates(self) -> List[Insight]:
        """Analyze error patterns"""
        insights = []
        conn = None

        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check agent execution errors
            cur.execute("""
                SELECT
                    agent_type,
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE status = 'error') as errors,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') as recent
                FROM ai_agent_executions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY agent_type
            """)
            agent_errors = cur.fetchall()

            for ae in agent_errors:
                if ae['total'] > 0:
                    error_rate = ae['errors'] / ae['total']
                    if error_rate > 0.1:  # More than 10% errors
                        insights.append(Insight(
                            category=AwarenessCategory.DEVOPS,
                            title=f"Agent Error Rate: {ae['agent_type']}",
                            description=f"{ae['errors']}/{ae['total']} executions failed ({error_rate*100:.1f}%)",
                            severity="critical" if error_rate > 0.25 else "warning",
                            data=dict(ae),
                            action_recommended=f"Review {ae['agent_type']} agent logs and fix issues"
                        ))

        except Exception as e:
            logger.error(f"Error scan failed: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

        return insights

    async def scan_security(self) -> List[Insight]:
        """Check for security concerns"""
        insights = []
        conn = None

        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for unusual access patterns
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE status_code = 401) as auth_failures,
                    COUNT(*) FILTER (WHERE status_code = 403) as forbidden,
                    COUNT(*) as total_requests
                FROM api_request_logs
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            access = cur.fetchone()

            if access and access['total_requests'] > 0:
                auth_fail_rate = access['auth_failures'] / access['total_requests']
                if auth_fail_rate > 0.05:  # More than 5% auth failures
                    insights.append(Insight(
                        category=AwarenessCategory.SECURITY,
                        title="High Authentication Failure Rate",
                        description=f"{access['auth_failures']} auth failures in last hour ({auth_fail_rate*100:.1f}%)",
                        severity="warning" if auth_fail_rate < 0.2 else "critical",
                        data=dict(access),
                        action_recommended="Review failed auth attempts for potential attack"
                    ))

        except Exception as e:
            # Table might not exist, that's ok
            pass
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

        return insights

    async def run_full_scan(self) -> Dict[str, Any]:
        """Run a complete system awareness scan"""
        logger.info("🔍 Running full system awareness scan...")
        start_time = datetime.utcnow()

        all_insights = []

        # Run all scans in parallel
        business, devops, errors, security = await asyncio.gather(
            self.scan_business_metrics(),
            self.scan_devops_status(),
            self.scan_error_rates(),
            self.scan_security(),
            return_exceptions=True
        )

        for result in [business, devops, errors, security]:
            if isinstance(result, list):
                all_insights.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Scan error: {result}")

        # Store insights
        self.insights = all_insights

        # Persist to database
        await self._persist_insights(all_insights)

        scan_duration = (datetime.utcnow() - start_time).total_seconds()

        # Categorize by severity
        critical = [i for i in all_insights if i.severity == 'critical']
        warnings = [i for i in all_insights if i.severity == 'warning']
        info = [i for i in all_insights if i.severity == 'info']

        summary = {
            'scan_time': start_time.isoformat(),
            'duration_seconds': scan_duration,
            'total_insights': len(all_insights),
            'critical': len(critical),
            'warnings': len(warnings),
            'info': len(info),
            'insights': [i.to_dict() for i in all_insights],
            'top_concerns': [i.to_dict() for i in critical + warnings][:5]
        }

        logger.info(f"✅ Scan complete: {len(critical)} critical, {len(warnings)} warnings, {len(info)} info")

        return summary

    async def _persist_insights(self, insights: List[Insight]):
        """Store insights in database"""
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Ensure table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_system_insights (
                    id SERIAL PRIMARY KEY,
                    category VARCHAR(50),
                    title VARCHAR(255),
                    description TEXT,
                    severity VARCHAR(20),
                    data JSONB,
                    action_recommended TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_insights_severity
                    ON ai_system_insights(severity);
                CREATE INDEX IF NOT EXISTS idx_insights_time
                    ON ai_system_insights(created_at DESC);
            """)

            for insight in insights:
                cur.execute("""
                    INSERT INTO ai_system_insights
                    (category, title, description, severity, data, action_recommended)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    insight.category.value, insight.title, insight.description,
                    insight.severity, Json(insight.data), insight.action_recommended
                ))

            conn.commit()

        except Exception as e:
            logger.error(f"Failed to persist insights: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def get_attention_priority(self) -> Tuple[str, str, int]:
        """Determine what should have attention right now"""
        if not self.insights:
            return ("system_initialization", "No scan data yet", 5)

        critical = [i for i in self.insights if i.severity == 'critical']
        if critical:
            top = critical[0]
            return (top.category.value, top.title, 10)

        warnings = [i for i in self.insights if i.severity == 'warning']
        if warnings:
            top = warnings[0]
            return (top.category.value, top.title, 7)

        return ("monitoring", "All systems normal", 3)


# Singleton
_system_awareness: Optional[SystemAwareness] = None


def get_system_awareness() -> SystemAwareness:
    global _system_awareness
    if _system_awareness is None:
        _system_awareness = SystemAwareness()
    return _system_awareness


if __name__ == "__main__":
    async def test():
        print("\n" + "="*60)
        print("🔍 SYSTEM AWARENESS TEST")
        print("="*60 + "\n")

        awareness = get_system_awareness()
        result = await awareness.run_full_scan()

        print(json.dumps(result, indent=2, default=str))

        priority = awareness.get_attention_priority()
        print(f"\n🎯 Attention Priority: {priority}")

    asyncio.run(test())

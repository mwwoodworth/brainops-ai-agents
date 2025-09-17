#!/usr/bin/env python3
"""
Automated Reporting System - Task 20
Real-time business intelligence and automated report generation
"""

import os
import json
import logging
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JSON encoder for Decimal types
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
    "port": os.getenv("DB_PORT", "5432")
}

class ReportType(Enum):
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_PERFORMANCE = "weekly_performance"
    MONTHLY_FINANCIAL = "monthly_financial"
    CUSTOMER_ANALYTICS = "customer_analytics"
    AGENT_PERFORMANCE = "agent_performance"
    LEAD_PIPELINE = "lead_pipeline"
    REVENUE_FORECAST = "revenue_forecast"
    OPERATIONAL_HEALTH = "operational_health"
    AI_SYSTEM_STATUS = "ai_system_status"
    CUSTOM = "custom"

class ReportFormat(Enum):
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    DASHBOARD = "dashboard"
    EMAIL = "email"

class ReportSchedule(Enum):
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_DEMAND = "on_demand"

class MetricType(Enum):
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    PERCENTAGE = "percentage"
    TREND = "trend"
    COMPARISON = "comparison"

class AutomatedReportingSystem:
    """Main automated reporting system with real-time analytics"""

    def __init__(self):
        self.conn = None
        self.report_generator = ReportGenerator()
        self.metric_calculator = MetricCalculator()
        self.visualization_engine = VisualizationEngine()
        self.distribution_manager = DistributionManager()
        self.alert_system = AlertSystem()
        self.cache = {}

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**DB_CONFIG)
        return self.conn

    async def create_report(
        self,
        report_type: ReportType,
        parameters: Dict[str, Any],
        format: ReportFormat = ReportFormat.JSON,
        schedule: Optional[ReportSchedule] = None
    ) -> str:
        """Create a new report or schedule recurring reports"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            report_id = str(uuid.uuid4())

            # Generate report data
            report_data = await self._generate_report_data(
                report_type, parameters
            )

            # Calculate metrics
            metrics = await self.metric_calculator.calculate_metrics(
                report_type, report_data
            )

            # Generate insights
            insights = await self._generate_insights(
                report_type, metrics, report_data
            )

            # Format report
            formatted_report = await self.report_generator.format_report(
                report_data, metrics, insights, format
            )

            # Generate title and text summary
            title = f"{report_type.value.replace('_', ' ').title()} - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

            # Create text summary
            if report_type == ReportType.DAILY_SUMMARY:
                summary_data = report_data.get('summary', {})
                text_summary = (
                    f"Daily report showing {summary_data.get('new_customers', 0)} new customers, "
                    f"{summary_data.get('new_jobs', 0)} new jobs, "
                    f"${summary_data.get('revenue', 0) or 0:,.2f} in revenue, "
                    f"and {summary_data.get('ai_executions', 0)} AI executions."
                )
            else:
                text_summary = f"{report_type.value.replace('_', ' ')} report generated successfully"

            # Store report
            cursor.execute("""
                INSERT INTO ai_generated_reports (
                    id, report_type, report_date, title, summary,
                    full_report, metrics, insights, recommendations,
                    ai_analysis, status, generated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                report_id, report_type.value, datetime.now(timezone.utc).date(),
                title,
                text_summary,
                json.dumps(report_data, cls=DecimalEncoder),
                json.dumps(metrics, cls=DecimalEncoder),
                json.dumps(insights, cls=DecimalEncoder),
                json.dumps([]),  # Recommendations - can be enhanced later
                formatted_report,  # AI analysis text
                'generated'
            ))

            # Schedule if needed
            if schedule and schedule != ReportSchedule.ON_DEMAND:
                await self._schedule_report(
                    report_id, report_type, parameters, schedule, cursor
                )

            # Check for alerts
            alerts = await self.alert_system.check_alerts(
                report_type, metrics
            )
            if alerts:
                await self._handle_alerts(alerts, report_id, cursor)

            conn.commit()

            logger.info(f"Created report {report_id} of type {report_type.value}")

            # Return full report details
            return {
                "report_id": report_id,
                "title": title,
                "summary": text_summary,
                "report_data": report_data,
                "metrics": metrics,
                "insights": insights,
                "formatted_report": formatted_report
            }

        except Exception as e:
            logger.error(f"Error creating report: {e}")
            if conn:
                conn.rollback()
            raise

    async def _generate_report_data(
        self,
        report_type: ReportType,
        parameters: Dict
    ) -> Dict:
        """Generate data for specific report type"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        if report_type == ReportType.DAILY_SUMMARY:
            return await self._get_daily_summary(cursor, parameters)
        elif report_type == ReportType.WEEKLY_PERFORMANCE:
            return await self._get_weekly_performance(cursor, parameters)
        elif report_type == ReportType.MONTHLY_FINANCIAL:
            return await self._get_monthly_financial(cursor, parameters)
        elif report_type == ReportType.CUSTOMER_ANALYTICS:
            return await self._get_customer_analytics(cursor, parameters)
        elif report_type == ReportType.AGENT_PERFORMANCE:
            return await self._get_agent_performance(cursor, parameters)
        elif report_type == ReportType.LEAD_PIPELINE:
            return await self._get_lead_pipeline(cursor, parameters)
        elif report_type == ReportType.AI_SYSTEM_STATUS:
            return await self._get_ai_system_status(cursor, parameters)
        else:
            return {}

    async def _get_daily_summary(
        self,
        cursor: Any,
        parameters: Dict
    ) -> Dict:
        """Get daily summary data"""
        date = parameters.get('date', datetime.now(timezone.utc).date())

        # Get key metrics
        cursor.execute("""
            SELECT
                (SELECT COUNT(*) FROM customers WHERE DATE(created_at) = %s) as new_customers,
                (SELECT COUNT(*) FROM jobs WHERE DATE(created_at) = %s) as new_jobs,
                (SELECT COUNT(*) FROM invoices WHERE DATE(created_at) = %s) as new_invoices,
                (SELECT SUM(amount) FROM invoices WHERE DATE(created_at) = %s) as revenue,
                (SELECT COUNT(*) FROM leads WHERE DATE(created_at) = %s) as new_leads,
                (SELECT COUNT(*) FROM agent_executions WHERE DATE(created_at) = %s) as ai_executions
        """, (date, date, date, date, date, date))

        summary = cursor.fetchone()

        # Get agent activity
        cursor.execute("""
            SELECT
                agent_type,
                COUNT(*) as executions,
                AVG(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100 as success_rate
            FROM agent_executions
            WHERE DATE(created_at) = %s
            GROUP BY agent_type
            ORDER BY executions DESC
            LIMIT 10
        """, (date,))

        agent_activity = cursor.fetchall()

        return {
            "date": str(date),
            "summary": dict(summary) if summary else {},
            "agent_activity": [dict(row) for row in agent_activity],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _get_weekly_performance(
        self,
        cursor: Any,
        parameters: Dict
    ) -> Dict:
        """Get weekly performance metrics"""
        end_date = parameters.get('end_date', datetime.now(timezone.utc).date())
        start_date = end_date - timedelta(days=7)

        # Get performance trends
        cursor.execute("""
            WITH daily_metrics AS (
                SELECT
                    DATE(created_at) as date,
                    COUNT(DISTINCT customer_id) as customers,
                    COUNT(*) as jobs,
                    SUM(amount) as revenue
                FROM jobs
                LEFT JOIN invoices USING (job_id)
                WHERE DATE(created_at) BETWEEN %s AND %s
                GROUP BY DATE(created_at)
            )
            SELECT
                date,
                customers,
                jobs,
                COALESCE(revenue, 0) as revenue,
                SUM(jobs) OVER (ORDER BY date) as cumulative_jobs
            FROM daily_metrics
            ORDER BY date
        """, (start_date, end_date))

        daily_trends = cursor.fetchall()

        # Get top performers
        cursor.execute("""
            SELECT
                a.name as agent_name,
                COUNT(ae.id) as executions,
                AVG(CASE WHEN ae.status = 'completed' THEN 1 ELSE 0 END) * 100 as success_rate
            FROM ai_agents a
            JOIN agent_executions ae ON a.id = ae.agent_id
            WHERE ae.created_at BETWEEN %s AND %s
            GROUP BY a.name
            ORDER BY executions DESC
            LIMIT 10
        """, (start_date, end_date))

        top_agents = cursor.fetchall()

        return {
            "period": {
                "start": str(start_date),
                "end": str(end_date)
            },
            "daily_trends": daily_trends,
            "top_agents": top_agents
        }

    async def _get_monthly_financial(
        self,
        cursor: Any,
        parameters: Dict
    ) -> Dict:
        """Get monthly financial report"""
        month = parameters.get('month', datetime.now(timezone.utc).month)
        year = parameters.get('year', datetime.now(timezone.utc).year)

        # Get financial summary
        cursor.execute("""
            SELECT
                COUNT(DISTINCT customer_id) as active_customers,
                COUNT(*) as total_invoices,
                SUM(amount) as total_revenue,
                AVG(amount) as avg_invoice,
                MAX(amount) as largest_invoice
            FROM invoices
            WHERE EXTRACT(MONTH FROM created_at) = %s
            AND EXTRACT(YEAR FROM created_at) = %s
        """, (month, year))

        financial_summary = cursor.fetchone()

        # Get revenue by customer segment
        cursor.execute("""
            SELECT
                COALESCE(c.segment, 'Unknown') as segment,
                COUNT(DISTINCT i.customer_id) as customers,
                SUM(i.amount) as revenue,
                AVG(i.amount) as avg_transaction
            FROM invoices i
            LEFT JOIN customers c ON i.customer_id = c.id
            WHERE EXTRACT(MONTH FROM i.created_at) = %s
            AND EXTRACT(YEAR FROM i.created_at) = %s
            GROUP BY c.segment
            ORDER BY revenue DESC
        """, (month, year))

        revenue_by_segment = cursor.fetchall()

        return {
            "period": f"{year}-{month:02d}",
            "financial_summary": dict(financial_summary) if financial_summary else {},
            "revenue_by_segment": [dict(row) for row in revenue_by_segment]
        }

    async def _get_customer_analytics(
        self,
        cursor: Any,
        parameters: Dict
    ) -> Dict:
        """Get customer analytics"""
        # Customer acquisition and retention
        cursor.execute("""
            WITH customer_metrics AS (
                SELECT
                    DATE_TRUNC('month', created_at) as month,
                    COUNT(*) as new_customers
                FROM customers
                WHERE created_at >= NOW() - INTERVAL '12 months'
                GROUP BY month
            ),
            customer_value AS (
                SELECT
                    customer_id,
                    SUM(amount) as total_value,
                    COUNT(*) as transaction_count,
                    MAX(created_at) as last_transaction
                FROM invoices
                GROUP BY customer_id
            )
            SELECT
                cm.month,
                cm.new_customers,
                AVG(cv.total_value) as avg_customer_value,
                COUNT(CASE WHEN cv.last_transaction > NOW() - INTERVAL '30 days'
                      THEN 1 END) as active_customers
            FROM customer_metrics cm
            LEFT JOIN customers c ON DATE_TRUNC('month', c.created_at) = cm.month
            LEFT JOIN customer_value cv ON c.id = cv.customer_id
            GROUP BY cm.month, cm.new_customers
            ORDER BY cm.month DESC
            LIMIT 12
        """)

        customer_trends = cursor.fetchall()

        # Top customers
        cursor.execute("""
            SELECT
                c.id,
                c.name,
                c.email,
                COUNT(j.id) as job_count,
                SUM(i.amount) as total_revenue,
                MAX(j.created_at) as last_job
            FROM customers c
            LEFT JOIN jobs j ON c.id = j.customer_id
            LEFT JOIN invoices i ON j.id = i.job_id
            GROUP BY c.id, c.name, c.email
            ORDER BY total_revenue DESC NULLS LAST
            LIMIT 20
        """)

        top_customers = cursor.fetchall()

        return {
            "customer_trends": [dict(row) for row in customer_trends],
            "top_customers": [dict(row) for row in top_customers]
        }

    async def _get_agent_performance(
        self,
        cursor: Any,
        parameters: Dict
    ) -> Dict:
        """Get AI agent performance metrics"""
        lookback_days = parameters.get('days', 7)

        # Overall agent performance
        cursor.execute("""
            SELECT
                agent_type,
                COUNT(*) as total_executions,
                AVG(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100 as success_rate,
                AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_duration_seconds,
                COUNT(DISTINCT DATE(created_at)) as active_days
            FROM agent_executions
            WHERE created_at >= NOW() - INTERVAL '%s days'
            GROUP BY agent_type
            ORDER BY total_executions DESC
        """, (lookback_days,))

        agent_performance = cursor.fetchall()

        # Error analysis
        cursor.execute("""
            SELECT
                agent_type,
                error_message,
                COUNT(*) as error_count
            FROM agent_executions
            WHERE status = 'failed'
            AND created_at >= NOW() - INTERVAL '%s days'
            AND error_message IS NOT NULL
            GROUP BY agent_type, error_message
            ORDER BY error_count DESC
            LIMIT 20
        """, (lookback_days,))

        error_analysis = cursor.fetchall()

        # AI system utilization
        cursor.execute("""
            SELECT
                DATE_TRUNC('hour', created_at) as hour,
                COUNT(*) as executions,
                COUNT(DISTINCT agent_id) as active_agents
            FROM agent_executions
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY hour
            ORDER BY hour DESC
        """)

        hourly_activity = cursor.fetchall()

        return {
            "performance": [dict(row) for row in agent_performance],
            "errors": [dict(row) for row in error_analysis],
            "hourly_activity": [dict(row) for row in hourly_activity],
            "period_days": lookback_days
        }

    async def _get_lead_pipeline(
        self,
        cursor: Any,
        parameters: Dict
    ) -> Dict:
        """Get lead pipeline analytics"""
        # Lead funnel
        cursor.execute("""
            SELECT
                status,
                COUNT(*) as count,
                AVG(EXTRACT(EPOCH FROM (NOW() - created_at))/86400) as avg_age_days,
                SUM(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 ELSE 0 END) as new_this_week
            FROM leads
            GROUP BY status
            ORDER BY count DESC
        """)

        lead_funnel = cursor.fetchall()

        # Lead sources
        cursor.execute("""
            SELECT
                COALESCE(source, 'Unknown') as source,
                COUNT(*) as count,
                AVG(CASE WHEN status = 'converted' THEN 1 ELSE 0 END) * 100 as conversion_rate
            FROM leads
            GROUP BY source
            ORDER BY count DESC
        """)

        lead_sources = cursor.fetchall()

        # Lead velocity
        cursor.execute("""
            SELECT
                DATE(created_at) as date,
                COUNT(*) as new_leads,
                SUM(COUNT(*)) OVER (ORDER BY DATE(created_at) ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) / 7 as rolling_avg
            FROM leads
            WHERE created_at >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """)

        lead_velocity = cursor.fetchall()

        return {
            "funnel": [dict(row) for row in lead_funnel],
            "sources": [dict(row) for row in lead_sources],
            "velocity": [dict(row) for row in lead_velocity]
        }

    async def _get_ai_system_status(
        self,
        cursor: Any,
        parameters: Dict
    ) -> Dict:
        """Get comprehensive AI system status"""
        # System health
        cursor.execute("""
            SELECT
                'Memory Contexts' as component,
                COUNT(*) as total,
                MAX(created_at) as last_update
            FROM ai_master_context
            UNION ALL
            SELECT 'Knowledge Graph', COUNT(*), MAX(created_at)
            FROM ai_knowledge_graph
            UNION ALL
            SELECT 'Conversations', COUNT(*), MAX(created_at)
            FROM ai_conversations
            UNION ALL
            SELECT 'Decision Trees', COUNT(*), MAX(created_at)
            FROM ai_decision_trees
            UNION ALL
            SELECT 'Predictions', COUNT(*), MAX(created_at)
            FROM ai_predictions
        """)

        component_status = cursor.fetchall()

        # Feature utilization
        cursor.execute("""
            WITH feature_usage AS (
                SELECT 'Lead Nurturing' as feature, COUNT(*) as usage
                FROM ai_nurture_sequences
                UNION ALL
                SELECT 'Customer Onboarding', COUNT(*)
                FROM ai_onboarding_journeys
                UNION ALL
                SELECT 'Follow-up System', COUNT(*)
                FROM ai_followup_sequences
                UNION ALL
                SELECT 'Training Pipeline', COUNT(*)
                FROM ai_training_jobs
            )
            SELECT * FROM feature_usage ORDER BY usage DESC
        """)

        feature_utilization = cursor.fetchall()

        # Real-time activity
        cursor.execute("""
            SELECT
                COUNT(*) as total_agents,
                SUM(CASE WHEN last_active > NOW() - INTERVAL '1 hour' THEN 1 ELSE 0 END) as active_last_hour,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as currently_active,
                AVG(EXTRACT(EPOCH FROM (NOW() - last_active))/3600) as avg_hours_since_active
            FROM ai_agents
        """)

        agent_status = cursor.fetchone()

        return {
            "components": [dict(row) for row in component_status],
            "features": [dict(row) for row in feature_utilization],
            "agents": dict(agent_status) if agent_status else {}
        }

    async def _generate_insights(
        self,
        report_type: ReportType,
        metrics: Dict,
        report_data: Dict
    ) -> List[Dict]:
        """Generate actionable insights from report data"""
        insights = []

        if report_type == ReportType.DAILY_SUMMARY:
            summary = report_data.get('summary', {})
            if summary.get('new_customers', 0) > 10:
                insights.append({
                    "type": "positive",
                    "category": "growth",
                    "message": f"Strong customer acquisition with {summary['new_customers']} new customers",
                    "action": "Scale onboarding resources"
                })

            revenue = summary.get('revenue') or 0
            if revenue > 10000:
                insights.append({
                    "type": "positive",
                    "category": "revenue",
                    "message": f"Excellent revenue day: ${revenue:,.2f}",
                    "action": "Analyze success factors for replication"
                })

        elif report_type == ReportType.AGENT_PERFORMANCE:
            for agent in report_data.get('performance', []):
                if agent.get('success_rate', 0) < 80:
                    insights.append({
                        "type": "warning",
                        "category": "performance",
                        "message": f"{agent['agent_type']} has low success rate: {agent['success_rate']:.1f}%",
                        "action": "Review error logs and optimize agent logic"
                    })

        elif report_type == ReportType.LEAD_PIPELINE:
            funnel = report_data.get('funnel', [])
            total_leads = sum(item.get('count', 0) for item in funnel)
            if total_leads > 0:
                new_leads = next((item['count'] for item in funnel if item.get('status') == 'new'), 0)
                if new_leads / total_leads > 0.5:
                    insights.append({
                        "type": "alert",
                        "category": "pipeline",
                        "message": f"{new_leads} leads need attention (>{(new_leads/total_leads)*100:.0f}% of pipeline)",
                        "action": "Prioritize lead qualification and outreach"
                    })

        return insights

    async def _schedule_report(
        self,
        report_id: str,
        report_type: ReportType,
        parameters: Dict,
        schedule: ReportSchedule,
        cursor: Any
    ) -> None:
        """Schedule recurring report generation"""
        schedule_id = str(uuid.uuid4())

        next_run = self._calculate_next_run(schedule)

        cursor.execute("""
            INSERT INTO ai_report_schedules (
                id, report_id, report_type, parameters,
                schedule, next_run, is_active, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, true, NOW())
        """, (
            schedule_id, report_id, report_type.value,
            json.dumps(parameters), schedule.value, next_run
        ))

    def _calculate_next_run(self, schedule: ReportSchedule) -> datetime:
        """Calculate next run time for scheduled report"""
        now = datetime.now(timezone.utc)

        if schedule == ReportSchedule.HOURLY:
            return now + timedelta(hours=1)
        elif schedule == ReportSchedule.DAILY:
            return now.replace(hour=8, minute=0, second=0) + timedelta(days=1)
        elif schedule == ReportSchedule.WEEKLY:
            return now + timedelta(days=7 - now.weekday())
        elif schedule == ReportSchedule.MONTHLY:
            next_month = now.month + 1 if now.month < 12 else 1
            year = now.year if now.month < 12 else now.year + 1
            return now.replace(year=year, month=next_month, day=1, hour=8, minute=0)
        else:
            return now + timedelta(days=1)

    async def _handle_alerts(
        self,
        alerts: List[Dict],
        report_id: str,
        cursor: Any
    ) -> None:
        """Handle alerts triggered by report metrics"""
        for alert in alerts:
            alert_id = str(uuid.uuid4())

            cursor.execute("""
                INSERT INTO ai_report_alerts (
                    id, report_id, alert_type, severity,
                    alert_message, threshold_value, actual_value,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                alert_id, report_id, alert['type'], alert.get('severity', 'info'),
                alert['message'], alert.get('threshold'),
                alert.get('actual_value')
            ))

            # Send notifications if critical
            if alert['severity'] == 'critical':
                await self.distribution_manager.send_alert(alert)

    async def get_report(self, report_id: str) -> Dict:
        """Retrieve a generated report"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM ai_reports
                WHERE id = %s
            """, (report_id,))

            report = cursor.fetchone()

            if not report:
                return {"error": "Report not found"}

            return {
                "id": report['id'],
                "type": report['report_type'],
                "format": report['format'],
                "data": json.loads(report['data']) if report['data'] else {},
                "metrics": json.loads(report['metrics']) if report['metrics'] else {},
                "insights": json.loads(report['insights']) if report['insights'] else [],
                "created_at": report['created_at'].isoformat() if report['created_at'] else None
            }

        except Exception as e:
            logger.error(f"Error retrieving report: {e}")
            return {"error": str(e)}

    async def execute_scheduled_reports(self) -> List[str]:
        """Execute all due scheduled reports"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get due reports
            cursor.execute("""
                SELECT * FROM ai_report_schedules
                WHERE is_active = true
                AND next_run <= NOW()
                ORDER BY next_run
                LIMIT 10
            """)

            scheduled_reports = cursor.fetchall()
            generated_reports = []

            for schedule in scheduled_reports:
                # Generate report
                report_id = await self.create_report(
                    ReportType(schedule['report_type']),
                    json.loads(schedule['parameters']),
                    ReportFormat.JSON,
                    ReportSchedule(schedule['schedule'])
                )

                generated_reports.append(report_id)

                # Update next run
                next_run = self._calculate_next_run(
                    ReportSchedule(schedule['schedule'])
                )

                cursor.execute("""
                    UPDATE ai_report_schedules
                    SET next_run = %s,
                        last_run = NOW()
                    WHERE id = %s
                """, (next_run, schedule['id']))

            conn.commit()

            logger.info(f"Generated {len(generated_reports)} scheduled reports")
            return generated_reports

        except Exception as e:
            logger.error(f"Error executing scheduled reports: {e}")
            if conn:
                conn.rollback()
            return []


class ReportGenerator:
    """Generate formatted reports"""

    async def format_report(
        self,
        data: Dict,
        metrics: Dict,
        insights: List[Dict],
        format: ReportFormat
    ) -> str:
        """Format report in requested format"""
        if format == ReportFormat.JSON:
            return json.dumps({
                "data": data,
                "metrics": metrics,
                "insights": insights,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }, default=str)

        elif format == ReportFormat.HTML:
            return self._generate_html_report(data, metrics, insights)

        elif format == ReportFormat.CSV:
            return self._generate_csv_report(data)

        else:
            return json.dumps(data, default=str)

    def _generate_html_report(
        self,
        data: Dict,
        metrics: Dict,
        insights: List[Dict]
    ) -> str:
        """Generate HTML formatted report"""
        html = f"""
        <html>
        <head>
            <title>Report - {datetime.now().strftime('%Y-%m-%d')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ padding: 10px; margin: 10px 0; background: #f0f0f0; }}
                .insight {{ padding: 10px; margin: 10px 0; border-left: 3px solid #007bff; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Report Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</h1>
        """

        # Add metrics
        html += "<h2>Key Metrics</h2>"
        for key, value in metrics.items():
            html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'

        # Add insights
        if insights:
            html += "<h2>Insights</h2>"
            for insight in insights:
                html += f'<div class="insight">{insight.get("message", "")}</div>'

        html += "</body></html>"
        return html

    def _generate_csv_report(self, data: Dict) -> str:
        """Generate CSV formatted report"""
        # Simple CSV generation - would be expanded based on data structure
        csv_lines = []

        # Headers
        if isinstance(data, dict) and data:
            first_item = next(iter(data.values()))
            if isinstance(first_item, list) and first_item:
                headers = list(first_item[0].keys())
                csv_lines.append(','.join(headers))

                # Data rows
                for item in first_item:
                    row = [str(item.get(h, '')) for h in headers]
                    csv_lines.append(','.join(row))

        return '\n'.join(csv_lines)


class MetricCalculator:
    """Calculate metrics from report data"""

    async def calculate_metrics(
        self,
        report_type: ReportType,
        data: Dict
    ) -> Dict:
        """Calculate relevant metrics for report type"""
        metrics = {}

        if report_type == ReportType.DAILY_SUMMARY:
            summary = data.get('summary', {})
            metrics['total_activity'] = sum(
                v for k, v in summary.items()
                if isinstance(v, (int, float)) and k != 'revenue'
            )
            metrics['revenue'] = float(summary.get('revenue', 0) or 0)
            metrics['ai_utilization'] = summary.get('ai_executions', 0)

        elif report_type == ReportType.AGENT_PERFORMANCE:
            performance = data.get('performance', [])
            if performance:
                metrics['avg_success_rate'] = sum(
                    p.get('success_rate', 0) for p in performance
                ) / len(performance)
                metrics['total_executions'] = sum(
                    p.get('total_executions', 0) for p in performance
                )

        elif report_type == ReportType.LEAD_PIPELINE:
            funnel = data.get('funnel', [])
            metrics['total_leads'] = sum(item.get('count', 0) for item in funnel)
            metrics['avg_lead_age'] = sum(
                item.get('avg_age_days', 0) * item.get('count', 0)
                for item in funnel
            ) / max(metrics['total_leads'], 1)

        return metrics


class VisualizationEngine:
    """Create data visualizations"""

    async def create_visualization(
        self,
        data: Dict,
        viz_type: str
    ) -> Dict:
        """Create visualization configuration"""
        # This would integrate with charting libraries
        return {
            "type": viz_type,
            "data": data,
            "config": {
                "responsive": True,
                "theme": "light"
            }
        }


class DistributionManager:
    """Manage report distribution"""

    async def distribute_report(
        self,
        report_id: str,
        recipients: List[str],
        format: ReportFormat
    ) -> Dict:
        """Distribute report to recipients"""
        # This would integrate with email/notification services
        return {
            "report_id": report_id,
            "sent_to": recipients,
            "format": format.value,
            "sent_at": datetime.now(timezone.utc).isoformat()
        }

    async def send_alert(self, alert: Dict) -> None:
        """Send critical alert"""
        logger.warning(f"ALERT: {alert.get('message')}")
        # Would integrate with notification services


class AlertSystem:
    """Monitor metrics and trigger alerts"""

    async def check_alerts(
        self,
        report_type: ReportType,
        metrics: Dict
    ) -> List[Dict]:
        """Check if metrics trigger any alerts"""
        alerts = []

        # Define alert thresholds
        if report_type == ReportType.AGENT_PERFORMANCE:
            if metrics.get('avg_success_rate', 100) < 70:
                alerts.append({
                    "type": "performance",
                    "severity": "warning",
                    "message": "Agent success rate below threshold",
                    "threshold": 70,
                    "actual_value": metrics.get('avg_success_rate')
                })

        elif report_type == ReportType.DAILY_SUMMARY:
            if metrics.get('revenue', 0) < 1000:
                alerts.append({
                    "type": "revenue",
                    "severity": "info",
                    "message": "Daily revenue below target",
                    "threshold": 1000,
                    "actual_value": metrics.get('revenue')
                })

        return alerts


# Singleton instance
_reporting_system = None

def get_automated_reporting_system():
    """Get singleton instance of automated reporting system"""
    global _reporting_system
    if _reporting_system is None:
        _reporting_system = AutomatedReportingSystem()
    return _reporting_system


# Export main components
__all__ = [
    'AutomatedReportingSystem',
    'ReportType',
    'ReportFormat',
    'ReportSchedule',
    'MetricType',
    'ReportGenerator',
    'MetricCalculator',
    'VisualizationEngine',
    'DistributionManager',
    'AlertSystem',
    'get_automated_reporting_system'
]
#!/usr/bin/env python3
"""
System Improvement Agent - Continuously Analyzes and Improves Systems
Monitors performance, detects inefficiencies, proposes optimizations

This agent is designed to achieve the ultimate vision of a self-improving system.
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


class ImprovementType(Enum):
    """Types of improvements the agent can propose"""
    PERFORMANCE = "performance"
    COST = "cost"
    RELIABILITY = "reliability"
    SECURITY = "security"
    USABILITY = "usability"
    SCALABILITY = "scalability"
    MAINTAINABILITY = "maintainability"


class ImprovementPriority(Enum):
    """Priority levels for improvements"""
    CRITICAL = 1  # Fix immediately
    HIGH = 2      # Fix this week
    MEDIUM = 3    # Fix this month
    LOW = 4       # Nice to have


@dataclass
class Inefficiency:
    """Detected system inefficiency"""
    id: str
    type: ImprovementType
    severity: float  # 0-100
    title: str
    description: str
    current_state: str
    desired_state: str
    impact: str
    detected_at: datetime
    affected_systems: List[str]
    metrics: Dict[str, Any]


@dataclass
class ImprovementProposal:
    """Proposed improvement to the system"""
    id: str
    inefficiency_id: str
    title: str
    description: str
    improvement_type: ImprovementType
    priority: ImprovementPriority
    estimated_effort_hours: float
    estimated_impact: str
    benefits: List[str]
    risks: List[str]
    implementation_steps: List[str]
    success_criteria: List[str]
    created_at: datetime
    status: str  # proposed, approved, implementing, completed, rejected


class SystemImprovementAgent:
    """Agent that continuously improves the system"""

    def __init__(self):
        self.db_config = DB_CONFIG
        self.conn = None
        self.inefficiencies_detected = []
        self.proposals_made = []
        self._init_database()
        logger.info("✅ System Improvement Agent initialized")

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Create inefficiencies table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_system_inefficiencies (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    type VARCHAR(50) NOT NULL,
                    severity FLOAT NOT NULL CHECK (severity >= 0 AND severity <= 100),
                    title TEXT NOT NULL,
                    description TEXT,
                    current_state TEXT,
                    desired_state TEXT,
                    impact TEXT,
                    detected_at TIMESTAMP DEFAULT NOW(),
                    affected_systems JSONB,
                    metrics JSONB DEFAULT '{}'::jsonb,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create improvement proposals table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_improvement_proposals (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    inefficiency_id UUID REFERENCES ai_system_inefficiencies(id),
                    title TEXT NOT NULL,
                    description TEXT,
                    improvement_type VARCHAR(50),
                    priority INTEGER CHECK (priority >= 1 AND priority <= 4),
                    estimated_effort_hours FLOAT,
                    estimated_impact TEXT,
                    benefits JSONB DEFAULT '[]'::jsonb,
                    risks JSONB DEFAULT '[]'::jsonb,
                    implementation_steps JSONB DEFAULT '[]'::jsonb,
                    success_criteria JSONB DEFAULT '[]'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW(),
                    approved_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'proposed'
                        CHECK (status IN ('proposed', 'approved', 'implementing', 'completed', 'rejected')),
                    approver TEXT,
                    implementation_notes TEXT,
                    actual_effort_hours FLOAT,
                    actual_impact TEXT,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create system metrics tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_system_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    metric_name VARCHAR(255) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    metric_unit VARCHAR(50),
                    metric_category VARCHAR(50),
                    baseline_value FLOAT,
                    target_value FLOAT,
                    threshold_warning FLOAT,
                    threshold_critical FLOAT,
                    measured_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            conn.commit()
            logger.info("✅ System Improvement Agent database tables ready")

        except Exception as e:
            logger.warning(f"Database initialization skipped: {e}. Operating without persistence")
        finally:
            if conn:
                conn.close()

    def analyze_performance_metrics(self) -> List[Inefficiency]:
        """Analyze system performance and detect inefficiencies"""
        inefficiencies = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for slow agent executions
            cur.execute("""
                SELECT
                    AVG(duration_ms) as avg_duration,
                    MAX(duration_ms) as max_duration,
                    COUNT(*) as total_executions
                FROM agent_executions
                WHERE completed_at > NOW() - INTERVAL '24 hours'
                AND status = 'completed'
            """)

            perf_data = cur.fetchone()

            if perf_data and perf_data['avg_duration']:
                avg_ms = float(perf_data['avg_duration'])
                max_ms = float(perf_data['max_duration'])

                # Flag if average execution > 5 seconds
                if avg_ms > 5000:
                    severity = min(100, (avg_ms / 10000) * 100)
                    inefficiencies.append(Inefficiency(
                        id=f"perf_slow_avg_{datetime.now().timestamp()}",
                        type=ImprovementType.PERFORMANCE,
                        severity=severity,
                        title="Slow Average Agent Execution Time",
                        description=f"Average agent execution time is {avg_ms/1000:.2f}s, which exceeds the 5s target",
                        current_state=f"{avg_ms/1000:.2f}s average execution time",
                        desired_state="< 5s average execution time",
                        impact=f"Slower user experience, {perf_data['total_executions']} executions affected in 24h",
                        detected_at=datetime.now(),
                        affected_systems=["Agent Execution Engine"],
                        metrics={"avg_ms": avg_ms, "max_ms": max_ms, "count": perf_data['total_executions']}
                    ))

            # Check for failed executions
            cur.execute("""
                SELECT
                    COUNT(*) as failed_count,
                    COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM agent_executions WHERE completed_at > NOW() - INTERVAL '24 hours'), 0) as failure_rate
                FROM agent_executions
                WHERE completed_at > NOW() - INTERVAL '24 hours'
                AND status = 'failed'
            """)

            failure_data = cur.fetchone()
            if failure_data and failure_data['failure_rate'] and failure_data['failure_rate'] > 5:
                inefficiencies.append(Inefficiency(
                    id=f"reliability_high_failure_{datetime.now().timestamp()}",
                    type=ImprovementType.RELIABILITY,
                    severity=min(100, failure_data['failure_rate'] * 10),
                    title="High Agent Execution Failure Rate",
                    description=f"Agent failure rate is {failure_data['failure_rate']:.1f}%, exceeding 5% threshold",
                    current_state=f"{failure_data['failure_rate']:.1f}% failure rate ({failure_data['failed_count']} failures)",
                    desired_state="< 5% failure rate",
                    impact="Reduced system reliability, user trust issues",
                    detected_at=datetime.now(),
                    affected_systems=["Agent Execution Engine", "Error Handling"],
                    metrics={"failure_rate": failure_data['failure_rate'], "failed_count": failure_data['failed_count']}
                ))

            conn.close()

        except Exception as e:
            logger.warning(f"Performance analysis failed: {e}")

        return inefficiencies

    def analyze_resource_usage(self) -> List[Inefficiency]:
        """Analyze resource usage for optimization opportunities"""
        inefficiencies = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for unused agents
            cur.execute("""
                SELECT
                    COUNT(*) as unused_agents
                FROM agents a
                WHERE enabled = true
                AND NOT EXISTS (
                    SELECT 1 FROM agent_executions ae
                    WHERE ae.agent_id = a.id
                    AND ae.completed_at > NOW() - INTERVAL '30 days'
                )
            """)

            unused_data = cur.fetchone()
            if unused_data and unused_data['unused_agents'] > 0:
                inefficiencies.append(Inefficiency(
                    id=f"cost_unused_agents_{datetime.now().timestamp()}",
                    type=ImprovementType.COST,
                    severity=min(100, unused_data['unused_agents'] * 5),
                    title="Unused Agents Enabled",
                    description=f"{unused_data['unused_agents']} enabled agents haven't been used in 30 days",
                    current_state=f"{unused_data['unused_agents']} unused agents consuming resources",
                    desired_state="All enabled agents actively used or disabled",
                    impact="Wasted resources, maintenance overhead, confusion",
                    detected_at=datetime.now(),
                    affected_systems=["Agent Management"],
                    metrics={"unused_count": unused_data['unused_agents']}
                ))

            conn.close()

        except Exception as e:
            logger.warning(f"Resource analysis failed: {e}")

        return inefficiencies

    def generate_improvement_proposals(self, inefficiency: Inefficiency) -> ImprovementProposal:
        """Generate improvement proposal from detected inefficiency"""

        proposals_map = {
            "perf_slow_avg": {
                "title": "Optimize Agent Execution Performance",
                "description": "Implement caching, async optimization, and database query improvements to reduce execution time",
                "priority": ImprovementPriority.HIGH,
                "effort": 16,
                "benefits": [
                    "50%+ faster agent execution times",
                    "Better user experience",
                    "Higher system throughput",
                    "Reduced server costs"
                ],
                "risks": [
                    "Caching may introduce stale data",
                    "Async changes need thorough testing"
                ],
                "steps": [
                    "Profile slow executions to find bottlenecks",
                    "Implement Redis caching for frequent queries",
                    "Optimize database indexes",
                    "Add async/await where beneficial",
                    "Load test improvements"
                ],
                "success_criteria": [
                    "Average execution time < 5s",
                    "95th percentile < 10s",
                    "No regressions in functionality"
                ]
            },
            "reliability_high_failure": {
                "title": "Improve Agent Execution Reliability",
                "description": "Enhanced error handling, retry logic, and validation to reduce failure rate",
                "priority": ImprovementPriority.CRITICAL,
                "effort": 12,
                "benefits": [
                    "Failure rate reduced below 5%",
                    "Better error messages for debugging",
                    "Automatic recovery from transient failures",
                    "Improved user trust"
                ],
                "risks": [
                    "Retry logic may mask underlying issues",
                    "Could increase execution times slightly"
                ],
                "steps": [
                    "Analyze failure patterns and root causes",
                    "Implement exponential backoff retry",
                    "Add input validation before execution",
                    "Enhance error logging and monitoring",
                    "Create failure rate dashboard"
                ],
                "success_criteria": [
                    "Failure rate < 5%",
                    "All failures logged with actionable info",
                    "Auto-recovery for > 80% of transient failures"
                ]
            },
            "cost_unused_agents": {
                "title": "Clean Up Unused Agents",
                "description": "Disable or remove agents that haven't been used in 30+ days to reduce maintenance",
                "priority": ImprovementPriority.MEDIUM,
                "effort": 4,
                "benefits": [
                    "Reduced system complexity",
                    "Lower maintenance burden",
                    "Clearer agent list for users",
                    "Faster agent listing queries"
                ],
                "risks": [
                    "May disable agents needed for seasonal tasks",
                    "Users may expect all agents available"
                ],
                "steps": [
                    "Review unused agents with stakeholders",
                    "Mark agents as deprecated first",
                    "Monitor for any usage after deprecation",
                    "Disable after 60 days if still unused",
                    "Document agent sunset process"
                ],
                "success_criteria": [
                    "90%+ of enabled agents used monthly",
                    "Clear deprecation process documented",
                    "No critical agents disabled by mistake"
                ]
            }
        }

        # Find matching proposal template
        inefficiency_prefix = inefficiency.id.split('_')[0] + '_' + inefficiency.id.split('_')[1]
        template = proposals_map.get(inefficiency_prefix, proposals_map.get("perf_slow_avg"))

        return ImprovementProposal(
            id=f"proposal_{inefficiency.id}",
            inefficiency_id=inefficiency.id,
            title=template["title"],
            description=template["description"],
            improvement_type=inefficiency.type,
            priority=template["priority"],
            estimated_effort_hours=template["effort"],
            estimated_impact=inefficiency.impact,
            benefits=template["benefits"],
            risks=template["risks"],
            implementation_steps=template["steps"],
            success_criteria=template["success_criteria"],
            created_at=datetime.now(),
            status="proposed"
        )

    async def continuous_improvement_loop(self, interval_hours: int = 6):
        """Main loop that continuously analyzes and proposes improvements"""
        logger.info(f"🔄 Starting continuous improvement loop (every {interval_hours}h)")

        while True:
            try:
                logger.info("🔍 Analyzing system for improvement opportunities...")

                # Collect inefficiencies from all analysis methods
                all_inefficiencies = []
                all_inefficiencies.extend(self.analyze_performance_metrics())
                all_inefficiencies.extend(self.analyze_resource_usage())

                logger.info(f"📊 Detected {len(all_inefficiencies)} inefficiencies")

                # Generate proposals for each inefficiency
                for inefficiency in all_inefficiencies:
                    proposal = self.generate_improvement_proposals(inefficiency)
                    self.proposals_made.append(proposal)

                    logger.info(f"💡 Proposal: {proposal.title} (Priority: {proposal.priority.name})")
                    logger.info(f"   Estimated effort: {proposal.estimated_effort_hours}h")
                    logger.info(f"   Impact: {proposal.estimated_impact}")

                # Store in database for review
                self._persist_findings(all_inefficiencies, self.proposals_made)

            except Exception as e:
                logger.error(f"❌ Improvement loop error: {e}")

            # Wait before next analysis
            await asyncio.sleep(interval_hours * 3600)

    def _persist_findings(self, inefficiencies: List[Inefficiency], proposals: List[ImprovementProposal]):
        """Persist inefficiencies and proposals to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            for ineff in inefficiencies:
                cur.execute("""
                    INSERT INTO ai_system_inefficiencies
                    (type, severity, title, description, current_state, desired_state, impact, affected_systems, metrics)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    ineff.type.value, ineff.severity, ineff.title, ineff.description,
                    ineff.current_state, ineff.desired_state, ineff.impact,
                    Json(ineff.affected_systems), Json(ineff.metrics)
                ))

            for proposal in proposals:
                cur.execute("""
                    INSERT INTO ai_improvement_proposals
                    (title, description, improvement_type, priority, estimated_effort_hours, estimated_impact,
                     benefits, risks, implementation_steps, success_criteria, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    proposal.title, proposal.description, proposal.improvement_type.value,
                    proposal.priority.value, proposal.estimated_effort_hours, proposal.estimated_impact,
                    Json(proposal.benefits), Json(proposal.risks), Json(proposal.implementation_steps),
                    Json(proposal.success_criteria), proposal.status
                ))

            conn.commit()
            conn.close()
            logger.info("✅ Findings persisted to database")

        except Exception as e:
            logger.warning(f"Failed to persist findings: {e}")


if __name__ == "__main__":
    agent = SystemImprovementAgent()
    asyncio.run(agent.continuous_improvement_loop(interval_hours=6))

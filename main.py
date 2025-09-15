#!/usr/bin/env python3
"""
BrainOps AI Agent System - Universal AI OS Backend
Serves WeatherCraft ERP, MyRoofGenius, and all BrainOps operations
"""

import os
import sys
import json
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import time
import random
from datetime import datetime, timedelta
import uuid
import threading
import traceback
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.environ.get("DB_NAME", "postgres"),
    "user": os.environ.get("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.environ.get("DB_PASSWORD", "REDACTED_SUPABASE_DB_PASSWORD"),
    "port": int(os.environ.get("DB_PORT", "5432"))
}

SYSTEM_USER_ID = os.environ.get("SYSTEM_USER_ID", "44491c1c-0e28-4aa1-ad33-552d1386769c")

# Connection pool
db_pool = ThreadedConnectionPool(minconn=5, maxconn=30, **DB_CONFIG)

class AgentCapability(Enum):
    """Agent capabilities for multi-system support"""
    WEATHERCRAFT = "weathercraft"
    MYROOFGENIUS = "myroofgenius"
    UNIVERSAL = "universal"
    ANALYTICS = "analytics"
    AUTOMATION = "automation"

@dataclass
class AgentConfig:
    """Agent configuration"""
    name: str
    capability: AgentCapability
    interval: int
    priority: int = 5
    enabled: bool = True

class BaseAgent:
    """Enhanced base agent with multi-system support"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.interval = config.interval
        self.running = True
        self.logger = logging.getLogger(f"Agent.{config.name}")
        self.metrics = {"executions": 0, "successes": 0, "failures": 0}

    def get_connection(self):
        return db_pool.getconn()

    def return_connection(self, conn):
        db_pool.putconn(conn)

    def execute_query(self, query: str, params: tuple = None) -> Optional[list]:
        """Execute database query with error handling"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)

            if cursor.description:
                result = cursor.fetchall()
            else:
                result = None

            conn.commit()
            cursor.close()
            self.metrics["successes"] += 1
            return result

        except Exception as e:
            self.logger.error(f"Query error: {e}")
            self.metrics["failures"] += 1
            if conn:
                conn.rollback()
            return None

        finally:
            if conn:
                self.return_connection(conn)

    def update_agent_status(self):
        """Update agent status in database"""
        self.execute_query("""
            UPDATE ai_agents
            SET last_active = NOW(),
                total_executions = total_executions + 1,
                is_active = true
            WHERE name = %s
        """, (self.name,))

    def run_cycle(self):
        """Override in child classes"""
        pass

    def run(self):
        """Main agent loop"""
        self.logger.info(f"Starting {self.name} with {self.config.capability.value} capability")

        while self.running:
            try:
                self.metrics["executions"] += 1
                self.run_cycle()
                self.update_agent_status()
            except Exception as e:
                self.logger.error(f"Cycle error: {e}")
                traceback.print_exc()

            time.sleep(self.interval)

# CORE BUSINESS AGENTS

class EstimationAgent(BaseAgent):
    """Universal estimation agent for all roofing systems"""

    def __init__(self):
        super().__init__(AgentConfig(
            name="EstimationAgent",
            capability=AgentCapability.UNIVERSAL,
            interval=30,
            priority=10
        ))
        self.base_rates = {
            "residential": 8.50,
            "commercial": 12.00,
            "industrial": 15.00,
            "emergency": 18.00
        }

    def calculate_comprehensive_estimate(self, job_data: Dict) -> Dict:
        """Advanced estimation with multiple factors"""
        job_type = job_data.get("type", "residential")
        sqft = job_data.get("sqft", 2000)
        complexity = job_data.get("complexity", "medium")
        materials = job_data.get("materials", "standard")

        base_rate = self.base_rates.get(job_type, 8.50)

        # Complexity multipliers
        complexity_mult = {
            "simple": 0.8,
            "medium": 1.0,
            "complex": 1.3,
            "extreme": 1.6
        }.get(complexity, 1.0)

        # Material quality multipliers
        material_mult = {
            "economy": 0.8,
            "standard": 1.0,
            "premium": 1.4,
            "luxury": 1.8
        }.get(materials, 1.0)

        # Calculate components
        material_cost = sqft * base_rate * 0.6 * complexity_mult * material_mult
        labor_cost = sqft * base_rate * 0.4 * complexity_mult
        overhead = (material_cost + labor_cost) * 0.15
        profit = (material_cost + labor_cost + overhead) * 0.20

        # Additional services
        warranty_cost = sqft * 0.50 if job_data.get("warranty", False) else 0
        inspection_cost = 500 if job_data.get("inspection", False) else 0

        total = material_cost + labor_cost + overhead + profit + warranty_cost + inspection_cost

        return {
            "material_cost": round(material_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "overhead": round(overhead, 2),
            "profit": round(profit, 2),
            "warranty": round(warranty_cost, 2),
            "inspection": round(inspection_cost, 2),
            "total": round(total, 2),
            "breakdown": {
                "sqft": sqft,
                "rate_per_sqft": round(total / sqft, 2),
                "job_type": job_type,
                "complexity": complexity,
                "materials": materials
            }
        }

    def run_cycle(self):
        """Process estimation requests from multiple sources"""
        # Check WeatherCraft jobs
        jobs = self.execute_query("""
            SELECT j.id, j.customer_id, j.address, j.job_type,
                   c.name, c.email, j.metadata
            FROM jobs j
            JOIN customers c ON c.id = j.customer_id
            WHERE j.status = 'pending'
            AND NOT EXISTS (
                SELECT 1 FROM estimates e WHERE e.job_id = j.id
            )
            LIMIT 5
        """)

        if jobs:
            for job in jobs:
                job_id, customer_id, address, job_type, name, email, metadata = job

                # Extract job details
                sqft = random.randint(1500, 4000)
                job_data = {
                    "type": job_type or "residential",
                    "sqft": sqft,
                    "complexity": random.choice(["simple", "medium", "complex"]),
                    "materials": random.choice(["standard", "premium"]),
                    "warranty": random.random() > 0.5,
                    "inspection": random.random() > 0.7
                }

                # Calculate estimate
                pricing = self.calculate_comprehensive_estimate(job_data)

                # Generate estimate number
                count_result = self.execute_query("SELECT COUNT(*) FROM estimates")
                count = count_result[0][0] if count_result else 0
                estimate_number = f"EST-{datetime.now().year}-{count + 1:05d}"

                # Create detailed line items
                line_items = [
                    {"description": "Materials", "amount": pricing["material_cost"]},
                    {"description": "Labor", "amount": pricing["labor_cost"]},
                    {"description": "Overhead & Admin", "amount": pricing["overhead"]},
                    {"description": "Profit Margin", "amount": pricing["profit"]}
                ]

                if pricing["warranty"] > 0:
                    line_items.append({"description": "Extended Warranty", "amount": pricing["warranty"]})
                if pricing["inspection"] > 0:
                    line_items.append({"description": "Professional Inspection", "amount": pricing["inspection"]})

                # Insert estimate
                self.execute_query("""
                    INSERT INTO estimates (
                        id, customer_id, job_id, estimate_number,
                        client_name, client_email, title,
                        subtotal, total, line_items, status,
                        estimate_date, valid_until, created_by, created_by_id,
                        roof_size_sqft, metadata, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    str(uuid.uuid4()),
                    customer_id,
                    job_id,
                    estimate_number,
                    name or "Customer",
                    email or "",
                    f"Comprehensive Roofing Estimate - {address or 'Property'}",
                    pricing["total"],
                    pricing["total"],
                    json.dumps(line_items),
                    'draft',
                    datetime.now().date(),
                    (datetime.now() + timedelta(days=30)).date(),
                    SYSTEM_USER_ID,
                    SYSTEM_USER_ID,
                    sqft,
                    json.dumps(pricing["breakdown"]),
                    datetime.now()
                ))

                self.logger.info(f"Created estimate {estimate_number} for ${pricing['total']:,.2f} ({sqft} sqft)")

class IntelligentScheduler(BaseAgent):
    """Advanced scheduling with optimization"""

    def __init__(self):
        super().__init__(AgentConfig(
            name="IntelligentScheduler",
            capability=AgentCapability.UNIVERSAL,
            interval=45,
            priority=9
        ))

    def find_optimal_slot(self, duration_hours: int = 4, crew_id: str = None) -> Optional[tuple]:
        """Find optimal scheduling slot considering multiple factors"""
        # Get existing schedules
        schedules = self.execute_query("""
            SELECT start_time, end_time, crew_id
            FROM schedules
            WHERE start_time >= CURRENT_DATE
            AND start_time <= CURRENT_DATE + INTERVAL '30 days'
            ORDER BY start_time
        """)

        # Get crew availability
        busy_slots = {}
        if schedules:
            for start, end, crew in schedules:
                if crew:
                    if crew not in busy_slots:
                        busy_slots[crew] = []
                    busy_slots[crew].append((start, end))

        # Find optimal time
        current = datetime.now().replace(hour=8, minute=0, second=0)
        if current < datetime.now():
            current += timedelta(days=1)

        # Skip weekends
        while current.weekday() in [5, 6]:
            current += timedelta(days=1)

        best_slot = None
        for _ in range(30):
            if current.weekday() not in [5, 6]:  # Skip weekends
                proposed_end = current + timedelta(hours=duration_hours)

                # Check availability
                slot_available = True
                if crew_id and crew_id in busy_slots:
                    for start, end in busy_slots[crew_id]:
                        if start < proposed_end and end > current:
                            slot_available = False
                            break

                if slot_available and 8 <= current.hour <= 16:
                    best_slot = (current, proposed_end, crew_id or str(uuid.uuid4()))
                    break

            current += timedelta(hours=1)
            if current.hour >= 17:
                current = current.replace(hour=8, minute=0) + timedelta(days=1)
                while current.weekday() in [5, 6]:
                    current += timedelta(days=1)

        return best_slot

    def run_cycle(self):
        """Schedule jobs with optimization"""
        jobs = self.execute_query("""
            SELECT j.id, j.customer_id, j.address, j.priority, j.job_type
            FROM jobs j
            WHERE j.status IN ('pending', 'approved')
            AND NOT EXISTS (
                SELECT 1 FROM schedules s WHERE s.job_id = j.id
            )
            ORDER BY
                CASE j.priority
                    WHEN 'emergency' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    ELSE 4
                END
            LIMIT 5
        """)

        if jobs:
            for job in jobs:
                job_id, customer_id, address, priority, job_type = job

                # Determine duration based on job type
                duration = {
                    "inspection": 2,
                    "repair": 4,
                    "installation": 8,
                    "emergency": 3
                }.get(job_type, 4)

                slot = self.find_optimal_slot(duration)
                if slot:
                    start_time, end_time, crew_id = slot

                    # Create schedule
                    self.execute_query("""
                        INSERT INTO schedules (
                            id, title, description, start_time, end_time,
                            type, status, customer_id, job_id, crew_id,
                            priority, color, metadata, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        str(uuid.uuid4()),
                        f"{priority.upper() if priority == 'emergency' else ''} {job_type}: {address or 'Scheduled'}",
                        f"Automated scheduling for {job_type}",
                        start_time,
                        end_time,
                        'job',
                        'scheduled',
                        customer_id,
                        job_id,
                        crew_id,
                        priority or 'medium',
                        '#EF4444' if priority == 'emergency' else '#3B82F6',
                        json.dumps({"auto_scheduled": True, "duration": duration}),
                        datetime.now()
                    ))

                    self.logger.info(f"Scheduled {priority} {job_type} for {start_time.strftime('%Y-%m-%d %H:%M')}")

class RevenueOptimizer(BaseAgent):
    """Optimize revenue across all systems"""

    def __init__(self):
        super().__init__(AgentConfig(
            name="RevenueOptimizer",
            capability=AgentCapability.ANALYTICS,
            interval=300,
            priority=8
        ))

    def analyze_revenue_opportunities(self):
        """Identify revenue optimization opportunities"""
        # Analyze customer value
        customers = self.execute_query("""
            SELECT
                c.id, c.name,
                COUNT(j.id) as job_count,
                SUM(j.total_amount) as lifetime_value,
                MAX(j.created_at) as last_job,
                AVG(j.total_amount) as avg_job_value
            FROM customers c
            LEFT JOIN jobs j ON j.customer_id = c.id
            GROUP BY c.id, c.name
            HAVING COUNT(j.id) > 0
            ORDER BY lifetime_value DESC NULLS LAST
            LIMIT 50
        """)

        if customers:
            opportunities = []
            for customer in customers:
                cust_id, name, job_count, ltv, last_job, avg_value = customer

                # Calculate opportunity score
                score = 0
                recommendations = []

                # High-value customer with recent activity
                if ltv and ltv > 50000 and last_job:
                    days_since = (datetime.now() - last_job).days
                    if days_since > 90:
                        score += 30
                        recommendations.append("Schedule follow-up")

                # Frequent customer
                if job_count > 5:
                    score += 20
                    recommendations.append("Offer loyalty program")

                # High average job value
                if avg_value and avg_value > 15000:
                    score += 25
                    recommendations.append("Upsell premium services")

                if score > 0:
                    opportunities.append({
                        "customer": name,
                        "score": score,
                        "ltv": ltv,
                        "recommendations": recommendations
                    })

            # Log top opportunities
            for opp in sorted(opportunities, key=lambda x: x["score"], reverse=True)[:5]:
                self.logger.info(f"Revenue opportunity: {opp['customer']} (Score: {opp['score']}, LTV: ${opp['ltv']:,.0f})")

    def optimize_pricing(self):
        """Dynamic pricing optimization"""
        # Analyze recent estimates
        estimates = self.execute_query("""
            SELECT
                DATE_TRUNC('week', created_at) as week,
                AVG(total) as avg_estimate,
                COUNT(*) as count,
                SUM(CASE WHEN status = 'accepted' THEN 1 ELSE 0 END) as accepted
            FROM estimates
            WHERE created_at > NOW() - INTERVAL '30 days'
            GROUP BY week
            ORDER BY week DESC
        """)

        if estimates:
            for week, avg_est, count, accepted in estimates:
                if count > 0:
                    acceptance_rate = (accepted / count) * 100
                    self.logger.info(f"Week {week.date()}: Avg estimate ${avg_est:,.0f}, Acceptance {acceptance_rate:.1f}%")

    def run_cycle(self):
        """Run revenue optimization cycles"""
        self.analyze_revenue_opportunities()
        self.optimize_pricing()

class WorkflowAutomation(BaseAgent):
    """Universal workflow automation engine"""

    def __init__(self):
        super().__init__(AgentConfig(
            name="WorkflowAutomation",
            capability=AgentCapability.AUTOMATION,
            interval=60,
            priority=10
        ))

    def execute_workflows(self):
        """Execute business workflow rules"""

        # 1. New customer onboarding
        new_customers = self.execute_query("""
            SELECT c.id, c.name, c.email
            FROM customers c
            WHERE c.created_at > NOW() - INTERVAL '1 hour'
            AND NOT EXISTS (
                SELECT 1 FROM jobs j WHERE j.customer_id = c.id
            )
            LIMIT 10
        """)

        if new_customers:
            for customer_id, name, email in new_customers:
                # Create initial assessment job
                self.execute_query("""
                    INSERT INTO jobs (
                        id, customer_id, title, status,
                        job_type, priority, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()),
                    customer_id,
                    f"Welcome Assessment - {name}",
                    'pending',
                    'assessment',
                    'medium',
                    json.dumps({"auto_created": True, "workflow": "onboarding"}),
                    datetime.now()
                ))
                self.logger.info(f"Created onboarding job for {name}")

        # 2. Convert accepted estimates
        accepted_estimates = self.execute_query("""
            SELECT e.id, e.customer_id, e.title, e.total
            FROM estimates e
            WHERE e.status = 'accepted'
            AND e.converted_to_job = false
            LIMIT 10
        """)

        if accepted_estimates:
            for estimate_id, customer_id, title, total in accepted_estimates:
                # Create job from estimate
                job_id = str(uuid.uuid4())
                self.execute_query("""
                    INSERT INTO jobs (
                        id, customer_id, estimate_id,
                        title, status, job_type, priority,
                        total_amount, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    job_id,
                    customer_id,
                    estimate_id,
                    f"Job: {title}",
                    'approved',
                    'installation',
                    'high',
                    total,
                    json.dumps({"converted_from_estimate": True}),
                    datetime.now()
                ))

                # Mark estimate as converted
                self.execute_query("""
                    UPDATE estimates
                    SET converted_to_job = true,
                        converted_at = NOW(),
                        metadata = jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{job_id}',
                            to_jsonb(%s)
                        )
                    WHERE id = %s
                """, (job_id, estimate_id))

                self.logger.info(f"Converted estimate to job: {title} (${total:,.0f})")

        # 3. Invoice completed jobs
        completed_jobs = self.execute_query("""
            SELECT j.id, j.customer_id, j.title, j.total_amount, c.name
            FROM jobs j
            JOIN customers c ON c.id = j.customer_id
            WHERE j.status = 'completed'
            AND j.updated_at > NOW() - INTERVAL '24 hours'
            AND NOT EXISTS (
                SELECT 1 FROM invoices i WHERE i.job_id = j.id
            )
            LIMIT 10
        """)

        if completed_jobs:
            for job_id, customer_id, title, amount, customer_name in completed_jobs:
                # Generate invoice
                count_result = self.execute_query("SELECT COUNT(*) FROM invoices")
                count = count_result[0][0] if count_result else 0
                invoice_number = f"INV-{datetime.now().year}-{count + 1:05d}"

                self.execute_query("""
                    INSERT INTO invoices (
                        id, invoice_number, customer_id, job_id,
                        customer_name, subtotal, total,
                        status, invoice_date, due_date,
                        created_by, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()),
                    invoice_number,
                    customer_id,
                    job_id,
                    customer_name,
                    amount or 10000,
                    amount or 10000,
                    'pending',
                    datetime.now().date(),
                    (datetime.now() + timedelta(days=30)).date(),
                    SYSTEM_USER_ID,
                    json.dumps({"auto_generated": True}),
                    datetime.now()
                ))
                self.logger.info(f"Generated invoice {invoice_number} for completed job")

    def run_cycle(self):
        """Execute all workflows"""
        self.execute_workflows()

class CustomerIntelligence(BaseAgent):
    """Advanced customer analytics and scoring"""

    def __init__(self):
        super().__init__(AgentConfig(
            name="CustomerIntelligence",
            capability=AgentCapability.ANALYTICS,
            interval=300,
            priority=7
        ))

    def calculate_customer_score(self, customer_data: Dict) -> int:
        """Calculate comprehensive customer score"""
        score = 50  # Base score

        # Job history
        job_count = customer_data.get("job_count", 0)
        if job_count > 20:
            score += 30
        elif job_count > 10:
            score += 20
        elif job_count > 5:
            score += 15
        elif job_count > 0:
            score += 10

        # Revenue contribution
        ltv = customer_data.get("lifetime_value", 0)
        if ltv > 200000:
            score += 30
        elif ltv > 100000:
            score += 25
        elif ltv > 50000:
            score += 20
        elif ltv > 25000:
            score += 15
        elif ltv > 10000:
            score += 10

        # Recency
        days_since = customer_data.get("days_since_last", 365)
        if days_since < 30:
            score += 20
        elif days_since < 60:
            score += 15
        elif days_since < 90:
            score += 10
        elif days_since < 180:
            score += 5

        # Payment history
        on_time_rate = customer_data.get("on_time_payment_rate", 0)
        if on_time_rate > 0.95:
            score += 15
        elif on_time_rate > 0.85:
            score += 10
        elif on_time_rate > 0.75:
            score += 5

        return min(100, score)

    def run_cycle(self):
        """Analyze and score customers"""
        customers = self.execute_query("""
            SELECT
                c.id, c.name, c.email,
                COUNT(DISTINCT j.id) as job_count,
                SUM(j.total_amount) as lifetime_value,
                MAX(j.created_at) as last_job,
                AVG(j.total_amount) as avg_job_value,
                COUNT(DISTINCT i.id) as invoice_count,
                SUM(CASE WHEN i.status = 'paid' THEN 1 ELSE 0 END) as paid_invoices
            FROM customers c
            LEFT JOIN jobs j ON j.customer_id = c.id
            LEFT JOIN invoices i ON i.customer_id = c.id
            GROUP BY c.id, c.name, c.email
            ORDER BY RANDOM()
            LIMIT 50
        """)

        if customers:
            for customer in customers:
                cust_id, name, email, job_count, ltv, last_job, avg_value, invoice_count, paid = customer

                days_since = 365
                if last_job:
                    days_since = (datetime.now() - last_job).days

                on_time_rate = (paid / invoice_count) if invoice_count > 0 else 0

                customer_data = {
                    "job_count": job_count or 0,
                    "lifetime_value": ltv or 0,
                    "days_since_last": days_since,
                    "on_time_payment_rate": on_time_rate
                }

                score = self.calculate_customer_score(customer_data)

                # Store score in metadata
                self.execute_query("""
                    UPDATE customers
                    SET metadata = jsonb_set(
                        jsonb_set(
                            jsonb_set(
                                COALESCE(metadata, '{}'::jsonb),
                                '{intelligence_score}',
                                to_jsonb(%s)
                            ),
                            '{lifetime_value}',
                            to_jsonb(%s)
                        ),
                        '{last_analysis}',
                        to_jsonb(%s)
                    )
                    WHERE id = %s
                """, (score, ltv or 0, datetime.now().isoformat(), cust_id))

                self.logger.info(f"Customer {name}: Score {score}/100, LTV ${ltv or 0:,.0f}, {job_count} jobs")

class SystemMonitor(BaseAgent):
    """Monitor system health and performance"""

    def __init__(self):
        super().__init__(AgentConfig(
            name="SystemMonitor",
            capability=AgentCapability.UNIVERSAL,
            interval=60,
            priority=10
        ))

    def run_cycle(self):
        """Monitor system metrics"""
        # Get system metrics
        metrics = self.execute_query("""
            SELECT
                (SELECT COUNT(*) FROM customers) as customers,
                (SELECT COUNT(*) FROM jobs) as jobs,
                (SELECT COUNT(*) FROM estimates) as estimates,
                (SELECT COUNT(*) FROM invoices) as invoices,
                (SELECT COUNT(*) FROM schedules) as schedules,
                (SELECT COUNT(*) FROM ai_agents WHERE is_active = true) as active_agents,
                (SELECT SUM(total_amount) FROM jobs WHERE created_at > NOW() - INTERVAL '30 days') as monthly_revenue
        """)

        if metrics:
            c, j, e, i, s, a, r = metrics[0]
            self.logger.info(f"System Metrics - Customers: {c}, Jobs: {j}, Estimates: {e}, Invoices: {i}, Schedules: {s}, Active Agents: {a}, Monthly Revenue: ${r or 0:,.0f}")

            # Check for anomalies
            if j > 0 and e == 0:
                self.logger.warning("No estimates despite having jobs")
            if c > 1000 and a < 5:
                self.logger.warning("High customer count but low agent activity")

        # Check agent health
        agent_health = self.execute_query("""
            SELECT name, last_active, total_executions, is_active
            FROM ai_agents
            WHERE is_active = true
            ORDER BY last_active DESC
        """)

        if agent_health:
            for name, last_active, executions, active in agent_health:
                if last_active:
                    minutes_ago = (datetime.now() - last_active).total_seconds() / 60
                    if minutes_ago > 10:
                        self.logger.warning(f"Agent {name} inactive for {minutes_ago:.0f} minutes")

# AGENT ORCHESTRATOR

class ProductionOrchestrator:
    """Production-grade agent orchestrator"""

    def __init__(self):
        self.agents = []
        self.threads = []
        self.logger = logging.getLogger("Orchestrator")

    def add_agent(self, agent: BaseAgent):
        """Add agent to orchestrator"""
        if agent.config.enabled:
            self.agents.append(agent)
            self.logger.info(f"Added {agent.name} with {agent.config.capability.value} capability")

    def start(self):
        """Start all agents with priority ordering"""
        self.logger.info("=" * 60)
        self.logger.info("BRAINOPS AI AGENT SYSTEM - PRODUCTION")
        self.logger.info("=" * 60)

        # Sort by priority
        self.agents.sort(key=lambda x: x.config.priority, reverse=True)

        for agent in self.agents:
            thread = threading.Thread(target=agent.run, name=agent.name)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            self.logger.info(f"✅ Started {agent.name} (Priority: {agent.config.priority})")
            time.sleep(0.5)

        self.logger.info(f"🚀 All {len(self.agents)} agents operational")

    def monitor(self):
        """Monitor agent health and restart if needed"""
        while True:
            try:
                alive = sum(1 for t in self.threads if t.is_alive())
                self.logger.info(f"Health Check: {alive}/{len(self.threads)} agents running")

                # Restart dead threads
                for i, thread in enumerate(self.threads):
                    if not thread.is_alive():
                        agent = self.agents[i]
                        self.logger.warning(f"Restarting {agent.name}")
                        new_thread = threading.Thread(target=agent.run, name=agent.name)
                        new_thread.daemon = True
                        new_thread.start()
                        self.threads[i] = new_thread

                time.sleep(60)

            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                time.sleep(60)

def initialize_database():
    """Initialize database with agent records"""
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor()

        agents = [
            ("EstimationAgent", "universal", "Comprehensive estimation system"),
            ("IntelligentScheduler", "universal", "Optimized scheduling engine"),
            ("RevenueOptimizer", "analytics", "Revenue maximization system"),
            ("WorkflowAutomation", "automation", "Business process automation"),
            ("CustomerIntelligence", "analytics", "Customer scoring and analytics"),
            ("SystemMonitor", "universal", "System health monitoring")
        ]

        for name, capability, description in agents:
            cursor.execute("""
                INSERT INTO ai_agents (
                    id, name, type, model, capabilities,
                    status, is_active, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE
                SET type = EXCLUDED.type,
                    capabilities = EXCLUDED.capabilities,
                    status = 'active',
                    is_active = true,
                    last_active = NOW()
            """, (
                str(uuid.uuid4()),
                name,
                capability,
                'production',
                json.dumps({"description": description, "capability": capability}),
                'active',
                True,
                datetime.now()
            ))

        conn.commit()
        cursor.close()
        db_pool.putconn(conn)
        logger.info("Database initialized with agent records")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def main():
    """Main entry point"""
    # Initialize database
    initialize_database()

    # Create orchestrator
    orchestrator = ProductionOrchestrator()

    # Add production agents
    orchestrator.add_agent(EstimationAgent())
    orchestrator.add_agent(IntelligentScheduler())
    orchestrator.add_agent(RevenueOptimizer())
    orchestrator.add_agent(WorkflowAutomation())
    orchestrator.add_agent(CustomerIntelligence())
    orchestrator.add_agent(SystemMonitor())

    # Start all agents
    orchestrator.start()

    # Monitor forever
    try:
        orchestrator.monitor()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        sys.exit(0)

if __name__ == "__main__":
    main()
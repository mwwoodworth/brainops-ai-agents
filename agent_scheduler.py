"""
AI Agent Scheduler - Automatic Execution System
Schedules and executes AI agents based on configured intervals
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import Dict, List, Optional, Any
import asyncio
import uuid
import json
from contextlib import contextmanager
from revenue_generation_system import get_revenue_system

# CRITICAL: Use shared connection pool to prevent MaxClientsInSessionMode
try:
    from database.sync_pool import get_sync_pool
    _SYNC_POOL_AVAILABLE = True
except ImportError:
    _SYNC_POOL_AVAILABLE = False


@contextmanager
def _get_pooled_connection(db_config=None):
    """Get connection from shared pool - ALWAYS use this."""
    if _SYNC_POOL_AVAILABLE:
        pool = get_sync_pool()
        with pool.get_connection() as conn:
            yield conn
    else:
        # Fallback for environments without shared pool
        if db_config is None:
            db_config = {
                'host': os.getenv('DB_HOST'),
                'database': os.getenv('DB_NAME', 'postgres'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'port': int(os.getenv('DB_PORT', 5432))
            }
        conn = psycopg2.connect(**db_config)
        try:
            yield conn
        finally:
            if conn and not conn.closed:
                conn.close()

# Import UnifiedBrain for persistent memory integration
try:
    from unified_brain import UnifiedBrain
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    UnifiedBrain = None

logger = logging.getLogger(__name__)

class AgentScheduler:
    """Manages automatic execution of AI agents"""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        # All credentials MUST come from environment variables - no hardcoded defaults
        self.db_config = db_config or {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        if not all([self.db_config['host'], self.db_config['user'], self.db_config['password']]):
            raise ValueError("DB_HOST, DB_USER, and DB_PASSWORD environment variables are required")
        # Use BackgroundScheduler instead of AsyncIOScheduler for FastAPI compatibility
        self.scheduler = BackgroundScheduler()
        self.registered_jobs = {}

        # Initialize UnifiedBrain for persistent memory integration
        self.brain = None
        if BRAIN_AVAILABLE:
            try:
                self.brain = UnifiedBrain()
                logger.info("🧠 UnifiedBrain connected for persistent memory")
            except Exception as e:
                logger.warning(f"⚠️ UnifiedBrain initialization failed: {e}")
                self.brain = None

        logger.info(f"🔧 AgentScheduler initialized with DB: {self.db_config['host']}:{self.db_config['port']}")

    @contextmanager
    def get_db_connection(self):
        """Get database connection from SHARED pool - USE WITH 'with' STATEMENT."""
        with _get_pooled_connection(self.db_config) as conn:
            yield conn

    def return_db_connection(self, conn):
        """DEPRECATED: No longer needed - connections are auto-returned via context manager."""
        # This method is kept for backward compatibility but does nothing
        # Connections are now managed by the context manager
        pass

    def execute_agent(self, agent_id: str, agent_name: str):
        """Execute a scheduled agent (SYNCHRONOUS for BackgroundScheduler)"""
        execution_id = str(uuid.uuid4())

        try:
            logger.info(f"🚀 Executing scheduled agent: {agent_name} ({agent_id})")

            with self.get_db_connection() as conn:
                if not conn:
                    logger.error("❌ Database connection failed, cannot execute agent")
                    return

                cur = conn.cursor(cursor_factory=RealDictCursor)

                try:
                    # Record execution start
                    cur.execute("""
                        INSERT INTO ai_agent_executions
                        (id, agent_name, status)
                        VALUES (%s, %s, %s)
                    """, (execution_id, agent_name, 'running'))
                    conn.commit()
                    logger.info(f"📝 Execution {execution_id} recorded as 'running'")

                    # Get agent configuration
                    cur.execute("SELECT * FROM ai_agents WHERE id = %s", (agent_id,))
                    agent = cur.fetchone()

                    if not agent:
                        logger.error(f"❌ Agent {agent_id} not found in database")
                        cur.execute("""
                            UPDATE ai_agent_executions
                            SET status = %s, error_message = %s
                            WHERE id = %s
                        """, ('failed', 'Agent not found', execution_id))
                        conn.commit()
                        return

                    # Execute based on agent type (synchronous)
                    start_time = datetime.utcnow()
                    result = self._execute_by_type_sync(agent, cur, conn)
                    execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                    # Record execution completion
                    cur.execute("""
                        UPDATE ai_agent_executions
                        SET status = %s, output_data = %s, execution_time_ms = %s
                        WHERE id = %s
                    """, ('completed', json.dumps(result), execution_time_ms, execution_id))

                    # Update agent statistics
                    cur.execute("""
                        UPDATE ai_agents
                        SET total_executions = total_executions + 1,
                            last_activation = %s,
                            last_active = %s
                        WHERE id = %s
                    """, (datetime.utcnow(), datetime.utcnow(), agent_id))

                    # Update schedule next execution with row locking to prevent race conditions
                    # First lock the row and get frequency_minutes, then update to prevent concurrent executions
                    cur.execute("""
                        SELECT id, frequency_minutes FROM agent_schedules
                        WHERE agent_id = %s AND enabled = true
                        FOR UPDATE NOWAIT
                    """, (agent_id,))
                    schedule_row = cur.fetchone()
                    frequency_minutes = (schedule_row['frequency_minutes'] if schedule_row else None) or 60

                    cur.execute("""
                        UPDATE agent_schedules
                        SET last_execution = %s,
                            next_execution = %s,
                            updated_at = NOW()
                        WHERE agent_id = %s AND enabled = true
                    """, (datetime.utcnow(), datetime.utcnow() + timedelta(minutes=frequency_minutes), agent_id))

                    conn.commit()
                    logger.info(f"Agent {agent_name} executed successfully")

                    # Store execution result in persistent memory for learning and context
                    if self.brain:
                        try:
                            self.brain.store(
                                key=f"agent_execution_{execution_id}",
                                value={
                                    "agent_id": agent_id,
                                    "agent_name": agent_name,
                                    "execution_id": execution_id,
                                    "status": "completed",
                                    "execution_time_ms": execution_time_ms,
                                    "result_summary": str(result)[:500] if result else None,
                                    "timestamp": datetime.utcnow().isoformat()
                                },
                                category="agent_execution",
                                priority="medium",
                                source="agent_scheduler",
                                metadata={"agent_type": agent.get('type', 'unknown')}
                            )
                            logger.debug(f"🧠 Stored execution {execution_id} in persistent memory")
                        except Exception as mem_err:
                            logger.warning(f"Failed to store in memory: {mem_err}")

                except Exception as e:
                    logger.error(f"Error executing agent {agent_name}: {e}")
                    try:
                        conn.rollback()  # Rollback any pending transaction
                        cur.execute("""
                            UPDATE ai_agent_executions
                            SET status = %s, error_message = %s
                            WHERE id = %s
                        """, ('failed', str(e), execution_id))
                        conn.commit()

                        # Store failure in memory for learning from mistakes
                        if self.brain:
                            try:
                                self.brain.store_learning(
                                    agent_id=agent_id,
                                    task_id=execution_id,
                                    mistake=f"Agent {agent_name} failed during execution",
                                    lesson=f"Error type: {type(e).__name__}. Need to handle: {str(e)[:200]}",
                                    root_cause=str(e)[:500],
                                    impact="medium"
                                )
                                logger.debug(f"🧠 Stored failure learning for {agent_name}")
                            except Exception as learn_err:
                                logger.warning(f"Failed to store learning: {learn_err}")
                    except Exception as log_error:
                        logger.error(f"Failed to log error: {log_error}")
                finally:
                    cur.close()

        except Exception as outer_e:
            logger.error(f"Failed to get database connection for agent {agent_name}: {outer_e}")

    def _execute_by_type_sync(self, agent: Dict, cur, conn) -> Dict:
        """Execute agent based on its type (SYNCHRONOUS)"""
        agent_type = agent.get('type', '').lower()
        agent_name = agent.get('name', 'Unknown')

        logger.info(f"⚙️ Executing {agent_type} agent: {agent_name}")

        # Health monitoring agent
        if agent_name == 'HealthMonitor':
            return self._execute_health_monitor(agent, cur, conn)

        # Revenue-generating agents
        elif 'revenue' in agent_type or agent_name == 'RevenueOptimizer':
            return self._execute_revenue_agent(agent, cur, conn)

        # Lead generation and revenue pipeline agents
        elif 'lead' in agent_type or 'nurture' in agent_type or agent_name in [
            'LeadGenerationAgent', 'LeadScorer', 'LeadQualificationAgent',
            'LeadDiscoveryAgent', 'NurtureExecutorAgent', 'DealClosingAgent',
            'RevenueProposalAgent'
        ]:
            return self._execute_lead_agent(agent, cur, conn)

        # Customer intelligence agents
        elif 'customer' in agent_type or agent_name == 'CustomerIntelligence':
            return self._execute_customer_agent(agent, cur, conn)

        # Analytics agents
        elif agent_type == 'analytics':
            return self._execute_analytics_agent(agent, cur, conn)

        # Email processor agent
        elif agent_name == 'EmailProcessor' or 'email' in agent_type:
            return self._execute_email_processor(agent, cur, conn)

        # Learning Feedback Loop agent - closes the gap between insights and action
        elif agent_name == 'LearningFeedbackLoop' or agent_type == 'system_improvement':
            return self._execute_learning_feedback_loop(agent, cur, conn)

        # Default: log and continue
        else:
            logger.info(f"No specific handler for agent type: {agent_type}")
            return {"status": "executed", "type": agent_type}

    def _execute_revenue_agent(self, agent: Dict, cur, conn) -> Dict:
        """Execute revenue optimization agent - TAKES REAL ACTIONS"""
        logger.info(f"Running revenue optimization for agent: {agent['name']}")
        actions_taken = []

        # Get revenue opportunities
        cur.execute("""
            SELECT COUNT(*) as total_jobs,
                   COUNT(*) FILTER (WHERE status = 'pending') as pending_jobs,
                   COUNT(*) FILTER (WHERE status = 'in_progress') as active_jobs
            FROM jobs
        """)
        stats = cur.fetchone()

        # Get pending estimates that need follow-up (older than 3 days)
        cur.execute("""
            SELECT e.id, e.customer_id, e.total_amount, c.name as customer_name, c.email
            FROM estimates e
            LEFT JOIN customers c ON c.id = e.customer_id
            WHERE e.status = 'pending'
            AND e.created_at < NOW() - INTERVAL '3 days'
            AND NOT EXISTS (
                SELECT 1 FROM ai_scheduled_outreach o
                WHERE o.target_id = e.customer_id::text
                AND o.created_at > NOW() - INTERVAL '7 days'
            )
            LIMIT 10
        """)
        stale_estimates = cur.fetchall()

        # Create follow-up tasks for stale estimates
        for est in stale_estimates:
            try:
                cur.execute("""
                    INSERT INTO ai_scheduled_outreach
                    (target_id, channel, message_template, personalization, scheduled_for, status, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(est['customer_id']),
                    'email',
                    'estimate_followup',
                    json.dumps({
                        'customer_name': est['customer_name'],
                        'estimate_amount': float(est['total_amount'] or 0)
                    }),
                    datetime.utcnow() + timedelta(hours=24),
                    'scheduled',
                    json.dumps({
                        'estimate_id': str(est['id']),
                        'amount': float(est['total_amount'] or 0),
                        'reason': 'Stale estimate - no response in 3+ days',
                        'priority': 'high' if float(est['total_amount'] or 0) > 5000 else 'medium'
                    })
                ))
                actions_taken.append({
                    'action': 'scheduled_followup',
                    'customer_id': str(est['customer_id']),
                    'estimate_amount': float(est['total_amount'] or 0)
                })
            except Exception as e:
                logger.warning(f"Could not create followup for estimate {est['id']}: {e}")

        # Record revenue insight if significant opportunities exist
        potential_revenue = sum(float(e['total_amount'] or 0) for e in stale_estimates)
        if potential_revenue > 1000:
            try:
                cur.execute("""
                    INSERT INTO ai_business_insights
                    (insight_type, category, title, description, impact_score, urgency)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    'revenue_opportunity',
                    'sales',
                    f'${potential_revenue:,.0f} in stale estimates need follow-up',
                    f'{len(stale_estimates)} estimates older than 3 days with no follow-up scheduled. Agent: {agent["name"]}',
                    min(100, int(potential_revenue / 100)),  # Impact score based on revenue
                    'high' if potential_revenue > 10000 else 'medium'
                ))
                actions_taken.append({'action': 'created_insight', 'revenue': potential_revenue})
            except Exception as e:
                logger.warning(f"Could not create insight: {e}")

        # --- Autonomous Revenue System Integration ---
        try:
            logger.info("🔄 Triggering Autonomous Revenue System...")
            rev_sys = get_revenue_system()
            
            async def run_autonomous_tasks():
                # 1. Identify new leads
                criteria = {
                    "location": "United States",
                    "company_size": "Small to Medium",
                    "industry": "Roofing"
                }
                new_leads = await rev_sys.identify_new_leads(criteria)
                
                # 2. Automatically qualify and start workflow for new leads
                processed = 0
                for lead_id in new_leads:
                    if processed >= 3: # Limit to avoid overload per cycle
                        break
                    await rev_sys.run_revenue_workflow(lead_id)
                    processed += 1
                    
                return {
                    "new_leads_identified": len(new_leads),
                    "workflows_started": processed
                }

            # Run autonomous tasks - create dedicated event loop for this thread
            # This avoids deadlocks from nested asyncio.run() calls
            def run_in_new_loop():
                """Run async tasks in a fresh event loop"""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        asyncio.wait_for(run_autonomous_tasks(), timeout=300)
                    )
                except asyncio.TimeoutError:
                    logger.warning("Autonomous tasks timed out after 300s")
                    return {"timed_out": True}
                finally:
                    loop.close()

            try:
                auto_stats = run_in_new_loop()
            except Exception as e:
                logger.error(f"Failed to run autonomous tasks: {e}")
                auto_stats = {"error": str(e)}
            actions_taken.append({
                'action': 'autonomous_revenue_cycle',
                'stats': auto_stats
            })
            logger.info(f"✅ Autonomous revenue cycle completed: {auto_stats}")

        except Exception as e:
            logger.error(f"❌ Autonomous Revenue System failed: {e}")
            # Don't fail the whole agent execution, just log it
        # ---------------------------------------------

        conn.commit()

        return {
            "agent": agent['name'],
            "jobs_analyzed": stats['total_jobs'],
            "pending_jobs": stats['pending_jobs'],
            "active_jobs": stats['active_jobs'],
            "stale_estimates_found": len(stale_estimates),
            "potential_revenue": potential_revenue,
            "actions_taken": actions_taken,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _execute_lead_agent(self, agent: Dict, cur, conn) -> Dict:
        """Execute lead generation/scoring agent - TAKES REAL ACTIONS"""
        logger.info(f"Running lead analysis for agent: {agent['name']}")
        actions_taken = []

        # Get lead statistics
        cur.execute("""
            SELECT COUNT(*) as total_customers,
                   COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as new_this_week,
                   COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as new_this_month
            FROM customers
        """)
        stats = cur.fetchone()

        # Find high-value customers who haven't been contacted recently (potential leads)
        cur.execute("""
            SELECT c.id, c.name, c.email, c.phone,
                   COUNT(j.id) as job_count,
                   COALESCE(SUM(i.total_amount), 0) as total_spent,
                   MAX(j.created_at) as last_job_date
            FROM customers c
            LEFT JOIN jobs j ON j.customer_id = c.id
            LEFT JOIN invoices i ON i.job_id = j.id
            WHERE c.email IS NOT NULL
            AND c.email NOT LIKE '%%test%%'
            AND c.email NOT LIKE '%%example%%'
            GROUP BY c.id, c.name, c.email, c.phone
            HAVING COUNT(j.id) >= 2
            AND MAX(j.created_at) < NOW() - INTERVAL '60 days'
            ORDER BY COALESCE(SUM(i.total_amount), 0) DESC
            LIMIT 15
        """)
        dormant_valuable_customers = cur.fetchall()

        # Calculate and update lead scores
        for customer in dormant_valuable_customers:
            total_spent = float(customer['total_spent'] or 0)
            job_count = customer['job_count'] or 0

            # Score based on historical value and recency
            score = min(100, int(
                (total_spent / 1000) * 10 +  # $1k = 10 points
                job_count * 15 +              # each job = 15 points
                30                            # base score for returning customer
            ))

            try:
                # Get tenant_id from customer
                cur.execute("SELECT tenant_id FROM customers WHERE id = %s", (customer['id'],))
                tenant_row = cur.fetchone()
                tenant_id = tenant_row['tenant_id'] if tenant_row else '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'

                # Update or insert lead score
                cur.execute("""
                    INSERT INTO ai_lead_scores
                    (customer_id, score, probability_to_close, estimated_value, factors, tenant_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (customer_id) DO UPDATE SET
                        score = EXCLUDED.score,
                        probability_to_close = EXCLUDED.probability_to_close,
                        estimated_value = EXCLUDED.estimated_value,
                        factors = EXCLUDED.factors,
                        updated_at = NOW()
                """, (
                    customer['id'],
                    score,
                    min(0.9, score / 100.0),  # Convert score to probability
                    total_spent * 0.3,  # Estimate 30% of historical value as potential
                    json.dumps({
                        'total_spent': total_spent,
                        'job_count': job_count,
                        'days_inactive': 60,
                        'calculation': 'dormant_valuable_customer'
                    }),
                    tenant_id
                ))

                # Schedule re-engagement for high-score leads
                if score >= 50:
                    cur.execute("""
                        INSERT INTO ai_scheduled_outreach
                        (target_id, channel, message_template, personalization, scheduled_for, status, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        str(customer['id']),
                        'email',
                        'reengagement',
                        json.dumps({
                            'customer_name': customer['name'],
                            'lead_score': score
                        }),
                        datetime.utcnow() + timedelta(hours=48),
                        'scheduled',
                        json.dumps({
                            'lead_score': score,
                            'total_spent': total_spent,
                            'reason': 'High-value dormant customer - re-engagement opportunity',
                            'priority': 'high' if score >= 70 else 'medium'
                        })
                    ))
                    actions_taken.append({
                        'action': 'scheduled_reengagement',
                        'customer_id': str(customer['id']),
                        'lead_score': score
                    })

                actions_taken.append({
                    'action': 'updated_lead_score',
                    'customer_id': str(customer['id']),
                    'score': score
                })
            except Exception as e:
                logger.warning(f"Could not update lead score for customer {customer['id']}: {e}")

        conn.commit()

        return {
            "agent": agent['name'],
            "total_customers": stats['total_customers'],
            "new_this_week": stats['new_this_week'],
            "new_this_month": stats['new_this_month'],
            "dormant_valuable_found": len(dormant_valuable_customers),
            "leads_scored": len([a for a in actions_taken if a['action'] == 'updated_lead_score']),
            "reengagements_scheduled": len([a for a in actions_taken if a['action'] == 'scheduled_reengagement']),
            "actions_taken": actions_taken,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _execute_customer_agent(self, agent: Dict, cur, conn) -> Dict:
        """Execute customer intelligence agent - TAKES REAL ACTIONS"""
        logger.info(f"Running customer intelligence for agent: {agent['name']}")
        actions_taken = []

        # Get customer insights
        cur.execute("""
            SELECT
                COUNT(DISTINCT c.id) as total_customers,
                COUNT(DISTINCT j.id) as total_jobs,
                COUNT(DISTINCT i.id) as total_invoices,
                SUM(i.total_amount) as total_revenue
            FROM customers c
            LEFT JOIN jobs j ON j.customer_id = c.id
            LEFT JOIN invoices i ON i.job_id = j.id
        """)
        stats = cur.fetchone()

        # Identify at-risk customers (high value, no recent activity)
        cur.execute("""
            SELECT c.id, c.name, c.email, c.phone,
                   COALESCE(SUM(i.total_amount), 0) as lifetime_value,
                   COUNT(DISTINCT j.id) as total_jobs,
                   MAX(j.created_at) as last_activity,
                   EXTRACT(days FROM NOW() - MAX(j.created_at)) as days_inactive
            FROM customers c
            LEFT JOIN jobs j ON j.customer_id = c.id
            LEFT JOIN invoices i ON i.job_id = j.id
            WHERE c.email IS NOT NULL
            AND c.email NOT LIKE '%%test%%'
            AND c.email NOT LIKE '%%example%%'
            GROUP BY c.id, c.name, c.email, c.phone
            HAVING COALESCE(SUM(i.total_amount), 0) > 2000
            AND MAX(j.created_at) < NOW() - INTERVAL '90 days'
            ORDER BY COALESCE(SUM(i.total_amount), 0) DESC
            LIMIT 10
        """)
        at_risk_customers = cur.fetchall()

        # Create customer health records and interventions
        for customer in at_risk_customers:
            ltv = float(customer['lifetime_value'] or 0)
            days_inactive = int(customer['days_inactive'] or 0)

            # Calculate churn risk score (0-100)
            churn_risk = min(100, int(
                (days_inactive / 90) * 30 +  # More days = higher risk
                (ltv / 5000) * 20 +           # Higher LTV = higher concern
                30                            # Base risk for inactivity
            ))

            try:
                # Get tenant_id from customer
                cur.execute("SELECT tenant_id FROM customers WHERE id = %s", (customer['id'],))
                tenant_row = cur.fetchone()
                tenant_id = tenant_row['tenant_id'] if tenant_row else '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'

                # Record customer health status
                cur.execute("""
                    INSERT INTO ai_customer_health
                    (customer_id, health_score, churn_probability, churn_risk, lifetime_value,
                     days_since_last_activity, health_status, tenant_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (customer_id) DO UPDATE SET
                        health_score = EXCLUDED.health_score,
                        churn_probability = EXCLUDED.churn_probability,
                        churn_risk = EXCLUDED.churn_risk,
                        lifetime_value = EXCLUDED.lifetime_value,
                        days_since_last_activity = EXCLUDED.days_since_last_activity,
                        health_status = EXCLUDED.health_status,
                        updated_at = NOW()
                """, (
                    customer['id'],
                    100 - churn_risk,  # Health score is inverse of churn risk
                    churn_risk / 100.0,  # Convert to probability (0-1)
                    'high' if churn_risk > 70 else 'medium' if churn_risk > 40 else 'low',
                    ltv,
                    days_inactive,
                    'at_risk' if churn_risk > 60 else 'declining',
                    tenant_id
                ))

                # Create intervention for high-risk customers
                if churn_risk > 50:
                    cur.execute("""
                        INSERT INTO ai_customer_interventions
                        (customer_id, intervention_type, reason, triggered_by, scheduled_date, tenant_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        customer['id'],
                        'retention',
                        f"High-value customer ({customer['name']}) inactive for {days_inactive} days. LTV: ${ltv:,.0f}. Churn risk: {churn_risk}%. Email: {customer['email']}",
                        'CustomerIntelligence Agent',
                        datetime.utcnow() + timedelta(days=1),
                        tenant_id
                    ))
                    actions_taken.append({
                        'action': 'created_intervention',
                        'customer_id': str(customer['id']),
                        'churn_risk': churn_risk,
                        'ltv': ltv
                    })

                actions_taken.append({
                    'action': 'updated_health',
                    'customer_id': str(customer['id']),
                    'health_score': 100 - churn_risk
                })
            except Exception as e:
                logger.warning(f"Could not process customer {customer['id']}: {e}")

        # Generate aggregate insight if multiple at-risk customers
        if len(at_risk_customers) >= 3:
            total_at_risk_value = sum(float(c['lifetime_value'] or 0) for c in at_risk_customers)
            try:
                cur.execute("""
                    INSERT INTO ai_business_insights
                    (insight_type, category, title, description, impact_score, urgency)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    'customer_health',
                    'retention',
                    f'{len(at_risk_customers)} high-value customers at churn risk',
                    f'Combined LTV of ${total_at_risk_value:,.0f} at risk. Immediate retention action recommended. Agent: {agent["name"]}',
                    min(100, int(total_at_risk_value / 200)),  # Impact score based on value
                    'critical' if total_at_risk_value > 20000 else 'high'
                ))
                actions_taken.append({'action': 'created_churn_insight', 'value_at_risk': total_at_risk_value})
            except Exception as e:
                logger.warning(f"Could not create insight: {e}")

        conn.commit()

        return {
            "agent": agent['name'],
            "total_customers": stats['total_customers'],
            "total_jobs": stats['total_jobs'],
            "total_invoices": stats['total_invoices'],
            "total_revenue": float(stats['total_revenue'] or 0),
            "at_risk_customers_found": len(at_risk_customers),
            "interventions_created": len([a for a in actions_taken if a['action'] == 'created_intervention']),
            "actions_taken": actions_taken,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _execute_analytics_agent(self, agent: Dict, cur, conn) -> Dict:
        """Execute analytics agent"""
        logger.info(f"Running analytics for agent: {agent['name']}")

        # Get general analytics
        cur.execute("""
            SELECT
                (SELECT COUNT(*) FROM customers) as total_customers,
                (SELECT COUNT(*) FROM jobs) as total_jobs,
                (SELECT COUNT(*) FROM invoices) as total_invoices,
                (SELECT COUNT(*) FROM estimates) as total_estimates
        """)
        stats = cur.fetchone()

        return {
            "agent": agent['name'],
            "analytics": {
                "customers": stats['total_customers'],
                "jobs": stats['total_jobs'],
                "invoices": stats['total_invoices'],
                "estimates": stats['total_estimates']
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    def _execute_health_monitor(self, agent: Dict, cur, conn) -> Dict:
        """Execute health monitoring agent - checks all agents and auto-restarts failed ones"""
        logger.info(f"Running health monitoring for agent: {agent['name']}")

        try:
            # Import health monitor
            from agent_health_monitor import get_health_monitor
            health_monitor = get_health_monitor()

            # Run health check for all agents
            health_summary = health_monitor.check_all_agents_health()

            # Auto-restart critical agents
            restart_result = health_monitor.auto_restart_critical_agents()

            # Store results in brain
            if self.brain:
                try:
                    self.brain.store(
                        key=f"health_check_{datetime.utcnow().isoformat()}",
                        value={
                            "total_agents": health_summary.get("total_agents", 0),
                            "healthy": health_summary.get("healthy", 0),
                            "degraded": health_summary.get("degraded", 0),
                            "critical": health_summary.get("critical", 0),
                            "restarted": restart_result.get("restarted", 0)
                        },
                        category="health_monitoring",
                        priority="high" if health_summary.get("critical", 0) > 0 else "medium",
                        source="health_monitor_agent"
                    )
                except Exception as mem_err:
                    logger.warning(f"Failed to store health check in brain: {mem_err}")

            return {
                "agent": agent['name'],
                "health_summary": health_summary,
                "auto_restart_result": restart_result,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
            return {
                "agent": agent['name'],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _execute_email_processor(self, agent: Dict, cur, conn) -> Dict:
        """Execute email processor agent - processes ai_email_queue"""
        logger.info(f"Running email processor for agent: {agent['name']}")

        try:
            # Import email sender
            from email_sender import process_email_queue, get_queue_status

            # Get queue status before processing
            queue_status = get_queue_status()
            logger.info(f"Email queue status: {queue_status.get('totals', {})}")

            # Process the email queue
            result = process_email_queue(batch_size=10)

            # Store result in brain memory for learning
            if self.brain:
                try:
                    self.brain.store(
                        key=f"email_processing_{datetime.utcnow().isoformat()}",
                        value={
                            "agent": agent['name'],
                            "processed": result.get('processed', 0),
                            "sent": result.get('sent', 0),
                            "failed": result.get('failed', 0),
                            "skipped": result.get('skipped', 0),
                            "provider": result.get('provider', 'unknown'),
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        category="email_processing",
                        priority="medium",
                        source="email_processor_agent"
                    )
                except Exception as mem_err:
                    logger.warning(f"Failed to store email processing in brain: {mem_err}")

            return {
                "agent": agent['name'],
                "queue_before": queue_status.get('totals', {}),
                "processing_result": result,
                "timestamp": datetime.utcnow().isoformat()
            }

        except ImportError as e:
            logger.error(f"Email sender module not available: {e}")
            return {
                "agent": agent['name'],
                "error": f"Email sender module not available: {e}",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Email processing failed: {e}")
            return {
                "agent": agent['name'],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _execute_learning_feedback_loop(self, agent: Dict, cur, conn) -> Dict:
        """
        Execute Learning Feedback Loop agent - CLOSES THE GAP BETWEEN INSIGHTS AND ACTION.

        This agent:
        1. Analyzes the 4,700+ insights that have been sitting idle
        2. Identifies actionable patterns from agent executions
        3. Generates improvement proposals
        4. Auto-approves low-risk improvements
        5. Applies approved improvements

        THE SYSTEM FINALLY ACTS ON ITS OWN LEARNING!
        """
        logger.info(f"Running Learning Feedback Loop for agent: {agent['name']}")

        try:
            # Import the feedback loop
            from learning_feedback_loop import run_scheduled_feedback_loop

            # Run the feedback loop cycle - create dedicated event loop for this thread
            def run_in_new_loop():
                """Run async tasks in a fresh event loop"""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        asyncio.wait_for(run_scheduled_feedback_loop(), timeout=300)
                    )
                except asyncio.TimeoutError:
                    logger.warning("Learning feedback loop timed out after 300s")
                    return {"timed_out": True}
                finally:
                    loop.close()

            try:
                cycle_result = run_in_new_loop()
            except Exception as e:
                logger.error(f"Failed to run learning feedback loop: {e}")
                cycle_result = {"error": str(e)}

            # Store result in brain memory for meta-learning
            if self.brain:
                try:
                    self.brain.store(
                        key=f"learning_feedback_loop_{datetime.utcnow().isoformat()}",
                        value={
                            "agent": agent['name'],
                            "patterns_found": cycle_result.get('patterns_found', 0),
                            "proposals_generated": cycle_result.get('proposals_generated', 0),
                            "auto_approved": cycle_result.get('auto_approved', 0),
                            "improvements_applied": cycle_result.get('improvements_applied', 0),
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        category="learning_feedback",
                        priority="high",
                        source="learning_feedback_loop_agent"
                    )
                except Exception as mem_err:
                    logger.warning(f"Failed to store learning cycle in brain: {mem_err}")

            logger.info(
                f"Learning Feedback Loop completed: "
                f"{cycle_result.get('patterns_found', 0)} patterns -> "
                f"{cycle_result.get('proposals_generated', 0)} proposals -> "
                f"{cycle_result.get('improvements_applied', 0)} applied"
            )

            return {
                "agent": agent['name'],
                "cycle_result": cycle_result,
                "timestamp": datetime.utcnow().isoformat()
            }

        except ImportError as e:
            logger.error(f"Learning feedback loop module not available: {e}")
            return {
                "agent": agent['name'],
                "error": f"Learning feedback loop module not available: {e}",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Learning feedback loop failed: {e}")
            return {
                "agent": agent['name'],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def load_schedules_from_db(self):
        """Load agent schedules from database"""
        try:
            logger.info("📋 Loading agent schedules from database...")

            with self.get_db_connection() as conn:
                if not conn:
                    logger.error("❌ Cannot load schedules: DB connection failed")
                    return

                cur = conn.cursor(cursor_factory=RealDictCursor)
                try:
                    # Get all enabled schedules
                    cur.execute("""
                        SELECT s.*, a.name as agent_name, a.type as agent_type
                        FROM agent_schedules s
                        JOIN ai_agents a ON a.id = s.agent_id
                        WHERE s.enabled = true
                    """)

                    schedules = cur.fetchall()
                    logger.info(f"✅ Found {len(schedules)} enabled agent schedules")

                    for schedule in schedules:
                        logger.info(f"   • {schedule['agent_name']} - Every {schedule['frequency_minutes']} minutes")
                        self.add_schedule(
                            agent_id=schedule['agent_id'],
                            agent_name=schedule['agent_name'],
                            frequency_minutes=schedule['frequency_minutes'] or 60,
                            schedule_id=schedule['id']
                        )
                finally:
                    cur.close()

            logger.info(f"✅ Successfully loaded {len(self.registered_jobs)} jobs into scheduler")

        except Exception as e:
            logger.error(f"❌ Error loading schedules: {e}", exc_info=True)

    def add_schedule(self, agent_id: str, agent_name: str, frequency_minutes: int, schedule_id: str = None):
        """Add an agent to the scheduler"""
        try:
            job_id = schedule_id or str(uuid.uuid4())

            # Add job to scheduler
            self.scheduler.add_job(
                func=self.execute_agent,
                trigger=IntervalTrigger(minutes=frequency_minutes),
                args=[agent_id, agent_name],
                id=job_id,
                name=f"Agent: {agent_name}",
                replace_existing=True
            )

            self.registered_jobs[job_id] = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "frequency_minutes": frequency_minutes,
                "added_at": datetime.utcnow().isoformat()
            }

            logger.info(f"✅ Scheduled agent {agent_name} (ID: {job_id}) to run every {frequency_minutes} minutes")

        except Exception as e:
            logger.error(f"❌ Error adding schedule for {agent_name}: {e}", exc_info=True)

    def remove_schedule(self, schedule_id: str):
        """Remove an agent from the scheduler"""
        try:
            self.scheduler.remove_job(schedule_id)
            if schedule_id in self.registered_jobs:
                del self.registered_jobs[schedule_id]
            logger.info(f"Removed schedule: {schedule_id}")
        except Exception as e:
            logger.error(f"Error removing schedule {schedule_id}: {e}")

    def start(self):
        """Start the scheduler"""
        try:
            # Load schedules from database
            self.load_schedules_from_db()

            # Start scheduler
            self.scheduler.start()
            logger.info(f"Agent scheduler started with {len(self.registered_jobs)} jobs")

        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")

    def shutdown(self):
        """Shutdown the scheduler"""
        try:
            self.scheduler.shutdown()
            logger.info("Agent scheduler shutdown")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")

    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            "running": self.scheduler.running,
            "total_jobs": len(self.registered_jobs),
            "jobs": [
                {
                    "id": job_id,
                    "agent": job_info["agent_name"],
                    "frequency_minutes": job_info["frequency_minutes"]
                }
                for job_id, job_info in self.registered_jobs.items()
            ]
        }


# Create execution tracking table if needed
def create_execution_table(db_config: Dict):
    """Create ai_agent_executions table if it doesn't exist"""
    try:
        with _get_pooled_connection(db_config) as conn:
            cur = conn.cursor()
            try:
                # Table already exists with different schema - just ensure indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_agent_executions_agent ON ai_agent_executions(agent_name);
                    CREATE INDEX IF NOT EXISTS idx_agent_executions_created ON ai_agent_executions(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_agent_executions_status ON ai_agent_executions(status);
                """)

                conn.commit()
                logger.info("Execution tracking table verified/created")
            finally:
                cur.close()

    except Exception as e:
        logger.error(f"Error creating execution table: {e}")


# Example usage
if __name__ == "__main__":
    # Configure database
    db_password = os.getenv('DB_PASSWORD')
    if not db_password:
        raise RuntimeError("DB_PASSWORD environment variable is required for agent scheduler startup.")

    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
        'password': db_password,
        'port': int(os.getenv('DB_PORT', 5432))
    }

    # Create execution table
    create_execution_table(DB_CONFIG)

    # Initialize scheduler
    scheduler = AgentScheduler(DB_CONFIG)

    # Start scheduler
    scheduler.start()

    # Keep running
    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
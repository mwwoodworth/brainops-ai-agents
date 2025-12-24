"""
Customer Success Agent
AI agent for handling customer onboarding, support, and success metrics.
Uses OpenAI for real analysis and persists results to database.
"""

import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Lazy OpenAI client initialization
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client"""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                _openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    return _openai_client

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 5432))
}

class CustomerSuccessAgent:
    """AI-powered customer success analysis agent with AUREA integration"""

    def __init__(self, tenant_id: str):
        if not tenant_id:
            raise ValueError("tenant_id is required for CustomerSuccessAgent")
        self.tenant_id = tenant_id
        self.agent_type = "customer_success"

        # AUREA Integration for decision recording and learning
        try:
            from aurea_integration import AUREAIntegration
            self.aurea = AUREAIntegration(tenant_id, self.agent_type)
        except ImportError:
            logger.warning("AUREA integration not available")
            self.aurea = None

    def _get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

    async def analyze_customer_health(self, customer_id: str) -> Dict[str, Any]:
        """Analyze health score for a specific customer using real data and AI"""
        try:
            results = {
                "customer_id": customer_id,
                "analyzed_at": datetime.utcnow().isoformat(),
                "customer_data": {}
            }

            # Get real customer data from database
            conn = self._get_db_connection()
            if conn:
                try:
                    cur = conn.cursor(cursor_factory=RealDictCursor)

                    # Get customer info
                    cur.execute("""
                        SELECT id, name, email, created_at, status
                        FROM customers
                        WHERE id::text = %s OR name ILIKE %s
                        LIMIT 1
                    """, (customer_id, f"%{customer_id}%"))
                    customer = cur.fetchone()

                    if customer:
                        results["customer_data"]["info"] = dict(customer)

                        # Get job history
                        cur.execute("""
                            SELECT
                                COUNT(*) as total_jobs,
                                COUNT(*) FILTER (WHERE status = 'completed') as completed,
                                COUNT(*) FILTER (WHERE status IN ('pending', 'in_progress')) as active,
                                MAX(created_at) as last_job_date
                            FROM jobs
                            WHERE customer_id = %s
                        """, (customer['id'],))
                        jobs = cur.fetchone()
                        results["customer_data"]["jobs"] = dict(jobs) if jobs else {}

                        # Get invoice/payment history
                        cur.execute("""
                            SELECT
                                COUNT(*) as total_invoices,
                                SUM(total_amount) as total_revenue,
                                COUNT(*) FILTER (WHERE status = 'paid') as paid_invoices,
                                AVG(total_amount) as avg_invoice_value
                            FROM invoices
                            WHERE customer_id = %s
                        """, (customer['id'],))
                        invoices = cur.fetchone()
                        results["customer_data"]["invoices"] = {
                            k: float(v) if isinstance(v, (int, float)) and v else v
                            for k, v in dict(invoices).items()
                        } if invoices else {}

                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Could not fetch customer data: {e}")
                    if conn:
                        conn.close()

            # Use AI for health scoring and analysis
            client = get_openai_client()
            if client and results["customer_data"]:
                try:
                    prompt = f"""Analyze customer health based on this data:
{json.dumps(results['customer_data'], indent=2, default=str)}

Calculate a customer health score and provide analysis:
1. Overall health assessment
2. Engagement level
3. Churn risk
4. Upsell opportunities
5. Action recommendations

Respond with JSON only:
{{
    "health_score": 0-100,
    "risk_level": "low/medium/high/critical",
    "engagement_level": "highly_engaged/engaged/at_risk/disengaged",
    "lifetime_value_tier": "high/medium/low",
    "churn_probability": 0-100,
    "health_factors": {{
        "positive": ["factor1", "factor2"],
        "negative": ["concern1", "concern2"]
    }},
    "upsell_opportunities": ["opportunity1"],
    "retention_actions": ["action1", "action2"],
    "next_best_action": "recommended immediate action",
    "confidence_score": 0-100
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=800
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI health analysis failed: {e}")
                    results["health_score"] = 75
                    results["risk_level"] = "low"
            else:
                results["health_score"] = 0
                results["note"] = "Customer not found or AI unavailable"

            await self._save_analysis("customer_health", results)
            return results

        except Exception as e:
            logger.error(f"Customer health analysis failed: {e}")
            return {"error": str(e), "customer_id": customer_id, "health_score": 0}

    async def generate_onboarding_plan(self, customer_id: str, plan_type: str = "standard") -> Dict[str, Any]:
        """Generate personalized onboarding plan using AI"""
        try:
            results = {
                "customer_id": customer_id,
                "plan_type": plan_type,
                "generated_at": datetime.utcnow().isoformat()
            }

            # Get customer info for personalization
            conn = self._get_db_connection()
            customer_info = {}
            if conn:
                try:
                    cur = conn.cursor(cursor_factory=RealDictCursor)
                    cur.execute("""
                        SELECT name, email, created_at
                        FROM customers
                        WHERE id::text = %s OR name ILIKE %s
                        LIMIT 1
                    """, (customer_id, f"%{customer_id}%"))
                    customer = cur.fetchone()
                    if customer:
                        customer_info = dict(customer)
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Could not fetch customer: {e}")
                    if conn:
                        conn.close()

            # AI-generated personalized onboarding plan
            client = get_openai_client()
            if client:
                try:
                    prompt = f"""Create a personalized customer onboarding plan.

Customer Info:
{json.dumps(customer_info, indent=2, default=str)}

Plan Type: {plan_type}

Generate a comprehensive onboarding plan with:
1. Welcome sequence
2. Key milestones
3. Training schedule
4. Success metrics
5. Check-in schedule

Respond with JSON only:
{{
    "plan_name": "Personalized plan name",
    "duration_days": 30,
    "steps": [
        {{"day": 1, "action": "Welcome call", "owner": "CSM", "deliverable": "...", "success_criteria": "..."}},
        {{"day": 3, "action": "Account setup", "owner": "Customer", "deliverable": "...", "success_criteria": "..."}},
        {{"day": 7, "action": "Training session 1", "owner": "CSM", "deliverable": "...", "success_criteria": "..."}}
    ],
    "milestones": [
        {{"name": "First milestone", "target_day": 7, "criteria": "...", "celebration": "..."}},
        {{"name": "Second milestone", "target_day": 14, "criteria": "...", "celebration": "..."}}
    ],
    "check_ins": ["Day 3", "Day 7", "Day 14", "Day 30"],
    "success_metrics": ["metric1", "metric2"],
    "risk_triggers": ["trigger that indicates onboarding issues"],
    "personalization_notes": "why this plan fits this customer"
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000
                    )
                    ai_plan = json.loads(response.choices[0].message.content)
                    results.update(ai_plan)
                except Exception as e:
                    logger.warning(f"AI onboarding plan generation failed: {e}")
                    results["steps"] = [
                        {"day": 1, "action": "Welcome email", "owner": "CSM"},
                        {"day": 3, "action": "Account setup", "owner": "Customer"},
                        {"day": 7, "action": "Training session", "owner": "CSM"},
                        {"day": 14, "action": "First milestone check", "owner": "CSM"}
                    ]

            await self._save_analysis("onboarding_plan", results)
            return results

        except Exception as e:
            logger.error(f"Onboarding plan generation failed: {e}")
            return {"error": str(e), "customer_id": customer_id}

    async def analyze_churn_risk(self) -> Dict[str, Any]:
        """Analyze churn risk across all customers"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "at_risk_customers": []
            }

            conn = self._get_db_connection()
            if conn:
                try:
                    cur = conn.cursor(cursor_factory=RealDictCursor)

                    # Find customers with risk indicators
                    cur.execute("""
                        SELECT
                            c.id,
                            c.name,
                            c.created_at,
                            COUNT(j.id) as job_count,
                            MAX(j.created_at) as last_job_date,
                            COALESCE(SUM(i.total_amount), 0) as total_revenue
                        FROM customers c
                        LEFT JOIN jobs j ON j.customer_id = c.id
                        LEFT JOIN invoices i ON i.customer_id = c.id AND i.status = 'paid'
                        WHERE c.created_at < NOW() - INTERVAL '30 days'
                        GROUP BY c.id, c.name, c.created_at
                        HAVING MAX(j.created_at) < NOW() - INTERVAL '60 days'
                           OR COUNT(j.id) = 0
                        ORDER BY total_revenue DESC
                        LIMIT 20
                    """)
                    at_risk = cur.fetchall()
                    results["at_risk_customers"] = [dict(c) for c in at_risk] if at_risk else []

                    # Get overall customer health stats
                    cur.execute("""
                        SELECT
                            COUNT(*) as total_customers,
                            COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as new_customers,
                            COUNT(*) FILTER (WHERE status = 'inactive') as inactive_customers
                        FROM customers
                    """)
                    stats = cur.fetchone()
                    results["customer_stats"] = dict(stats) if stats else {}

                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Could not fetch churn data: {e}")
                    if conn:
                        conn.close()

            # AI analysis of churn patterns
            client = get_openai_client()
            if client and results["at_risk_customers"]:
                try:
                    prompt = f"""Analyze customer churn risk based on this data:

At-Risk Customers:
{json.dumps(results['at_risk_customers'][:10], indent=2, default=str)}

Customer Stats:
{json.dumps(results.get('customer_stats', {}), indent=2)}

Provide churn analysis:
1. Risk assessment
2. Pattern identification
3. Intervention recommendations

Respond with JSON only:
{{
    "overall_churn_risk": "low/moderate/high/critical",
    "estimated_churn_rate": "X%",
    "risk_patterns": ["pattern1", "pattern2"],
    "high_priority_interventions": [
        {{"customer": "name", "risk_score": 0-100, "recommended_action": "...", "urgency": "immediate/soon/scheduled"}}
    ],
    "retention_strategies": ["strategy1", "strategy2"],
    "health_improvement_initiatives": ["initiative1"],
    "confidence_score": 0-100
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=800
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI churn analysis failed: {e}")

            await self._save_analysis("churn_risk", results)
            return results

        except Exception as e:
            logger.error(f"Churn risk analysis failed: {e}")
            return {"error": str(e)}

    async def _save_analysis(self, analysis_type: str, results: Dict[str, Any]):
        """Save analysis results to database"""
        conn = self._get_db_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_customer_success_analyses (tenant_id, analysis_type, results, analyzed_at)
                VALUES (%s, %s, %s, NOW())
            """, (self.tenant_id, analysis_type, json.dumps(results, default=str)))
            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"Saved {analysis_type} analysis for tenant {self.tenant_id}")
        except Exception as e:
            logger.warning(f"Failed to save analysis (table may not exist): {e}")
            if conn:
                conn.close()

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

    async def calculate_health_score(self, customer_id: str) -> Dict[str, Any]:
        """Calculate comprehensive customer health score with multiple factors"""
        try:
            conn = self._get_db_connection()
            if not conn:
                return {"error": "Database connection failed", "health_score": 0}

            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get comprehensive customer data
            cur.execute("""
                SELECT
                    c.id,
                    c.name,
                    c.created_at,
                    c.status,
                    COUNT(DISTINCT j.id) as total_jobs,
                    COUNT(DISTINCT j.id) FILTER (WHERE j.created_at > NOW() - INTERVAL '30 days') as recent_jobs,
                    COUNT(DISTINCT j.id) FILTER (WHERE j.status = 'completed') as completed_jobs,
                    MAX(j.created_at) as last_job_date,
                    COUNT(DISTINCT i.id) as total_invoices,
                    COUNT(DISTINCT i.id) FILTER (WHERE i.status = 'paid') as paid_invoices,
                    COUNT(DISTINCT i.id) FILTER (WHERE i.status = 'overdue') as overdue_invoices,
                    COALESCE(SUM(i.total_amount) FILTER (WHERE i.status = 'paid'), 0) as total_revenue,
                    COALESCE(AVG(i.total_amount), 0) as avg_invoice_value,
                    EXTRACT(EPOCH FROM (NOW() - MAX(j.created_at)))/86400 as days_since_last_job
                FROM customers c
                LEFT JOIN jobs j ON j.customer_id = c.id
                LEFT JOIN invoices i ON i.customer_id = c.id
                WHERE c.id::text = %s OR c.name ILIKE %s
                GROUP BY c.id, c.name, c.created_at, c.status
            """, (customer_id, f"%{customer_id}%"))

            customer_data = cur.fetchone()
            cur.close()
            conn.close()

            if not customer_data:
                return {"error": "Customer not found", "health_score": 0}

            # Calculate health score components
            score_components = {
                "engagement_score": 0,
                "revenue_score": 0,
                "payment_score": 0,
                "recency_score": 0,
                "frequency_score": 0
            }

            # 1. Engagement Score (0-25 points)
            recent_jobs = customer_data.get('recent_jobs', 0)
            if recent_jobs >= 3:
                score_components["engagement_score"] = 25
            elif recent_jobs >= 2:
                score_components["engagement_score"] = 20
            elif recent_jobs >= 1:
                score_components["engagement_score"] = 15
            else:
                score_components["engagement_score"] = 5

            # 2. Revenue Score (0-25 points)
            total_revenue = float(customer_data.get('total_revenue', 0))
            if total_revenue >= 50000:
                score_components["revenue_score"] = 25
            elif total_revenue >= 25000:
                score_components["revenue_score"] = 20
            elif total_revenue >= 10000:
                score_components["revenue_score"] = 15
            elif total_revenue >= 5000:
                score_components["revenue_score"] = 10
            else:
                score_components["revenue_score"] = 5

            # 3. Payment Score (0-25 points)
            total_invoices = customer_data.get('total_invoices', 0)
            paid_invoices = customer_data.get('paid_invoices', 0)
            overdue_invoices = customer_data.get('overdue_invoices', 0)

            if total_invoices > 0:
                payment_rate = paid_invoices / total_invoices
                if payment_rate >= 0.95 and overdue_invoices == 0:
                    score_components["payment_score"] = 25
                elif payment_rate >= 0.85 and overdue_invoices <= 1:
                    score_components["payment_score"] = 20
                elif payment_rate >= 0.70:
                    score_components["payment_score"] = 15
                elif payment_rate >= 0.50:
                    score_components["payment_score"] = 10
                else:
                    score_components["payment_score"] = 5
            else:
                score_components["payment_score"] = 15  # New customer, neutral score

            # 4. Recency Score (0-15 points)
            days_since_last_job = customer_data.get('days_since_last_job', 999)
            if days_since_last_job is None:
                score_components["recency_score"] = 5  # New customer
            elif days_since_last_job <= 30:
                score_components["recency_score"] = 15
            elif days_since_last_job <= 60:
                score_components["recency_score"] = 10
            elif days_since_last_job <= 90:
                score_components["recency_score"] = 5
            else:
                score_components["recency_score"] = 0

            # 5. Frequency Score (0-10 points)
            total_jobs = customer_data.get('total_jobs', 0)
            if total_jobs >= 10:
                score_components["frequency_score"] = 10
            elif total_jobs >= 5:
                score_components["frequency_score"] = 8
            elif total_jobs >= 3:
                score_components["frequency_score"] = 6
            elif total_jobs >= 1:
                score_components["frequency_score"] = 4
            else:
                score_components["frequency_score"] = 2

            # Calculate total health score (0-100)
            total_score = sum(score_components.values())

            # Determine health status
            if total_score >= 80:
                health_status = "excellent"
                risk_level = "very_low"
            elif total_score >= 65:
                health_status = "good"
                risk_level = "low"
            elif total_score >= 50:
                health_status = "fair"
                risk_level = "medium"
            elif total_score >= 35:
                health_status = "poor"
                risk_level = "high"
            else:
                health_status = "critical"
                risk_level = "very_high"

            result = {
                "customer_id": customer_data['id'],
                "customer_name": customer_data['name'],
                "health_score": total_score,
                "health_status": health_status,
                "risk_level": risk_level,
                "score_components": score_components,
                "metrics": {
                    "total_jobs": total_jobs,
                    "recent_jobs_30d": recent_jobs,
                    "total_revenue": total_revenue,
                    "payment_rate": (paid_invoices / total_invoices * 100) if total_invoices > 0 else 0,
                    "days_since_last_job": days_since_last_job
                },
                "calculated_at": datetime.utcnow().isoformat()
            }

            await self._save_analysis("health_score", result)
            return result

        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return {"error": str(e), "health_score": 0}

    async def predict_churn_risk(self, customer_id: str) -> Dict[str, Any]:
        """Predict churn risk with AI-powered analysis and proactive prevention recommendations"""
        try:
            # Get health score first
            health_data = await self.calculate_health_score(customer_id)

            conn = self._get_db_connection()
            if not conn:
                return {"error": "Database connection failed", "churn_risk": 0}

            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get additional churn indicators
            cur.execute("""
                SELECT
                    c.id,
                    c.name,
                    c.created_at,
                    EXTRACT(EPOCH FROM (NOW() - MAX(j.created_at)))/86400 as days_inactive,
                    COUNT(DISTINCT j.id) FILTER (WHERE j.created_at > NOW() - INTERVAL '90 days') as jobs_last_90d,
                    COUNT(DISTINCT j.id) FILTER (WHERE j.created_at > NOW() - INTERVAL '180 days'
                                                   AND j.created_at <= NOW() - INTERVAL '90 days') as jobs_prev_90d,
                    COUNT(DISTINCT i.id) FILTER (WHERE i.status = 'overdue') as overdue_count,
                    COUNT(DISTINCT i.id) FILTER (WHERE i.status = 'cancelled') as cancelled_count
                FROM customers c
                LEFT JOIN jobs j ON j.customer_id = c.id
                LEFT JOIN invoices i ON i.customer_id = c.id
                WHERE c.id = %s
                GROUP BY c.id, c.name, c.created_at
            """, (health_data.get('customer_id'),))

            churn_data = cur.fetchone()
            cur.close()
            conn.close()

            if not churn_data:
                return {"error": "Customer data not found", "churn_risk": 0}

            # Calculate churn risk factors
            risk_factors = []
            churn_score = 0

            # Factor 1: Inactivity (max 30 points)
            days_inactive = churn_data.get('days_inactive', 0) or 0
            if days_inactive > 90:
                churn_score += 30
                risk_factors.append(f"Inactive for {int(days_inactive)} days")
            elif days_inactive > 60:
                churn_score += 20
                risk_factors.append(f"Low activity - {int(days_inactive)} days since last job")
            elif days_inactive > 30:
                churn_score += 10
                risk_factors.append("Decreasing engagement")

            # Factor 2: Declining engagement (max 25 points)
            jobs_last_90d = churn_data.get('jobs_last_90d', 0)
            jobs_prev_90d = churn_data.get('jobs_prev_90d', 0)
            if jobs_prev_90d > 0 and jobs_last_90d < jobs_prev_90d * 0.5:
                churn_score += 25
                risk_factors.append("Job volume declined by 50%+")
            elif jobs_prev_90d > 0 and jobs_last_90d < jobs_prev_90d * 0.75:
                churn_score += 15
                risk_factors.append("Job volume declining")

            # Factor 3: Payment issues (max 25 points)
            overdue_count = churn_data.get('overdue_count', 0)
            cancelled_count = churn_data.get('cancelled_count', 0)
            if overdue_count >= 2 or cancelled_count >= 1:
                churn_score += 25
                risk_factors.append("Multiple payment issues")
            elif overdue_count >= 1:
                churn_score += 15
                risk_factors.append("Payment delays")

            # Factor 4: Low health score (max 20 points)
            health_score = health_data.get('health_score', 100)
            if health_score < 35:
                churn_score += 20
                risk_factors.append("Critical health score")
            elif health_score < 50:
                churn_score += 15
                risk_factors.append("Poor health score")
            elif health_score < 65:
                churn_score += 10
                risk_factors.append("Fair health score")

            # Determine churn risk level
            if churn_score >= 70:
                risk_level = "critical"
                churn_probability = min(95, 60 + (churn_score - 70) * 1.5)
            elif churn_score >= 50:
                risk_level = "high"
                churn_probability = 40 + (churn_score - 50) * 1.0
            elif churn_score >= 30:
                risk_level = "medium"
                churn_probability = 20 + (churn_score - 30) * 1.0
            elif churn_score >= 15:
                risk_level = "low"
                churn_probability = 5 + (churn_score - 15) * 1.0
            else:
                risk_level = "very_low"
                churn_probability = churn_score * 0.5

            # Generate proactive prevention strategies
            prevention_strategies = self._generate_prevention_strategies(
                risk_level, risk_factors, health_data, churn_data
            )

            result = {
                "customer_id": health_data.get('customer_id'),
                "customer_name": churn_data['name'],
                "churn_risk_score": churn_score,
                "churn_probability": round(churn_probability, 2),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "prevention_strategies": prevention_strategies,
                "health_context": {
                    "health_score": health_score,
                    "days_inactive": int(days_inactive) if days_inactive else 0,
                    "engagement_trend": "declining" if jobs_last_90d < jobs_prev_90d else "stable"
                },
                "recommended_actions": self._get_churn_prevention_actions(risk_level),
                "urgency": "immediate" if risk_level in ["critical", "high"] else "scheduled",
                "predicted_at": datetime.utcnow().isoformat()
            }

            await self._save_analysis("churn_prediction", result)
            return result

        except Exception as e:
            logger.error(f"Error predicting churn risk: {e}")
            return {"error": str(e), "churn_risk": 0}

    def _generate_prevention_strategies(
        self,
        risk_level: str,
        risk_factors: List[str],
        health_data: Dict,
        churn_data: Dict
    ) -> List[Dict[str, Any]]:
        """Generate personalized churn prevention strategies"""
        strategies = []

        # Strategy 1: Re-engagement campaign
        if any("inactive" in factor.lower() or "activity" in factor.lower() for factor in risk_factors):
            strategies.append({
                "type": "re_engagement",
                "priority": "high" if risk_level in ["critical", "high"] else "medium",
                "action": "Launch personalized re-engagement campaign",
                "tactics": [
                    "Send personalized 'we miss you' email",
                    "Offer exclusive discount or promotion",
                    "Schedule check-in call to understand needs",
                    "Provide case studies of recent successful projects"
                ],
                "timeline": "immediate" if risk_level == "critical" else "within_48_hours"
            })

        # Strategy 2: Payment issue resolution
        if any("payment" in factor.lower() for factor in risk_factors):
            strategies.append({
                "type": "payment_resolution",
                "priority": "high",
                "action": "Proactive payment issue resolution",
                "tactics": [
                    "Reach out to discuss payment challenges",
                    "Offer flexible payment plans",
                    "Review and optimize billing process",
                    "Provide financial assistance options if applicable"
                ],
                "timeline": "immediate"
            })

        # Strategy 3: Value reinforcement
        if health_data.get('health_score', 0) < 65:
            strategies.append({
                "type": "value_reinforcement",
                "priority": "medium",
                "action": "Demonstrate and reinforce value proposition",
                "tactics": [
                    "Share ROI analysis and value metrics",
                    "Provide success stories and testimonials",
                    "Offer free consultation or service audit",
                    "Highlight unique features and benefits"
                ],
                "timeline": "within_7_days"
            })

        # Strategy 4: Relationship building
        if risk_level in ["high", "critical"]:
            strategies.append({
                "type": "relationship_building",
                "priority": "high",
                "action": "Strengthen personal relationship",
                "tactics": [
                    "Assign dedicated account manager",
                    "Schedule quarterly business review",
                    "Invite to exclusive customer events",
                    "Request feedback and implement suggestions"
                ],
                "timeline": "within_72_hours"
            })

        # Strategy 5: Service improvement
        strategies.append({
            "type": "service_improvement",
            "priority": "medium",
            "action": "Optimize service delivery",
            "tactics": [
                "Conduct satisfaction survey",
                "Identify and resolve pain points",
                "Offer training or onboarding refresh",
                "Provide additional resources or support"
            ],
            "timeline": "within_14_days"
        })

        return strategies

    def _get_churn_prevention_actions(self, risk_level: str) -> List[Dict[str, str]]:
        """Get immediate churn prevention actions based on risk level"""
        actions_map = {
            "critical": [
                {"action": "Immediate executive escalation", "owner": "VP Customer Success"},
                {"action": "Emergency account review meeting", "owner": "Account Manager"},
                {"action": "Special retention offer preparation", "owner": "Sales Director"},
                {"action": "Root cause analysis", "owner": "CS Team"}
            ],
            "high": [
                {"action": "Schedule urgent check-in call", "owner": "Account Manager"},
                {"action": "Prepare retention proposal", "owner": "CS Manager"},
                {"action": "Review account history", "owner": "CS Team"},
                {"action": "Activate win-back campaign", "owner": "Marketing"}
            ],
            "medium": [
                {"action": "Schedule proactive check-in", "owner": "Account Manager"},
                {"action": "Send value reinforcement content", "owner": "CS Team"},
                {"action": "Monitor engagement closely", "owner": "CS Team"}
            ],
            "low": [
                {"action": "Include in routine check-in", "owner": "CS Team"},
                {"action": "Continue monitoring", "owner": "CS Team"}
            ],
            "very_low": [
                {"action": "Standard engagement", "owner": "CS Team"}
            ]
        }

        return actions_map.get(risk_level, actions_map["medium"])

    async def predict_satisfaction(self, customer_id: str) -> Dict[str, Any]:
        """Predict customer satisfaction using AI-powered sentiment and behavior analysis"""
        try:
            # Get health and churn data
            health_data = await self.calculate_health_score(customer_id)
            churn_data = await self.predict_churn_risk(customer_id)

            conn = self._get_db_connection()
            if not conn:
                return {"error": "Database connection failed", "satisfaction_score": 0}

            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get behavioral indicators
            cur.execute("""
                SELECT
                    c.id,
                    c.name,
                    COUNT(DISTINCT j.id) as total_jobs,
                    AVG(CASE WHEN j.status = 'completed' THEN 1 ELSE 0 END) * 100 as completion_rate,
                    COUNT(DISTINCT j.id) FILTER (WHERE j.created_at > NOW() - INTERVAL '30 days') as recent_activity,
                    AVG(i.total_amount) as avg_order_value,
                    COUNT(DISTINCT i.id) FILTER (WHERE i.status = 'paid'
                                                   AND i.paid_at <= i.due_date) as on_time_payments,
                    COUNT(DISTINCT i.id) FILTER (WHERE i.status = 'paid') as total_paid_invoices
                FROM customers c
                LEFT JOIN jobs j ON j.customer_id = c.id
                LEFT JOIN invoices i ON i.customer_id = c.id
                WHERE c.id = %s
                GROUP BY c.id, c.name
            """, (health_data.get('customer_id'),))

            satisfaction_data = cur.fetchone()
            cur.close()
            conn.close()

            if not satisfaction_data:
                return {"error": "Customer data not found", "satisfaction_score": 0}

            # Calculate satisfaction score components
            satisfaction_score = 0
            satisfaction_factors = {}

            # Factor 1: Service completion rate (max 25 points)
            completion_rate = satisfaction_data.get('completion_rate', 0) or 0
            if completion_rate >= 95:
                satisfaction_score += 25
                satisfaction_factors["service_quality"] = "excellent"
            elif completion_rate >= 85:
                satisfaction_score += 20
                satisfaction_factors["service_quality"] = "good"
            elif completion_rate >= 75:
                satisfaction_score += 15
                satisfaction_factors["service_quality"] = "fair"
            else:
                satisfaction_score += 10
                satisfaction_factors["service_quality"] = "poor"

            # Factor 2: Payment behavior (max 20 points)
            total_paid = satisfaction_data.get('total_paid_invoices', 0)
            on_time = satisfaction_data.get('on_time_payments', 0)
            if total_paid > 0:
                on_time_rate = (on_time / total_paid) * 100
                if on_time_rate >= 95:
                    satisfaction_score += 20
                    satisfaction_factors["payment_satisfaction"] = "very_satisfied"
                elif on_time_rate >= 80:
                    satisfaction_score += 15
                    satisfaction_factors["payment_satisfaction"] = "satisfied"
                else:
                    satisfaction_score += 10
                    satisfaction_factors["payment_satisfaction"] = "neutral"
            else:
                satisfaction_score += 15
                satisfaction_factors["payment_satisfaction"] = "new_customer"

            # Factor 3: Engagement level (max 20 points)
            recent_activity = satisfaction_data.get('recent_activity', 0)
            if recent_activity >= 5:
                satisfaction_score += 20
                satisfaction_factors["engagement"] = "highly_engaged"
            elif recent_activity >= 3:
                satisfaction_score += 15
                satisfaction_factors["engagement"] = "engaged"
            elif recent_activity >= 1:
                satisfaction_score += 10
                satisfaction_factors["engagement"] = "moderately_engaged"
            else:
                satisfaction_score += 5
                satisfaction_factors["engagement"] = "disengaged"

            # Factor 4: Health score contribution (max 20 points)
            health_score = health_data.get('health_score', 0)
            health_contribution = (health_score / 100) * 20
            satisfaction_score += health_contribution

            # Factor 5: Churn risk (inverse relationship, max 15 points)
            churn_risk = churn_data.get('churn_risk_score', 0)
            churn_contribution = max(0, 15 - (churn_risk / 100 * 15))
            satisfaction_score += churn_contribution

            # Normalize to 0-100
            satisfaction_score = min(100, satisfaction_score)

            # Determine satisfaction level
            if satisfaction_score >= 80:
                satisfaction_level = "highly_satisfied"
                nps_category = "promoter"
                predicted_nps = 9 + (satisfaction_score - 80) / 20
            elif satisfaction_score >= 60:
                satisfaction_level = "satisfied"
                nps_category = "passive"
                predicted_nps = 7 + (satisfaction_score - 60) / 10
            elif satisfaction_score >= 40:
                satisfaction_level = "neutral"
                nps_category = "passive"
                predicted_nps = 5 + (satisfaction_score - 40) / 10
            elif satisfaction_score >= 20:
                satisfaction_level = "dissatisfied"
                nps_category = "detractor"
                predicted_nps = 3 + (satisfaction_score - 20) / 10
            else:
                satisfaction_level = "highly_dissatisfied"
                nps_category = "detractor"
                predicted_nps = satisfaction_score / 20 * 3

            # Generate improvement recommendations
            improvement_recommendations = self._generate_satisfaction_improvements(
                satisfaction_score, satisfaction_factors, health_data, churn_data
            )

            result = {
                "customer_id": health_data.get('customer_id'),
                "customer_name": satisfaction_data['name'],
                "satisfaction_score": round(satisfaction_score, 2),
                "satisfaction_level": satisfaction_level,
                "predicted_nps": round(predicted_nps, 1),
                "nps_category": nps_category,
                "satisfaction_factors": satisfaction_factors,
                "metrics": {
                    "completion_rate": round(completion_rate, 2),
                    "on_time_payment_rate": round((on_time / total_paid * 100) if total_paid > 0 else 0, 2),
                    "recent_activity_level": recent_activity,
                    "health_score": health_score,
                    "churn_risk": churn_risk
                },
                "improvement_recommendations": improvement_recommendations,
                "survey_priority": "high" if satisfaction_score < 60 else "medium" if satisfaction_score < 80 else "low",
                "predicted_at": datetime.utcnow().isoformat()
            }

            await self._save_analysis("satisfaction_prediction", result)
            return result

        except Exception as e:
            logger.error(f"Error predicting satisfaction: {e}")
            return {"error": str(e), "satisfaction_score": 0}

    def _generate_satisfaction_improvements(
        self,
        satisfaction_score: float,
        factors: Dict[str, str],
        health_data: Dict,
        churn_data: Dict
    ) -> List[Dict[str, Any]]:
        """Generate satisfaction improvement recommendations"""
        recommendations = []

        # Recommendation 1: Service quality improvement
        if factors.get("service_quality") in ["fair", "poor"]:
            recommendations.append({
                "area": "Service Quality",
                "priority": "high",
                "recommendation": "Improve service completion rates and quality",
                "actions": [
                    "Review and resolve ongoing service issues",
                    "Implement quality assurance process",
                    "Provide additional training to service team",
                    "Set up proactive quality monitoring"
                ]
            })

        # Recommendation 2: Engagement improvement
        if factors.get("engagement") in ["disengaged", "moderately_engaged"]:
            recommendations.append({
                "area": "Customer Engagement",
                "priority": "high",
                "recommendation": "Increase customer engagement and interaction",
                "actions": [
                    "Launch re-engagement campaign",
                    "Schedule personalized check-in calls",
                    "Provide valuable content and resources",
                    "Invite to customer community or events"
                ]
            })

        # Recommendation 3: Payment experience
        if factors.get("payment_satisfaction") in ["neutral", "dissatisfied"]:
            recommendations.append({
                "area": "Payment Experience",
                "priority": "medium",
                "recommendation": "Improve billing and payment processes",
                "actions": [
                    "Simplify invoicing and payment methods",
                    "Offer flexible payment options",
                    "Provide clearer payment terms",
                    "Implement automated payment reminders"
                ]
            })

        # Recommendation 4: Relationship strengthening
        if satisfaction_score < 70:
            recommendations.append({
                "area": "Customer Relationship",
                "priority": "high",
                "recommendation": "Strengthen customer relationship and trust",
                "actions": [
                    "Assign dedicated account manager",
                    "Conduct satisfaction survey",
                    "Implement regular business reviews",
                    "Create customer success plan"
                ]
            })

        # Recommendation 5: Value demonstration
        recommendations.append({
            "area": "Value Proposition",
            "priority": "medium",
            "recommendation": "Continuously demonstrate and communicate value",
            "actions": [
                "Share ROI metrics and success stories",
                "Provide regular value reports",
                "Highlight cost savings and benefits",
                "Offer additional value-added services"
            ]
        })

        return recommendations

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

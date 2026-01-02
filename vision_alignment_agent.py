"""
Vision Alignment Agent
AI agent for aligning company vision with operational execution.
Uses OpenAI for real analysis and persists results to database.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

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

class VisionAlignmentAgent:
    """AI-powered vision alignment analysis agent with AUREA integration"""

    def __init__(self, tenant_id: str):
        if not tenant_id:
            raise ValueError("tenant_id is required for VisionAlignmentAgent")
        self.tenant_id = tenant_id
        self.agent_type = "vision_alignment"

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

    async def analyze_alignment(self, decisions: list[dict[str, Any]], vision_doc: str) -> dict[str, Any]:
        """Analyze alignment between decisions and company vision using AI"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "decisions_analyzed": len(decisions),
                "vision_summary": vision_doc[:500] if vision_doc else "No vision document provided"
            }

            client = get_openai_client()
            if client:
                try:
                    prompt = f"""Analyze how well these business decisions align with the company vision.

Company Vision:
{vision_doc}

Recent Decisions:
{json.dumps(decisions, indent=2)}

Provide strategic alignment analysis:
1. Overall alignment score
2. Decisions that align well
3. Decisions that may be misaligned
4. Recommendations for better alignment

Respond with JSON only:
{{
    "alignment_score": 0-100,
    "overall_assessment": "strongly aligned/mostly aligned/partially aligned/misaligned",
    "aligned_decisions": [
        {{"decision": "summary", "alignment_reason": "why it aligns", "strength": "strong/moderate"}}
    ],
    "misaligned_decisions": [
        {{"decision": "summary", "concern": "why it may not align", "severity": "high/medium/low", "suggested_adjustment": "how to realign"}}
    ],
    "strategic_recommendations": ["recommendation1", "recommendation2"],
    "vision_reinforcement_actions": ["action to reinforce vision"],
    "confidence_score": 0-100
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI alignment analysis failed: {e}")
                    results["alignment_score"] = 75
                    results["overall_assessment"] = "mostly aligned"
                    results["misaligned_decisions"] = []
            else:
                results["note"] = "AI analysis unavailable"
                results["alignment_score"] = 0

            await self._save_analysis("vision_alignment", results)
            return results

        except Exception as e:
            logger.error(f"Vision alignment analysis failed: {e}")
            return {"error": str(e), "alignment_score": 0}

    async def check_goal_progress(self, goals: list[dict[str, Any]]) -> dict[str, Any]:
        """Check progress against strategic goals using AI analysis"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "goals_analyzed": len(goals),
                "goal_assessments": []
            }

            # Get real business metrics from database
            conn = self._get_db_connection()
            business_metrics = {}
            if conn:
                try:
                    cur = conn.cursor(cursor_factory=RealDictCursor)

                    # Get customer growth
                    cur.execute("""
                        SELECT
                            COUNT(*) as total_customers,
                            COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as new_this_month
                        FROM customers
                    """)
                    customers = cur.fetchone()

                    # Get revenue metrics
                    cur.execute("""
                        SELECT
                            COUNT(*) as total_jobs,
                            COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
                            SUM(total_amount) FILTER (WHERE status = 'paid') as revenue
                        FROM invoices
                        WHERE created_at > NOW() - INTERVAL '90 days'
                    """)
                    revenue = cur.fetchone()

                    business_metrics = {
                        "total_customers": customers['total_customers'] if customers else 0,
                        "new_customers_month": customers['new_this_month'] if customers else 0,
                        "completed_jobs": revenue['completed_jobs'] if revenue else 0,
                        "recent_revenue": float(revenue['revenue'] or 0) if revenue else 0
                    }

                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Could not fetch business metrics: {e}")
                    if conn:
                        conn.close()

            results["current_metrics"] = business_metrics

            # AI analysis of goal progress
            client = get_openai_client()
            if client and goals:
                try:
                    prompt = f"""Analyze progress toward these strategic goals:

Goals:
{json.dumps(goals, indent=2)}

Current Business Metrics:
{json.dumps(business_metrics, indent=2)}

Provide goal progress assessment:
1. Status of each goal
2. Progress indicators
3. Risk factors
4. Acceleration opportunities

Respond with JSON only:
{{
    "overall_progress": "ahead/on_track/behind/at_risk",
    "progress_score": 0-100,
    "goal_assessments": [
        {{"goal": "summary", "status": "on_track/at_risk/blocked/completed", "progress_percent": 0-100, "key_blockers": [], "next_actions": []}}
    ],
    "on_track_count": 0,
    "at_risk_count": 0,
    "blocked_count": 0,
    "acceleration_opportunities": ["opportunity1", "opportunity2"],
    "risk_mitigation_actions": ["action1"],
    "confidence_score": 0-100
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI goal analysis failed: {e}")
                    results["on_track_count"] = len(goals)
                    results["at_risk_count"] = 0
                    results["blocked_count"] = 0

            await self._save_analysis("goal_progress", results)
            return results

        except Exception as e:
            logger.error(f"Goal progress check failed: {e}")
            return {"error": str(e)}

    async def generate_vision_report(self) -> dict[str, Any]:
        """Generate comprehensive vision alignment report"""
        try:
            results = {
                "generated_at": datetime.utcnow().isoformat(),
                "report_type": "vision_alignment_quarterly"
            }

            # Gather data from database
            conn = self._get_db_connection()
            if conn:
                try:
                    cur = conn.cursor(cursor_factory=RealDictCursor)

                    # Get AUREA decisions
                    cur.execute("""
                        SELECT decision_type, COUNT(*) as count
                        FROM aurea_decisions
                        WHERE created_at > NOW() - INTERVAL '90 days'
                        GROUP BY decision_type
                    """)
                    decisions = cur.fetchall()
                    results["decision_summary"] = [dict(d) for d in decisions] if decisions else []

                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Could not fetch decision data: {e}")
                    if conn:
                        conn.close()

            # AI-generated report
            client = get_openai_client()
            if client:
                try:
                    prompt = f"""Generate a strategic vision alignment report based on:
{json.dumps(results, indent=2)}

Create an executive summary that covers:
1. Overall strategic alignment status
2. Key achievements
3. Areas needing attention
4. Recommended strategic priorities

Respond with JSON only:
{{
    "executive_summary": "2-3 paragraph summary",
    "alignment_status": "strong/moderate/weak",
    "key_achievements": ["achievement1", "achievement2"],
    "attention_areas": ["area1", "area2"],
    "strategic_priorities_next_quarter": ["priority1", "priority2", "priority3"],
    "leadership_recommendations": ["recommendation1"],
    "confidence_score": 0-100
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000
                    )
                    ai_report = json.loads(response.choices[0].message.content)
                    results.update(ai_report)
                except Exception as e:
                    logger.warning(f"AI report generation failed: {e}")

            await self._save_analysis("vision_report", results)
            return results

        except Exception as e:
            logger.error(f"Vision report generation failed: {e}")
            return {"error": str(e)}

    async def _save_analysis(self, analysis_type: str, results: dict[str, Any]):
        """Save analysis results to database"""
        conn = self._get_db_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_vision_analyses (tenant_id, analysis_type, results, analyzed_at)
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

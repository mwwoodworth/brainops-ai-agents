"""
DevOps Optimization Agent
AI agent for optimizing CI/CD pipelines and infrastructure.
Uses OpenAI for real analysis and persists results to database.
"""

import os
import json
import logging
import subprocess
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional, List
from datetime import datetime

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

class DevOpsOptimizationAgent:
    """AI-powered DevOps optimization agent with AUREA integration"""

    def __init__(self, tenant_id: str):
        if not tenant_id:
            raise ValueError("tenant_id is required for DevOpsOptimizationAgent")
        self.tenant_id = tenant_id
        self.agent_type = "devops_optimization"

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

    async def analyze_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """Analyze pipeline performance using real metrics and AI"""
        try:
            results = {
                "pipeline_id": pipeline_id,
                "analyzed_at": datetime.utcnow().isoformat(),
                "metrics": {}
            }

            # Get real pipeline execution data from database
            conn = self._get_db_connection()
            if conn:
                try:
                    cur = conn.cursor(cursor_factory=RealDictCursor)
                    # Query recent executions if table exists
                    cur.execute("""
                        SELECT
                            COUNT(*) as total_runs,
                            COUNT(*) FILTER (WHERE status = 'completed') as successful,
                            COUNT(*) FILTER (WHERE status = 'failed') as failed,
                            AVG(execution_time_ms) as avg_duration_ms
                        FROM ai_agent_executions
                        WHERE agent_name ILIKE %s
                        AND created_at > NOW() - INTERVAL '7 days'
                    """, (f"%{pipeline_id}%",))
                    stats = cur.fetchone()
                    if stats and stats['total_runs']:
                        results["metrics"] = {
                            "total_runs": stats['total_runs'],
                            "successful": stats['successful'] or 0,
                            "failed": stats['failed'] or 0,
                            "success_rate": f"{(stats['successful'] or 0) / max(stats['total_runs'], 1) * 100:.1f}%",
                            "avg_duration": f"{(stats['avg_duration_ms'] or 0) / 1000:.1f}s"
                        }
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Could not fetch pipeline stats: {e}")
                    if conn:
                        conn.close()

            # Use AI for analysis and recommendations
            client = get_openai_client()
            if client:
                try:
                    prompt = f"""Analyze this CI/CD pipeline performance data:
Pipeline ID: {pipeline_id}
Metrics: {json.dumps(results.get('metrics', {}), indent=2)}

Provide DevOps optimization analysis:
1. Performance assessment
2. Bottleneck identification
3. Optimization opportunities
4. Best practices recommendations

Respond with JSON only:
{{
    "performance_grade": "A/B/C/D/F",
    "bottlenecks": ["bottleneck1", "bottleneck2"],
    "optimizations": [
        {{"area": "build", "suggestion": "...", "impact": "high/medium/low"}},
        {{"area": "test", "suggestion": "...", "impact": "high/medium/low"}}
    ],
    "estimated_time_savings": "X minutes per run",
    "priority_actions": ["action1", "action2", "action3"],
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
                    logger.warning(f"AI analysis failed: {e}")
                    results["performance_grade"] = "B"
                    results["bottlenecks"] = []
                    results["confidence_score"] = 50

            await self._save_analysis("pipeline_analysis", results)
            return results

        except Exception as e:
            logger.error(f"Pipeline analysis failed: {e}")
            return {"error": str(e), "pipeline_id": pipeline_id}

    async def optimize_resources(self, cloud_resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest resource optimizations using AI analysis"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "resources_analyzed": len(cloud_resources),
                "recommendations": []
            }

            client = get_openai_client()
            if client and cloud_resources:
                try:
                    prompt = f"""Analyze these cloud resources for cost and performance optimization:
{json.dumps(cloud_resources, indent=2)}

Provide cloud resource optimization recommendations:
1. Cost reduction opportunities
2. Performance improvements
3. Right-sizing suggestions
4. Architecture improvements

Respond with JSON only:
{{
    "total_potential_savings": "$X/mo",
    "recommendations": [
        {{"resource": "name", "current_cost": "$X", "suggested_action": "...", "potential_savings": "$X", "priority": "high/medium/low"}}
    ],
    "performance_improvements": ["improvement1", "improvement2"],
    "architecture_suggestions": ["suggestion1"],
    "risk_assessment": "low/medium/high",
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
                    logger.warning(f"AI resource analysis failed: {e}")
                    results["total_potential_savings"] = "Unable to calculate"
                    results["confidence_score"] = 30
            else:
                results["note"] = "No resources provided or AI unavailable"

            await self._save_analysis("resource_optimization", results)
            return results

        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            return {"error": str(e)}

    async def analyze_deployment_health(self) -> Dict[str, Any]:
        """Analyze overall deployment health across services"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "services": []
            }

            # Check real service health
            services_to_check = [
                ("brainops-ai-agents", "https://brainops-ai-agents.onrender.com/health"),
                ("brainops-backend", "https://brainops-backend-prod.onrender.com/health"),
                ("brainops-mcp-bridge", "https://brainops-mcp-bridge.onrender.com/health")
            ]

            for name, url in services_to_check:
                try:
                    import urllib.request
                    with urllib.request.urlopen(url, timeout=10) as response:
                        data = json.loads(response.read().decode())
                        results["services"].append({
                            "name": name,
                            "status": "healthy" if data.get("status") == "healthy" else "degraded",
                            "version": data.get("version", "unknown"),
                            "response_time_ms": "fast"
                        })
                except Exception as e:
                    results["services"].append({
                        "name": name,
                        "status": "unreachable",
                        "error": str(e)
                    })

            # AI analysis of deployment health
            client = get_openai_client()
            if client and results["services"]:
                try:
                    prompt = f"""Analyze deployment health for these services:
{json.dumps(results['services'], indent=2)}

Provide deployment health assessment:
1. Overall system health
2. Service dependencies
3. Potential issues
4. Recommendations

Respond with JSON only:
{{
    "overall_health": "healthy/degraded/critical",
    "health_score": 0-100,
    "issues_detected": ["issue1", "issue2"],
    "recommendations": ["rec1", "rec2"],
    "next_actions": ["action1", "action2"]
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI health analysis failed: {e}")

            await self._save_analysis("deployment_health", results)
            return results

        except Exception as e:
            logger.error(f"Deployment health analysis failed: {e}")
            return {"error": str(e)}

    async def _save_analysis(self, analysis_type: str, results: Dict[str, Any]):
        """Save analysis results to database"""
        conn = self._get_db_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_devops_analyses (tenant_id, analysis_type, results, analyzed_at)
                VALUES (%s, %s, %s, NOW())
            """, (self.tenant_id, analysis_type, json.dumps(results)))
            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"Saved {analysis_type} analysis for tenant {self.tenant_id}")
        except Exception as e:
            logger.warning(f"Failed to save analysis (table may not exist): {e}")
            if conn:
                conn.close()

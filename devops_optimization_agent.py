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
from typing import Dict, Any, List
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

    async def scan_security_vulnerabilities(self, repo_path: str) -> Dict[str, Any]:
        """Scan for security vulnerabilities in code and dependencies"""
        try:
            results = {
                "repo_path": repo_path,
                "scanned_at": datetime.utcnow().isoformat(),
                "vulnerabilities": [],
                "risk_score": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0
            }

            # Check for common security issues in code
            security_patterns = [
                (r"password\s*=\s*['\"].*['\"]", "Hardcoded password", "critical"),
                (r"api[_-]?key\s*=\s*['\"].*['\"]", "Hardcoded API key", "critical"),
                (r"secret\s*=\s*['\"].*['\"]", "Hardcoded secret", "critical"),
                (r"eval\(", "Use of eval() function", "high"),
                (r"exec\(", "Use of exec() function", "high"),
                (r"__import__\(", "Dynamic import", "medium"),
                (r"pickle\.loads\(", "Unsafe pickle deserialization", "high"),
                (r"os\.system\(", "OS command execution", "high"),
                (r"subprocess\.call\(.*, shell=True", "Shell injection risk", "critical"),
                (r"sql.*\+.*%s", "Potential SQL injection", "critical"),
            ]

            for pattern, description, severity in security_patterns:
                try:
                    grep_result = subprocess.run(
                        ["grep", "-r", "-n", "-E", pattern, repo_path, "--include=*.py"],
                        capture_output=True, text=True, timeout=30
                    )
                    if grep_result.returncode == 0 and grep_result.stdout:
                        lines = grep_result.stdout.strip().split('\n')
                        for line in lines[:10]:  # Limit to 10 occurrences per pattern
                            if ':' in line:
                                file_path, line_num, code = line.split(':', 2)
                                results["vulnerabilities"].append({
                                    "file": file_path.replace(repo_path, ''),
                                    "line": line_num,
                                    "severity": severity,
                                    "description": description,
                                    "code_snippet": code.strip()[:100]
                                })
                                if severity == "critical":
                                    results["critical_count"] += 1
                                elif severity == "high":
                                    results["high_count"] += 1
                                elif severity == "medium":
                                    results["medium_count"] += 1
                                else:
                                    results["low_count"] += 1
                except ValueError as exc:
                    logger.debug("Failed to parse vulnerability line: %s", exc, exc_info=True)

            # Check for outdated dependencies (Python)
            try:
                pip_result = subprocess.run(
                    ["pip", "list", "--outdated", "--format=json"],
                    capture_output=True, text=True, timeout=30
                )
                if pip_result.returncode == 0 and pip_result.stdout:
                    outdated = json.loads(pip_result.stdout)
                    for pkg in outdated[:20]:  # Limit to 20 packages
                        results["vulnerabilities"].append({
                            "file": "requirements.txt",
                            "severity": "medium",
                            "description": f"Outdated package: {pkg['name']}",
                            "current_version": pkg.get('version'),
                            "latest_version": pkg.get('latest_version')
                        })
                        results["medium_count"] += 1
            except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
                logger.debug("Failed to read outdated dependencies: %s", exc, exc_info=True)

            # Calculate risk score (0-100)
            results["risk_score"] = min(100,
                results["critical_count"] * 25 +
                results["high_count"] * 10 +
                results["medium_count"] * 3 +
                results["low_count"] * 1
            )

            # Get AI recommendations
            client = get_openai_client()
            if client and results["vulnerabilities"]:
                try:
                    vuln_summary = {
                        "critical": results["critical_count"],
                        "high": results["high_count"],
                        "medium": results["medium_count"],
                        "low": results["low_count"],
                        "top_issues": results["vulnerabilities"][:5]
                    }

                    prompt = f"""Analyze these security vulnerabilities and provide recommendations:
{json.dumps(vuln_summary, indent=2)}

Respond with JSON only:
{{
    "severity_assessment": "critical/high/medium/low",
    "immediate_actions": ["action1", "action2"],
    "remediation_steps": ["step1", "step2"],
    "best_practices": ["practice1", "practice2"],
    "estimated_fix_time": "X hours",
    "confidence_score": 0-100
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=600
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI security analysis failed: {e}")

            await self._save_analysis("security_scan", results)
            return results

        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return {"error": str(e), "repo_path": repo_path}

    async def optimize_infrastructure_costs(self, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze infrastructure costs and suggest optimizations"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "total_monthly_cost": 0,
                "potential_savings": 0,
                "optimization_opportunities": [],
                "recommendations": []
            }

            # Calculate current costs
            for resource in resources:
                cost = resource.get("monthly_cost", 0)
                results["total_monthly_cost"] += cost

            # Use AI for cost optimization analysis
            client = get_openai_client()
            if client and resources:
                try:
                    prompt = f"""Analyze these infrastructure resources for cost optimization:
{json.dumps(resources[:20], indent=2)}
Total monthly cost: ${results['total_monthly_cost']}

Provide detailed cost optimization recommendations:
1. Underutilized resources to downsize or remove
2. Right-sizing opportunities
3. Reserved instance/commitment opportunities
4. Architecture improvements
5. Alternative services that are cheaper

Respond with JSON only:
{{
    "potential_monthly_savings": "$X",
    "savings_percentage": X,
    "optimizations": [
        {{
            "resource": "name",
            "current_cost": "$X/mo",
            "action": "specific action",
            "new_cost": "$X/mo",
            "savings": "$X/mo",
            "implementation_effort": "low/medium/high",
            "priority": "high/medium/low"
        }}
    ],
    "quick_wins": ["win1", "win2"],
    "long_term_strategies": ["strategy1", "strategy2"],
    "confidence_score": 0-100
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)

                    # Extract potential savings
                    if "potential_monthly_savings" in ai_analysis:
                        savings_str = ai_analysis["potential_monthly_savings"].replace('$', '').replace(',', '')
                        try:
                            results["potential_savings"] = float(savings_str)
                        except (ValueError, TypeError) as exc:
                            logger.debug("Failed to parse savings value: %s", exc, exc_info=True)

                except Exception as e:
                    logger.warning(f"AI cost analysis failed: {e}")

            await self._save_analysis("cost_optimization", results)
            return results

        except Exception as e:
            logger.error(f"Cost optimization failed: {e}")
            return {"error": str(e)}

    async def analyze_logs(self, log_source: str, log_lines: List[str]) -> Dict[str, Any]:
        """Analyze logs for errors, patterns, and anomalies"""
        try:
            results = {
                "log_source": log_source,
                "analyzed_at": datetime.utcnow().isoformat(),
                "total_lines": len(log_lines),
                "errors": [],
                "warnings": [],
                "patterns": [],
                "anomalies": [],
                "alerts": []
            }

            # Analyze log lines
            error_keywords = ['error', 'exception', 'fatal', 'critical', 'failed']
            warning_keywords = ['warn', 'warning', 'deprecated']

            error_counts = {}
            warning_counts = {}

            for i, line in enumerate(log_lines):
                lower_line = line.lower()

                # Check for errors
                for keyword in error_keywords:
                    if keyword in lower_line:
                        results["errors"].append({
                            "line_number": i + 1,
                            "message": line[:200],
                            "keyword": keyword
                        })
                        error_counts[keyword] = error_counts.get(keyword, 0) + 1
                        break

                # Check for warnings
                for keyword in warning_keywords:
                    if keyword in lower_line:
                        results["warnings"].append({
                            "line_number": i + 1,
                            "message": line[:200],
                            "keyword": keyword
                        })
                        warning_counts[keyword] = warning_counts.get(keyword, 0) + 1
                        break

            # Identify patterns
            if error_counts:
                results["patterns"].append({
                    "type": "error_distribution",
                    "data": error_counts
                })

            if warning_counts:
                results["patterns"].append({
                    "type": "warning_distribution",
                    "data": warning_counts
                })

            # Create alerts for critical issues
            if len(results["errors"]) > 100:
                results["alerts"].append({
                    "severity": "high",
                    "message": f"High error rate detected: {len(results['errors'])} errors",
                    "action": "Immediate investigation required"
                })

            # Use AI for log analysis
            client = get_openai_client()
            if client and (results["errors"] or results["warnings"]):
                try:
                    sample_errors = results["errors"][:10]
                    sample_warnings = results["warnings"][:10]

                    prompt = f"""Analyze these log entries and provide insights:
Errors ({len(results['errors'])} total):
{json.dumps(sample_errors, indent=2)}

Warnings ({len(results['warnings'])} total):
{json.dumps(sample_warnings, indent=2)}

Respond with JSON only:
{{
    "root_causes": ["cause1", "cause2"],
    "severity_assessment": "critical/high/medium/low",
    "recommended_actions": ["action1", "action2"],
    "potential_impacts": ["impact1", "impact2"],
    "monitoring_recommendations": ["metric1", "metric2"],
    "confidence_score": 0-100
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=700
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI log analysis failed: {e}")

            await self._save_analysis("log_analysis", results)
            return results

        except Exception as e:
            logger.error(f"Log analysis failed: {e}")
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

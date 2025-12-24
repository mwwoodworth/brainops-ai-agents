"""
Code Quality Agent
AI agent for monitoring and improving codebase quality.
Uses OpenAI for real code analysis and persists results to database.
"""

import os
import json
import logging
import subprocess
from typing import Dict, Any, Optional, List
from datetime import datetime
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

class CodeQualityAgent:
    """AI-powered code quality analysis agent with AUREA integration"""

    def __init__(self, tenant_id: str):
        if not tenant_id:
            raise ValueError("tenant_id is required for CodeQualityAgent")
        self.tenant_id = tenant_id
        self.agent_type = "code_quality"

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

    async def analyze_codebase(self, repo_path: str) -> Dict[str, Any]:
        """Analyze codebase for quality metrics using real static analysis"""
        try:
            results = {
                "repo_path": repo_path,
                "analyzed_at": datetime.utcnow().isoformat(),
                "metrics": {}
            }

            # Run actual static analysis tools if available
            # Count lines of code
            try:
                loc_result = subprocess.run(
                    ["find", repo_path, "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
                    capture_output=True, text=True, timeout=30
                )
                if loc_result.returncode == 0:
                    lines = loc_result.stdout.strip().split('\n')
                    if lines:
                        total = lines[-1].split()[0] if lines[-1] else "0"
                        results["metrics"]["total_lines"] = int(total)
            except Exception as e:
                logger.warning(f"LOC count failed: {e}")

            # Check for common code issues using grep patterns
            issues = []
            issue_patterns = [
                ("TODO", "Unfinished tasks"),
                ("FIXME", "Known bugs"),
                ("XXX", "Problematic code"),
                ("HACK", "Temporary workarounds"),
                ("pass$", "Empty implementations")
            ]

            for pattern, description in issue_patterns:
                try:
                    grep_result = subprocess.run(
                        ["grep", "-r", "-c", pattern, repo_path, "--include=*.py"],
                        capture_output=True, text=True, timeout=30
                    )
                    count = sum(int(line.split(':')[-1]) for line in grep_result.stdout.strip().split('\n') if ':' in line)
                    if count > 0:
                        issues.append({"pattern": pattern, "count": count, "description": description})
                except Exception:
                    pass

            results["metrics"]["issues"] = issues
            results["metrics"]["issue_count"] = sum(i["count"] for i in issues)

            # Use AI to generate quality score and recommendations
            client = get_openai_client()
            if client and issues:
                try:
                    prompt = f"""Analyze these code quality findings and provide a JSON response:
Issues found: {json.dumps(issues)}
Total lines of code: {results['metrics'].get('total_lines', 'unknown')}

Respond with JSON only:
{{"quality_score": 0-100, "technical_debt": "low/medium/high", "recommendations": ["recommendation1", "recommendation2"]}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results["metrics"].update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI analysis failed: {e}")
                    # Fallback scoring
                    issue_count = results["metrics"].get("issue_count", 0)
                    results["metrics"]["quality_score"] = max(40, 100 - issue_count * 2)
                    results["metrics"]["technical_debt"] = "high" if issue_count > 50 else "medium" if issue_count > 20 else "low"
            else:
                # No issues or no AI - good score
                results["metrics"]["quality_score"] = 95
                results["metrics"]["technical_debt"] = "low"

            # Persist results to database
            await self._save_analysis(results)

            return results

        except Exception as e:
            logger.error(f"Codebase analysis failed: {e}")
            return {"error": str(e), "quality_score": 0}

    async def review_pr(self, pr_details: Dict[str, Any]) -> Dict[str, Any]:
        """Review a pull request using AI analysis"""
        try:
            client = get_openai_client()
            if not client:
                return {"error": "OpenAI client not available", "status": "pending"}

            # Extract PR info
            title = pr_details.get("title", "Unknown PR")
            description = pr_details.get("description", "")
            diff = pr_details.get("diff", "")[:4000]  # Limit diff size
            files_changed = pr_details.get("files_changed", [])

            prompt = f"""Review this pull request and provide constructive feedback.

Title: {title}
Description: {description}
Files changed: {files_changed}
Diff (partial):
{diff}

Respond with JSON only:
{{
    "status": "approved" or "changes_requested" or "needs_discussion",
    "summary": "one sentence summary",
    "comments": ["specific comment 1", "specific comment 2"],
    "suggestions": ["improvement suggestion 1"],
    "security_concerns": ["any security issues"],
    "score": 0-100
}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )

            review = json.loads(response.choices[0].message.content)
            review["reviewed_at"] = datetime.utcnow().isoformat()
            review["pr_title"] = title

            # Persist review
            await self._save_review(pr_details, review)

            return review

        except Exception as e:
            logger.error(f"PR review failed: {e}")
            return {"error": str(e), "status": "error"}

    async def _save_analysis(self, results: Dict[str, Any]):
        """Save analysis results to database"""
        conn = self._get_db_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_code_analyses (tenant_id, repo_path, metrics, analyzed_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (tenant_id, repo_path) DO UPDATE SET
                    metrics = EXCLUDED.metrics,
                    analyzed_at = NOW()
            """, (self.tenant_id, results.get("repo_path", ""), json.dumps(results.get("metrics", {}))))
            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"Saved code analysis for {results.get('repo_path')}")
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")

    async def _save_review(self, pr_details: Dict, review: Dict):
        """Save PR review to database"""
        conn = self._get_db_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_pr_reviews (tenant_id, pr_id, pr_title, review_data, reviewed_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, (self.tenant_id, pr_details.get("id", ""), pr_details.get("title", ""), json.dumps(review)))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save PR review: {e}")

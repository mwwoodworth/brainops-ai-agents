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
        """Review a pull request using AI analysis with detailed suggestions"""
        try:
            client = get_openai_client()
            if not client:
                return {"error": "OpenAI client not available", "status": "pending"}

            # Extract PR info
            title = pr_details.get("title", "Unknown PR")
            description = pr_details.get("description", "")
            diff = pr_details.get("diff", "")[:6000]  # Increased limit
            files_changed = pr_details.get("files_changed", [])
            author = pr_details.get("author", "unknown")

            prompt = f"""You are an expert code reviewer. Provide a comprehensive pull request review.

Title: {title}
Author: {author}
Description: {description}
Files changed: {files_changed}
Diff (partial):
{diff}

Provide detailed analysis covering:
1. Code quality and best practices
2. Potential bugs or logic errors
3. Performance implications
4. Security vulnerabilities
5. Test coverage
6. Documentation needs
7. Specific improvement suggestions with code examples

Respond with JSON only:
{{
    "status": "approved" or "changes_requested" or "needs_discussion",
    "summary": "one sentence summary of the PR",
    "quality_score": 0-100,
    "comments": [
        {{
            "file": "filename",
            "line": number,
            "severity": "critical/high/medium/low",
            "category": "bug/performance/security/style/docs",
            "comment": "specific issue description",
            "suggestion": "how to fix it"
        }}
    ],
    "code_suggestions": [
        {{
            "file": "filename",
            "original_code": "current code",
            "suggested_code": "improved code",
            "explanation": "why this is better"
        }}
    ],
    "security_concerns": ["concern 1", "concern 2"],
    "performance_notes": ["note 1", "note 2"],
    "test_recommendations": ["test 1", "test 2"],
    "documentation_needs": ["doc 1", "doc 2"],
    "best_practices": ["practice 1", "practice 2"],
    "overall_assessment": "detailed paragraph",
    "confidence_score": 0-100
}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500
            )

            review = json.loads(response.choices[0].message.content)
            review["reviewed_at"] = datetime.utcnow().isoformat()
            review["pr_title"] = title
            review["pr_author"] = author

            # Persist review
            await self._save_review(pr_details, review)

            return review

        except Exception as e:
            logger.error(f"PR review failed: {e}")
            return {"error": str(e), "status": "error"}

    async def automated_code_review(self, file_path: str, code_content: str) -> Dict[str, Any]:
        """Perform automated code review with AI suggestions"""
        try:
            client = get_openai_client()
            if not client:
                return {"error": "OpenAI client not available"}

            results = {
                "file_path": file_path,
                "reviewed_at": datetime.utcnow().isoformat(),
                "issues": [],
                "suggestions": [],
                "metrics": {}
            }

            # Basic static analysis
            lines = code_content.split('\n')
            results["metrics"]["total_lines"] = len(lines)
            results["metrics"]["code_lines"] = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
            results["metrics"]["comment_lines"] = sum(1 for line in lines if line.strip().startswith('#'))

            # Check for common issues
            issues = []
            for i, line in enumerate(lines, 1):
                stripped = line.strip()

                # Check for long lines
                if len(line) > 120:
                    issues.append({
                        "line": i,
                        "severity": "low",
                        "category": "style",
                        "message": f"Line exceeds 120 characters ({len(line)} chars)"
                    })

                # Check for print statements (debugging code)
                if 'print(' in stripped and not stripped.startswith('#'):
                    issues.append({
                        "line": i,
                        "severity": "medium",
                        "category": "debug",
                        "message": "Found print statement (consider using logging)"
                    })

                # Check for TODO/FIXME
                if 'TODO' in stripped or 'FIXME' in stripped:
                    issues.append({
                        "line": i,
                        "severity": "low",
                        "category": "maintenance",
                        "message": "Unfinished task marker found"
                    })

                # Check for bare except
                if 'except:' in stripped and 'except Exception' not in stripped:
                    issues.append({
                        "line": i,
                        "severity": "high",
                        "category": "bug",
                        "message": "Bare except clause catches all exceptions"
                    })

            results["issues"] = issues
            results["metrics"]["issue_count"] = len(issues)

            # Use AI for deeper analysis (on a sample if file is large)
            code_sample = code_content[:4000] if len(code_content) > 4000 else code_content

            prompt = f"""Analyze this Python code and provide improvement suggestions:

File: {file_path}
Code:
{code_sample}

Provide analysis covering:
1. Code structure and organization
2. Potential bugs or edge cases
3. Performance optimizations
4. Security best practices
5. Pythonic improvements
6. Type hints and documentation
7. Error handling

Respond with JSON only:
{{
    "code_quality_grade": "A/B/C/D/F",
    "maintainability_score": 0-100,
    "complexity_assessment": "low/medium/high",
    "suggestions": [
        {{
            "category": "performance/security/style/bug/architecture",
            "priority": "high/medium/low",
            "description": "what to improve",
            "code_example": "example of better code",
            "impact": "expected impact"
        }}
    ],
    "refactoring_opportunities": ["opportunity 1", "opportunity 2"],
    "security_recommendations": ["rec 1", "rec 2"],
    "performance_tips": ["tip 1", "tip 2"],
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "confidence_score": 0-100
}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200
            )

            ai_analysis = json.loads(response.choices[0].message.content)
            results.update(ai_analysis)

            await self._save_code_review(file_path, results)
            return results

        except Exception as e:
            logger.error(f"Automated code review failed: {e}")
            return {"error": str(e)}

    async def suggest_refactoring(self, code_content: str, context: str = "") -> Dict[str, Any]:
        """Suggest refactoring improvements for code"""
        try:
            client = get_openai_client()
            if not client:
                return {"error": "OpenAI client not available"}

            prompt = f"""Suggest refactoring improvements for this code:

Context: {context}
Code:
{code_content[:3000]}

Provide specific refactoring suggestions with before/after examples.

Respond with JSON only:
{{
    "refactorings": [
        {{
            "type": "extract_method/rename/simplify/optimize",
            "priority": "high/medium/low",
            "description": "what to refactor and why",
            "before": "original code snippet",
            "after": "refactored code snippet",
            "benefits": ["benefit 1", "benefit 2"]
        }}
    ],
    "estimated_effort": "X hours",
    "complexity_reduction": "X%",
    "readability_improvement": "high/medium/low",
    "confidence_score": 0-100
}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )

            suggestions = json.loads(response.choices[0].message.content)
            suggestions["suggested_at"] = datetime.utcnow().isoformat()

            return suggestions

        except Exception as e:
            logger.error(f"Refactoring suggestions failed: {e}")
            return {"error": str(e)}

    async def _save_code_review(self, file_path: str, results: Dict[str, Any]):
        """Save code review results to database"""
        conn = self._get_db_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_code_reviews (tenant_id, file_path, review_data, reviewed_at)
                VALUES (%s, %s, %s, NOW())
            """, (self.tenant_id, file_path, json.dumps(results)))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save code review: {e}")
            if conn:
                conn.close()

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

"""
TRUE AI INTELLIGENCE ENGINE
===========================

This replaces hardcoded pattern matching with REAL AI-powered analysis.
Uses actual LLM calls to understand issues, not regex patterns.

NO MORE FAKE INTELLIGENCE.

Created: 2026-01-27
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AnalysisDepth(Enum):
    """How deep should the AI analyze?"""
    QUICK = "quick"        # Fast pattern match + quick AI check
    STANDARD = "standard"  # Full AI analysis
    DEEP = "deep"          # Multi-model consensus with reasoning


@dataclass
class AIAnalysisResult:
    """Result from true AI analysis"""
    issue: str
    root_cause: str
    confidence: float
    severity: str  # critical, high, medium, low
    fix_strategies: List[Dict[str, Any]]
    auto_fixable: bool
    reasoning: str
    model_used: str
    analysis_depth: AnalysisDepth
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TrueAIIntelligence:
    """
    TRUE AI INTELLIGENCE - Not hardcoded patterns

    Uses actual LLM calls to:
    1. Understand what's wrong (not pattern match)
    2. Reason about root causes
    3. Suggest intelligent fixes
    4. Learn from outcomes
    """

    def __init__(self):
        self.ai_core = None
        self.analysis_history: List[AIAnalysisResult] = []
        self.learning_data: List[Dict[str, Any]] = []
        self._init_ai_core()

    def _init_ai_core(self):
        """Initialize the AI core for real intelligence"""
        try:
            from ai_core import AICore
            self.ai_core = AICore()
            logger.info("✅ True AI Intelligence initialized with AICore")
        except ImportError:
            logger.warning("⚠️ AICore not available, will use fallback")
            self.ai_core = None

    async def analyze_issue(
        self,
        issue_description: str,
        context: Dict[str, Any] = None,
        depth: AnalysisDepth = AnalysisDepth.STANDARD
    ) -> AIAnalysisResult:
        """
        ANALYZE AN ISSUE WITH REAL AI

        Not pattern matching - actual intelligent analysis.
        """
        context = context or {}

        # Build the analysis prompt
        prompt = self._build_analysis_prompt(issue_description, context)

        # Get AI analysis
        if self.ai_core:
            try:
                response = await self._get_ai_analysis(prompt, depth)
                result = self._parse_ai_response(response, issue_description, depth)
                self.analysis_history.append(result)
                return result
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
                # Fall back to heuristic analysis
                return self._heuristic_analysis(issue_description, context)
        else:
            return self._heuristic_analysis(issue_description, context)

    def _build_analysis_prompt(self, issue: str, context: Dict[str, Any]) -> str:
        """Build a comprehensive prompt for AI analysis"""

        # Include historical context for learning
        recent_similar = self._find_similar_past_issues(issue)

        prompt = f"""You are an expert DevOps and infrastructure AI analyzing a system issue.

ISSUE TO ANALYZE:
{issue}

SYSTEM CONTEXT:
- System: BrainOps AI Operating System
- Services: Backend API, AI Agents, MCP Bridge, Frontend Apps
- Infrastructure: Render (Backend), Vercel (Frontend), Supabase (Database)
{json.dumps(context, indent=2) if context else "No additional context provided."}

{"SIMILAR PAST ISSUES AND OUTCOMES:" + chr(10) + json.dumps(recent_similar, indent=2) if recent_similar else ""}

ANALYZE THIS ISSUE AND PROVIDE:
1. ROOT CAUSE: What is the actual underlying cause of this issue?
2. SEVERITY: critical/high/medium/low - with reasoning
3. FIX STRATEGIES: List 3-5 specific actions to fix this, ordered by likelihood of success
4. AUTO-FIXABLE: Can this be automatically fixed via API? (restart, redeploy, scale)
5. REASONING: Explain your analysis step by step

Respond in this exact JSON format:
{{
    "root_cause": "specific description of the root cause",
    "severity": "critical|high|medium|low",
    "severity_reasoning": "why this severity level",
    "fix_strategies": [
        {{"action": "action_name", "description": "what to do", "confidence": 0.0-1.0, "auto": true/false}},
        ...
    ],
    "auto_fixable": true/false,
    "reasoning": "step by step analysis"
}}"""

        return prompt

    async def _get_ai_analysis(self, prompt: str, depth: AnalysisDepth) -> str:
        """Get actual AI analysis using available models"""

        if depth == AnalysisDepth.QUICK:
            # Use fastest model
            return await self._call_ai(prompt, model="fast")

        elif depth == AnalysisDepth.STANDARD:
            # Use standard model
            return await self._call_ai(prompt, model="standard")

        elif depth == AnalysisDepth.DEEP:
            # Multi-model consensus
            responses = await asyncio.gather(
                self._call_ai(prompt, model="anthropic"),
                self._call_ai(prompt, model="openai"),
                self._call_ai(prompt, model="gemini"),
                return_exceptions=True
            )

            # Filter out errors and get consensus
            valid_responses = [r for r in responses if isinstance(r, str)]
            if valid_responses:
                return await self._build_consensus(valid_responses, prompt)
            raise Exception("All AI models failed")

    async def _call_ai(self, prompt: str, model: str = "standard") -> str:
        """Call an AI model for analysis"""

        if self.ai_core:
            try:
                # Try to use AICore's smart routing
                if model == "anthropic":
                    response = await self.ai_core.call_anthropic(prompt)
                elif model == "openai":
                    response = await self.ai_core.call_openai(prompt)
                elif model == "gemini":
                    response = await self.ai_core.call_gemini(prompt)
                else:
                    # Use smart routing
                    response = await self.ai_core.generate(prompt)

                return response
            except Exception as e:
                logger.warning(f"AI call failed ({model}): {e}")

        # Fallback: try direct API calls
        return await self._direct_api_call(prompt, model)

    async def _direct_api_call(self, prompt: str, model: str) -> str:
        """Direct API call as fallback - tries multiple providers"""
        import httpx

        errors = []

        # Try Anthropic first (best for analysis)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": anthropic_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json"
                        },
                        json={
                            "model": "claude-3-haiku-20240307",
                            "max_tokens": 2000,
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return data["content"][0]["text"]
                    errors.append(f"Anthropic: HTTP {response.status_code}")
            except Exception as e:
                errors.append(f"Anthropic: {e}")
                logger.warning(f"Anthropic direct call failed: {e}")

        # Try OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openai_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 2000
                        },
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return data["choices"][0]["message"]["content"]
                    errors.append(f"OpenAI: HTTP {response.status_code}")
            except Exception as e:
                errors.append(f"OpenAI: {e}")
                logger.warning(f"OpenAI direct call failed: {e}")

        # Try Gemini
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}",
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"maxOutputTokens": 2000}
                        },
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [])
                            if parts:
                                return parts[0].get("text", "")
                    errors.append(f"Gemini: HTTP {response.status_code}")
            except Exception as e:
                errors.append(f"Gemini: {e}")
                logger.warning(f"Gemini direct call failed: {e}")

        # Try HuggingFace as last resort (try multiple models)
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
        hf_models = [
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/DialoGPT-large",
            "gpt2"  # Always available
        ]
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

        for hf_model in hf_models:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"https://api-inference.huggingface.co/models/{hf_model}",
                        headers=headers,
                        json={
                            "inputs": prompt[:2000],  # Limit prompt size for smaller models
                            "parameters": {
                                "max_new_tokens": 500,
                                "temperature": 0.7,
                                "return_full_text": False
                            }
                        },
                        timeout=15.0
                    )
                    if response.status_code == 200:
                        result = response.json()
                        text = ""
                        if isinstance(result, list) and result:
                            text = result[0].get("generated_text", "")
                        elif isinstance(result, dict):
                            text = result.get("generated_text", "")
                        if text:
                            logger.info(f"HuggingFace success with {hf_model}")
                            return text
            except Exception as e:
                logger.debug(f"HuggingFace {hf_model} failed: {e}")
                continue

        errors.append("HuggingFace: All models failed")

        raise Exception(f"No AI API available. Tried: {'; '.join(errors)}")

    async def _build_consensus(self, responses: List[str], original_prompt: str) -> str:
        """Build consensus from multiple AI responses"""

        consensus_prompt = f"""You are analyzing multiple AI responses to the same infrastructure issue.
Build a consensus analysis that combines the best insights from each.

ORIGINAL QUESTION:
{original_prompt}

AI RESPONSES:
{chr(10).join([f"Response {i+1}:{chr(10)}{r}" for i, r in enumerate(responses)])}

Create a SINGLE consensus response in the same JSON format that:
1. Takes the most confident root cause analysis
2. Combines the best fix strategies from all responses
3. Uses the most conservative (highest) severity if they differ
4. Synthesizes the reasoning

Return ONLY the final JSON response."""

        return await self._call_ai(consensus_prompt, model="standard")

    def _parse_ai_response(
        self,
        response: str,
        original_issue: str,
        depth: AnalysisDepth
    ) -> AIAnalysisResult:
        """Parse AI response into structured result"""

        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                return AIAnalysisResult(
                    issue=original_issue,
                    root_cause=data.get("root_cause", "Unknown"),
                    confidence=0.85 if depth == AnalysisDepth.DEEP else 0.7,
                    severity=data.get("severity", "medium"),
                    fix_strategies=data.get("fix_strategies", []),
                    auto_fixable=data.get("auto_fixable", False),
                    reasoning=data.get("reasoning", ""),
                    model_used="multi-model" if depth == AnalysisDepth.DEEP else "standard",
                    analysis_depth=depth
                )
        except json.JSONDecodeError:
            pass

        # Fallback: extract what we can
        return AIAnalysisResult(
            issue=original_issue,
            root_cause=response[:200] if response else "Analysis failed",
            confidence=0.5,
            severity="medium",
            fix_strategies=[{"action": "investigate", "description": "Manual investigation needed", "confidence": 0.5, "auto": False}],
            auto_fixable=False,
            reasoning=response,
            model_used="standard",
            analysis_depth=depth
        )

    def _heuristic_analysis(self, issue: str, context: Dict[str, Any]) -> AIAnalysisResult:
        """Enhanced heuristic analysis when AI is not available"""

        issue_lower = issue.lower()

        # 500 Internal Server Error
        if "500" in issue or "internal server error" in issue_lower:
            return AIAnalysisResult(
                issue=issue,
                root_cause="Internal server error - application crash or unhandled exception",
                confidence=0.7,
                severity="high",
                fix_strategies=[
                    {"action": "check_logs", "description": "Check application logs for stack trace", "confidence": 0.9, "auto": False},
                    {"action": "restart", "description": "Restart the service", "confidence": 0.7, "auto": True},
                    {"action": "check_memory", "description": "Check memory usage", "confidence": 0.6, "auto": False}
                ],
                auto_fixable=True,
                reasoning="HTTP 500 indicates unhandled exception. Check logs for root cause.",
                model_used="heuristic",
                analysis_depth=AnalysisDepth.QUICK
            )

        # Timeout issues
        if "timeout" in issue_lower:
            return AIAnalysisResult(
                issue=issue,
                root_cause="Request timeout - slow response from service or database",
                confidence=0.7,
                severity="medium",
                fix_strategies=[
                    {"action": "check_db_queries", "description": "Check for slow database queries", "confidence": 0.8, "auto": False},
                    {"action": "check_network", "description": "Check network latency", "confidence": 0.7, "auto": False},
                    {"action": "scale_up", "description": "Consider scaling up resources", "confidence": 0.5, "auto": False}
                ],
                auto_fixable=False,
                reasoning="Timeouts typically indicate slow queries, network issues, or insufficient resources.",
                model_used="heuristic",
                analysis_depth=AnalysisDepth.QUICK
            )

        # Database issues
        if any(kw in issue_lower for kw in ["database", "db", "postgres", "sql", "query", "connection pool"]):
            return AIAnalysisResult(
                issue=issue,
                root_cause="Database connectivity or performance issue",
                confidence=0.7,
                severity="high",
                fix_strategies=[
                    {"action": "check_connections", "description": "Check database connection pool usage", "confidence": 0.8, "auto": False},
                    {"action": "check_slow_queries", "description": "Identify slow queries", "confidence": 0.9, "auto": False},
                    {"action": "restart_pool", "description": "Restart connection pool", "confidence": 0.6, "auto": True}
                ],
                auto_fixable=True,
                reasoning="Database issues affect all services. Check connection pool and query performance.",
                model_used="heuristic",
                analysis_depth=AnalysisDepth.QUICK
            )

        # Memory issues
        if any(kw in issue_lower for kw in ["memory", "oom", "out of memory", "heap"]):
            return AIAnalysisResult(
                issue=issue,
                root_cause="Memory pressure or leak detected",
                confidence=0.8,
                severity="critical",
                fix_strategies=[
                    {"action": "restart", "description": "Restart service to reclaim memory", "confidence": 0.9, "auto": True},
                    {"action": "profile_memory", "description": "Profile memory usage to find leak", "confidence": 0.7, "auto": False},
                    {"action": "increase_limits", "description": "Increase memory limits if available", "confidence": 0.5, "auto": False}
                ],
                auto_fixable=True,
                reasoning="Memory issues require immediate restart, then investigation for leaks.",
                model_used="heuristic",
                analysis_depth=AnalysisDepth.QUICK
            )

        # Performance/slow issues
        if any(kw in issue_lower for kw in ["slow", "performance", "degradation", "latency", "lag"]):
            return AIAnalysisResult(
                issue=issue,
                root_cause="Service performance degradation",
                confidence=0.6,
                severity="medium",
                fix_strategies=[
                    {"action": "check_cpu", "description": "Check CPU utilization", "confidence": 0.7, "auto": False},
                    {"action": "check_db", "description": "Check database performance", "confidence": 0.8, "auto": False},
                    {"action": "check_external", "description": "Check external API dependencies", "confidence": 0.6, "auto": False},
                    {"action": "scale", "description": "Consider horizontal scaling", "confidence": 0.5, "auto": False}
                ],
                auto_fixable=False,
                reasoning="Performance issues require profiling to identify bottleneck.",
                model_used="heuristic",
                analysis_depth=AnalysisDepth.QUICK
            )

        # Authentication/authorization issues
        if any(kw in issue_lower for kw in ["401", "403", "auth", "unauthorized", "forbidden", "token", "api key"]):
            return AIAnalysisResult(
                issue=issue,
                root_cause="Authentication or authorization failure",
                confidence=0.8,
                severity="high",
                fix_strategies=[
                    {"action": "check_credentials", "description": "Verify API keys/tokens are valid", "confidence": 0.9, "auto": False},
                    {"action": "check_expiry", "description": "Check if tokens have expired", "confidence": 0.8, "auto": False},
                    {"action": "rotate_keys", "description": "Rotate credentials if compromised", "confidence": 0.5, "auto": False}
                ],
                auto_fixable=False,
                reasoning="Auth issues often indicate expired or invalid credentials.",
                model_used="heuristic",
                analysis_depth=AnalysisDepth.QUICK
            )

        # Deployment/crash issues
        if any(kw in issue_lower for kw in ["crash", "restart", "deploy", "failed", "unhealthy"]):
            return AIAnalysisResult(
                issue=issue,
                root_cause="Service crash or deployment failure",
                confidence=0.7,
                severity="high",
                fix_strategies=[
                    {"action": "check_logs", "description": "Review deployment/crash logs", "confidence": 0.9, "auto": False},
                    {"action": "rollback", "description": "Rollback to previous version", "confidence": 0.7, "auto": False},
                    {"action": "restart", "description": "Force restart the service", "confidence": 0.6, "auto": True}
                ],
                auto_fixable=True,
                reasoning="Crashes require log analysis to identify root cause.",
                model_used="heuristic",
                analysis_depth=AnalysisDepth.QUICK
            )

        # Quota/rate limit issues
        if any(kw in issue_lower for kw in ["quota", "rate limit", "429", "too many requests"]):
            return AIAnalysisResult(
                issue=issue,
                root_cause="Rate limit or quota exceeded",
                confidence=0.9,
                severity="medium",
                fix_strategies=[
                    {"action": "wait", "description": "Wait for rate limit to reset", "confidence": 0.9, "auto": True},
                    {"action": "add_credits", "description": "Add credits or upgrade plan", "confidence": 0.8, "auto": False},
                    {"action": "implement_backoff", "description": "Implement exponential backoff", "confidence": 0.7, "auto": False}
                ],
                auto_fixable=True,
                reasoning="Rate limits require waiting or plan upgrades.",
                model_used="heuristic",
                analysis_depth=AnalysisDepth.QUICK
            )

        # Generic fallback with better guidance
        return AIAnalysisResult(
            issue=issue,
            root_cause="Issue requires investigation - no specific pattern matched",
            confidence=0.4,
            severity="medium",
            fix_strategies=[
                {"action": "check_logs", "description": "Review application and system logs", "confidence": 0.8, "auto": False},
                {"action": "check_metrics", "description": "Review system metrics (CPU, memory, network)", "confidence": 0.7, "auto": False},
                {"action": "investigate", "description": "Manual investigation needed", "confidence": 1.0, "auto": False}
            ],
            auto_fixable=False,
            reasoning="AI providers unavailable - using pattern-based analysis. Check logs for more details.",
            model_used="heuristic",
            analysis_depth=AnalysisDepth.QUICK
        )

    def _find_similar_past_issues(self, issue: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find similar issues from history for learning"""

        # Simple keyword matching for now
        # TODO: Use embeddings for semantic similarity
        issue_words = set(issue.lower().split())

        similar = []
        for past in self.analysis_history[-100:]:  # Last 100
            past_words = set(past.issue.lower().split())
            overlap = len(issue_words & past_words)
            if overlap > 2:  # At least 3 words in common
                similar.append({
                    "issue": past.issue,
                    "root_cause": past.root_cause,
                    "fix_applied": past.fix_strategies[0] if past.fix_strategies else None,
                    "severity": past.severity
                })

        return similar[:limit]

    async def analyze_system_health(self, systems: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall system health with AI"""

        # Build context about all systems
        system_summary = []
        for sys_id, sys_data in systems.items():
            system_summary.append(f"- {sys_data.get('name', sys_id)}: health={sys_data.get('health_score', 'unknown')}, state={sys_data.get('state', 'unknown')}, issues={sys_data.get('issues', [])}")

        prompt = f"""Analyze the overall health of this AI Operating System and provide insights.

SYSTEM STATUS:
{chr(10).join(system_summary)}

Provide:
1. Overall health assessment
2. Critical concerns (if any)
3. Proactive recommendations
4. Risk assessment for the next 24 hours

Respond in JSON format:
{{
    "overall_health": "excellent|good|concerning|critical",
    "health_score": 0.0-1.0,
    "critical_concerns": ["list of concerns"],
    "recommendations": ["list of proactive recommendations"],
    "risk_level": "low|medium|high",
    "risk_reasoning": "explanation"
}}"""

        try:
            response = await self._call_ai(prompt, model="standard")
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0:
                return json.loads(response[json_start:json_end])
        except Exception as e:
            logger.error(f"System health analysis failed: {e}")

        # Fallback
        healthy_count = sum(1 for s in systems.values() if s.get("health_score", 0) >= 0.8)
        total = len(systems)

        return {
            "overall_health": "good" if healthy_count == total else "concerning",
            "health_score": healthy_count / total if total > 0 else 0,
            "critical_concerns": [],
            "recommendations": ["Continue monitoring"],
            "risk_level": "low" if healthy_count == total else "medium",
            "risk_reasoning": "AI analysis not available - using basic metrics"
        }

    async def generate_fix_plan(self, issues: List[str]) -> Dict[str, Any]:
        """Generate a comprehensive fix plan for multiple issues"""

        if not issues:
            return {"plan": [], "message": "No issues to fix"}

        prompt = f"""You are creating an automated fix plan for an AI Operating System.

ISSUES TO FIX:
{chr(10).join([f"- {issue}" for issue in issues])}

Create an ORDERED fix plan that:
1. Prioritizes by severity and dependency
2. Groups related fixes together
3. Includes rollback steps
4. Estimates risk for each step

Respond in JSON format:
{{
    "plan": [
        {{
            "step": 1,
            "action": "action to take",
            "target": "system/service affected",
            "auto_executable": true/false,
            "risk": "low|medium|high",
            "rollback": "how to rollback if needed",
            "depends_on": [list of step numbers this depends on]
        }}
    ],
    "total_estimated_risk": "low|medium|high",
    "recommended_approach": "automatic|supervised|manual"
}}"""

        try:
            response = await self._call_ai(prompt, model="standard")
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0:
                return json.loads(response[json_start:json_end])
        except Exception as e:
            logger.error(f"Fix plan generation failed: {e}")

        # Fallback
        return {
            "plan": [{"step": 1, "action": "investigate", "target": "all", "auto_executable": False, "risk": "low", "rollback": "N/A", "depends_on": []}],
            "total_estimated_risk": "medium",
            "recommended_approach": "manual"
        }

    def record_fix_outcome(self, issue: str, fix_applied: str, success: bool, details: str = ""):
        """Record the outcome of a fix for learning"""
        self.learning_data.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "issue": issue,
            "fix_applied": fix_applied,
            "success": success,
            "details": details
        })

        logger.info(f"📚 Learning recorded: {fix_applied} for '{issue[:50]}...' - {'SUCCESS' if success else 'FAILED'}")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about what the AI has learned"""

        if not self.learning_data:
            return {"total_learnings": 0, "success_rate": 0}

        total = len(self.learning_data)
        successes = sum(1 for l in self.learning_data if l["success"])

        # Group by fix type
        fix_stats = {}
        for l in self.learning_data:
            fix = l["fix_applied"]
            if fix not in fix_stats:
                fix_stats[fix] = {"total": 0, "success": 0}
            fix_stats[fix]["total"] += 1
            if l["success"]:
                fix_stats[fix]["success"] += 1

        return {
            "total_learnings": total,
            "success_rate": successes / total if total > 0 else 0,
            "by_fix_type": {
                k: {"total": v["total"], "success_rate": v["success"] / v["total"] if v["total"] > 0 else 0}
                for k, v in fix_stats.items()
            },
            "recent_learnings": self.learning_data[-5:]
        }


# Global instance
_true_ai_intelligence: Optional[TrueAIIntelligence] = None


def get_ai_intelligence() -> TrueAIIntelligence:
    """Get the global AI intelligence instance"""
    global _true_ai_intelligence
    if _true_ai_intelligence is None:
        _true_ai_intelligence = TrueAIIntelligence()
    return _true_ai_intelligence


async def analyze_with_ai(issue: str, context: Dict[str, Any] = None, depth: str = "standard") -> Dict[str, Any]:
    """Convenience function to analyze an issue with AI"""
    ai = get_ai_intelligence()
    depth_enum = AnalysisDepth(depth)
    result = await ai.analyze_issue(issue, context, depth_enum)

    return {
        "issue": result.issue,
        "root_cause": result.root_cause,
        "confidence": result.confidence,
        "severity": result.severity,
        "fix_strategies": result.fix_strategies,
        "auto_fixable": result.auto_fixable,
        "reasoning": result.reasoning,
        "model_used": result.model_used,
        "analysis_depth": result.analysis_depth.value,
        "timestamp": result.timestamp.isoformat()
    }

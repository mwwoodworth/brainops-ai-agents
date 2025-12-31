"""
Competitive Intelligence Agent
AI agent for monitoring competitors and market trends.
Uses OpenAI for real analysis and persists results to database.
"""

import os
import json
import logging
import psycopg2
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

class CompetitiveIntelligenceAgent:
    """AI-powered competitive intelligence analysis agent with AUREA integration"""

    def __init__(self, tenant_id: str):
        if not tenant_id:
            raise ValueError("tenant_id is required for CompetitiveIntelligenceAgent")
        self.tenant_id = tenant_id
        self.agent_type = "competitive_intelligence"

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

    async def monitor_competitors(self, competitors: List[str]) -> Dict[str, Any]:
        """Monitor competitor activities using AI analysis"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "competitors_tracked": len(competitors),
                "competitors": competitors,
                "analyses": []
            }

            client = get_openai_client()
            if client and competitors:
                try:
                    prompt = f"""Analyze competitive landscape for these competitors: {', '.join(competitors)}

For each competitor, provide strategic intelligence:
1. Market positioning
2. Strengths and weaknesses
3. Recent market moves or trends
4. Threat level to our business

Respond with JSON only:
{{
    "market_overview": "brief market state",
    "competitor_analyses": [
        {{"name": "competitor", "positioning": "...", "strengths": [...], "weaknesses": [...], "threat_level": "low/medium/high"}}
    ],
    "market_shifts": "stable/growing/declining/volatile",
    "strategic_recommendations": ["recommendation1", "recommendation2"]
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI analysis failed: {e}")
                    # Fallback analysis
                    results["market_shifts"] = "stable"
                    results["strategic_recommendations"] = ["Continue monitoring", "Gather more data"]
            else:
                results["market_shifts"] = "stable"
                results["note"] = "AI analysis unavailable - using default assessment"

            # Persist results
            await self._save_analysis("competitor_monitoring", results)
            return results

        except Exception as e:
            logger.error(f"Competitor monitoring failed: {e}")
            return {"error": str(e), "competitors_tracked": 0}

    async def analyze_pricing(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pricing positioning using AI"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "market_data_received": bool(market_data)
            }

            client = get_openai_client()
            if client and market_data:
                try:
                    prompt = f"""Analyze pricing strategy based on this market data:
{json.dumps(market_data, indent=2)}

Provide strategic pricing analysis:
1. Current market position
2. Price elasticity assessment
3. Competitive price gaps
4. Recommended pricing strategy

Respond with JSON only:
{{
    "position": "budget/value/premium/luxury",
    "price_gap_analysis": {{"vs_market_avg": "X%", "vs_premium": "X%", "vs_budget": "X%"}},
    "elasticity": "low/medium/high",
    "recommendation": "maintain/increase/decrease/restructure",
    "confidence_score": 0-100,
    "rationale": "explanation",
    "suggested_actions": ["action1", "action2"]
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=800
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI pricing analysis failed: {e}")
                    results["position"] = "value"
                    results["recommendation"] = "maintain"
                    results["confidence_score"] = 50
            else:
                results["position"] = "value"
                results["recommendation"] = "gather_more_data"
                results["note"] = "Insufficient data for AI analysis"

            await self._save_analysis("pricing_analysis", results)
            return results

        except Exception as e:
            logger.error(f"Pricing analysis failed: {e}")
            return {"error": str(e)}

    async def analyze_market_trends(self, industry: str, timeframe: str = "quarterly") -> Dict[str, Any]:
        """Analyze market trends for an industry"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "industry": industry,
                "timeframe": timeframe
            }

            client = get_openai_client()
            if client:
                try:
                    prompt = f"""Analyze market trends for the {industry} industry over the {timeframe} timeframe.

Provide comprehensive market intelligence:
1. Key trends
2. Growth drivers
3. Risk factors
4. Opportunity areas

Respond with JSON only:
{{
    "trend_direction": "growing/stable/declining",
    "growth_rate_estimate": "X%",
    "key_trends": ["trend1", "trend2", "trend3"],
    "growth_drivers": ["driver1", "driver2"],
    "risk_factors": ["risk1", "risk2"],
    "opportunities": ["opportunity1", "opportunity2"],
    "market_outlook": "positive/neutral/cautious/negative",
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
                    logger.warning(f"AI market analysis failed: {e}")
                    results["trend_direction"] = "stable"
                    results["market_outlook"] = "neutral"

            await self._save_analysis("market_trends", results)
            return results

        except Exception as e:
            logger.error(f"Market trend analysis failed: {e}")
            return {"error": str(e)}

    async def benchmark_competitors(
        self,
        competitors: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive competitive benchmarking across multiple metrics

        Args:
            competitors: List of competitor data with metrics
            metrics: Metrics to benchmark (default: all available)

        Returns:
            Benchmarking analysis with rankings and gaps
        """
        if not metrics:
            metrics = ["market_share", "pricing", "features", "customer_satisfaction", "growth_rate"]

        try:
            results = {
                "benchmarked_at": datetime.utcnow().isoformat(),
                "competitors_analyzed": len(competitors),
                "metrics_analyzed": metrics,
                "benchmarks": {},
                "rankings": {},
                "gaps": {},
                "opportunities": []
            }

            # Calculate benchmarks for each metric
            for metric in metrics:
                metric_values = []
                competitor_scores = {}

                for comp in competitors:
                    value = comp.get(metric, 0)
                    metric_values.append(float(value))
                    competitor_scores[comp.get("name", "Unknown")] = float(value)

                if metric_values:
                    results["benchmarks"][metric] = {
                        "average": float(sum(metric_values) / len(metric_values)),
                        "max": float(max(metric_values)),
                        "min": float(min(metric_values)),
                        "median": float(sorted(metric_values)[len(metric_values) // 2]),
                        "top_performer": max(competitor_scores, key=competitor_scores.get),
                        "scores": competitor_scores
                    }

                    # Calculate rankings
                    ranked = sorted(competitor_scores.items(), key=lambda x: x[1], reverse=True)
                    results["rankings"][metric] = [
                        {"rank": i + 1, "name": name, "value": value}
                        for i, (name, value) in enumerate(ranked)
                    ]

            # Identify gaps and opportunities
            client = get_openai_client()
            if client:
                try:
                    prompt = f"""Analyze this competitive benchmark data and identify:
1. Key competitive gaps
2. Market opportunities
3. Strategic recommendations

Benchmark Data:
{json.dumps(results["benchmarks"], indent=2)}

Respond with JSON only:
{{
    "competitive_gaps": ["gap1", "gap2", "gap3"],
    "opportunities": [
        {{"type": "market", "description": "...", "priority": "high/medium/low"}},
        {{"type": "product", "description": "...", "priority": "high/medium/low"}}
    ],
    "strategic_recommendations": ["rec1", "rec2", "rec3"],
    "overall_competitive_position": "leader/strong/average/weak"
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)

                except Exception as e:
                    logger.warning(f"AI benchmarking analysis failed: {e}")
                    results["opportunities"] = [
                        {
                            "type": "analysis",
                            "description": "Manual analysis required",
                            "priority": "medium"
                        }
                    ]

            await self._save_analysis("competitive_benchmarking", results)
            return results

        except Exception as e:
            logger.error(f"Competitive benchmarking failed: {e}")
            return {"error": str(e)}

    async def calculate_competitive_score(
        self,
        our_metrics: Dict[str, float],
        competitor_metrics: List[Dict[str, float]],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate competitive score based on weighted metrics

        Args:
            our_metrics: Our company metrics
            competitor_metrics: List of competitor metrics
            weights: Optional weights for each metric (default: equal weights)

        Returns:
            Competitive score and analysis
        """
        if not weights:
            # Default equal weights
            metrics = list(our_metrics.keys())
            weights = {m: 1.0 / len(metrics) for m in metrics}

        try:
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            normalized_weights = {k: v / total_weight for k, v in weights.items()}

            # Calculate our score
            our_score = 0
            metric_scores = {}

            for metric, weight in normalized_weights.items():
                our_value = our_metrics.get(metric, 0)

                # Get competitor values for this metric
                comp_values = [c.get(metric, 0) for c in competitor_metrics]

                if comp_values:
                    # Normalize: 0-100 scale based on min-max
                    all_values = comp_values + [our_value]
                    min_val = min(all_values)
                    max_val = max(all_values)

                    if max_val > min_val:
                        normalized_score = ((our_value - min_val) / (max_val - min_val)) * 100
                    else:
                        normalized_score = 50  # Neutral if all equal

                    weighted_score = normalized_score * weight
                    our_score += weighted_score

                    metric_scores[metric] = {
                        "our_value": our_value,
                        "market_min": min_val,
                        "market_max": max_val,
                        "market_avg": sum(comp_values) / len(comp_values),
                        "normalized_score": normalized_score,
                        "weighted_contribution": weighted_score,
                        "percentile": sum(1 for v in comp_values if v < our_value) / len(comp_values) * 100
                    }

            # Determine competitive position
            if our_score >= 80:
                position = "market_leader"
            elif our_score >= 60:
                position = "strong_competitor"
            elif our_score >= 40:
                position = "average_competitor"
            else:
                position = "challenger"

            return {
                "overall_score": round(our_score, 2),
                "competitive_position": position,
                "metric_breakdown": metric_scores,
                "weights_used": normalized_weights,
                "strengths": [
                    metric for metric, data in metric_scores.items()
                    if data["percentile"] >= 75
                ],
                "weaknesses": [
                    metric for metric, data in metric_scores.items()
                    if data["percentile"] <= 25
                ],
                "calculated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Competitive score calculation failed: {e}")
            return {"error": str(e)}

    async def _save_analysis(self, analysis_type: str, results: Dict[str, Any]):
        """Save analysis results to database"""
        conn = self._get_db_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_competitive_analyses (tenant_id, analysis_type, results, analyzed_at)
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

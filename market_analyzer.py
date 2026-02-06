#!/usr/bin/env python3
"""
Market Analyzer Agent
=====================
Provides real-time market pricing analysis, competitor tracking, and trend forecasting
for the roofing and construction industry.

Features:
- Local material pricing analysis
- Labor rate benchmarking
- Competitor price estimation
- Market trend analysis

Part of BrainOps AI OS.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from agent_executor import BaseAgent, ai_core

logger = logging.getLogger(__name__)

class MarketAnalyzerAgent(BaseAgent):
    """
    AI Agent for analyzing market conditions, pricing, and competitors.
    """

    def __init__(self):
        super().__init__("MarketAnalyzer", "market-analyzer")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market analysis tasks."""
        action = task.get('action')

        if action == 'get_current_pricing':
            return await self.get_current_pricing(task)
        elif action == 'analyze_competitor':
            return await self.analyze_competitor(task)
        elif action == 'get_market_trends':
            return await self.get_market_trends(task)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def get_current_pricing(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market pricing for materials and labor in a specific location.
        """
        location = task.get('location', 'Unknown Location')
        materials = task.get('materials', [])
        labor_categories = task.get('labor_categories', [])
        
        # Use AI to estimate local pricing based on its knowledge base
        # In a fully connected system, this would also query vendor APIs (ABC Supply, SRS, etc.)
        
        prompt = f"""
        You are an expert construction estimator and market analyst.
        Analyze the current market pricing for the following items in {location}.
        
        Materials: {', '.join(materials) if materials else 'Standard roofing materials'}
        Labor Categories: {', '.join(labor_categories) if labor_categories else 'Roofing labor'}
        
        Provide a JSON response with the following structure:
        {{
            "local_rates": {{ "Material Name": price_per_unit_float }},
            "labor_rates": {{ "Category": hourly_rate_float }},
            "average_competitor_price": estimated_total_sqft_price_float,
            "trends": [
                {{
                    "material": "Material Name",
                    "trend": "increasing" | "stable" | "decreasing",
                    "forecast_change": percentage_float,
                    "recommendation": "buy_now" | "wait" | "bulk_order"
                }}
            ],
            "competitor_analysis": {{
                "estimated_competitor_price": price_per_sq_float,
                "our_advantage": ["point 1", "point 2"],
                "their_advantage": ["point 1", "point 2"]
            }}
        }}
        
        Ensure prices are realistic for the current US market (2025/2026).
        """
        
        try:
            response_text = await ai_core.generate(
                prompt, 
                model="gpt-4", 
                temperature=0.2,
                json_mode=True
            )
            
            # Parse the JSON response
            # ai_core.generate usually returns a string, we need to ensure it's valid JSON
            # The BaseAgent helper might handle this, but let's be safe
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from markdown code blocks
                import re
                match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if match:
                    data = json.loads(match.group(1))
                else:
                    match = re.search(r'(\{.*?\})', response_text, re.DOTALL)
                    if match:
                        data = json.loads(match.group(1))
                    else:
                        raise ValueError("Could not parse AI response as JSON")

            # Enrich with metadata
            data['timestamp'] = datetime.now(timezone.utc).isoformat()
            data['location'] = location
            data['required_materials'] = materials
            data['labor_categories'] = labor_categories
            data['status'] = 'success'
            
            return data

        except Exception as e:
            logger.error(f"Market pricing analysis failed: {e}")
            # Return safe fallbacks if AI fails
            return {
                "status": "partial_success",
                "error": str(e),
                "local_rates": {},
                "labor_rates": {"General": 65.0, "Roofing": 75.0},
                "average_competitor_price": 450.0, # per square
                "trends": [],
                "competitor_analysis": None
            }

    async def analyze_competitor(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific competitor."""
        competitor_name = task.get('competitor_name')
        location = task.get('location')
        
        prompt = f"""
        Analyze the roofing competitor "{competitor_name}" in {location}.
        Identify their likely pricing strategy, strengths, weaknesses, and estimated market share.
        Return JSON.
        """
        
        try:
            response = await ai_core.generate(prompt, model="gpt-4", json_mode=True)
            return json.loads(response)
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_market_trends(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get general market trends for a region."""
        location = task.get('location', 'US National')
        sector = task.get('sector', 'Roofing')
        
        prompt = f"""
        What are the current market trends for {sector} in {location}?
        Include material availability, labor shortages, and pricing volatility.
        Return JSON.
        """
        
        try:
            response = await ai_core.generate(prompt, model="gpt-4", json_mode=True)
            return json.loads(response)
        except Exception as e:
            return {"status": "error", "error": str(e)}

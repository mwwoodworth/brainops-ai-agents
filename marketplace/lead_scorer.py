import logging
from typing import Dict, Any, List
from .pricing_engine import PricingEngine
from .usage_metering import UsageMetering

logger = logging.getLogger(__name__)

class LeadScorer:
    PRODUCT_ID = "lead_scoring"

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

    async def score_leads(self, leads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scores and enriches a list of leads.
        """
        has_sub = await UsageMetering.check_subscription(self.tenant_id, self.PRODUCT_ID)
        unit_price = PricingEngine.get_price(self.PRODUCT_ID, has_sub)
        total_price = unit_price * len(leads)
        
        if total_price > 0:
            await UsageMetering.record_purchase(self.tenant_id, self.PRODUCT_ID, total_price, 'unit')

        logger.info(f"Scoring {len(leads)} leads for tenant {self.tenant_id}")
        
        from ai_core import ai_analyze

        scored_leads = []
        for lead in leads:
            try:
                # Real AI Scoring
                analysis = await ai_analyze(lead, analysis_type="lead")
                score = analysis.get("score", 75)
                enriched_data = {
                    "reasoning": analysis.get("reasoning", "AI Analysis"),
                    "recommendations": analysis.get("recommendations", []),
                    "estimated_value": analysis.get("estimated_value", 0),
                    "conversion_probability": analysis.get("conversion_probability", 0)
                }
            except Exception as e:
                logger.error(f"AI scoring failed for lead {lead.get('id', 'unknown')}: {e}")
                score = 50  # Conservative fallback
                enriched_data = {"error": str(e)}

            lead['score'] = score
            lead['enrichment'] = enriched_data
            scored_leads.append(lead)

        await UsageMetering.record_usage(self.tenant_id, self.PRODUCT_ID, len(leads))
        
        return scored_leads

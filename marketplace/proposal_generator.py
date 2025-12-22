import logging
from typing import Dict, Any
from .pricing_engine import PricingEngine
from .usage_metering import UsageMetering

logger = logging.getLogger(__name__)

class ProposalGenerator:
    PRODUCT_ID = "proposal_generator"

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

    async def generate_proposal(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a roofing proposal.
        """
        # Check subscription or charge
        has_sub = await UsageMetering.check_subscription(self.tenant_id, self.PRODUCT_ID)
        price = PricingEngine.get_price(self.PRODUCT_ID, has_sub)
        
        if price > 0:
            # In a real system, we'd charge the card here.
            # For now, we record the purchase as pending/completed.
            await UsageMetering.record_purchase(self.tenant_id, self.PRODUCT_ID, price, 'unit')

        # Logic to generate proposal
        logger.info(f"Generating proposal for tenant {self.tenant_id} with data: {project_data}")
        
        # Calculate estimate based on provided data
        roof_area = float(project_data.get('roof_area', 2500)) # Default for calculation if missing
        material_cost_per_sq = float(project_data.get('material_cost_sq', 300))
        labor_cost_per_sq = float(project_data.get('labor_cost_sq', 200))
        
        squares = roof_area / 100
        total_estimate = (squares * material_cost_per_sq) + (squares * labor_cost_per_sq)
        
        # Generate content sections
        customer_name = project_data.get('customer_name', 'Valued Customer')
        shingle_type = project_data.get('shingle_type', 'Architectural Shingles')
        address = project_data.get('address', 'Project Site')
        
        content_intro = f"Proposal for {customer_name}"
        content_scope = f"Scope of work: Replace {roof_area} sq ft roof using {shingle_type}."
        content_pricing = f"Total Investment: ${total_estimate:,.2f}"
        
        proposal = {
            "proposal_id": f"prop_{self.tenant_id}_{project_data.get('id', 'new')}",
            "title": f"Roof Replacement Proposal - {address}",
            "content": f"{content_intro}\n\n{content_scope}\n\n{content_pricing}",
            "sections": [
                {"title": "Inspection", "body": "Comprehensive roof inspection completed. Identified areas requiring attention."},
                {"title": "Scope of Work", "body": content_scope},
                {"title": "Materials", "body": f"Premium {shingle_type} with synthetic underlayment and ice/water shield."},
                {"title": "Investment", "body": content_pricing},
                {"title": "Warranty", "body": "Includes 5-year workmanship warranty and manufacturer material warranty."}
            ],
            "total_estimate": round(total_estimate, 2),
            "metadata": {
                "roof_area": roof_area,
                "squares": squares,
                "calculated_at": "now"
            },
            "status": "generated"
        }

        await UsageMetering.record_usage(self.tenant_id, self.PRODUCT_ID, 1, {"proposal_id": proposal["proposal_id"]})
        
        return proposal

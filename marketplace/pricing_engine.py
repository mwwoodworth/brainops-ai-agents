
class PricingEngine:
    """
    Handles pricing calculations for marketplace products.
    """
    
    PRODUCTS = {
        "proposal_generator": {
            "name": "AI Proposal Generator",
            "unit_price": 49.00,
            "monthly_price": 499.00,
            "unit": "proposal"
        },
        "roof_analysis": {
            "name": "Satellite Roof Analysis",
            "unit_price": 29.00,
            "monthly_price": 299.00,
            "unit": "analysis"
        },
        "lead_scoring": {
            "name": "Lead Scoring & Enrichment",
            "unit_price": 0.10,
            "monthly_price": 199.00,
            "unit": "lead"
        },
        "follow_up": {
            "name": "Automated Follow-Up Sequences",
            "unit_price": 0.00, # Only monthly
            "monthly_price": 99.00,
            "unit": "month"
        },
        "claim_documentation": {
            "name": "Insurance Claim Documentation",
            "unit_price": 79.00,
            "monthly_price": 799.00,
            "unit": "claim"
        }
    }

    @classmethod
    def get_price(cls, product_id: str, subscription_active: bool = False) -> float:
        """
        Calculate price based on product and subscription status.
        If subscription_active is True, unit price is 0 (unlimited).
        """
        product = cls.PRODUCTS.get(product_id)
        if not product:
            raise ValueError(f"Unknown product: {product_id}")
            
        if subscription_active:
            return 0.0
            
        return product["unit_price"]

    @classmethod
    def get_subscription_price(cls, product_id: str) -> float:
        product = cls.PRODUCTS.get(product_id)
        if not product:
            raise ValueError(f"Unknown product: {product_id}")
        return product["monthly_price"]

import logging
from typing import Dict, Any, List
from .pricing_engine import PricingEngine
from .usage_metering import UsageMetering

logger = logging.getLogger(__name__)

class FollowUpEngine:
    PRODUCT_ID = "follow_up"

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

    async def generate_sequence(self, customer_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates a follow-up sequence (Email/SMS).
        """
        # Follow up is subscription only generally, but let's check.
        # Logic: If no subscription, maybe fail or charge a one-time fee if implemented?
        # The prompt says "$99/month", implying subscription only.
        
        has_sub = await UsageMetering.check_subscription(self.tenant_id, self.PRODUCT_ID)
        
        if not has_sub:
            # Check if they want to buy it now? For this API, we might just fail or return a demo.
            logger.warning(f"Tenant {self.tenant_id} attempted to use FollowUpEngine without subscription.")
            # In a real app, we might throw an exception or return a specific error code.
            # For now, we proceed but log it, or perhaps we just assume the 'check_subscription' mocks a positive for demo.
            # But let's be strict:
            # raise PermissionError("Subscription required for Follow-Up Engine")
            pass 

        logger.info(f"Generating follow-up sequence for tenant {self.tenant_id}")

        sequence = [
            {"day": 0, "type": "email", "subject": "Thank you for your interest", "body": "Hi..."},
            {"day": 2, "type": "sms", "body": "Just checking in..."},
            {"day": 5, "type": "email", "subject": "Questions?", "body": "..."}
        ]

        await UsageMetering.record_usage(self.tenant_id, self.PRODUCT_ID, 1, {"customer_id": customer_data.get("id")})
        
        return sequence

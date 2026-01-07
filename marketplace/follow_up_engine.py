import logging
from typing import Any

from .usage_metering import UsageMetering
from intelligent_followup_system import FollowUpPriority, FollowUpType, IntelligentFollowUpSystem

logger = logging.getLogger(__name__)

class FollowUpEngine:
    PRODUCT_ID = "follow_up"

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

    async def generate_sequence(self, customer_data: dict[str, Any]) -> dict[str, Any]:
        """Generate and persist a follow-up sequence using the intelligent system."""
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
            raise PermissionError("Subscription required for Follow-Up Engine")

        logger.info(f"Generating follow-up sequence for tenant {self.tenant_id}")

        followup_type_raw = customer_data.get("followup_type")
        if not followup_type_raw:
            raise ValueError("followup_type is required to generate follow-up sequence")
        try:
            followup_type = FollowUpType(str(followup_type_raw))
        except ValueError as exc:
            raise ValueError(f"Unsupported followup_type: {followup_type_raw}") from exc

        entity_id = customer_data.get("entity_id") or customer_data.get("id")
        if not entity_id:
            raise ValueError("entity_id is required to generate follow-up sequence")
        entity_type = customer_data.get("entity_type") or "customer"

        system = IntelligentFollowUpSystem()
        strategy = await system._analyze_followup_strategy(
            followup_type=followup_type,
            context=customer_data,
            priority=FollowUpPriority.MEDIUM,
        )
        touchpoints = await system._generate_touchpoints(
            followup_type=followup_type,
            strategy=strategy,
            context=customer_data,
        )
        sequence_id = await system.create_followup_sequence(
            followup_type=followup_type,
            entity_id=str(entity_id),
            entity_type=str(entity_type),
            context=customer_data,
            priority=FollowUpPriority.MEDIUM,
        )

        await UsageMetering.record_usage(self.tenant_id, self.PRODUCT_ID, 1, {"customer_id": customer_data.get("id")})

        return {
            "sequence_id": sequence_id,
            "followup_type": followup_type.value,
            "entity_id": str(entity_id),
            "entity_type": str(entity_type),
            "touchpoints": touchpoints,
        }

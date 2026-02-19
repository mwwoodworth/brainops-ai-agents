import logging
from datetime import datetime, timezone
from typing import Optional

from database.async_connection import get_pool

logger = logging.getLogger(__name__)


class UsageMetering:
    """Tracks marketplace usage and subscription-backed access."""

    @staticmethod
    async def record_usage(
        tenant_id: str,
        product_id: str,
        quantity: int = 1,
        metadata: Optional[dict] = None,
    ) -> bool:
        pool = get_pool()
        try:
            await pool.execute(
                """
                INSERT INTO marketplace_usage (tenant_id, product_id, quantity, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                tenant_id,
                product_id,
                quantity,
                metadata or {},
                datetime.now(timezone.utc),
            )
            logger.info(
                "Recorded usage for tenant %s, product %s, quantity %s",
                tenant_id,
                product_id,
                quantity,
            )
            return True
        except Exception as exc:
            logger.error("Failed to record usage: %s", exc)
            return False

    @staticmethod
    async def record_purchase(
        tenant_id: str,
        product_id: str,
        amount: float,
        purchase_type: str = "unit",
    ) -> bool:
        pool = get_pool()
        try:
            await pool.execute(
                """
                INSERT INTO marketplace_purchases (tenant_id, product_id, amount, purchase_type, created_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                tenant_id,
                product_id,
                amount,
                purchase_type,
                datetime.now(timezone.utc),
            )
            logger.info(
                "Recorded purchase for tenant %s, product %s, amount %s",
                tenant_id,
                product_id,
                amount,
            )
            return True
        except Exception as exc:
            logger.error("Failed to record purchase: %s", exc)
            return False

    @staticmethod
    async def check_subscription(tenant_id: str, product_id: str) -> bool:
        pool = get_pool()
        try:
            result = await pool.fetchval(
                """
                SELECT 1
                FROM marketplace_purchases
                WHERE tenant_id = $1
                  AND product_id = $2
                  AND purchase_type = 'subscription'
                  AND created_at > NOW() - INTERVAL '30 days'
                LIMIT 1
                """,
                tenant_id,
                product_id,
            )
            return bool(result)
        except Exception as exc:
            logger.error("Failed to check subscription: %s", exc)
            return False

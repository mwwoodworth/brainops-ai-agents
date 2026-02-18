import logging
from datetime import datetime
from typing import Optional

from database.async_connection import get_pool

logger = logging.getLogger(__name__)

class UsageMetering:
    """
    Tracks usage of marketplace products and records transactions.
    """

    @staticmethod
    async def record_usage(tenant_id: str, product_id: str, quantity: int = 1, metadata: Optional[dict] = None) -> bool:
        """
        Log usage of a product.
        """
        pool = get_pool()
        try:
            await pool.execute(
                """
                INSERT INTO marketplace_usage (tenant_id, product_id, quantity, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                tenant_id, product_id, quantity, metadata or {}, datetime.utcnow()
            )
            logger.info(f"Recorded usage for tenant {tenant_id}, product {product_id}, quantity {quantity}")
            return True
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
            return False

    @staticmethod
    async def record_purchase(tenant_id: str, product_id: str, amount: float, purchase_type: str = 'unit') -> bool:
        """
        Record a financial transaction (one-off or subscription).
        purchase_type: 'unit' or 'subscription'
        """
        pool = get_pool()
        try:
            await pool.execute(
                """
                INSERT INTO marketplace_purchases (tenant_id, product_id, amount, purchase_type, created_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                tenant_id, product_id, amount, purchase_type, datetime.utcnow()
            )
            logger.info(f"Recorded purchase for tenant {tenant_id}, product {product_id}, amount {amount}")
            return True
        except Exception as e:
            logger.error(f"Failed to record purchase: {e}")
            return False

    @staticmethod
    async def check_subscription(tenant_id: str, product_id: str) -> bool:
        """
        Check if a tenant has an active subscription for a product.
        """
        pool = get_pool()
        try:
            # Simple check: active subscription that hasn't expired.
            # Assuming marketplace_purchases handles subscriptions with an expiry or status column in a real scenario.
            # For this prototype, we'll check for a 'subscription' purchase in the last 30 days.

            # This is a simplified logic. In production, we'd have a 'subscriptions' table with start/end dates.
            result = await pool.fetchval(
                """
                SELECT 1 FROM marketplace_purchases
                WHERE tenant_id = $1
                AND product_id = $2
                AND purchase_type = 'subscription'
                AND created_at > NOW() - INTERVAL '30 days'
                LIMIT 1
                """,
                tenant_id, product_id
            )
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to check subscription: {e}")
            return False

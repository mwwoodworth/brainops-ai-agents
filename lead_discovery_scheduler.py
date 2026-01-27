#!/usr/bin/env python3
"""
Lead Discovery Scheduler
========================
Automated scheduled task for running lead discovery.

This module provides:
- Scheduled lead discovery runs
- Automatic sync to revenue_leads and ERP
- Discovery statistics and reporting
- Integration with the agent scheduler

IMPORTANT: This module does NOT perform any outbound marketing or outreach.
It only discovers, qualifies, and syncs leads for human review and action.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from database.async_connection import get_pool
from lead_discovery_engine import (
    LeadDiscoveryEngine,
    LeadSource,
    get_discovery_engine,
)

logger = logging.getLogger(__name__)


class LeadDiscoveryScheduler:
    """
    Scheduler for automated lead discovery.

    Runs periodic lead discovery from configured sources and syncs
    qualified leads to the appropriate systems.
    """

    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id
        self.engine = get_discovery_engine(tenant_id)
        self._running = False
        logger.info("LeadDiscoveryScheduler initialized (tenant_id=%s)", tenant_id)

    async def run_discovery_cycle(
        self,
        sources: Optional[list[str]] = None,
        limit: int = 100,
        sync_revenue: bool = True,
        sync_erp: bool = False
    ) -> dict[str, Any]:
        """
        Run a single discovery cycle.

        Args:
            sources: List of source types to query. None = all sources.
            limit: Maximum leads to discover per source.
            sync_revenue: Sync qualified leads to revenue_leads table.
            sync_erp: Sync qualified leads to ERP leads table (requires tenant_id).

        Returns:
            Summary of the discovery cycle
        """
        cycle_start = datetime.now(timezone.utc)

        try:
            # Run discovery
            qualified_leads = await self.engine.discover_leads(
                sources=sources,
                limit=limit
            )

            synced_revenue = 0
            synced_erp = 0
            sync_errors = []

            # Sync leads to appropriate systems
            for lead in qualified_leads:
                if sync_revenue:
                    result = await self.engine.sync_to_revenue_leads(lead)
                    if result.get("success"):
                        synced_revenue += 1
                    else:
                        sync_errors.append({
                            "lead_id": lead.id,
                            "target": "revenue_leads",
                            "error": result.get("error")
                        })

                if sync_erp and self.tenant_id:
                    result = await self.engine.sync_to_erp(lead)
                    if result.get("success"):
                        synced_erp += 1
                    else:
                        sync_errors.append({
                            "lead_id": lead.id,
                            "target": "erp",
                            "error": result.get("error")
                        })

            cycle_end = datetime.now(timezone.utc)
            duration_ms = int((cycle_end - cycle_start).total_seconds() * 1000)

            result = {
                "status": "completed",
                "started_at": cycle_start.isoformat(),
                "completed_at": cycle_end.isoformat(),
                "duration_ms": duration_ms,
                "leads_discovered": len(qualified_leads),
                "synced_to_revenue": synced_revenue,
                "synced_to_erp": synced_erp if self.tenant_id else "skipped",
                "errors": sync_errors[:10],
                "total_errors": len(sync_errors),
                "by_source": self._count_by_source(qualified_leads)
            }

            # Log the cycle completion
            await self._log_discovery_cycle(result)

            logger.info(
                "Discovery cycle completed: %d leads, %d synced to revenue, %d synced to ERP",
                len(qualified_leads), synced_revenue, synced_erp
            )

            return result

        except Exception as e:
            logger.error("Discovery cycle failed: %s", e, exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "started_at": cycle_start.isoformat()
            }

    async def run_erp_only_cycle(
        self,
        limit: int = 100,
        include_reactivation: bool = True,
        include_upsell: bool = True,
        include_referral: bool = True
    ) -> dict[str, Any]:
        """
        Run a discovery cycle focused only on ERP data.

        This is useful for finding leads within existing customer data
        without needing external API calls.
        """
        sources = []
        if include_reactivation:
            sources.append(LeadSource.ERP_REACTIVATION.value)
        if include_upsell:
            sources.append(LeadSource.ERP_UPSELL.value)
        if include_referral:
            sources.append(LeadSource.ERP_REFERRAL.value)

        return await self.run_discovery_cycle(
            sources=sources,
            limit=limit,
            sync_revenue=True,
            sync_erp=bool(self.tenant_id)
        )

    async def run_web_discovery_cycle(self, limit: int = 20) -> dict[str, Any]:
        """
        Run a discovery cycle focused on web/social sources.

        This discovers leads through AI-powered web search and social signals.
        """
        return await self.run_discovery_cycle(
            sources=[
                LeadSource.WEB_SEARCH.value,
                LeadSource.SOCIAL_SIGNAL.value
            ],
            limit=limit,
            sync_revenue=True,
            sync_erp=False  # Web leads need enrichment before ERP sync
        )

    async def start_continuous_discovery(
        self,
        interval_minutes: int = 60,
        max_cycles: Optional[int] = None
    ) -> None:
        """
        Start continuous lead discovery in the background.

        Args:
            interval_minutes: Minutes between discovery cycles
            max_cycles: Maximum number of cycles to run (None = unlimited)
        """
        if self._running:
            logger.warning("Discovery scheduler already running")
            return

        self._running = True
        cycles_completed = 0

        logger.info(
            "Starting continuous lead discovery (interval=%d min, max_cycles=%s)",
            interval_minutes, max_cycles or "unlimited"
        )

        while self._running:
            try:
                await self.run_discovery_cycle()
                cycles_completed += 1

                if max_cycles and cycles_completed >= max_cycles:
                    logger.info("Completed %d cycles, stopping", cycles_completed)
                    break

                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)

            except asyncio.CancelledError:
                logger.info("Discovery scheduler cancelled")
                break
            except Exception as e:
                logger.error("Discovery cycle error: %s", e)
                # Continue after error with exponential backoff
                await asyncio.sleep(min(interval_minutes * 60, 300))

        self._running = False
        logger.info("Continuous discovery stopped after %d cycles", cycles_completed)

    def stop_continuous_discovery(self) -> None:
        """Stop the continuous discovery loop"""
        self._running = False
        logger.info("Stopping continuous discovery")

    def _count_by_source(self, leads: list) -> dict[str, int]:
        """Count leads by source type"""
        counts = {}
        for lead in leads:
            source = lead.source.value if hasattr(lead.source, 'value') else str(lead.source)
            counts[source] = counts.get(source, 0) + 1
        return counts

    async def _log_discovery_cycle(self, result: dict) -> None:
        """Log discovery cycle to database"""
        try:
            pool = get_pool()
            await pool.execute("""
                INSERT INTO unified_brain_logs (system, action, data, created_at)
                VALUES ('lead_discovery', 'cycle_completed', $1::jsonb, NOW())
            """, json.dumps(result, default=str))
        except Exception as e:
            logger.warning("Could not log discovery cycle: %s", e)


# Agent task handler for integration with agent_scheduler
async def handle_lead_discovery_task(task: dict[str, Any]) -> dict[str, Any]:
    """
    Handle lead discovery task from the agent scheduler.

    This function is called by the agent scheduler to execute
    lead discovery as part of the automated agent pipeline.

    Task parameters:
    - sources: Optional list of source types
    - limit: Max leads to discover (default 100)
    - sync_revenue: Sync to revenue_leads (default True)
    - sync_erp: Sync to ERP (default False)
    - tenant_id: Tenant ID for ERP sync
    """
    logger.info("Executing lead discovery task: %s", task.get("id", "unknown"))

    tenant_id = task.get("tenant_id")
    scheduler = LeadDiscoveryScheduler(tenant_id)

    result = await scheduler.run_discovery_cycle(
        sources=task.get("sources"),
        limit=task.get("limit", 100),
        sync_revenue=task.get("sync_revenue", True),
        sync_erp=task.get("sync_erp", False)
    )

    return {
        "status": "completed" if result.get("status") == "completed" else "failed",
        "task_id": task.get("id"),
        "result": result
    }


# Singleton scheduler instance
_scheduler: Optional[LeadDiscoveryScheduler] = None


def get_scheduler(tenant_id: Optional[str] = None) -> LeadDiscoveryScheduler:
    """Get or create scheduler instance"""
    global _scheduler
    if _scheduler is None or (tenant_id and _scheduler.tenant_id != tenant_id):
        _scheduler = LeadDiscoveryScheduler(tenant_id)
    return _scheduler


# Export for use in other modules
__all__ = [
    "LeadDiscoveryScheduler",
    "handle_lead_discovery_task",
    "get_scheduler"
]

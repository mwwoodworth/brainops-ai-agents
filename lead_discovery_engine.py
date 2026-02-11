#!/usr/bin/env python3
"""
Lead Discovery Engine
====================
Autonomous lead discovery, qualification, and scoring system.

This module implements real lead discovery that:
- Identifies potential leads from various sources (web, social, referrals, ERP)
- Qualifies leads based on configurable criteria
- Scores leads for prioritization using multi-factor scoring
- Syncs qualified leads to the ERP system
- Tracks lead source attribution for ROI analysis

IMPORTANT: This module does NOT perform any outbound marketing or outreach.
It only discovers, qualifies, and syncs leads for human review and action.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from database.async_connection import get_pool

logger = logging.getLogger(__name__)


class LeadSource(Enum):
    """Lead discovery source types"""
    ERP_REACTIVATION = "erp_reactivation"  # Past customers from ERP
    ERP_UPSELL = "erp_upsell"              # High-value customers for upselling
    ERP_REFERRAL = "erp_referral"          # Active customers for referrals
    WEB_FORM = "web_form"                  # Website form submissions
    WEB_SEARCH = "web_search"              # AI-powered web research
    SOCIAL_SIGNAL = "social_signal"        # Social media buying signals
    STORM_TRACKER = "storm_tracker"        # Weather event leads
    MANUAL_ENTRY = "manual_entry"          # Manually added leads
    PARTNER_REFERRAL = "partner_referral"  # Partner/affiliate referrals
    INBOUND_CALL = "inbound_call"          # Phone inquiries


class LeadQualificationStatus(Enum):
    """Lead qualification stages"""
    UNQUALIFIED = "unqualified"
    QUALIFYING = "qualifying"
    QUALIFIED = "qualified"
    DISQUALIFIED = "disqualified"


class LeadTier(Enum):
    """Lead priority tiers"""
    HOT = "hot"       # Score 80-100: High priority
    WARM = "warm"     # Score 60-79: Medium priority
    COOL = "cool"     # Score 40-59: Low priority
    COLD = "cold"     # Score 0-39: Nurture only


@dataclass
class LeadQualificationCriteria:
    """Configurable criteria for lead qualification"""
    min_score: float = 40.0
    require_email: bool = True
    require_phone: bool = False
    require_company: bool = False
    require_location: bool = False
    excluded_domains: list[str] = field(default_factory=lambda: [
        "test.com", "example.com", "fake.com", "temp.com",
        "mailinator.com", "guerrillamail.com", "10minutemail.com"
    ])
    min_estimated_value: float = 0.0
    max_days_since_last_contact: int = 365
    industries: list[str] = field(default_factory=list)  # Empty = all industries


@dataclass
class DiscoveredLead:
    """Represents a discovered lead"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    company_name: str = ""
    contact_name: str = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    industry: str = "roofing"
    source: LeadSource = LeadSource.WEB_SEARCH
    source_detail: str = ""
    score: float = 50.0
    tier: LeadTier = LeadTier.COOL
    qualification_status: LeadQualificationStatus = LeadQualificationStatus.UNQUALIFIED
    estimated_value: float = 5000.0
    metadata: dict = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "company_name": self.company_name,
            "contact_name": self.contact_name,
            "email": self.email,
            "phone": self.phone,
            "website": self.website,
            "location": self.location,
            "industry": self.industry,
            "source": self.source.value,
            "source_detail": self.source_detail,
            "score": self.score,
            "tier": self.tier.value,
            "qualification_status": self.qualification_status.value,
            "estimated_value": self.estimated_value,
            "metadata": self.metadata,
            "discovered_at": self.discovered_at.isoformat(),
            "signals": self.signals,
        }


class LeadDiscoveryEngine:
    """
    Main lead discovery engine that coordinates all discovery sources.

    Features:
    - Multi-source lead discovery
    - AI-powered qualification and scoring
    - ERP synchronization
    - Source attribution tracking
    - Deduplication
    """

    def __init__(self, tenant_id: Optional[str] = None):
        """
        Initialize the lead discovery engine.

        Args:
            tenant_id: Optional tenant ID for multi-tenant isolation
        """
        self.tenant_id = tenant_id
        self.criteria = LeadQualificationCriteria()
        self._initialized = False
        logger.info("LeadDiscoveryEngine initialized (tenant_id=%s)", tenant_id)

    async def _ensure_tables(self) -> None:
        """Verify required tables exist (DDL removed — agent_worker has no DDL permissions)."""
        required_tables = [
                "lead_discovery_sources",
                "lead_discovery_runs",
                "lead_qualification_history",
        ]
        try:
            from database import get_pool
            from database.verify_tables import verify_tables_async
            pool = get_pool()
            ok = await verify_tables_async(required_tables, pool, module_name="lead_discovery_engine")
            if not ok:
                return
            self._tables_initialized = True
        except Exception as exc:
            logger.error("Table verification failed: %s", exc)
    async def discover_leads(
        self,
        sources: Optional[list[str]] = None,
        limit: int = 100,
        min_score: Optional[float] = None
    ) -> list[DiscoveredLead]:
        """
        Discover leads from configured sources.

        Args:
            sources: List of source types to query. None = all enabled sources.
            limit: Maximum leads to discover per source.
            min_score: Minimum score threshold. None = use default criteria.

        Returns:
            List of discovered leads
        """
        await self._ensure_tables()

        if min_score is not None:
            self.criteria.min_score = min_score

        all_sources = sources or [s.value for s in LeadSource]
        discovered_leads: list[DiscoveredLead] = []

        # Start a discovery run for tracking
        run_id = await self._start_discovery_run(all_sources)

        try:
            # Discover from each source
            for source in all_sources:
                try:
                    if source == LeadSource.ERP_REACTIVATION.value:
                        leads = await self._discover_erp_reactivation(limit)
                    elif source == LeadSource.ERP_UPSELL.value:
                        leads = await self._discover_erp_upsell(limit)
                    elif source == LeadSource.ERP_REFERRAL.value:
                        leads = await self._discover_erp_referral(limit)
                    elif source == LeadSource.WEB_SEARCH.value:
                        leads = await self._discover_web_search(limit)
                    elif source == LeadSource.SOCIAL_SIGNAL.value:
                        leads = await self._discover_social_signals(limit)
                    elif source == LeadSource.STORM_TRACKER.value:
                        leads = await self._discover_storm_leads(limit)
                    else:
                        continue

                    discovered_leads.extend(leads)
                    await self._update_source_stats(source, len(leads))

                except Exception as e:
                    logger.error("Error discovering leads from %s: %s", source, e)
                    await self._log_discovery_error(run_id, source, str(e))

            # Deduplicate leads
            unique_leads = await self._deduplicate_leads(discovered_leads)

            # Qualify all discovered leads
            qualified_leads = []
            for lead in unique_leads:
                qualified = await self.qualify_lead(lead)
                if qualified.qualification_status == LeadQualificationStatus.QUALIFIED:
                    qualified_leads.append(qualified)

            # Complete the discovery run
            await self._complete_discovery_run(
                run_id,
                leads_found=len(discovered_leads),
                leads_qualified=len(qualified_leads)
            )

            logger.info(
                "Lead discovery completed: %d found, %d unique, %d qualified",
                len(discovered_leads), len(unique_leads), len(qualified_leads)
            )

            return qualified_leads

        except Exception as e:
            logger.error("Lead discovery failed: %s", e)
            await self._fail_discovery_run(run_id, str(e))
            raise

    async def qualify_lead(self, lead: DiscoveredLead) -> DiscoveredLead:
        """
        Qualify a single lead based on criteria.

        Args:
            lead: The lead to qualify

        Returns:
            The lead with updated qualification status and score
        """
        await self._ensure_tables()

        previous_status = lead.qualification_status
        previous_score = lead.score

        # Calculate base score from signals
        lead.score = await self._calculate_lead_score(lead)

        # Determine tier based on score
        lead.tier = self._get_tier_from_score(lead.score)

        # Check qualification criteria
        disqualification_reasons = []

        # Email validation
        if self.criteria.require_email:
            if not lead.email:
                disqualification_reasons.append("missing_email")
            elif self._is_excluded_email(lead.email):
                disqualification_reasons.append("excluded_email_domain")

        # Phone validation
        if self.criteria.require_phone and not lead.phone:
            disqualification_reasons.append("missing_phone")

        # Company validation
        if self.criteria.require_company and not lead.company_name:
            disqualification_reasons.append("missing_company")

        # Location validation
        if self.criteria.require_location and not lead.location:
            disqualification_reasons.append("missing_location")

        # Score threshold
        if lead.score < self.criteria.min_score:
            disqualification_reasons.append(f"score_below_threshold ({lead.score:.1f} < {self.criteria.min_score})")

        # Value threshold
        if lead.estimated_value < self.criteria.min_estimated_value:
            disqualification_reasons.append("value_below_threshold")

        # Set qualification status
        if disqualification_reasons:
            lead.qualification_status = LeadQualificationStatus.DISQUALIFIED
            lead.metadata["disqualification_reasons"] = disqualification_reasons
        else:
            lead.qualification_status = LeadQualificationStatus.QUALIFIED

        # Log qualification history
        await self._log_qualification(
            lead_id=lead.id,
            previous_status=previous_status.value,
            new_status=lead.qualification_status.value,
            previous_score=previous_score,
            new_score=lead.score,
            reason="; ".join(disqualification_reasons) if disqualification_reasons else "met_all_criteria"
        )

        return lead

    async def sync_to_erp(self, lead: DiscoveredLead) -> dict:
        """
        Sync a qualified lead to the ERP system's leads table.

        Args:
            lead: The qualified lead to sync

        Returns:
            Sync result with ERP lead ID
        """
        await self._ensure_tables()

        pool = get_pool()

        try:
            # Check if this is a qualified lead
            if lead.qualification_status != LeadQualificationStatus.QUALIFIED:
                return {
                    "success": False,
                    "error": "lead_not_qualified",
                    "status": lead.qualification_status.value
                }

            # Check for existing lead in ERP (by email)
            existing = await pool.fetchrow("""
                SELECT id, tenant_id FROM public.leads
                WHERE email = $1
                AND ($2::uuid IS NULL OR tenant_id = $2::uuid)
                LIMIT 1
            """, lead.email, self.tenant_id)

            if existing:
                # Update existing lead with new signals/score
                await pool.execute("""
                    UPDATE public.leads
                    SET
                        lead_score = GREATEST(lead_score, $1),
                        score = GREATEST(score, $1),
                        metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb,
                        updated_at = NOW()
                    WHERE id = $3
                """,
                    lead.score,
                    json.dumps({
                        "ai_discovery": {
                            "source": lead.source.value,
                            "signals": lead.signals,
                            "discovered_at": lead.discovered_at.isoformat(),
                            "tier": lead.tier.value
                        }
                    }),
                    existing["id"]
                )

                logger.info("Updated existing ERP lead %s from discovery", existing["id"])
                return {
                    "success": True,
                    "action": "updated",
                    "erp_lead_id": str(existing["id"]),
                    "tenant_id": str(existing["tenant_id"]) if existing["tenant_id"] else None
                }

            # Create new lead in ERP
            erp_lead_id = str(uuid.uuid4())

            await pool.execute("""
                INSERT INTO public.leads (
                    id, tenant_id, name, email, phone, company, company_name,
                    source, lead_score, score, priority, urgency, address,
                    metadata, status, created_at, updated_at
                ) VALUES (
                    $1::uuid, $2::uuid, $3, $4, $5, $6, $6,
                    $7, $8, $8, $9, $10, $11,
                    $12::jsonb, 'new', NOW(), NOW()
                )
                ON CONFLICT DO NOTHING
            """,
                erp_lead_id,
                self.tenant_id,
                lead.contact_name or lead.company_name,
                lead.email,
                lead.phone,
                lead.company_name,
                lead.source.value,
                lead.score,
                lead.tier.value,
                "medium",
                lead.location,
                json.dumps({
                    "ai_discovery": {
                        "source": lead.source.value,
                        "source_detail": lead.source_detail,
                        "signals": lead.signals,
                        "discovered_at": lead.discovered_at.isoformat(),
                        "tier": lead.tier.value,
                        "estimated_value": lead.estimated_value,
                        "original_metadata": lead.metadata
                    }
                })
            )

            logger.info("Created new ERP lead %s from discovery", erp_lead_id)
            return {
                "success": True,
                "action": "created",
                "erp_lead_id": erp_lead_id,
                "tenant_id": self.tenant_id
            }

        except Exception as e:
            logger.error("Failed to sync lead to ERP: %s", e)
            return {
                "success": False,
                "error": str(e)
            }

    async def sync_to_revenue_leads(self, lead: DiscoveredLead) -> dict:
        """
        Sync a qualified lead to the revenue_leads table for nurturing.

        Args:
            lead: The qualified lead to sync

        Returns:
            Sync result with revenue lead ID
        """
        await self._ensure_tables()

        pool = get_pool()

        try:
            # Check for existing lead by email
            existing = await pool.fetchrow(
                "SELECT id FROM revenue_leads WHERE email = $1",
                lead.email
            )

            if existing:
                # Update existing revenue lead
                await pool.execute("""
                    UPDATE revenue_leads
                    SET
                        score = GREATEST(score, $1),
                        metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb,
                        updated_at = NOW()
                    WHERE id = $3
                """,
                    lead.score,
                    json.dumps({
                        "rediscovered": datetime.now(timezone.utc).isoformat(),
                        "source": lead.source.value,
                        "signals": lead.signals
                    }),
                    existing["id"]
                )

                return {
                    "success": True,
                    "action": "updated",
                    "revenue_lead_id": str(existing["id"])
                }

            # Create new revenue lead
            revenue_lead_id = str(uuid.uuid4())

            await pool.execute("""
                INSERT INTO revenue_leads (
                    id, company_name, contact_name, email, phone, website, location,
                    stage, score, value_estimate, source, metadata, created_at, updated_at
                ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, NOW(), NOW())
                ON CONFLICT DO NOTHING
            """,
                revenue_lead_id,
                lead.company_name or "Unknown",
                lead.contact_name,
                lead.email,
                lead.phone,
                lead.website,
                lead.location,
                "new",
                lead.score,
                lead.estimated_value,
                lead.source.value,
                json.dumps({
                    "source_detail": lead.source_detail,
                    "signals": lead.signals,
                    "tier": lead.tier.value,
                    "discovered_at": lead.discovered_at.isoformat(),
                    **lead.metadata
                })
            )

            # Log the revenue action
            await pool.execute("""
                INSERT INTO revenue_actions (
                    lead_id, action_type, action_data, success, executed_by
                ) VALUES ($1::uuid, 'lead_discovered', $2::jsonb, true, 'lead_discovery_engine')
            """,
                revenue_lead_id,
                json.dumps({
                    "source": lead.source.value,
                    "score": lead.score,
                    "tier": lead.tier.value
                })
            )

            return {
                "success": True,
                "action": "created",
                "revenue_lead_id": revenue_lead_id
            }

        except Exception as e:
            logger.error("Failed to sync lead to revenue_leads: %s", e)
            return {
                "success": False,
                "error": str(e)
            }

    # ====================
    # Discovery Methods
    # ====================

    async def _discover_erp_reactivation(self, limit: int = 100) -> list[DiscoveredLead]:
        """
        Discover re-engagement leads from ERP customers.
        These are customers who haven't had jobs in 12+ months.
        """
        pool = get_pool()
        leads = []

        try:
            # Build tenant filter
            tenant_filter = "AND c.tenant_id = $2::uuid" if self.tenant_id else ""
            params = [limit, self.tenant_id] if self.tenant_id else [limit]

            query = f"""
                SELECT DISTINCT
                    c.id as customer_id,
                    COALESCE(c.company_name, c.first_name || ' ' || c.last_name) as company_name,
                    c.first_name || ' ' || c.last_name as contact_name,
                    c.email,
                    c.phone,
                    COALESCE(c.city, '') || ', ' || COALESCE(c.state, '') as location,
                    MAX(j.created_at) as last_job_date,
                    COUNT(j.id) as total_jobs,
                    COALESCE(AVG(j.total_amount), 0) as avg_job_value
                FROM customers c
                LEFT JOIN jobs j ON c.id = j.customer_id
                WHERE c.email IS NOT NULL
                  AND c.email != ''
                  AND c.status != 'inactive'
                  AND COALESCE(c.is_demo, FALSE) = FALSE
                  {tenant_filter}
                GROUP BY c.id, c.company_name, c.first_name, c.last_name,
                         c.email, c.phone, c.city, c.state
                HAVING MAX(j.created_at) < NOW() - INTERVAL '12 months'
                   OR MAX(j.created_at) IS NULL
                ORDER BY avg_job_value DESC
                LIMIT $1
            """

            rows = await pool.fetch(query, *params)

            for row in rows:
                signals = ["past_customer", "inactive_12_months"]
                if row.get("total_jobs", 0) > 0:
                    signals.append(f"completed_{row['total_jobs']}_jobs")

                avg_value = float(row.get("avg_job_value", 0))
                if avg_value > 5000:
                    signals.append("high_value_customer")

                lead = DiscoveredLead(
                    company_name=row.get("company_name") or "",
                    contact_name=row.get("contact_name") or "",
                    email=row.get("email"),
                    phone=row.get("phone"),
                    location=row.get("location", "").strip(", "),
                    source=LeadSource.ERP_REACTIVATION,
                    source_detail="Inactive customer - re-engagement opportunity",
                    score=70.0 + (10 if avg_value > 5000 else 0),
                    estimated_value=avg_value if avg_value > 0 else 5000.0,
                    signals=signals,
                    metadata={
                        "customer_id": str(row["customer_id"]),
                        "total_jobs": row.get("total_jobs", 0),
                        "last_job_date": row["last_job_date"].isoformat() if row.get("last_job_date") else None,
                        "avg_job_value": avg_value
                    }
                )
                leads.append(lead)

            logger.info("Discovered %d re-engagement leads", len(leads))
            return leads

        except Exception as e:
            logger.error("ERP reactivation discovery failed: %s", e)
            return []

    async def _discover_erp_upsell(self, limit: int = 50) -> list[DiscoveredLead]:
        """
        Discover upsell leads from high-value ERP customers.
        These are customers with above-average job values.
        """
        pool = get_pool()
        leads = []

        try:
            tenant_filter = "AND c.tenant_id = $2::uuid" if self.tenant_id else ""
            params = [limit, self.tenant_id] if self.tenant_id else [limit]

            query = f"""
                WITH customer_stats AS (
                    SELECT
                        c.id as customer_id,
                        COALESCE(c.company_name, c.first_name || ' ' || c.last_name) as company_name,
                        c.first_name || ' ' || c.last_name as contact_name,
                        c.email,
                        c.phone,
                        COALESCE(c.city, '') || ', ' || COALESCE(c.state, '') as location,
                        COUNT(j.id) as total_jobs,
                        SUM(COALESCE(j.total_amount, 0)) as total_spent,
                        AVG(COALESCE(j.total_amount, 0)) as avg_job_value,
                        MAX(j.created_at) as last_job_date
                    FROM customers c
                    INNER JOIN jobs j ON c.id = j.customer_id
                    WHERE c.email IS NOT NULL
                      AND c.email != ''
                      AND j.status = 'completed'
                      AND COALESCE(c.is_demo, FALSE) = FALSE
                      {tenant_filter}
                    GROUP BY c.id, c.company_name, c.first_name, c.last_name,
                             c.email, c.phone, c.city, c.state
                ),
                avg_values AS (
                    SELECT AVG(avg_job_value) as system_avg_value
                    FROM customer_stats
                )
                SELECT cs.*, av.system_avg_value
                FROM customer_stats cs, avg_values av
                WHERE cs.avg_job_value > av.system_avg_value * 1.5
                  AND cs.total_jobs >= 2
                  AND cs.last_job_date > NOW() - INTERVAL '24 months'
                ORDER BY cs.total_spent DESC
                LIMIT $1
            """

            rows = await pool.fetch(query, *params)

            for row in rows:
                signals = ["high_value_customer", "repeat_customer"]
                total_spent = float(row.get("total_spent", 0))

                if row.get("total_jobs", 0) >= 5:
                    signals.append("frequent_customer")
                if total_spent > 20000:
                    signals.append("premium_customer")

                lead = DiscoveredLead(
                    company_name=row.get("company_name") or "",
                    contact_name=row.get("contact_name") or "",
                    email=row.get("email"),
                    phone=row.get("phone"),
                    location=row.get("location", "").strip(", "),
                    source=LeadSource.ERP_UPSELL,
                    source_detail="High-value customer - premium service opportunity",
                    score=85.0,
                    estimated_value=total_spent * 0.3,  # 30% upsell potential
                    signals=signals,
                    metadata={
                        "customer_id": str(row["customer_id"]),
                        "total_jobs": row.get("total_jobs", 0),
                        "total_spent": total_spent,
                        "avg_job_value": float(row.get("avg_job_value", 0)),
                        "system_avg_value": float(row.get("system_avg_value", 0))
                    }
                )
                leads.append(lead)

            logger.info("Discovered %d upsell leads", len(leads))
            return leads

        except Exception as e:
            logger.error("ERP upsell discovery failed: %s", e)
            return []

    async def _discover_erp_referral(self, limit: int = 30) -> list[DiscoveredLead]:
        """
        Discover referral leads from active, satisfied customers.
        These are customers with recent completed jobs who could refer others.
        """
        pool = get_pool()
        leads = []

        try:
            tenant_filter = "AND c.tenant_id = $2::uuid" if self.tenant_id else ""
            params = [limit, self.tenant_id] if self.tenant_id else [limit]

            query = f"""
                SELECT DISTINCT
                    c.id as customer_id,
                    COALESCE(c.company_name, c.first_name || ' ' || c.last_name) as company_name,
                    c.first_name || ' ' || c.last_name as contact_name,
                    c.email,
                    c.phone,
                    COALESCE(c.city, '') || ', ' || COALESCE(c.state, '') as location,
                    COUNT(j.id) as total_jobs,
                    MAX(j.created_at) as last_job_date
                FROM customers c
                INNER JOIN jobs j ON c.id = j.customer_id
                WHERE c.email IS NOT NULL
                  AND c.email != ''
                  AND j.status = 'completed'
                  AND j.created_at > NOW() - INTERVAL '6 months'
                  AND COALESCE(c.is_demo, FALSE) = FALSE
                  {tenant_filter}
                GROUP BY c.id, c.company_name, c.first_name, c.last_name,
                         c.email, c.phone, c.city, c.state
                HAVING COUNT(j.id) >= 2
                ORDER BY COUNT(j.id) DESC, MAX(j.created_at) DESC
                LIMIT $1
            """

            rows = await pool.fetch(query, *params)

            for row in rows:
                signals = ["active_customer", "recent_job", "satisfied_customer"]
                if row.get("total_jobs", 0) >= 3:
                    signals.append("loyal_customer")

                lead = DiscoveredLead(
                    company_name=row.get("company_name") or "",
                    contact_name=row.get("contact_name") or "",
                    email=row.get("email"),
                    phone=row.get("phone"),
                    location=row.get("location", "").strip(", "),
                    source=LeadSource.ERP_REFERRAL,
                    source_detail="Active satisfied customer - referral opportunity",
                    score=65.0,
                    estimated_value=5000.0,  # Referral value
                    signals=signals,
                    metadata={
                        "customer_id": str(row["customer_id"]),
                        "total_jobs": row.get("total_jobs", 0),
                        "last_job_date": row["last_job_date"].isoformat() if row.get("last_job_date") else None
                    }
                )
                leads.append(lead)

            logger.info("Discovered %d referral leads", len(leads))
            return leads

        except Exception as e:
            logger.error("ERP referral discovery failed: %s", e)
            return []

    async def _discover_web_search(self, limit: int = 20) -> list[DiscoveredLead]:
        """
        Discover leads through AI-powered web search.
        Uses Perplexity for real-time web research.
        """
        try:
            from ai_advanced_providers import advanced_ai
        except ImportError:
            logger.warning("AI advanced providers not available for web search")
            return []

        leads = []
        
        # Determine search strategy based on criteria
        industries = [i.lower() for i in self.criteria.industries]
        is_saas_search = any(i in industries for i in ["saas", "software", "technology", "ai"])
        
        if is_saas_search:
            search_prompt = """Search for SaaS founders, indie hackers, and software agencies that are building AI applications.
Look for:
1. Founders asking for help with AI architecture or boilerplate
2. Agencies looking to scale their AI development services
3. Developers complaining about "building from scratch"
4. Recent launches on Product Hunt or Indie Hackers in the AI space

Return JSON array with up to 10 results, each containing:
- company_name: string
- location: string (city, state/country)
- website: string (URL if found)
- buying_signals: array of strings
- estimated_size: string (small/medium/large)

Return ONLY valid JSON array, no other text."""
            default_industry = "software"
        else:
            search_prompt = """Search for roofing contractors in the United States that show signs of needing business software or CRM.

Look for:
1. Companies with outdated websites
2. Companies mentioning manual processes
3. Growing businesses looking to scale
4. Businesses hiring operations or admin staff

Return JSON array with up to 10 results, each containing:
- company_name: string
- location: string (city, state)
- website: string (URL if found)
- buying_signals: array of strings
- estimated_size: string (small/medium/large)

Return ONLY valid JSON array, no other text."""

        try:
            result = advanced_ai.search_with_perplexity(search_prompt)

            if result and result.get("answer"):
                answer = result["answer"]

                # Extract JSON from response
                json_match = re.search(r'\[[\s\S]*\]', answer)
                if json_match:
                    found_leads = json.loads(json_match.group())

                    for data in found_leads[:limit]:
                        signals = data.get("buying_signals", [])
                        if isinstance(signals, str):
                            signals = [signals]

                        lead = DiscoveredLead(
                            company_name=data.get("company_name", "Unknown"),
                            location=data.get("location", ""),
                            website=data.get("website"),
                            industry=default_industry,
                            source=LeadSource.WEB_SEARCH,
                            source_detail="AI web research - buying signals detected",
                            score=60.0,
                            estimated_value=5000.0,
                            signals=signals,
                            metadata={
                                "estimated_size": data.get("estimated_size", "unknown"),
                                "search_source": "perplexity",
                                "target_market": "saas" if is_saas_search else "roofing"
                            }
                        )
                        leads.append(lead)

            logger.info(f"Discovered {len(leads)} leads from web search (SaaS={is_saas_search})")
            return leads

        except Exception as e:
            logger.error("Web search discovery failed: %s", e)
            return []

    async def _discover_social_signals(self, limit: int = 15) -> list[DiscoveredLead]:
        """
        Discover leads from social media buying signals.
        """
        try:
            from ai_advanced_providers import advanced_ai
        except ImportError:
            logger.warning("AI advanced providers not available for social signals")
            return []

        leads = []
        keywords = [
            "need roofing software",
            "looking for roofing CRM",
            "roofing business automation"
        ]

        try:
            for keyword in keywords[:2]:  # Limit API calls
                search_prompt = f"""Search social media and forums for roofing businesses posting about: {keyword}

Look for:
1. Twitter/X posts from roofing contractors
2. LinkedIn posts about roofing business challenges
3. Reddit discussions in contractor/roofing subreddits

Find posts showing buying intent for roofing software.

Return JSON array with up to 5 signals, each containing:
- platform: string (twitter/linkedin/reddit)
- company_hint: string (company name if mentioned)
- post_summary: string (key content)
- intent_level: string (high/medium/low)

Return ONLY valid JSON array."""

                result = advanced_ai.search_with_perplexity(search_prompt)

                if result and result.get("answer"):
                    answer = result["answer"]
                    json_match = re.search(r'\[[\s\S]*\]', answer)

                    if json_match:
                        signals = json.loads(json_match.group())

                        for signal in signals:
                            if signal.get("company_hint"):
                                intent = signal.get("intent_level", "medium")
                                score = {"high": 75, "medium": 55, "low": 40}.get(intent, 55)

                                lead = DiscoveredLead(
                                    company_name=signal.get("company_hint", "Unknown"),
                                    source=LeadSource.SOCIAL_SIGNAL,
                                    source_detail=f"{signal.get('platform', 'social')} - {signal.get('post_summary', '')[:50]}",
                                    score=float(score),
                                    estimated_value=5000.0,
                                    signals=[f"social_{signal.get('platform', 'unknown')}", f"intent_{intent}", keyword.replace(" ", "_")],
                                    metadata={
                                        "platform": signal.get("platform"),
                                        "post_summary": signal.get("post_summary"),
                                        "intent_level": intent
                                    }
                                )
                                leads.append(lead)

            logger.info("Discovered %d leads from social signals", len(leads))
            return leads[:limit]

        except Exception as e:
            logger.error("Social signal discovery failed: %s", e)
            return []

    async def _discover_storm_leads(self, limit: int = 20) -> list[DiscoveredLead]:
        """
        Discover leads from weather/storm events.
        Would integrate with weather APIs for real storm data.
        """
        # This would integrate with weather APIs like NOAA, Tomorrow.io
        # For now, just check if there are configured storm sources
        pool = get_pool()

        try:
            storm_source = await pool.fetchrow("""
                SELECT * FROM lead_discovery_sources
                WHERE source_type = 'storm_tracker'
                AND enabled = true
            """)

            if not storm_source:
                logger.debug("No storm tracker source configured")
                return []

            # Would query weather API here
            # For now, return empty list as this requires external API integration
            return []

        except Exception as e:
            logger.error("Storm lead discovery failed: %s", e)
            return []

    # ====================
    # Scoring Methods
    # ====================

    async def _calculate_lead_score(self, lead: DiscoveredLead) -> float:
        """
        Calculate a comprehensive lead score based on multiple factors.

        Scoring breakdown (0-100):
        - Contact completeness: 0-20 points
        - Source quality: 0-20 points
        - Signals strength: 0-30 points
        - Estimated value: 0-15 points
        - Recency: 0-15 points
        """
        score = 0.0

        # Contact completeness (0-20)
        if lead.email:
            score += 8
        if lead.phone:
            score += 6
        if lead.company_name and lead.company_name != "Unknown":
            score += 4
        if lead.location:
            score += 2

        # Source quality (0-20)
        source_scores = {
            LeadSource.ERP_UPSELL: 18,
            LeadSource.ERP_REACTIVATION: 15,
            LeadSource.PARTNER_REFERRAL: 14,
            LeadSource.ERP_REFERRAL: 12,
            LeadSource.INBOUND_CALL: 12,
            LeadSource.WEB_FORM: 10,
            LeadSource.STORM_TRACKER: 8,
            LeadSource.WEB_SEARCH: 6,
            LeadSource.SOCIAL_SIGNAL: 5,
            LeadSource.MANUAL_ENTRY: 5,
        }
        score += source_scores.get(lead.source, 5)

        # Signals strength (0-30)
        high_value_signals = [
            "high_value_customer", "premium_customer", "repeat_customer",
            "loyal_customer", "intent_high", "recent_job"
        ]
        medium_value_signals = [
            "past_customer", "active_customer", "satisfied_customer",
            "intent_medium", "buying_signal"
        ]

        signal_score = 0
        for signal in lead.signals:
            if any(hv in signal for hv in high_value_signals):
                signal_score += 6
            elif any(mv in signal for mv in medium_value_signals):
                signal_score += 3
            else:
                signal_score += 1
        score += min(30, signal_score)

        # Estimated value (0-15)
        if lead.estimated_value >= 20000:
            score += 15
        elif lead.estimated_value >= 10000:
            score += 12
        elif lead.estimated_value >= 5000:
            score += 8
        elif lead.estimated_value >= 2000:
            score += 4

        # Recency (0-15) - newer discoveries get bonus
        days_old = (datetime.now(timezone.utc) - lead.discovered_at).days
        if days_old <= 1:
            score += 15
        elif days_old <= 7:
            score += 10
        elif days_old <= 30:
            score += 5

        return min(100.0, max(0.0, score))

    def _get_tier_from_score(self, score: float) -> LeadTier:
        """Determine lead tier based on score"""
        if score >= 80:
            return LeadTier.HOT
        elif score >= 60:
            return LeadTier.WARM
        elif score >= 40:
            return LeadTier.COOL
        else:
            return LeadTier.COLD

    def _is_excluded_email(self, email: str) -> bool:
        """Check if email domain is excluded"""
        if not email or "@" not in email:
            return True
        domain = email.split("@")[1].lower()
        return domain in self.criteria.excluded_domains

    # ====================
    # Utility Methods
    # ====================

    async def _deduplicate_leads(self, leads: list[DiscoveredLead]) -> list[DiscoveredLead]:
        """Remove duplicate leads based on email"""
        seen_emails = set()
        unique_leads = []

        pool = get_pool()

        # Also check existing leads in database
        existing_emails = set()
        try:
            rows = await pool.fetch("SELECT email FROM revenue_leads WHERE email IS NOT NULL")
            existing_emails = {row["email"].lower() for row in rows if row.get("email")}
        except Exception:
            pass

        for lead in leads:
            if not lead.email:
                # Allow leads without email if they have other identifiers
                if lead.company_name and lead.company_name != "Unknown":
                    unique_leads.append(lead)
                continue

            email_lower = lead.email.lower()
            if email_lower not in seen_emails and email_lower not in existing_emails:
                seen_emails.add(email_lower)
                unique_leads.append(lead)

        return unique_leads

    async def _start_discovery_run(self, sources: list[str]) -> str:
        """Start a new discovery run and return its ID"""
        pool = get_pool()
        run_id = str(uuid.uuid4())

        try:
            await pool.execute("""
                INSERT INTO lead_discovery_runs (id, source_type, metadata)
                VALUES ($1::uuid, $2, $3::jsonb)
            """, run_id, ",".join(sources), json.dumps({"sources": sources}))
        except Exception as e:
            logger.warning("Could not log discovery run start: %s", e)

        return run_id

    async def _complete_discovery_run(
        self,
        run_id: str,
        leads_found: int,
        leads_qualified: int
    ) -> None:
        """Complete a discovery run"""
        pool = get_pool()

        try:
            await pool.execute("""
                UPDATE lead_discovery_runs
                SET completed_at = NOW(),
                    status = 'completed',
                    leads_found = $2,
                    leads_qualified = $3
                WHERE id = $1::uuid
            """, run_id, leads_found, leads_qualified)
        except Exception as e:
            logger.warning("Could not complete discovery run: %s", e)

    async def _fail_discovery_run(self, run_id: str, error: str) -> None:
        """Mark a discovery run as failed"""
        pool = get_pool()

        try:
            await pool.execute("""
                UPDATE lead_discovery_runs
                SET completed_at = NOW(),
                    status = 'failed',
                    errors = errors || $2::jsonb
                WHERE id = $1::uuid
            """, run_id, json.dumps([{"error": error, "time": datetime.now(timezone.utc).isoformat()}]))
        except Exception as e:
            logger.warning("Could not fail discovery run: %s", e)

    async def _log_discovery_error(self, run_id: str, source: str, error: str) -> None:
        """Log an error during discovery"""
        pool = get_pool()

        try:
            await pool.execute("""
                UPDATE lead_discovery_runs
                SET errors = errors || $2::jsonb
                WHERE id = $1::uuid
            """, run_id, json.dumps([{
                "source": source,
                "error": error,
                "time": datetime.now(timezone.utc).isoformat()
            }]))
        except Exception as e:
            logger.warning("Could not log discovery error: %s", e)

    async def _update_source_stats(self, source: str, leads_found: int) -> None:
        """Update source statistics"""
        pool = get_pool()

        try:
            await pool.execute("""
                INSERT INTO lead_discovery_sources (source_type, source_name, leads_discovered, last_run_at)
                VALUES ($1, $1, $2, NOW())
                ON CONFLICT (source_type, source_name)
                DO UPDATE SET
                    leads_discovered = lead_discovery_sources.leads_discovered + $2,
                    last_run_at = NOW(),
                    updated_at = NOW()
            """, source, leads_found)
        except Exception as e:
            logger.warning("Could not update source stats: %s", e)

    async def _log_qualification(
        self,
        lead_id: str,
        previous_status: str,
        new_status: str,
        previous_score: float,
        new_score: float,
        reason: str
    ) -> None:
        """Log qualification decision for audit trail"""
        pool = get_pool()

        try:
            await pool.execute("""
                INSERT INTO lead_qualification_history (
                    lead_id, previous_status, new_status,
                    previous_score, new_score, reason
                ) VALUES ($1::uuid, $2, $3, $4, $5, $6)
            """, lead_id, previous_status, new_status, previous_score, new_score, reason)
        except Exception as e:
            logger.warning("Could not log qualification: %s", e)

    # ====================
    # Public Utility Methods
    # ====================

    async def get_discovery_stats(self) -> dict:
        """Get discovery statistics"""
        pool = get_pool()

        try:
            # Get source stats
            sources = await pool.fetch("""
                SELECT source_type, leads_discovered, leads_qualified,
                       conversion_rate, last_run_at
                FROM lead_discovery_sources
                WHERE enabled = true
                ORDER BY leads_discovered DESC
            """)

            # Get recent runs
            recent_runs = await pool.fetch("""
                SELECT id, source_type, started_at, completed_at, status,
                       leads_found, leads_qualified
                FROM lead_discovery_runs
                ORDER BY started_at DESC
                LIMIT 10
            """)

            # Get totals
            totals = await pool.fetchrow("""
                SELECT
                    SUM(leads_discovered) as total_discovered,
                    SUM(leads_qualified) as total_qualified,
                    AVG(conversion_rate) as avg_conversion_rate
                FROM lead_discovery_sources
            """)

            return {
                "sources": [dict(s) for s in sources],
                "recent_runs": [dict(r) for r in recent_runs],
                "totals": {
                    "discovered": totals["total_discovered"] or 0,
                    "qualified": totals["total_qualified"] or 0,
                    "conversion_rate": float(totals["avg_conversion_rate"] or 0)
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error("Failed to get discovery stats: %s", e)
            return {"error": str(e)}


# Singleton instance
_discovery_engine: Optional[LeadDiscoveryEngine] = None


def get_discovery_engine(tenant_id: Optional[str] = None) -> LeadDiscoveryEngine:
    """Get or create discovery engine instance"""
    global _discovery_engine
    if _discovery_engine is None or (tenant_id and _discovery_engine.tenant_id != tenant_id):
        _discovery_engine = LeadDiscoveryEngine(tenant_id)
    return _discovery_engine


# Export for use in other modules
__all__ = [
    "LeadDiscoveryEngine",
    "LeadSource",
    "LeadQualificationStatus",
    "LeadTier",
    "LeadQualificationCriteria",
    "DiscoveredLead",
    "get_discovery_engine"
]

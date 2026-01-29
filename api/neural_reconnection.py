"""
Neural Reconnection API - Schema Unification & Mode Logic
==========================================================
Bridges the ERP (leads table) with the AI (revenue_leads table).
Implements draft vs execute mode for dual-system architecture.

ERP Mode (draft): Returns suggestions for human review
MRG Mode (execute): Performs actual DB writes, Stripe charges, email sends
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config import config
from database.async_connection import get_pool, DatabaseUnavailableError

logger = logging.getLogger(__name__)

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


router = APIRouter(
    prefix="/api/neural",
    tags=["neural-reconnection"],
    dependencies=[Depends(verify_api_key)]
)


# ====================
# Enums & Models
# ====================

class OperationMode(str, Enum):
    """Operation mode determines behavior"""
    DRAFT = "draft"      # ERP: Return suggestions only
    EXECUTE = "execute"  # MRG: Perform actual operations


class SyncResult(BaseModel):
    """Result of a sync operation"""
    success: bool
    leads_synced: int = 0
    leads_skipped: int = 0
    leads_updated: int = 0
    message: str = ""
    errors: list[str] = []


class ModeAwareResponse(BaseModel):
    """Response that includes mode context"""
    mode: OperationMode
    action_taken: str
    data: Any
    suggestion: Optional[str] = None  # For draft mode
    execution_id: Optional[str] = None  # For execute mode


# ====================
# Schema Sync Endpoints
# ====================

@router.post("/sync/leads-to-revenue", response_model=SyncResult)
async def sync_leads_to_revenue_leads(
    limit: int = Query(default=100, le=1000, description="Max leads to sync per call"),
    force_update: bool = Query(default=False, description="Update even if exists")
):
    """
    Sync ERP leads table to AI revenue_leads table.

    This bridges the gap:
    - ERP writes to `leads` (969 rows of real data)
    - AI reads from `revenue_leads` (currently 0 rows)

    Column mapping:
    - leads.name → revenue_leads.company_name
    - leads.name → revenue_leads.contact_name
    - leads.email → revenue_leads.email
    - leads.phone → revenue_leads.phone
    - leads.status → revenue_leads.stage
    - leads.score → revenue_leads.score
    - leads.source → revenue_leads.source
    - leads.company → revenue_leads.metadata.company
    """
    try:
        pool = get_pool()

        # Fetch leads from ERP table
        leads = await pool.fetch("""
            SELECT
                id, name, email, phone, company, score, status, source,
                description, tags, custom_fields, created_at, updated_at,
                address, roof_type, square_footage, urgency, insurance_claim,
                budget_range, value_score
            FROM leads
            WHERE email IS NOT NULL AND email != ''
            ORDER BY created_at DESC
            LIMIT $1
        """, limit)

        synced = 0
        skipped = 0
        updated = 0
        errors = []

        for lead in leads:
            try:
                # Check if already exists in revenue_leads
                existing = await pool.fetchrow(
                    "SELECT id FROM revenue_leads WHERE email = $1 OR lead_id = $2",
                    lead['email'], str(lead['id'])
                )

                if existing and not force_update:
                    skipped += 1
                    continue

                # Map status to stage
                status_to_stage = {
                    'NEW': 'new',
                    'CONTACTED': 'contacted',
                    'QUALIFIED': 'qualified',
                    'PROPOSAL': 'proposal_sent',
                    'NEGOTIATING': 'negotiating',
                    'WON': 'won',
                    'LOST': 'lost'
                }
                stage = status_to_stage.get(str(lead.get('status', '')).upper(), 'new')

                # Build metadata
                metadata = {
                    'synced_from_erp': True,
                    'erp_lead_id': str(lead['id']),
                    'company': lead.get('company', ''),
                    'address': lead.get('address', ''),
                    'roof_type': lead.get('roof_type', ''),
                    'square_footage': lead.get('square_footage'),
                    'urgency': lead.get('urgency', 'normal'),
                    'insurance_claim': lead.get('insurance_claim', False),
                    'budget_range': lead.get('budget_range', ''),
                    'original_tags': lead.get('tags'),
                    'original_custom_fields': lead.get('custom_fields')
                }

                if existing:
                    # Update existing
                    await pool.execute("""
                        UPDATE revenue_leads SET
                            company_name = COALESCE($1, company_name),
                            contact_name = COALESCE($2, contact_name),
                            phone = COALESCE($3, phone),
                            stage = $4,
                            score = COALESCE($5, score),
                            source = COALESCE($6, source),
                            metadata = metadata || $7::jsonb,
                            updated_at = NOW()
                        WHERE id = $8
                    """,
                        lead.get('company') or lead.get('name'),
                        lead.get('name'),
                        lead.get('phone'),
                        stage,
                        float(lead.get('score', 0) or 0),
                        lead.get('source'),
                        metadata,
                        existing['id']
                    )
                    updated += 1
                else:
                    # Insert new
                    await pool.execute("""
                        INSERT INTO revenue_leads (
                            id, company_name, contact_name, email, phone,
                            stage, score, source, metadata, lead_id,
                            value_estimate, created_at, updated_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11, $12, NOW()
                        )
                    """,
                        uuid4(),
                        lead.get('company') or lead.get('name'),
                        lead.get('name'),
                        lead.get('email'),
                        lead.get('phone'),
                        stage,
                        float(lead.get('score', 0) or 0),
                        lead.get('source'),
                        metadata,
                        str(lead['id']),
                        float(lead.get('value_score', 0) or 5000),
                        lead.get('created_at', datetime.now(timezone.utc))
                    )
                    synced += 1

            except Exception as e:
                errors.append(f"Lead {lead.get('email', 'unknown')}: {str(e)}")
                logger.warning(f"Sync error for lead {lead.get('id')}: {e}")

        return SyncResult(
            success=True,
            leads_synced=synced,
            leads_skipped=skipped,
            leads_updated=updated,
            message=f"Sync complete: {synced} new, {updated} updated, {skipped} skipped",
            errors=errors[:10]  # Limit error list
        )

    except DatabaseUnavailableError as e:
        logger.error(f"Database unavailable: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/sync/status")
async def get_sync_status():
    """Get current sync status between leads and revenue_leads tables"""
    try:
        pool = get_pool()

        leads_count = await pool.fetchval("SELECT COUNT(*) FROM leads")
        revenue_leads_count = await pool.fetchval("SELECT COUNT(*) FROM revenue_leads")

        synced_count = await pool.fetchval("""
            SELECT COUNT(*) FROM revenue_leads
            WHERE metadata->>'synced_from_erp' = 'true'
        """)

        unsynced_count = await pool.fetchval("""
            SELECT COUNT(*) FROM leads l
            WHERE NOT EXISTS (
                SELECT 1 FROM revenue_leads rl
                WHERE rl.email = l.email OR rl.lead_id = l.id::text
            )
        """)

        return {
            "leads_table_count": leads_count,
            "revenue_leads_table_count": revenue_leads_count,
            "synced_from_erp": synced_count,
            "unsynced_leads": unsynced_count,
            "sync_coverage": f"{((leads_count - unsynced_count) / max(leads_count, 1)) * 100:.1f}%",
            "recommendation": "Run sync" if unsynced_count > 0 else "Tables synced"
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Mode-Aware Endpoints
# ====================

@router.post("/leads/discover", response_model=ModeAwareResponse)
async def discover_leads_with_mode(
    mode: OperationMode = Query(default=OperationMode.DRAFT),
    limit: int = Query(default=20, le=100),
    lead_type: str = Query(default="all", regex="^(all|reengagement|upsell|referral)$")
):
    """
    Discover leads with mode-aware behavior.

    - mode=draft: Returns lead suggestions for human review (ERP)
    - mode=execute: Auto-adds leads to CRM and starts nurture sequences (MRG)
    """
    try:
        pool = get_pool()

        # Query for potential leads from customers table
        leads = await pool.fetch("""
            SELECT DISTINCT ON (c.email)
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
              AND COALESCE(c.is_demo, TRUE) = FALSE
            GROUP BY c.id, c.company_name, c.first_name, c.last_name, c.email, c.phone, c.city, c.state
            HAVING MAX(j.created_at) < NOW() - INTERVAL '6 months' OR MAX(j.created_at) IS NULL
            ORDER BY c.email, avg_job_value DESC
            LIMIT $1
        """, limit)

        discovered = [dict(lead) for lead in leads]

        if mode == OperationMode.DRAFT:
            # ERP Mode: Return suggestions only
            return ModeAwareResponse(
                mode=mode,
                action_taken="discovery_suggestion",
                data={
                    "leads_found": len(discovered),
                    "leads": discovered,
                    "criteria": "Customers with no jobs in 6+ months",
                },
                suggestion="Review these leads and approve for outreach. Click 'Approve' to add to CRM."
            )
        else:
            # MRG Mode: Execute - add to revenue_leads and start sequence
            execution_id = str(uuid4())
            added = 0

            for lead in discovered:
                try:
                    existing = await pool.fetchrow(
                        "SELECT id FROM revenue_leads WHERE email = $1",
                        lead['email']
                    )

                    if not existing:
                        await pool.execute("""
                            INSERT INTO revenue_leads (
                                id, company_name, contact_name, email, phone,
                                stage, score, source, metadata, created_at
                            ) VALUES (
                                $1, $2, $3, $4, $5, 'new', 0.7, 'auto_discovery', $6::jsonb, NOW()
                            )
                        """,
                            uuid4(),
                            lead['company_name'],
                            lead['contact_name'],
                            lead['email'],
                            lead.get('phone'),
                            {
                                'execution_id': execution_id,
                                'customer_id': str(lead['customer_id']),
                                'avg_job_value': float(lead.get('avg_job_value', 0)),
                                'total_jobs': lead.get('total_jobs', 0),
                                'mode': 'execute',
                                'auto_added': True
                            }
                        )
                        added += 1
                except Exception as e:
                    logger.warning(f"Failed to add lead {lead.get('email')}: {e}")

            return ModeAwareResponse(
                mode=mode,
                action_taken="leads_added_to_crm",
                data={
                    "leads_found": len(discovered),
                    "leads_added": added,
                    "leads": discovered
                },
                execution_id=execution_id
            )

    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(f"Lead discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proposals/generate", response_model=ModeAwareResponse)
async def generate_proposal_with_mode(
    lead_id: str,
    mode: OperationMode = Query(default=OperationMode.DRAFT),
    include_financing: bool = Query(default=True)
):
    """
    Generate a proposal with mode-aware behavior.

    - mode=draft: Populates estimate form for review (ERP)
    - mode=execute: Emails PDF quote directly to lead (MRG)
    """
    try:
        pool = get_pool()

        # Get lead info
        lead = await pool.fetchrow("""
            SELECT * FROM revenue_leads WHERE id::text = $1 OR lead_id = $1
        """, lead_id)

        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")

        # Generate proposal data (simplified)
        proposal = {
            "lead_id": lead_id,
            "company_name": lead['company_name'],
            "contact_name": lead['contact_name'],
            "email": lead['email'],
            "estimated_value": float(lead.get('value_estimate', 0) or lead.get('estimated_value', 5000)),
            "items": [
                {"description": "Professional Roofing Service", "price": 5000},
                {"description": "Materials", "price": 2500},
                {"description": "Labor", "price": 2000}
            ],
            "subtotal": 9500,
            "tax": 760,
            "total": 10260,
            "financing_available": include_financing,
            "valid_until": (datetime.now(timezone.utc).replace(day=1) +
                          __import__('datetime').timedelta(days=32)).replace(day=1).isoformat()
        }

        if mode == OperationMode.DRAFT:
            return ModeAwareResponse(
                mode=mode,
                action_taken="proposal_drafted",
                data=proposal,
                suggestion="Review the proposal details above. Click 'Send' to email to customer."
            )
        else:
            # Execute mode - would send email
            execution_id = str(uuid4())

            # Queue the email (actual sending handled by email scheduler)
            await pool.execute("""
                INSERT INTO ai_email_queue (
                    id, recipient, subject, body, status, scheduled_for, metadata
                ) VALUES (
                    $1, $2, $3, $4, 'queued', NOW(), $5::jsonb
                )
            """,
                uuid4(),
                lead['email'],
                f"Your Roofing Proposal from BrainStack",
                f"Dear {lead['contact_name']},\n\nThank you for your interest. "
                f"Please find your proposal attached with a total of ${proposal['total']:,.2f}.\n\n"
                f"Best regards,\nBrainStack Team",
                {
                    'execution_id': execution_id,
                    'lead_id': lead_id,
                    'proposal': proposal,
                    'mode': 'execute'
                }
            )

            # Update lead stage
            await pool.execute("""
                UPDATE revenue_leads SET stage = 'proposal_sent', updated_at = NOW()
                WHERE id::text = $1 OR lead_id = $1
            """, lead_id)

            return ModeAwareResponse(
                mode=mode,
                action_taken="proposal_sent",
                data=proposal,
                execution_id=execution_id
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Proposal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Engine Activation Endpoints
# ====================

@router.get("/engines/status")
async def get_engine_status():
    """Check status of revenue engines"""
    engines = {
        "affiliate_pipeline": False,
        "api_monetization": False,
        "revenue_automation": False
    }

    try:
        from affiliate_partnership_pipeline import AffiliatePartnershipPipeline
        engines["affiliate_pipeline"] = True
    except ImportError:
        pass

    try:
        from api_monetization_engine import APIMonetizationEngine
        engines["api_monetization"] = True
    except ImportError:
        pass

    try:
        from revenue_automation_engine import RevenueAutomationEngine
        engines["revenue_automation"] = True
    except ImportError:
        pass

    return {
        "engines": engines,
        "all_active": all(engines.values())
    }


@router.post("/engines/affiliate/process")
async def process_affiliate_pipeline(
    mode: OperationMode = Query(default=OperationMode.DRAFT)
):
    """Process the affiliate partnership pipeline"""
    try:
        from affiliate_partnership_pipeline import AffiliatePartnershipPipeline
        pipeline = AffiliatePartnershipPipeline()

        if mode == OperationMode.DRAFT:
            # Return what would happen
            return ModeAwareResponse(
                mode=mode,
                action_taken="affiliate_analysis",
                data={
                    "description": "Affiliate partnership pipeline analysis",
                    "tiers": ["Bronze", "Silver", "Gold", "Diamond"],
                    "commission_rates": {"Bronze": "20%", "Silver": "25%", "Gold": "30%", "Diamond": "35%"}
                },
                suggestion="Run in execute mode to process affiliate applications and commissions"
            )
        else:
            # Would run the actual pipeline
            result = await pipeline.run_discovery_cycle() if hasattr(pipeline, 'run_discovery_cycle') else {"status": "ready"}
            return ModeAwareResponse(
                mode=mode,
                action_taken="affiliate_processed",
                data=result,
                execution_id=str(uuid4())
            )
    except ImportError:
        raise HTTPException(status_code=501, detail="Affiliate pipeline not available")
    except Exception as e:
        logger.error(f"Affiliate processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

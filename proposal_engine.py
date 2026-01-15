#!/usr/bin/env python3
"""
Proposal Engine
===============
Creates, manages, and sends proposals for BrainOps services.

Features:
- Standardized offer catalog
- Proposal draft → approval → send workflow
- Invoice-ready artifacts
- Integration with Pipeline State Machine

Part of Revenue Perfection Session.
"""

import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
import json

logger = logging.getLogger(__name__)


class ProposalStatus(str, Enum):
    """Proposal lifecycle states."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SENT = "sent"
    VIEWED = "viewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OfferTier(str, Enum):
    """Standard offer tiers."""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


# =============================================================================
# OFFER CATALOG - The sellable products/services
# =============================================================================

OFFER_CATALOG = {
    # MRG - MyRoofGenius SaaS
    "mrg_starter": {
        "id": "mrg_starter",
        "name": "MyRoofGenius Starter",
        "product": "MyRoofGenius",
        "tier": OfferTier.STARTER.value,
        "deliverables": [
            "Satellite roof measurement access",
            "Up to 50 reports/month",
            "Email support",
            "Basic CRM integration"
        ],
        "term_days": 30,
        "price": Decimal("299.00"),
        "price_unit": "month",
        "guarantees": ["14-day money-back guarantee"],
        "limitations": ["50 reports/month limit", "Email support only"]
    },
    "mrg_pro": {
        "id": "mrg_pro",
        "name": "MyRoofGenius Professional",
        "product": "MyRoofGenius",
        "tier": OfferTier.PROFESSIONAL.value,
        "deliverables": [
            "Unlimited satellite roof measurements",
            "AI-powered proposal generation",
            "Priority support",
            "Full CRM integration",
            "Team accounts (up to 5 users)"
        ],
        "term_days": 30,
        "price": Decimal("799.00"),
        "price_unit": "month",
        "guarantees": ["14-day money-back guarantee", "99.5% uptime SLA"],
        "limitations": ["5 user limit"]
    },

    # BrainOps AI OS
    "brainops_automation": {
        "id": "brainops_automation",
        "name": "BrainOps Managed Automation",
        "product": "BrainOps AI OS",
        "tier": OfferTier.PROFESSIONAL.value,
        "deliverables": [
            "Custom AI agent setup",
            "Lead discovery & enrichment automation",
            "Automated email sequences",
            "Revenue pipeline monitoring",
            "Weekly optimization reports"
        ],
        "term_days": 90,
        "price": Decimal("2500.00"),
        "price_unit": "month",
        "guarantees": ["ROI guarantee: 3x pipeline value in 90 days or 50% refund"],
        "limitations": ["Requires 90-day commitment"]
    },
    "brainops_starter": {
        "id": "brainops_starter",
        "name": "BrainOps API Access",
        "product": "BrainOps AI OS",
        "tier": OfferTier.STARTER.value,
        "deliverables": [
            "API access to AI agents",
            "10,000 API calls/month",
            "Documentation & examples",
            "Community support"
        ],
        "term_days": 30,
        "price": Decimal("199.00"),
        "price_unit": "month",
        "guarantees": ["7-day trial included"],
        "limitations": ["10,000 API calls/month", "Community support only"]
    },

    # ERP-as-a-Service
    "erp_implementation": {
        "id": "erp_implementation",
        "name": "Weathercraft ERP Implementation",
        "product": "Weathercraft ERP",
        "tier": OfferTier.ENTERPRISE.value,
        "deliverables": [
            "Full ERP platform setup",
            "Data migration assistance",
            "Custom workflow configuration",
            "Staff training (up to 20 users)",
            "90-day support included"
        ],
        "term_days": 60,
        "price": Decimal("15000.00"),
        "price_unit": "one-time",
        "guarantees": ["Go-live within 60 days or 20% discount"],
        "limitations": ["Requires dedicated project manager"]
    },
    "erp_subscription": {
        "id": "erp_subscription",
        "name": "Weathercraft ERP Subscription",
        "product": "Weathercraft ERP",
        "tier": OfferTier.PROFESSIONAL.value,
        "deliverables": [
            "Full ERP platform access",
            "Unlimited users",
            "All modules included",
            "Priority support",
            "Monthly updates"
        ],
        "term_days": 30,
        "price": Decimal("1500.00"),
        "price_unit": "month",
        "guarantees": ["99.9% uptime SLA"],
        "limitations": ["Annual commitment recommended"]
    }
}


@dataclass
class Proposal:
    """Represents a proposal for a lead."""
    id: str
    lead_id: str
    offer_id: str
    status: ProposalStatus
    created_at: datetime
    updated_at: datetime

    # Client info
    client_name: str
    client_email: str
    client_company: str

    # Offer details
    offer_name: str
    deliverables: list[str]
    price: Decimal
    price_unit: str
    term_days: int
    guarantees: list[str]
    limitations: list[str]

    # Customizations
    custom_notes: Optional[str] = None
    discount_percent: Optional[Decimal] = None
    final_price: Optional[Decimal] = None

    # Lifecycle
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    viewed_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None

    # Artifact
    public_link: Optional[str] = None
    pdf_url: Optional[str] = None


class ProposalEngine:
    """
    Manages proposal lifecycle from draft to acceptance.
    """

    def __init__(self):
        self._pool = None

    def _get_pool(self):
        """Get database pool."""
        try:
            from database.async_connection import get_pool
            return get_pool()
        except Exception as e:
            logger.error(f"Failed to get database pool: {e}")
            return None

    def get_offers(self) -> dict:
        """Return the offer catalog."""
        return OFFER_CATALOG

    async def draft_proposal(
        self,
        lead_id: str,
        offer_id: str,
        custom_notes: Optional[str] = None,
        discount_percent: Optional[float] = None
    ) -> tuple[bool, str, Optional[Proposal]]:
        """
        Create a draft proposal for a lead.

        Args:
            lead_id: UUID of the lead
            offer_id: ID from OFFER_CATALOG
            custom_notes: Optional custom notes for the proposal
            discount_percent: Optional discount (0-100)

        Returns:
            (success, message, proposal)
        """
        pool = self._get_pool()
        if not pool:
            return False, "Database not available", None

        # Validate offer
        offer = OFFER_CATALOG.get(offer_id)
        if not offer:
            return False, f"Invalid offer_id: {offer_id}", None

        # Get lead info
        lead = await pool.fetchrow("""
            SELECT id, company_name, contact_name, email
            FROM revenue_leads
            WHERE id = $1
        """, uuid.UUID(lead_id) if isinstance(lead_id, str) else lead_id)

        if not lead:
            return False, f"Lead not found: {lead_id}", None

        # Calculate final price
        base_price = Decimal(str(offer["price"]))
        final_price = base_price
        if discount_percent and discount_percent > 0:
            discount = base_price * Decimal(str(discount_percent)) / 100
            final_price = base_price - discount

        proposal_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=30)

        proposal = Proposal(
            id=proposal_id,
            lead_id=str(lead["id"]),
            offer_id=offer_id,
            status=ProposalStatus.DRAFT,
            created_at=now,
            updated_at=now,
            client_name=lead["contact_name"] or lead["company_name"],
            client_email=lead["email"],
            client_company=lead["company_name"],
            offer_name=offer["name"],
            deliverables=offer["deliverables"],
            price=base_price,
            price_unit=offer["price_unit"],
            term_days=offer["term_days"],
            guarantees=offer["guarantees"],
            limitations=offer["limitations"],
            custom_notes=custom_notes,
            discount_percent=Decimal(str(discount_percent)) if discount_percent else None,
            final_price=final_price,
            expires_at=expires_at
        )

        # Store in database
        try:
            await pool.execute("""
                INSERT INTO ai_proposals (
                    id, lead_id, offer_id, status, client_name, client_email, client_company,
                    offer_name, deliverables, price, price_unit, term_days, guarantees, limitations,
                    custom_notes, discount_percent, final_price, expires_at, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            """,
                uuid.UUID(proposal.id),
                uuid.UUID(proposal.lead_id),
                offer_id,
                proposal.status.value,
                proposal.client_name,
                proposal.client_email,
                proposal.client_company,
                proposal.offer_name,
                json.dumps(proposal.deliverables),
                float(proposal.price),
                proposal.price_unit,
                proposal.term_days,
                json.dumps(proposal.guarantees),
                json.dumps(proposal.limitations),
                proposal.custom_notes,
                float(proposal.discount_percent) if proposal.discount_percent else None,
                float(proposal.final_price) if proposal.final_price else None,
                proposal.expires_at,
                proposal.created_at,
                proposal.updated_at
            )

            logger.info(f"Proposal {proposal_id[:8]}... drafted for lead {lead_id[:8]}...")
            return True, "Proposal drafted successfully", proposal

        except Exception as e:
            logger.error(f"Failed to create proposal: {e}")
            return False, str(e), None

    async def submit_for_approval(self, proposal_id: str) -> tuple[bool, str]:
        """Submit a draft proposal for approval."""
        pool = self._get_pool()
        if not pool:
            return False, "Database not available"

        try:
            result = await pool.execute("""
                UPDATE ai_proposals
                SET status = $1, updated_at = $2
                WHERE id = $3 AND status = 'draft'
            """, ProposalStatus.PENDING_APPROVAL.value, datetime.now(timezone.utc), uuid.UUID(proposal_id))

            if result == "UPDATE 0":
                return False, "Proposal not found or not in draft status"

            # Create approval item
            await pool.execute("""
                INSERT INTO ai_improvement_proposals (id, title, description, status, created_at, updated_at)
                VALUES ($1, $2, $3, 'proposed', $4, $4)
            """,
                uuid.uuid4(),
                f"Approve Proposal {proposal_id[:8]}",
                f"Revenue proposal pending human approval. Proposal ID: {proposal_id}",
                datetime.now(timezone.utc)
            )

            logger.info(f"Proposal {proposal_id[:8]}... submitted for approval")
            return True, "Submitted for approval"

        except Exception as e:
            logger.error(f"Failed to submit for approval: {e}")
            return False, str(e)

    async def approve_proposal(self, proposal_id: str, approved_by: str) -> tuple[bool, str]:
        """Approve a proposal (human-in-the-loop)."""
        pool = self._get_pool()
        if not pool:
            return False, "Database not available"

        now = datetime.now(timezone.utc)

        try:
            result = await pool.execute("""
                UPDATE ai_proposals
                SET status = $1, approved_by = $2, approved_at = $3, updated_at = $3
                WHERE id = $4 AND status = 'pending_approval'
            """, ProposalStatus.APPROVED.value, approved_by, now, uuid.UUID(proposal_id))

            if result == "UPDATE 0":
                return False, "Proposal not found or not pending approval"

            logger.info(f"Proposal {proposal_id[:8]}... approved by {approved_by}")
            return True, "Proposal approved"

        except Exception as e:
            logger.error(f"Failed to approve proposal: {e}")
            return False, str(e)

    async def send_proposal(self, proposal_id: str) -> tuple[bool, str, Optional[str]]:
        """
        Send an approved proposal to the client.

        Returns (success, message, public_link)
        """
        pool = self._get_pool()
        if not pool:
            return False, "Database not available", None

        # Get proposal
        proposal = await pool.fetchrow("""
            SELECT * FROM ai_proposals WHERE id = $1 AND status = 'approved'
        """, uuid.UUID(proposal_id))

        if not proposal:
            return False, "Proposal not found or not approved", None

        now = datetime.now(timezone.utc)
        public_link = f"https://brainops-ai-agents.onrender.com/proposals/view/{proposal_id}"

        # Generate HTML content (simplified for now)
        html_content = self._generate_proposal_html(proposal)

        try:
            # Update proposal status
            await pool.execute("""
                UPDATE ai_proposals
                SET status = $1, sent_at = $2, public_link = $3, updated_at = $2
                WHERE id = $4
            """, ProposalStatus.SENT.value, now, public_link, uuid.UUID(proposal_id))

            # Queue email
            await pool.execute("""
                INSERT INTO ai_email_queue (id, recipient_email, subject, body, status, send_after, created_at)
                VALUES ($1, $2, $3, $4, 'queued', $5, $5)
            """,
                uuid.uuid4(),
                proposal["client_email"],
                f"Proposal from BrainOps: {proposal['offer_name']}",
                f"Hi {proposal['client_name']},\n\nPlease review your proposal here: {public_link}\n\nBest,\nBrainOps Team",
                now
            )

            # Update lead state via state machine
            from pipeline_state_machine import get_state_machine, PipelineState
            sm = get_state_machine()
            await sm.transition(
                str(proposal["lead_id"]),
                PipelineState.PROPOSAL_SENT,
                trigger="proposal_sent",
                actor="system:proposal_engine",
                metadata={"proposal_id": proposal_id}
            )

            logger.info(f"Proposal {proposal_id[:8]}... sent to {proposal['client_email'][:3]}***")
            return True, "Proposal sent", public_link

        except Exception as e:
            logger.error(f"Failed to send proposal: {e}")
            return False, str(e), None

    def _generate_proposal_html(self, proposal: dict) -> str:
        """Generate simple HTML proposal content."""
        deliverables = json.loads(proposal["deliverables"]) if isinstance(proposal["deliverables"], str) else proposal["deliverables"]
        guarantees = json.loads(proposal["guarantees"]) if isinstance(proposal["guarantees"], str) else proposal["guarantees"]

        deliverables_html = "\n".join([f"<li>{d}</li>" for d in deliverables])
        guarantees_html = "\n".join([f"<li>{g}</li>" for g in guarantees])

        return f"""
        <html>
        <head><title>Proposal: {proposal['offer_name']}</title></head>
        <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1>Proposal for {proposal['client_company']}</h1>
            <p>Dear {proposal['client_name']},</p>
            <h2>{proposal['offer_name']}</h2>
            <h3>What You Get:</h3>
            <ul>{deliverables_html}</ul>
            <h3>Investment:</h3>
            <p style="font-size: 24px; color: #2196F3;">
                ${float(proposal['final_price'] or proposal['price']):.2f} / {proposal['price_unit']}
            </p>
            <h3>Our Guarantees:</h3>
            <ul>{guarantees_html}</ul>
            {f"<h3>Notes:</h3><p>{proposal['custom_notes']}</p>" if proposal.get('custom_notes') else ""}
            <p>Valid until: {proposal['expires_at'].strftime('%B %d, %Y') if proposal['expires_at'] else 'N/A'}</p>
            <hr>
            <p>To accept this proposal, reply to this email or contact us directly.</p>
            <p>Best regards,<br>BrainOps Team</p>
        </body>
        </html>
        """

    async def get_proposal(self, proposal_id: str) -> Optional[dict]:
        """Get a proposal by ID."""
        pool = self._get_pool()
        if not pool:
            return None

        row = await pool.fetchrow("""
            SELECT * FROM ai_proposals WHERE id = $1
        """, uuid.UUID(proposal_id))

        if row:
            return dict(row)
        return None

    async def get_proposals_for_lead(self, lead_id: str) -> list[dict]:
        """Get all proposals for a lead."""
        pool = self._get_pool()
        if not pool:
            return []

        rows = await pool.fetch("""
            SELECT * FROM ai_proposals
            WHERE lead_id = $1
            ORDER BY created_at DESC
        """, uuid.UUID(lead_id))

        return [dict(r) for r in rows]


# Check if ai_proposals table exists and create if needed
async def ensure_proposals_table():
    """Ensure the ai_proposals table exists."""
    try:
        from database.async_connection import get_pool
        pool = get_pool()
        if not pool:
            return False

        await pool.execute("""
            CREATE TABLE IF NOT EXISTS ai_proposals (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                lead_id UUID REFERENCES revenue_leads(id),
                offer_id VARCHAR(100) NOT NULL,
                status VARCHAR(50) DEFAULT 'draft',
                client_name VARCHAR(255),
                client_email VARCHAR(255),
                client_company VARCHAR(255),
                offer_name VARCHAR(255),
                deliverables JSONB DEFAULT '[]',
                price DECIMAL(12,2),
                price_unit VARCHAR(50),
                term_days INTEGER,
                guarantees JSONB DEFAULT '[]',
                limitations JSONB DEFAULT '[]',
                custom_notes TEXT,
                discount_percent DECIMAL(5,2),
                final_price DECIMAL(12,2),
                approved_by VARCHAR(255),
                approved_at TIMESTAMPTZ,
                sent_at TIMESTAMPTZ,
                expires_at TIMESTAMPTZ,
                viewed_at TIMESTAMPTZ,
                responded_at TIMESTAMPTZ,
                public_link TEXT,
                pdf_url TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await pool.execute("""
            CREATE INDEX IF NOT EXISTS idx_ai_proposals_lead_id ON ai_proposals(lead_id);
            CREATE INDEX IF NOT EXISTS idx_ai_proposals_status ON ai_proposals(status);
        """)

        logger.info("ai_proposals table ensured")
        return True

    except Exception as e:
        logger.error(f"Failed to ensure ai_proposals table: {e}")
        return False


# Singleton instance
_proposal_engine: Optional[ProposalEngine] = None


def get_proposal_engine() -> ProposalEngine:
    """Get singleton proposal engine instance."""
    global _proposal_engine
    if _proposal_engine is None:
        _proposal_engine = ProposalEngine()
    return _proposal_engine

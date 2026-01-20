#!/usr/bin/env python3
"""
Payment Capture Module
======================
Handles payment collection for proposals and revenue tracking.

Features:
- Stripe checkout session creation
- Manual payment path (fallback)
- Invoice generation
- Payment status tracking
- Integration with pipeline state machine

Part of Revenue Perfection Session.
"""

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Optional
import json

logger = logging.getLogger(__name__)

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY_LIVE")
STRIPE_AVAILABLE = bool(STRIPE_SECRET_KEY)


@dataclass
class Invoice:
    """Represents an invoice for a proposal."""
    id: str
    proposal_id: str
    lead_id: str
    amount: Decimal
    currency: str
    status: str  # pending, sent, paid, cancelled
    stripe_checkout_id: Optional[str]
    stripe_payment_intent: Optional[str]
    payment_link: Optional[str]
    due_date: datetime
    created_at: datetime
    paid_at: Optional[datetime]


class PaymentCapture:
    """
    Handles payment collection for proposals.
    """

    def __init__(self):
        self._pool = None
        self._stripe = None

    def _get_pool(self):
        """Get database pool."""
        try:
            from database.async_connection import get_pool
            return get_pool()
        except Exception as e:
            logger.error(f"Failed to get database pool: {e}")
            return None

    def _get_stripe(self):
        """Get Stripe client."""
        if not STRIPE_AVAILABLE:
            return None
        if self._stripe is None:
            try:
                import stripe
                stripe.api_key = STRIPE_SECRET_KEY
                self._stripe = stripe
            except ImportError:
                logger.warning("Stripe SDK not installed")
        return self._stripe

    async def create_invoice(
        self,
        proposal_id: str,
        due_days: int = 14
    ) -> tuple[bool, str, Optional[Invoice]]:
        """
        Create an invoice from an approved/sent proposal.

        Args:
            proposal_id: UUID of the proposal
            due_days: Days until invoice is due

        Returns:
            (success, message, invoice)
        """
        pool = self._get_pool()
        if not pool:
            return False, "Database not available", None

        # Get proposal
        proposal = await pool.fetchrow("""
            SELECT * FROM ai_proposals
            WHERE id = $1 AND status IN ('approved', 'sent')
        """, uuid.UUID(proposal_id))

        if not proposal:
            return False, "Proposal not found or not in approved/sent status", None

        invoice_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        due_date = now + timedelta(days=due_days)

        amount = Decimal(str(proposal["final_price"] or proposal["price"] or 0))

        invoice = Invoice(
            id=invoice_id,
            proposal_id=str(proposal["id"]),
            lead_id=str(proposal["lead_id"]),
            amount=amount,
            currency="USD",
            status="pending",
            stripe_checkout_id=None,
            stripe_payment_intent=None,
            payment_link=None,
            due_date=due_date,
            created_at=now,
            paid_at=None
        )

        # Create Stripe checkout session if available
        stripe = self._get_stripe()
        if stripe:
            try:
                session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{
                        'price_data': {
                            'currency': 'usd',
                            'product_data': {
                                'name': proposal["offer_name"],
                                'description': f"Proposal for {proposal['client_company']}"
                            },
                            'unit_amount': int(amount * 100),  # Convert to cents
                        },
                        'quantity': 1,
                    }],
                    mode='payment',
                    success_url=f"https://brainops-ai-agents.onrender.com/payments/success?invoice_id={invoice_id}",
                    cancel_url=f"https://brainops-ai-agents.onrender.com/payments/cancel?invoice_id={invoice_id}",
                    customer_email=proposal["client_email"],
                    metadata={
                        'invoice_id': invoice_id,
                        'proposal_id': proposal_id,
                        'lead_id': str(proposal["lead_id"])
                    }
                )

                invoice.stripe_checkout_id = session.id
                invoice.payment_link = session.url
                logger.info(f"Created Stripe checkout session: {session.id}")

            except Exception as e:
                logger.error(f"Stripe checkout creation failed: {e}")
                # Continue with manual payment path

        # If no Stripe, generate manual payment link
        if not invoice.payment_link:
            invoice.payment_link = f"https://brainops-ai-agents.onrender.com/payments/manual/{invoice_id}"

        # Store invoice
        try:
            await pool.execute("""
                INSERT INTO ai_invoices (
                    id, proposal_id, lead_id, amount, currency, status,
                    stripe_checkout_id, stripe_payment_intent, payment_link,
                    due_date, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                uuid.UUID(invoice.id),
                uuid.UUID(invoice.proposal_id),
                uuid.UUID(invoice.lead_id),
                float(invoice.amount),
                invoice.currency,
                invoice.status,
                invoice.stripe_checkout_id,
                invoice.stripe_payment_intent,
                invoice.payment_link,
                invoice.due_date,
                invoice.created_at
            )

            # Update lead state
            from pipeline_state_machine import get_state_machine, PipelineState
            sm = get_state_machine()
            await sm.transition(
                invoice.lead_id,
                PipelineState.WON_INVOICE_PENDING,
                trigger="invoice_created",
                actor="system:payment_capture",
                metadata={"invoice_id": invoice_id, "amount": float(amount)}
            )

            logger.info(f"Invoice {invoice_id[:8]}... created for proposal {proposal_id[:8]}...")
            return True, "Invoice created", invoice

        except Exception as e:
            logger.error(f"Failed to create invoice: {e}")
            return False, str(e), None

    async def send_invoice(self, invoice_id: str) -> tuple[bool, str]:
        """
        Send an invoice to the client.
        """
        pool = self._get_pool()
        if not pool:
            return False, "Database not available"

        invoice = await pool.fetchrow("""
            SELECT i.*, p.client_email, p.client_name
            FROM ai_invoices i
            JOIN ai_proposals p ON i.proposal_id = p.id
            WHERE i.id = $1
        """, uuid.UUID(invoice_id))

        if not invoice:
            return False, "Invoice not found"

        now = datetime.now(timezone.utc)

        # Queue email
        email_metadata = {
            "source": "payment_capture",
            "invoice_id": invoice_id,
            "proposal_id": str(invoice["proposal_id"]),
            "lead_id": str(invoice["lead_id"]),
            "payment_link": invoice["payment_link"],
        }
        await pool.execute("""
            INSERT INTO ai_email_queue (id, recipient, subject, body, status, scheduled_for, created_at, metadata)
            VALUES ($1, $2, $3, $4, 'queued', $5, $5, $6::jsonb)
        """,
            uuid.uuid4(),
            invoice["client_email"],
            f"Invoice from BrainOps - ${float(invoice['amount']):.2f}",
            f"""Hi {invoice['client_name']},

Your invoice for ${float(invoice['amount']):.2f} is ready.

Pay securely here: {invoice['payment_link']}

Due by: {invoice['due_date'].strftime('%B %d, %Y')}

If you have any questions, just reply to this email.

Best,
BrainOps Team""",
            now,
            json.dumps(email_metadata),
        )

        # Update invoice status
        await pool.execute("""
            UPDATE ai_invoices SET status = 'sent', updated_at = $1 WHERE id = $2
        """, now, uuid.UUID(invoice_id))

        # Update lead state
        from pipeline_state_machine import get_state_machine, PipelineState
        sm = get_state_machine()
        await sm.transition(
            str(invoice["lead_id"]),
            PipelineState.INVOICED,
            trigger="invoice_sent",
            actor="system:payment_capture",
            metadata={"invoice_id": invoice_id}
        )

        logger.info(f"Invoice {invoice_id[:8]}... sent to {invoice['client_email'][:3]}***")
        return True, "Invoice sent"

    async def mark_paid(
        self,
        invoice_id: str,
        payment_method: str = "manual",
        payment_reference: Optional[str] = None,
        verified_by: str = "system"
    ) -> tuple[bool, str]:
        """
        Mark an invoice as paid.

        This is the final step - leads to REAL REVENUE.
        """
        pool = self._get_pool()
        if not pool:
            return False, "Database not available"

        invoice = await pool.fetchrow("""
            SELECT i.*, rl.email AS client_email
            FROM ai_invoices i
            LEFT JOIN revenue_leads rl ON i.lead_id = rl.id
            WHERE i.id = $1
        """, uuid.UUID(invoice_id))

        if not invoice:
            return False, "Invoice not found"

        if invoice["status"] == "paid":
            return False, "Invoice already marked as paid"

        now = datetime.now(timezone.utc)

        # Update invoice
        await pool.execute("""
            UPDATE ai_invoices
            SET status = 'paid', paid_at = $1, updated_at = $1,
                stripe_payment_intent = COALESCE(stripe_payment_intent, $2)
            WHERE id = $3
        """, now, payment_reference, uuid.UUID(invoice_id))

        # Update lead state to PAID - THIS IS REAL REVENUE
        from pipeline_state_machine import get_state_machine, PipelineState
        sm = get_state_machine()
        await sm.transition(
            str(invoice["lead_id"]),
            PipelineState.PAID,
            trigger=f"payment_received_{payment_method}",
            actor=f"human:{verified_by}" if payment_method == "manual" else "system:stripe",
            metadata={
                "invoice_id": invoice_id,
                "amount": float(invoice["amount"]),
                "payment_method": payment_method,
                "payment_reference": payment_reference
            }
        )

        # Record real revenue
        revenue_date = now.date().isoformat()
        stripe_payment_id = payment_reference if payment_method.startswith("stripe") else None
        description = f"Invoice {invoice_id} paid via {payment_method}"

        customer_email = invoice["client_email"]
        currency = invoice["currency"] or "USD"

        await pool.execute("""
            INSERT INTO real_revenue_tracking (
                id,
                tenant_id,
                revenue_date,
                source,
                amount,
                description,
                customer_email,
                is_verified,
                stripe_payment_id,
                created_at,
                currency,
                is_recurring,
                metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, true, $8, $9, $10, false, $11)
        """,
            uuid.uuid4(),
            invoice["tenant_id"],
            revenue_date,
            "proposal",
            float(invoice["amount"]),
            description,
            customer_email,
            stripe_payment_id,
            now,
            currency,
            json.dumps({
                "invoice_id": invoice_id,
                "lead_id": str(invoice["lead_id"]),
                "payment_method": payment_method,
                "payment_reference": payment_reference
            })
        )

        logger.info(f"REAL REVENUE: Invoice {invoice_id[:8]}... marked PAID - ${float(invoice['amount']):.2f}")
        return True, f"Payment confirmed. REAL REVENUE: ${float(invoice['amount']):.2f}"

    async def handle_stripe_webhook(
        self,
        event_type: str,
        data: dict
    ) -> tuple[bool, str]:
        """
        Handle Stripe webhook events for payment confirmation.
        """
        if event_type == "checkout.session.completed":
            session_id = data.get("id")
            payment_status = data.get("payment_status")
            metadata = data.get("metadata", {})

            if payment_status == "paid":
                invoice_id = metadata.get("invoice_id")
                if invoice_id:
                    return await self.mark_paid(
                        invoice_id,
                        payment_method="stripe",
                        payment_reference=session_id,
                        verified_by="stripe_webhook"
                    )

        elif event_type == "invoice.paid":
            invoice_data = data.get("lines", {}).get("data", [{}])[0]
            metadata = invoice_data.get("metadata", {})
            invoice_id = metadata.get("invoice_id")

            if invoice_id:
                return await self.mark_paid(
                    invoice_id,
                    payment_method="stripe_invoice",
                    payment_reference=data.get("id"),
                    verified_by="stripe_webhook"
                )

        return False, f"Unhandled event type: {event_type}"

    async def get_revenue_summary(self, real_only: bool = True) -> dict:
        """
        Get revenue summary from PAID invoices.
        """
        pool = self._get_pool()
        if not pool:
            return {}

        # Get paid invoices
        query = """
            SELECT
                COUNT(*) as total_paid,
                COALESCE(SUM(amount), 0) as total_revenue
            FROM ai_invoices
            WHERE status = 'paid'
        """

        if real_only:
            # Join with leads to filter test data
            query = """
                SELECT
                    COUNT(*) as total_paid,
                    COALESCE(SUM(i.amount), 0) as total_revenue
                FROM ai_invoices i
                JOIN revenue_leads rl ON i.lead_id = rl.id
                WHERE i.status = 'paid'
                AND rl.email NOT ILIKE '%test%'
                AND rl.email NOT ILIKE '%example%'
                AND rl.email NOT ILIKE '%demo%'
            """

        row = await pool.fetchrow(query)

        return {
            "total_paid_invoices": row["total_paid"] if row else 0,
            "total_real_revenue": float(row["total_revenue"] or 0) if row else 0,
            "data_mode": "REAL_ONLY" if real_only else "ALL_DATA"
        }


# Ensure ai_invoices table exists
async def ensure_invoices_table():
    """Ensure the ai_invoices table exists."""
    try:
        from database.async_connection import get_pool, DatabaseUnavailableError
        try:
            pool = get_pool()
        except DatabaseUnavailableError:
            # Pool not initialized yet - will be called again later
            logger.debug("Database pool not ready yet for ai_invoices table - will retry later")
            return False
        if not pool:
            return False

        await pool.execute("""
            CREATE TABLE IF NOT EXISTS ai_invoices (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                proposal_id UUID REFERENCES ai_proposals(id),
                lead_id UUID REFERENCES revenue_leads(id),
                amount DECIMAL(12,2) NOT NULL,
                currency VARCHAR(10) DEFAULT 'USD',
                status VARCHAR(50) DEFAULT 'pending',
                stripe_checkout_id TEXT,
                stripe_payment_intent TEXT,
                payment_link TEXT,
                due_date TIMESTAMPTZ,
                paid_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await pool.execute("""
            CREATE INDEX IF NOT EXISTS idx_ai_invoices_proposal_id ON ai_invoices(proposal_id);
            CREATE INDEX IF NOT EXISTS idx_ai_invoices_lead_id ON ai_invoices(lead_id);
            CREATE INDEX IF NOT EXISTS idx_ai_invoices_status ON ai_invoices(status);
        """)

        logger.info("ai_invoices table ensured")
        return True

    except Exception as e:
        logger.error(f"Failed to ensure ai_invoices table: {e}")
        return False


# Singleton instance
_payment_capture: Optional[PaymentCapture] = None


def get_payment_capture() -> PaymentCapture:
    """Get singleton payment capture instance."""
    global _payment_capture
    if _payment_capture is None:
        _payment_capture = PaymentCapture()
    return _payment_capture

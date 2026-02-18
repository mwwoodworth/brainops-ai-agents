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

def resolve_stripe_secret_key_with_source() -> tuple[str, str]:
    """
    Resolve the Stripe secret key from the environment.

    We prefer a single canonical key name (`STRIPE_SECRET_KEY`) but support
    legacy aliases to reduce production drift during migration.
    """
    for key_name in (
        "STRIPE_SECRET_KEY",
        "STRIPE_API_KEY_LIVE",
        "STRIPE_API_KEY",
        "STRIPE_SECRET_KEY_LIVE",
    ):
        value = (os.getenv(key_name) or "").strip()
        if value:
            return value, key_name
    return "", "missing"


def resolve_stripe_secret_key() -> str:
    key, _ = resolve_stripe_secret_key_with_source()
    return key


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
        self._stripe_key = None

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
        key = resolve_stripe_secret_key()
        if not key:
            return None
        if self._stripe is None:
            try:
                import stripe
                self._stripe = stripe
            except ImportError:
                logger.warning("Stripe SDK not installed")
                return None
        if self._stripe is not None and self._stripe_key != key:
            # Allow key rotation / env fixes without requiring a full process restart.
            self._stripe.api_key = key
            self._stripe_key = key
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

        # GUARD: Check if the proposal is associated with a test/demo tenant.
        # If so, skip Stripe session creation to avoid polluting live Stripe data.
        _skip_stripe = False
        _client_email = str(proposal.get("client_email") or "").strip().lower()
        _test_email_domains = ("@example.com", "@test.com", "@demo.com", "@localhost", "@fake.com")
        if _client_email and any(_client_email.endswith(d) for d in _test_email_domains):
            _skip_stripe = True
            logger.warning(
                "Skipping Stripe checkout for proposal %s: test email domain (%s)",
                proposal_id, _client_email,
            )

        # Also check if the lead belongs to a test tenant via the tenants table.
        if not _skip_stripe:
            try:
                _is_test = await pool.fetchval(
                    """
                    SELECT COALESCE(t.is_test, false)
                    FROM revenue_leads rl
                    JOIN tenants t ON t.id = rl.tenant_id
                    WHERE rl.id = $1
                    """,
                    uuid.UUID(str(proposal["lead_id"])),
                )
                if _is_test:
                    _skip_stripe = True
                    logger.warning(
                        "Skipping Stripe checkout for proposal %s: lead %s belongs to a test tenant",
                        proposal_id, proposal["lead_id"],
                    )
            except Exception:
                # revenue_leads may not have tenant_id column; that's OK -- proceed cautiously.
                pass

        # Create Stripe checkout session if available and not a test context
        stripe = self._get_stripe()
        if stripe and not _skip_stripe:
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
        elif _skip_stripe:
            logger.info("Using manual payment path for test/demo proposal %s", proposal_id)

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

        # Phase 2: Revenue reinforcement
        # Best-effort: queue a recompilation so outreach learns from real cash outcomes.
        try:
            from optimization.revenue_prompt_compile_queue import enqueue_revenue_prompt_compile_task

            await enqueue_revenue_prompt_compile_task(
                pool=pool,
                tenant_id=str(invoice.get("tenant_id") or "") or None,
                lead_id=str(invoice.get("lead_id") or ""),
                reason="real_revenue_tracking:invoice_paid",
                priority=95,
                force=True,
            )
        except Exception as exc:
            logger.debug("Revenue prompt compile enqueue skipped after payment: %s", exc)

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


# Verify ai_invoices table exists (no DDL - agent_worker has no DDL perms)
async def ensure_invoices_table():
    """Verify the ai_invoices table exists. Returns False if missing."""
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

        from database.verify_tables import verify_tables_async
        tables_ok = await verify_tables_async(
            ["ai_invoices"],
            pool,
            module_name="payment_capture",
        )
        if not tables_ok:
            logger.error(
                "ai_invoices table missing - run migrations to create it"
            )
            return False

        logger.info("ai_invoices table verified")
        return True

    except Exception as e:
        logger.error(f"Failed to verify ai_invoices table: {e}")
        return False


# Singleton instance
_payment_capture: Optional[PaymentCapture] = None


def get_payment_capture() -> PaymentCapture:
    """Get singleton payment capture instance."""
    global _payment_capture
    if _payment_capture is None:
        _payment_capture = PaymentCapture()
    return _payment_capture

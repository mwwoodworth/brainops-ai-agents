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


def _env_flag(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or "").strip() or default)
    except (TypeError, ValueError):
        return default


ENABLE_PARTIAL_PAYMENTS = _env_flag("PAYMENT_CAPTURE_ENABLE_PARTIAL_PAYMENTS", "true")
ENABLE_PAYMENT_RETRY = _env_flag("PAYMENT_CAPTURE_ENABLE_RETRY", "true")
ENABLE_PAYMENT_PLAN_ENFORCEMENT = _env_flag("PAYMENT_CAPTURE_ENFORCE_PLAN", "true")
ENABLE_PAYMENT_FOLLOWUP_ESCALATION = _env_flag("PAYMENT_CAPTURE_ENABLE_FOLLOWUP_ESCALATION", "false")
ENABLE_COLLECTION_FORECAST = _env_flag("PAYMENT_CAPTURE_ENABLE_COLLECTION_FORECAST", "false")
PAYMENT_RETRY_MAX_ATTEMPTS = max(1, _env_int("PAYMENT_CAPTURE_RETRY_ATTEMPTS", 3))

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

    def _get_tenant_pool(self, tenant_id: str | None):
        """Get tenant scoped pool for new collection intelligence operations."""
        resolved_tenant = (tenant_id or "").strip() or os.getenv("DEFAULT_TENANT_ID") or os.getenv("TENANT_ID")
        if not resolved_tenant:
            return self._get_pool()
        try:
            from database.async_connection import get_tenant_pool

            return get_tenant_pool(resolved_tenant)
        except Exception as exc:
            logger.warning("Tenant pool unavailable for %s: %s", resolved_tenant, exc)
            return self._get_pool()

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

    @staticmethod
    def _to_uuid(value: str | uuid.UUID) -> uuid.UUID:
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))

    async def _get_invoice_row(self, invoice_id: str) -> Optional[dict[str, Any]]:
        pool = self._get_pool()
        if not pool:
            return None
        row = await pool.fetchrow(
            """
            SELECT i.*, rl.email AS client_email
            FROM ai_invoices i
            LEFT JOIN revenue_leads rl ON i.lead_id = rl.id
            WHERE i.id = $1
            """,
            self._to_uuid(invoice_id),
        )
        return dict(row) if row else None

    async def _get_recorded_payment_total(self, invoice_id: str) -> Decimal:
        """Calculate total captured amount from payment events."""
        pool = self._get_pool()
        if not pool:
            return Decimal("0")

        try:
            raw = await pool.fetchval(
                """
                SELECT COALESCE(SUM((action_data->>'amount')::numeric), 0)
                FROM revenue_actions
                WHERE action_type = 'payment_event'
                  AND action_data->>'invoice_id' = $1
                """,
                str(invoice_id),
            )
            return Decimal(str(raw or 0))
        except Exception:
            return Decimal("0")

    async def _record_payment_event(
        self,
        invoice_id: str,
        lead_id: str,
        amount: Decimal,
        payment_method: str,
        payment_reference: Optional[str],
        actor: str,
        event: str = "captured",
    ) -> None:
        pool = self._get_pool()
        if not pool:
            return
        try:
            await pool.execute(
                """
                INSERT INTO revenue_actions (
                    id, lead_id, action_type, action_data, result, success, created_at, executed_by
                ) VALUES ($1, $2, 'payment_event', $3, $4, true, $5, $6)
                """,
                uuid.uuid4(),
                self._to_uuid(lead_id),
                {
                    "invoice_id": str(invoice_id),
                    "event": event,
                    "amount": float(amount),
                    "payment_method": payment_method,
                    "payment_reference": payment_reference,
                },
                {"status": "recorded"},
                datetime.now(timezone.utc),
                actor,
            )
        except Exception as exc:
            logger.debug("Payment event logging skipped for invoice %s: %s", invoice_id, exc)

    async def _get_payment_plan(self, invoice_id: str) -> Optional[dict[str, Any]]:
        """Fetch latest payment plan configuration for an invoice."""
        pool = self._get_pool()
        if not pool:
            return None
        try:
            row = await pool.fetchrow(
                """
                SELECT action_data
                FROM revenue_actions
                WHERE action_type = 'payment_plan_config'
                  AND action_data->>'invoice_id' = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                str(invoice_id),
            )
            if not row:
                return None
            plan = row.get("action_data")
            if isinstance(plan, str):
                plan = json.loads(plan)
            return plan if isinstance(plan, dict) else None
        except Exception:
            return None

    async def configure_payment_plan(
        self,
        invoice_id: str,
        installment_count: int,
        interval_days: int = 30,
        min_installment_amount: Optional[float] = None,
        configured_by: str = "system",
    ) -> tuple[bool, str]:
        """Attach a payment plan policy to an invoice."""
        if installment_count < 2:
            return False, "Installment count must be >= 2"
        if interval_days < 1:
            return False, "Interval days must be >= 1"

        invoice = await self._get_invoice_row(invoice_id)
        if not invoice:
            return False, "Invoice not found"

        total_amount = Decimal(str(invoice.get("amount") or 0))
        min_installment = Decimal(str(min_installment_amount)) if min_installment_amount else (
            total_amount / Decimal(str(installment_count))
        )

        pool = self._get_pool()
        if not pool:
            return False, "Database not available"

        payload = {
            "invoice_id": str(invoice_id),
            "installment_count": int(installment_count),
            "interval_days": int(interval_days),
            "min_installment_amount": float(min_installment),
            "configured_by": configured_by,
        }
        await pool.execute(
            """
            INSERT INTO revenue_actions (
                id, lead_id, action_type, action_data, success, created_at, executed_by
            ) VALUES ($1, $2, 'payment_plan_config', $3, true, $4, $5)
            """,
            uuid.uuid4(),
            self._to_uuid(str(invoice["lead_id"])),
            payload,
            datetime.now(timezone.utc),
            f"human:{configured_by}",
        )
        return True, "Payment plan configured"

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

    async def capture_payment(
        self,
        invoice_id: str,
        amount: Optional[float] = None,
        payment_method: str = "manual",
        payment_reference: Optional[str] = None,
        verified_by: str = "system",
    ) -> tuple[bool, str]:
        """
        Capture a payment against an invoice.

        Supports partial capture when enabled via PAYMENT_CAPTURE_ENABLE_PARTIAL_PAYMENTS.
        Enforces configured payment plan minimums when PAYMENT_CAPTURE_ENFORCE_PLAN is enabled.
        """
        pool = self._get_pool()
        if not pool:
            return False, "Database not available"

        invoice = await self._get_invoice_row(invoice_id)
        if not invoice:
            return False, "Invoice not found"
        if invoice.get("status") == "paid":
            return False, "Invoice already marked as paid"

        invoice_total = Decimal(str(invoice.get("amount") or 0))
        total_paid_before = await self._get_recorded_payment_total(invoice_id)
        outstanding = max(Decimal("0"), invoice_total - total_paid_before)
        if outstanding <= Decimal("0"):
            return False, "Invoice has no outstanding balance"

        payment_amount = Decimal(str(amount)) if amount is not None else outstanding
        if payment_amount <= 0:
            return False, "Payment amount must be greater than 0"
        if payment_amount > outstanding:
            payment_amount = outstanding

        if payment_amount < outstanding and not ENABLE_PARTIAL_PAYMENTS:
            return False, "Partial payments are disabled by configuration"

        plan = await self._get_payment_plan(invoice_id)
        if ENABLE_PAYMENT_PLAN_ENFORCEMENT and plan:
            min_installment = Decimal(str(plan.get("min_installment_amount") or 0))
            if min_installment > 0 and payment_amount < min_installment:
                return False, (
                    f"Payment amount ${float(payment_amount):.2f} is below plan minimum "
                    f"${float(min_installment):.2f}"
                )

        now = datetime.now(timezone.utc)
        actor = f"human:{verified_by}" if payment_method == "manual" else "system:payment_capture"
        await self._record_payment_event(
            invoice_id=invoice_id,
            lead_id=str(invoice["lead_id"]),
            amount=payment_amount,
            payment_method=payment_method,
            payment_reference=payment_reference,
            actor=actor,
            event="captured",
        )

        total_paid_after = min(invoice_total, total_paid_before + payment_amount)
        remaining = max(Decimal("0"), invoice_total - total_paid_after)
        fully_paid = remaining <= Decimal("0.0001")
        new_status = "paid" if fully_paid else "partial"

        await pool.execute(
            """
            UPDATE ai_invoices
            SET status = $1,
                paid_at = CASE WHEN $1 = 'paid' THEN $2 ELSE paid_at END,
                updated_at = $2,
                stripe_payment_intent = COALESCE(stripe_payment_intent, $3)
            WHERE id = $4
            """,
            new_status,
            now,
            payment_reference,
            self._to_uuid(invoice_id),
        )

        if not fully_paid:
            logger.info(
                "Partial payment captured for invoice %s: +$%.2f (remaining $%.2f)",
                str(invoice_id)[:8],
                float(payment_amount),
                float(remaining),
            )
            return True, (
                f"Partial payment recorded: ${float(payment_amount):.2f}. "
                f"Remaining balance: ${float(remaining):.2f}"
            )

        # Update lead state to PAID - this is revenue-complete lifecycle closeout.
        from pipeline_state_machine import get_state_machine, PipelineState
        sm = get_state_machine()
        await sm.transition(
            str(invoice["lead_id"]),
            PipelineState.PAID,
            trigger=f"payment_received_{payment_method}",
            actor=f"human:{verified_by}" if payment_method == "manual" else "system:stripe",
            metadata={
                "invoice_id": invoice_id,
                "amount": float(invoice_total),
                "payment_method": payment_method,
                "payment_reference": payment_reference,
                "partial_payment_count": 0 if total_paid_before == 0 else 1,
            }
        )

        # Record real revenue
        revenue_date = now.date().isoformat()
        stripe_payment_id = payment_reference if payment_method.startswith("stripe") else None
        description = f"Invoice {invoice_id} paid via {payment_method}"
        customer_email = invoice.get("client_email")
        currency = invoice.get("currency") or "USD"
        tenant_id = invoice.get("tenant_id")

        await pool.execute(
            """
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
            tenant_id,
            revenue_date,
            "proposal",
            float(invoice_total),
            description,
            customer_email,
            stripe_payment_id,
            now,
            currency,
            json.dumps({
                "invoice_id": str(invoice_id),
                "lead_id": str(invoice["lead_id"]),
                "payment_method": payment_method,
                "payment_reference": payment_reference,
                "partial_payments_total": float(total_paid_after),
            }),
        )

        # Phase 2: Revenue reinforcement
        try:
            from optimization.revenue_prompt_compile_queue import enqueue_revenue_prompt_compile_task

            await enqueue_revenue_prompt_compile_task(
                pool=pool,
                tenant_id=str(tenant_id or "") or None,
                lead_id=str(invoice.get("lead_id") or ""),
                reason="real_revenue_tracking:invoice_paid",
                priority=95,
                force=True,
            )
        except Exception as exc:
            logger.debug("Revenue prompt compile enqueue skipped after payment: %s", exc)

        logger.info(
            "REAL REVENUE: Invoice %s marked PAID - $%.2f",
            str(invoice_id)[:8],
            float(invoice_total),
        )
        return True, f"Payment confirmed. REAL REVENUE: ${float(invoice_total):.2f}"

    async def mark_paid(
        self,
        invoice_id: str,
        payment_method: str = "manual",
        payment_reference: Optional[str] = None,
        verified_by: str = "system",
    ) -> tuple[bool, str]:
        """Backward-compatible full payment API."""
        return await self.capture_payment(
            invoice_id=invoice_id,
            amount=None,
            payment_method=payment_method,
            payment_reference=payment_reference,
            verified_by=verified_by,
        )

    async def retry_outstanding_payments(
        self,
        max_invoices: int = 25,
        include_not_due: bool = False,
    ) -> dict[str, Any]:
        """
        Retry payment collection for outstanding invoices.

        Strategy:
        - Queue reminder emails
        - Track retry attempts in revenue_actions
        - Enforce max retries via PAYMENT_CAPTURE_RETRY_ATTEMPTS
        """
        if not ENABLE_PAYMENT_RETRY:
            return {"status": "disabled", "reason": "PAYMENT_CAPTURE_ENABLE_RETRY=false"}

        pool = self._get_pool()
        if not pool:
            return {"status": "error", "error": "Database not available"}

        due_filter = "" if include_not_due else "AND i.due_date <= NOW()"
        invoices = await pool.fetch(
            f"""
            SELECT i.id, i.lead_id, i.amount, i.due_date, i.status, p.client_email, p.client_name, p.offer_name
            FROM ai_invoices i
            LEFT JOIN ai_proposals p ON i.proposal_id = p.id
            WHERE i.status IN ('pending', 'sent', 'partial')
              {due_filter}
            ORDER BY i.due_date ASC NULLS LAST, i.created_at ASC
            LIMIT $1
            """,
            max(1, max_invoices),
        )

        now = datetime.now(timezone.utc)
        retried = 0
        skipped = 0
        failed = 0

        for invoice in invoices:
            invoice_id = str(invoice["id"])
            attempt_count = await pool.fetchval(
                """
                SELECT COUNT(*)
                FROM revenue_actions
                WHERE action_type = 'payment_retry_attempt'
                  AND action_data->>'invoice_id' = $1
                """,
                invoice_id,
            ) or 0

            if attempt_count >= PAYMENT_RETRY_MAX_ATTEMPTS:
                skipped += 1
                continue

            try:
                await pool.execute(
                    """
                    INSERT INTO revenue_actions (
                        id, lead_id, action_type, action_data, success, created_at, executed_by
                    ) VALUES ($1, $2, 'payment_retry_attempt', $3, true, $4, 'system:payment_capture_retry')
                    """,
                    uuid.uuid4(),
                    invoice["lead_id"],
                    {
                        "invoice_id": invoice_id,
                        "attempt": int(attempt_count) + 1,
                        "due_date": invoice["due_date"].isoformat() if invoice["due_date"] else None,
                    },
                    now,
                )

                await pool.execute(
                    """
                    INSERT INTO ai_email_queue (id, recipient, subject, body, status, scheduled_for, created_at, metadata)
                    VALUES ($1, $2, $3, $4, 'queued', $5, $5, $6::jsonb)
                    """,
                    uuid.uuid4(),
                    invoice.get("client_email"),
                    f"Payment reminder: Invoice {invoice_id[:8]}",
                    (
                        f"Hi {invoice.get('client_name') or 'there'},\n\n"
                        f"This is a reminder for invoice {invoice_id[:8]} "
                        f"(${float(invoice.get('amount') or 0):.2f}).\n"
                        f"Please complete payment at your earliest convenience.\n\n"
                        "If you need a payment plan, reply to this email."
                    ),
                    now,
                    json.dumps(
                        {
                            "source": "payment_retry",
                            "invoice_id": invoice_id,
                            "attempt": int(attempt_count) + 1,
                        }
                    ),
                )
                retried += 1
            except Exception as exc:
                failed += 1
                logger.warning("Payment retry failed for invoice %s: %s", invoice_id, exc)

        result = {
            "status": "completed",
            "max_attempts": PAYMENT_RETRY_MAX_ATTEMPTS,
            "evaluated": len(invoices),
            "retried": retried,
            "skipped": skipped,
            "failed": failed,
        }
        if ENABLE_PAYMENT_FOLLOWUP_ESCALATION:
            result["followup_escalation"] = await self.schedule_collection_followups(
                max_invoices=max(1, max_invoices)
            )
        return result

    async def schedule_collection_followups(
        self,
        max_invoices: int = 50,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Schedule escalated collection followups for overdue invoices."""
        if not ENABLE_PAYMENT_FOLLOWUP_ESCALATION:
            return {"status": "skipped", "reason": "PAYMENT_CAPTURE_ENABLE_FOLLOWUP_ESCALATION=false"}

        pool = self._get_tenant_pool(tenant_id)
        if not pool:
            return {"status": "error", "error": "Database not available"}

        rows = await pool.fetch(
            """
            WITH attempts AS (
                SELECT
                    action_data->>'invoice_id' AS invoice_id,
                    COUNT(*) AS retries
                FROM revenue_actions
                WHERE action_type = 'payment_retry_attempt'
                GROUP BY action_data->>'invoice_id'
            )
            SELECT
                i.id,
                i.lead_id,
                i.amount,
                i.due_date,
                i.status,
                p.client_email,
                p.client_name,
                COALESCE(a.retries, 0) AS retries,
                GREATEST(0, EXTRACT(DAY FROM (NOW() - i.due_date))) AS days_overdue
            FROM ai_invoices i
            LEFT JOIN ai_proposals p ON p.id = i.proposal_id
            LEFT JOIN attempts a ON a.invoice_id = i.id::text
            WHERE i.status IN ('pending', 'sent', 'partial')
              AND i.due_date <= NOW()
            ORDER BY days_overdue DESC, retries DESC
            LIMIT $1
            """,
            max(1, max_invoices),
        )

        now = datetime.now(timezone.utc)
        scheduled = 0
        escalated = 0
        for row in rows:
            invoice_id = str(row["id"])
            retries = int(row.get("retries") or 0)
            days_overdue = float(row.get("days_overdue") or 0.0)
            escalation_level = min(3, max(1, retries + (1 if days_overdue >= 7 else 0)))
            template = f"collection_escalation_{escalation_level}"
            due_suffix = f"{int(days_overdue)}d_overdue" if days_overdue > 0 else "due_today"

            exists = await pool.fetchval(
                """
                SELECT target_id
                FROM ai_scheduled_outreach
                WHERE target_id = $1
                  AND message_template = $2
                  AND status IN ('scheduled', 'queued')
                  AND created_at > NOW() - INTERVAL '12 hours'
                LIMIT 1
                """,
                invoice_id,
                template,
            )
            if exists:
                continue

            scheduled_for = now + timedelta(hours=3 if escalation_level >= 3 else 12)
            await pool.execute(
                """
                INSERT INTO ai_scheduled_outreach
                (target_id, channel, message_template, personalization, scheduled_for, status, metadata, created_at)
                VALUES ($1, 'email', $2, $3, $4, 'scheduled', $5, $6)
                """,
                invoice_id,
                template,
                json.dumps(
                    {
                        "invoice_id": invoice_id,
                        "client_name": row.get("client_name"),
                        "client_email": row.get("client_email"),
                        "amount": float(row.get("amount") or 0.0),
                        "retries": retries,
                        "days_overdue": days_overdue,
                    }
                ),
                scheduled_for,
                json.dumps(
                    {
                        "source": "payment_capture",
                        "escalation_level": escalation_level,
                        "due_state": due_suffix,
                    }
                ),
                now,
            )

            await pool.execute(
                """
                INSERT INTO revenue_actions
                (id, lead_id, action_type, action_data, success, created_at, executed_by)
                VALUES ($1, $2, 'collection_followup_escalation', $3, true, $4, 'system:payment_capture')
                """,
                uuid.uuid4(),
                row["lead_id"],
                {
                    "invoice_id": invoice_id,
                    "escalation_level": escalation_level,
                    "days_overdue": days_overdue,
                    "retry_count": retries,
                    "scheduled_for": scheduled_for.isoformat(),
                },
                now,
            )
            scheduled += 1
            if escalation_level >= 2:
                escalated += 1

        return {
            "status": "completed",
            "evaluated": len(rows),
            "scheduled": scheduled,
            "escalated": escalated,
        }

    async def get_collection_forecast(
        self,
        tenant_id: str | None = None,
        horizon_days: int = 30,
    ) -> dict[str, Any]:
        """Forecast expected collections from open invoices."""
        if not ENABLE_COLLECTION_FORECAST:
            return {"status": "skipped", "reason": "PAYMENT_CAPTURE_ENABLE_COLLECTION_FORECAST=false"}

        pool = self._get_tenant_pool(tenant_id)
        if not pool:
            return {"status": "error", "error": "Database not available"}

        rows = await pool.fetch(
            """
            SELECT id, amount, due_date, status
            FROM ai_invoices
            WHERE status IN ('pending', 'sent', 'partial')
              AND COALESCE(due_date, NOW()) <= NOW() + ($1::text || ' days')::interval
            ORDER BY due_date ASC NULLS LAST
            LIMIT 500
            """,
            max(7, horizon_days),
        )

        now = datetime.now(timezone.utc)
        total_open = 0.0
        expected_collection = 0.0
        buckets: dict[str, float] = {"current": 0.0, "overdue_1_7": 0.0, "overdue_8_plus": 0.0}
        for row in rows:
            amount = float(row.get("amount") or 0.0)
            total_open += amount
            due_date = row.get("due_date")
            days_overdue = 0.0
            if due_date:
                days_overdue = (now - due_date).total_seconds() / 86400.0
            if days_overdue <= 0:
                probability = 0.85
                buckets["current"] += amount
            elif days_overdue <= 7:
                probability = 0.60
                buckets["overdue_1_7"] += amount
            else:
                probability = 0.35
                buckets["overdue_8_plus"] += amount
            expected_collection += amount * probability

        return {
            "status": "completed",
            "horizon_days": horizon_days,
            "open_invoice_count": len(rows),
            "open_amount": round(total_open, 2),
            "expected_collection": round(expected_collection, 2),
            "bucket_amounts": {k: round(v, 2) for k, v in buckets.items()},
            "generated_at": now.isoformat(),
        }

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
        partial_collected = await pool.fetchval(
            """
            SELECT COALESCE(SUM((action_data->>'amount')::numeric), 0)
            FROM revenue_actions
            WHERE action_type = 'payment_event'
            """
        )

        return {
            "total_paid_invoices": row["total_paid"] if row else 0,
            "total_real_revenue": float(row["total_revenue"] or 0) if row else 0,
            "total_collected_including_partials": float(partial_collected or 0),
            "data_mode": "REAL_ONLY" if real_only else "ALL_DATA",
            "collection_forecast": await self.get_collection_forecast(),
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

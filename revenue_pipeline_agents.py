#!/usr/bin/env python3
"""
Revenue Pipeline Agents - REAL implementations that query customer/jobs data.

These agents are the production implementations that:
1. Query real customer and jobs data from the database
2. Identify leads based on business rules (re-engagement, upsell, referral)
3. Create nurture sequences
4. Queue emails for the email scheduler daemon

Part of the AI OS Revenue Pipeline fix.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

from database.async_connection import get_pool

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for agents"""

    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.type = agent_type
        self.logger = logging.getLogger(f"Agent.{name}")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(f"Agent {self.name} must implement execute method")

    async def log_execution(self, task: Dict, result: Dict):
        """Log execution to database"""
        try:
            pool = get_pool()
            exec_id = str(uuid.uuid4())
            await pool.execute("""
                INSERT INTO agent_executions (
                    id, task_execution_id, agent_type, prompt,
                    response, status, created_at, completed_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
            """,
                exec_id, exec_id, self.type,
                json.dumps(task), json.dumps(result, default=str),
                result.get('status', 'completed')
            )
        except Exception as e:
            self.logger.error(f"Execution logging failed: {e}")


class LeadDiscoveryAgentReal(BaseAgent):
    """
    REAL Lead Discovery Agent - Queries actual customer/jobs data to find leads.

    Identifies:
    1. Re-engagement leads: Customers who haven't had jobs in 12+ months
    2. Upsell leads: Customers with high job values (potential for premium services)
    3. Referral leads: Active customers who could refer others

    Stores leads in `revenue_leads` table for nurturing.
    """

    def __init__(self):
        super().__init__("LeadDiscoveryAgentReal", "revenue")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute lead discovery from real customer data"""
        action = task.get('action', 'discover_all')

        try:
            if action == 'discover_reengagement':
                return await self.discover_reengagement_leads()
            elif action == 'discover_upsell':
                return await self.discover_upsell_leads()
            elif action == 'discover_referral':
                return await self.discover_referral_leads()
            else:
                return await self.discover_all_leads()
        except Exception as e:
            self.logger.error(f"Lead discovery failed: {e}")
            return {"status": "error", "error": str(e)}

    async def discover_all_leads(self) -> Dict[str, Any]:
        """Run all lead discovery methods and aggregate results"""
        results = {
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "leads_discovered": 0,
            "leads_stored": 0,
            "categories": {}
        }

        # Discover re-engagement leads
        reengagement = await self.discover_reengagement_leads()
        results["categories"]["reengagement"] = reengagement
        results["leads_discovered"] += reengagement.get("leads_found", 0)
        results["leads_stored"] += reengagement.get("leads_stored", 0)

        # Discover upsell leads
        upsell = await self.discover_upsell_leads()
        results["categories"]["upsell"] = upsell
        results["leads_discovered"] += upsell.get("leads_found", 0)
        results["leads_stored"] += upsell.get("leads_stored", 0)

        # Discover referral candidates
        referral = await self.discover_referral_leads()
        results["categories"]["referral"] = referral
        results["leads_discovered"] += referral.get("leads_found", 0)
        results["leads_stored"] += referral.get("leads_stored", 0)

        await self.log_execution({"action": "discover_all"}, results)
        return results

    async def discover_reengagement_leads(self) -> Dict[str, Any]:
        """Find customers who haven't had jobs in 12+ months."""
        try:
            pool = get_pool()

            leads = await pool.fetch("""
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
                GROUP BY c.id, c.company_name, c.first_name, c.last_name, c.email, c.phone, c.city, c.state
                HAVING MAX(j.created_at) < NOW() - INTERVAL '12 months'
                   OR MAX(j.created_at) IS NULL
                ORDER BY avg_job_value DESC
                LIMIT 100
            """)

            leads_stored = 0
            for lead in leads:
                stored = await self._store_lead(
                    lead=lead,
                    source='reengagement_discovery',
                    lead_type='reengagement',
                    score=0.7
                )
                if stored:
                    leads_stored += 1

            return {
                "status": "completed",
                "leads_found": len(leads),
                "leads_stored": leads_stored,
                "lead_type": "reengagement",
                "criteria": "No jobs in 12+ months"
            }

        except Exception as e:
            self.logger.error(f"Re-engagement discovery failed: {e}")
            return {"status": "error", "error": str(e), "leads_found": 0, "leads_stored": 0}

    async def discover_upsell_leads(self) -> Dict[str, Any]:
        """Find high-value customers for upselling premium services."""
        try:
            pool = get_pool()

            leads = await pool.fetch("""
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
                    GROUP BY c.id, c.company_name, c.first_name, c.last_name, c.email, c.phone, c.city, c.state
                ),
                avg_values AS (
                    SELECT
                        AVG(avg_job_value) as system_avg_value,
                        AVG(total_spent) as system_avg_spent
                    FROM customer_stats
                )
                SELECT
                    cs.*,
                    av.system_avg_value,
                    av.system_avg_spent
                FROM customer_stats cs, avg_values av
                WHERE cs.avg_job_value > av.system_avg_value * 1.5
                  AND cs.total_jobs >= 2
                  AND cs.last_job_date > NOW() - INTERVAL '24 months'
                ORDER BY cs.total_spent DESC
                LIMIT 50
            """)

            leads_stored = 0
            for lead in leads:
                stored = await self._store_lead(
                    lead=lead,
                    source='upsell_discovery',
                    lead_type='upsell',
                    score=0.85,
                    value_estimate=float(lead.get('total_spent', 0)) * 0.3
                )
                if stored:
                    leads_stored += 1

            return {
                "status": "completed",
                "leads_found": len(leads),
                "leads_stored": leads_stored,
                "lead_type": "upsell",
                "criteria": "High-value repeat customers (50% above average)"
            }

        except Exception as e:
            self.logger.error(f"Upsell discovery failed: {e}")
            return {"status": "error", "error": str(e), "leads_found": 0, "leads_stored": 0}

    async def discover_referral_leads(self) -> Dict[str, Any]:
        """Find active, satisfied customers who could provide referrals."""
        try:
            pool = get_pool()

            leads = await pool.fetch("""
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
                GROUP BY c.id, c.company_name, c.first_name, c.last_name, c.email, c.phone, c.city, c.state
                HAVING COUNT(j.id) >= 2
                ORDER BY COUNT(j.id) DESC, MAX(j.created_at) DESC
                LIMIT 30
            """)

            leads_stored = 0
            for lead in leads:
                stored = await self._store_lead(
                    lead=lead,
                    source='referral_discovery',
                    lead_type='referral',
                    score=0.6,
                    value_estimate=5000.0
                )
                if stored:
                    leads_stored += 1

            return {
                "status": "completed",
                "leads_found": len(leads),
                "leads_stored": leads_stored,
                "lead_type": "referral",
                "criteria": "Active satisfied customers with 2+ recent jobs"
            }

        except Exception as e:
            self.logger.error(f"Referral discovery failed: {e}")
            return {"status": "error", "error": str(e), "leads_found": 0, "leads_stored": 0}

    async def _store_lead(
        self,
        lead: dict,
        source: str,
        lead_type: str,
        score: float = 0.5,
        value_estimate: float = None
    ) -> bool:
        """Store discovered lead in revenue_leads table"""
        try:
            pool = get_pool()

            email = lead.get('email')
            if not email:
                return False

            existing = await pool.fetchrow(
                "SELECT id FROM revenue_leads WHERE email = $1",
                email
            )

            if existing:
                await pool.execute("""
                    UPDATE revenue_leads
                    SET updated_at = NOW(),
                        metadata = metadata || $1::jsonb
                    WHERE email = $2
                """,
                    json.dumps({
                        f"rediscovered_{lead_type}": datetime.now(timezone.utc).isoformat(),
                        "customer_id": str(lead.get('customer_id', ''))
                    }),
                    email
                )
                return False

            if value_estimate is None:
                avg_job_value = lead.get('avg_job_value') or lead.get('total_spent', 0)
                value_estimate = float(avg_job_value) if avg_job_value else 5000.0

            await pool.execute("""
                INSERT INTO revenue_leads (
                    company_name, contact_name, email, phone, location,
                    stage, score, value_estimate, source, metadata, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW())
            """,
                lead.get('company_name', 'Unknown'),
                lead.get('contact_name', ''),
                email,
                lead.get('phone', ''),
                lead.get('location', ''),
                'new',
                score,
                value_estimate,
                source,
                json.dumps({
                    "lead_type": lead_type,
                    "customer_id": str(lead.get('customer_id', '')),
                    "total_jobs": lead.get('total_jobs', 0),
                    "last_job_date": str(lead.get('last_job_date', '')),
                    "discovered_at": datetime.now(timezone.utc).isoformat()
                })
            )

            self.logger.info(f"Stored new {lead_type} lead: {email}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store lead: {e}")
            return False


class NurtureExecutorAgentReal(BaseAgent):
    """
    REAL Nurture Executor Agent - Creates and executes nurture sequences.

    This agent:
    1. Reads leads from revenue_leads table
    2. Creates nurture sequences based on lead type
    3. Queues emails in ai_email_queue (for SendGrid to process)

    Does NOT send emails directly - just queues them for the email scheduler daemon.
    """

    def __init__(self):
        super().__init__("NurtureExecutorAgentReal", "revenue")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute nurture sequence creation and email queueing"""
        action = task.get('action', 'nurture_new_leads')

        try:
            if action == 'nurture_new_leads':
                return await self.nurture_new_leads()
            elif action == 'create_sequence':
                lead_id = task.get('lead_id')
                sequence_type = task.get('sequence_type', 'reengagement')
                return await self.create_nurture_sequence(lead_id, sequence_type)
            elif action == 'queue_emails':
                return await self.queue_pending_emails()
            else:
                return await self.nurture_new_leads()
        except Exception as e:
            self.logger.error(f"Nurture execution failed: {e}")
            return {"status": "error", "error": str(e)}

    async def nurture_new_leads(self) -> Dict[str, Any]:
        """Process all new leads and create nurture sequences"""
        try:
            pool = get_pool()

            leads = await pool.fetch("""
                SELECT rl.*
                FROM revenue_leads rl
                WHERE rl.stage = 'new'
                  AND rl.email IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM ai_nurture_sequences ns
                      WHERE ns.configuration->>'lead_id' = rl.id::text
                  )
                ORDER BY rl.score DESC, rl.created_at
                LIMIT 50
            """)

            sequences_created = 0
            emails_queued = 0

            for lead in leads:
                metadata = lead.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                lead_type = metadata.get('lead_type', 'reengagement')

                result = await self.create_nurture_sequence(
                    lead_id=str(lead['id']),
                    sequence_type=lead_type,
                    lead_data=dict(lead)
                )

                if result.get('status') == 'completed':
                    sequences_created += 1
                    emails_queued += result.get('emails_queued', 0)

                    await pool.execute("""
                        UPDATE revenue_leads
                        SET stage = 'contacted',
                            updated_at = NOW(),
                            last_contact = NOW()
                        WHERE id = $1
                    """, lead['id'])

            result = {
                "status": "completed",
                "leads_processed": len(leads),
                "sequences_created": sequences_created,
                "emails_queued": emails_queued,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            await self.log_execution({"action": "nurture_new_leads"}, result)
            return result

        except Exception as e:
            self.logger.error(f"Nurture new leads failed: {e}")
            return {"status": "error", "error": str(e)}

    async def create_nurture_sequence(
        self,
        lead_id: str,
        sequence_type: str,
        lead_data: dict = None
    ) -> Dict[str, Any]:
        """Create a nurture sequence for a specific lead"""
        try:
            pool = get_pool()

            if not lead_data:
                lead_row = await pool.fetchrow(
                    "SELECT * FROM revenue_leads WHERE id = $1",
                    lead_id
                )
                lead_data = dict(lead_row) if lead_row else {}

            if not lead_data:
                return {"status": "error", "error": f"Lead {lead_id} not found"}

            sequence = self._generate_sequence(sequence_type, lead_data)

            sequence_id = str(uuid.uuid4())
            # Use correct column names matching ai_nurture_sequences table schema:
            # name (not sequence_name), is_active (not active)
            # touchpoint_count and days_duration go in configuration JSON
            await pool.execute("""
                INSERT INTO ai_nurture_sequences (
                    id, name, sequence_type, target_segment,
                    configuration, is_active, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, true, NOW(), NOW())
            """,
                sequence_id,
                f"{sequence_type.title()} Sequence for {lead_data.get('contact_name', 'Lead')}",
                sequence_type,
                sequence_type,
                json.dumps({
                    "lead_id": lead_id,
                    "touchpoints": sequence['touchpoints'],
                    "touchpoint_count": len(sequence['touchpoints']),
                    "days_duration": sequence['duration_days']
                })
            )

            emails_queued = await self._queue_sequence_emails(
                lead_id=lead_id,
                lead_data=lead_data,
                sequence_id=sequence_id,
                touchpoints=sequence['touchpoints']
            )

            return {
                "status": "completed",
                "sequence_id": sequence_id,
                "sequence_type": sequence_type,
                "touchpoints": len(sequence['touchpoints']),
                "emails_queued": emails_queued
            }

        except Exception as e:
            self.logger.error(f"Create sequence failed: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_sequence(self, sequence_type: str, lead_data: dict) -> dict:
        """Generate touchpoint sequence based on type"""
        contact_name = lead_data.get('contact_name', 'Valued Customer')
        first_name = contact_name.split()[0] if contact_name else 'Friend'
        company_name = lead_data.get('company_name', '')

        sequences = {
            'reengagement': {
                'duration_days': 21,
                'touchpoints': [
                    {
                        'day': 0,
                        'type': 'email',
                        'subject': f"We miss you, {first_name}! Here's what's new",
                        'body': f"Hi {first_name},\n\nIt's been a while since we last worked together, and we wanted to reach out!\n\nWe've been thinking about past customers like you and {company_name or 'your property'}, and we'd love to help with any roofing needs you might have.\n\nSince we last connected, we've:\n- Added new service options\n- Improved our response times\n- Introduced seasonal maintenance plans\n\nIs there anything we can help you with? Reply to this email or give us a call.\n\nBest regards,\nYour Roofing Team"
                    },
                    {
                        'day': 7,
                        'type': 'email',
                        'subject': "Quick check-in: How's your roof holding up?",
                        'body': f"Hi {first_name},\n\nJust wanted to follow up on my last email. Winter weather can be tough on roofs, and I wanted to make sure everything is in good shape for you.\n\nWe're offering free roof inspections this month. Would you like to schedule one?\n\nNo pressure - just want to make sure you're protected.\n\nBest,\nYour Roofing Team"
                    },
                    {
                        'day': 14,
                        'type': 'email',
                        'subject': "Special offer for valued customers",
                        'body': f"Hi {first_name},\n\nAs a valued past customer, we'd like to offer you 15% off your next service.\n\nThis offer is valid for the next 30 days and can be applied to:\n- Roof repairs\n- Maintenance inspections\n- Gutter cleaning\n- New installations\n\nUse code WELCOME15 when you book.\n\nHope to hear from you soon!\n\nYour Roofing Team"
                    }
                ]
            },
            'upsell': {
                'duration_days': 14,
                'touchpoints': [
                    {
                        'day': 0,
                        'type': 'email',
                        'subject': f"Exclusive premium services for {company_name or first_name}",
                        'body': f"Hi {first_name},\n\nThank you for being such a great customer! Your trust in us means everything.\n\nGiven your history with us, I wanted to personally introduce you to some of our premium services:\n\n1. Annual Maintenance Plan - Proactive care to extend roof life\n2. Priority Response Service - 24-hour emergency coverage\n3. Extended Warranty Options - Peace of mind for years to come\n\nThese are only available to select customers like yourself. Would you like to learn more?\n\nBest,\nYour Roofing Team"
                    },
                    {
                        'day': 5,
                        'type': 'email',
                        'subject': "Quick question about your property",
                        'body': f"Hi {first_name},\n\nI hope this email finds you well!\n\nI was reviewing your account and noticed you might benefit from our maintenance plan. It includes:\n- Semi-annual inspections\n- Priority scheduling\n- 10% off all services\n- Extended warranties\n\nWould a quick call work for you this week? I'd love to explain how it could save you money long-term.\n\nBest,\nYour Roofing Team"
                    }
                ]
            },
            'referral': {
                'duration_days': 7,
                'touchpoints': [
                    {
                        'day': 0,
                        'type': 'email',
                        'subject': f"Thank you, {first_name}! A small gift for you",
                        'body': f"Hi {first_name},\n\nI just wanted to say THANK YOU for being such a wonderful customer.\n\nYour continued trust in us means the world, and we'd love to extend that same great service to your friends, family, or neighbors.\n\nOur Referral Program:\n- You get $100 credit for each successful referral\n- Your referral gets 10% off their first service\n- No limit on how many people you can refer!\n\nKnow anyone who might need roofing help? Just reply with their name and number, and we'll take it from there (mentioning you, of course!).\n\nThanks again!\n\nYour Roofing Team"
                    }
                ]
            }
        }

        return sequences.get(sequence_type, sequences['reengagement'])

    async def _queue_sequence_emails(
        self,
        lead_id: str,
        lead_data: dict,
        sequence_id: str,
        touchpoints: list
    ) -> int:
        """Queue emails from touchpoints to ai_email_queue"""
        try:
            pool = get_pool()
            emails_queued = 0

            for touchpoint in touchpoints:
                if touchpoint.get('type') != 'email':
                    continue

                days_delay = touchpoint.get('day', 0)
                scheduled_for = datetime.now(timezone.utc) + timedelta(days=days_delay)

                email_id = str(uuid.uuid4())
                await pool.execute("""
                    INSERT INTO ai_email_queue (
                        id, recipient, subject, body,
                        scheduled_for, status, metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                """,
                    email_id,
                    lead_data.get('email'),
                    touchpoint.get('subject', 'Update from Your Roofing Team'),
                    touchpoint.get('body', ''),
                    scheduled_for,
                    'queued',
                    json.dumps({
                        "source": "nurture_sequence",
                        "sequence_id": sequence_id,
                        "lead_id": lead_id,
                        "touchpoint_day": days_delay,
                        "lead_type": lead_data.get('metadata', {}).get('lead_type', 'unknown')
                    })
                )

                emails_queued += 1
                self.logger.info(f"Queued email for {lead_data.get('email')} (day {days_delay})")

            return emails_queued

        except Exception as e:
            self.logger.error(f"Failed to queue emails: {e}")
            return 0

    async def queue_pending_emails(self) -> Dict[str, Any]:
        """Process any pending emails that haven't been queued yet"""
        try:
            pool = get_pool()

            sequences = await pool.fetch("""
                SELECT id, configuration
                FROM ai_nurture_sequences
                WHERE active = true
                  AND created_at > NOW() - INTERVAL '30 days'
            """)

            total_queued = 0
            for seq in sequences:
                config = seq.get('configuration', {})
                if isinstance(config, str):
                    config = json.loads(config)

                lead_id = config.get('lead_id')
                touchpoints = config.get('touchpoints', [])

                if lead_id and touchpoints:
                    lead_row = await pool.fetchrow(
                        "SELECT * FROM revenue_leads WHERE id = $1",
                        lead_id
                    )
                    if lead_row:
                        queued = await self._queue_sequence_emails(
                            lead_id=lead_id,
                            lead_data=dict(lead_row),
                            sequence_id=str(seq['id']),
                            touchpoints=touchpoints
                        )
                        total_queued += queued

            return {
                "status": "completed",
                "sequences_processed": len(sequences),
                "emails_queued": total_queued
            }

        except Exception as e:
            self.logger.error(f"Queue pending emails failed: {e}")
            return {"status": "error", "error": str(e)}


# Export for use in agent_executor.py
__all__ = ['LeadDiscoveryAgentReal', 'NurtureExecutorAgentReal']

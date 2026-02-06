#!/usr/bin/env python3
"""
Outreach Engine
===============
Handles lead enrichment and outreach sequences for the 23 REAL leads.

Features:
- Lead enrichment within limits (100/day)
- Personalized outreach sequences
- Approval-gated sending
- Dedupe and deliverability checks
- Throttled to 50 emails/day

Part of Revenue Perfection Session.
"""

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
import json

import httpx
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Email regex pattern for scraping
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

TEST_EMAIL_PATTERNS = (
    'test', 'example', 'demo', 'sample', 'fake', 'placeholder', 'localhost'
)

BAD_EMAIL_DOMAINS = (
    'example.com', 'test.com', 'sentry.io', 'google.com',
    'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com'
)

def _parse_metadata(metadata: Any) -> dict:
    """Parse metadata field which may be a string or dict."""
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


# Product URLs
MYROOFGENIUS_URL = "https://myroofgenius.com"
MYROOFGENIUS_PRICING = "https://myroofgenius.com/pricing"
BRAINSTACK_URL = "https://brainstackstudio.com"
GUMROAD_URL = "https://woodworthia.gumroad.com"

# Email templates for outreach - ROOFING COMPANIES
ROOFING_TEMPLATES = {
    "initial_contact": {
        "subject": "AI estimating tool for {company_name}",
        "body": """Hi {contact_name},

I built MyRoofGenius specifically for roofing contractors like {company_name}.

It's an AI-powered estimating tool that:
- Generates accurate roof estimates in under 2 minutes
- Analyzes roof photos to detect damage automatically
- Tracks all your jobs from lead to completion

Contractors using it are cutting estimate time by 70% and winning more bids.

Plans start at $49/month. You can see the full pricing and features here:
{pricing_url}

Would love to get your feedback on it.

Best,
Matt Woodworth
MyRoofGenius

P.S. Free trial available - no credit card required."""
    },
    "followup_1": {
        "subject": "Re: AI estimating for {company_name}",
        "body": """Hi {contact_name},

Quick follow-up on MyRoofGenius.

I know estimating takes forever - measuring, calculating materials, writing up proposals.

Our AI does it in 2 minutes from a roof photo.

Here's a quick look at how it works: {pricing_url}

Worth 5 minutes to check out?

Best,
Matt"""
    },
    "followup_2": {
        "subject": "Last note - {company_name}",
        "body": """Hi {contact_name},

Final note from me.

If you're ever looking to speed up your estimating process, MyRoofGenius is ready when you are: {main_url}

Plans from $49/mo, free trial available.

All the best with {company_name},
Matt"""
    }
}

# Email templates for outreach - TECH/DEV/GENERAL
TECH_TEMPLATES = {
    "initial_contact": {
        "subject": "AI automation tools for {company_name}",
        "body": """Hi {contact_name},

I've been building AI automation tools that might be useful for {company_name}.

Two things that might interest you:

1. **BrainStack Studio** - Multi-AI playground with GPT-5, Claude, Gemini in one interface
   {brainstack_url}

2. **Code Kits & Prompts** - Ready-to-use AI templates, MCP servers, and automation scripts
   {gumroad_url}

If you're building anything with AI, these could save you weeks of development time.

Worth a look?

Best,
Matt Woodworth
BrainStack"""
    },
    "followup_1": {
        "subject": "Re: AI tools for {company_name}",
        "body": """Hi {contact_name},

Just a quick follow-up.

If you're working with AI at all, I have:
- MCP Server Starter Kit ($97)
- AI Orchestration Framework ($147)
- Command Center UI Kit ($149)

All on Gumroad: {gumroad_url}

These are production-ready code, not tutorials.

Best,
Matt"""
    },
    "followup_2": {
        "subject": "Resources for {company_name}",
        "body": """Hi {contact_name},

Last note from me.

If you ever need AI development resources:
- Code kits: {gumroad_url}
- Multi-AI platform: {brainstack_url}

All the best,
Matt"""
    }
}

# Legacy template mapping for backwards compatibility
OUTREACH_TEMPLATES = ROOFING_TEMPLATES


@dataclass
class EnrichmentResult:
    """Result of lead enrichment."""
    lead_id: str
    decision_maker: Optional[str]
    company_size: Optional[str]
    region: Optional[str]
    pain_points: list[str]
    estimated_value_band: str
    enriched_at: datetime
    source: str


@dataclass
class OutreachDraft:
    """A draft outreach message pending approval."""
    id: str
    lead_id: str
    sequence_step: int  # 1, 2, or 3
    subject: str
    body: str
    status: str  # draft, pending_approval, approved, sent
    created_at: datetime


class OutreachEngine:
    """
    Manages lead enrichment and outreach sequences.
    """

    def __init__(self):
        self._daily_enrichment_count = 0
        self._daily_outreach_count = 0
        self._last_count_reset = datetime.now(timezone.utc).date()

    def _get_pool(self):
        """Get database pool."""
        try:
            from database.async_connection import get_pool
            return get_pool()
        except Exception as e:
            logger.error(f"Failed to get database pool: {e}")
            return None

    def _check_limits(self) -> tuple[int, int]:
        """Check and reset daily limits."""
        today = datetime.now(timezone.utc).date()
        if today != self._last_count_reset:
            self._daily_enrichment_count = 0
            self._daily_outreach_count = 0
            self._last_count_reset = today

        return self._daily_enrichment_count, self._daily_outreach_count

    async def get_real_leads(self, limit: int = 100, campaign_id: str = None) -> list[dict]:
        """Get REAL leads (not test/demo), optionally filtered by campaign."""
        pool = self._get_pool()
        if not pool:
            return []

        if campaign_id:
            rows = await pool.fetch("""
                SELECT id, company_name, contact_name, email, stage, score,
                       value_estimate, source, industry, metadata, created_at, location
                FROM revenue_leads
                WHERE is_test = false AND is_demo = false
                AND metadata->>'campaign_id' = $1
                ORDER BY score DESC, created_at DESC
                LIMIT $2
            """, campaign_id, limit)
        else:
            rows = await pool.fetch("""
                SELECT id, company_name, contact_name, email, stage, score,
                       value_estimate, source, industry, metadata, created_at, location
                FROM revenue_leads
                WHERE is_test = false AND is_demo = false
                AND email NOT ILIKE '%test%'
                AND email NOT ILIKE '%example%'
                AND email NOT ILIKE '%demo%'
                ORDER BY score DESC, created_at DESC
                LIMIT $1
            """, limit)

        return [dict(r) for r in rows]

    async def enrich_lead(self, lead_id: str) -> tuple[bool, str, Optional[EnrichmentResult]]:
        """
        Enrich a lead with additional data.

        Limited to 100 enrichments per day.
        """
        enrichment_count, _ = self._check_limits()
        if enrichment_count >= 100:
            return False, "Daily enrichment limit (100) reached", None

        pool = self._get_pool()
        if not pool:
            return False, "Database not available", None

        # Get lead
        lead = await pool.fetchrow("""
            SELECT * FROM revenue_leads WHERE id = $1
        """, uuid.UUID(lead_id) if isinstance(lead_id, str) else lead_id)

        if not lead:
            return False, "Lead not found", None

        # Simulate enrichment (in production, this would call external APIs)
        # For now, generate reasonable defaults based on existing data
        company = lead["company_name"] or "Unknown Company"
        industry = lead.get("industry", "generic")

        # Determine company size based on metadata if available
        metadata = _parse_metadata(lead.get("metadata"))
        company_size = metadata.get("company_size", "small")
        if "enterprise" in company.lower():
            company_size = "enterprise"
        elif "inc" in company.lower() or "corp" in company.lower():
            company_size = "mid-market"

        # Determine region
        region = metadata.get("region", "US")

        # Generate pain points based on industry
        pain_points_map = {
            "roofing": ["Manual estimating taking too long", "Losing bids to competitors", "Difficulty tracking jobs"],
            "construction": ["Project cost overruns", "Scheduling inefficiencies", "Documentation gaps"],
            "technology": ["Scaling operations", "Customer acquisition cost", "Churn reduction"],
            "generic": ["Operational efficiency", "Revenue growth", "Process automation"]
        }
        pain_points = pain_points_map.get(industry, pain_points_map["generic"])

        # Estimate value band
        value_bands = {
            "small": "$1K-5K/month",
            "mid-market": "$5K-15K/month",
            "enterprise": "$15K-50K/month"
        }
        value_band = value_bands.get(company_size, "$1K-5K/month")

        enrichment = EnrichmentResult(
            lead_id=str(lead["id"]),
            decision_maker=lead.get("contact_name") or f"{company} Decision Maker",
            company_size=company_size,
            region=region,
            pain_points=pain_points,
            estimated_value_band=value_band,
            enriched_at=datetime.now(timezone.utc),
            source="internal_enrichment"
        )

        # Store enrichment in metadata
        try:
            new_metadata = {
                **(metadata or {}),
                "enrichment": {
                    "decision_maker": enrichment.decision_maker,
                    "company_size": enrichment.company_size,
                    "region": enrichment.region,
                    "pain_points": enrichment.pain_points,
                    "value_band": enrichment.estimated_value_band,
                    "enriched_at": enrichment.enriched_at.isoformat()
                }
            }

            await pool.execute("""
                UPDATE revenue_leads
                SET metadata = $1, updated_at = $2
                WHERE id = $3
            """, json.dumps(new_metadata), datetime.now(timezone.utc), lead["id"])

            # Record in engagement history
            await pool.execute("""
                INSERT INTO lead_engagement_history (lead_id, event_type, event_data, channel, timestamp)
                VALUES ($1, 'enriched', $2, 'system', $3)
            """,
                lead["id"],
                json.dumps({"source": "internal", "fields_enriched": ["decision_maker", "company_size", "region", "pain_points"]}),
                datetime.now(timezone.utc)
            )

            # Update state machine
            from pipeline_state_machine import get_state_machine, PipelineState
            sm = get_state_machine()
            await sm.transition(
                str(lead["id"]),
                PipelineState.ENRICHED,
                trigger="enrichment_completed",
                actor="system:outreach_engine",
                metadata={"enrichment": new_metadata["enrichment"]}
            )

            self._daily_enrichment_count += 1
            logger.info(f"Lead {lead_id[:8]}... enriched (daily count: {self._daily_enrichment_count})")
            return True, "Lead enriched successfully", enrichment

        except Exception as e:
            logger.error(f"Failed to store enrichment: {e}")
            return False, str(e), None

    async def generate_outreach_draft(
        self,
        lead_id: str,
        sequence_step: int = 1
    ) -> tuple[bool, str, Optional[OutreachDraft]]:
        """
        Generate an outreach message draft for a lead.

        Does NOT send - creates a draft for approval.
        """
        pool = self._get_pool()
        if not pool:
            return False, "Database not available", None

        # Get lead with enrichment
        lead = await pool.fetchrow("""
            SELECT * FROM revenue_leads WHERE id = $1
        """, uuid.UUID(lead_id) if isinstance(lead_id, str) else lead_id)

        if not lead:
            return False, "Lead not found", None

        # Get enrichment data
        metadata = _parse_metadata(lead.get("metadata"))
        enrichment = metadata.get("enrichment", {})
        if isinstance(enrichment, str):
            try:
                enrichment = json.loads(enrichment)
            except (json.JSONDecodeError, TypeError):
                enrichment = {}

        # Check if this lead has a campaign_id - use campaign templates if so
        campaign_id = metadata.get("campaign_id")
        if campaign_id:
            try:
                from campaign_manager import get_campaign, personalize_template as cm_personalize
                campaign = get_campaign(campaign_id)
                if campaign and sequence_step <= len(campaign.templates):
                    template = campaign.templates[sequence_step - 1]
                    subject, body = cm_personalize(template, dict(lead), campaign)

                    # Create draft
                    draft_id = str(uuid.uuid4())
                    draft = OutreachDraft(
                        id=draft_id,
                        lead_id=str(lead["id"]),
                        sequence_step=sequence_step,
                        subject=subject,
                        body=body,
                        status="draft",
                        created_at=datetime.now(timezone.utc)
                    )

                    # Store draft
                    await pool.execute("""
                        INSERT INTO revenue_actions (id, lead_id, action_type, action_data, success, created_at, executed_by)
                        VALUES ($1, $2, 'outreach_draft', $3, true, $4, 'system:outreach_engine')
                    """,
                        uuid.UUID(draft_id),
                        lead["id"],
                        json.dumps({
                            "sequence_step": sequence_step,
                            "subject": subject,
                            "body": body,
                            "status": "draft",
                            "campaign_id": campaign_id,
                        }),
                        datetime.now(timezone.utc)
                    )

                    logger.info(f"Campaign outreach draft {draft_id[:8]}... created for lead {lead_id[:8]}...")
                    return True, "Campaign outreach draft created", draft
            except ImportError:
                logger.warning("campaign_manager not available, falling back to standard templates")

        # Detect if roofing company (use roofing templates) or tech (use tech templates)
        company_name = (lead["company_name"] or "").lower()
        industry = (lead.get("industry") or "").lower()
        source = (lead.get("source") or "").lower()

        is_roofing = any(word in company_name or word in industry for word in
                        ["roof", "roofing", "construction", "contractor", "solar", "siding", "gutter"])

        # Select appropriate template set
        if is_roofing:
            template_set = ROOFING_TEMPLATES
        else:
            template_set = TECH_TEMPLATES

        # Get template for sequence step
        templates = {1: "initial_contact", 2: "followup_1", 3: "followup_2"}
        template_key = templates.get(sequence_step, "initial_contact")
        template = template_set[template_key]

        # Personalize template
        pain_points = enrichment.get("pain_points", ["estimating efficiency" if is_roofing else "AI development"])
        first_pain = pain_points[0] if pain_points else ("estimating efficiency" if is_roofing else "AI development")

        variables = {
            "company_name": lead["company_name"] or "your company",
            "contact_name": lead.get("contact_name") or "there",
            "industry": lead.get("industry", "roofing" if is_roofing else "technology"),
            "pain_point": first_pain,
            "benefit": "70% faster estimates" if is_roofing else "weeks of development time saved",
            "calendar_link": "https://calendly.com/brainops/15min",
            # Product URLs
            "pricing_url": MYROOFGENIUS_PRICING,
            "main_url": MYROOFGENIUS_URL,
            "brainstack_url": BRAINSTACK_URL,
            "gumroad_url": GUMROAD_URL
        }

        subject = template["subject"].format(**variables)
        body = template["body"].format(**variables)

        # Create draft
        draft_id = str(uuid.uuid4())
        draft = OutreachDraft(
            id=draft_id,
            lead_id=str(lead["id"]),
            sequence_step=sequence_step,
            subject=subject,
            body=body,
            status="draft",
            created_at=datetime.now(timezone.utc)
        )

        # Store draft
        try:
            await pool.execute("""
                INSERT INTO revenue_actions (id, lead_id, action_type, action_data, success, created_at, executed_by)
                VALUES ($1, $2, 'outreach_draft', $3, true, $4, 'system:outreach_engine')
            """,
                uuid.UUID(draft_id),
                lead["id"],
                json.dumps({
                    "sequence_step": sequence_step,
                    "subject": subject,
                    "body": body,
                    "status": "draft"
                }),
                datetime.now(timezone.utc)
            )

            logger.info(f"Outreach draft {draft_id[:8]}... created for lead {lead_id[:8]}...")
            return True, "Outreach draft created", draft

        except Exception as e:
            logger.error(f"Failed to create outreach draft: {e}")
            return False, str(e), None

    async def submit_outreach_for_approval(self, draft_id: str) -> tuple[bool, str]:
        """Submit an outreach draft for human approval."""
        pool = self._get_pool()
        if not pool:
            return False, "Database not available"

        try:
            # Update draft status
            result = await pool.execute("""
                UPDATE revenue_actions
                SET action_data = action_data || '{"status": "pending_approval"}'::jsonb
                WHERE id = $1 AND action_type = 'outreach_draft'
            """, uuid.UUID(draft_id))

            if result == "UPDATE 0":
                return False, "Draft not found"

            # Create approval item
            await pool.execute("""
                INSERT INTO ai_improvement_proposals (id, title, description, status, created_at, updated_at)
                VALUES ($1, $2, $3, 'proposed', $4, $4)
            """,
                uuid.uuid4(),
                f"Approve Outreach: {draft_id[:8]}",
                f"Outreach email draft pending approval. Draft ID: {draft_id}",
                datetime.now(timezone.utc)
            )

            # Update lead state
            draft = await pool.fetchrow("""
                SELECT lead_id FROM revenue_actions WHERE id = $1
            """, uuid.UUID(draft_id))

            if draft:
                from pipeline_state_machine import get_state_machine, PipelineState
                sm = get_state_machine()
                await sm.transition(
                    str(draft["lead_id"]),
                    PipelineState.OUTREACH_PENDING_APPROVAL,
                    trigger="outreach_draft_submitted",
                    actor="system:outreach_engine",
                    metadata={"draft_id": draft_id}
                )

            logger.info(f"Outreach draft {draft_id[:8]}... submitted for approval")
            return True, "Submitted for approval"

        except Exception as e:
            logger.error(f"Failed to submit for approval: {e}")
            return False, str(e)

    async def approve_and_send_outreach(
        self,
        draft_id: str,
        approved_by: str
    ) -> tuple[bool, str]:
        """
        Approve and send an outreach message.

        Limited to 50 emails per day.
        """
        _, outreach_count = self._check_limits()
        if outreach_count >= 50:
            return False, "Daily outreach limit (50) reached"

        pool = self._get_pool()
        if not pool:
            return False, "Database not available"

        try:
            # Get draft
            draft = await pool.fetchrow("""
                SELECT * FROM revenue_actions
                WHERE id = $1 AND action_type = 'outreach_draft'
            """, uuid.UUID(draft_id))

            if not draft:
                return False, "Draft not found"

            action_data = draft["action_data"]
            # Parse JSON if it's a string
            if isinstance(action_data, str):
                action_data = json.loads(action_data)
            if action_data.get("status") not in ["draft", "pending_approval"]:
                return False, f"Draft status is {action_data.get('status')}, cannot send"

            # Get lead email
            lead = await pool.fetchrow("""
                SELECT email, company_name FROM revenue_leads WHERE id = $1
            """, draft["lead_id"])

            if not lead or not lead["email"]:
                return False, "Lead has no email address"

            # Check for duplicate sends
            existing = await pool.fetchrow("""
                SELECT id FROM ai_email_queue
                WHERE recipient = $1
                AND subject = $2
                AND status = 'sent'
            """, lead["email"], action_data.get("subject"))

            if existing:
                return False, "Duplicate email already sent"

            # Queue email
            now = datetime.now(timezone.utc)
            email_metadata = {
                "source": "outreach_engine",
                "draft_id": draft_id,
                "lead_id": str(draft["lead_id"]),
                "approved_by": approved_by,
                "sequence_step": action_data.get("sequence_step"),
            }
            await pool.execute("""
                INSERT INTO ai_email_queue (id, recipient, subject, body, status, scheduled_for, created_at, metadata)
                VALUES ($1, $2, $3, $4, 'queued', $5, $5, $6::jsonb)
            """,
                uuid.uuid4(),
                lead["email"],
                action_data.get("subject"),
                action_data.get("body"),
                now,
                json.dumps(email_metadata),
            )

            # Update draft status
            await pool.execute("""
                UPDATE revenue_actions
                SET action_data = action_data || '{"status": "sent", "approved_by": "' || $1 || '", "sent_at": "' || $2 || '"}'::jsonb
                WHERE id = $3
            """, approved_by, now.isoformat(), uuid.UUID(draft_id))

            # Update lead state
            from pipeline_state_machine import get_state_machine, PipelineState
            sm = get_state_machine()
            await sm.transition(
                str(draft["lead_id"]),
                PipelineState.OUTREACH_SENT,
                trigger="outreach_sent",
                actor=f"human:{approved_by}",
                metadata={"draft_id": draft_id, "approved_by": approved_by}
            )

            # Record engagement
            await pool.execute("""
                INSERT INTO lead_engagement_history (lead_id, event_type, event_data, channel, timestamp)
                VALUES ($1, 'outreach_sent', $2, 'email', $3)
            """,
                draft["lead_id"],
                json.dumps({"draft_id": draft_id, "sequence_step": action_data.get("sequence_step", 1)}),
                now
            )

            self._daily_outreach_count += 1
            logger.info(f"Outreach {draft_id[:8]}... approved by {approved_by} and queued (daily count: {self._daily_outreach_count})")
            return True, "Outreach approved and queued"

        except Exception as e:
            logger.error(f"Failed to approve and send: {e}")
            return False, str(e)

    async def log_reply(self, lead_id: str, reply_summary: str) -> tuple[bool, str]:
        """Log a reply received from a lead (manual entry)."""
        pool = self._get_pool()
        if not pool:
            return False, "Database not available"

        now = datetime.now(timezone.utc)

        try:
            # Record engagement
            await pool.execute("""
                INSERT INTO lead_engagement_history (lead_id, event_type, event_data, channel, timestamp)
                VALUES ($1, 'reply_received', $2, 'email', $3)
            """,
                uuid.UUID(lead_id),
                json.dumps({"summary": reply_summary}),
                now
            )

            # Update lead state
            from pipeline_state_machine import get_state_machine, PipelineState
            sm = get_state_machine()
            await sm.transition(
                lead_id,
                PipelineState.REPLIED,
                trigger="reply_received",
                actor="human:manual_entry",
                metadata={"summary": reply_summary}
            )

            logger.info(f"Reply logged for lead {lead_id[:8]}...")
            return True, "Reply logged and state updated"

        except Exception as e:
            logger.error(f"Failed to log reply: {e}")
            return False, str(e)

    async def get_outreach_stats(self) -> dict:
        """Get outreach statistics."""
        pool = self._get_pool()
        if not pool:
            return {}

        enrichment_count, outreach_count = self._check_limits()

        drafts = await pool.fetch("""
            SELECT action_data->>'status' as status, COUNT(*) as count
            FROM revenue_actions
            WHERE action_type = 'outreach_draft'
            GROUP BY action_data->>'status'
        """)

        status_counts = {r["status"]: r["count"] for r in drafts}

        return {
            "daily_limits": {
                "enrichment": {"used": enrichment_count, "limit": 100},
                "outreach": {"used": outreach_count, "limit": 50}
            },
            "drafts": {
                "total": sum(status_counts.values()),
                "by_status": status_counts
            }
        }


    async def scrape_email_from_website(self, website_url: str) -> Optional[str]:
        """
        Scrape contact email from a company website.

        Tries common contact pages and extracts email addresses.
        Filters out generic emails like noreply@, info@, etc.
        """
        if not website_url:
            return None

        # Normalize URL
        if not website_url.startswith(('http://', 'https://')):
            website_url = f'https://{website_url}'

        # Common pages to check for contact info
        pages_to_check = [
            '',              # Homepage
            '/contact',
            '/contact-us',
            '/about',
            '/about-us',
            '/team',
        ]

        emails_found = set()

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            for page in pages_to_check:
                try:
                    url = website_url.rstrip('/') + page
                    response = await client.get(url, headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; BrainOps/1.0; business research)'
                    })

                    if response.status_code == 200:
                        # Extract emails from page content
                        found = EMAIL_PATTERN.findall(response.text.lower())
                        emails_found.update(found)

                except Exception as e:
                    logger.debug(f"Failed to fetch {url}: {e}")
                    continue

        # Filter out generic/unwanted emails
        bad_prefixes = ('noreply', 'no-reply', 'donotreply', 'mailer-daemon',
                        'postmaster', 'webmaster', 'admin@', 'support@',
                        'sales@', 'marketing@', 'info@')
        bad_domains = ('example.com', 'test.com', 'sentry.io', 'google.com',
                       'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com')

        good_emails = []
        for email in emails_found:
            email_lower = email.lower()
            if any(email_lower.startswith(p) for p in bad_prefixes):
                continue
            if any(d in email_lower for d in bad_domains):
                continue
            # Prefer emails with names (contain letters before @)
            local_part = email.split('@')[0]
            if re.match(r'^[a-z]+[a-z.]+$', local_part):
                good_emails.insert(0, email)  # Prioritize name-based emails
            else:
                good_emails.append(email)

        if good_emails:
            return good_emails[0]

        # Fallback to generic emails if nothing better found
        for email in emails_found:
            if 'info@' in email or 'contact@' in email:
                return email

        return None

    async def enrich_leads_with_emails(self, limit: int = 10) -> dict[str, Any]:
        """
        Find leads with websites but no emails, and scrape emails from their websites.

        Returns summary of enrichment results.
        """
        pool = self._get_pool()
        if not pool:
            return {"status": "error", "error": "Database not available"}

        # Get leads with websites but no emails
        leads = await pool.fetch("""
            SELECT id, company_name, website
            FROM revenue_leads
            WHERE website IS NOT NULL
              AND website != ''
              AND (email IS NULL OR email = '')
              AND is_test = false
            ORDER BY score DESC NULLS LAST
            LIMIT $1
        """, limit)

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "leads_processed": 0,
            "emails_found": 0,
            "enriched_leads": [],
            "failed_leads": []
        }

        for lead in leads:
            lead_id = str(lead['id'])
            company = lead['company_name']
            website = lead['website']

            results["leads_processed"] += 1

            email = await self.scrape_email_from_website(website)

            if email:
                # Update lead with found email
                await pool.execute("""
                    UPDATE revenue_leads
                    SET email = $1, updated_at = $2
                    WHERE id = $3
                """, email, datetime.now(timezone.utc), lead['id'])

                results["emails_found"] += 1
                results["enriched_leads"].append({
                    "company": company,
                    "website": website,
                    "email": email
                })
                logger.info(f"Found email for {company}: {email}")
            else:
                results["failed_leads"].append({
                    "company": company,
                    "website": website,
                    "reason": "No email found"
                })
                logger.info(f"No email found for {company} ({website})")

        return results

    def _extract_domain(self, website_url: str) -> Optional[str]:
        if not website_url:
            return None
        if not website_url.startswith(('http://', 'https://')):
            website_url = f'https://{website_url}'
        try:
            parsed = urlparse(website_url)
            host = parsed.netloc.lower().strip()
            if host.startswith('www.'):
                host = host[4:]
            return host or None
        except Exception:
            return None

    def _is_real_email(self, email: str) -> bool:
        email_lower = email.lower().strip()
        if not email_lower or '@' not in email_lower:
            return False
        if any(token in email_lower for token in TEST_EMAIL_PATTERNS):
            return False
        domain = email_lower.split('@')[-1]
        if domain in BAD_EMAIL_DOMAINS:
            return False
        return True

    async def enrich_ai_revenue_leads(self, limit: int = 10) -> dict[str, Any]:
        """
        Promote AI-researched leads into revenue_leads ONLY if we can find
        a real contact email from the company's website (no guessing).
        """
        pool = self._get_pool()
        if not pool:
            return {"status": "error", "error": "Database not available"}

        leads = await pool.fetch("""
            SELECT id, company_name, contact_name, contact_email, contact_phone,
                   website, score, industry, metadata
            FROM ai_revenue_leads
            WHERE status IN ('research', 'new', 'enriched')
              AND website IS NOT NULL
              AND website != ''
            ORDER BY score DESC NULLS LAST
            LIMIT $1
        """, limit)

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "leads_processed": 0,
            "promoted": 0,
            "duplicates": 0,
            "failed": 0,
            "items": []
        }

        for lead in leads:
            lead_id = str(lead["id"])
            company = lead["company_name"]
            website = lead["website"]
            email = (lead.get("contact_email") or "").strip()

            results["leads_processed"] += 1

            if not email:
                email = await self.scrape_email_from_website(website)

            if not email or not self._is_real_email(email):
                results["failed"] += 1
                results["items"].append({
                    "lead_id": lead_id,
                    "company": company,
                    "status": "no_valid_email"
                })
                await pool.execute("""
                    UPDATE ai_revenue_leads
                    SET status = 'no_email', updated_at = $1
                    WHERE id = $2
                """, datetime.now(timezone.utc), lead["id"])
                continue

            website_domain = self._extract_domain(website)
            email_domain = email.lower().split("@")[-1]
            if website_domain and not (email_domain == website_domain or email_domain.endswith("." + website_domain)):
                results["failed"] += 1
                results["items"].append({
                    "lead_id": lead_id,
                    "company": company,
                    "status": "domain_mismatch",
                    "email": email
                })
                await pool.execute("""
                    UPDATE ai_revenue_leads
                    SET status = 'email_domain_mismatch', updated_at = $1
                    WHERE id = $2
                """, datetime.now(timezone.utc), lead["id"])
                continue

            existing = await pool.fetchval(
                "SELECT id FROM revenue_leads WHERE email = $1 LIMIT 1",
                email
            )
            if existing:
                results["duplicates"] += 1
                results["items"].append({
                    "lead_id": lead_id,
                    "company": company,
                    "status": "duplicate_email",
                    "email": email
                })
                await pool.execute("""
                    UPDATE ai_revenue_leads
                    SET status = 'duplicate', contact_email = $1, updated_at = $2
                    WHERE id = $3
                """, email, datetime.now(timezone.utc), lead["id"])
                continue

            metadata = _parse_metadata(lead.get("metadata"))
            metadata.update({
                "email_source": "website_scrape",
                "email_source_url": website,
                "email_verified_at": datetime.now(timezone.utc).isoformat(),
                "ai_revenue_lead_id": lead_id
            })

            await pool.execute("""
                INSERT INTO revenue_leads (
                    id, company_name, contact_name, email, phone, website,
                    stage, status, score, source, industry, metadata, created_at, updated_at,
                    is_test, is_demo
                ) VALUES ($1, $2, $3, $4, $5, $6,
                          'new', 'new', $7, 'ai_revenue_enrichment', $8, $9, NOW(), NOW(),
                          FALSE, FALSE)
            """,
                uuid.uuid4(),
                company,
                lead.get("contact_name"),
                email,
                lead.get("contact_phone"),
                website,
                lead.get("score") or 0,
                lead.get("industry") or "roofing",
                json.dumps(metadata)
            )

            await pool.execute("""
                UPDATE ai_revenue_leads
                SET status = 'promoted', contact_email = $1, updated_at = $2
                WHERE id = $3
            """, email, datetime.now(timezone.utc), lead["id"])

            results["promoted"] += 1
            results["items"].append({
                "lead_id": lead_id,
                "company": company,
                "status": "promoted",
                "email": email
            })

        return results


# Singleton instance
_outreach_engine: Optional[OutreachEngine] = None


def get_outreach_engine() -> OutreachEngine:
    """Get singleton outreach engine instance."""
    global _outreach_engine
    if _outreach_engine is None:
        _outreach_engine = OutreachEngine()
    return _outreach_engine

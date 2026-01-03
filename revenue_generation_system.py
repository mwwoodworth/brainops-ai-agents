#!/usr/bin/env python3
"""
Autonomous Revenue Generation System
Implements AI-driven revenue workflows for automatic lead-to-close operations
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data safeguards
TEST_EMAIL_SUFFIXES = (".test", ".example", ".invalid")
TEST_EMAIL_TOKENS = ("@example.", "@test.", "@demo.", "@invalid.")
TEST_EMAIL_DOMAINS = ("test.com", "example.com", "localhost", "demo@roofing.com")


def _is_test_email(email: str | None) -> bool:
    if not email:
        return True
    lowered = email.lower().strip()
    if any(lowered.endswith(suffix) for suffix in TEST_EMAIL_SUFFIXES):
        return True
    if any(token in lowered for token in TEST_EMAIL_TOKENS):
        return True
    return any(domain in lowered for domain in TEST_EMAIL_DOMAINS)

# Use unified AI core instead of direct clients
try:
    from ai_core import RealAICore, ai_analyze, ai_generate
    _ai_core = RealAICore()
    AI_CORE_AVAILABLE = True
    logger.info("Revenue System using unified AI Core")
except ImportError:
    AI_CORE_AVAILABLE = False
    import anthropic
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    logger.warning("AI Core not available - using direct clients")

# Database configuration - use config module for consistency
# NO hardcoded credentials - all values MUST come from environment variables
try:
    from config import config
    DB_CONFIG = {
        "host": config.database.host,
        "database": config.database.database,
        "user": config.database.user,
        "password": config.database.password,
        "port": config.database.port
    }
except (ImportError, AttributeError):
    # Fallback to environment variables directly - supports DATABASE_URL
    from urllib.parse import urlparse
    _DB_HOST = os.getenv("DB_HOST")
    _DB_NAME = os.getenv("DB_NAME")
    _DB_USER = os.getenv("DB_USER")
    _DB_PASSWORD = os.getenv("DB_PASSWORD")
    _DB_PORT = os.getenv("DB_PORT", "5432")

    # Fallback to DATABASE_URL if individual vars not set
    if not all([_DB_HOST, _DB_NAME, _DB_USER, _DB_PASSWORD]):
        _DATABASE_URL = os.getenv('DATABASE_URL', '')
        if _DATABASE_URL:
            _parsed = urlparse(_DATABASE_URL)
            _DB_HOST = _parsed.hostname or ''
            _DB_NAME = _parsed.path.lstrip('/') if _parsed.path else ''
            _DB_USER = _parsed.username or ''
            _DB_PASSWORD = _parsed.password or ''
            _DB_PORT = str(_parsed.port) if _parsed.port else '5432'

    if not all([_DB_HOST, _DB_NAME, _DB_USER, _DB_PASSWORD]):
        raise RuntimeError(
            "Database configuration is incomplete. "
            "Set DB_HOST/DB_NAME/DB_USER/DB_PASSWORD or DATABASE_URL."
        ) from None

    DB_CONFIG = {
        "host": _DB_HOST,
        "database": _DB_NAME,
        "user": _DB_USER,
        "password": _DB_PASSWORD,
        "port": int(_DB_PORT)
    }

# Connection pool for sync operations (reuse connections)
_sync_connection_pool = []
_MAX_POOL_SIZE = 5

def get_sync_connection():
    """Get a connection from pool or create new one"""
    global _sync_connection_pool
    if _sync_connection_pool:
        conn = _sync_connection_pool.pop()
        try:
            # Test if connection is still valid
            conn.cursor().execute("SELECT 1")
            return conn
        except Exception as e:
            logger.warning(f"Stale connection in pool, creating new: {e}")
            try:
                conn.close()
            except Exception:
                logger.debug("Connection already closed while clearing pool")
    return psycopg2.connect(**DB_CONFIG, connect_timeout=10)

def return_sync_connection(conn):
    """Return connection to pool for reuse"""
    global _sync_connection_pool
    if len(_sync_connection_pool) < _MAX_POOL_SIZE:
        try:
            conn.rollback()  # Reset any uncommitted state
            _sync_connection_pool.append(conn)
        except Exception as e:
            logger.warning(f"Failed to return connection to pool: {e}")
            try:
                conn.close()
            except Exception as close_err:
                logger.debug(f"Error closing connection: {close_err}")
    else:
        try:
            conn.close()
        except Exception as e:
            logger.debug(f"Error closing excess connection: {e}")

class LeadStage(Enum):
    """Lead progression stages"""
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    PROPOSAL_SENT = "proposal_sent"
    NEGOTIATING = "negotiating"
    WON = "won"
    LOST = "lost"

class RevenueAction(Enum):
    """Revenue generation actions"""
    IDENTIFY_LEAD = "identify_lead"
    QUALIFY_LEAD = "qualify_lead"
    SEND_OUTREACH = "send_outreach"
    SCHEDULE_FOLLOWUP = "schedule_followup"
    CREATE_PROPOSAL = "create_proposal"
    NEGOTIATE_DEAL = "negotiate_deal"
    CLOSE_DEAL = "close_deal"
    NURTURE_LEAD = "nurture_lead"

@dataclass
class Lead:
    """Represents a potential customer"""
    id: str
    company_name: str
    contact_name: str
    email: str
    phone: Optional[str]
    stage: LeadStage
    score: float
    value_estimate: float
    metadata: dict[str, Any]
    created_at: datetime
    last_contact: Optional[datetime]
    next_action: Optional[RevenueAction]
    next_action_date: Optional[datetime]

@dataclass
class RevenueOpportunity:
    """Represents a revenue opportunity"""
    id: str
    lead_id: str
    title: str
    value: float
    probability: float
    expected_close_date: datetime
    stage: str
    notes: str
    created_at: datetime

class AutonomousRevenueSystem:
    """Fully autonomous revenue generation system"""

    def __init__(self):
        """Initialize the revenue system"""
        self._ensure_tables()
        self.active_workflows = {}
        logger.info("Autonomous Revenue System initialized")

    def _ensure_tables(self):
        """Ensure revenue tables exist"""
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Add location column if it doesn't exist (migration for existing tables)
        cursor.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'revenue_leads' AND column_name = 'location'
                ) THEN
                    ALTER TABLE revenue_leads ADD COLUMN location VARCHAR(255);
                END IF;
            END $$;
        """)
        conn.commit()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS revenue_leads (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                company_name VARCHAR(255) NOT NULL,
                contact_name VARCHAR(255),
                email VARCHAR(255),
                phone VARCHAR(50),
                website VARCHAR(255),
                location VARCHAR(255),
                stage VARCHAR(50) DEFAULT 'new',
                score FLOAT DEFAULT 0.0,
                value_estimate FLOAT DEFAULT 0.0,
                source VARCHAR(100),
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                last_contact TIMESTAMPTZ,
                next_action VARCHAR(50),
                next_action_date TIMESTAMPTZ,
                assigned_agent_id UUID
            );

            CREATE TABLE IF NOT EXISTS revenue_opportunities (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                lead_id UUID REFERENCES revenue_leads(id),
                title VARCHAR(255),
                value FLOAT,
                probability FLOAT DEFAULT 0.5,
                expected_close_date DATE,
                stage VARCHAR(50),
                notes TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                closed_at TIMESTAMPTZ,
                won BOOLEAN
            );

            CREATE TABLE IF NOT EXISTS revenue_actions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                lead_id UUID REFERENCES revenue_leads(id),
                action_type VARCHAR(50),
                action_data JSONB,
                result JSONB,
                success BOOLEAN,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                executed_by VARCHAR(100)
            );

            CREATE TABLE IF NOT EXISTS email_templates (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                template_type VARCHAR(50),
                subject VARCHAR(255),
                body TEXT,
                variables JSONB,
                performance_score FLOAT DEFAULT 0.5,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS revenue_metrics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                metric_date DATE DEFAULT CURRENT_DATE,
                leads_generated INT DEFAULT 0,
                leads_qualified INT DEFAULT 0,
                proposals_sent INT DEFAULT 0,
                deals_closed INT DEFAULT 0,
                revenue_generated FLOAT DEFAULT 0.0,
                conversion_rate FLOAT DEFAULT 0.0,
                avg_deal_size FLOAT DEFAULT 0.0,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_revenue_leads_stage ON revenue_leads(stage);
            CREATE INDEX IF NOT EXISTS idx_revenue_leads_score ON revenue_leads(score DESC);
            CREATE INDEX IF NOT EXISTS idx_revenue_opportunities_value ON revenue_opportunities(value DESC);
            CREATE INDEX IF NOT EXISTS idx_revenue_actions_lead ON revenue_actions(lead_id);

            -- New enhancement tables
            CREATE TABLE IF NOT EXISTS ai_email_sequences (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                lead_id UUID REFERENCES revenue_leads(id),
                sequence_type VARCHAR(50),
                emails JSONB DEFAULT '[]'::jsonb,
                status VARCHAR(50) DEFAULT 'draft',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                executed_at TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS ai_competitor_analysis (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                lead_id UUID REFERENCES revenue_leads(id),
                competitors JSONB DEFAULT '[]'::jsonb,
                analysis JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS ai_churn_predictions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                lead_id UUID REFERENCES revenue_leads(id),
                churn_probability FLOAT DEFAULT 0.0,
                risk_level VARCHAR(20),
                prediction_data JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS ai_upsell_recommendations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                lead_id UUID REFERENCES revenue_leads(id),
                recommendations JSONB DEFAULT '[]'::jsonb,
                total_potential FLOAT DEFAULT 0.0,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS ai_revenue_forecasts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                months_ahead INT,
                forecast_data JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS unified_brain_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                system VARCHAR(100),
                action VARCHAR(100),
                data JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Create indexes for new tables
            CREATE INDEX IF NOT EXISTS idx_email_sequences_lead ON ai_email_sequences(lead_id);
            CREATE INDEX IF NOT EXISTS idx_competitor_analysis_lead ON ai_competitor_analysis(lead_id);
            CREATE INDEX IF NOT EXISTS idx_churn_predictions_lead ON ai_churn_predictions(lead_id);
            CREATE INDEX IF NOT EXISTS idx_upsell_recommendations_lead ON ai_upsell_recommendations(lead_id);
            CREATE INDEX IF NOT EXISTS idx_unified_brain_logs_system ON unified_brain_logs(system);
            CREATE INDEX IF NOT EXISTS idx_unified_brain_logs_action ON unified_brain_logs(action);
            CREATE INDEX IF NOT EXISTS idx_unified_brain_logs_created ON unified_brain_logs(created_at DESC);
        """)

        conn.commit()
        cursor.close()
        conn.close()

    async def identify_new_leads(self, criteria: dict[str, Any]) -> list[Lead]:
        """Autonomously identify new leads based on criteria"""
        try:
            # Use AI to generate lead search parameters
            prompt = f"""Based on these criteria: {json.dumps(criteria)}
            Generate search parameters for finding roofing contractor leads.
            Include: location, company size, indicators of need, budget range.
            Return as JSON."""

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a lead generation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            search_params = json.loads(response.choices[0].message.content)

            # Simulate lead discovery (would integrate with real data sources)
            new_leads = await self._discover_leads(search_params)

            # Store leads in database
            stored_leads = []
            for lead_data in new_leads:
                lead_id = await self._store_lead(lead_data)
                if lead_id:
                    stored_leads.append(lead_id)
                    await self._log_action(lead_id, RevenueAction.IDENTIFY_LEAD, lead_data)

            logger.info(f"Identified {len(stored_leads)} new leads")
            return stored_leads

        except Exception as e:
            logger.error(f"Failed to identify leads: {e}")
            return []

    async def qualify_lead(self, lead_id: str) -> tuple[float, dict]:
        """Autonomously qualify a lead using AI analysis with advanced ML-based scoring"""
        try:
            # Get lead data
            lead = await self._get_lead(lead_id)
            if not lead:
                return 0.0, {}

            # AI qualification prompt with enhanced scoring
            prompt = f"""Analyze this lead for roofing services potential:
            Company: {lead.get('company_name')}
            Contact: {lead.get('contact_name')}
            Email: {lead.get('email')}
            Phone: {lead.get('phone')}
            Website: {lead.get('website')}
            Source: {lead.get('source')}
            Metadata: {json.dumps(lead.get('metadata', {}))}

            Perform ADVANCED AI LEAD SCORING based on:
            1. Likelihood to need roofing services (0-25 points)
            2. Budget availability and financial capacity (0-25 points)
            3. Decision-making authority and role (0-20 points)
            4. Timeline urgency and buying signals (0-15 points)
            5. Fit with our services and company size (0-15 points)

            Additional analysis:
            6. Digital presence quality (website, social media)
            7. Competitor engagement likelihood
            8. Churn risk if converted
            9. Upsell/cross-sell potential
            10. Customer lifetime value estimate

            Return JSON with:
            - score: 0-100 total points
            - reasons: array of key factors
            - recommended_action: next step
            - estimated_value: deal size estimate
            - churn_risk: 0-1 probability
            - upsell_potential: low/medium/high
            - lifetime_value: estimated LTV
            - buying_signals: array of detected signals
            - competitor_risk: 0-1 probability they're with competitor"""

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an advanced AI lead qualification expert with deep sales intelligence capabilities. Provide detailed, data-driven analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )

            qualification = json.loads(response.choices[0].message.content)
            score = qualification.get('score', 0) / 100.0

            # Update lead with qualification
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE revenue_leads
                SET score = %s,
                    stage = %s,
                    value_estimate = %s,
                    metadata = metadata || %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (
                score,
                'qualified' if score > 0.6 else 'contacted',
                qualification.get('estimated_value', 0),
                json.dumps({'qualification': qualification}),
                lead_id
            ))

            conn.commit()
            cursor.close()
            conn.close()

            await self._log_action(lead_id, RevenueAction.QUALIFY_LEAD, qualification)

            # Log to unified brain
            await self._log_to_unified_brain(
                action='lead_qualification',
                lead_id=lead_id,
                score=score,
                qualification=qualification
            )

            logger.info(f"Qualified lead {lead_id} with score {score} (LTV: ${qualification.get('lifetime_value', 0)}, Churn Risk: {qualification.get('churn_risk', 0):.1%})")
            return score, qualification

        except Exception as e:
            logger.error(f"Failed to qualify lead: {e}")
            await self._log_to_unified_brain(
                action='lead_qualification_error',
                lead_id=lead_id,
                error=str(e)
            )
            return 0.0, {}

    async def create_personalized_outreach(self, lead_id: str) -> dict:
        """Create personalized outreach using AI"""
        try:
            lead = await self._get_lead(lead_id)
            if not lead:
                return {}

            # Generate personalized email
            prompt = f"""Create a personalized cold email for:
            Company: {lead.get('company_name')}
            Contact: {lead.get('contact_name', 'Business Owner')}

            Our service: AI-powered roofing business automation
            Goal: Schedule a demo or consultation

            Make it personal, compelling, and focused on their potential ROI.
            Include subject line and body."""

            response = anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=500
            )

            email_content = response.content[0].text

            # Parse and structure email
            lines = email_content.split('\n')
            subject = lines[0].replace('Subject:', '').strip()
            body = '\n'.join(lines[2:]).strip()

            outreach = {
                'subject': subject,
                'body': body,
                'lead_id': lead_id,
                'scheduled_send': datetime.now(timezone.utc) + timedelta(hours=1)
            }

            # Store outreach
            await self._schedule_email(outreach)
            await self._log_action(lead_id, RevenueAction.SEND_OUTREACH, outreach)

            # Update lead stage
            await self._update_lead_stage(lead_id, LeadStage.CONTACTED)

            logger.info(f"Created outreach for lead {lead_id}")
            return outreach

        except Exception as e:
            logger.error(f"Failed to create outreach: {e}")
            return {}

    async def generate_proposal(self, lead_id: str, requirements: dict) -> dict:
        """Generate AI-powered proposal"""
        try:
            lead = await self._get_lead(lead_id)
            if not lead:
                return {}

            # Generate proposal using AI
            prompt = f"""Create a roofing services proposal for:
            Company: {lead.get('company_name')}
            Requirements: {json.dumps(requirements)}

            Include:
            1. Executive summary
            2. Scope of services
            3. Pricing breakdown
            4. Timeline
            5. ROI projection
            6. Next steps

            Make it professional and compelling."""

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a proposal writing expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            proposal_content = response.choices[0].message.content

            # Calculate pricing
            pricing = await self._calculate_dynamic_pricing(requirements)

            proposal = {
                'lead_id': lead_id,
                'content': proposal_content,
                'pricing': pricing,
                'valid_until': datetime.now(timezone.utc) + timedelta(days=30),
                'created_at': datetime.now(timezone.utc).isoformat()
            }

            # Create opportunity
            await self._create_opportunity(lead_id, pricing.get('total', 0))

            # Update lead stage
            await self._update_lead_stage(lead_id, LeadStage.PROPOSAL_SENT)
            await self._log_action(lead_id, RevenueAction.CREATE_PROPOSAL, proposal)

            logger.info(f"Generated proposal for lead {lead_id}")
            return proposal

        except Exception as e:
            logger.error(f"Failed to generate proposal: {e}")
            return {}

    async def handle_negotiation(self, lead_id: str, client_response: str) -> dict:
        """AI-powered negotiation handling"""
        try:
            # Analyze client response
            prompt = f"""Analyze this client response to our proposal:
            "{client_response}"

            Determine:
            1. Client sentiment (positive/negative/neutral)
            2. Key objections or concerns
            3. Negotiation strategy
            4. Recommended response
            5. Suggested concessions (if any)

            Return as JSON."""

            response = anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=500
            )

            analysis = json.loads(response.content[0].text)

            # Generate negotiation response
            negotiation_response = await self._generate_negotiation_response(
                lead_id,
                analysis
            )

            # Update opportunity
            await self._update_opportunity_stage(lead_id, "negotiating")
            await self._log_action(lead_id, RevenueAction.NEGOTIATE_DEAL, negotiation_response)

            logger.info(f"Handled negotiation for lead {lead_id}")
            return negotiation_response

        except Exception as e:
            logger.error(f"Failed to handle negotiation: {e}")
            return {}

    async def close_deal(self, lead_id: str, terms: dict) -> bool:
        """Autonomously close the deal"""
        try:
            # Generate closing documents
            await self._generate_closing_documents(lead_id, terms)

            # Update opportunity as won
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE revenue_opportunities
                SET stage = 'closed_won',
                    won = true,
                    closed_at = NOW(),
                    value = %s
                WHERE lead_id = %s
            """, (terms.get('final_value', 0), lead_id))

            # Update lead stage
            cursor.execute("""
                UPDATE revenue_leads
                SET stage = 'won',
                    updated_at = NOW()
                WHERE id = %s
            """, (lead_id,))

            # Update metrics
            cursor.execute("""
                INSERT INTO revenue_metrics (metric_date, deals_closed, revenue_generated)
                VALUES (CURRENT_DATE, 1, %s)
                ON CONFLICT (metric_date)
                DO UPDATE SET
                    deals_closed = revenue_metrics.deals_closed + 1,
                    revenue_generated = revenue_metrics.revenue_generated + EXCLUDED.revenue_generated
            """, (terms.get('final_value', 0),))

            conn.commit()
            cursor.close()
            conn.close()

            await self._log_action(lead_id, RevenueAction.CLOSE_DEAL, terms)

            # Trigger onboarding workflow
            await self._initiate_onboarding(lead_id)

            logger.info(f"Closed deal for lead {lead_id} - Value: ${terms.get('final_value', 0)}")
            return True

        except Exception as e:
            logger.error(f"Failed to close deal: {e}")
            return False

    async def run_revenue_workflow(self, lead_id: str):
        """Run complete autonomous revenue workflow for a lead"""
        try:
            workflow_id = str(uuid.uuid4())
            self.active_workflows[workflow_id] = {
                'lead_id': lead_id,
                'start_time': datetime.now(timezone.utc),
                'status': 'running'
            }

            # 1. Qualify the lead
            score, qualification = await self.qualify_lead(lead_id)

            if score < 0.3:
                # Low quality - nurture
                await self._schedule_nurture_campaign(lead_id)
                return

            # 2. Send personalized outreach
            await self.create_personalized_outreach(lead_id)

            # 3. Wait for response (simulated)
            await asyncio.sleep(2)

            # 4. Generate proposal if qualified
            if score > 0.6:
                requirements = qualification.get('requirements', {})
                await self.generate_proposal(lead_id, requirements)

                # 5. Handle negotiation (if needed)
                # This would be triggered by client response

                # 6. Close deal
                # This would be triggered by acceptance

            self.active_workflows[workflow_id]['status'] = 'completed'
            self.active_workflows[workflow_id]['end_time'] = datetime.now(timezone.utc)

            logger.info(f"Completed revenue workflow {workflow_id} for lead {lead_id}")

        except Exception as e:
            logger.error(f"Revenue workflow failed: {e}")
            self.active_workflows[workflow_id]['status'] = 'failed'

    # Helper methods
    async def _get_lead(self, lead_id: str) -> Optional[dict]:
        """Get lead data from database"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM revenue_leads WHERE id = %s", (lead_id,))
            lead = cursor.fetchone()
            cursor.close()
            conn.close()
            return lead
        except Exception as e:
            logger.error(f"Failed to get lead: {e}")
            return None

    async def _store_lead(self, lead_data: dict) -> Optional[str]:
        """Store new lead in database"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            lead_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO revenue_leads
                (id, company_name, contact_name, email, phone, website, source, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                lead_id,
                lead_data.get('company_name'),
                lead_data.get('contact_name'),
                lead_data.get('email'),
                lead_data.get('phone'),
                lead_data.get('website'),
                lead_data.get('source', 'ai_discovery'),
                json.dumps(lead_data.get('metadata', {}))
            ))

            conn.commit()
            cursor.close()
            conn.close()

            return lead_id
        except Exception as e:
            logger.error(f"Failed to store lead: {e}")
            return None

    async def _log_action(self, lead_id: str, action: RevenueAction, data: dict):
        """Log revenue action"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO revenue_actions
                (lead_id, action_type, action_data, result, success, executed_by)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                lead_id,
                action.value,
                json.dumps(data),
                json.dumps({}),
                True,
                "autonomous_system"
            ))

            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log action: {e}")

    async def _update_lead_stage(self, lead_id: str, stage: LeadStage):
        """Update lead stage"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE revenue_leads
                SET stage = %s, updated_at = NOW()
                WHERE id = %s
            """, (stage.value, lead_id))

            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update lead stage: {e}")

    async def _discover_leads(self, search_params: dict) -> list[dict]:
        """Discover leads using Perplexity AI for real-time web search"""
        try:
            from ai_advanced_providers import advanced_ai

            # Build search query from params
            location = search_params.get('location', 'United States')
            company_size = search_params.get('company_size', 'small to medium')
            indicators = search_params.get('indicators', ['growth', 'hiring', 'expansion'])

            discovery_prompt = f"""Search for roofing contractor businesses that need CRM/automation software.

Search criteria:
- Location: {location}
- Company size: {company_size}
- Looking for indicators: {', '.join(indicators) if isinstance(indicators, list) else indicators}

Find real roofing companies showing these buying signals:
1. Outdated or no website
2. Manual scheduling/estimating processes
3. Growing rapidly and need systems
4. Recently funded or expanding
5. Posting job ads for office staff
6. Complaints about disorganization online

Return JSON array with 5-10 leads, each containing:
- company_name: string
- contact_name: string (owner/manager if found)
- email: string (if available, otherwise null)
- phone: string (if available, otherwise null)
- website: string (if available)
- location: string (city, state)
- source: string (where found - yelp/google/linkedin/etc)
- buying_signals: array of strings
- estimated_value: number (potential deal size 1000-50000)
- confidence_score: number (0.0-1.0)

Return ONLY valid JSON array, no other text."""

            result = advanced_ai.search_with_perplexity(discovery_prompt)

            if result and result.get("answer"):
                try:
                    answer = result["answer"]
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', answer)
                    if json_match:
                        leads = json.loads(json_match.group())
                        logger.info(f"Discovered {len(leads)} leads via Perplexity")
                        return leads
                except json.JSONDecodeError:
                    logger.warning("Could not parse lead discovery response as JSON")

                # Fallback: use OpenAI to extract structured data
                extraction_response = openai.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "Extract lead data from search results. Return only valid JSON array."},
                        {"role": "user", "content": f"Extract leads from: {result['answer']}\n\nReturn JSON array with company_name, contact_name, email, phone, website, location, source, buying_signals, estimated_value, confidence_score for each lead."}
                    ],
                    temperature=0.3
                )

                extracted = json.loads(extraction_response.choices[0].message.content)
                if isinstance(extracted, list):
                    logger.info(f"Extracted {len(extracted)} leads from discovery")
                    return extracted

            # Perplexity unavailable - check database for opportunities
            logger.warning("Perplexity unavailable, checking database for opportunities")
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor

                # Use the validated DB config - no hardcoded defaults
                from revenue_generation_system import DB_CONFIG
                conn = psycopg2.connect(**DB_CONFIG)
                cur = conn.cursor(cursor_factory=RealDictCursor)

                # First, check for existing leads in NEW or CONTACTED status
                cur.execute("""
                    SELECT id, company_name, contact_name, email,
                           phone, website as location, source,
                           COALESCE(metadata->>'buying_signals', '[]')::jsonb as buying_signals,
                           COALESCE(value_estimate, 0) as estimated_value,
                           COALESCE(score, 0.5) as confidence_score
                    FROM revenue_leads
                    WHERE stage IN ('new', 'contacted')
                      AND created_at > NOW() - INTERVAL '30 days'
                    ORDER BY value_estimate DESC NULLS LAST
                    LIMIT 10
                """)

                existing_leads = cur.fetchall()
                if existing_leads:
                    logger.info(f"Found {len(existing_leads)} existing leads in database")
                    cur.close()
                    conn.close()
                    return [dict(lead) for lead in existing_leads]

                # Second, find existing customers with upsell/cross-sell potential
                cur.execute("""
                    SELECT DISTINCT
                        c.id::text as id,
                        c.company_name as company_name,
                        c.first_name || ' ' || c.last_name as contact_name,
                        c.email,
                        c.phone,
                        COALESCE(c.city, '') || ', ' || COALESCE(c.state, '') as location,
                        'existing_customer' as source,
                        ARRAY['repeat_customer', 'has_history'] as buying_signals,
                        COALESCE(
                            (SELECT AVG(total_amount) FROM invoices WHERE customer_id = c.id AND status = 'paid'),
                            5000
                        )::float as estimated_value,
                        0.75 as confidence_score
                    FROM customers c
                    WHERE c.email IS NOT NULL
                      AND c.created_at < NOW() - INTERVAL '30 days'
                      AND NOT EXISTS (
                          SELECT 1 FROM jobs j
                          WHERE j.customer_id = c.id
                            AND j.created_at > NOW() - INTERVAL '90 days'
                      )
                    ORDER BY estimated_value DESC
                    LIMIT 10
                """)

                customer_opportunities = cur.fetchall()
                cur.close()
                conn.close()

                if customer_opportunities:
                    logger.info(f"Found {len(customer_opportunities)} customer upsell opportunities")
                    return [dict(opp) for opp in customer_opportunities]

                logger.info("No leads or opportunities found - need Perplexity API for cold outreach")
                return []

            except Exception as db_error:
                logger.warning(f"Could not query leads/opportunities: {db_error}")
                return []

        except Exception as e:
            logger.error(f"Lead discovery failed: {e}")
            return []

    async def _calculate_dynamic_pricing(self, requirements: dict) -> dict:
        """Calculate dynamic pricing using AI analysis of requirements"""
        try:
            prompt = f"""Calculate dynamic pricing for a roofing software project based on these requirements:
            {json.dumps(requirements)}

            Consider: complexity, scope, market rates (base ~$5000), and value provided.

            Return JSON with:
            - base: base price (float)
            - total: total price with adjustments (float)
            - discount: discount amount if applicable (float)
            - terms: payment terms string
            - reasoning: short explanation
            """

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a pricing expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            pricing = json.loads(response.choices[0].message.content)
            return pricing
        except Exception as e:
            logger.error(f"AI pricing calculation failed: {e}")
            # Fallback
            base_price = 5000
            return {
                'base': base_price,
                'total': base_price * 1.2,
                'discount': 0,
                'terms': '50% upfront, 50% on completion',
                'reasoning': 'Fallback pricing due to AI error'
            }

    async def _schedule_email(self, email_data: dict):
        """Schedule email for sending"""
        try:
            recipient = email_data.get('to')
            metadata = email_data.get('metadata') or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            metadata['is_test'] = bool(email_data.get('is_test')) or _is_test_email(recipient)

            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Store email in outbound queue
            cursor.execute("""
                INSERT INTO ai_email_queue
                (id, recipient, subject, body, scheduled_for, status, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                str(uuid.uuid4()),
                recipient,
                email_data.get('subject'),
                email_data.get('body'),
                datetime.now(timezone.utc),
                'queued',
                json.dumps(metadata)
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Email queued for {email_data.get('to')}")
        except Exception as e:
            logger.error(f"Failed to queue email: {e}")

    async def _create_opportunity(self, lead_id: str, value: float) -> str:
        """Create revenue opportunity"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            opp_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO revenue_opportunities
                (id, lead_id, title, value, probability, expected_close_date, stage)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                opp_id,
                lead_id,
                "Roofing Services Opportunity",
                value,
                0.5,
                datetime.now(timezone.utc) + timedelta(days=30),
                "proposal_sent"
            ))

            conn.commit()
            cursor.close()
            conn.close()

            return opp_id
        except Exception as e:
            logger.error(f"Failed to create opportunity: {e}")
            return ""

    async def _schedule_nurture_campaign(self, lead_id: str):
        """Schedule nurture campaign for low-quality leads"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Create nurture campaign
            campaign_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_nurture_campaigns
                (id, lead_id, campaign_type, status, next_touch_date, touch_count)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                campaign_id,
                lead_id,
                'educational_drip',
                'active',
                datetime.now(timezone.utc) + timedelta(days=3),
                0
            ))

            # Schedule first touch
            cursor.execute("""
                INSERT INTO ai_campaign_touches
                (id, campaign_id, touch_type, scheduled_for, content_template)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                str(uuid.uuid4()),
                campaign_id,
                'email',
                datetime.now(timezone.utc) + timedelta(days=3),
                'educational_content_1'
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Nurture campaign scheduled for lead {lead_id}")
        except Exception as e:
            logger.error(f"Failed to schedule nurture campaign: {e}")

    async def _generate_closing_documents(self, lead_id: str, terms: dict) -> dict:
        """Generate closing documents using AI"""
        try:
            lead = await self._get_lead(lead_id)
            prompt = f"""Generate the structure for closing documents for:
            Client: {lead.get('company_name', 'Client')}
            Terms: {json.dumps(terms)}

            Return JSON with:
            - contract_id: generated UUID
            - sections: list of contract sections (Scope, Terms, etc.)
            - key_clauses: list of important clauses based on terms
            - signature_block: text for signature area
            """

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a legal document assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            docs = json.loads(response.choices[0].message.content)
            docs['generated_at'] = datetime.now(timezone.utc).isoformat()
            return docs

        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def _initiate_onboarding(self, lead_id: str):
        """Initiate customer onboarding"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Create onboarding record
            onboarding_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_onboarding_workflows
                (id, lead_id, status, current_step, total_steps, started_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                onboarding_id,
                lead_id,
                'in_progress',
                1,
                5,
                datetime.now(timezone.utc)
            ))

            # Create onboarding steps
            steps = [
                ('welcome_email', 'Send welcome email and credentials'),
                ('data_collection', 'Collect necessary customer information'),
                ('system_setup', 'Configure customer account and preferences'),
                ('training_schedule', 'Schedule training sessions'),
                ('first_project', 'Initiate first project or service')
            ]

            for idx, (step_name, description) in enumerate(steps, 1):
                cursor.execute("""
                    INSERT INTO ai_onboarding_steps
                    (id, workflow_id, step_number, step_name, description, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()),
                    onboarding_id,
                    idx,
                    step_name,
                    description,
                    'pending' if idx > 1 else 'in_progress'
                ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Onboarding initiated for lead {lead_id}")
            return onboarding_id
        except Exception as e:
            logger.error(f"Failed to initiate onboarding: {e}")
            return None

    async def _generate_negotiation_response(self, lead_id: str, analysis: dict) -> dict:
        """Generate negotiation response based on analysis"""
        return {
            'response': "We understand your concerns and are happy to work with you...",
            'concessions': analysis.get('suggested_concessions', []),
            'next_steps': "Schedule a call to finalize terms"
        }

    async def _update_opportunity_stage(self, lead_id: str, stage: str):
        """Update opportunity stage"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE revenue_opportunities
                SET stage = %s, updated_at = NOW()
                WHERE lead_id = %s
            """, (stage, lead_id))

            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update opportunity: {e}")

    # NEW ENHANCEMENTS

    async def generate_email_sequence(self, lead_id: str, sequence_type: str = "nurture") -> dict:
        """Generate automated multi-touch email sequence using AI"""
        try:
            lead = await self._get_lead(lead_id)
            if not lead:
                return {}

            prompt = f"""Generate a 5-email automated sequence for:

            Lead: {lead.get('company_name')} ({lead.get('contact_name')})
            Stage: {lead.get('stage')}
            Score: {lead.get('score')}
            Value: ${lead.get('value_estimate', 0)}
            Source: {lead.get('source')}
            Type: {sequence_type}

            Create emails for days 0, 3, 7, 14, 21 with:
            1. Day 0: Initial value proposition
            2. Day 3: Educational content + social proof
            3. Day 7: Case study + ROI calculator
            4. Day 14: Personalized demo offer
            5. Day 21: Urgency-based close attempt

            For each email return JSON:
            - day: number
            - subject: compelling subject line
            - preview_text: email preview
            - body: full email HTML/text
            - cta: call to action
            - goal: email objective
            - personalization_tokens: fields to customize

            Make highly personalized, benefit-focused, and conversion-optimized."""

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert email marketing copywriter specializing in B2B sales sequences with proven conversion rates."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2500
            )

            sequence = json.loads(response.choices[0].message.content)

            # Store sequence in database
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            sequence_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_email_sequences
                (id, lead_id, sequence_type, emails, created_at, status)
                VALUES (%s, %s, %s, %s, NOW(), %s)
            """, (sequence_id, lead_id, sequence_type, json.dumps(sequence), 'active'))

            conn.commit()
            cursor.close()
            conn.close()

            await self._log_to_unified_brain(
                action='email_sequence_generated',
                lead_id=lead_id,
                sequence_type=sequence_type,
                emails_count=len(sequence.get('emails', []))
            )

            logger.info(f"Generated {sequence_type} email sequence for lead {lead_id}")
            return {'sequence_id': sequence_id, 'emails': sequence}

        except Exception as e:
            logger.error(f"Failed to generate email sequence: {e}")
            return {}

    async def analyze_competitor_pricing(self, lead_id: str, competitors: list[str] = None) -> dict:
        """Analyze competitor pricing and positioning using AI"""
        try:
            lead = await self._get_lead(lead_id)

            prompt = f"""Analyze competitor pricing for roofing software targeting:

            Lead: {lead.get('company_name')}
            Size: {lead.get('metadata', {}).get('company_size', 'unknown')}
            Location: {lead.get('location', 'unknown')}

            Main competitors: {', '.join(competitors) if competitors else 'JobNimbus, AccuLynx, CompanyCam, Roofr'}

            Provide competitive intelligence:
            1. Estimated competitor pricing (monthly/annual)
            2. Feature comparison vs our offering
            3. Competitor strengths and weaknesses
            4. Market positioning gaps
            5. Recommended pricing strategy
            6. Differentiation opportunities
            7. Win/loss factors

            Return JSON with detailed competitive analysis."""

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a competitive intelligence analyst with deep knowledge of the roofing software market."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )

            analysis = json.loads(response.choices[0].message.content)

            # Store analysis
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_competitor_analysis
                (id, lead_id, competitors, analysis, created_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, (str(uuid.uuid4()), lead_id, json.dumps(competitors or []), json.dumps(analysis)))

            conn.commit()
            cursor.close()
            conn.close()

            await self._log_to_unified_brain(
                action='competitor_analysis',
                lead_id=lead_id,
                competitors=competitors,
                analysis=analysis
            )

            return analysis

        except Exception as e:
            logger.error(f"Competitor analysis failed: {e}")
            return {}

    async def predict_churn_risk(self, lead_id: str) -> dict:
        """Predict churn risk for a lead/customer using AI"""
        try:
            lead = await self._get_lead(lead_id)

            # Get historical data
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM revenue_actions
                WHERE lead_id = %s
                ORDER BY created_at DESC
                LIMIT 50
            """, (lead_id,))

            actions = cursor.fetchall()
            cursor.close()
            conn.close()

            prompt = f"""Predict churn risk for this lead/customer:

            Company: {lead.get('company_name')}
            Stage: {lead.get('stage')}
            Score: {lead.get('score')}
            Last Contact: {lead.get('last_contact')}
            Value: ${lead.get('value_estimate', 0)}
            Recent Actions: {len(actions)} interactions

            Analyze churn indicators:
            1. Engagement decline
            2. Support ticket patterns
            3. Feature adoption
            4. Payment history
            5. Competitive signals
            6. Contract status

            Return JSON:
            - churn_probability: 0-1 score
            - risk_level: low/medium/high/critical
            - key_factors: array of risk indicators
            - retention_actions: recommended interventions
            - estimated_impact: revenue at risk
            - timeline: expected churn window
            """

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a customer success AI specializing in churn prediction and retention strategies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            prediction = json.loads(response.choices[0].message.content)

            # Store prediction
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_churn_predictions
                (id, lead_id, churn_probability, risk_level, prediction_data, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                str(uuid.uuid4()),
                lead_id,
                prediction.get('churn_probability', 0),
                prediction.get('risk_level', 'unknown'),
                json.dumps(prediction)
            ))

            conn.commit()
            cursor.close()
            conn.close()

            await self._log_to_unified_brain(
                action='churn_prediction',
                lead_id=lead_id,
                churn_risk=prediction.get('churn_probability'),
                risk_level=prediction.get('risk_level')
            )

            return prediction

        except Exception as e:
            logger.error(f"Churn prediction failed: {e}")
            return {}

    async def generate_upsell_recommendations(self, lead_id: str) -> dict:
        """Generate AI-powered upsell and cross-sell recommendations"""
        try:
            lead = await self._get_lead(lead_id)

            prompt = f"""Generate upsell/cross-sell recommendations for:

            Customer: {lead.get('company_name')}
            Current Value: ${lead.get('value_estimate', 0)}
            Stage: {lead.get('stage')}
            Metadata: {json.dumps(lead.get('metadata', {}))}

            Analyze opportunities for:
            1. Feature upgrades
            2. Additional users/licenses
            3. Premium support packages
            4. Training and consulting
            5. Integration add-ons
            6. Advanced analytics
            7. API access
            8. White-label options

            For each opportunity return:
            - product_name: what to sell
            - value_proposition: why they need it
            - price_range: expected pricing
            - probability: likelihood to buy (0-1)
            - timing: when to pitch (immediate/30d/90d)
            - expected_revenue: additional MRR/ARR
            - effort_level: sales effort required

            Prioritize by revenue potential and fit."""

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying upsell and cross-sell opportunities with high conversion rates."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )

            recommendations = json.loads(response.choices[0].message.content)

            # Store recommendations
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_upsell_recommendations
                (id, lead_id, recommendations, total_potential, created_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, (
                str(uuid.uuid4()),
                lead_id,
                json.dumps(recommendations),
                sum([r.get('expected_revenue', 0) for r in recommendations.get('opportunities', [])])
            ))

            conn.commit()
            cursor.close()
            conn.close()

            await self._log_to_unified_brain(
                action='upsell_recommendations',
                lead_id=lead_id,
                opportunities_count=len(recommendations.get('opportunities', [])),
                total_potential=sum([r.get('expected_revenue', 0) for r in recommendations.get('opportunities', [])])
            )

            return recommendations

        except Exception as e:
            logger.error(f"Upsell recommendations failed: {e}")
            return {}

    async def forecast_revenue(self, months_ahead: int = 6) -> dict:
        """Generate AI-powered revenue forecast"""
        try:
            # Get historical data
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    DATE_TRUNC('month', created_at) as month,
                    COUNT(*) as leads,
                    COUNT(*) FILTER (WHERE stage = 'won') as wins,
                    SUM(value_estimate) FILTER (WHERE stage = 'won') as revenue,
                    AVG(value_estimate) as avg_deal_size
                FROM revenue_leads
                WHERE created_at > NOW() - INTERVAL '12 months'
                GROUP BY DATE_TRUNC('month', created_at)
                ORDER BY month
            """)

            historical = cursor.fetchall()

            cursor.execute("""
                SELECT
                    stage,
                    COUNT(*) as count,
                    SUM(value_estimate) as value
                FROM revenue_leads
                WHERE stage IN ('qualified', 'proposal_sent', 'negotiating')
                GROUP BY stage
            """)

            pipeline = cursor.fetchall()
            cursor.close()
            conn.close()

            prompt = f"""Generate revenue forecast for next {months_ahead} months:

            Historical Performance (last 12 months):
            {json.dumps([dict(h) for h in historical], default=str)}

            Current Pipeline:
            {json.dumps([dict(p) for p in pipeline], default=str)}

            Provide month-by-month forecast with:
            - expected_revenue: forecasted revenue
            - confidence_interval: (low, high) range
            - new_leads_needed: acquisition target
            - conversion_assumptions: expected rates
            - risk_factors: potential issues
            - growth_rate: month-over-month %
            - cumulative_total: running total

            Include best-case, likely-case, and worst-case scenarios."""

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a revenue forecasting expert with deep statistical modeling capabilities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )

            forecast = json.loads(response.choices[0].message.content)

            # Store forecast
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            forecast_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_revenue_forecasts
                (id, months_ahead, forecast_data, created_at)
                VALUES (%s, %s, %s, NOW())
            """, (forecast_id, months_ahead, json.dumps(forecast)))

            conn.commit()
            cursor.close()
            conn.close()

            await self._log_to_unified_brain(
                action='revenue_forecast',
                months_ahead=months_ahead,
                forecast_id=forecast_id,
                total_forecast=sum([m.get('expected_revenue', 0) for m in forecast.get('monthly_forecast', [])])
            )

            return forecast

        except Exception as e:
            logger.error(f"Revenue forecasting failed: {e}")
            return {}

    async def _log_to_unified_brain(self, action: str, **kwargs):
        """Log all revenue actions to unified brain for centralized monitoring"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO unified_brain_logs
                (id, system, action, data, created_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, (
                str(uuid.uuid4()),
                'revenue_generation_system',
                action,
                json.dumps(kwargs)
            ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            # Non-critical - don't fail on logging errors
            logger.warning(f"Failed to log to unified brain: {e}")

# Global instance - create lazily
revenue_system = None

def get_revenue_system():
    """Get or create revenue system instance"""
    global revenue_system
    if revenue_system is None:
        revenue_system = AutonomousRevenueSystem()
    return revenue_system

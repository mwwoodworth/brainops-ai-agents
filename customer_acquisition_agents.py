#!/usr/bin/env python3
"""
Customer Acquisition AI Agents
Autonomous agents that find, qualify, and convert leads automatically
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

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - NO hardcoded fallback credentials
def _get_db_config():
    """Get database configuration from environment variables."""
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_port = os.getenv("DB_PORT", "5432")

    missing = []
    if not db_host:
        missing.append("DB_HOST")
    if not db_name:
        missing.append("DB_NAME")
    if not db_user:
        missing.append("DB_USER")
    if not db_password:
        missing.append("DB_PASSWORD")

    if missing:
        raise RuntimeError(
            f"Required environment variables not set: {', '.join(missing)}. "
            "Set these variables before using customer acquisition agents."
        )

    return {
        "host": db_host,
        "database": db_name,
        "user": db_user,
        "password": db_password,
        "port": int(db_port)
    }


def _get_db_connection(**kwargs):
    """Get database connection with validated config."""
    db_config = _get_db_config()
    db_config.update(kwargs)
    return psycopg2.connect(**db_config)

# AI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_AVAILABLE = openai is not None and bool(OPENAI_API_KEY)
if openai is not None and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
elif openai is None:
    logger.warning("OpenAI SDK not installed - OpenAI features disabled")
else:
    logger.warning("OpenAI API key not found - OpenAI features disabled")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_AVAILABLE = anthropic is not None and bool(ANTHROPIC_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_AVAILABLE else None
if anthropic is None:
    logger.warning("Anthropic SDK not installed - Anthropic features disabled")
elif not ANTHROPIC_API_KEY:
    logger.warning("Anthropic API key not found - Anthropic features disabled")


def _require_openai(feature: str) -> bool:
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI unavailable - %s", feature)
        return False
    return True


def _require_anthropic(feature: str) -> bool:
    if not ANTHROPIC_AVAILABLE:
        logger.warning("Anthropic unavailable - %s", feature)
        return False
    return True

class AcquisitionChannel(Enum):
    """Customer acquisition channels"""
    WEB_SEARCH = "web_search"
    SOCIAL_MEDIA = "social_media"
    REFERRAL = "referral"
    CONTENT_MARKETING = "content_marketing"
    EMAIL_CAMPAIGN = "email_campaign"
    COLD_CALLING = "cold_calling"
    PARTNERSHIPS = "partnerships"
    EVENTS = "events"

class LeadIntent(Enum):
    """Lead intent signals"""
    HIGH_INTENT = "high_intent"  # Actively looking for solution
    MEDIUM_INTENT = "medium_intent"  # Interested but not urgent
    LOW_INTENT = "low_intent"  # Future potential
    EDUCATIONAL = "educational"  # Just researching

@dataclass
class AcquisitionTarget:
    """Represents a potential customer target"""
    id: str
    company_name: str
    industry: str
    size: str
    location: str
    website: Optional[str]
    social_profiles: dict[str, str]
    decision_makers: list[dict]
    pain_points: list[str]
    budget_range: tuple[float, float]
    intent_score: float
    acquisition_channel: AcquisitionChannel
    metadata: dict[str, Any]

class CustomerAcquisitionAgent:
    """Base agent for customer acquisition"""

    _tables_ensured = False  # Class-level flag to prevent repeated table checks

    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.agent_id = str(uuid.uuid4())
        # Lazy table initialization - don't connect at import time
        logger.info(f"Initialized {agent_type} acquisition agent")

    def _ensure_tables(self):
        """Ensure acquisition tables exist - lazy initialization"""
        if CustomerAcquisitionAgent._tables_ensured:
            return
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS acquisition_targets (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    company_name VARCHAR(255) NOT NULL,
                    industry VARCHAR(100),
                    company_size VARCHAR(50),
                    location VARCHAR(255),
                    website VARCHAR(255),
                    social_profiles JSONB DEFAULT '{}'::jsonb,
                    decision_makers JSONB DEFAULT '[]'::jsonb,
                    pain_points TEXT[],
                    budget_min FLOAT,
                    budget_max FLOAT,
                    intent_score FLOAT DEFAULT 0.0,
                    acquisition_channel VARCHAR(50),
                    status VARCHAR(50) DEFAULT 'identified',
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    contacted_at TIMESTAMPTZ,
                    converted_at TIMESTAMPTZ
                );

                CREATE TABLE IF NOT EXISTS acquisition_campaigns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    campaign_name VARCHAR(255),
                    campaign_type VARCHAR(50),
                    target_criteria JSONB,
                    budget FLOAT,
                    start_date DATE,
                    end_date DATE,
                    status VARCHAR(50) DEFAULT 'active',
                    metrics JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS acquisition_activities (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    target_id UUID REFERENCES acquisition_targets(id),
                    activity_type VARCHAR(50),
                    activity_data JSONB,
                    outcome VARCHAR(50),
                    engagement_score FLOAT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    agent_id UUID
                );

                CREATE INDEX IF NOT EXISTS idx_acquisition_targets_intent ON acquisition_targets(intent_score DESC);
                CREATE INDEX IF NOT EXISTS idx_acquisition_targets_status ON acquisition_targets(status);
                CREATE INDEX IF NOT EXISTS idx_acquisition_activities_target ON acquisition_activities(target_id);
            """)

            conn.commit()
            cursor.close()
            conn.close()
            CustomerAcquisitionAgent._tables_ensured = True
        except Exception as e:
            logger.warning(f"Could not ensure acquisition tables: {e}")

class WebSearchAgent(CustomerAcquisitionAgent):
    """Agent that searches the web for potential customers"""

    def __init__(self):
        super().__init__("WebSearchAgent")

    async def search_for_leads(self, criteria: dict) -> list[AcquisitionTarget]:
        """Search web for potential customers matching criteria"""
        try:
            if not _require_openai("lead search query generation"):
                return []

            # Generate search queries using AI
            prompt = f"""Generate 10 specific Google search queries to find roofing contractors that need:
            {json.dumps(criteria)}

            Focus on finding companies showing buying signals like:
            - Outdated websites
            - Poor online reviews management
            - No online booking system
            - Manual processes mentioned
            - Growth indicators

            Return as JSON array of search queries."""

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a lead generation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            search_queries = json.loads(response.choices[0].message.content)

            # Simulate web search (would integrate with real search APIs)
            targets = []
            for query in search_queries[:5]:  # Limit for demo
                found_targets = await self._execute_search(query)
                targets.extend(found_targets)

            # Analyze and score targets
            qualified_targets = []
            for target_data in targets:
                target = await self._analyze_target(target_data)
                if target.intent_score > 0.3:
                    stored_id = await self._store_target(target)
                    if stored_id:
                        qualified_targets.append(target)

            logger.info(f"Found {len(qualified_targets)} qualified targets via web search")
            return qualified_targets

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    async def _execute_search(self, query: str) -> list[dict]:
        """Execute web search using Perplexity AI for real-time results"""
        try:
            from ai_advanced_providers import advanced_ai

            # Use Perplexity for real web search
            search_prompt = f"""Search for roofing contractors matching this query: {query}

Find companies that:
1. Are roofing contractors or related businesses
2. Show signs they need better software/automation
3. Have contact information available

Return JSON array with up to 5 results, each containing:
- company_name: string
- location: string (city, state)
- website: string (URL if found)
- contact_info: object with email/phone if available
- buying_signals: array of strings (e.g., "outdated website", "manual processes mentioned")
- estimated_size: string (small/medium/large)

Return ONLY valid JSON array, no other text."""

            result = advanced_ai.search_with_perplexity(search_prompt)

            if result and result.get("answer"):
                try:
                    # Try to parse JSON from response
                    answer = result["answer"]
                    # Find JSON array in response
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', answer)
                    if json_match:
                        leads = json.loads(json_match.group())
                        logger.info(f"Found {len(leads)} leads from Perplexity search")
                        return leads
                except json.JSONDecodeError:
                    logger.warning("Could not parse Perplexity response as JSON")

                # Fallback: use AI to extract structured data
                extraction_prompt = f"""Extract business leads from this search result:
{result['answer']}

Return JSON array with company_name, location, website, and buying_signals for each."""

                if not _require_openai("search result extraction"):
                    return []

                extraction_response = openai.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "Extract structured lead data. Return only valid JSON array."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    temperature=0.3
                )

                extracted = json.loads(extraction_response.choices[0].message.content)
                if isinstance(extracted, list):
                    logger.info(f"Extracted {len(extracted)} leads from search")
                    return extracted

            logger.warning(f"No results from search query: {query}")
            return []

        except Exception as e:
            logger.error(f"Web search execution failed: {e}")
            return []

    async def _analyze_target(self, target_data: dict) -> AcquisitionTarget:
        """Analyze target company for fit and intent"""
        try:
            if not _require_openai("target analysis"):
                return AcquisitionTarget(
                    id=str(uuid.uuid4()),
                    company_name=target_data.get('company_name', 'Unknown'),
                    industry="roofing",
                    size=target_data.get('estimated_size', 'unknown'),
                    location=target_data.get('location', ''),
                    website=target_data.get('website'),
                    social_profiles={},
                    decision_makers=[],
                    pain_points=target_data.get('buying_signals', []),
                    budget_range=(0.0, 0.0),
                    intent_score=0.0,
                    acquisition_channel=AcquisitionChannel.WEB_SEARCH,
                    metadata={"fallback": "openai_unavailable"}
                )

            # Use AI to analyze company
            prompt = f"""Analyze this company for roofing software potential:
            {json.dumps(target_data)}

            Determine:
            1. Company size (employees)
            2. Likely revenue range
            3. Technology adoption level
            4. Pain points they might have
            5. Budget capacity
            6. Intent signals (0-1 score)
            7. Best contact approach

            Return as JSON."""

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a B2B sales intelligence expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )

            analysis = json.loads(response.choices[0].message.content)

            return AcquisitionTarget(
                id=str(uuid.uuid4()),
                company_name=target_data.get('company_name', 'Unknown'),
                industry="roofing",
                size=analysis.get('company_size', 'small'),
                location=target_data.get('location', ''),
                website=target_data.get('website'),
                social_profiles=target_data.get('social', {}),
                decision_makers=[],
                pain_points=analysis.get('pain_points', []),
                budget_range=(analysis.get('budget_min', 0), analysis.get('budget_max', 10000)),
                intent_score=analysis.get('intent_signals', 0),
                acquisition_channel=AcquisitionChannel.WEB_SEARCH,
                metadata=analysis
            )

        except Exception as e:
            logger.error(f"Target analysis failed: {e}")
            return None

    async def _store_target(self, target: AcquisitionTarget) -> Optional[str]:
        """Store acquisition target in database"""
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO acquisition_targets
                (company_name, industry, company_size, location, website,
                 social_profiles, pain_points, budget_min, budget_max,
                 intent_score, acquisition_channel, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                target.company_name,
                target.industry,
                target.size,
                target.location,
                target.website,
                json.dumps(target.social_profiles),
                target.pain_points,
                target.budget_range[0],
                target.budget_range[1],
                target.intent_score,
                target.acquisition_channel.value,
                json.dumps(target.metadata)
            ))

            target_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

            return str(target_id)

        except Exception as e:
            logger.error(f"Failed to store target: {e}")
            return None

class SocialMediaAgent(CustomerAcquisitionAgent):
    """Agent that finds leads through social media"""

    def __init__(self):
        super().__init__("SocialMediaAgent")

    async def monitor_social_signals(self) -> list[dict]:
        """Monitor social media for buying signals"""
        try:
            # Social media monitoring logic
            signals = []

            # Keywords to monitor
            keywords = [
                "need roofing software",
                "looking for roofing CRM",
                "roofing business automation",
                "tired of manual estimates",
                "roofing contractor recommendations",
                "scale roofing business"
            ]

            # Simulate social monitoring (would integrate with social APIs)
            for keyword in keywords:
                found_signals = await self._search_social_platforms(keyword)
                signals.extend(found_signals)

            # Process signals into leads
            leads = []
            for signal in signals:
                lead = await self._process_social_signal(signal)
                if lead:
                    leads.append(lead)

            logger.info(f"Found {len(leads)} leads from social signals")
            return leads

        except Exception as e:
            logger.error(f"Social monitoring failed: {e}")
            return []

    async def _search_social_platforms(self, keyword: str) -> list[dict]:
        """Search social platforms for buying signals using Perplexity AI"""
        try:
            from ai_advanced_providers import advanced_ai

            # Use Perplexity to search for social media posts about roofing software needs
            social_search_prompt = f"""Search social media and forums for people/businesses posting about: {keyword}

Look for:
1. Twitter/X posts from roofing contractors
2. LinkedIn posts about roofing business challenges
3. Reddit discussions in contractor/roofing subreddits
4. Facebook business group discussions

Find posts showing buying intent for roofing software/CRM/automation.

Return JSON array with up to 5 signals, each containing:
- platform: string (twitter/linkedin/reddit/facebook)
- username: string (anonymized if needed)
- post_summary: string (key content)
- intent_level: string (high/medium/low)
- company_hint: string (company name if mentioned)
- timestamp_hint: string (recent/last_week/last_month)

Return ONLY valid JSON array, no other text."""

            result = advanced_ai.search_with_perplexity(social_search_prompt)

            if result and result.get("answer"):
                try:
                    answer = result["answer"]
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', answer)
                    if json_match:
                        signals = json.loads(json_match.group())
                        logger.info(f"Found {len(signals)} social signals for '{keyword}'")
                        return signals
                except json.JSONDecodeError:
                    logger.warning("Could not parse social search response as JSON")

                # Fallback: use AI to extract structured data
                if not _require_openai("social signal extraction"):
                    return []

                extraction_response = openai.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "Extract social media buying signals. Return only valid JSON array."},
                        {"role": "user", "content": f"Extract social signals from: {result['answer']}"}
                    ],
                    temperature=0.3
                )

                extracted = json.loads(extraction_response.choices[0].message.content)
                if isinstance(extracted, list):
                    logger.info(f"Extracted {len(extracted)} social signals")
                    return extracted

            logger.info(f"No social signals found for: {keyword}")
            return []

        except Exception as e:
            logger.error(f"Social platform search failed: {e}")
            return []

    async def _process_social_signal(self, signal: dict) -> Optional[dict]:
        """Process social signal into lead"""
        try:
            if not _require_anthropic("social signal analysis"):
                return None

            # Analyze social signal
            prompt = f"""Analyze this social media post for sales potential:
            {json.dumps(signal)}

            Determine if this is a qualified lead for roofing software.
            Extract: company name, contact info, urgency level, pain points.

            Return as JSON with qualification score 0-1."""

            response = anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )

            analysis = json.loads(response.content[0].text)

            if analysis.get('qualification_score', 0) > 0.5:
                return analysis

            return None

        except Exception as e:
            logger.error(f"Signal processing failed: {e}")
            return None

class OutreachAgent(CustomerAcquisitionAgent):
    """Agent that handles personalized outreach"""

    def __init__(self):
        super().__init__("OutreachAgent")

    async def create_outreach_sequence(self, target_id: str) -> dict:
        """Create multi-touch outreach sequence"""
        try:
            if not _require_anthropic("outreach sequence generation"):
                return {}

            # Get target data
            target = await self._get_target(target_id)
            if not target:
                return {}

            # Generate outreach sequence
            # STRATEGY: 'The Tease' / 'Iron Man Armor'
            # Do NOT sell software. Validate pain. Offer 'Superpowers'.
            prompt = f"""Create a 5-touch outreach sequence for a Roofing Consultant.
            Target: {target.get('company_name')}
            Role: {target.get('title', 'Consultant')}
            Pain Points: {json.dumps(target.get('pain_points', []))}

            STRATEGY:
            - Tone: Peer-to-peer, not salesperson-to-prospect.
            - Concept: "Iron Man Armor" for consultants.
            - Hook: "I saw you handle [Project Type]. My AI analyzed a similar roof in 4 seconds."
            - Call to Action: "Want to test the flight suit?" (i.e., see a demo/data).

            SEQUENCE STRUCTURE:
            1. The "X-Ray" Tease: Reference a specific pain point (e.g., reporting time). "My AI did this in 4 seconds."
            2. The "Data" Proof: "Here is the raw data output. No manual entry."
            3. The "Iron Man" Analogy: "We don't replace you. We are the suit. You are Tony Stark."
            4. The Case Study: "A consultant in Florida cut report time by 90%."
            5. The Break-up: "I'll keep the flight suit ready if you change your mind."

            Return as JSON with 'subject' and 'body' for each touch."""

            response = anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )

            sequence = json.loads(response.content[0].text)

            # Store sequence
            await self._store_outreach_sequence(target_id, sequence)

            # Schedule first touch
            await self._schedule_outreach(target_id, sequence['touches'][0], delay_hours=0)

            logger.info(f"Created outreach sequence for target {target_id}")
            return sequence

        except Exception as e:
            logger.error(f"Failed to create outreach sequence: {e}")
            return {}

    async def _get_target(self, target_id: str) -> Optional[dict]:
        """Get target data from database"""
        try:
            conn = _get_db_connection(cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM acquisition_targets WHERE id = %s", (target_id,))
            target = cursor.fetchone()
            cursor.close()
            conn.close()
            return target
        except Exception as e:
            logger.error(f"Failed to get target: {e}")
            return None

    async def _store_outreach_sequence(self, target_id: str, sequence: dict):
        """Store outreach sequence"""
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO acquisition_activities
                (target_id, activity_type, activity_data, agent_id)
                VALUES (%s, %s, %s, %s)
            """, (
                target_id,
                "outreach_sequence_created",
                json.dumps(sequence),
                self.agent_id
            ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store sequence: {e}")

    async def _schedule_outreach(self, target_id: str, touch: dict, delay_hours: int):
        """Schedule outreach touch"""
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            # Schedule the outreach
            scheduled_time = datetime.now(timezone.utc) + timedelta(hours=delay_hours)
            cursor.execute("""
                INSERT INTO ai_scheduled_outreach
                (id, target_id, channel, message_template, personalization,
                 scheduled_for, status, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(uuid.uuid4()),
                target_id,
                touch.get('channel', 'email'),
                touch.get('template'),
                json.dumps(touch.get('personalization', {})),
                scheduled_time,
                'scheduled',
                json.dumps(touch.get('metadata', {}))
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Outreach scheduled for {target_id} at {scheduled_time}")
        except Exception as e:
            logger.error(f"Failed to schedule outreach: {e}")

class ConversionAgent(CustomerAcquisitionAgent):
    """Agent that optimizes conversion"""

    def __init__(self):
        super().__init__("ConversionAgent")

    async def optimize_conversion_path(self, target_id: str) -> dict:
        """Optimize the conversion path for a target"""
        try:
            if not _require_openai("conversion optimization"):
                return {}

            # Analyze target's engagement
            engagement = await self._analyze_engagement(target_id)

            # Determine best conversion strategy
            prompt = f"""Based on this engagement data:
            {json.dumps(engagement)}

            Recommend the best conversion strategy:
            1. Next best action
            2. Optimal timing
            3. Channel to use
            4. Message angle
            5. Offer to present

            Return as JSON."""

            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a conversion optimization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )

            strategy = json.loads(response.choices[0].message.content)

            # Execute conversion strategy
            result = await self._execute_conversion_strategy(target_id, strategy)

            logger.info(f"Optimized conversion path for target {target_id}")
            return result

        except Exception as e:
            logger.error(f"Conversion optimization failed: {e}")
            return {}

    async def _analyze_engagement(self, target_id: str) -> dict:
        """Analyze target's engagement history"""
        try:
            conn = _get_db_connection(cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_activities,
                    AVG(engagement_score) as avg_engagement,
                    MAX(created_at) as last_activity,
                    array_agg(activity_type) as activity_types
                FROM acquisition_activities
                WHERE target_id = %s
                GROUP BY target_id
            """, (target_id,))

            engagement = cursor.fetchone() or {}
            cursor.close()
            conn.close()

            return engagement

        except Exception as e:
            logger.error(f"Failed to analyze engagement: {e}")
            return {}

    async def _execute_conversion_strategy(self, target_id: str, strategy: dict) -> dict:
        """Execute the conversion strategy"""
        # Implement conversion strategy execution
        return {"status": "executed", "strategy": strategy}

class AcquisitionOrchestrator:
    """Orchestrates all customer acquisition agents"""

    def __init__(self):
        self.web_agent = WebSearchAgent()
        self.social_agent = SocialMediaAgent()
        self.outreach_agent = OutreachAgent()
        self.conversion_agent = ConversionAgent()
        logger.info("Customer Acquisition Orchestrator initialized")

    async def run_acquisition_pipeline(self, criteria: dict):
        """Run full customer acquisition pipeline"""
        try:
            logger.info("Starting customer acquisition pipeline")

            # Phase 1: Discovery
            web_leads = await self.web_agent.search_for_leads(criteria)
            social_leads = await self.social_agent.monitor_social_signals()

            all_targets = web_leads + social_leads
            logger.info(f"Discovered {len(all_targets)} potential targets")

            # Phase 2: Outreach
            for target in all_targets[:10]:  # Limit for demo
                if hasattr(target, 'id'):
                    await self.outreach_agent.create_outreach_sequence(target.id)
                    await asyncio.sleep(1)  # Rate limiting

            # Phase 3: Conversion optimization
            # This would run continuously, monitoring engagement

            logger.info("Acquisition pipeline completed")
            return {
                "targets_found": len(all_targets),
                "outreach_started": min(10, len(all_targets)),
                "status": "running"
            }

        except Exception as e:
            logger.error(f"Acquisition pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def get_acquisition_metrics(self) -> dict:
        """Get acquisition performance metrics"""
        try:
            conn = _get_db_connection(cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_targets,
                    COUNT(*) FILTER (WHERE status = 'contacted') as contacted,
                    COUNT(*) FILTER (WHERE status = 'qualified') as qualified,
                    COUNT(*) FILTER (WHERE status = 'converted') as converted,
                    AVG(intent_score) as avg_intent_score,
                    COUNT(DISTINCT acquisition_channel) as channels_used
                FROM acquisition_targets
                WHERE created_at > NOW() - INTERVAL '30 days'
            """)

            metrics = cursor.fetchone()
            cursor.close()
            conn.close()

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}

# Global orchestrator instance - create lazily
acquisition_orchestrator = None

def get_acquisition_orchestrator():
    """Get or create acquisition orchestrator instance"""
    global acquisition_orchestrator
    if acquisition_orchestrator is None:
        acquisition_orchestrator = AcquisitionOrchestrator()
    return acquisition_orchestrator

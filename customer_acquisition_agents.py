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


def _safe_json_parse(raw: Any, default: Any) -> Any:
    """Best-effort JSON parsing for model outputs."""
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re

        obj_match = re.search(r"\{[\s\S]*\}", raw)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except Exception:
                pass
        arr_match = re.search(r"\[[\s\S]*\]", raw)
        if arr_match:
            try:
                return json.loads(arr_match.group())
            except Exception:
                pass
    return default

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
        """Verify required tables exist (DDL removed — agent_worker has no DDL permissions)."""
        required_tables = [
                "acquisition_targets",
                "acquisition_campaigns",
                "acquisition_activities",
        ]
        try:
            from database.verify_tables import verify_tables_sync
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()
            ok = verify_tables_sync(required_tables, cursor, module_name="customer_acquisition_agents")
            cursor.close()
            conn.close()
            if not ok:
                return
            self._tables_initialized = True
        except Exception as exc:
            logger.error("Table verification failed: %s", exc)
class WebSearchAgent(CustomerAcquisitionAgent):
    """Agent that searches the web for potential customers"""

    def __init__(self):
        super().__init__("WebSearchAgent")

    async def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute web-search lead discovery."""
        criteria = {
            "industry": task.get("industry", "roofing"),
            "location": task.get("location", "United States"),
            "company_size": task.get("company_size", "5-50 employees"),
        }
        leads = await self.search_for_leads(criteria)
        return {
            "status": "completed",
            "channel": AcquisitionChannel.WEB_SEARCH.value,
            "criteria": criteria,
            "leads_found": len(leads),
            "top_leads": [
                {
                    "company_name": lead.company_name,
                    "email": lead.metadata.get("email") if isinstance(lead.metadata, dict) else None,
                    "intent_score": lead.intent_score,
                    "source": lead.acquisition_channel.value,
                }
                for lead in leads[:10]
            ],
        }

    async def search_for_leads(self, criteria: dict) -> list[AcquisitionTarget]:
        """Search web for potential customers matching criteria"""
        try:
            if not _require_anthropic("lead search query generation"):
                return []

            # Generate search queries using AI (Claude)
            prompt = f"""Generate 5 specific search queries to find roofing contractors who fit the Ideal Customer Profile for 'Weathercraft ERP' (roofing software).
            
            Criteria:
            {json.dumps(criteria)}

            Focus on finding companies that:
            - Are actively growing but struggling with manual processes.
            - Have 5-50 employees (sweet spot for SaaS adoption).
            - Are located in storm-prone areas (high volume needs).

            Return as JSON array of strings (the queries)."""

            response = anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )

            import re
            json_match = re.search(r'\[[\s\S]*\]', response.content[0].text)
            if not json_match:
                logger.warning("Could not parse search queries from Claude")
                return []
                
            search_queries = json.loads(json_match.group())

            # Execute web search using DuckDuckGo
            targets = []
            for query in search_queries[:3]:  # Limit for rate safety
                found_targets = await self._execute_search(query)
                targets.extend(found_targets)

            # Analyze and score targets
            qualified_targets = []
            for target_data in targets:
                target = await self._analyze_target(target_data)
                if target and target.intent_score > 0.3:
                    stored_id = await self._store_target(target)
                    if stored_id:
                        qualified_targets.append(target)

            logger.info(f"Found {len(qualified_targets)} qualified targets via web search")
            return qualified_targets

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    async def _execute_search(self, query: str) -> list[dict]:
        """Execute web search using DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS
            
            logger.info(f"Searching DuckDuckGo for: {query}")
            results = []
            
            # Use synchronous DDGS in a way that doesn't block (simplification for this context)
            # In production, run in executor
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=5))
                
                for r in ddg_results:
                    results.append({
                        "company_name": r.get('title', '').split('-')[0].strip(), # Simple heuristic
                        "website": r.get('href'),
                        "snippet": r.get('body'),
                        "source": "duckduckgo"
                    })

            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    async def _analyze_target(self, target_data: dict) -> Optional[AcquisitionTarget]:
        """Analyze target company for fit and intent using Claude"""
        try:
            if not _require_anthropic("target analysis"):
                return None

            # Use AI to analyze company
            prompt = f"""Analyze this search result for roofing software potential:
            {json.dumps(target_data)}

            Determine:
            1. Company size (guess based on snippet)
            2. Tech adoption (does snippet mention software?)
            3. Pain points (inferred)
            4. Intent score (0.0 to 1.0)

            Return as JSON."""

            response = anthropic_client.messages.create(
                model="claude-3-haiku-20240307", # Use Haiku for speed/cost
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )

            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content[0].text)
            if not json_match:
                return None
                
            analysis = json.loads(json_match.group())

            return AcquisitionTarget(
                id=str(uuid.uuid4()),
                company_name=target_data.get('company_name', 'Unknown'),
                industry="roofing",
                size=analysis.get('company_size', 'unknown'),
                location=analysis.get('location', 'Unknown'), # AI might infer this
                website=target_data.get('website'),
                social_profiles={},
                decision_makers=[],
                pain_points=analysis.get('pain_points', []),
                budget_range=(0.0, 10000.0), # Default
                intent_score=float(analysis.get('intent_score', 0.0)),
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

    async def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute social signal monitoring and qualification."""
        leads = await self.monitor_social_signals()
        return {
            "status": "completed",
            "channel": AcquisitionChannel.SOCIAL_MEDIA.value,
            "signals_processed": len(leads),
            "qualified_leads": leads[:10],
        }

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

    async def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute outreach actions for one target or a campaign batch."""
        action = str(task.get("action") or task.get("type") or "run_campaign").lower()

        if action in {"create_sequence", "create_outreach_sequence"}:
            target_id = task.get("target_id")
            if not target_id:
                return {"status": "error", "error": "target_id required for create_sequence"}
            sequence = await self.create_outreach_sequence(str(target_id))
            if not sequence:
                return {"status": "error", "error": f"Failed to create outreach sequence for {target_id}"}
            return {
                "status": "completed",
                "target_id": str(target_id),
                "touches": len(sequence.get("touches", [])) if isinstance(sequence, dict) else 0,
            }

        if action in {"run_campaign", "campaign", "scheduled", "scheduled_run"}:
            return await self.run_acquisition_campaign(
                limit=int(task.get("limit", 10)),
                min_intent_score=float(task.get("min_intent_score", 0.55)),
            )

        return {"status": "error", "error": f"Unknown outreach action: {action}"}

    async def run_acquisition_campaign(self, limit: int = 10, min_intent_score: float = 0.55) -> dict:
        """Create outreach sequences for top intent targets not yet contacted."""
        try:
            conn = _get_db_connection(cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, company_name, intent_score, status
                FROM acquisition_targets
                WHERE COALESCE(status, 'new') IN ('new', 'qualified')
                  AND COALESCE(intent_score, 0) >= %s
                ORDER BY intent_score DESC, created_at DESC
                LIMIT %s
                """,
                (min_intent_score, max(1, limit)),
            )
            targets = cursor.fetchall() or []
            cursor.close()
            conn.close()

            contacted_ids: list[str] = []
            for target in targets:
                target_id = str(target.get("id"))
                sequence = await self.create_outreach_sequence(target_id)
                if sequence:
                    contacted_ids.append(target_id)

            if contacted_ids:
                conn = _get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE acquisition_targets
                    SET status = 'contacted',
                        updated_at = NOW()
                    WHERE id::text = ANY(%s)
                    """,
                    (contacted_ids,),
                )
                conn.commit()
                cursor.close()
                conn.close()

            return {
                "status": "completed",
                "targets_processed": len(targets),
                "outreach_started": len(contacted_ids),
                "target_ids": contacted_ids,
            }
        except Exception as e:
            logger.error(f"Acquisition campaign run failed: {e}")
            return {"status": "error", "error": str(e)}

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
            # STRATEGY: Direct Value / Product-Led Growth
            # Sell the specific solution that matches their pain.
            prompt = f"""Create a 5-touch email outreach sequence for a Roofing Company Owner.
            Target: {target.get('company_name')}
            Pain Points: {json.dumps(target.get('pain_points', []))}

            PRODUCT TO PITCH (Choose best fit based on pain points):
            1. 'Commercial Roofing Estimation Intelligence Bundle' (if pain is estimation time/accuracy)
            2. 'AI-Enhanced Project Management Accelerator' (if pain is chaos/updates)
            3. 'Intelligent Client Onboarding System' (if pain is admin/onboarding)
            4. 'Weathercraft ERP' (if they need a full system replacement)

            STRATEGY:
            - Tone: Helpful, expert, direct. Not salesy fluff.
            - Approach: "I built this to fix exactly what you're struggling with."
            - Proof: Mention specific time savings (e.g., "Cut estimation time by 5 hours/bid").
            - Call to Action: Link to the specific Gumroad product page or demo.

            SEQUENCE STRUCTURE:
            1. The "Pattern Match": "I noticed [pain point]. I built [Product] to fix exactly that."
            2. The "Value Drop": Share a free tip or insight related to the product.
            3. The "Case Study": "How [Similar Company] saved [X hours] using this tool."
            4. The "No-Brainer": Emphasize the low cost vs. high ROI (especially for the $97-$297 products).
            5. The "Break-up": "Still here if you need to fix [pain point]."

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

    async def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute conversion optimization for one target or an active funnel batch."""
        action = str(task.get("action") or task.get("type") or "optimize_funnel").lower()

        if action in {"optimize_conversion_path", "optimize_target", "optimize"}:
            target_id = task.get("target_id")
            if not target_id:
                return {"status": "error", "error": "target_id required for optimize_conversion_path"}
            return await self.optimize_conversion_path(str(target_id))

        if action in {"optimize_funnel", "scheduled", "scheduled_run"}:
            return await self.optimize_conversion_funnel(
                limit=int(task.get("limit", 25)),
                lookback_days=int(task.get("lookback_days", 30)),
            )

        return {"status": "error", "error": f"Unknown conversion action: {action}"}

    async def optimize_conversion_funnel(self, limit: int = 25, lookback_days: int = 30) -> dict:
        """Optimize conversion for recently active/engaged acquisition targets."""
        try:
            conn = _get_db_connection(cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, company_name, intent_score, status
                FROM acquisition_targets
                WHERE COALESCE(status, 'new') IN ('contacted', 'qualified', 'engaged')
                  AND created_at > NOW() - (%s || ' days')::interval
                ORDER BY intent_score DESC NULLS LAST, updated_at DESC NULLS LAST
                LIMIT %s
                """,
                (max(1, lookback_days), max(1, limit)),
            )
            targets = cursor.fetchall() or []
            cursor.close()
            conn.close()

            optimized = 0
            scheduled_actions = 0
            failures = 0
            for target in targets:
                result = await self.optimize_conversion_path(str(target.get("id")))
                if result.get("status") == "completed":
                    optimized += 1
                    if result.get("outreach_scheduled"):
                        scheduled_actions += 1
                else:
                    failures += 1

            return {
                "status": "completed",
                "targets_processed": len(targets),
                "optimized_targets": optimized,
                "outreach_scheduled": scheduled_actions,
                "failed_targets": failures,
            }
        except Exception as e:
            logger.error(f"Conversion funnel optimization failed: {e}")
            return {"status": "error", "error": str(e)}

    async def optimize_conversion_path(self, target_id: str) -> dict:
        """Optimize the conversion path for a target"""
        try:
            if not _require_openai("conversion optimization"):
                return {
                    "status": "skipped",
                    "reason": "OPENAI_API_KEY missing",
                    "target_id": target_id,
                }

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

            strategy = _safe_json_parse(response.choices[0].message.content, {})
            if not isinstance(strategy, dict):
                strategy = {"next_best_action": "follow_up", "channel": "email"}

            # Execute conversion strategy
            result = await self._execute_conversion_strategy(target_id, strategy)

            logger.info(f"Optimized conversion path for target {target_id}")
            return result

        except Exception as e:
            logger.error(f"Conversion optimization failed: {e}")
            return {"status": "error", "error": str(e), "target_id": target_id}

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
        """Persist and schedule a concrete conversion action."""
        try:
            conn = _get_db_connection(cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            channel = str(
                strategy.get("channel")
                or strategy.get("optimal_channel")
                or strategy.get("delivery_channel")
                or "email"
            ).lower()
            next_action = str(
                strategy.get("next_best_action")
                or strategy.get("message_angle")
                or strategy.get("action")
                or "follow_up"
            )
            message_angle = str(
                strategy.get("message_angle")
                or strategy.get("offer_to_present")
                or strategy.get("offer")
                or "value_reinforcement"
            )
            timing_text = str(strategy.get("optimal_timing") or strategy.get("timing") or "24h")
            timing_lower = timing_text.lower()
            if "immediate" in timing_lower or "now" in timing_lower:
                delay_hours = 1
            elif "week" in timing_lower:
                delay_hours = 24 * 7
            elif "day" in timing_lower:
                delay_hours = 24
            else:
                delay_hours = 6

            scheduled_for = datetime.now(timezone.utc) + timedelta(hours=delay_hours)

            cursor.execute(
                """
                INSERT INTO acquisition_activities
                (target_id, activity_type, activity_data, agent_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    target_id,
                    "conversion_strategy_generated",
                    json.dumps(
                        {
                            "strategy": strategy,
                            "next_action": next_action,
                            "message_angle": message_angle,
                            "scheduled_for": scheduled_for.isoformat(),
                        }
                    ),
                    self.agent_id,
                ),
            )
            strategy_activity = cursor.fetchone()
            strategy_activity_id = (
                str(strategy_activity.get("id")) if isinstance(strategy_activity, dict) else None
            )

            outreach_scheduled = False
            outreach_id = None
            if channel in {"email", "sms", "call", "linkedin"}:
                cursor.execute(
                    """
                    INSERT INTO ai_scheduled_outreach
                    (id, target_id, channel, message_template, personalization,
                     scheduled_for, status, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        str(uuid.uuid4()),
                        target_id,
                        channel,
                        "conversion_followup",
                        json.dumps(
                            {
                                "next_best_action": next_action,
                                "message_angle": message_angle,
                                "offer": strategy.get("offer_to_present") or strategy.get("offer"),
                            }
                        ),
                        scheduled_for,
                        "scheduled",
                        json.dumps(
                            {
                                "source": "conversion_optimizer",
                                "strategy": strategy,
                                "delay_hours": delay_hours,
                            }
                        ),
                    ),
                )
                outreach_row = cursor.fetchone()
                outreach_id = str(outreach_row.get("id")) if isinstance(outreach_row, dict) else None
                outreach_scheduled = True

            conversion_probability = strategy.get("conversion_probability")
            try:
                probability_value = float(conversion_probability) if conversion_probability is not None else None
            except Exception:
                probability_value = None
            if probability_value is not None and probability_value >= 0.7:
                cursor.execute(
                    """
                    UPDATE acquisition_targets
                    SET status = 'qualified',
                        updated_at = NOW()
                    WHERE id::text = %s
                    """,
                    (target_id,),
                )

            conn.commit()
            cursor.close()
            conn.close()

            return {
                "status": "completed",
                "target_id": target_id,
                "strategy_activity_id": strategy_activity_id,
                "outreach_scheduled": outreach_scheduled,
                "scheduled_outreach_id": outreach_id,
                "scheduled_for": scheduled_for.isoformat(),
                "strategy": strategy,
            }
        except Exception as e:
            logger.error(f"Failed to execute conversion strategy: {e}")
            return {"status": "error", "error": str(e), "target_id": target_id}

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

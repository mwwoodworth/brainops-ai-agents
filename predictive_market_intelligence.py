"""
Predictive Market Intelligence System
=====================================
AI-powered market intelligence for real-time data integration, trend analysis,
and autonomous content optimization.

Based on 2025 best practices from Gartner, RTInsights, and leading market research.
"""

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class MarketSignalType(Enum):
    """Types of market signals"""
    TREND = "trend"
    COMPETITOR = "competitor"
    PRICING = "pricing"
    DEMAND = "demand"
    SENTIMENT = "sentiment"
    NEWS = "news"
    REGULATORY = "regulatory"
    TECHNOLOGY = "technology"


class ContentOptimizationAction(Enum):
    """Content optimization actions"""
    SWAP_CREATIVE = "swap_creative"
    TUNE_BIDS = "tune_bids"
    REALLOCATE_BUDGET = "reallocate_budget"
    ADJUST_TARGETING = "adjust_targeting"
    CHANGE_MESSAGING = "change_messaging"
    UPDATE_SEO = "update_seo"


@dataclass
class MarketSignal:
    """A market signal from external data"""
    signal_id: str
    signal_type: MarketSignalType
    source: str
    data: dict[str, Any]
    strength: float  # 0-1
    confidence: float  # 0-1
    timestamp: str
    expires_at: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class MarketPrediction:
    """A market prediction based on signals"""
    prediction_id: str
    prediction_type: str
    target_metric: str
    predicted_value: float
    confidence_interval: tuple[float, float]
    confidence: float
    time_horizon: str  # e.g., "24h", "7d", "30d"
    contributing_signals: list[str]
    created_at: str
    expires_at: str


@dataclass
class ContentOptimization:
    """An autonomous content optimization action"""
    optimization_id: str
    action: ContentOptimizationAction
    target: str  # e.g., campaign ID, content ID
    old_value: Any
    new_value: Any
    expected_impact: dict[str, float]
    confidence: float
    executed_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None


@dataclass
class CompetitorIntelligence:
    """Intelligence about a competitor"""
    competitor_id: str
    name: str
    recent_actions: list[dict[str, Any]]
    pricing_changes: list[dict[str, Any]]
    product_launches: list[dict[str, Any]]
    market_share_estimate: float
    threat_level: str  # low, medium, high, critical
    last_updated: str



# Connection pool helper - prefer shared pool, fallback to direct connection
async def _get_db_connection(db_url: str = None):
    """Get database connection, preferring shared pool"""
    try:
        from database.async_connection import get_pool
        pool = get_pool()
        return await pool.acquire()
    except Exception as exc:
        logger.warning("Shared pool unavailable, falling back to direct connection: %s", exc, exc_info=True)
        # Fallback to direct connection if pool unavailable
        if db_url:
            import asyncpg
            return await asyncpg.connect(db_url)
        return None

class PredictiveMarketIntelligence:
    """
    Core Predictive Market Intelligence Engine

    Capabilities:
    - Real-time market signal collection
    - Trend analysis and prediction
    - Competitor intelligence
    - Autonomous content optimization
    - Dynamic pricing recommendations
    """

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.signals: dict[str, MarketSignal] = {}
        self.predictions: dict[str, MarketPrediction] = {}
        self.optimizations: list[ContentOptimization] = []
        self.competitors: dict[str, CompetitorIntelligence] = {}
        self._initialized = False

        # External data source configurations
        self.data_sources = {
            "news_api": os.getenv("NEWS_API_KEY"),
            "google_trends": os.getenv("GOOGLE_TRENDS_API"),
            "social_sentiment": os.getenv("SENTIMENT_API_KEY"),
        }

    async def initialize(self):
        """Initialize the Market Intelligence Engine"""
        if self._initialized:
            return

        logger.info("Initializing Predictive Market Intelligence Engine...")

        # Create database tables
        await self._create_tables()

        # Load existing signals and predictions
        await self._load_from_db()

        self._initialized = True
        logger.info("Market Intelligence Engine initialized")

    async def _create_tables(self):
        """Verify required tables exist (DDL removed — agent_worker has no DDL permissions)."""
        required_tables = [
                "market_signals",
                "market_predictions",
                "content_optimizations",
                "competitor_intelligence",
                "market_insights",
                "market_opportunity_scores",
        ]
        try:
            from database import get_pool
            from database.verify_tables import verify_tables_async
            pool = get_pool()
            ok = await verify_tables_async(required_tables, pool, module_name="predictive_market_intelligence")
            if not ok:
                return
            self._tables_initialized = True
        except Exception as exc:
            logger.error("Table verification failed: %s", exc)
    async def _load_from_db(self):
        """Load existing data from database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                # Load recent signals (last 24 hours)
                rows = await conn.fetch("""
                    SELECT * FROM market_signals
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    AND (expires_at IS NULL OR expires_at > NOW())
                """)

                for row in rows:
                    signal = MarketSignal(
                        signal_id=row['signal_id'],
                        signal_type=MarketSignalType(row['signal_type']),
                        source=row['source'],
                        data=row['data'],
                        strength=row['strength'],
                        confidence=row['confidence'],
                        timestamp=row['created_at'].isoformat() if row['created_at'] else "",
                        expires_at=row['expires_at'].isoformat() if row['expires_at'] else None,
                        tags=row['tags'] or []
                    )
                    self.signals[signal.signal_id] = signal

                # Load active predictions
                pred_rows = await conn.fetch("""
                    SELECT * FROM market_predictions
                    WHERE expires_at > NOW()
                """)

                for row in pred_rows:
                    pred = MarketPrediction(
                        prediction_id=row['prediction_id'],
                        prediction_type=row['prediction_type'],
                        target_metric=row['target_metric'],
                        predicted_value=row['predicted_value'],
                        confidence_interval=(row['confidence_lower'], row['confidence_upper']),
                        confidence=row['confidence'],
                        time_horizon=row['time_horizon'],
                        contributing_signals=row['contributing_signals'] or [],
                        created_at=row['created_at'].isoformat() if row['created_at'] else "",
                        expires_at=row['expires_at'].isoformat() if row['expires_at'] else ""
                    )
                    self.predictions[pred.prediction_id] = pred

            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error loading market intelligence from DB: {e}")

    async def ingest_signal(
        self,
        signal_type: MarketSignalType,
        source: str,
        data: dict[str, Any],
        strength: float = 0.5,
        confidence: float = 0.7,
        tags: list[str] = None
    ) -> MarketSignal:
        """
        Ingest a market signal from an external source

        Args:
            signal_type: Type of the signal
            source: Source of the signal (e.g., "google_trends", "news_api")
            data: Signal data payload
            strength: Signal strength (0-1)
            confidence: Confidence in the signal (0-1)
            tags: Tags for categorization

        Returns:
            Created MarketSignal
        """
        signal_id = self._generate_id(f"sig:{source}:{signal_type.value}")
        now = datetime.utcnow()

        signal = MarketSignal(
            signal_id=signal_id,
            signal_type=signal_type,
            source=source,
            data=data,
            strength=strength,
            confidence=confidence,
            timestamp=now.isoformat(),
            expires_at=(now + timedelta(hours=24)).isoformat(),
            tags=tags or []
        )

        self.signals[signal_id] = signal
        await self._persist_signal(signal)

        # Check if signal triggers any predictions
        await self._process_signal_for_predictions(signal)

        return signal

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        hash_input = f"{prefix}:{datetime.utcnow().timestamp()}"
        return f"{prefix.split(':')[0]}_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"

    async def _persist_signal(self, signal: MarketSignal):
        """Persist signal to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO market_signals
                    (signal_id, signal_type, source, data, strength, confidence, expires_at, tags)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (signal_id) DO UPDATE SET
                        data = EXCLUDED.data,
                        strength = EXCLUDED.strength,
                        confidence = EXCLUDED.confidence
                """,
                    signal.signal_id,
                    signal.signal_type.value,
                    signal.source,
                    json.dumps(signal.data),
                    signal.strength,
                    signal.confidence,
                    datetime.fromisoformat(signal.expires_at) if signal.expires_at else None,
                    json.dumps(signal.tags)
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting signal: {e}")

    async def _process_signal_for_predictions(self, signal: MarketSignal):
        """Process a new signal to update predictions"""
        # Aggregate signals by type
        signals_by_type = {}
        for sig in self.signals.values():
            if sig.signal_type not in signals_by_type:
                signals_by_type[sig.signal_type] = []
            signals_by_type[sig.signal_type].append(sig)

        # Generate predictions based on signal patterns
        if signal.signal_type == MarketSignalType.TREND:
            await self._generate_trend_prediction(signals_by_type.get(MarketSignalType.TREND, []))
        elif signal.signal_type == MarketSignalType.DEMAND:
            await self._generate_demand_prediction(signals_by_type.get(MarketSignalType.DEMAND, []))
        elif signal.signal_type == MarketSignalType.COMPETITOR:
            await self._update_competitor_intelligence(signal)

    async def _generate_trend_prediction(self, trend_signals: list[MarketSignal]):
        """Generate trend-based predictions"""
        if len(trend_signals) < 3:
            return

        # Aggregate trend data
        trend_strengths = [s.strength for s in trend_signals]
        avg_strength = sum(trend_strengths) / len(trend_strengths)

        prediction = MarketPrediction(
            prediction_id=self._generate_id("pred:trend"),
            prediction_type="market_trend",
            target_metric="demand_index",
            predicted_value=avg_strength * 100,
            confidence_interval=(avg_strength * 80, avg_strength * 120),
            confidence=sum(s.confidence for s in trend_signals) / len(trend_signals),
            time_horizon="7d",
            contributing_signals=[s.signal_id for s in trend_signals[:5]],
            created_at=datetime.utcnow().isoformat(),
            expires_at=(datetime.utcnow() + timedelta(days=7)).isoformat()
        )

        self.predictions[prediction.prediction_id] = prediction
        await self._persist_prediction(prediction)

    async def _generate_demand_prediction(self, demand_signals: list[MarketSignal]):
        """Generate demand predictions"""
        if len(demand_signals) < 2:
            return

        # Analyze demand patterns
        demand_values = [s.data.get("demand_index", s.strength * 100) for s in demand_signals]
        avg_demand = sum(demand_values) / len(demand_values)

        # Simple trend calculation
        if len(demand_values) >= 2:
            trend = (demand_values[-1] - demand_values[0]) / len(demand_values)
            predicted = avg_demand + (trend * 7)  # 7-day projection
        else:
            predicted = avg_demand

        prediction = MarketPrediction(
            prediction_id=self._generate_id("pred:demand"),
            prediction_type="demand_forecast",
            target_metric="demand_level",
            predicted_value=predicted,
            confidence_interval=(predicted * 0.85, predicted * 1.15),
            confidence=0.7,
            time_horizon="7d",
            contributing_signals=[s.signal_id for s in demand_signals[:5]],
            created_at=datetime.utcnow().isoformat(),
            expires_at=(datetime.utcnow() + timedelta(days=7)).isoformat()
        )

        self.predictions[prediction.prediction_id] = prediction
        await self._persist_prediction(prediction)

    async def _persist_prediction(self, prediction: MarketPrediction):
        """Persist prediction to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO market_predictions
                    (prediction_id, prediction_type, target_metric, predicted_value,
                     confidence_lower, confidence_upper, confidence, time_horizon,
                     contributing_signals, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (prediction_id) DO UPDATE SET
                        predicted_value = EXCLUDED.predicted_value,
                        confidence = EXCLUDED.confidence
                """,
                    prediction.prediction_id,
                    prediction.prediction_type,
                    prediction.target_metric,
                    prediction.predicted_value,
                    prediction.confidence_interval[0],
                    prediction.confidence_interval[1],
                    prediction.confidence,
                    prediction.time_horizon,
                    json.dumps(prediction.contributing_signals),
                    datetime.fromisoformat(prediction.expires_at) if prediction.expires_at else None
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting prediction: {e}")

    async def _update_competitor_intelligence(self, signal: MarketSignal):
        """Update competitor intelligence from a signal"""
        competitor_name = signal.data.get("competitor_name")
        if not competitor_name:
            return

        competitor_id = hashlib.sha256(competitor_name.encode()).hexdigest()[:16]

        if competitor_id not in self.competitors:
            self.competitors[competitor_id] = CompetitorIntelligence(
                competitor_id=competitor_id,
                name=competitor_name,
                recent_actions=[],
                pricing_changes=[],
                product_launches=[],
                market_share_estimate=signal.data.get("market_share", 0),
                threat_level="medium",
                last_updated=datetime.utcnow().isoformat()
            )

        competitor = self.competitors[competitor_id]

        # Add the new action
        action_data = {
            "timestamp": signal.timestamp,
            "action_type": signal.data.get("action_type", "unknown"),
            "details": signal.data.get("details", {})
        }
        competitor.recent_actions.append(action_data)

        # Keep only last 50 actions
        if len(competitor.recent_actions) > 50:
            competitor.recent_actions = competitor.recent_actions[-50:]

        # Update threat level based on activity
        if len(competitor.recent_actions) > 10:
            competitor.threat_level = "high"
        elif len(competitor.recent_actions) > 5:
            competitor.threat_level = "medium"

        competitor.last_updated = datetime.utcnow().isoformat()

        await self._persist_competitor(competitor)

    async def _persist_competitor(self, competitor: CompetitorIntelligence):
        """Persist competitor intelligence to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO competitor_intelligence
                    (competitor_id, name, recent_actions, pricing_changes, product_launches,
                     market_share_estimate, threat_level, last_updated)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (competitor_id) DO UPDATE SET
                        recent_actions = EXCLUDED.recent_actions,
                        pricing_changes = EXCLUDED.pricing_changes,
                        product_launches = EXCLUDED.product_launches,
                        market_share_estimate = EXCLUDED.market_share_estimate,
                        threat_level = EXCLUDED.threat_level,
                        last_updated = EXCLUDED.last_updated
                """,
                    competitor.competitor_id,
                    competitor.name,
                    json.dumps(competitor.recent_actions),
                    json.dumps(competitor.pricing_changes),
                    json.dumps(competitor.product_launches),
                    competitor.market_share_estimate,
                    competitor.threat_level,
                    datetime.fromisoformat(competitor.last_updated)
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting competitor: {e}")

    async def generate_content_optimization(
        self,
        target: str,
        current_performance: dict[str, float],
        context: dict[str, Any] = None
    ) -> ContentOptimization:
        """
        Generate an autonomous content optimization recommendation

        Args:
            target: Target content/campaign ID
            current_performance: Current performance metrics
            context: Additional context for optimization

        Returns:
            ContentOptimization recommendation
        """
        # Analyze performance gaps
        click_rate = current_performance.get("click_rate", 0)
        conversion_rate = current_performance.get("conversion_rate", 0)
        cost_per_action = current_performance.get("cost_per_action", 0)

        # Determine best optimization action
        if click_rate < 0.02:  # <2% CTR
            action = ContentOptimizationAction.SWAP_CREATIVE
            expected_impact = {"click_rate": 0.3, "conversions": 0.15}
            new_value = {"creative_variant": "high_engagement_v2"}
        elif conversion_rate < 0.01:  # <1% CVR
            action = ContentOptimizationAction.ADJUST_TARGETING
            expected_impact = {"conversion_rate": 0.25, "cost_per_action": -0.1}
            new_value = {"targeting": "refined_high_intent"}
        elif cost_per_action > 50:  # High CPA
            action = ContentOptimizationAction.TUNE_BIDS
            expected_impact = {"cost_per_action": -0.2, "volume": -0.05}
            new_value = {"bid_adjustment": -15}
        else:
            action = ContentOptimizationAction.REALLOCATE_BUDGET
            expected_impact = {"roi": 0.1}
            new_value = {"budget_shift": "top_performers"}

        optimization = ContentOptimization(
            optimization_id=self._generate_id("opt"),
            action=action,
            target=target,
            old_value=current_performance,
            new_value=new_value,
            expected_impact=expected_impact,
            confidence=0.75
        )

        self.optimizations.append(optimization)
        await self._persist_optimization(optimization)

        return optimization

    async def _persist_optimization(self, optimization: ContentOptimization):
        """Persist optimization to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO content_optimizations
                    (optimization_id, action, target, old_value, new_value,
                     expected_impact, confidence, executed_at, result)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    optimization.optimization_id,
                    optimization.action.value,
                    optimization.target,
                    json.dumps(optimization.old_value),
                    json.dumps(optimization.new_value),
                    json.dumps(optimization.expected_impact),
                    optimization.confidence,
                    datetime.fromisoformat(optimization.executed_at) if optimization.executed_at else None,
                    json.dumps(optimization.result) if optimization.result else None
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting optimization: {e}")

    async def execute_optimization(self, optimization_id: str) -> dict[str, Any]:
        """Execute a content optimization"""
        for opt in self.optimizations:
            if opt.optimization_id == optimization_id:
                opt.executed_at = datetime.utcnow().isoformat()

                # Simulate execution result
                opt.result = {
                    "status": "executed",
                    "changes_applied": opt.new_value,
                    "execution_time_ms": 150
                }

                await self._persist_optimization(opt)

                return {
                    "optimization_id": optimization_id,
                    "status": "executed",
                    "action": opt.action.value,
                    "target": opt.target,
                    "expected_impact": opt.expected_impact
                }

        return {"error": f"Optimization {optimization_id} not found"}

    async def get_market_insights(self, category: str = None) -> dict[str, Any]:
        """Get current market insights and predictions"""
        await self.initialize()

        # Aggregate insights
        insights = {
            "generated_at": datetime.utcnow().isoformat(),
            "signals_count": len(self.signals),
            "active_predictions": len(self.predictions),
            "competitors_tracked": len(self.competitors),
            "pending_optimizations": len([o for o in self.optimizations if not o.executed_at])
        }

        # Top signals
        recent_signals = sorted(
            self.signals.values(),
            key=lambda s: s.strength * s.confidence,
            reverse=True
        )[:10]

        insights["top_signals"] = [
            {
                "type": s.signal_type.value,
                "source": s.source,
                "strength": s.strength,
                "confidence": s.confidence,
                "summary": s.data.get("summary", str(s.data)[:100])
            }
            for s in recent_signals
        ]

        # Active predictions
        insights["predictions"] = [
            {
                "type": p.prediction_type,
                "metric": p.target_metric,
                "predicted_value": p.predicted_value,
                "confidence": p.confidence,
                "horizon": p.time_horizon
            }
            for p in self.predictions.values()
        ]

        # Competitor overview
        high_threat_competitors = [
            c for c in self.competitors.values()
            if c.threat_level in ["high", "critical"]
        ]

        insights["competitor_alerts"] = [
            {
                "name": c.name,
                "threat_level": c.threat_level,
                "recent_actions_count": len(c.recent_actions),
                "market_share": c.market_share_estimate
            }
            for c in high_threat_competitors
        ]

        return insights

    async def fetch_external_signals(self) -> list[MarketSignal]:
        """Fetch signals from configured external data sources"""
        signals = []

        # Fetch from each configured source
        if self.data_sources.get("news_api"):
            news_signals = await self._fetch_news_signals()
            signals.extend(news_signals)

        # Add more source integrations here

        return signals

    async def _fetch_news_signals(self) -> list[MarketSignal]:
        """Fetch news-based market signals"""
        api_key = self.data_sources.get("news_api")
        if not api_key:
            raise RuntimeError("NEWS_API_KEY not configured")

        query = os.getenv("NEWS_API_QUERY", "").strip()
        if not query:
            raise RuntimeError("NEWS_API_QUERY not configured")

        base_url = os.getenv("NEWS_API_BASE_URL", "https://newsapi.org/v2/everything")
        try:
            page_size = int(os.getenv("NEWS_API_PAGE_SIZE", "20"))
        except ValueError:
            page_size = 20

        timeout_seconds = float(os.getenv("NEWS_API_TIMEOUT", "10"))
        params = {
            "q": query,
            "apiKey": api_key,
            "pageSize": min(max(page_size, 1), 100),
            "sortBy": "publishedAt",
            "language": "en",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.get(base_url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("News API request failed: %s", exc, exc_info=True)
            raise RuntimeError("News API request failed") from exc

        payload = response.json()
        if payload.get("status") != "ok":
            message = payload.get("message", "Unknown error")
            raise RuntimeError(f"News API response error: {message}")

        articles = payload.get("articles") or []
        signals: list[MarketSignal] = []
        now = datetime.utcnow().isoformat()

        for article in articles:
            title = article.get("title") or ""
            description = article.get("description") or ""
            content = article.get("content") or description
            url = article.get("url") or ""
            published_at = article.get("publishedAt") or now

            if not title and not content:
                continue

            signal_seed = f"{title}|{url}|{published_at}"
            signal_id = hashlib.sha256(signal_seed.encode("utf-8")).hexdigest()[:16]

            signals.append(MarketSignal(
                signal_id=signal_id,
                signal_type=MarketSignalType.NEWS,
                source="news_api",
                data={
                    "title": title,
                    "description": description,
                    "content": content,
                    "url": url,
                    "author": article.get("author"),
                    "source_name": (article.get("source") or {}).get("name"),
                    "published_at": published_at,
                    "query": query,
                },
                strength=0.5,
                confidence=0.6,
                timestamp=published_at,
                tags=[query],
            ))

        if not signals:
            logger.warning("News API returned no articles for query=%s", query)

        return signals

    async def generate_seo_recommendations(
        self,
        current_keywords: list[str],
        industry: str
    ) -> dict[str, Any]:
        """Generate SEO recommendations based on market trends"""
        # Analyze trend signals for keywords
        trend_signals = [
            s for s in self.signals.values()
            if s.signal_type == MarketSignalType.TREND
        ]

        # Extract trending keywords
        trending_keywords = []
        for sig in trend_signals:
            if "keywords" in sig.data:
                trending_keywords.extend(sig.data["keywords"])

        return {
            "current_keywords": current_keywords,
            "recommended_additions": list(set(trending_keywords) - set(current_keywords))[:10],
            "recommended_removals": [],  # Would need performance data
            "content_suggestions": [
                f"Create content about {kw}" for kw in trending_keywords[:5]
            ],
            "industry": industry
        }

    async def score_opportunity(
        self,
        opportunity_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Score a market opportunity using multi-factor analysis

        Scoring Factors:
        - Market size and growth (30%)
        - Competitive intensity (20%)
        - Strategic fit (20%)
        - Resource requirements (15%)
        - Time to market (10%)
        - Risk level (5%)

        Args:
            opportunity_data: Dictionary containing opportunity metrics

        Returns:
            Comprehensive opportunity score and recommendation
        """
        try:
            # Extract opportunity metrics
            market_size = opportunity_data.get("market_size", 0)
            market_growth = opportunity_data.get("market_growth_rate", 0)
            competition_level = opportunity_data.get("competition_level", "high")  # low/medium/high
            strategic_fit = opportunity_data.get("strategic_fit", 5)  # 1-10 scale
            resource_requirements = opportunity_data.get("resource_requirements", "high")  # low/medium/high
            time_to_market = opportunity_data.get("time_to_market_months", 12)
            risk_level = opportunity_data.get("risk_level", "medium")  # low/medium/high

            # Calculate component scores (0-100 scale)
            scores = {}

            # 1. Market size & growth score (30%)
            market_score = 0
            if market_size > 10_000_000:  # $10M+
                market_score += 50
            elif market_size > 1_000_000:  # $1M+
                market_score += 30
            elif market_size > 100_000:  # $100K+
                market_score += 10

            if market_growth > 0.20:  # 20%+ growth
                market_score += 50
            elif market_growth > 0.10:  # 10%+ growth
                market_score += 30
            elif market_growth > 0:  # Positive growth
                market_score += 10

            scores["market_attractiveness"] = min(100, market_score)

            # 2. Competition score (20%) - inverse scoring
            competition_map = {"low": 90, "medium": 60, "high": 30}
            scores["competitive_advantage"] = competition_map.get(competition_level.lower(), 50)

            # 3. Strategic fit score (20%)
            scores["strategic_fit"] = strategic_fit * 10  # Convert 1-10 to 0-100

            # 4. Resource requirements score (15%) - inverse scoring
            resource_map = {"low": 90, "medium": 60, "high": 30}
            scores["resource_feasibility"] = resource_map.get(resource_requirements.lower(), 50)

            # 5. Time to market score (10%) - faster is better
            if time_to_market <= 3:
                time_score = 90
            elif time_to_market <= 6:
                time_score = 70
            elif time_to_market <= 12:
                time_score = 50
            else:
                time_score = 30
            scores["time_to_market"] = time_score

            # 6. Risk score (5%) - inverse scoring
            risk_map = {"low": 90, "medium": 60, "high": 30}
            scores["risk_tolerance"] = risk_map.get(risk_level.lower(), 50)

            # Calculate weighted overall score
            weights = {
                "market_attractiveness": 0.30,
                "competitive_advantage": 0.20,
                "strategic_fit": 0.20,
                "resource_feasibility": 0.15,
                "time_to_market": 0.10,
                "risk_tolerance": 0.05
            }

            overall_score = sum(scores[k] * weights[k] for k in scores.keys())

            # Determine recommendation
            if overall_score >= 80:
                recommendation = "pursue_immediately"
                priority = "high"
            elif overall_score >= 60:
                recommendation = "pursue_with_planning"
                priority = "medium"
            elif overall_score >= 40:
                recommendation = "explore_further"
                priority = "low"
            else:
                recommendation = "deprioritize"
                priority = "very_low"

            # Identify strengths and weaknesses
            strengths = [k for k, v in scores.items() if v >= 70]
            weaknesses = [k for k, v in scores.items() if v <= 40]

            # Calculate expected ROI estimate
            if overall_score >= 70:
                roi_estimate = "high"
                roi_range = "150-300%"
            elif overall_score >= 50:
                roi_estimate = "medium"
                roi_range = "50-150%"
            else:
                roi_estimate = "low"
                roi_range = "0-50%"

            result = {
                "overall_score": round(overall_score, 2),
                "recommendation": recommendation,
                "priority": priority,
                "component_scores": scores,
                "weights_applied": weights,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "expected_roi": {
                    "estimate": roi_estimate,
                    "range": roi_range
                },
                "next_steps": self._generate_opportunity_next_steps(
                    recommendation, weaknesses, opportunity_data
                ),
                "scored_at": datetime.utcnow().isoformat()
            }

            # Persist the opportunity score
            await self._persist_opportunity_score(opportunity_data, result)

            return result

        except Exception as e:
            logger.error(f"Opportunity scoring failed: {e}")
            return {"error": str(e)}

    def _generate_opportunity_next_steps(
        self,
        recommendation: str,
        weaknesses: list[str],
        opportunity_data: dict[str, Any]
    ) -> list[str]:
        """Generate actionable next steps based on opportunity analysis"""
        next_steps = []

        if recommendation == "pursue_immediately":
            next_steps = [
                "Allocate resources and budget",
                "Develop detailed execution plan",
                "Set up success metrics and KPIs",
                "Begin implementation within 30 days"
            ]
        elif recommendation == "pursue_with_planning":
            next_steps = [
                "Conduct detailed market research",
                "Develop business case and ROI model",
                "Identify and mitigate key risks",
                "Create phased rollout plan"
            ]

            # Address weaknesses
            if "competitive_advantage" in weaknesses:
                next_steps.append("Develop differentiation strategy")
            if "resource_feasibility" in weaknesses:
                next_steps.append("Secure additional resources or partnerships")
            if "time_to_market" in weaknesses:
                next_steps.append("Explore ways to accelerate timeline")

        elif recommendation == "explore_further":
            next_steps = [
                "Conduct pilot or proof of concept",
                "Validate market assumptions",
                "Assess resource availability",
                "Re-evaluate in 3-6 months"
            ]
        else:  # deprioritize
            next_steps = [
                "Monitor market conditions",
                "Reassess if fundamentals change",
                "Focus resources on higher-priority opportunities"
            ]

        return next_steps

    async def _persist_opportunity_score(
        self,
        opportunity_data: dict[str, Any],
        score_result: dict[str, Any]
    ):
        """Persist opportunity score to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:

                await conn.execute("""
                    INSERT INTO market_opportunity_scores
                    (opportunity_name, opportunity_data, overall_score, recommendation,
                     priority, component_scores, strengths, weaknesses, next_steps)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    opportunity_data.get("name", "Unnamed Opportunity"),
                    json.dumps(opportunity_data),
                    score_result["overall_score"],
                    score_result["recommendation"],
                    score_result["priority"],
                    json.dumps(score_result["component_scores"]),
                    json.dumps(score_result["strengths"]),
                    json.dumps(score_result["weaknesses"]),
                    json.dumps(score_result["next_steps"])
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting opportunity score: {e}")


# Singleton instance
market_intelligence = PredictiveMarketIntelligence()


# API Functions
async def ingest_market_signal(
    signal_type: str,
    source: str,
    data: dict[str, Any],
    strength: float = 0.5,
    confidence: float = 0.7
) -> dict[str, Any]:
    """Ingest a new market signal"""
    await market_intelligence.initialize()
    signal = await market_intelligence.ingest_signal(
        signal_type=MarketSignalType(signal_type),
        source=source,
        data=data,
        strength=strength,
        confidence=confidence
    )
    return asdict(signal)


async def get_market_insights(category: str = None) -> dict[str, Any]:
    """Get current market insights"""
    return await market_intelligence.get_market_insights(category)


async def generate_optimization(
    target: str,
    performance: dict[str, float]
) -> dict[str, Any]:
    """Generate content optimization recommendation"""
    await market_intelligence.initialize()
    opt = await market_intelligence.generate_content_optimization(target, performance)
    return {
        "optimization_id": opt.optimization_id,
        "action": opt.action.value,
        "target": opt.target,
        "expected_impact": opt.expected_impact,
        "confidence": opt.confidence
    }


async def execute_market_optimization(optimization_id: str) -> dict[str, Any]:
    """Execute a pending optimization"""
    return await market_intelligence.execute_optimization(optimization_id)


async def get_competitor_intelligence() -> list[dict[str, Any]]:
    """Get competitor intelligence overview"""
    await market_intelligence.initialize()
    return [
        {
            "competitor_id": c.competitor_id,
            "name": c.name,
            "threat_level": c.threat_level,
            "market_share": c.market_share_estimate,
            "recent_actions_count": len(c.recent_actions),
            "last_updated": c.last_updated
        }
        for c in market_intelligence.competitors.values()
    ]


async def score_market_opportunity(
    opportunity_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Score a market opportunity based on multiple factors

    Args:
        opportunity_data: Data about the opportunity including market size,
                         competition, trends, etc.

    Returns:
        Opportunity score and detailed analysis
    """
    await market_intelligence.initialize()
    return await market_intelligence.score_opportunity(opportunity_data)

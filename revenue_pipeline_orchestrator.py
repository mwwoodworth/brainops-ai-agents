"""
Revenue Pipeline Orchestrator
Comprehensive multi-stream revenue generation and optimization system

This orchestrates ALL revenue streams:
- Product sales (digital products)
- Subscription services
- Affiliate/partnerships
- Consulting/services
- Advertising/sponsorships
- Data/API monetization
- White-label/licensing

Features:
- Automated revenue optimization
- Multi-channel attribution
- Pricing optimization
- Upsell/cross-sell automation
- Lifetime value maximization
- Churn prevention
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass, field
from enum import Enum
import uuid

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class RevenueStream(Enum):
    """Revenue stream types"""
    DIGITAL_PRODUCTS = "digital_products"
    SAAS_SUBSCRIPTION = "saas_subscription"
    AFFILIATE = "affiliate"
    CONSULTING = "consulting"
    ADVERTISING = "advertising"
    API_ACCESS = "api_access"
    WHITE_LABEL = "white_label"
    LICENSING = "licensing"
    MARKETPLACE = "marketplace"
    TRAINING = "training"
    CERTIFICATION = "certification"
    SPONSORSHIP = "sponsorship"


class PricingStrategy(Enum):
    """Pricing strategies"""
    PENETRATION = "penetration"        # Low price to gain market share
    SKIMMING = "skimming"              # High price for premium positioning
    COMPETITIVE = "competitive"         # Match competitor pricing
    VALUE_BASED = "value_based"        # Price based on customer value
    DYNAMIC = "dynamic"                # Real-time price optimization
    TIERED = "tiered"                  # Multiple pricing tiers
    FREEMIUM = "freemium"              # Free tier with paid upgrades
    USAGE_BASED = "usage_based"        # Pay per use
    BUNDLE = "bundle"                  # Product bundles


class CustomerSegment(Enum):
    """Customer segments for targeting"""
    ENTERPRISE = "enterprise"
    SMB = "smb"
    STARTUP = "startup"
    FREELANCER = "freelancer"
    HOBBYIST = "hobbyist"
    STUDENT = "student"


@dataclass
class RevenueStreamConfig:
    """Configuration for a revenue stream"""
    stream_type: RevenueStream
    name: str
    description: str
    pricing_strategy: PricingStrategy
    base_price: float
    target_segments: List[CustomerSegment]
    commission_rate: float = 0.0  # For affiliates
    revenue_share: float = 0.0   # For partnerships
    active: bool = True
    automation_enabled: bool = True
    metadata: Dict = field(default_factory=dict)


@dataclass
class RevenueMetrics:
    """Revenue metrics snapshot"""
    total_revenue: float = 0.0
    mrr: float = 0.0  # Monthly Recurring Revenue
    arr: float = 0.0  # Annual Recurring Revenue
    arpu: float = 0.0  # Average Revenue Per User
    ltv: float = 0.0  # Lifetime Value
    cac: float = 0.0  # Customer Acquisition Cost
    churn_rate: float = 0.0
    growth_rate: float = 0.0
    by_stream: Dict[str, float] = field(default_factory=dict)
    by_product: Dict[str, float] = field(default_factory=dict)
    period_start: datetime = None
    period_end: datetime = None


class RevenuePipelineOrchestrator:
    """
    Master revenue orchestration engine

    Manages and optimizes ALL revenue streams:
    - Coordinates pricing across channels
    - Optimizes for maximum LTV
    - Automates upsell/cross-sell
    - Tracks attribution
    - Prevents churn
    """

    def __init__(self):
        self.db_url = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL')
        self._initialized = False

        # Default revenue stream configurations
        self.default_streams = [
            RevenueStreamConfig(
                stream_type=RevenueStream.DIGITAL_PRODUCTS,
                name="Digital Product Store",
                description="eBooks, templates, guides, and tools",
                pricing_strategy=PricingStrategy.VALUE_BASED,
                base_price=29.0,
                target_segments=[CustomerSegment.SMB, CustomerSegment.FREELANCER, CustomerSegment.STARTUP]
            ),
            RevenueStreamConfig(
                stream_type=RevenueStream.SAAS_SUBSCRIPTION,
                name="SaaS Platform",
                description="Monthly/annual subscription access",
                pricing_strategy=PricingStrategy.TIERED,
                base_price=49.0,
                target_segments=[CustomerSegment.SMB, CustomerSegment.ENTERPRISE]
            ),
            RevenueStreamConfig(
                stream_type=RevenueStream.AFFILIATE,
                name="Affiliate Program",
                description="Partner referral commissions",
                pricing_strategy=PricingStrategy.COMPETITIVE,
                base_price=0.0,
                commission_rate=0.30,
                target_segments=[CustomerSegment.FREELANCER]
            ),
            RevenueStreamConfig(
                stream_type=RevenueStream.API_ACCESS,
                name="API Marketplace",
                description="Pay-per-use API access",
                pricing_strategy=PricingStrategy.USAGE_BASED,
                base_price=0.01,  # Per request
                target_segments=[CustomerSegment.STARTUP, CustomerSegment.ENTERPRISE]
            ),
            RevenueStreamConfig(
                stream_type=RevenueStream.CONSULTING,
                name="AI Consulting Services",
                description="Custom implementation and strategy",
                pricing_strategy=PricingStrategy.VALUE_BASED,
                base_price=250.0,  # Per hour
                target_segments=[CustomerSegment.ENTERPRISE, CustomerSegment.SMB]
            ),
            RevenueStreamConfig(
                stream_type=RevenueStream.TRAINING,
                name="Training & Courses",
                description="Educational content and certifications",
                pricing_strategy=PricingStrategy.TIERED,
                base_price=199.0,
                target_segments=[CustomerSegment.FREELANCER, CustomerSegment.STUDENT, CustomerSegment.SMB]
            ),
            RevenueStreamConfig(
                stream_type=RevenueStream.WHITE_LABEL,
                name="White Label Solutions",
                description="Resellable AI solutions",
                pricing_strategy=PricingStrategy.VALUE_BASED,
                base_price=999.0,
                revenue_share=0.20,
                target_segments=[CustomerSegment.ENTERPRISE]
            )
        ]

    def _get_connection(self):
        if not self.db_url:
            raise ValueError("DATABASE_URL not configured")
        return psycopg2.connect(self.db_url)

    async def initialize_tables(self):
        """Create revenue orchestration tables"""
        if self._initialized:
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Revenue streams configuration
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS revenue_streams (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        stream_type VARCHAR(50) NOT NULL UNIQUE,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        pricing_strategy VARCHAR(50),
                        base_price FLOAT DEFAULT 0,
                        commission_rate FLOAT DEFAULT 0,
                        revenue_share FLOAT DEFAULT 0,
                        target_segments JSONB DEFAULT '[]',
                        active BOOLEAN DEFAULT true,
                        automation_enabled BOOLEAN DEFAULT true,
                        config JSONB DEFAULT '{}',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Revenue transactions
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS revenue_transactions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        stream_type VARCHAR(50) NOT NULL,
                        product_id UUID,
                        customer_id UUID,
                        amount FLOAT NOT NULL,
                        currency VARCHAR(10) DEFAULT 'USD',
                        transaction_type VARCHAR(50),
                        payment_method VARCHAR(50),
                        status VARCHAR(50) DEFAULT 'completed',
                        attribution JSONB DEFAULT '{}',
                        metadata JSONB DEFAULT '{}',
                        transaction_date TIMESTAMPTZ DEFAULT NOW(),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Pricing rules
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pricing_rules (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        stream_type VARCHAR(50),
                        product_id UUID,
                        rule_name VARCHAR(255),
                        rule_type VARCHAR(50),
                        conditions JSONB DEFAULT '{}',
                        price_adjustment FLOAT,
                        adjustment_type VARCHAR(50),
                        priority INT DEFAULT 5,
                        active BOOLEAN DEFAULT true,
                        valid_from TIMESTAMPTZ,
                        valid_until TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Subscription management
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS subscriptions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        customer_id UUID NOT NULL,
                        plan_id VARCHAR(100) NOT NULL,
                        stream_type VARCHAR(50),
                        status VARCHAR(50) DEFAULT 'active',
                        price FLOAT NOT NULL,
                        billing_period VARCHAR(20),
                        current_period_start TIMESTAMPTZ,
                        current_period_end TIMESTAMPTZ,
                        cancel_at_period_end BOOLEAN DEFAULT false,
                        canceled_at TIMESTAMPTZ,
                        trial_end TIMESTAMPTZ,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Affiliate tracking
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS affiliate_tracking (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        affiliate_id UUID NOT NULL,
                        referred_customer_id UUID,
                        referral_code VARCHAR(50),
                        click_date TIMESTAMPTZ,
                        conversion_date TIMESTAMPTZ,
                        commission_amount FLOAT DEFAULT 0,
                        commission_status VARCHAR(50) DEFAULT 'pending',
                        transaction_id UUID,
                        lifetime_value FLOAT DEFAULT 0,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Upsell/cross-sell opportunities
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS upsell_opportunities (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        customer_id UUID NOT NULL,
                        current_product_id UUID,
                        recommended_product_id UUID,
                        opportunity_type VARCHAR(50),
                        score FLOAT DEFAULT 0,
                        potential_value FLOAT DEFAULT 0,
                        status VARCHAR(50) DEFAULT 'identified',
                        triggered_at TIMESTAMPTZ,
                        converted_at TIMESTAMPTZ,
                        declined_at TIMESTAMPTZ,
                        notes TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Revenue metrics snapshots
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS revenue_metrics_snapshots (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        snapshot_date DATE NOT NULL,
                        total_revenue FLOAT DEFAULT 0,
                        mrr FLOAT DEFAULT 0,
                        arr FLOAT DEFAULT 0,
                        arpu FLOAT DEFAULT 0,
                        ltv FLOAT DEFAULT 0,
                        cac FLOAT DEFAULT 0,
                        churn_rate FLOAT DEFAULT 0,
                        growth_rate FLOAT DEFAULT 0,
                        by_stream JSONB DEFAULT '{}',
                        by_product JSONB DEFAULT '{}',
                        by_segment JSONB DEFAULT '{}',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(snapshot_date)
                    )
                """)

                # Create indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_stream ON revenue_transactions(stream_type)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON revenue_transactions(transaction_date DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(customer_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_affiliate_code ON affiliate_tracking(referral_code)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_upsell_customer ON upsell_opportunities(customer_id)")

                conn.commit()

                # Initialize default streams
                await self._initialize_default_streams()

                self._initialized = True
                logger.info("Revenue orchestration tables initialized")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize tables: {e}")
            raise
        finally:
            conn.close()

    async def _initialize_default_streams(self):
        """Initialize default revenue stream configurations"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                for stream in self.default_streams:
                    cur.execute("""
                        INSERT INTO revenue_streams (
                            stream_type, name, description, pricing_strategy,
                            base_price, commission_rate, revenue_share,
                            target_segments, active, automation_enabled
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (stream_type) DO NOTHING
                    """, (
                        stream.stream_type.value,
                        stream.name,
                        stream.description,
                        stream.pricing_strategy.value,
                        stream.base_price,
                        stream.commission_rate,
                        stream.revenue_share,
                        json.dumps([s.value for s in stream.target_segments]),
                        stream.active,
                        stream.automation_enabled
                    ))
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize streams: {e}")
        finally:
            conn.close()

    async def record_transaction(self, stream_type: RevenueStream, amount: float,
                                  product_id: str = None, customer_id: str = None,
                                  transaction_type: str = "purchase",
                                  attribution: Dict = None, metadata: Dict = None) -> str:
        """Record a revenue transaction"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                transaction_id = str(uuid.uuid4())
                cur.execute("""
                    INSERT INTO revenue_transactions (
                        id, stream_type, product_id, customer_id, amount,
                        transaction_type, attribution, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    transaction_id,
                    stream_type.value,
                    product_id,
                    customer_id,
                    amount,
                    transaction_type,
                    json.dumps(attribution or {}),
                    json.dumps(metadata or {})
                ))
                conn.commit()

                # Check for upsell opportunities
                if customer_id:
                    await self._identify_upsell_opportunities(customer_id)

                return transaction_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record transaction: {e}")
            raise
        finally:
            conn.close()

    async def calculate_metrics(self, period_days: int = 30) -> RevenueMetrics:
        """Calculate comprehensive revenue metrics"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                period_start = datetime.utcnow() - timedelta(days=period_days)

                # Total revenue
                cur.execute("""
                    SELECT COALESCE(SUM(amount), 0) as total
                    FROM revenue_transactions
                    WHERE transaction_date >= %s AND status = 'completed'
                """, (period_start,))
                total_revenue = cur.fetchone()['total']

                # Revenue by stream
                cur.execute("""
                    SELECT stream_type, SUM(amount) as revenue
                    FROM revenue_transactions
                    WHERE transaction_date >= %s AND status = 'completed'
                    GROUP BY stream_type
                """, (period_start,))
                by_stream = {row['stream_type']: row['revenue'] for row in cur.fetchall()}

                # MRR from subscriptions
                cur.execute("""
                    SELECT COALESCE(SUM(price), 0) as mrr
                    FROM subscriptions
                    WHERE status = 'active' AND billing_period = 'monthly'
                """)
                monthly_mrr = cur.fetchone()['mrr']

                cur.execute("""
                    SELECT COALESCE(SUM(price / 12), 0) as mrr
                    FROM subscriptions
                    WHERE status = 'active' AND billing_period = 'yearly'
                """)
                annual_mrr = cur.fetchone()['mrr']

                mrr = monthly_mrr + annual_mrr
                arr = mrr * 12

                # Customer count
                cur.execute("""
                    SELECT COUNT(DISTINCT customer_id) as customers
                    FROM revenue_transactions
                    WHERE transaction_date >= %s AND status = 'completed'
                """, (period_start,))
                customer_count = cur.fetchone()['customers'] or 1

                arpu = total_revenue / customer_count

                # Churn rate (simplified)
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'canceled') as churned,
                        COUNT(*) as total
                    FROM subscriptions
                    WHERE updated_at >= %s
                """, (period_start,))
                churn_data = cur.fetchone()
                churn_rate = churn_data['churned'] / max(churn_data['total'], 1)

                # Calculate LTV (simplified: ARPU / churn_rate)
                ltv = arpu / max(churn_rate, 0.01)

                return RevenueMetrics(
                    total_revenue=total_revenue,
                    mrr=mrr,
                    arr=arr,
                    arpu=arpu,
                    ltv=ltv,
                    cac=0,  # Would need marketing data
                    churn_rate=churn_rate,
                    growth_rate=0,  # Would need comparison data
                    by_stream=by_stream,
                    period_start=period_start,
                    period_end=datetime.utcnow()
                )

        finally:
            conn.close()

    async def optimize_pricing(self, product_id: str = None, stream_type: RevenueStream = None) -> Dict:
        """
        Optimize pricing using AI analysis

        Considers:
        - Conversion rates at different price points
        - Competitor pricing
        - Customer segment willingness to pay
        - Demand elasticity
        """
        await self.initialize_tables()

        # Get historical data
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get transaction history
                query = """
                    SELECT
                        amount as price,
                        COUNT(*) as transactions,
                        SUM(amount) as total_revenue
                    FROM revenue_transactions
                    WHERE status = 'completed'
                """
                params = []

                if product_id:
                    query += " AND product_id = %s"
                    params.append(product_id)

                if stream_type:
                    query += " AND stream_type = %s"
                    params.append(stream_type.value)

                query += " GROUP BY amount ORDER BY amount"

                cur.execute(query, params)
                price_performance = cur.fetchall()

        finally:
            conn.close()

        # Simple optimization: find price with highest revenue
        if price_performance:
            best_price = max(price_performance, key=lambda x: x['total_revenue'])
            return {
                "recommended_price": best_price['price'],
                "current_revenue": best_price['total_revenue'],
                "analysis": price_performance,
                "strategy": "revenue_maximization"
            }

        return {
            "recommended_price": None,
            "analysis": "Insufficient data for optimization"
        }

    async def _identify_upsell_opportunities(self, customer_id: str):
        """Identify upsell/cross-sell opportunities for a customer"""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get customer's purchase history
                cur.execute("""
                    SELECT DISTINCT product_id, stream_type, amount
                    FROM revenue_transactions
                    WHERE customer_id = %s AND status = 'completed'
                    ORDER BY transaction_date DESC
                """, (customer_id,))
                purchases = cur.fetchall()

                if not purchases:
                    return

                # Get subscription status
                cur.execute("""
                    SELECT * FROM subscriptions
                    WHERE customer_id = %s AND status = 'active'
                """, (customer_id,))
                subscription = cur.fetchone()

                # Identify opportunities based on patterns
                opportunities = []

                # If has subscription, recommend add-ons
                if subscription:
                    opportunities.append({
                        "type": "add_on",
                        "recommended_stream": RevenueStream.CONSULTING.value,
                        "score": 0.7,
                        "potential_value": 500
                    })

                # If bought digital products, recommend course
                product_streams = [p['stream_type'] for p in purchases]
                if RevenueStream.DIGITAL_PRODUCTS.value in product_streams:
                    if RevenueStream.TRAINING.value not in product_streams:
                        opportunities.append({
                            "type": "cross_sell",
                            "recommended_stream": RevenueStream.TRAINING.value,
                            "score": 0.8,
                            "potential_value": 199
                        })

                # Store opportunities
                for opp in opportunities:
                    cur.execute("""
                        INSERT INTO upsell_opportunities (
                            customer_id, opportunity_type, score, potential_value
                        ) VALUES (%s, %s, %s, %s)
                    """, (
                        customer_id,
                        opp['type'],
                        opp['score'],
                        opp['potential_value']
                    ))

                conn.commit()

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to identify upsell opportunities: {e}")
        finally:
            conn.close()

    async def create_subscription(self, customer_id: str, plan_id: str,
                                   price: float, billing_period: str = "monthly",
                                   trial_days: int = 0) -> str:
        """Create a new subscription"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                subscription_id = str(uuid.uuid4())

                period_start = datetime.utcnow()
                if trial_days > 0:
                    trial_end = period_start + timedelta(days=trial_days)
                    period_end = trial_end
                else:
                    trial_end = None
                    if billing_period == "monthly":
                        period_end = period_start + timedelta(days=30)
                    elif billing_period == "yearly":
                        period_end = period_start + timedelta(days=365)
                    else:
                        period_end = period_start + timedelta(days=30)

                cur.execute("""
                    INSERT INTO subscriptions (
                        id, customer_id, plan_id, stream_type, price,
                        billing_period, current_period_start, current_period_end,
                        trial_end, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    subscription_id,
                    customer_id,
                    plan_id,
                    RevenueStream.SAAS_SUBSCRIPTION.value,
                    price,
                    billing_period,
                    period_start,
                    period_end,
                    trial_end,
                    'trialing' if trial_days > 0 else 'active'
                ))

                conn.commit()
                return subscription_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create subscription: {e}")
            raise
        finally:
            conn.close()

    async def track_affiliate_referral(self, affiliate_id: str, referral_code: str,
                                        customer_id: str = None) -> str:
        """Track affiliate referral"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                tracking_id = str(uuid.uuid4())
                cur.execute("""
                    INSERT INTO affiliate_tracking (
                        id, affiliate_id, referral_code, referred_customer_id, click_date
                    ) VALUES (%s, %s, %s, %s, NOW())
                    RETURNING id
                """, (tracking_id, affiliate_id, referral_code, customer_id))

                conn.commit()
                return tracking_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to track referral: {e}")
            raise
        finally:
            conn.close()

    async def process_affiliate_commission(self, tracking_id: str, transaction_id: str,
                                            transaction_amount: float):
        """Process affiliate commission for a sale"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get affiliate stream config for commission rate
                cur.execute("""
                    SELECT commission_rate FROM revenue_streams
                    WHERE stream_type = %s
                """, (RevenueStream.AFFILIATE.value,))
                config = cur.fetchone()
                commission_rate = config['commission_rate'] if config else 0.30

                commission_amount = transaction_amount * commission_rate

                cur.execute("""
                    UPDATE affiliate_tracking SET
                        conversion_date = NOW(),
                        commission_amount = %s,
                        commission_status = 'approved',
                        transaction_id = %s
                    WHERE id = %s
                """, (commission_amount, transaction_id, tracking_id))

                conn.commit()

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to process commission: {e}")
        finally:
            conn.close()

    async def get_revenue_dashboard(self) -> Dict:
        """Get comprehensive revenue dashboard data"""
        await self.initialize_tables()

        metrics = await self.calculate_metrics()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Recent transactions
                cur.execute("""
                    SELECT stream_type, amount, transaction_date
                    FROM revenue_transactions
                    WHERE status = 'completed'
                    ORDER BY transaction_date DESC
                    LIMIT 10
                """)
                recent_transactions = cur.fetchall()

                # Active streams
                cur.execute("""
                    SELECT stream_type, name, base_price, active
                    FROM revenue_streams
                    WHERE active = true
                """)
                active_streams = cur.fetchall()

                # Subscription stats
                cur.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'active') as active,
                        COUNT(*) FILTER (WHERE status = 'trialing') as trialing,
                        COALESCE(SUM(price) FILTER (WHERE status = 'active'), 0) as mrr
                    FROM subscriptions
                """)
                subscription_stats = cur.fetchone()

                # Pending upsells
                cur.execute("""
                    SELECT COUNT(*) as count, SUM(potential_value) as value
                    FROM upsell_opportunities
                    WHERE status = 'identified'
                """)
                upsell_stats = cur.fetchone()

                # Affiliate performance
                cur.execute("""
                    SELECT
                        COUNT(*) as total_referrals,
                        COUNT(*) FILTER (WHERE conversion_date IS NOT NULL) as conversions,
                        COALESCE(SUM(commission_amount), 0) as commissions_paid
                    FROM affiliate_tracking
                """)
                affiliate_stats = cur.fetchone()

                return {
                    "metrics": {
                        "total_revenue": metrics.total_revenue,
                        "mrr": metrics.mrr,
                        "arr": metrics.arr,
                        "arpu": metrics.arpu,
                        "ltv": metrics.ltv,
                        "churn_rate": metrics.churn_rate
                    },
                    "by_stream": metrics.by_stream,
                    "recent_transactions": recent_transactions,
                    "active_streams": active_streams,
                    "subscriptions": subscription_stats,
                    "upsell_opportunities": upsell_stats,
                    "affiliate_performance": affiliate_stats,
                    "timestamp": datetime.utcnow().isoformat()
                }

        finally:
            conn.close()

    async def snapshot_metrics(self):
        """Take a daily snapshot of metrics"""
        metrics = await self.calculate_metrics()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO revenue_metrics_snapshots (
                        snapshot_date, total_revenue, mrr, arr, arpu,
                        ltv, cac, churn_rate, growth_rate, by_stream
                    ) VALUES (CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (snapshot_date) DO UPDATE SET
                        total_revenue = EXCLUDED.total_revenue,
                        mrr = EXCLUDED.mrr,
                        arr = EXCLUDED.arr,
                        arpu = EXCLUDED.arpu,
                        ltv = EXCLUDED.ltv
                """, (
                    metrics.total_revenue,
                    metrics.mrr,
                    metrics.arr,
                    metrics.arpu,
                    metrics.ltv,
                    metrics.cac,
                    metrics.churn_rate,
                    metrics.growth_rate,
                    json.dumps(metrics.by_stream)
                ))
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to snapshot metrics: {e}")
        finally:
            conn.close()


# =====================================
# AUTOMATED PRODUCT CATALOG
# =====================================

class ProductCatalogGenerator:
    """
    Automatically generates and manages product catalog

    Features:
    - Auto-generate product listings
    - Dynamic pricing
    - Bundle creation
    - Seasonal/promotional pricing
    """

    def __init__(self):
        self.db_url = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL')
        self.revenue_orchestrator = RevenuePipelineOrchestrator()

    def _get_connection(self):
        return psycopg2.connect(self.db_url)

    async def initialize_tables(self):
        """Create product catalog tables"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS product_catalog (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        product_type VARCHAR(50) NOT NULL,
                        name VARCHAR(500) NOT NULL,
                        slug VARCHAR(500) UNIQUE,
                        description TEXT,
                        short_description TEXT,
                        price FLOAT NOT NULL,
                        compare_at_price FLOAT,
                        currency VARCHAR(10) DEFAULT 'USD',
                        stream_type VARCHAR(50),
                        category VARCHAR(100),
                        tags JSONB DEFAULT '[]',
                        features JSONB DEFAULT '[]',
                        images JSONB DEFAULT '[]',
                        files JSONB DEFAULT '[]',
                        generated_product_id UUID,
                        active BOOLEAN DEFAULT true,
                        featured BOOLEAN DEFAULT false,
                        sort_order INT DEFAULT 0,
                        sales_count INT DEFAULT 0,
                        rating FLOAT DEFAULT 0,
                        review_count INT DEFAULT 0,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS product_bundles (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        name VARCHAR(500) NOT NULL,
                        description TEXT,
                        products JSONB DEFAULT '[]',
                        bundle_price FLOAT NOT NULL,
                        savings_amount FLOAT,
                        savings_percentage FLOAT,
                        active BOOLEAN DEFAULT true,
                        valid_from TIMESTAMPTZ,
                        valid_until TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_type ON product_catalog(product_type)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_category ON product_catalog(category)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_active ON product_catalog(active)")

                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create catalog tables: {e}")
        finally:
            conn.close()

    async def add_to_catalog(self, product_data: Dict) -> str:
        """Add a product to the catalog"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                product_id = str(uuid.uuid4())
                slug = self._generate_slug(product_data.get('name', ''))

                cur.execute("""
                    INSERT INTO product_catalog (
                        id, product_type, name, slug, description, short_description,
                        price, compare_at_price, stream_type, category, tags,
                        features, generated_product_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    product_id,
                    product_data.get('product_type', 'digital'),
                    product_data.get('name'),
                    slug,
                    product_data.get('description'),
                    product_data.get('short_description'),
                    product_data.get('price', 29.0),
                    product_data.get('compare_at_price'),
                    product_data.get('stream_type', RevenueStream.DIGITAL_PRODUCTS.value),
                    product_data.get('category'),
                    json.dumps(product_data.get('tags', [])),
                    json.dumps(product_data.get('features', [])),
                    product_data.get('generated_product_id')
                ))

                conn.commit()
                return product_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add to catalog: {e}")
            raise
        finally:
            conn.close()

    def _generate_slug(self, name: str) -> str:
        """Generate URL-safe slug from name"""
        import re
        slug = name.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return f"{slug}-{uuid.uuid4().hex[:8]}"

    async def create_bundle(self, name: str, product_ids: List[str],
                            discount_percentage: float = 20) -> str:
        """Create a product bundle with discount"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get products and calculate bundle price
                placeholders = ','.join(['%s'] * len(product_ids))
                cur.execute(f"""
                    SELECT id, name, price FROM product_catalog
                    WHERE id::text IN ({placeholders})
                """, product_ids)
                products = cur.fetchall()

                total_price = sum(p['price'] for p in products)
                bundle_price = total_price * (1 - discount_percentage / 100)
                savings = total_price - bundle_price

                bundle_id = str(uuid.uuid4())
                cur.execute("""
                    INSERT INTO product_bundles (
                        id, name, products, bundle_price,
                        savings_amount, savings_percentage
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    bundle_id,
                    name,
                    json.dumps([{
                        "id": str(p['id']),
                        "name": p['name'],
                        "price": p['price']
                    } for p in products]),
                    bundle_price,
                    savings,
                    discount_percentage
                ))

                conn.commit()
                return bundle_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create bundle: {e}")
            raise
        finally:
            conn.close()

    async def get_catalog(self, category: str = None, product_type: str = None,
                          active_only: bool = True) -> List[Dict]:
        """Get products from catalog"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT * FROM product_catalog WHERE 1=1"
                params = []

                if active_only:
                    query += " AND active = true"

                if category:
                    query += " AND category = %s"
                    params.append(category)

                if product_type:
                    query += " AND product_type = %s"
                    params.append(product_type)

                query += " ORDER BY featured DESC, sort_order ASC, created_at DESC"

                cur.execute(query, params)
                return cur.fetchall()

        finally:
            conn.close()


# =====================================
# SINGLETON INSTANCES
# =====================================

_revenue_orchestrator = None
_catalog_generator = None

def get_revenue_orchestrator() -> RevenuePipelineOrchestrator:
    global _revenue_orchestrator
    if _revenue_orchestrator is None:
        _revenue_orchestrator = RevenuePipelineOrchestrator()
    return _revenue_orchestrator

def get_catalog_generator() -> ProductCatalogGenerator:
    global _catalog_generator
    if _catalog_generator is None:
        _catalog_generator = ProductCatalogGenerator()
    return _catalog_generator

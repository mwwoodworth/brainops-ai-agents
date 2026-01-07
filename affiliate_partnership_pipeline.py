"""
AFFILIATE & PARTNERSHIP PIPELINE - BRAINOPS AI OS
==================================================
Multi-tier affiliate program, partnership management, and commission automation.

Features:
- Multi-tier affiliate tracking (up to 3 levels)
- Partner program management (reseller, white-label, referral)
- Commission calculation and payout automation
- Performance analytics and attribution
- Affiliate content generation (AI-powered)
- Partner onboarding automation
- Fraud detection and prevention

Author: BrainOps AI OS
Version: 1.0.0
"""

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import stripe
from loguru import logger

from ai_core import ai_generate
from email_sender import send_email

# =============================================================================
# ENUMERATIONS
# =============================================================================

class PartnerType(Enum):
    """Types of partners in the ecosystem."""
    AFFILIATE = "affiliate"           # Commission-based referrals
    RESELLER = "reseller"             # Buys at wholesale, sells at retail
    WHITE_LABEL = "white_label"       # Full rebrand partner
    REFERRAL = "referral"             # Simple referral (one-time bonus)
    INFLUENCER = "influencer"         # Social media/content creator
    AGENCY = "agency"                 # Agency partner (client management)
    TECHNOLOGY = "technology"         # Integration/tech partners
    STRATEGIC = "strategic"           # Strategic alliance partners


class PartnerTier(Enum):
    """Partner performance tiers."""
    BRONZE = "bronze"       # Entry level
    SILVER = "silver"       # $1K-$5K monthly
    GOLD = "gold"           # $5K-$25K monthly
    PLATINUM = "platinum"   # $25K-$100K monthly
    DIAMOND = "diamond"     # $100K+ monthly


class CommissionType(Enum):
    """Types of commission structures."""
    PERCENTAGE = "percentage"         # % of sale
    FLAT_RATE = "flat_rate"          # Fixed amount per sale
    TIERED = "tiered"                # Increases with volume
    RECURRING = "recurring"          # Ongoing % of subscription
    HYBRID = "hybrid"                # Base + percentage
    PERFORMANCE_BONUS = "performance_bonus"  # Bonuses for targets


class PayoutStatus(Enum):
    """Payout processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"


class AffiliateStatus(Enum):
    """Affiliate account status."""
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    BANNED = "banned"
    INACTIVE = "inactive"


class AttributionModel(Enum):
    """Attribution models for tracking."""
    LAST_CLICK = "last_click"
    FIRST_CLICK = "first_click"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CommissionStructure:
    """Defines commission rates and rules."""
    structure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    commission_type: CommissionType = CommissionType.PERCENTAGE
    base_rate: Decimal = Decimal("0.20")  # 20% default
    tier_rates: dict[str, Decimal] = field(default_factory=dict)  # tier -> rate
    recurring_months: int = 12  # How long recurring commissions last
    minimum_payout: Decimal = Decimal("50.00")
    payout_frequency: str = "monthly"  # weekly, bi-weekly, monthly
    cookie_duration_days: int = 90  # Attribution window
    products: list[str] = field(default_factory=list)  # Specific products or "all"

    def get_rate_for_tier(self, tier: PartnerTier) -> Decimal:
        """Get commission rate for a specific tier."""
        tier_rate = self.tier_rates.get(tier.value)
        if tier_rate:
            return tier_rate
        return self.base_rate


@dataclass
class Affiliate:
    """Affiliate/partner profile."""
    affiliate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    partner_type: PartnerType = PartnerType.AFFILIATE
    tier: PartnerTier = PartnerTier.BRONZE
    status: AffiliateStatus = AffiliateStatus.PENDING_APPROVAL

    # Contact Info
    company_name: str = ""
    contact_name: str = ""
    email: str = ""
    phone: str = ""
    website: str = ""

    # Tracking
    affiliate_code: str = field(default_factory=lambda: secrets.token_urlsafe(8))
    tracking_links: dict[str, str] = field(default_factory=dict)  # campaign -> link
    referral_source: str = ""

    # Commission Settings
    commission_structure_id: str = ""
    custom_commission_rate: Optional[Decimal] = None
    payout_method: str = "stripe"  # stripe, paypal, wire, check
    payout_details: dict[str, Any] = field(default_factory=dict)

    # Multi-tier tracking
    parent_affiliate_id: Optional[str] = None  # Who referred this affiliate
    tier_level: int = 1  # 1 = direct, 2 = second tier, 3 = third tier

    # Performance
    total_referrals: int = 0
    total_conversions: int = 0
    total_revenue_generated: Decimal = Decimal("0")
    total_commissions_earned: Decimal = Decimal("0")
    total_commissions_paid: Decimal = Decimal("0")
    pending_commission: Decimal = Decimal("0")

    # Dates
    joined_date: datetime = field(default_factory=datetime.utcnow)
    last_activity_date: Optional[datetime] = None
    next_payout_date: Optional[datetime] = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    custom_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Referral:
    """Individual referral/conversion record."""
    referral_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    affiliate_id: str = ""

    # Visitor info
    visitor_id: str = ""  # Anonymous ID before conversion
    customer_id: Optional[str] = None  # After conversion
    ip_address: str = ""
    user_agent: str = ""

    # Tracking
    tracking_code: str = ""
    landing_page: str = ""
    referrer_url: str = ""
    utm_source: str = ""
    utm_medium: str = ""
    utm_campaign: str = ""
    utm_content: str = ""

    # Conversion
    converted: bool = False
    conversion_date: Optional[datetime] = None
    product_id: Optional[str] = None
    product_name: str = ""
    order_id: Optional[str] = None
    order_value: Decimal = Decimal("0")

    # Commission
    commission_amount: Decimal = Decimal("0")
    commission_status: str = "pending"  # pending, approved, paid, refunded

    # Multi-tier
    tier_level: int = 1
    parent_referral_id: Optional[str] = None  # For tier 2/3 tracking

    # Attribution
    attribution_model: AttributionModel = AttributionModel.LAST_CLICK
    attribution_weight: float = 1.0
    touchpoints: list[dict[str, Any]] = field(default_factory=list)

    # Dates
    click_date: datetime = field(default_factory=datetime.utcnow)
    cookie_expires: Optional[datetime] = None


@dataclass
class Commission:
    """Commission record for tracking payouts."""
    commission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    affiliate_id: str = ""
    referral_id: str = ""

    # Amounts
    gross_amount: Decimal = Decimal("0")
    adjustments: Decimal = Decimal("0")  # Refunds, chargebacks
    net_amount: Decimal = Decimal("0")

    # Details
    commission_type: CommissionType = CommissionType.PERCENTAGE
    rate_applied: Decimal = Decimal("0")
    tier_level: int = 1
    tier_rate_modifier: float = 1.0  # 1.0 for tier 1, 0.5 for tier 2, etc.

    # Status
    status: PayoutStatus = PayoutStatus.PENDING
    payout_id: Optional[str] = None

    # Dates
    earned_date: datetime = field(default_factory=datetime.utcnow)
    available_date: Optional[datetime] = None  # After hold period
    paid_date: Optional[datetime] = None

    # Metadata
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Payout:
    """Payout batch for affiliate payments."""
    payout_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    affiliate_id: str = ""

    # Amounts
    gross_amount: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")  # Processing fees
    net_amount: Decimal = Decimal("0")

    # Commissions included
    commission_ids: list[str] = field(default_factory=list)
    commission_count: int = 0

    # Payment
    payment_method: str = ""  # stripe, paypal, wire
    payment_reference: str = ""  # External transaction ID
    currency: str = "USD"

    # Status
    status: PayoutStatus = PayoutStatus.PENDING
    failure_reason: Optional[str] = None
    retry_count: int = 0

    # Dates
    created_date: datetime = field(default_factory=datetime.utcnow)
    processed_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None


@dataclass
class PayoutQueueItem:
    """Payout queue item for batch processing."""
    queue_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payout_id: str = ""
    affiliate_id: str = ""

    # Queue status
    status: str = "queued"  # queued, processing, completed, failed, cancelled
    priority: int = 0  # Higher = higher priority

    # Amounts
    amount: Decimal = Decimal("0")
    currency: str = "USD"
    payment_method: str = ""

    # Processing
    attempts: int = 0
    max_attempts: int = 3
    last_attempt_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    error_message: str = ""

    # Timing
    queued_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PartnerContent:
    """AI-generated content for affiliates."""
    content_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    affiliate_id: str = ""

    # Content details
    content_type: str = ""  # email, social_post, blog, banner, video_script
    title: str = ""
    content: str = ""

    # Assets
    images: list[str] = field(default_factory=list)
    tracking_link: str = ""

    # Performance
    times_used: int = 0
    clicks_generated: int = 0
    conversions_generated: int = 0

    # Metadata
    created_date: datetime = field(default_factory=datetime.utcnow)
    ai_model_used: str = ""
    generation_prompt: str = ""


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_COMMISSION_STRUCTURES: dict[PartnerType, CommissionStructure] = {
    PartnerType.AFFILIATE: CommissionStructure(
        name="Standard Affiliate",
        commission_type=CommissionType.RECURRING,
        base_rate=Decimal("0.25"),  # 25% recurring
        tier_rates={
            "bronze": Decimal("0.20"),
            "silver": Decimal("0.25"),
            "gold": Decimal("0.30"),
            "platinum": Decimal("0.35"),
            "diamond": Decimal("0.40"),
        },
        recurring_months=24,
        cookie_duration_days=90,
    ),
    PartnerType.RESELLER: CommissionStructure(
        name="Reseller Program",
        commission_type=CommissionType.PERCENTAGE,
        base_rate=Decimal("0.40"),  # 40% margin
        tier_rates={
            "bronze": Decimal("0.35"),
            "silver": Decimal("0.40"),
            "gold": Decimal("0.45"),
            "platinum": Decimal("0.50"),
        },
        cookie_duration_days=365,
    ),
    PartnerType.INFLUENCER: CommissionStructure(
        name="Influencer Program",
        commission_type=CommissionType.HYBRID,
        base_rate=Decimal("0.30"),  # 30% + flat bonus
        tier_rates={
            "bronze": Decimal("0.25"),
            "silver": Decimal("0.30"),
            "gold": Decimal("0.35"),
            "platinum": Decimal("0.40"),
        },
        recurring_months=12,
        cookie_duration_days=30,
    ),
    PartnerType.AGENCY: CommissionStructure(
        name="Agency Partner",
        commission_type=CommissionType.RECURRING,
        base_rate=Decimal("0.20"),
        tier_rates={
            "bronze": Decimal("0.15"),
            "silver": Decimal("0.20"),
            "gold": Decimal("0.25"),
            "platinum": Decimal("0.30"),
            "diamond": Decimal("0.35"),
        },
        recurring_months=36,
        cookie_duration_days=180,
    ),
}

MULTI_TIER_RATES = {
    1: Decimal("1.0"),    # Tier 1: 100% of commission
    2: Decimal("0.10"),   # Tier 2: 10% of tier 1's commission
    3: Decimal("0.05"),   # Tier 3: 5% of tier 1's commission
}

TIER_THRESHOLDS = {
    PartnerTier.BRONZE: Decimal("0"),
    PartnerTier.SILVER: Decimal("1000"),      # $1K/month
    PartnerTier.GOLD: Decimal("5000"),        # $5K/month
    PartnerTier.PLATINUM: Decimal("25000"),   # $25K/month
    PartnerTier.DIAMOND: Decimal("100000"),   # $100K/month
}

# Volume-based tiered commission rates (cumulative monthly sales thresholds)
VOLUME_TIERED_COMMISSION_RATES = {
    # (min_volume, max_volume): rate
    (Decimal("0"), Decimal("1000")): Decimal("0.15"),         # $0-$1K: 15%
    (Decimal("1000"), Decimal("5000")): Decimal("0.20"),      # $1K-$5K: 20%
    (Decimal("5000"), Decimal("15000")): Decimal("0.25"),     # $5K-$15K: 25%
    (Decimal("15000"), Decimal("50000")): Decimal("0.30"),    # $15K-$50K: 30%
    (Decimal("50000"), Decimal("999999999")): Decimal("0.35"), # $50K+: 35%
}


# =============================================================================
# FRAUD DETECTION
# =============================================================================

@dataclass
class FraudSignal:
    """Fraud detection signal."""
    signal_type: str
    severity: str  # low, medium, high, critical
    score: float
    details: str


class FraudDetector:
    """Detects fraudulent affiliate activity."""

    # Known fraud patterns
    SUSPICIOUS_IP_RANGES: set[str] = {
        "10.0.0.",
        "192.168.",
        "172.16.",
    }

    # Thresholds
    MAX_CLICKS_PER_IP_HOUR = 10
    MAX_CONVERSIONS_PER_IP_DAY = 3
    MIN_TIME_ON_SITE_SECONDS = 5
    MAX_REFUND_RATE = 0.15  # 15%

    def __init__(self):
        self.ip_click_counts: dict[str, list[datetime]] = {}
        self.ip_conversion_counts: dict[str, list[datetime]] = {}

    async def analyze_click(
        self,
        referral: Referral,
        affiliate: Affiliate
    ) -> list[FraudSignal]:
        """Analyze a click for fraud signals."""
        signals = []

        # Check for VPN/proxy IPs
        if any(referral.ip_address.startswith(r) for r in self.SUSPICIOUS_IP_RANGES):
            signals.append(FraudSignal(
                signal_type="suspicious_ip",
                severity="medium",
                score=0.5,
                details=f"IP {referral.ip_address} appears to be from private range"
            ))

        # Check click velocity
        ip = referral.ip_address
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)

        if ip not in self.ip_click_counts:
            self.ip_click_counts[ip] = []

        # Clean old entries
        self.ip_click_counts[ip] = [
            t for t in self.ip_click_counts[ip] if t > hour_ago
        ]
        self.ip_click_counts[ip].append(now)

        if len(self.ip_click_counts[ip]) > self.MAX_CLICKS_PER_IP_HOUR:
            signals.append(FraudSignal(
                signal_type="click_velocity",
                severity="high",
                score=0.8,
                details=f"IP {ip} exceeded {self.MAX_CLICKS_PER_IP_HOUR} clicks/hour"
            ))

        # Check for self-referral
        if affiliate.website and affiliate.website in (referral.landing_page or ""):
            signals.append(FraudSignal(
                signal_type="self_referral",
                severity="critical",
                score=1.0,
                details="Potential self-referral detected"
            ))

        # Check user agent for bots
        ua_lower = referral.user_agent.lower()
        bot_indicators = ["bot", "crawler", "spider", "scraper", "curl", "wget"]
        if any(indicator in ua_lower for indicator in bot_indicators):
            signals.append(FraudSignal(
                signal_type="bot_traffic",
                severity="high",
                score=0.9,
                details=f"Bot-like user agent detected: {referral.user_agent[:50]}"
            ))

        return signals

    async def analyze_conversion(
        self,
        referral: Referral,
        affiliate: Affiliate
    ) -> list[FraudSignal]:
        """Analyze a conversion for fraud signals."""
        signals = []

        # Check conversion velocity
        ip = referral.ip_address
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)

        if ip not in self.ip_conversion_counts:
            self.ip_conversion_counts[ip] = []

        self.ip_conversion_counts[ip] = [
            t for t in self.ip_conversion_counts[ip] if t > day_ago
        ]
        self.ip_conversion_counts[ip].append(now)

        if len(self.ip_conversion_counts[ip]) > self.MAX_CONVERSIONS_PER_IP_DAY:
            signals.append(FraudSignal(
                signal_type="conversion_velocity",
                severity="critical",
                score=0.95,
                details=f"IP {ip} exceeded {self.MAX_CONVERSIONS_PER_IP_DAY} conversions/day"
            ))

        # Check for rapid conversion (clicked and converted too fast)
        if referral.conversion_date and referral.click_date:
            time_diff = (referral.conversion_date - referral.click_date).total_seconds()
            if time_diff < self.MIN_TIME_ON_SITE_SECONDS:
                signals.append(FraudSignal(
                    signal_type="rapid_conversion",
                    severity="high",
                    score=0.7,
                    details=f"Converted {time_diff:.1f}s after click (min: {self.MIN_TIME_ON_SITE_SECONDS}s)"
                ))

        # Check affiliate's refund rate
        if affiliate.total_conversions > 10:  # Need enough data
            refund_rate = None
            custom_refund_rate = affiliate.custom_data.get("refund_rate")
            if isinstance(custom_refund_rate, (int, float)):
                refund_rate = float(custom_refund_rate)
            else:
                total_refunds = affiliate.custom_data.get("total_refunds")
                if isinstance(total_refunds, (int, float)) and affiliate.total_conversions:
                    refund_rate = float(total_refunds) / float(affiliate.total_conversions)

            if refund_rate is not None and refund_rate > self.MAX_REFUND_RATE:
                signals.append(FraudSignal(
                    signal_type="high_refund_rate",
                    severity="medium",
                    score=0.6,
                    details=f"Refund rate {refund_rate:.2%} exceeds {self.MAX_REFUND_RATE:.0%}"
                ))

        return signals

    def calculate_fraud_score(self, signals: list[FraudSignal]) -> float:
        """Calculate overall fraud score from signals."""
        if not signals:
            return 0.0

        # Weighted average with severity multipliers
        severity_weights = {
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "critical": 1.0,
        }

        total_weight = 0.0
        weighted_score = 0.0

        for signal in signals:
            weight = severity_weights.get(signal.severity, 0.5)
            total_weight += weight
            weighted_score += signal.score * weight

        return weighted_score / total_weight if total_weight > 0 else 0.0


# =============================================================================
# AFFILIATE CONTENT GENERATOR
# =============================================================================

class AffiliateContentGenerator:
    """AI-powered content generation for affiliates."""

    def __init__(self, openai_api_key: str = None, anthropic_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    async def generate_email_template(
        self,
        affiliate: Affiliate,
        product_name: str,
        product_description: str,
        target_audience: str,
        tone: str = "professional"
    ) -> PartnerContent:
        """Generate email marketing template for affiliate."""
        prompt = f"""Create a compelling email marketing template for an affiliate promoting:

Product: {product_name}
Description: {product_description}
Target Audience: {target_audience}
Tone: {tone}
Affiliate: {affiliate.company_name or affiliate.contact_name}

Requirements:
1. Engaging subject line options (3 variants)
2. Personal opening
3. Problem-agitation-solution structure
4. Clear value proposition
5. Social proof placeholder
6. Strong call-to-action with tracking link placeholder {{TRACKING_LINK}}
7. P.S. line for urgency

Format as ready-to-send email with clear sections."""

        content = await self._call_claude(prompt)

        return PartnerContent(
            affiliate_id=affiliate.affiliate_id,
            content_type="email",
            title=f"Email Template: {product_name}",
            content=content,
            tracking_link="{TRACKING_LINK}",
            ai_model_used="claude-3-opus",
            generation_prompt=prompt[:500],
        )

    async def generate_social_posts(
        self,
        affiliate: Affiliate,
        product_name: str,
        key_benefits: list[str],
        platforms: list[str] = None
    ) -> list[PartnerContent]:
        """Generate social media posts for multiple platforms."""
        platforms = platforms or ["twitter", "linkedin", "facebook", "instagram"]
        contents = []

        for platform in platforms:
            prompt = f"""Create an engaging {platform} post for promoting:

Product: {product_name}
Key Benefits: {', '.join(key_benefits)}

Requirements for {platform}:
- Twitter: Max 280 chars, punchy, use hashtags
- LinkedIn: Professional tone, 300-500 chars, no excessive hashtags
- Facebook: Conversational, 100-300 chars, emotional hook
- Instagram: Visual-focused caption, lifestyle angle, 5-10 hashtags

Include {{TRACKING_LINK}} placeholder.
Make it sound authentic, not salesy."""

            content = await self._call_claude(prompt)

            contents.append(PartnerContent(
                affiliate_id=affiliate.affiliate_id,
                content_type=f"social_{platform}",
                title=f"{platform.title()} Post: {product_name}",
                content=content,
                tracking_link="{TRACKING_LINK}",
                ai_model_used="claude-3-opus",
                generation_prompt=prompt[:500],
            ))

        return contents

    async def generate_blog_review(
        self,
        affiliate: Affiliate,
        product_name: str,
        product_features: list[str],
        target_keywords: list[str],
        word_count: int = 1500
    ) -> PartnerContent:
        """Generate SEO-optimized blog review article."""
        prompt = f"""Write a comprehensive, SEO-optimized blog review article:

Product: {product_name}
Features to Cover: {', '.join(product_features)}
Target Keywords: {', '.join(target_keywords)}
Word Count: ~{word_count} words

Structure:
1. Compelling headline with primary keyword
2. Hook introduction (problem the reader has)
3. Product overview section
4. Detailed feature breakdown
5. Pros and cons (balanced, honest)
6. Who is it best for?
7. Pricing discussion
8. Final verdict with rating (X/10)
9. CTA with {{TRACKING_LINK}}

SEO Requirements:
- Use primary keyword in H1, first 100 words, and conclusion
- Include related keywords naturally
- Use H2/H3 subheadings
- Write meta description (155 chars)
- Suggest featured image alt text

Write authentically, as if you've genuinely used the product."""

        content = await self._call_claude(prompt)

        return PartnerContent(
            affiliate_id=affiliate.affiliate_id,
            content_type="blog_review",
            title=f"Blog Review: {product_name}",
            content=content,
            tracking_link="{TRACKING_LINK}",
            ai_model_used="claude-3-opus",
            generation_prompt=prompt[:500],
        )

    async def generate_comparison_content(
        self,
        affiliate: Affiliate,
        main_product: str,
        competitors: list[str],
        comparison_criteria: list[str]
    ) -> PartnerContent:
        """Generate product comparison content."""
        prompt = f"""Create a detailed product comparison article:

Main Product (we're promoting): {main_product}
Competitors to Compare: {', '.join(competitors)}
Comparison Criteria: {', '.join(comparison_criteria)}

Requirements:
1. Fair, balanced comparison (builds trust)
2. Clear comparison table/matrix
3. Category-by-category breakdown
4. Highlight where main product excels
5. Acknowledge competitor strengths
6. Clear winner recommendation
7. "Best for" scenarios for each product
8. Strong CTA for main product with {{TRACKING_LINK}}

Write objectively - readers should trust this is unbiased
even though we favor the main product where justified."""

        content = await self._call_claude(prompt)

        return PartnerContent(
            affiliate_id=affiliate.affiliate_id,
            content_type="comparison",
            title=f"Comparison: {main_product} vs {', '.join(competitors)}",
            content=content,
            tracking_link="{TRACKING_LINK}",
            ai_model_used="claude-3-opus",
            generation_prompt=prompt[:500],
        )

    async def generate_video_script(
        self,
        affiliate: Affiliate,
        product_name: str,
        video_type: str,  # review, tutorial, unboxing
        duration_minutes: int = 5
    ) -> PartnerContent:
        """Generate video script for YouTube/social."""
        prompt = f"""Write a {video_type} video script for:

Product: {product_name}
Video Type: {video_type}
Target Duration: {duration_minutes} minutes
Platform: YouTube

Script Structure:
1. Hook (first 5 seconds - critical!)
2. Channel intro + subscribe CTA
3. Video overview (what they'll learn)
4. Main content sections with timestamps
5. Demo/walkthrough sections
6. Pros/cons discussion
7. Final verdict
8. Call to action with link mention
9. End screen suggestions

Include:
- Speaking notes (conversational, not robotic)
- B-roll suggestions [in brackets]
- On-screen text suggestions {{in curly braces}}
- Timestamp markers [0:00]
- Engagement prompts (ask questions, polls)

Mention affiliate link: "Link in description" at strategic points."""

        content = await self._call_claude(prompt)

        return PartnerContent(
            affiliate_id=affiliate.affiliate_id,
            content_type="video_script",
            title=f"Video Script ({video_type}): {product_name}",
            content=content,
            tracking_link="Link in description",
            ai_model_used="claude-3-opus",
            generation_prompt=prompt[:500],
        )

    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API for content generation using RealAICore."""
        try:
            # Use ai_core.ai_generate which handles fallback and real keys
            return await ai_generate(
                prompt=prompt,
                model="claude-3-opus-20240229",
                system_prompt="You are an expert affiliate marketing copywriter."
            )
        except Exception as e:
            logger.error(f"Content generation error: {e}")
            return f"[Content generation failed: {str(e)}]"


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

AFFILIATE_TABLES_SQL = """
-- Affiliates table
CREATE TABLE IF NOT EXISTS affiliates (
    affiliate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    partner_type VARCHAR(50) NOT NULL DEFAULT 'affiliate',
    tier VARCHAR(20) NOT NULL DEFAULT 'bronze',
    status VARCHAR(30) NOT NULL DEFAULT 'pending_approval',

    -- Contact Info
    company_name VARCHAR(255),
    contact_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    phone VARCHAR(50),
    website VARCHAR(500),

    -- Tracking
    affiliate_code VARCHAR(50) NOT NULL UNIQUE,
    tracking_links JSONB DEFAULT '{}',
    referral_source VARCHAR(255),

    -- Commission Settings
    commission_structure_id VARCHAR(50),
    custom_commission_rate DECIMAL(10, 4),
    payout_method VARCHAR(50) DEFAULT 'stripe',
    payout_details JSONB DEFAULT '{}',

    -- Multi-tier tracking
    parent_affiliate_id UUID REFERENCES affiliates(affiliate_id),
    tier_level INTEGER DEFAULT 1,

    -- Performance
    total_referrals INTEGER DEFAULT 0,
    total_conversions INTEGER DEFAULT 0,
    total_revenue_generated DECIMAL(15, 2) DEFAULT 0,
    total_commissions_earned DECIMAL(15, 2) DEFAULT 0,
    total_commissions_paid DECIMAL(15, 2) DEFAULT 0,
    pending_commission DECIMAL(15, 2) DEFAULT 0,

    -- Dates
    joined_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity_date TIMESTAMP WITH TIME ZONE,
    next_payout_date TIMESTAMP WITH TIME ZONE,

    -- Metadata
    tags TEXT[],
    notes TEXT,
    custom_data JSONB DEFAULT '{}',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Referrals table
CREATE TABLE IF NOT EXISTS affiliate_referrals (
    referral_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    affiliate_id UUID NOT NULL REFERENCES affiliates(affiliate_id),

    -- Visitor info
    visitor_id VARCHAR(255) NOT NULL,
    customer_id VARCHAR(255),
    ip_address VARCHAR(45),
    user_agent TEXT,

    -- Tracking
    tracking_code VARCHAR(50) NOT NULL,
    landing_page TEXT,
    referrer_url TEXT,
    utm_source VARCHAR(255),
    utm_medium VARCHAR(255),
    utm_campaign VARCHAR(255),
    utm_content VARCHAR(255),

    -- Conversion
    converted BOOLEAN DEFAULT FALSE,
    conversion_date TIMESTAMP WITH TIME ZONE,
    product_id VARCHAR(255),
    product_name VARCHAR(255),
    order_id VARCHAR(255),
    order_value DECIMAL(15, 2) DEFAULT 0,

    -- Commission
    commission_amount DECIMAL(15, 2) DEFAULT 0,
    commission_status VARCHAR(30) DEFAULT 'pending',

    -- Multi-tier
    tier_level INTEGER DEFAULT 1,
    parent_referral_id UUID REFERENCES affiliate_referrals(referral_id),

    -- Attribution
    attribution_model VARCHAR(30) DEFAULT 'last_click',
    attribution_weight DECIMAL(5, 4) DEFAULT 1.0,
    touchpoints JSONB DEFAULT '[]',

    -- Dates
    click_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    cookie_expires TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Commissions table
CREATE TABLE IF NOT EXISTS affiliate_commissions (
    commission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    affiliate_id UUID NOT NULL REFERENCES affiliates(affiliate_id),
    referral_id UUID REFERENCES affiliate_referrals(referral_id),

    -- Amounts
    gross_amount DECIMAL(15, 2) NOT NULL DEFAULT 0,
    adjustments DECIMAL(15, 2) DEFAULT 0,
    net_amount DECIMAL(15, 2) NOT NULL DEFAULT 0,

    -- Details
    commission_type VARCHAR(30) NOT NULL DEFAULT 'percentage',
    rate_applied DECIMAL(10, 4) NOT NULL,
    tier_level INTEGER DEFAULT 1,
    tier_rate_modifier DECIMAL(5, 4) DEFAULT 1.0,

    -- Status
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    payout_id UUID,

    -- Dates
    earned_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    available_date TIMESTAMP WITH TIME ZONE,
    paid_date TIMESTAMP WITH TIME ZONE,

    -- Metadata
    description TEXT,
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Payouts table
CREATE TABLE IF NOT EXISTS affiliate_payouts (
    payout_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    affiliate_id UUID NOT NULL REFERENCES affiliates(affiliate_id),

    -- Amounts
    gross_amount DECIMAL(15, 2) NOT NULL DEFAULT 0,
    fees DECIMAL(15, 2) DEFAULT 0,
    net_amount DECIMAL(15, 2) NOT NULL DEFAULT 0,

    -- Commissions included
    commission_ids UUID[],
    commission_count INTEGER DEFAULT 0,

    -- Payment
    payment_method VARCHAR(50) NOT NULL,
    payment_reference VARCHAR(255),
    currency VARCHAR(3) DEFAULT 'USD',

    -- Status
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    failure_reason TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Dates
    created_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_date TIMESTAMP WITH TIME ZONE,
    completed_date TIMESTAMP WITH TIME ZONE
);

-- Payout Queue table
CREATE TABLE IF NOT EXISTS affiliate_payout_queue (
    queue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payout_id UUID,
    affiliate_id UUID NOT NULL REFERENCES affiliates(affiliate_id),

    -- Queue status
    status VARCHAR(30) NOT NULL DEFAULT 'queued',
    priority INTEGER DEFAULT 0,

    -- Amounts
    amount DECIMAL(15, 2) NOT NULL DEFAULT 0,
    currency VARCHAR(3) DEFAULT 'USD',
    payment_method VARCHAR(50),

    -- Processing
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    last_attempt_at TIMESTAMP WITH TIME ZONE,
    next_retry_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,

    -- Timing
    queued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Partner Content table
CREATE TABLE IF NOT EXISTS affiliate_content (
    content_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    affiliate_id UUID NOT NULL REFERENCES affiliates(affiliate_id),

    -- Content details
    content_type VARCHAR(50) NOT NULL,
    title VARCHAR(500),
    content TEXT,

    -- Assets
    images TEXT[],
    tracking_link TEXT,

    -- Performance
    times_used INTEGER DEFAULT 0,
    clicks_generated INTEGER DEFAULT 0,
    conversions_generated INTEGER DEFAULT 0,

    -- Metadata
    created_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ai_model_used VARCHAR(100),
    generation_prompt TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_affiliates_email ON affiliates(email);
CREATE INDEX IF NOT EXISTS idx_affiliates_code ON affiliates(affiliate_code);
CREATE INDEX IF NOT EXISTS idx_affiliates_status ON affiliates(status);
CREATE INDEX IF NOT EXISTS idx_affiliates_parent ON affiliates(parent_affiliate_id);

CREATE INDEX IF NOT EXISTS idx_referrals_affiliate ON affiliate_referrals(affiliate_id);
CREATE INDEX IF NOT EXISTS idx_referrals_visitor ON affiliate_referrals(visitor_id);
CREATE INDEX IF NOT EXISTS idx_referrals_converted ON affiliate_referrals(converted);
CREATE INDEX IF NOT EXISTS idx_referrals_click_date ON affiliate_referrals(click_date);

CREATE INDEX IF NOT EXISTS idx_commissions_affiliate ON affiliate_commissions(affiliate_id);
CREATE INDEX IF NOT EXISTS idx_commissions_status ON affiliate_commissions(status);
CREATE INDEX IF NOT EXISTS idx_commissions_payout ON affiliate_commissions(payout_id);

CREATE INDEX IF NOT EXISTS idx_payouts_affiliate ON affiliate_payouts(affiliate_id);
CREATE INDEX IF NOT EXISTS idx_payouts_status ON affiliate_payouts(status);

CREATE INDEX IF NOT EXISTS idx_queue_status ON affiliate_payout_queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_priority ON affiliate_payout_queue(priority DESC, queued_at ASC);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_affiliates_updated_at ON affiliates;
CREATE TRIGGER update_affiliates_updated_at
    BEFORE UPDATE ON affiliates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_commissions_updated_at ON affiliate_commissions;
CREATE TRIGGER update_commissions_updated_at
    BEFORE UPDATE ON affiliate_commissions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
"""


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class AffiliatePartnershipPipeline:
    """
    Main orchestrator for affiliate and partnership programs.

    Capabilities:
    - Partner onboarding and management
    - Click/conversion tracking
    - Multi-tier commission calculation
    - Payout processing
    - Performance analytics
    - Fraud detection
    - AI content generation
    """

    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        env = os.getenv("ENVIRONMENT", "production").strip().lower()
        allow_inmemory = os.getenv("ALLOW_AFFILIATE_INMEMORY", "").strip().lower() in {"1", "true", "yes"}

        if env in {"production", "prod"} and allow_inmemory:
            raise RuntimeError("ALLOW_AFFILIATE_INMEMORY is not permitted in production environments")

        if not self.db_url and not allow_inmemory:
            raise RuntimeError(
                "Affiliate pipeline requires DATABASE_URL or ALLOW_AFFILIATE_INMEMORY=true for explicit dev-only mode"
            )

        # In-memory storage (would be database in production)
        self.affiliates: dict[str, Affiliate] = {}
        self.referrals: dict[str, Referral] = {}
        self.commissions: dict[str, Commission] = {}
        self.payouts: dict[str, Payout] = {}
        self.content: dict[str, PartnerContent] = {}
        self.payout_queue: dict[str, PayoutQueueItem] = {}  # Payout queue for batch processing

        # Components
        self.fraud_detector = FraudDetector()
        self.content_generator = AffiliateContentGenerator()

        # Tracking code -> affiliate_id mapping
        self.tracking_codes: dict[str, str] = {}

        # Commission structures
        self.commission_structures = DEFAULT_COMMISSION_STRUCTURES.copy()

        # Database connection pool (lazy initialization)
        self._db_pool = None

        logger.info(
            "AffiliatePartnershipPipeline initialized (env=%s, storage=%s)",
            env,
            "database" if self.db_url else "inmemory",
        )

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================

    async def _get_db_pool(self):
        """Get or create database connection pool."""
        if self._db_pool is None and self.db_url:
            try:
                import asyncpg
                self._db_pool = await asyncpg.create_pool(
                    self.db_url,
                    min_size=2,
                    max_size=10,
                    command_timeout=60
                )
                logger.info("Database connection pool created")
            except ImportError:
                logger.warning("asyncpg not installed, database features disabled")
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}")
        return self._db_pool

    async def initialize_database(self) -> bool:
        """
        Initialize database tables for the affiliate system.

        Returns:
            True if successful, False otherwise
        """
        pool = await self._get_db_pool()
        if not pool:
            logger.warning("No database pool available, using in-memory storage only")
            return False

        try:
            async with pool.acquire() as conn:
                await conn.execute(AFFILIATE_TABLES_SQL)
                logger.info("Affiliate database tables created/verified")
                return True
        except Exception as e:
            logger.error(f"Failed to initialize affiliate tables: {e}")
            return False

    async def sync_to_database(self) -> dict[str, int]:
        """
        Sync in-memory data to database.

        Returns:
            Dict with counts of synced records
        """
        pool = await self._get_db_pool()
        if not pool:
            return {"error": "No database connection"}

        counts = {
            "affiliates": 0,
            "referrals": 0,
            "commissions": 0,
            "payouts": 0,
            "queue_items": 0,
        }

        try:
            async with pool.acquire() as conn:
                # Sync affiliates
                for affiliate in self.affiliates.values():
                    await conn.execute("""
                        INSERT INTO affiliates (
                            affiliate_id, partner_type, tier, status,
                            company_name, contact_name, email, phone, website,
                            affiliate_code, tracking_links, referral_source,
                            commission_structure_id, custom_commission_rate,
                            payout_method, payout_details,
                            parent_affiliate_id, tier_level,
                            total_referrals, total_conversions,
                            total_revenue_generated, total_commissions_earned,
                            total_commissions_paid, pending_commission,
                            joined_date, last_activity_date, next_payout_date,
                            tags, notes, custom_data
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                                  $11, $12, $13, $14, $15, $16, $17, $18,
                                  $19, $20, $21, $22, $23, $24, $25, $26, $27,
                                  $28, $29, $30)
                        ON CONFLICT (affiliate_id) DO UPDATE SET
                            tier = EXCLUDED.tier,
                            status = EXCLUDED.status,
                            total_referrals = EXCLUDED.total_referrals,
                            total_conversions = EXCLUDED.total_conversions,
                            total_revenue_generated = EXCLUDED.total_revenue_generated,
                            total_commissions_earned = EXCLUDED.total_commissions_earned,
                            total_commissions_paid = EXCLUDED.total_commissions_paid,
                            pending_commission = EXCLUDED.pending_commission,
                            last_activity_date = EXCLUDED.last_activity_date
                    """,
                        affiliate.affiliate_id,
                        affiliate.partner_type.value,
                        affiliate.tier.value,
                        affiliate.status.value,
                        affiliate.company_name,
                        affiliate.contact_name,
                        affiliate.email,
                        affiliate.phone,
                        affiliate.website,
                        affiliate.affiliate_code,
                        json.dumps(affiliate.tracking_links),
                        affiliate.referral_source,
                        affiliate.commission_structure_id,
                        float(affiliate.custom_commission_rate) if affiliate.custom_commission_rate else None,
                        affiliate.payout_method,
                        json.dumps(affiliate.payout_details),
                        affiliate.parent_affiliate_id,
                        affiliate.tier_level,
                        affiliate.total_referrals,
                        affiliate.total_conversions,
                        float(affiliate.total_revenue_generated),
                        float(affiliate.total_commissions_earned),
                        float(affiliate.total_commissions_paid),
                        float(affiliate.pending_commission),
                        affiliate.joined_date,
                        affiliate.last_activity_date,
                        affiliate.next_payout_date,
                        affiliate.tags,
                        affiliate.notes,
                        json.dumps(affiliate.custom_data),
                    )
                    counts["affiliates"] += 1

                # Sync commissions
                for commission in self.commissions.values():
                    await conn.execute("""
                        INSERT INTO affiliate_commissions (
                            commission_id, affiliate_id, referral_id,
                            gross_amount, adjustments, net_amount,
                            commission_type, rate_applied, tier_level, tier_rate_modifier,
                            status, payout_id,
                            earned_date, available_date, paid_date,
                            description, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                                  $11, $12, $13, $14, $15, $16, $17)
                        ON CONFLICT (commission_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            payout_id = EXCLUDED.payout_id,
                            paid_date = EXCLUDED.paid_date
                    """,
                        commission.commission_id,
                        commission.affiliate_id,
                        commission.referral_id,
                        float(commission.gross_amount),
                        float(commission.adjustments),
                        float(commission.net_amount),
                        commission.commission_type.value,
                        float(commission.rate_applied),
                        commission.tier_level,
                        commission.tier_rate_modifier,
                        commission.status.value,
                        commission.payout_id,
                        commission.earned_date,
                        commission.available_date,
                        commission.paid_date,
                        commission.description,
                        json.dumps(commission.metadata),
                    )
                    counts["commissions"] += 1

                logger.info(f"Synced to database: {counts}")
                return counts

        except Exception as e:
            logger.error(f"Failed to sync to database: {e}")
            return {"error": str(e)}

    async def load_from_database(self) -> dict[str, int]:
        """
        Load data from database into in-memory storage.

        Returns:
            Dict with counts of loaded records
        """
        pool = await self._get_db_pool()
        if not pool:
            return {"error": "No database connection"}

        counts = {
            "affiliates": 0,
            "referrals": 0,
            "commissions": 0,
            "payouts": 0,
        }

        try:
            async with pool.acquire() as conn:
                # Load affiliates
                rows = await conn.fetch("SELECT * FROM affiliates")
                for row in rows:
                    affiliate = Affiliate(
                        affiliate_id=str(row["affiliate_id"]),
                        partner_type=PartnerType(row["partner_type"]),
                        tier=PartnerTier(row["tier"]),
                        status=AffiliateStatus(row["status"]),
                        company_name=row["company_name"] or "",
                        contact_name=row["contact_name"],
                        email=row["email"],
                        phone=row["phone"] or "",
                        website=row["website"] or "",
                        affiliate_code=row["affiliate_code"],
                        tracking_links=row["tracking_links"] or {},
                        referral_source=row["referral_source"] or "",
                        commission_structure_id=row["commission_structure_id"] or "",
                        custom_commission_rate=Decimal(str(row["custom_commission_rate"])) if row["custom_commission_rate"] else None,
                        payout_method=row["payout_method"] or "stripe",
                        payout_details=row["payout_details"] or {},
                        parent_affiliate_id=str(row["parent_affiliate_id"]) if row["parent_affiliate_id"] else None,
                        tier_level=row["tier_level"] or 1,
                        total_referrals=row["total_referrals"] or 0,
                        total_conversions=row["total_conversions"] or 0,
                        total_revenue_generated=Decimal(str(row["total_revenue_generated"] or 0)),
                        total_commissions_earned=Decimal(str(row["total_commissions_earned"] or 0)),
                        total_commissions_paid=Decimal(str(row["total_commissions_paid"] or 0)),
                        pending_commission=Decimal(str(row["pending_commission"] or 0)),
                        joined_date=row["joined_date"],
                        last_activity_date=row["last_activity_date"],
                        next_payout_date=row["next_payout_date"],
                        tags=row["tags"] or [],
                        notes=row["notes"] or "",
                        custom_data=row["custom_data"] or {},
                    )
                    self.affiliates[affiliate.affiliate_id] = affiliate
                    self.tracking_codes[affiliate.affiliate_code] = affiliate.affiliate_id
                    counts["affiliates"] += 1

                # Load commissions
                rows = await conn.fetch("SELECT * FROM affiliate_commissions")
                for row in rows:
                    commission = Commission(
                        commission_id=str(row["commission_id"]),
                        affiliate_id=str(row["affiliate_id"]),
                        referral_id=str(row["referral_id"]) if row["referral_id"] else "",
                        gross_amount=Decimal(str(row["gross_amount"] or 0)),
                        adjustments=Decimal(str(row["adjustments"] or 0)),
                        net_amount=Decimal(str(row["net_amount"] or 0)),
                        commission_type=CommissionType(row["commission_type"]),
                        rate_applied=Decimal(str(row["rate_applied"])),
                        tier_level=row["tier_level"] or 1,
                        tier_rate_modifier=float(row["tier_rate_modifier"] or 1.0),
                        status=PayoutStatus(row["status"]),
                        payout_id=str(row["payout_id"]) if row["payout_id"] else None,
                        earned_date=row["earned_date"],
                        available_date=row["available_date"],
                        paid_date=row["paid_date"],
                        description=row["description"] or "",
                        metadata=row["metadata"] or {},
                    )
                    self.commissions[commission.commission_id] = commission
                    counts["commissions"] += 1

                logger.info(f"Loaded from database: {counts}")
                return counts

        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            return {"error": str(e)}

    # =========================================================================
    # AFFILIATE MANAGEMENT
    # =========================================================================

    async def register_affiliate(
        self,
        email: str,
        contact_name: str,
        company_name: str = "",
        partner_type: PartnerType = PartnerType.AFFILIATE,
        website: str = "",
        parent_affiliate_code: str = None,
        metadata: dict[str, Any] = None
    ) -> Affiliate:
        """Register a new affiliate."""

        # Check for existing affiliate
        for aff in self.affiliates.values():
            if aff.email == email:
                raise ValueError(f"Affiliate with email {email} already exists")

        # Create affiliate
        affiliate = Affiliate(
            email=email,
            contact_name=contact_name,
            company_name=company_name,
            partner_type=partner_type,
            website=website,
            status=AffiliateStatus.PENDING_APPROVAL,
            custom_data=metadata or {},
        )

        # Set commission structure
        structure = self.commission_structures.get(partner_type)
        if structure:
            affiliate.commission_structure_id = structure.structure_id

        # Handle multi-tier referral
        if parent_affiliate_code:
            parent = await self.get_affiliate_by_code(parent_affiliate_code)
            if parent:
                affiliate.parent_affiliate_id = parent.affiliate_id
                affiliate.tier_level = parent.tier_level + 1
                affiliate.referral_source = f"affiliate:{parent.affiliate_code}"

        # Generate tracking link
        base_url = os.getenv("BASE_URL", "https://app.example.com")
        affiliate.tracking_links["default"] = f"{base_url}?ref={affiliate.affiliate_code}"

        # Store
        self.affiliates[affiliate.affiliate_id] = affiliate
        self.tracking_codes[affiliate.affiliate_code] = affiliate.affiliate_id

        logger.info(f"Registered new affiliate: {affiliate.affiliate_id} ({email})")

        return affiliate

    async def approve_affiliate(self, affiliate_id: str) -> Affiliate:
        """Approve a pending affiliate."""
        affiliate = self.affiliates.get(affiliate_id)
        if not affiliate:
            raise ValueError(f"Affiliate {affiliate_id} not found")

        affiliate.status = AffiliateStatus.ACTIVE
        logger.info(f"Approved affiliate: {affiliate_id}")

        # Send welcome email
        await self._send_welcome_email(affiliate)

        return affiliate

    async def update_affiliate_tier(
        self,
        affiliate_id: str,
        new_tier: PartnerTier
    ) -> Affiliate:
        """Update affiliate's tier based on performance."""
        affiliate = self.affiliates.get(affiliate_id)
        if not affiliate:
            raise ValueError(f"Affiliate {affiliate_id} not found")

        old_tier = affiliate.tier
        affiliate.tier = new_tier

        logger.info(f"Updated affiliate {affiliate_id} tier: {old_tier.value} -> {new_tier.value}")

        return affiliate

    async def get_affiliate_by_code(self, code: str) -> Optional[Affiliate]:
        """Get affiliate by tracking code."""
        affiliate_id = self.tracking_codes.get(code)
        if affiliate_id:
            return self.affiliates.get(affiliate_id)
        return None

    async def auto_tier_affiliates(self) -> list[tuple[str, PartnerTier, PartnerTier]]:
        """Automatically update affiliate tiers based on monthly revenue."""
        changes = []

        # Calculate monthly revenue for each affiliate
        now = datetime.utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        for affiliate in self.affiliates.values():
            if affiliate.status != AffiliateStatus.ACTIVE:
                continue

            # Calculate monthly revenue
            monthly_revenue = Decimal("0")
            for ref in self.referrals.values():
                if (ref.affiliate_id == affiliate.affiliate_id and
                    ref.converted and
                    ref.conversion_date and
                    ref.conversion_date >= month_start):
                    monthly_revenue += ref.order_value

            # Determine appropriate tier
            new_tier = PartnerTier.BRONZE
            for tier, threshold in sorted(
                TIER_THRESHOLDS.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if monthly_revenue >= threshold:
                    new_tier = tier
                    break

            # Update if changed
            if new_tier != affiliate.tier:
                old_tier = affiliate.tier
                affiliate.tier = new_tier
                changes.append((affiliate.affiliate_id, old_tier, new_tier))
                logger.info(
                    f"Auto-tiered {affiliate.affiliate_id}: "
                    f"{old_tier.value} -> {new_tier.value} "
                    f"(${monthly_revenue}/month)"
                )

        return changes

    # =========================================================================
    # TRACKING
    # =========================================================================

    async def track_click(
        self,
        tracking_code: str,
        visitor_id: str,
        ip_address: str,
        user_agent: str,
        landing_page: str,
        referrer_url: str = "",
        utm_params: dict[str, str] = None
    ) -> tuple[Referral, list[FraudSignal]]:
        """Track an affiliate click."""

        # Get affiliate
        affiliate = await self.get_affiliate_by_code(tracking_code)
        if not affiliate:
            raise ValueError(f"Invalid tracking code: {tracking_code}")

        if affiliate.status != AffiliateStatus.ACTIVE:
            raise ValueError(f"Affiliate {affiliate.affiliate_id} is not active")

        # Get commission structure
        structure = self.commission_structures.get(affiliate.partner_type)
        cookie_days = structure.cookie_duration_days if structure else 90

        # Create referral
        utm = utm_params or {}
        referral = Referral(
            affiliate_id=affiliate.affiliate_id,
            visitor_id=visitor_id,
            tracking_code=tracking_code,
            ip_address=ip_address,
            user_agent=user_agent,
            landing_page=landing_page,
            referrer_url=referrer_url,
            utm_source=utm.get("utm_source", ""),
            utm_medium=utm.get("utm_medium", ""),
            utm_campaign=utm.get("utm_campaign", ""),
            utm_content=utm.get("utm_content", ""),
            cookie_expires=datetime.utcnow() + timedelta(days=cookie_days),
            tier_level=affiliate.tier_level,
        )

        # Fraud detection
        fraud_signals = await self.fraud_detector.analyze_click(referral, affiliate)
        fraud_score = self.fraud_detector.calculate_fraud_score(fraud_signals)

        if fraud_score >= 0.9:
            logger.warning(
                f"High fraud score ({fraud_score:.2f}) for click - "
                f"affiliate: {affiliate.affiliate_id}, IP: {ip_address}"
            )
            # Still track but flag
            referral.touchpoints.append({
                "type": "fraud_flag",
                "score": fraud_score,
                "signals": [s.__dict__ for s in fraud_signals],
            })

        # Store
        self.referrals[referral.referral_id] = referral

        # Update affiliate stats
        affiliate.total_referrals += 1
        affiliate.last_activity_date = datetime.utcnow()

        # Track touchpoint
        referral.touchpoints.append({
            "type": "click",
            "timestamp": datetime.utcnow().isoformat(),
            "page": landing_page,
        })

        logger.info(
            f"Tracked click: affiliate={affiliate.affiliate_id}, "
            f"visitor={visitor_id}, fraud_score={fraud_score:.2f}"
        )

        return referral, fraud_signals

    async def track_conversion(
        self,
        visitor_id: str = None,
        referral_id: str = None,
        customer_id: str = "",
        order_id: str = "",
        order_value: Decimal = Decimal("0"),
        product_id: str = "",
        product_name: str = "",
        attribution_model: AttributionModel = AttributionModel.LAST_CLICK
    ) -> Optional[Commission]:
        """Track a conversion and calculate commission."""

        # Find referral
        referral = None
        if referral_id:
            referral = self.referrals.get(referral_id)
        elif visitor_id:
            # Find most recent referral for visitor within cookie window
            now = datetime.utcnow()
            candidates = [
                r for r in self.referrals.values()
                if (r.visitor_id == visitor_id and
                    not r.converted and
                    r.cookie_expires and
                    r.cookie_expires > now)
            ]
            if candidates:
                # Apply attribution model
                if attribution_model == AttributionModel.LAST_CLICK:
                    referral = max(candidates, key=lambda r: r.click_date)
                elif attribution_model == AttributionModel.FIRST_CLICK:
                    referral = min(candidates, key=lambda r: r.click_date)

        if not referral:
            logger.info(f"No valid referral found for conversion (visitor={visitor_id})")
            return None

        # Get affiliate
        affiliate = self.affiliates.get(referral.affiliate_id)
        if not affiliate or affiliate.status != AffiliateStatus.ACTIVE:
            logger.warning(f"Affiliate not active for referral {referral.referral_id}")
            return None

        # Fraud detection
        referral.conversion_date = datetime.utcnow()
        referral.order_value = order_value
        fraud_signals = await self.fraud_detector.analyze_conversion(referral, affiliate)
        fraud_score = self.fraud_detector.calculate_fraud_score(fraud_signals)

        # Update referral
        referral.converted = True
        referral.customer_id = customer_id
        referral.order_id = order_id
        referral.product_id = product_id
        referral.product_name = product_name
        referral.attribution_model = attribution_model

        # Add conversion touchpoint
        referral.touchpoints.append({
            "type": "conversion",
            "timestamp": datetime.utcnow().isoformat(),
            "order_id": order_id,
            "value": str(order_value),
        })

        # Calculate commission
        commission = await self._calculate_commission(
            referral, affiliate, order_value, fraud_score
        )

        if commission:
            self.commissions[commission.commission_id] = commission

            # Update affiliate stats
            affiliate.total_conversions += 1
            affiliate.total_revenue_generated += order_value
            affiliate.total_commissions_earned += commission.net_amount
            affiliate.pending_commission += commission.net_amount

            # Handle multi-tier commissions
            await self._process_multi_tier_commissions(referral, order_value)

        logger.info(
            f"Tracked conversion: affiliate={affiliate.affiliate_id}, "
            f"order={order_id}, value=${order_value}, "
            f"commission=${commission.net_amount if commission else 0}"
        )

        return commission

    async def _calculate_commission(
        self,
        referral: Referral,
        affiliate: Affiliate,
        order_value: Decimal,
        fraud_score: float
    ) -> Optional[Commission]:
        """Calculate commission for a conversion."""

        # Get commission structure
        structure = self.commission_structures.get(affiliate.partner_type)
        if not structure:
            structure = DEFAULT_COMMISSION_STRUCTURES[PartnerType.AFFILIATE]

        # Get rate (custom or tier-based)
        if affiliate.custom_commission_rate:
            rate = affiliate.custom_commission_rate
        else:
            rate = structure.get_rate_for_tier(affiliate.tier)

        # Calculate base commission
        if structure.commission_type == CommissionType.PERCENTAGE:
            gross_amount = order_value * rate
        elif structure.commission_type == CommissionType.FLAT_RATE:
            gross_amount = Decimal(str(rate))  # Rate is flat amount
        elif structure.commission_type == CommissionType.RECURRING:
            # Same as percentage but tracked for recurring
            gross_amount = order_value * rate
        elif structure.commission_type == CommissionType.HYBRID:
            # Base + percentage
            base_bonus = Decimal("25")  # $25 base
            gross_amount = base_bonus + (order_value * rate)
        else:
            gross_amount = order_value * rate

        # Apply fraud penalty
        if fraud_score > 0.5:
            penalty = Decimal(str(fraud_score * 0.5))  # Up to 50% reduction
            gross_amount = gross_amount * (1 - penalty)
            logger.info(
                f"Applied fraud penalty ({fraud_score:.2f}): "
                f"commission reduced by {penalty*100:.0f}%"
            )

        # Create commission record
        commission = Commission(
            affiliate_id=affiliate.affiliate_id,
            referral_id=referral.referral_id,
            gross_amount=gross_amount,
            net_amount=gross_amount,  # Adjustments would modify this
            commission_type=structure.commission_type,
            rate_applied=rate,
            tier_level=referral.tier_level,
            description=f"Commission for order {referral.order_id}",
            metadata={
                "order_value": str(order_value),
                "fraud_score": fraud_score,
                "product": referral.product_name,
            }
        )

        # Set availability date (hold period for chargebacks)
        hold_days = 30  # Standard hold period
        commission.available_date = datetime.utcnow() + timedelta(days=hold_days)

        return commission

    def calculate_flat_percentage_commission(
        self,
        order_value: Decimal,
        rate: Decimal
    ) -> Decimal:
        """
        Calculate a flat percentage commission on an order.

        Args:
            order_value: The total value of the order
            rate: The commission rate as a decimal (e.g., 0.25 for 25%)

        Returns:
            The commission amount
        """
        if order_value <= 0 or rate <= 0:
            return Decimal("0")

        commission = order_value * rate
        # Round to 2 decimal places
        return commission.quantize(Decimal("0.01"))

    def calculate_tiered_volume_commission(
        self,
        order_value: Decimal,
        affiliate_monthly_volume: Decimal,
        custom_tiers: dict[tuple[Decimal, Decimal], Decimal] = None
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate commission based on tiered volume thresholds.
        Higher monthly sales volume = higher commission rate.

        Args:
            order_value: The value of the current order
            affiliate_monthly_volume: The affiliate's total sales this month (before this order)
            custom_tiers: Optional custom tier structure, defaults to VOLUME_TIERED_COMMISSION_RATES

        Returns:
            Tuple of (commission_amount, rate_applied)
        """
        if order_value <= 0:
            return Decimal("0"), Decimal("0")

        tiers = custom_tiers or VOLUME_TIERED_COMMISSION_RATES

        # Determine which tier the affiliate falls into based on their monthly volume
        applicable_rate = Decimal("0.15")  # Default base rate

        for (min_vol, max_vol), rate in sorted(tiers.items(), key=lambda x: x[0][0]):
            if min_vol <= affiliate_monthly_volume < max_vol:
                applicable_rate = rate
                break

        commission = order_value * applicable_rate
        return commission.quantize(Decimal("0.01")), applicable_rate

    def calculate_progressive_tiered_commission(
        self,
        order_value: Decimal,
        affiliate_monthly_volume: Decimal
    ) -> tuple[Decimal, dict[str, Any]]:
        """
        Calculate commission using progressive tiers (like tax brackets).
        Each portion of the sale is taxed at the rate for that tier.

        Args:
            order_value: The value of the current order
            affiliate_monthly_volume: The affiliate's total sales this month (before this order)

        Returns:
            Tuple of (total_commission, breakdown_dict)
        """
        if order_value <= 0:
            return Decimal("0"), {"tiers": [], "total": "0"}

        total_commission = Decimal("0")
        breakdown = []
        remaining_value = order_value
        current_volume = affiliate_monthly_volume

        # Sort tiers by minimum volume
        sorted_tiers = sorted(VOLUME_TIERED_COMMISSION_RATES.items(), key=lambda x: x[0][0])

        for (min_vol, max_vol), rate in sorted_tiers:
            if remaining_value <= 0:
                break

            # How much of the order falls into this tier?
            if current_volume >= max_vol:
                # Already past this tier
                continue
            elif current_volume >= min_vol:
                # Currently in this tier
                space_in_tier = max_vol - current_volume
                value_in_tier = min(remaining_value, space_in_tier)

                tier_commission = value_in_tier * rate
                total_commission += tier_commission

                breakdown.append({
                    "tier": f"${min_vol}-${max_vol}",
                    "rate": str(rate),
                    "value": str(value_in_tier),
                    "commission": str(tier_commission.quantize(Decimal("0.01")))
                })

                remaining_value -= value_in_tier
                current_volume += value_in_tier

        return total_commission.quantize(Decimal("0.01")), {
            "tiers": breakdown,
            "total": str(total_commission.quantize(Decimal("0.01")))
        }

    async def calculate_multi_level_commission(
        self,
        order_value: Decimal,
        affiliate_id: str,
        max_levels: int = 3
    ) -> list[dict[str, Any]]:
        """
        Calculate multi-level marketing (MLM) style commissions for an affiliate chain.
        Up to 3 levels of upline affiliates receive a portion of the commission.

        Args:
            order_value: The value of the order
            affiliate_id: The ID of the referring affiliate (level 1)
            max_levels: Maximum number of levels to pay out (default 3)

        Returns:
            List of commission records for each level
        """
        commissions = []

        affiliate = self.affiliates.get(affiliate_id)
        if not affiliate:
            return commissions

        current_affiliate = affiliate
        current_level = 1

        while current_affiliate and current_level <= max_levels:
            # Get the tier rate modifier for this level
            tier_modifier = MULTI_TIER_RATES.get(current_level, Decimal("0"))

            if tier_modifier <= 0:
                break

            # Get the affiliate's base rate from their commission structure
            structure = self.commission_structures.get(current_affiliate.partner_type)
            if not structure:
                structure = DEFAULT_COMMISSION_STRUCTURES[PartnerType.AFFILIATE]

            base_rate = structure.get_rate_for_tier(current_affiliate.tier)

            # Calculate commission for this level
            # Level 1 gets full rate, Level 2 gets modifier of Level 1's rate, etc.
            if current_level == 1:
                level_rate = base_rate
            else:
                level_rate = base_rate * tier_modifier

            commission_amount = (order_value * level_rate).quantize(Decimal("0.01"))

            commissions.append({
                "level": current_level,
                "affiliate_id": current_affiliate.affiliate_id,
                "affiliate_code": current_affiliate.affiliate_code,
                "affiliate_name": current_affiliate.contact_name,
                "base_rate": str(base_rate),
                "tier_modifier": str(tier_modifier),
                "effective_rate": str(level_rate),
                "order_value": str(order_value),
                "commission_amount": str(commission_amount),
            })

            # Move up the chain
            if current_affiliate.parent_affiliate_id:
                current_affiliate = self.affiliates.get(current_affiliate.parent_affiliate_id)
                current_level += 1
            else:
                break

        return commissions

    async def get_affiliate_monthly_volume(
        self,
        affiliate_id: str,
        month_start: datetime = None
    ) -> Decimal:
        """
        Get the total sales volume for an affiliate in the current month.

        Args:
            affiliate_id: The affiliate's ID
            month_start: Optional start of the month (defaults to current month)

        Returns:
            Total sales volume for the month
        """
        if month_start is None:
            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        total_volume = Decimal("0")

        for referral in self.referrals.values():
            if (referral.affiliate_id == affiliate_id and
                referral.converted and
                referral.conversion_date and
                referral.conversion_date >= month_start):
                total_volume += referral.order_value

        return total_volume

    async def _process_multi_tier_commissions(
        self,
        referral: Referral,
        order_value: Decimal
    ) -> list[Commission]:
        """Process multi-tier commissions for parent affiliates."""
        tier_commissions = []

        # Get the original affiliate
        affiliate = self.affiliates.get(referral.affiliate_id)
        if not affiliate or not affiliate.parent_affiliate_id:
            return tier_commissions

        # Walk up the tier chain
        current_parent_id = affiliate.parent_affiliate_id
        current_tier = 2

        while current_parent_id and current_tier <= 3:
            parent = self.affiliates.get(current_parent_id)
            if not parent or parent.status != AffiliateStatus.ACTIVE:
                break

            # Get tier rate modifier
            tier_modifier = MULTI_TIER_RATES.get(current_tier, Decimal("0"))
            if tier_modifier <= 0:
                break

            # Get parent's base commission rate
            structure = self.commission_structures.get(parent.partner_type)
            if not structure:
                structure = DEFAULT_COMMISSION_STRUCTURES[PartnerType.AFFILIATE]

            base_rate = structure.get_rate_for_tier(parent.tier)
            tier_rate = base_rate * tier_modifier

            # Calculate tier commission
            tier_commission = Commission(
                affiliate_id=parent.affiliate_id,
                referral_id=referral.referral_id,
                gross_amount=order_value * tier_rate,
                net_amount=order_value * tier_rate,
                commission_type=CommissionType.PERCENTAGE,
                rate_applied=tier_rate,
                tier_level=current_tier,
                tier_rate_modifier=float(tier_modifier),
                description=f"Tier {current_tier} commission from {affiliate.affiliate_code}",
                metadata={
                    "original_affiliate": affiliate.affiliate_id,
                    "order_value": str(order_value),
                }
            )
            tier_commission.available_date = datetime.utcnow() + timedelta(days=30)

            self.commissions[tier_commission.commission_id] = tier_commission
            tier_commissions.append(tier_commission)

            # Update parent stats
            parent.total_commissions_earned += tier_commission.net_amount
            parent.pending_commission += tier_commission.net_amount

            logger.info(
                f"Created tier {current_tier} commission: "
                f"parent={parent.affiliate_id}, amount=${tier_commission.net_amount}"
            )

            # Move up the chain
            current_parent_id = parent.parent_affiliate_id
            current_tier += 1

        return tier_commissions

    # =========================================================================
    # PAYOUTS
    # =========================================================================

    async def process_payouts(
        self,
        affiliate_ids: list[str] = None
    ) -> list[Payout]:
        """Process pending payouts for affiliates."""
        payouts = []

        # Get affiliates to process
        if affiliate_ids:
            affiliates = [
                self.affiliates[aid] for aid in affiliate_ids
                if aid in self.affiliates
            ]
        else:
            affiliates = [
                a for a in self.affiliates.values()
                if a.status == AffiliateStatus.ACTIVE
            ]

        now = datetime.utcnow()

        for affiliate in affiliates:
            # Get available commissions
            available_commissions = [
                c for c in self.commissions.values()
                if (c.affiliate_id == affiliate.affiliate_id and
                    c.status == PayoutStatus.PENDING and
                    c.available_date and
                    c.available_date <= now)
            ]

            if not available_commissions:
                continue

            # Calculate total
            total_amount = sum(c.net_amount for c in available_commissions)

            # Check minimum payout
            structure = self.commission_structures.get(affiliate.partner_type)
            min_payout = structure.minimum_payout if structure else Decimal("50")

            if total_amount < min_payout:
                logger.info(
                    f"Affiliate {affiliate.affiliate_id} below minimum payout "
                    f"(${total_amount} < ${min_payout})"
                )
                continue

            # Create payout
            fees = total_amount * Decimal("0.02")  # 2% processing fee
            payout = Payout(
                affiliate_id=affiliate.affiliate_id,
                gross_amount=total_amount,
                fees=fees,
                net_amount=total_amount - fees,
                commission_ids=[c.commission_id for c in available_commissions],
                commission_count=len(available_commissions),
                payment_method=affiliate.payout_method,
                currency="USD",
            )

            # Update commission status
            for commission in available_commissions:
                commission.status = PayoutStatus.PROCESSING
                commission.payout_id = payout.payout_id

            # Process payment
            payout.status = PayoutStatus.PROCESSING
            payment_result = await self._process_payment(payout, affiliate)

            if payment_result["success"]:
                payout.status = PayoutStatus.COMPLETED
                payout.payment_reference = payment_result.get("reference", "")
                payout.completed_date = datetime.utcnow()

                # Update commissions
                for commission in available_commissions:
                    commission.status = PayoutStatus.COMPLETED
                    commission.paid_date = datetime.utcnow()

                # Update affiliate
                affiliate.total_commissions_paid += payout.net_amount
                affiliate.pending_commission -= payout.gross_amount
            else:
                payout.status = PayoutStatus.FAILED
                payout.failure_reason = payment_result.get("error", "Unknown error")

                # Reset commissions
                for commission in available_commissions:
                    commission.status = PayoutStatus.PENDING
                    commission.payout_id = None

            self.payouts[payout.payout_id] = payout
            payouts.append(payout)

            logger.info(
                f"Processed payout: affiliate={affiliate.affiliate_id}, "
                f"amount=${payout.net_amount}, status={payout.status.value}"
            )

        return payouts

    async def _process_payment(
        self,
        payout: Payout,
        affiliate: Affiliate
    ) -> dict[str, Any]:
        """Process actual payment using Stripe Connect."""
        stripe_api_key = os.getenv("STRIPE_API_KEY") or os.getenv("STRIPE_SECRET_KEY")

        method = payout.payment_method

        if method == "stripe":
            if not stripe_api_key:
                return {"success": False, "error": "Stripe API key not configured"}

            stripe.api_key = stripe_api_key

            try:
                # Expect 'stripe_account_id' in payout_details for Stripe Connect
                destination = affiliate.payout_details.get("stripe_account_id")

                if not destination:
                     return {"success": False, "error": "No Stripe Connect account ID found in payout_details"}

                # Create a Transfer to the connected account
                transfer = stripe.Transfer.create(
                    amount=int(payout.net_amount * 100), # cents
                    currency=payout.currency.lower(),
                    destination=destination,
                    description=f"Payout {payout.payout_id} for {affiliate.company_name}",
                    metadata={"payout_id": payout.payout_id}
                )

                return {
                    "success": True,
                    "reference": transfer.id,
                    "processor": "stripe",
                }
            except Exception as e:
                logger.error(f"Stripe payout error: {e}")
                return {"success": False, "error": str(e)}

        elif method == "paypal":
            logger.error("PayPal payouts are not implemented for affiliate %s", affiliate.affiliate_id)
            return {
                "success": False,
                "error": "PayPal payouts not implemented. Configure Stripe Connect or add PayPal integration.",
                "processor": "paypal",
            }
        elif method == "wire":
            logger.error("Wire payouts are not implemented for affiliate %s", affiliate.affiliate_id)
            return {
                "success": False,
                "error": "Wire payouts not implemented. Configure Stripe Connect or add manual payout workflow.",
                "processor": "wire",
            }
        else:
            return {
                "success": False,
                "error": f"Unsupported payment method: {method}",
            }

    # =========================================================================
    # PAYOUT QUEUE MANAGEMENT
    # =========================================================================

    async def queue_payout(
        self,
        affiliate_id: str,
        priority: int = 0,
        force: bool = False
    ) -> Optional[PayoutQueueItem]:
        """
        Queue a payout for processing.

        Args:
            affiliate_id: The affiliate's ID
            priority: Priority level (higher = processed first)
            force: If True, queue even if below minimum payout threshold

        Returns:
            The queue item if created, None if no payout needed
        """
        affiliate = self.affiliates.get(affiliate_id)
        if not affiliate:
            raise ValueError(f"Affiliate {affiliate_id} not found")

        if affiliate.status != AffiliateStatus.ACTIVE:
            raise ValueError(f"Affiliate {affiliate_id} is not active")

        # Check if already in queue
        existing = [
            q for q in self.payout_queue.values()
            if q.affiliate_id == affiliate_id and q.status in ("queued", "processing")
        ]
        if existing:
            logger.info(f"Affiliate {affiliate_id} already has pending payout in queue")
            return existing[0]

        # Get available commissions
        now = datetime.utcnow()
        available_commissions = [
            c for c in self.commissions.values()
            if (c.affiliate_id == affiliate_id and
                c.status == PayoutStatus.PENDING and
                c.available_date and
                c.available_date <= now)
        ]

        if not available_commissions:
            logger.info(f"No available commissions for affiliate {affiliate_id}")
            return None

        total_amount = sum(c.net_amount for c in available_commissions)

        # Check minimum payout threshold
        structure = self.commission_structures.get(affiliate.partner_type)
        min_payout = structure.minimum_payout if structure else Decimal("50")

        if total_amount < min_payout and not force:
            logger.info(
                f"Affiliate {affiliate_id} below minimum payout "
                f"(${total_amount} < ${min_payout})"
            )
            return None

        # Calculate fees
        fees = total_amount * Decimal("0.02")
        net_amount = total_amount - fees

        # Create queue item
        queue_item = PayoutQueueItem(
            affiliate_id=affiliate_id,
            amount=net_amount,
            currency="USD",
            payment_method=affiliate.payout_method,
            priority=priority,
            metadata={
                "gross_amount": str(total_amount),
                "fees": str(fees),
                "commission_count": len(available_commissions),
                "commission_ids": [c.commission_id for c in available_commissions],
            }
        )

        self.payout_queue[queue_item.queue_id] = queue_item

        logger.info(
            f"Queued payout for affiliate {affiliate_id}: "
            f"${net_amount} ({len(available_commissions)} commissions)"
        )

        return queue_item

    async def process_payout_queue(
        self,
        batch_size: int = 10
    ) -> list[dict[str, Any]]:
        """
        Process pending payouts from the queue.

        Args:
            batch_size: Maximum number of payouts to process in this batch

        Returns:
            List of processing results
        """
        results = []

        # Get pending queue items, sorted by priority (highest first) then queued time
        pending_items = sorted(
            [q for q in self.payout_queue.values() if q.status == "queued"],
            key=lambda x: (-x.priority, x.queued_at)
        )[:batch_size]

        for queue_item in pending_items:
            result = await self._process_queue_item(queue_item)
            results.append(result)

        return results

    async def _process_queue_item(
        self,
        queue_item: PayoutQueueItem
    ) -> dict[str, Any]:
        """Process a single payout queue item."""
        queue_item.status = "processing"
        queue_item.started_at = datetime.utcnow()
        queue_item.attempts += 1
        queue_item.last_attempt_at = datetime.utcnow()

        affiliate = self.affiliates.get(queue_item.affiliate_id)
        if not affiliate:
            queue_item.status = "failed"
            queue_item.error_message = "Affiliate not found"
            return {
                "queue_id": queue_item.queue_id,
                "status": "failed",
                "error": "Affiliate not found",
            }

        try:
            # Get commission IDs from metadata
            commission_ids = queue_item.metadata.get("commission_ids", [])
            commissions = [
                self.commissions[cid] for cid in commission_ids
                if cid in self.commissions
            ]

            # Create payout record
            payout = Payout(
                affiliate_id=queue_item.affiliate_id,
                gross_amount=Decimal(queue_item.metadata.get("gross_amount", "0")),
                fees=Decimal(queue_item.metadata.get("fees", "0")),
                net_amount=queue_item.amount,
                commission_ids=commission_ids,
                commission_count=len(commissions),
                payment_method=queue_item.payment_method,
                currency=queue_item.currency,
            )

            # Update commission status
            for commission in commissions:
                commission.status = PayoutStatus.PROCESSING
                commission.payout_id = payout.payout_id

            # Process the actual payment
            payment_result = await self._process_payment(payout, affiliate)

            if payment_result["success"]:
                payout.status = PayoutStatus.COMPLETED
                payout.payment_reference = payment_result.get("reference", "")
                payout.completed_date = datetime.utcnow()

                # Update commissions
                for commission in commissions:
                    commission.status = PayoutStatus.COMPLETED
                    commission.paid_date = datetime.utcnow()

                # Update affiliate
                affiliate.total_commissions_paid += payout.net_amount
                affiliate.pending_commission -= payout.gross_amount

                # Update queue item
                queue_item.status = "completed"
                queue_item.completed_at = datetime.utcnow()
                queue_item.payout_id = payout.payout_id

                self.payouts[payout.payout_id] = payout

                logger.info(
                    f"Payout completed: {payout.payout_id}, "
                    f"affiliate={affiliate.affiliate_id}, amount=${payout.net_amount}"
                )

                return {
                    "queue_id": queue_item.queue_id,
                    "payout_id": payout.payout_id,
                    "status": "completed",
                    "amount": str(payout.net_amount),
                    "reference": payout.payment_reference,
                }

            else:
                # Payment failed
                error_msg = payment_result.get("error", "Unknown error")

                # Reset commissions
                for commission in commissions:
                    commission.status = PayoutStatus.PENDING
                    commission.payout_id = None

                # Update queue item for retry
                if queue_item.attempts < queue_item.max_attempts:
                    queue_item.status = "queued"
                    queue_item.error_message = error_msg
                    # Exponential backoff for retries
                    retry_delay = timedelta(minutes=5 * (2 ** (queue_item.attempts - 1)))
                    queue_item.next_retry_at = datetime.utcnow() + retry_delay

                    logger.warning(
                        f"Payout failed (attempt {queue_item.attempts}), "
                        f"will retry at {queue_item.next_retry_at}: {error_msg}"
                    )
                else:
                    queue_item.status = "failed"
                    queue_item.error_message = f"Max retries exceeded: {error_msg}"

                    logger.error(
                        f"Payout permanently failed after {queue_item.attempts} attempts: {error_msg}"
                    )

                return {
                    "queue_id": queue_item.queue_id,
                    "status": queue_item.status,
                    "error": error_msg,
                    "attempts": queue_item.attempts,
                    "next_retry_at": queue_item.next_retry_at.isoformat() if queue_item.next_retry_at else None,
                }

        except Exception as e:
            logger.error(f"Error processing queue item {queue_item.queue_id}: {e}")
            queue_item.status = "failed"
            queue_item.error_message = str(e)

            return {
                "queue_id": queue_item.queue_id,
                "status": "failed",
                "error": str(e),
            }

    async def get_payout_queue_status(self) -> dict[str, Any]:
        """Get the current status of the payout queue."""
        queue_items = list(self.payout_queue.values())

        by_status = {}
        for item in queue_items:
            status = item.status
            if status not in by_status:
                by_status[status] = {"count": 0, "total_amount": Decimal("0")}
            by_status[status]["count"] += 1
            by_status[status]["total_amount"] += item.amount

        # Convert Decimal to string for JSON serialization
        for status_data in by_status.values():
            status_data["total_amount"] = str(status_data["total_amount"])

        return {
            "total_items": len(queue_items),
            "by_status": by_status,
            "pending_items": [
                {
                    "queue_id": q.queue_id,
                    "affiliate_id": q.affiliate_id,
                    "amount": str(q.amount),
                    "priority": q.priority,
                    "queued_at": q.queued_at.isoformat(),
                    "attempts": q.attempts,
                }
                for q in sorted(
                    [q for q in queue_items if q.status == "queued"],
                    key=lambda x: (-x.priority, x.queued_at)
                )[:20]  # Top 20
            ],
        }

    async def cancel_queued_payout(self, queue_id: str) -> bool:
        """Cancel a queued payout before it's processed."""
        queue_item = self.payout_queue.get(queue_id)
        if not queue_item:
            raise ValueError(f"Queue item {queue_id} not found")

        if queue_item.status not in ("queued",):
            raise ValueError(f"Cannot cancel payout in status: {queue_item.status}")

        queue_item.status = "cancelled"
        queue_item.completed_at = datetime.utcnow()

        logger.info(f"Cancelled queued payout: {queue_id}")
        return True

    async def retry_failed_payout(self, queue_id: str) -> PayoutQueueItem:
        """Retry a failed payout."""
        queue_item = self.payout_queue.get(queue_id)
        if not queue_item:
            raise ValueError(f"Queue item {queue_id} not found")

        if queue_item.status != "failed":
            raise ValueError(f"Can only retry failed payouts, current status: {queue_item.status}")

        # Reset for retry
        queue_item.status = "queued"
        queue_item.attempts = 0
        queue_item.error_message = ""
        queue_item.next_retry_at = None

        logger.info(f"Reset failed payout for retry: {queue_id}")
        return queue_item

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    async def get_affiliate_dashboard(
        self,
        affiliate_id: str,
        period_days: int = 30
    ) -> dict[str, Any]:
        """Get affiliate dashboard data."""
        affiliate = self.affiliates.get(affiliate_id)
        if not affiliate:
            raise ValueError(f"Affiliate {affiliate_id} not found")

        now = datetime.utcnow()
        period_start = now - timedelta(days=period_days)

        # Get referrals in period
        period_referrals = [
            r for r in self.referrals.values()
            if (r.affiliate_id == affiliate_id and
                r.click_date >= period_start)
        ]

        # Get conversions in period
        period_conversions = [
            r for r in period_referrals if r.converted
        ]

        # Get commissions in period
        period_commissions = [
            c for c in self.commissions.values()
            if (c.affiliate_id == affiliate_id and
                c.earned_date >= period_start)
        ]

        # Calculate metrics
        clicks = len(period_referrals)
        conversions = len(period_conversions)
        conversion_rate = (conversions / clicks * 100) if clicks > 0 else 0

        revenue = sum(r.order_value for r in period_conversions)
        commissions_earned = sum(c.net_amount for c in period_commissions)
        avg_order_value = revenue / conversions if conversions > 0 else Decimal("0")

        # EPC (Earnings Per Click)
        epc = commissions_earned / clicks if clicks > 0 else Decimal("0")

        return {
            "affiliate": {
                "id": affiliate.affiliate_id,
                "code": affiliate.affiliate_code,
                "tier": affiliate.tier.value,
                "status": affiliate.status.value,
            },
            "period": {
                "days": period_days,
                "start": period_start.isoformat(),
                "end": now.isoformat(),
            },
            "metrics": {
                "clicks": clicks,
                "conversions": conversions,
                "conversion_rate": round(conversion_rate, 2),
                "revenue": str(revenue),
                "commissions_earned": str(commissions_earned),
                "avg_order_value": str(avg_order_value),
                "epc": str(round(epc, 2)),
            },
            "lifetime": {
                "total_referrals": affiliate.total_referrals,
                "total_conversions": affiliate.total_conversions,
                "total_revenue": str(affiliate.total_revenue_generated),
                "total_earned": str(affiliate.total_commissions_earned),
                "total_paid": str(affiliate.total_commissions_paid),
                "pending": str(affiliate.pending_commission),
            },
            "tracking_links": affiliate.tracking_links,
        }

    async def get_program_analytics(
        self,
        period_days: int = 30
    ) -> dict[str, Any]:
        """Get overall affiliate program analytics."""
        now = datetime.utcnow()
        period_start = now - timedelta(days=period_days)

        # Affiliate stats
        total_affiliates = len(self.affiliates)
        active_affiliates = len([
            a for a in self.affiliates.values()
            if a.status == AffiliateStatus.ACTIVE
        ])

        # Referral stats
        period_referrals = [
            r for r in self.referrals.values()
            if r.click_date >= period_start
        ]
        period_conversions = [r for r in period_referrals if r.converted]

        total_clicks = len(period_referrals)
        total_conversions = len(period_conversions)
        total_revenue = sum(r.order_value for r in period_conversions)

        # Commission stats
        period_commissions = [
            c for c in self.commissions.values()
            if c.earned_date >= period_start
        ]
        total_commissions = sum(c.net_amount for c in period_commissions)

        # Top performers
        affiliate_performance = {}
        for referral in period_conversions:
            aid = referral.affiliate_id
            if aid not in affiliate_performance:
                affiliate_performance[aid] = {
                    "conversions": 0,
                    "revenue": Decimal("0"),
                }
            affiliate_performance[aid]["conversions"] += 1
            affiliate_performance[aid]["revenue"] += referral.order_value

        top_affiliates = sorted(
            affiliate_performance.items(),
            key=lambda x: x[1]["revenue"],
            reverse=True
        )[:10]

        # Tier distribution
        tier_distribution = {}
        for affiliate in self.affiliates.values():
            tier = affiliate.tier.value
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

        return {
            "period": {
                "days": period_days,
                "start": period_start.isoformat(),
                "end": now.isoformat(),
            },
            "affiliates": {
                "total": total_affiliates,
                "active": active_affiliates,
                "tier_distribution": tier_distribution,
            },
            "performance": {
                "clicks": total_clicks,
                "conversions": total_conversions,
                "conversion_rate": round(total_conversions / total_clicks * 100, 2) if total_clicks > 0 else 0,
                "revenue": str(total_revenue),
                "commissions": str(total_commissions),
                "avg_commission_rate": str(
                    round(total_commissions / total_revenue * 100, 2) if total_revenue > 0 else 0
                ) + "%",
            },
            "top_affiliates": [
                {
                    "affiliate_id": aid,
                    "name": self.affiliates[aid].contact_name if aid in self.affiliates else "Unknown",
                    "conversions": data["conversions"],
                    "revenue": str(data["revenue"]),
                }
                for aid, data in top_affiliates
            ],
        }

    # =========================================================================
    # CONTENT GENERATION
    # =========================================================================

    async def generate_affiliate_content(
        self,
        affiliate_id: str,
        content_type: str,
        product_info: dict[str, Any]
    ) -> PartnerContent:
        """Generate marketing content for an affiliate."""
        affiliate = self.affiliates.get(affiliate_id)
        if not affiliate:
            raise ValueError(f"Affiliate {affiliate_id} not found")

        # Generate based on type
        if content_type == "email":
            content = await self.content_generator.generate_email_template(
                affiliate=affiliate,
                product_name=product_info.get("name", ""),
                product_description=product_info.get("description", ""),
                target_audience=product_info.get("audience", ""),
                tone=product_info.get("tone", "professional"),
            )
        elif content_type == "social":
            contents = await self.content_generator.generate_social_posts(
                affiliate=affiliate,
                product_name=product_info.get("name", ""),
                key_benefits=product_info.get("benefits", []),
                platforms=product_info.get("platforms"),
            )
            # Return first one, store all
            for c in contents:
                c.tracking_link = affiliate.tracking_links.get("default", "")
                self.content[c.content_id] = c
            content = contents[0] if contents else None
        elif content_type == "blog":
            content = await self.content_generator.generate_blog_review(
                affiliate=affiliate,
                product_name=product_info.get("name", ""),
                product_features=product_info.get("features", []),
                target_keywords=product_info.get("keywords", []),
                word_count=product_info.get("word_count", 1500),
            )
        elif content_type == "comparison":
            content = await self.content_generator.generate_comparison_content(
                affiliate=affiliate,
                main_product=product_info.get("name", ""),
                competitors=product_info.get("competitors", []),
                comparison_criteria=product_info.get("criteria", []),
            )
        elif content_type == "video":
            content = await self.content_generator.generate_video_script(
                affiliate=affiliate,
                product_name=product_info.get("name", ""),
                video_type=product_info.get("video_type", "review"),
                duration_minutes=product_info.get("duration", 5),
            )
        else:
            raise ValueError(f"Unknown content type: {content_type}")

        if content:
            # Replace tracking link placeholder
            content.tracking_link = affiliate.tracking_links.get("default", "")
            content.content = content.content.replace(
                "{TRACKING_LINK}",
                content.tracking_link
            )
            self.content[content.content_id] = content

        return content

    # =========================================================================
    # LEADERBOARD AND COMMISSIONS
    # =========================================================================

    async def get_leaderboard(
        self,
        period: str = "month",
        limit: int = 10,
        tenant_id: str = None
    ) -> list[dict]:
        """
        Get affiliate leaderboard by performance.

        Args:
            period: Time period - "week", "month", "quarter", "year", "all"
            limit: Maximum number of affiliates to return
            tenant_id: Optional tenant filter

        Returns:
            List of affiliate performance dicts sorted by revenue
        """
        from datetime import timedelta

        now = datetime.utcnow()

        # Determine date cutoff based on period
        if period == "week":
            cutoff = now - timedelta(days=7)
        elif period == "month":
            cutoff = now - timedelta(days=30)
        elif period == "quarter":
            cutoff = now - timedelta(days=90)
        elif period == "year":
            cutoff = now - timedelta(days=365)
        else:  # "all"
            cutoff = datetime.min

        # Calculate performance for each affiliate
        leaderboard = []
        for affiliate_id, affiliate in self.affiliates.items():
            # Get conversions in period
            conversions = [
                r for r in self.referrals.values()
                if r.affiliate_id == affiliate_id and
                r.converted and
                r.clicked_at and r.clicked_at >= cutoff
            ]

            # Get commissions in period
            commissions = [
                c for c in self.commissions.values()
                if c.affiliate_id == affiliate_id and
                c.created_at >= cutoff
            ]

            total_revenue = sum(
                Decimal(str(r.order_value or 0)) for r in conversions
            )
            total_commissions = sum(c.net_amount for c in commissions)

            if total_revenue > 0 or len(conversions) > 0:
                leaderboard.append({
                    "rank": 0,  # Will be set after sorting
                    "affiliate_id": affiliate_id,
                    "name": affiliate.contact_name or affiliate.company_name or "Anonymous",
                    "tier": affiliate.tier.value if hasattr(affiliate.tier, 'value') else str(affiliate.tier),
                    "conversions": len(conversions),
                    "revenue": float(total_revenue),
                    "commissions": float(total_commissions),
                    "conversion_rate": round(
                        len(conversions) / max(affiliate.total_clicks, 1) * 100, 2
                    ),
                })

        # Sort by revenue descending
        leaderboard.sort(key=lambda x: x["revenue"], reverse=True)

        # Assign ranks and limit
        for i, entry in enumerate(leaderboard[:limit], 1):
            entry["rank"] = i

        return leaderboard[:limit]

    async def get_pending_commissions(
        self,
        tenant_id: str = None
    ) -> list[dict]:
        """
        Get all pending commissions awaiting payout.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of pending commission dicts
        """
        pending = []

        for commission in self.commissions.values():
            if commission.status == PayoutStatus.PENDING:
                affiliate = self.affiliates.get(commission.affiliate_id)
                pending.append({
                    "commission_id": commission.commission_id,
                    "affiliate_id": commission.affiliate_id,
                    "affiliate_name": affiliate.contact_name if affiliate else "Unknown",
                    "amount": float(commission.net_amount),
                    "commission_type": commission.commission_type.value if hasattr(commission.commission_type, 'value') else str(commission.commission_type),
                    "created_at": commission.created_at.isoformat() if commission.created_at else None,
                    "available_date": commission.available_date.isoformat() if commission.available_date else None,
                    "description": commission.description,
                })

        # Sort by created date
        pending.sort(key=lambda x: x["created_at"] or "", reverse=True)

        return pending

    async def get_dashboard(self, affiliate_id: str, tenant_id: str = None) -> dict:
        """Get affiliate dashboard data."""
        affiliate = self.affiliates.get(affiliate_id)
        if not affiliate:
            raise ValueError(f"Affiliate {affiliate_id} not found")

        return {
            "affiliate_id": affiliate.affiliate_id,
            "name": affiliate.contact_name,
            "email": affiliate.email,
            "company": affiliate.company_name,
            "referral_code": affiliate.affiliate_code,
            "partner_type": affiliate.partner_type.value,
            "tier": affiliate.tier.value,
            "status": affiliate.status.value,
            "commission_rate": float(affiliate.custom_commission_rate) if affiliate.custom_commission_rate else 0.20,
            "tracking_link": affiliate.tracking_links.get("default", ""),
            "metrics": {
                "total_referrals": affiliate.total_referrals,
                "total_conversions": affiliate.total_conversions,
                "conversion_rate": round(affiliate.total_conversions / max(affiliate.total_referrals, 1) * 100, 2),
                "total_revenue_generated": float(affiliate.total_revenue_generated),
                "total_commissions_earned": float(affiliate.total_commissions_earned),
                "total_commissions_paid": float(affiliate.total_commissions_paid),
                "pending_commission": float(affiliate.pending_commission),
            },
            "dates": {
                "joined": affiliate.joined_date.isoformat() if affiliate.joined_date else None,
                "last_activity": affiliate.last_activity_date.isoformat() if affiliate.last_activity_date else None,
                "next_payout": affiliate.next_payout_date.isoformat() if affiliate.next_payout_date else None,
            },
        }

    async def get_stats(self, affiliate_id: str, period: str = "30d", tenant_id: str = None) -> dict:
        """Get affiliate statistics."""
        affiliate = self.affiliates.get(affiliate_id)
        if not affiliate:
            raise ValueError(f"Affiliate {affiliate_id} not found")

        return {
            "affiliate_id": affiliate.affiliate_id,
            "referral_code": affiliate.affiliate_code,
            "total_referrals": affiliate.total_referrals,
            "total_conversions": affiliate.total_conversions,
            "conversion_rate": round(affiliate.total_conversions / max(affiliate.total_referrals, 1) * 100, 2),
            "total_revenue": float(affiliate.total_revenue_generated),
            "total_earned": float(affiliate.total_commissions_earned),
            "total_paid": float(affiliate.total_commissions_paid),
            "pending": float(affiliate.pending_commission),
            "tier": affiliate.tier.value,
            "status": affiliate.status.value,
        }

    # =========================================================================
    # UTILITIES
    # =========================================================================

    async def _send_welcome_email(self, affiliate: Affiliate):
        """Send welcome email to new affiliate."""
        subject = f"Welcome to the Partner Program, {affiliate.contact_name}!"
        body = f"""
        <h1>Welcome to the Team!</h1>
        <p>Hi {affiliate.contact_name},</p>
        <p>We are thrilled to have you on board as a {affiliate.partner_type.value}.</p>
        <p>Your affiliate code is: <strong>{affiliate.affiliate_code}</strong></p>
        <p><a href="{affiliate.tracking_links.get('default')}">Access your dashboard here</a></p>
        """
        success, msg = send_email(affiliate.email, subject, body)
        if not success:
            logger.error(f"Failed to send welcome email to {affiliate.email}: {msg}")
        else:
            logger.info(f"Sent welcome email to {affiliate.email}")

    def generate_tracking_link(
        self,
        affiliate: Affiliate,
        campaign: str = "default",
        landing_page: str = None
    ) -> str:
        """Generate a tracking link for an affiliate."""
        base_url = landing_page or os.getenv("BASE_URL", "https://app.example.com")

        # Build tracking URL
        params = {
            "ref": affiliate.affiliate_code,
            "utm_source": "affiliate",
            "utm_medium": affiliate.partner_type.value,
            "utm_campaign": campaign,
        }

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        link = f"{base_url}?{query_string}"

        # Store for affiliate
        affiliate.tracking_links[campaign] = link

        return link

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
        secret: str
    ) -> bool:
        """Verify webhook signature for security."""
        expected = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected, signature)


# =============================================================================
# FASTAPI ROUTER
# =============================================================================

def create_affiliate_router():
    """Create FastAPI router for affiliate endpoints."""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    router = APIRouter(prefix="/affiliate", tags=["Affiliate"])
    pipeline = AffiliatePartnershipPipeline()

    class AffiliateRegistration(BaseModel):
        email: str
        contact_name: str
        company_name: str = ""
        partner_type: str = "affiliate"
        website: str = ""
        parent_code: str = None

    class ClickTracking(BaseModel):
        tracking_code: str
        visitor_id: str
        ip_address: str
        user_agent: str
        landing_page: str
        referrer_url: str = ""
        utm_params: dict = None

    class ConversionTracking(BaseModel):
        visitor_id: str = None
        referral_id: str = None
        customer_id: str
        order_id: str
        order_value: float
        product_id: str = ""
        product_name: str = ""

    @router.post("/register")
    async def register_affiliate(data: AffiliateRegistration):
        """Register a new affiliate."""
        try:
            affiliate = await pipeline.register_affiliate(
                email=data.email,
                contact_name=data.contact_name,
                company_name=data.company_name,
                partner_type=PartnerType(data.partner_type),
                website=data.website,
                parent_affiliate_code=data.parent_code,
            )
            return {
                "success": True,
                "affiliate_id": affiliate.affiliate_id,
                "code": affiliate.affiliate_code,
                "status": affiliate.status.value,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.post("/track/click")
    async def track_click(data: ClickTracking):
        """Track an affiliate click."""
        try:
            referral, fraud_signals = await pipeline.track_click(
                tracking_code=data.tracking_code,
                visitor_id=data.visitor_id,
                ip_address=data.ip_address,
                user_agent=data.user_agent,
                landing_page=data.landing_page,
                referrer_url=data.referrer_url,
                utm_params=data.utm_params,
            )
            return {
                "success": True,
                "referral_id": referral.referral_id,
                "cookie_expires": referral.cookie_expires.isoformat() if referral.cookie_expires else None,
                "fraud_signals": len(fraud_signals),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.post("/track/conversion")
    async def track_conversion(data: ConversionTracking):
        """Track a conversion."""
        commission = await pipeline.track_conversion(
            visitor_id=data.visitor_id,
            referral_id=data.referral_id,
            customer_id=data.customer_id,
            order_id=data.order_id,
            order_value=Decimal(str(data.order_value)),
            product_id=data.product_id,
            product_name=data.product_name,
        )

        if commission:
            return {
                "success": True,
                "commission_id": commission.commission_id,
                "amount": str(commission.net_amount),
                "status": commission.status.value,
            }
        return {
            "success": False,
            "message": "No valid referral found for this conversion",
        }

    @router.get("/dashboard/{affiliate_id}")
    async def get_dashboard(affiliate_id: str, days: int = 30):
        """Get affiliate dashboard data."""
        try:
            return await pipeline.get_affiliate_dashboard(affiliate_id, days)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @router.get("/analytics")
    async def get_analytics(days: int = 30):
        """Get program-wide analytics."""
        return await pipeline.get_program_analytics(days)

    @router.post("/content/{affiliate_id}")
    async def generate_content(
        affiliate_id: str,
        content_type: str,
        product_info: dict
    ):
        """Generate marketing content for affiliate."""
        try:
            content = await pipeline.generate_affiliate_content(
                affiliate_id=affiliate_id,
                content_type=content_type,
                product_info=product_info,
            )
            return {
                "success": True,
                "content_id": content.content_id,
                "content": content.content,
                "tracking_link": content.tracking_link,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.post("/payouts/process")
    async def process_payouts(affiliate_ids: list[str] = None):
        """Process pending payouts."""
        payouts = await pipeline.process_payouts(affiliate_ids)
        return {
            "success": True,
            "payouts_processed": len(payouts),
            "payouts": [
                {
                    "payout_id": p.payout_id,
                    "affiliate_id": p.affiliate_id,
                    "amount": str(p.net_amount),
                    "status": p.status.value,
                }
                for p in payouts
            ],
        }

    return router


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Demo the affiliate pipeline."""
    print("=" * 70)
    print("AFFILIATE & PARTNERSHIP PIPELINE DEMO")
    print("=" * 70)

    pipeline = AffiliatePartnershipPipeline()

    # Register affiliates
    print("\n1. Registering affiliates...")

    affiliate1 = await pipeline.register_affiliate(
        email="john@roofingblog.com",
        contact_name="John Smith",
        company_name="Roofing Blog Pro",
        partner_type=PartnerType.AFFILIATE,
        website="https://roofingblog.com",
    )
    print(f"   Created: {affiliate1.affiliate_code} (pending approval)")

    # Approve
    await pipeline.approve_affiliate(affiliate1.affiliate_id)
    print(f"   Approved: {affiliate1.affiliate_code}")

    # Register tier-2 affiliate
    affiliate2 = await pipeline.register_affiliate(
        email="mary@homeimprovement.com",
        contact_name="Mary Johnson",
        company_name="Home Improvement Tips",
        partner_type=PartnerType.INFLUENCER,
        website="https://homeimprovement.com",
        parent_affiliate_code=affiliate1.affiliate_code,
    )
    await pipeline.approve_affiliate(affiliate2.affiliate_id)
    print(f"   Created tier-2: {affiliate2.affiliate_code} (parent: {affiliate1.affiliate_code})")

    # Track clicks
    print("\n2. Tracking clicks...")

    referral1, signals1 = await pipeline.track_click(
        tracking_code=affiliate1.affiliate_code,
        visitor_id="visitor_001",
        ip_address="203.0.113.42",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        landing_page="https://app.example.com/pricing",
        referrer_url="https://roofingblog.com/best-roofing-software",
        utm_params={"utm_campaign": "roofing-review"},
    )
    print(f"   Click tracked: {referral1.referral_id}")
    print(f"   Fraud signals: {len(signals1)}")

    # Track conversion
    print("\n3. Tracking conversion...")

    commission = await pipeline.track_conversion(
        visitor_id="visitor_001",
        customer_id="cust_123456",
        order_id="order_789",
        order_value=Decimal("299.00"),
        product_id="prod_premium",
        product_name="Premium Plan Annual",
    )
    print(f"   Commission: ${commission.net_amount} ({commission.status.value})")

    # Get dashboard
    print("\n4. Affiliate Dashboard:")
    dashboard = await pipeline.get_affiliate_dashboard(affiliate1.affiliate_id)
    print(f"   Clicks: {dashboard['metrics']['clicks']}")
    print(f"   Conversions: {dashboard['metrics']['conversions']}")
    print(f"   Revenue: ${dashboard['metrics']['revenue']}")
    print(f"   Commissions: ${dashboard['metrics']['commissions_earned']}")

    # Program analytics
    print("\n5. Program Analytics:")
    analytics = await pipeline.get_program_analytics()
    print(f"   Total affiliates: {analytics['affiliates']['total']}")
    print(f"   Active: {analytics['affiliates']['active']}")
    print(f"   Total revenue: ${analytics['performance']['revenue']}")

    # Generate content
    print("\n6. Generating affiliate content...")
    content = await pipeline.generate_affiliate_content(
        affiliate_id=affiliate1.affiliate_id,
        content_type="email",
        product_info={
            "name": "WeatherCraft ERP",
            "description": "AI-powered roofing business management software",
            "audience": "Roofing contractors and business owners",
            "tone": "professional",
        },
    )
    print(f"   Generated: {content.content_type}")
    print(f"   Preview: {content.content[:200]}...")

    print("\n" + "=" * 70)
    print("AFFILIATE PIPELINE DEMO COMPLETE")
    print("=" * 70)


# Singleton instance for API usage
_affiliate_pipeline_instance: Optional[AffiliatePartnershipPipeline] = None

def get_affiliate_pipeline() -> AffiliatePartnershipPipeline:
    """Get or create the singleton affiliate pipeline instance."""
    global _affiliate_pipeline_instance
    if _affiliate_pipeline_instance is None:
        _affiliate_pipeline_instance = AffiliatePartnershipPipeline()
    return _affiliate_pipeline_instance


if __name__ == "__main__":
    asyncio.run(main())

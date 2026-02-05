#!/usr/bin/env python3
"""
Revenue Intelligence System - Complete AI OS Subsystem
=======================================================
The CENTRAL source of truth for all revenue streams, products,
social presence, and business intelligence.

This system is accessed by ALL AI agents to maintain complete
awareness of business state at all times.

Features:
- Product inventory across all platforms
- Revenue tracking and analytics
- Social media presence tracking
- Email/marketing pipeline status
- Automation health monitoring
- Real-time business intelligence
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field, asdict
from enum import Enum

import psycopg2
from psycopg2.extras import RealDictCursor
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Platform(Enum):
    GUMROAD = "gumroad"
    MYROOFGENIUS = "myroofgenius"
    BRAINSTACK = "brainstack_studio"
    STRIPE = "stripe"


class ProductType(Enum):
    CODE_KIT = "code_kit"
    PROMPT_PACK = "prompt_pack"
    AUTOMATION = "automation"
    TEMPLATE = "template"
    SUBSCRIPTION = "subscription"
    FREE = "free"


class RevenueStreamType(Enum):
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    FREEMIUM = "freemium"


@dataclass
class Product:
    """Product definition"""
    id: str
    name: str
    platform: str
    price: float
    product_type: str
    url: str
    status: str = "active"
    description: str = ""
    features: List[str] = field(default_factory=list)
    created_at: str = ""
    sales_count: int = 0
    revenue: float = 0


@dataclass
class SocialPresence:
    """Social media presence"""
    platform: str
    url: str
    name: str
    status: str = "active"
    followers: int = 0
    posts: int = 0
    last_post: str = ""


@dataclass
class RevenueStream:
    """Revenue stream definition"""
    id: str
    name: str
    platform: str
    stream_type: str
    current_revenue: float = 0
    mrr: float = 0
    products: List[str] = field(default_factory=list)
    status: str = "active"


@dataclass
class BusinessState:
    """Complete business state snapshot"""
    timestamp: str
    products: Dict[str, List[Product]] = field(default_factory=dict)
    revenue_streams: List[RevenueStream] = field(default_factory=list)
    social_presence: List[SocialPresence] = field(default_factory=list)
    automations: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _get_db_config():
    """Get database configuration."""
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        match = re.match(
            r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)',
            database_url
        )
        if match:
            return {
                'host': match.group(3),
                'database': match.group(5),
                'user': match.group(1),
                'password': match.group(2),
                'port': int(match.group(4))
            }
    return {
        "host": os.getenv("DB_HOST", ""),
        "database": os.getenv("DB_NAME", ""),
        "user": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
        "port": int(os.getenv("DB_PORT", "5432"))
    }


def _get_db_connection(**kwargs):
    """Get database connection."""
    db_config = _get_db_config()
    db_config.update(kwargs)
    return psycopg2.connect(**db_config)


class RevenueIntelligenceSystem:
    """
    Central Revenue Intelligence System

    Maintains complete awareness of all revenue streams, products,
    social presence, and business metrics. Accessed by all AI agents.
    """

    _tables_ensured = False

    # ==================== STATIC PRODUCT INVENTORY ====================
    # These are the REAL products that exist

    GUMROAD_PRODUCTS = [
        Product(
            id="HJHMSM",
            name="MCP Server Starter Kit",
            platform="gumroad",
            price=97,
            product_type="code_kit",
            url="https://woodworthia.gumroad.com/l/hjhmsm",
            status="active",
            description="Build AI tool integrations fast with MCP Server patterns",
            features=["Full source code", "AI tool patterns", "Documentation", "Lifetime updates"]
        ),
        Product(
            id="GSAAVB",
            name="AI Orchestration Framework",
            platform="gumroad",
            price=147,
            product_type="code_kit",
            url="https://woodworthia.gumroad.com/l/gsaavb",
            status="active",
            description="Multi-LLM smart routing and orchestration system",
            features=["Multi-LLM routing", "Production-ready", "Full docs", "Priority support"]
        ),
        Product(
            id="VJXCEW",
            name="SaaS Automation Scripts",
            platform="gumroad",
            price=67,
            product_type="code_kit",
            url="https://woodworthia.gumroad.com/l/vjxcew",
            status="active"
        ),
        Product(
            id="UPSYKR",
            name="Command Center UI Kit",
            platform="gumroad",
            price=149,
            product_type="code_kit",
            url="https://woodworthia.gumroad.com/l/upsykr",
            status="active"
        ),
        Product(
            id="XGFKP",
            name="AI Prompt Engineering Pack",
            platform="gumroad",
            price=47,
            product_type="prompt_pack",
            url="https://woodworthia.gumroad.com/l/xgfkp",
            status="active"
        ),
        Product(
            id="CAWVO",
            name="Business Automation Toolkit",
            platform="gumroad",
            price=49,
            product_type="prompt_pack",
            url="https://woodworthia.gumroad.com/l/cawvo",
            status="active"
        ),
        Product(
            id="GR-ERP-START",
            name="SaaS ERP Starter Kit",
            platform="gumroad",
            price=197,
            product_type="code_kit",
            url="https://woodworthia.gumroad.com/l/gr-erp-start",
            status="active",
            description="Multi-tenant SaaS foundation with auth, CRM, jobs, invoicing",
            features=["Next.js 14", "Supabase", "TypeScript", "Multi-tenant"]
        ),
        Product(
            id="GR-CONTENT",
            name="AI Content Production Pipeline",
            platform="gumroad",
            price=347,
            product_type="automation",
            url="https://woodworthia.gumroad.com/l/gr-content",
            status="active",
            description="Scale content 10x with multi-stage AI pipeline",
            features=["Python", "FastAPI", "OpenAI/Anthropic", "SEO optimization"]
        ),
        Product(
            id="GR-ONBOARD",
            name="Intelligent Client Onboarding",
            platform="gumroad",
            price=297,
            product_type="automation",
            url="https://woodworthia.gumroad.com/l/gr-onboard",
            status="active",
            features=["Next.js", "TypeScript", "Make.com integration"]
        ),
        Product(
            id="GR-PMCMD",
            name="AI Project Command Center",
            platform="gumroad",
            price=197,
            product_type="template",
            url="https://woodworthia.gumroad.com/l/gr-pmcmd",
            status="active",
            description="BrainOps Knowledge Base with linked databases and automations",
            features=["Knowledge Base", "Make/Zapier", "AI-Ready", "Executive dashboard"]
        ),
    ]

    MRG_PRODUCTS = [
        Product(
            id="mrg-starter",
            name="Starter",
            platform="myroofgenius",
            price=49,
            product_type="subscription",
            url="https://myroofgenius.com/pricing",
            status="ready",
            features=["1-3 users", "2-10 jobs/month", "Basic analysis"]
        ),
        Product(
            id="mrg-professional",
            name="Professional",
            platform="myroofgenius",
            price=99,
            product_type="subscription",
            url="https://myroofgenius.com/pricing",
            status="ready",
            features=["Up to 10 users", "10-30 jobs/month", "Advanced analytics"]
        ),
        Product(
            id="mrg-enterprise",
            name="Enterprise",
            platform="myroofgenius",
            price=199,
            product_type="subscription",
            url="https://myroofgenius.com/pricing",
            status="ready",
            features=["Unlimited users", "30+ jobs/month", "Full features"]
        ),
    ]

    BSS_PRODUCTS = [
        Product(
            id="bss-playground",
            name="AI Playground",
            platform="brainstack_studio",
            price=0,
            product_type="free",
            url="https://brainstackstudio.com/playground",
            status="active",
            features=["Claude, GPT, Gemini", "Local storage", "Basic highlighting"]
        ),
    ]

    SOCIAL_PRESENCE = [
        SocialPresence(
            platform="linkedin",
            url="https://www.linkedin.com/company/brainops-ai-os",
            name="BrainOps AI OS",
            status="active"
        ),
        SocialPresence(
            platform="facebook",
            url="https://www.facebook.com/brainopsai",
            name="BrainOps",
            status="active"
        ),
        SocialPresence(
            platform="gumroad",
            url="https://woodworthia.gumroad.com",
            name="Woodworthia",
            status="active"
        ),
    ]

    WEBSITES = [
        {"name": "BrainStack Studio", "url": "https://brainstackstudio.com", "status": "active"},
        {"name": "MyRoofGenius", "url": "https://myroofgenius.com", "status": "active"},
        {"name": "Weathercraft ERP", "url": "https://weathercraft-erp.vercel.app", "status": "active", "note": "internal"},
    ]

    def __init__(self):
        self.system_id = "revenue-intelligence-v1"
        logger.info("Revenue Intelligence System initialized")

    def _ensure_tables(self):
        """Ensure revenue tracking tables exist."""
        if RevenueIntelligenceSystem._tables_ensured:
            return
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_business_state (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    snapshot_type VARCHAR(50) NOT NULL,
                    state_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_business_state_type ON ai_business_state(snapshot_type);
                CREATE INDEX IF NOT EXISTS idx_business_state_created ON ai_business_state(created_at DESC);

                CREATE TABLE IF NOT EXISTS ai_revenue_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type VARCHAR(50) NOT NULL,
                    platform VARCHAR(50) NOT NULL,
                    amount DECIMAL(10,2),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_revenue_events_type ON ai_revenue_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_revenue_events_platform ON ai_revenue_events(platform);
            """)

            conn.commit()
            cursor.close()
            conn.close()
            RevenueIntelligenceSystem._tables_ensured = True
            logger.info("Revenue tracking tables ensured")
        except Exception as e:
            logger.warning(f"Could not ensure tables: {e}")

    # ==================== CORE INTELLIGENCE METHODS ====================

    def get_all_products(self) -> Dict[str, List[Dict]]:
        """Get complete product inventory across all platforms."""
        return {
            "gumroad": [asdict(p) for p in self.GUMROAD_PRODUCTS],
            "myroofgenius": [asdict(p) for p in self.MRG_PRODUCTS],
            "brainstack_studio": [asdict(p) for p in self.BSS_PRODUCTS]
        }

    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """Find a product by ID across all platforms."""
        all_products = self.GUMROAD_PRODUCTS + self.MRG_PRODUCTS + self.BSS_PRODUCTS
        for product in all_products:
            if product.id == product_id:
                return asdict(product)
        return None

    def get_social_presence(self) -> List[Dict]:
        """Get all social media presence."""
        return [asdict(s) for s in self.SOCIAL_PRESENCE]

    def get_websites(self) -> List[Dict]:
        """Get all websites."""
        return self.WEBSITES

    async def get_live_revenue(self) -> Dict[str, Any]:
        """Get live revenue data from database."""
        self._ensure_tables()

        try:
            conn = _get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Gumroad revenue
            cursor.execute("""
                SELECT
                    COUNT(*) as total_sales,
                    COALESCE(SUM(price::numeric), 0) as total_revenue,
                    MAX(sale_timestamp) as last_sale
                FROM gumroad_sales
                WHERE is_test = false OR is_test IS NULL
            """)
            gumroad = cursor.fetchone() or {}

            # MRG subscriptions
            cursor.execute("""
                SELECT
                    COUNT(*) as active_subs,
                    COALESCE(SUM(
                        CASE
                            WHEN billing_cycle = 'monthly' THEN amount
                            WHEN billing_cycle = 'annual' THEN amount / 12
                            ELSE 0
                        END
                    ), 0) as mrr
                FROM mrg_subscriptions
                WHERE status = 'active'
            """)
            mrg = cursor.fetchone() or {}

            # Recent sales (last 30 days)
            cursor.execute("""
                SELECT
                    COUNT(*) as count,
                    COALESCE(SUM(price::numeric), 0) as revenue
                FROM gumroad_sales
                WHERE (is_test = false OR is_test IS NULL)
                AND sale_timestamp > NOW() - INTERVAL '30 days'
            """)
            last_30 = cursor.fetchone() or {}

            cursor.close()
            conn.close()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gumroad": {
                    "lifetime_sales": gumroad.get("total_sales", 0),
                    "lifetime_revenue": float(gumroad.get("total_revenue", 0)),
                    "last_sale": str(gumroad.get("last_sale")) if gumroad.get("last_sale") else None,
                    "last_30_days": {
                        "sales": last_30.get("count", 0),
                        "revenue": float(last_30.get("revenue", 0))
                    }
                },
                "myroofgenius": {
                    "active_subscriptions": mrg.get("active_subs", 0),
                    "mrr": float(mrg.get("mrr", 0)),
                    "arr": float(mrg.get("mrr", 0)) * 12
                },
                "totals": {
                    "lifetime_revenue": float(gumroad.get("total_revenue", 0)),
                    "current_mrr": float(mrg.get("mrr", 0)),
                    "projected_arr": float(mrg.get("mrr", 0)) * 12 + float(last_30.get("revenue", 0)) * 12
                }
            }

        except Exception as e:
            logger.error(f"Failed to get live revenue: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

    async def get_automation_status(self) -> Dict[str, Any]:
        """Get status of all revenue-related automations."""
        self._ensure_tables()

        try:
            conn = _get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Email automation status
            cursor.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'queued') as queued,
                    COUNT(*) FILTER (WHERE status = 'sent') as sent,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed,
                    MAX(CASE WHEN status = 'sent' THEN sent_at END) as last_sent
                FROM ai_email_queue
                WHERE created_at > NOW() - INTERVAL '7 days'
            """)
            email = cursor.fetchone() or {}

            # Content generation status
            cursor.execute("""
                SELECT
                    COUNT(*) as total_content,
                    COUNT(*) FILTER (WHERE status = 'published') as published,
                    MAX(created_at) as last_created
                FROM content_posts
                WHERE created_at > NOW() - INTERVAL '30 days'
            """)
            content = cursor.fetchone() or {}

            # Lead pipeline status
            cursor.execute("""
                SELECT
                    COUNT(*) as total_leads,
                    COUNT(*) FILTER (WHERE stage = 'WON') as won,
                    COUNT(*) FILTER (WHERE stage = 'QUALIFIED') as qualified,
                    COALESCE(SUM(CASE WHEN stage = 'WON' THEN estimated_value ELSE 0 END), 0) as won_value
                FROM revenue_leads
                WHERE created_at > NOW() - INTERVAL '90 days'
            """)
            leads = cursor.fetchone() or {}

            cursor.close()
            conn.close()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "email_automation": {
                    "queued": email.get("queued", 0),
                    "sent_last_7_days": email.get("sent", 0),
                    "failed": email.get("failed", 0),
                    "last_sent": str(email.get("last_sent")) if email.get("last_sent") else None,
                    "status": "healthy" if email.get("failed", 0) < 5 else "degraded"
                },
                "content_pipeline": {
                    "total_created_30d": content.get("total_content", 0),
                    "published": content.get("published", 0),
                    "last_created": str(content.get("last_created")) if content.get("last_created") else None,
                    "status": "active" if content.get("total_content", 0) > 0 else "idle"
                },
                "lead_pipeline": {
                    "total_leads_90d": leads.get("total_leads", 0),
                    "qualified": leads.get("qualified", 0),
                    "won": leads.get("won", 0),
                    "won_value": float(leads.get("won_value", 0)),
                    "status": "active" if leads.get("total_leads", 0) > 0 else "idle"
                }
            }

        except Exception as e:
            logger.error(f"Failed to get automation status: {e}")
            return {"error": str(e)}

    async def get_complete_business_state(self) -> Dict[str, Any]:
        """
        Get COMPLETE business state snapshot.
        This is the PRIMARY method for AI agents to understand business state.
        """
        self._ensure_tables()

        products = self.get_all_products()
        social = self.get_social_presence()
        websites = self.get_websites()
        revenue = await self.get_live_revenue()
        automations = await self.get_automation_status()

        # Calculate totals
        gumroad_product_count = len(products.get("gumroad", []))
        gumroad_product_value = sum(p["price"] for p in products.get("gumroad", []))

        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_version": self.system_id,

            "products": {
                "platforms": products,
                "summary": {
                    "gumroad": {
                        "count": gumroad_product_count,
                        "total_catalog_value": gumroad_product_value,
                        "pricing_model": "one_time"
                    },
                    "myroofgenius": {
                        "count": len(products.get("myroofgenius", [])),
                        "pricing_model": "subscription",
                        "price_range": "$49-$199/mo"
                    },
                    "brainstack_studio": {
                        "count": len(products.get("brainstack_studio", [])),
                        "pricing_model": "freemium"
                    }
                }
            },

            "revenue": revenue,

            "social_presence": {
                "platforms": social,
                "websites": websites
            },

            "automations": automations,

            "action_items": self._generate_action_items(revenue, automations),

            "warnings": [
                "ERP customer/job/invoice data is DEMO - not real revenue",
                "MRG has 0 active subscribers - needs customer acquisition"
            ]
        }

        # Save snapshot
        await self._save_state_snapshot(state)

        return state

    def _generate_action_items(self, revenue: Dict, automations: Dict) -> List[str]:
        """Generate actionable recommendations based on current state."""
        items = []

        # Revenue-based recommendations
        if revenue.get("myroofgenius", {}).get("active_subscriptions", 0) == 0:
            items.append("PRIORITY: Activate MRG customer acquisition - 0 subscribers")

        last_30 = revenue.get("gumroad", {}).get("last_30_days", {})
        if last_30.get("sales", 0) == 0:
            items.append("No Gumroad sales in 30 days - increase marketing")

        # Automation-based recommendations
        email_status = automations.get("email_automation", {})
        if email_status.get("failed", 0) > 0:
            items.append(f"Fix {email_status['failed']} failed emails in queue")

        content_status = automations.get("content_pipeline", {})
        if content_status.get("status") == "idle":
            items.append("Content pipeline idle - schedule content generation")

        return items

    async def _save_state_snapshot(self, state: Dict) -> str:
        """Save business state snapshot to database."""
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_business_state (snapshot_type, state_data)
                VALUES ('full_state', %s)
                RETURNING id
            """, (json.dumps(state),))

            snapshot_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

            return str(snapshot_id)

        except Exception as e:
            logger.error(f"Failed to save state snapshot: {e}")
            return ""

    async def record_revenue_event(
        self,
        event_type: str,
        platform: str,
        amount: float = 0,
        metadata: Dict = None
    ) -> str:
        """Record a revenue event for tracking."""
        self._ensure_tables()

        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_revenue_events (event_type, platform, amount, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (event_type, platform, amount, json.dumps(metadata or {})))

            event_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Revenue event recorded: {event_type} on {platform} (${amount})")
            return str(event_id)

        except Exception as e:
            logger.error(f"Failed to record revenue event: {e}")
            return ""

    # ==================== STORE TO BRAIN ====================

    async def store_to_brain(self) -> Dict[str, Any]:
        """
        Store complete business state to the AI brain knowledge system.
        This ensures ALL AI agents have access to current business state.
        """
        state = await self.get_complete_business_state()

        try:
            # Store to unified brain
            conn = _get_db_connection()
            cursor = conn.cursor()

            # Store as brain knowledge
            brain_entries = [
                {
                    "key": "business_state_snapshot",
                    "value": state,
                    "category": "revenue_intelligence"
                },
                {
                    "key": "product_inventory",
                    "value": self.get_all_products(),
                    "category": "revenue_intelligence"
                },
                {
                    "key": "social_presence",
                    "value": {
                        "platforms": self.get_social_presence(),
                        "websites": self.get_websites()
                    },
                    "category": "revenue_intelligence"
                },
                {
                    "key": "revenue_metrics",
                    "value": state.get("revenue", {}),
                    "category": "revenue_intelligence"
                }
            ]

            for entry in brain_entries:
                cursor.execute("""
                    INSERT INTO unified_brain (key, value, category, priority, last_updated)
                    VALUES (%s, %s, %s, 5, NOW())
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        category = EXCLUDED.category,
                        last_updated = NOW()
                """, (entry["key"], json.dumps(entry["value"]), entry["category"]))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("Business state stored to brain successfully")

            return {
                "status": "success",
                "entries_stored": len(brain_entries),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to store to brain: {e}")
            return {"status": "error", "error": str(e)}


# ==================== SINGLETON ACCESS ====================

_revenue_system: Optional[RevenueIntelligenceSystem] = None


def get_revenue_system() -> RevenueIntelligenceSystem:
    """Get singleton instance of Revenue Intelligence System."""
    global _revenue_system
    if _revenue_system is None:
        _revenue_system = RevenueIntelligenceSystem()
    return _revenue_system


# ==================== CONVENIENCE FUNCTIONS ====================

async def get_business_state() -> Dict[str, Any]:
    """Quick access to complete business state."""
    system = get_revenue_system()
    return await system.get_complete_business_state()


async def get_revenue() -> Dict[str, Any]:
    """Quick access to live revenue."""
    system = get_revenue_system()
    return await system.get_live_revenue()


async def sync_to_brain() -> Dict[str, Any]:
    """Sync current state to brain for all AI agents."""
    system = get_revenue_system()
    return await system.store_to_brain()


if __name__ == "__main__":
    async def test():
        system = RevenueIntelligenceSystem()

        print("\n=== PRODUCT INVENTORY ===")
        products = system.get_all_products()
        for platform, prods in products.items():
            print(f"\n{platform.upper()}: {len(prods)} products")
            for p in prods:
                print(f"  - {p['name']}: ${p['price']}")

        print("\n=== LIVE REVENUE ===")
        revenue = await system.get_live_revenue()
        print(json.dumps(revenue, indent=2))

        print("\n=== COMPLETE STATE ===")
        state = await system.get_complete_business_state()
        print(f"Action items: {state.get('action_items', [])}")

    asyncio.run(test())

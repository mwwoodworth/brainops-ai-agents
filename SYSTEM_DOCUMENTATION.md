# BRAINOPS AI OS - COMPREHENSIVE SYSTEM DOCUMENTATION

## Version: 10.0.0 | Last Updated: 2025-12-27

---

## TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Core Pipelines](#core-pipelines)
3. [Revenue Systems](#revenue-systems)
4. [Product Generation](#product-generation)
5. [Knowledge Management](#knowledge-management)
6. [Integration Guide](#integration-guide)
7. [API Reference](#api-reference)
8. [Deployment](#deployment)

---

## SYSTEM OVERVIEW

The BrainOps AI OS is a comprehensive AI-native operating system designed for autonomous business operations. It combines multiple AI models (Claude, GPT, Gemini) with sophisticated pipelines for revenue generation, product creation, knowledge management, and operational automation.

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│         FastAPI Routers | CLI Interfaces | Dashboards       │
├─────────────────────────────────────────────────────────────┤
│                   ORCHESTRATION LAYER                        │
│  AUREA Master Orchestrator | Agent Scheduler | Event Bus    │
├─────────────────────────────────────────────────────────────┤
│                    PIPELINE LAYER                            │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  │
│  │ Revenue   │ │ Product   │ │ Affiliate │ │ Knowledge │  │
│  │ Pipeline  │ │ Generator │ │ Pipeline  │ │ Base      │  │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  │
│  │ SOP       │ │ Lead      │ │ Conversion│ │ Customer  │  │
│  │ Generator │ │ Scoring   │ │ Analytics │ │ Acquisition│  │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   AI MODEL LAYER                             │
│   Claude Opus 4.5 | GPT-4 Turbo | Gemini Pro | Embeddings   │
├─────────────────────────────────────────────────────────────┤
│                   DATA LAYER                                 │
│       Supabase PostgreSQL | Vector Store | Redis Cache      │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Files |
|-----------|---------|-------|
| Revenue Pipeline | Multi-stream revenue management | `revenue_pipeline_orchestrator.py` |
| Product Generator | AI-powered content/product creation | `product_generation_pipeline.py` |
| Affiliate System | Partner/affiliate program management | `affiliate_partnership_pipeline.py` |
| Knowledge Base | Semantic knowledge repository | `master_knowledge_base.py` |
| SOP Generator | Automated procedure documentation | `automated_sop_generator.py` |
| Lead Scoring | Advanced multi-factor lead scoring | `advanced_lead_scoring.py` |
| Conversion Analytics | Funnel metrics and optimization | `conversion_analytics.py` |

---

## CORE PIPELINES

### 1. Revenue Pipeline Orchestrator

**File:** `revenue_pipeline_orchestrator.py` (~900 lines)

#### Purpose
Manages and optimizes multiple revenue streams with AI-powered pricing, catalog management, and revenue tracking.

#### Revenue Streams
- Digital Products (eBooks, courses, templates)
- SaaS Subscriptions (monthly, annual, enterprise)
- Affiliate Programs
- Consulting Services
- API Access (metered, tiered)
- White-Label Licensing

#### Key Features
```python
# Initialize orchestrator
orchestrator = RevenuePipelineOrchestrator()

# Get revenue metrics
metrics = await orchestrator.get_revenue_metrics(
    time_range=TimeRange.LAST_30_DAYS
)

# Generate product catalog with AI
catalog = ProductCatalogGenerator()
products = await catalog.generate_tiered_products(
    base_product="WeatherCraft ERP",
    tiers=["starter", "professional", "enterprise"]
)
```

#### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/revenue/streams` | GET | List all revenue streams |
| `/revenue/metrics` | GET | Get revenue metrics by period |
| `/revenue/products` | POST | Add product to catalog |
| `/revenue/forecast` | GET | Get revenue forecast |

---

### 2. Product Generation Pipeline

**File:** `product_generation_pipeline.py` (~1600 lines)

#### Purpose
Multi-AI content factory for generating high-quality digital products at scale.

#### Product Types
| Type | Description | Quality Tiers |
|------|-------------|---------------|
| EBOOK | Full-length books (50K+ words) | Standard, Premium, Ultimate |
| GUIDE | Comprehensive guides (10K-30K words) | Standard, Premium |
| TEMPLATE_BUSINESS | Business templates | Standard, Premium |
| TEMPLATE_CODE | Code templates/boilerplates | Standard, Premium |
| COURSE | Full courses with curriculum | Premium, Ultimate |
| SOP | Standard operating procedures | All tiers |
| EMAIL_SEQUENCE | Marketing email campaigns | Standard, Premium |
| PROMPT_PACK | AI prompt collections | Standard, Premium |
| MICRO_TOOL | Small utility tools | Premium, Ultimate |

#### Multi-AI Orchestration
```python
# The pipeline uses all three AI models:
# 1. Claude: Research, writing, style capture
# 2. GPT: Technical accuracy, completeness
# 3. Gemini: Analysis, optimization, visual concepts

generator = ProductGenerator()
ebook = await generator.generate_ebook(
    topic="AI-Powered Roofing Business",
    chapters=12,
    quality_tier=QualityTier.PREMIUM,
    style="professional",
    target_audience="Roofing contractors"
)
```

#### Generation Process
```
1. Topic Research (Claude) → Extract key information
2. Outline Generation (GPT) → Create structure
3. Content Writing (Claude) → Draft chapters/sections
4. Technical Review (GPT) → Accuracy check
5. Optimization (Gemini) → Improve clarity
6. Design Generation → Cover/layout
7. Quality Assurance → Multi-point validation
8. Format Conversion → EPUB, PDF, HTML
```

---

### 3. Affiliate & Partnership Pipeline

**File:** `affiliate_partnership_pipeline.py` (~1400 lines)

#### Purpose
Complete affiliate program management with multi-tier tracking, fraud detection, and AI-powered content generation for partners.

#### Partner Types
- **Affiliate**: Commission-based referrals (20-40%)
- **Reseller**: Wholesale to retail (35-50% margin)
- **White-Label**: Full rebrand partnerships
- **Influencer**: Social media/content creators
- **Agency**: Client management partners
- **Strategic**: Alliance partnerships

#### Commission Structures
```python
# Default commission rates by tier
TIER_RATES = {
    "bronze": 0.20,   # 20% - Entry level
    "silver": 0.25,   # 25% - $1K-$5K/month
    "gold": 0.30,     # 30% - $5K-$25K/month
    "platinum": 0.35, # 35% - $25K-$100K/month
    "diamond": 0.40,  # 40% - $100K+/month
}

# Multi-tier rates
MULTI_TIER_RATES = {
    1: 1.0,   # Tier 1: 100% of commission
    2: 0.10,  # Tier 2: 10% of tier 1's commission
    3: 0.05,  # Tier 3: 5% of tier 1's commission
}
```

#### Fraud Detection
```python
class FraudDetector:
    MAX_CLICKS_PER_IP_HOUR = 10
    MAX_CONVERSIONS_PER_IP_DAY = 3
    MIN_TIME_ON_SITE_SECONDS = 5
    MAX_REFUND_RATE = 0.15  # 15%

    async def analyze_click(self, referral, affiliate):
        # Checks: IP velocity, bot detection, self-referral
        signals = []
        # ... fraud analysis logic
        return signals
```

#### Content Generation for Affiliates
- Email marketing templates
- Social media posts (Twitter, LinkedIn, Facebook, Instagram)
- Blog review articles
- Comparison content
- Video scripts

---

### 4. Master Knowledge Base

**File:** `master_knowledge_base.py` (~1100 lines)

#### Purpose
Central semantic knowledge repository with vector search, hierarchical organization, and AI-powered extraction.

#### Knowledge Types
- SOPs, Policies, Processes, Guides
- API Documentation, Code References
- Product Information, FAQs
- AI Prompt Templates, Agent Configurations
- Runbooks, Incident Reports
- Customer Profiles, Use Cases

#### Features
```python
# Create knowledge entry
kb = MasterKnowledgeBase()
entry = await kb.create_entry(
    title="Customer Refund SOP",
    content="...",
    knowledge_type=KnowledgeType.SOP,
    category="support",
    tags=["refunds", "customer-service"]
)

# Semantic search
results = await kb.search(
    query="how to handle refunds",
    knowledge_types=[KnowledgeType.SOP],
    top_k=5
)

# AI agent query interface
response = await kb.query_for_agent(
    agent_id="support_agent_001",
    query="What is the refund policy?"
)
```

#### Vector Store Integration
- Uses embeddings for semantic search
- Supports Pinecone, Weaviate, or in-memory
- Automatic embedding generation via OpenAI

---

### 5. Automated SOP Generator

**File:** `automated_sop_generator.py` (~1200 lines)

#### Purpose
Multi-AI SOP generation from various sources including documentation, logs, and conversations.

#### SOP Types
- Technical (IT/Engineering)
- Operational (Day-to-day)
- Customer Service
- Sales Processes
- HR Procedures
- Security/Compliance
- Emergency Response

#### Generation Sources
1. **Manual**: Human-written
2. **AI Generated**: Created from prompts
3. **Process Mined**: Extracted from logs
4. **Template**: Generated from templates
5. **Recorded**: From screen recordings

#### Multi-AI Generation Flow
```python
# Stage 1: Claude generates structure
initial = await generator._claude_generate_structure(...)

# Stage 2: GPT reviews for completeness
enhanced = await generator._gpt_enhance_content(initial, ...)

# Stage 3: Gemini optimizes
optimized = await generator._gemini_optimize(enhanced, ...)

# Build final SOP
sop = generator._build_sop_from_content(optimized)
```

#### Export Formats
- Markdown
- JSON
- HTML
- PDF (via additional converter)

---

## REVENUE SYSTEMS

### Lead Scoring Engine

**File:** `advanced_lead_scoring.py` (~800 lines)

#### 5-Dimension Scoring (100 points total)

| Dimension | Points | Components |
|-----------|--------|------------|
| Behavioral | 0-30 | Email engagement, website activity, content downloads, demo attendance |
| Firmographic | 0-25 | Company size, industry fit, revenue alignment, growth indicators |
| Intent Signals | 0-25 | Search behavior, competitor shopping, job postings, modernization signals |
| Deal Velocity | 0-15 | Time since engagement, touchpoint frequency |
| Financial Health | 0-5 | Payment history, contract value trend |

#### Lead Tiers
- **HOT** (80-100): Immediate action required
- **WARM** (60-79): Active nurturing
- **COOL** (40-59): Long-term nurturing
- **COLD** (0-39): Low priority

### Conversion Analytics

**File:** `conversion_analytics.py` (~700 lines)

#### Features
- 6-stage funnel tracking
- Industry benchmark comparison
- Bottleneck identification
- Win/loss analysis
- Predictive conversion modeling
- Revenue forecasting

#### Funnel Stages
```
NEW → CONTACTED → QUALIFIED → PROPOSAL_SENT → NEGOTIATING → WON/LOST
```

---

## INTEGRATION GUIDE

### FastAPI Integration

All pipelines include FastAPI routers that can be mounted in the main app:

```python
from fastapi import FastAPI
from revenue_pipeline_orchestrator import create_revenue_router
from affiliate_partnership_pipeline import create_affiliate_router
from master_knowledge_base import create_knowledge_router
from automated_sop_generator import create_sop_router

app = FastAPI()

app.include_router(create_revenue_router())
app.include_router(create_affiliate_router())
app.include_router(create_knowledge_router())
app.include_router(create_sop_router())
```

### Environment Variables

```bash
# AI APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Database
DATABASE_URL=postgresql://...
SUPABASE_URL=https://...
SUPABASE_SERVICE_ROLE_KEY=...

# Storage
S3_BUCKET=...
CLOUDFLARE_R2_BUCKET=...

# Payments
STRIPE_SECRET_KEY=sk_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

### CLI Usage

Each pipeline includes CLI interfaces for testing:

```bash
# Test product generation
python product_generation_pipeline.py

# Test affiliate pipeline
python affiliate_partnership_pipeline.py

# Test knowledge base
python master_knowledge_base.py

# Test SOP generator
python automated_sop_generator.py
```

---

## API REFERENCE

### Revenue API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /revenue/products` | Create product |
| `GET /revenue/products` | List products |
| `GET /revenue/metrics` | Get metrics |
| `POST /revenue/forecast` | Generate forecast |

### Affiliate API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /affiliate/register` | Register affiliate |
| `POST /affiliate/track/click` | Track click |
| `POST /affiliate/track/conversion` | Track conversion |
| `GET /affiliate/dashboard/{id}` | Get dashboard |
| `POST /affiliate/content/{id}` | Generate content |
| `POST /affiliate/payouts/process` | Process payouts |

### Knowledge API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /knowledge/entries` | Create entry |
| `GET /knowledge/entries/{id}` | Get entry |
| `POST /knowledge/search` | Search knowledge |
| `POST /knowledge/agent/query` | Agent query |
| `GET /knowledge/statistics` | Get statistics |

### SOP API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /sop/generate` | Generate SOP |
| `GET /sop/{id}` | Get SOP |
| `GET /sop/{id}/export/{format}` | Export SOP |
| `PUT /sop/{id}` | Update SOP |
| `POST /sop/{id}/approve` | Approve SOP |
| `POST /sop/{id}/publish` | Publish SOP |

---

## DEPLOYMENT

### Production Checklist

- [ ] Set all environment variables
- [ ] Configure database tables (migrations)
- [ ] Set up vector store (Pinecone/Weaviate)
- [ ] Configure Stripe webhooks
- [ ] Set up cron jobs for scheduled tasks
- [ ] Enable monitoring/alerting
- [ ] Configure rate limiting
- [ ] Set up backup procedures

### Database Tables Required

```sql
-- Revenue
CREATE TABLE revenue_streams (...);
CREATE TABLE product_catalog (...);
CREATE TABLE revenue_transactions (...);

-- Affiliates
CREATE TABLE affiliates (...);
CREATE TABLE referrals (...);
CREATE TABLE commissions (...);
CREATE TABLE payouts (...);

-- Knowledge
CREATE TABLE knowledge_entries (...);
CREATE TABLE knowledge_categories (...);

-- SOPs
CREATE TABLE sops (...);
CREATE TABLE sop_sections (...);
CREATE TABLE sop_steps (...);
```

### Monitoring

All pipelines include logging via `loguru`:

```python
from loguru import logger

logger.info("Pipeline started", pipeline="revenue")
logger.error("Processing failed", error=str(e))
```

---

## APPENDIX

### Performance Benchmarks

| Operation | Avg Time | Notes |
|-----------|----------|-------|
| Product generation (eBook) | 15-30 min | Full 50K word book |
| SOP generation | 2-5 min | Multi-AI review |
| Knowledge search | 50-200ms | Semantic search |
| Lead scoring | 100-300ms | Full 5-dimension |
| Commission calculation | <50ms | Per transaction |

### AI Model Usage

| Model | Use Cases | Cost Factor |
|-------|-----------|-------------|
| Claude Opus | Writing, research, complex reasoning | High |
| Claude Haiku | Quick processing, extraction | Low |
| GPT-4 Turbo | Technical accuracy, completeness | High |
| Gemini Pro | Analysis, optimization, multimodal | Medium |
| Embeddings | Vector search, similarity | Low |

---

## CHANGELOG

### v10.0.0 (2025-12-27)
- Added Product Generation Pipeline
- Added Revenue Pipeline Orchestrator
- Added Affiliate/Partnership Pipeline
- Added Master Knowledge Base
- Added Automated SOP Generator
- Multi-AI orchestration throughout

---

*Documentation generated by BrainOps AI OS*

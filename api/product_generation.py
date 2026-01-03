"""
Product Generation API Router
Secure, authenticated endpoints for the Multi-AI Product Generation Pipeline
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# API Key Security - use centralized config
from config import config

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

router = APIRouter(prefix="/products", tags=["Product Generation"])

# Import the product generator with fallback
try:
    from product_generation_pipeline import (
        ProductSpec,
        ProductType,
        QualityTier,
        get_product_generator,
    )
    PRODUCT_GENERATOR_AVAILABLE = True
    logger.info("Product Generation Pipeline loaded")
except ImportError as e:
    PRODUCT_GENERATOR_AVAILABLE = False
    logger.warning(f"Product Generation Pipeline not available: {e}")


# Pydantic models for API
class ProductRequest(BaseModel):
    """Request to generate a product"""
    product_type: str = Field(..., description="Type of product to generate")
    title: str = Field(..., max_length=200, description="Product title")
    description: str = Field(..., max_length=2000, description="Product description")
    target_audience: str = Field(..., max_length=500, description="Target audience")
    quality_tier: str = Field(default="premium", description="Quality tier: standard, premium, ultimate")
    word_count_target: int = Field(default=5000, ge=500, le=100000)
    style: str = Field(default="professional", max_length=100)
    tone: str = Field(default="authoritative", max_length=100)
    industry: str = Field(default="general", max_length=100)
    keywords: list[str] = Field(default_factory=list, max_items=20)
    include_visuals: bool = True
    include_templates: bool = True
    include_examples: bool = True
    custom_instructions: str = Field(default="", max_length=5000)
    tenant_id: str = Field(default="default", description="Tenant ID for multi-tenant isolation")


class ProductResponse(BaseModel):
    """Response with product generation result"""
    product_id: str
    status: str
    message: str
    estimated_completion_time: Optional[str] = None


class ProductStatusResponse(BaseModel):
    """Response with product status"""
    product_id: str
    status: str
    progress_percent: float
    content_preview: Optional[str] = None
    quality_score: Optional[float] = None
    models_used: list[str] = []
    created_at: str
    updated_at: Optional[str] = None


@router.get("/health")
async def product_generation_health():
    """Check product generation system health"""
    return {
        "status": "available" if PRODUCT_GENERATOR_AVAILABLE else "unavailable",
        "timestamp": datetime.utcnow().isoformat(),
        "capabilities": {
            "ebook_generation": PRODUCT_GENERATOR_AVAILABLE,
            "guide_generation": PRODUCT_GENERATOR_AVAILABLE,
            "template_generation": PRODUCT_GENERATOR_AVAILABLE,
            "course_generation": PRODUCT_GENERATOR_AVAILABLE,
            "multi_ai_orchestration": PRODUCT_GENERATOR_AVAILABLE
        }
    }


@router.get("/types")
async def list_product_types(api_key: str = Depends(verify_api_key)):
    """List available product types"""
    if not PRODUCT_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Product generation not available")

    return {
        "product_types": [
            {"id": "ebook", "name": "eBook", "description": "Full-length digital books (50K+ words)"},
            {"id": "guide", "name": "Guide", "description": "Comprehensive guides (10K-30K words)"},
            {"id": "template_business", "name": "Business Template", "description": "Professional business templates"},
            {"id": "template_code", "name": "Code Template", "description": "Code boilerplates and starters"},
            {"id": "course", "name": "Course", "description": "Full courses with curriculum"},
            {"id": "sop", "name": "SOP", "description": "Standard operating procedures"},
            {"id": "playbook", "name": "Playbook", "description": "Strategic playbooks"},
            {"id": "email_sequence", "name": "Email Sequence", "description": "Marketing email campaigns"},
            {"id": "prompt_pack", "name": "Prompt Pack", "description": "AI prompt collections"},
            {"id": "micro_tool", "name": "Micro Tool", "description": "Small utility tools/calculators"}
        ],
        "quality_tiers": [
            {"id": "standard", "name": "Standard", "description": "Single model, basic review"},
            {"id": "premium", "name": "Premium", "description": "Multi-model, enhanced review"},
            {"id": "ultimate", "name": "Ultimate", "description": "Full pipeline, human-like quality"}
        ]
    }


@router.post("/generate", response_model=ProductResponse)
async def generate_product(
    request: ProductRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Queue a product for generation.

    This endpoint queues the product and returns immediately.
    Use the /status/{product_id} endpoint to check progress.
    """
    if not PRODUCT_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Product generation not available")

    try:
        generator = get_product_generator()

        # Map string to enums with validation
        try:
            product_type = ProductType(request.product_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid product type: {request.product_type}") from None

        try:
            quality_tier = QualityTier(request.quality_tier)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid quality tier: {request.quality_tier}") from None

        # Create product spec with tenant isolation (using UUID for DB compatibility)
        import uuid
        spec = ProductSpec(
            product_id=str(uuid.uuid4()),  # Use real UUID for database compatibility
            product_type=product_type,
            title=request.title,
            description=request.description,
            target_audience=request.target_audience,
            quality_tier=quality_tier,
            word_count_target=request.word_count_target,
            style=request.style,
            tone=request.tone,
            industry=request.industry,
            keywords=request.keywords,
            include_visuals=request.include_visuals,
            include_templates=request.include_templates,
            include_examples=request.include_examples,
            custom_instructions=request.custom_instructions,
            metadata={"tenant_id": request.tenant_id}
        )

        # Initialize tables and create initial record
        await generator.initialize_tables()
        await _create_product_record(spec)

        # Queue for background generation
        background_tasks.add_task(generator.generate_product, spec)

        return ProductResponse(
            product_id=spec.product_id,
            status="queued",
            message="Product generation queued successfully",
            estimated_completion_time=_estimate_completion_time(product_type, quality_tier)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Product generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _create_product_record(spec):
    """Create initial product record in database"""
    import json
    import os

    import psycopg2

    db_url = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL')
    if not db_url:
        logger.warning("No DATABASE_URL - product not persisted initially")
        return

    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO generated_products (
                    id, product_type, title, description, spec, status
                ) VALUES (%s, %s, %s, %s, %s, 'queued')
                ON CONFLICT (id) DO NOTHING
            """, (
                spec.product_id,
                spec.product_type.value,
                spec.title,
                spec.description,
                json.dumps({
                    "target_audience": spec.target_audience,
                    "quality_tier": spec.quality_tier.value,
                    "word_count_target": spec.word_count_target,
                    "style": spec.style,
                    "tone": spec.tone,
                    "industry": spec.industry,
                    "keywords": spec.keywords,
                    "custom_instructions": spec.custom_instructions,
                    "tenant_id": spec.metadata.get("tenant_id", "default")
                })
            ))
            conn.commit()
            logger.info(f"Product {spec.product_id} record created in database")
    except Exception as e:
        logger.error(f"Failed to create product record: {e}")
    finally:
        conn.close()


@router.get("/status/{product_id}", response_model=ProductStatusResponse)
async def get_product_status(
    product_id: str,
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Get the status of a product generation job"""
    if not PRODUCT_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Product generation not available")

    try:
        generator = get_product_generator()
        status = await generator.get_status(product_id)

        if not status:
            raise HTTPException(status_code=404, detail="Product not found")

        # Verify tenant isolation
        if status.get("metadata", {}).get("tenant_id") != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return ProductStatusResponse(
            product_id=product_id,
            status=status.get("status", "unknown"),
            progress_percent=status.get("progress", 0),
            content_preview=status.get("preview"),
            quality_score=status.get("quality_score"),
            models_used=status.get("models_used", []),
            created_at=status.get("created_at", datetime.utcnow().isoformat()),
            updated_at=status.get("updated_at")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/list")
async def list_products(
    tenant_id: str = "default",
    status: Optional[str] = None,
    product_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    api_key: str = Depends(verify_api_key)
):
    """List products for a tenant"""
    if not PRODUCT_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Product generation not available")

    try:
        generator = get_product_generator()
        products = await generator.list_products(
            tenant_id=tenant_id,
            status=status,
            product_type=product_type,
            limit=limit,
            offset=offset
        )

        return {
            "products": products,
            "total": len(products),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"List products error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{product_id}/download")
async def download_product(
    product_id: str,
    format: str = "json",
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Download a completed product"""
    if not PRODUCT_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Product generation not available")

    try:
        generator = get_product_generator()
        product = await generator.get_product(product_id)

        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        # Verify tenant isolation
        if product.get("metadata", {}).get("tenant_id") != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        if product.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Product not yet completed")

        return {
            "product_id": product_id,
            "format": format,
            "content": product.get("content"),
            "assets": product.get("assets", []),
            "metadata": {
                "quality_score": product.get("quality_score"),
                "models_used": product.get("models_used"),
                "generation_time": product.get("generation_time"),
                "word_count": product.get("word_count")
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _estimate_completion_time(product_type: ProductType, quality_tier: QualityTier) -> str:
    """Estimate completion time based on product type and quality"""
    base_times = {
        ProductType.EBOOK: 30,
        ProductType.GUIDE: 15,
        ProductType.TEMPLATE_BUSINESS: 5,
        ProductType.TEMPLATE_CODE: 10,
        ProductType.COURSE: 45,
        ProductType.SOP: 10,
        ProductType.EMAIL_SEQUENCE: 5,
        ProductType.PROMPT_PACK: 8,
        ProductType.MICRO_TOOL: 12
    }

    multipliers = {
        QualityTier.STANDARD: 1.0,
        QualityTier.PREMIUM: 1.5,
        QualityTier.ULTIMATE: 2.5
    }

    base = base_times.get(product_type, 15)
    multiplier = multipliers.get(quality_tier, 1.5)
    minutes = int(base * multiplier)

    return f"{minutes} minutes"

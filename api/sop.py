"""
SOP Generator API Router
Secure, authenticated endpoints for the Automated SOP Generator
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.responses import Response
from pydantic import BaseModel, Field
import bleach  # For HTML sanitization

logger = logging.getLogger(__name__)

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = {"brainops_prod_key_2025", "brainops_dev_key_2025"}
APPROVER_API_KEYS = {"brainops_prod_key_2025"}  # Only approvers can approve SOPs

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

async def verify_approver_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify approver API key for SOP approval operations"""
    if not api_key or api_key not in APPROVER_API_KEYS:
        raise HTTPException(status_code=403, detail="Approver access required")
    return api_key

router = APIRouter(prefix="/sop", tags=["SOP Generator"])

# HTML sanitization allowed tags
ALLOWED_TAGS = [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'hr',
    'ul', 'ol', 'li', 'dl', 'dt', 'dd',
    'table', 'thead', 'tbody', 'tr', 'th', 'td',
    'a', 'strong', 'em', 'code', 'pre', 'blockquote',
    'div', 'span', 'img'
]
ALLOWED_ATTRIBUTES = {
    'a': ['href', 'title'],
    'img': ['src', 'alt', 'title'],
    'div': ['class'],
    'span': ['class'],
    'td': ['colspan', 'rowspan'],
    'th': ['colspan', 'rowspan']
}

# Import the SOP generator with fallback
try:
    from automated_sop_generator import (
        AutomatedSOPGenerator, SOPType, SOPStatus, GenerationSource,
        get_sop_generator
    )
    SOP_GENERATOR_AVAILABLE = True
    logger.info("Automated SOP Generator loaded")
except ImportError as e:
    SOP_GENERATOR_AVAILABLE = False
    logger.warning(f"Automated SOP Generator not available: {e}")


# Pydantic models
class SOPGenerationRequest(BaseModel):
    """Request to generate an SOP"""
    title: str = Field(..., min_length=5, max_length=300)
    description: str = Field(..., min_length=20, max_length=5000)
    sop_type: str = Field(default="operational")
    department: str = Field(default="general", max_length=100)
    target_audience: str = Field(default="all staff", max_length=200)
    complexity_level: str = Field(default="standard", pattern="^(simple|standard|complex)$")
    include_visuals: bool = True
    include_checklists: bool = True
    include_flowcharts: bool = False
    related_processes: List[str] = Field(default_factory=list, max_items=10)
    compliance_requirements: List[str] = Field(default_factory=list, max_items=10)
    custom_instructions: str = Field(default="", max_length=3000)
    tenant_id: str = Field(default="default")
    author: str = Field(default="system", max_length=200)


class SOPFromProcessRequest(BaseModel):
    """Request to generate SOP from process mining"""
    process_logs: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000)
    process_name: str = Field(..., min_length=3, max_length=200)
    department: str = Field(default="general", max_length=100)
    tenant_id: str = Field(default="default")


class SOPUpdateRequest(BaseModel):
    """Request to update an SOP"""
    title: Optional[str] = Field(None, min_length=5, max_length=300)
    content_updates: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    tenant_id: str = Field(default="default")


@router.get("/health")
async def sop_generator_health():
    """Check SOP generator system health"""
    return {
        "status": "available" if SOP_GENERATOR_AVAILABLE else "unavailable",
        "timestamp": datetime.utcnow().isoformat(),
        "capabilities": {
            "multi_ai_generation": SOP_GENERATOR_AVAILABLE,
            "process_mining": SOP_GENERATOR_AVAILABLE,
            "template_generation": SOP_GENERATOR_AVAILABLE,
            "version_control": SOP_GENERATOR_AVAILABLE,
            "approval_workflow": SOP_GENERATOR_AVAILABLE,
            "export_formats": ["markdown", "json", "html"]
        }
    }


@router.post("/generate")
async def generate_sop(
    request: SOPGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate a new SOP using multi-AI pipeline.

    The SOP generation runs in the background. Use the /status endpoint
    to check progress.
    """
    if not SOP_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SOP generator not available")

    try:
        generator = get_sop_generator()

        # Map SOP type
        try:
            sop_type = SOPType(request.sop_type)
        except ValueError:
            sop_type = SOPType.OPERATIONAL

        # Queue generation
        sop_id = await generator.queue_generation(
            title=request.title,
            description=request.description,
            sop_type=sop_type,
            department=request.department,
            target_audience=request.target_audience,
            complexity_level=request.complexity_level,
            include_visuals=request.include_visuals,
            include_checklists=request.include_checklists,
            include_flowcharts=request.include_flowcharts,
            related_processes=request.related_processes,
            compliance_requirements=request.compliance_requirements,
            custom_instructions=request.custom_instructions,
            tenant_id=request.tenant_id,
            author=request.author
        )

        return {
            "status": "queued",
            "sop_id": sop_id,
            "message": "SOP generation queued successfully",
            "estimated_time": _estimate_generation_time(request.complexity_level),
            "queued_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"SOP generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-from-process")
async def generate_sop_from_process(
    request: SOPFromProcessRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate SOP from process logs (process mining)"""
    if not SOP_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SOP generator not available")

    try:
        generator = get_sop_generator()

        sop_id = await generator.generate_from_process_logs(
            process_logs=request.process_logs,
            process_name=request.process_name,
            department=request.department,
            tenant_id=request.tenant_id
        )

        return {
            "status": "generating",
            "sop_id": sop_id,
            "message": "SOP generation from process logs started",
            "queued_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Process mining SOP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{sop_id}")
async def get_sop(
    sop_id: str,
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Get an SOP by ID"""
    if not SOP_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SOP generator not available")

    try:
        generator = get_sop_generator()
        sop = await generator.get_sop(sop_id)

        if not sop:
            raise HTTPException(status_code=404, detail="SOP not found")

        # Verify tenant isolation
        if sop.get("tenant_id") != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return sop

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get SOP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{sop_id}/status")
async def get_sop_status(
    sop_id: str,
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Get the generation status of an SOP"""
    if not SOP_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SOP generator not available")

    try:
        generator = get_sop_generator()
        status = await generator.get_status(sop_id, tenant_id)

        if not status:
            raise HTTPException(status_code=404, detail="SOP not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{sop_id}")
async def update_sop(
    sop_id: str,
    request: SOPUpdateRequest,
    api_key: str = Depends(verify_api_key)
):
    """Update an SOP"""
    if not SOP_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SOP generator not available")

    try:
        generator = get_sop_generator()

        # Verify tenant isolation first
        existing = await generator.get_sop(sop_id)
        if not existing:
            raise HTTPException(status_code=404, detail="SOP not found")

        if existing.get("tenant_id") != request.tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        updated = await generator.update_sop(
            sop_id=sop_id,
            title=request.title,
            content_updates=request.content_updates,
            status=request.status
        )

        return {
            "status": "updated",
            "sop_id": sop_id,
            "version": updated.get("version"),
            "updated_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update SOP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{sop_id}/approve")
async def approve_sop(
    sop_id: str,
    tenant_id: str = "default",
    approval_notes: str = "",
    api_key: str = Depends(verify_approver_key)  # Approver access required!
):
    """
    Approve an SOP.

    **Approver access required** - Only authorized approvers can approve SOPs.
    The approver identity is derived from the authenticated API key.
    """
    if not SOP_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SOP generator not available")

    try:
        generator = get_sop_generator()

        # Verify tenant isolation
        existing = await generator.get_sop(sop_id)
        if not existing:
            raise HTTPException(status_code=404, detail="SOP not found")

        if existing.get("tenant_id") != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Approver identity from API key (in production, from JWT)
        approved_by = "api_key_holder"  # In production: extract from JWT

        approved = await generator.approve_sop(
            sop_id=sop_id,
            approved_by=approved_by,
            approval_notes=approval_notes
        )

        return {
            "status": "approved",
            "sop_id": sop_id,
            "approved_by": approved_by,
            "approved_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approve SOP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{sop_id}/publish")
async def publish_sop(
    sop_id: str,
    tenant_id: str = "default",
    api_key: str = Depends(verify_approver_key)
):
    """Publish an approved SOP"""
    if not SOP_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SOP generator not available")

    try:
        generator = get_sop_generator()

        # Verify tenant and status
        existing = await generator.get_sop(sop_id)
        if not existing:
            raise HTTPException(status_code=404, detail="SOP not found")

        if existing.get("tenant_id") != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        if existing.get("status") != "approved":
            raise HTTPException(status_code=400, detail="SOP must be approved before publishing")

        published = await generator.publish_sop(sop_id)

        return {
            "status": "published",
            "sop_id": sop_id,
            "published_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Publish SOP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{sop_id}/export/{format}")
async def export_sop(
    sop_id: str,
    format: str,
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """
    Export SOP in various formats.

    Supported formats: markdown, json, html
    """
    if not SOP_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SOP generator not available")

    if format not in ["markdown", "json", "html"]:
        raise HTTPException(status_code=400, detail="Invalid format. Use: markdown, json, or html")

    try:
        generator = get_sop_generator()

        # Verify tenant isolation
        sop = await generator.get_sop(sop_id)
        if not sop:
            raise HTTPException(status_code=404, detail="SOP not found")

        if sop.get("tenant_id") != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        exported = await generator.export_sop(sop_id, format)

        if format == "html":
            # Sanitize HTML to prevent XSS
            sanitized = bleach.clean(
                exported.get("content", ""),
                tags=ALLOWED_TAGS,
                attributes=ALLOWED_ATTRIBUTES,
                strip=True
            )
            return Response(
                content=sanitized,
                media_type="text/html",
                headers={"Content-Disposition": f"attachment; filename={sop_id}.html"}
            )
        elif format == "markdown":
            return Response(
                content=exported.get("content", ""),
                media_type="text/markdown",
                headers={"Content-Disposition": f"attachment; filename={sop_id}.md"}
            )
        else:
            return exported

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export SOP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_sops(
    tenant_id: str = "default",
    status: Optional[str] = None,
    sop_type: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    api_key: str = Depends(verify_api_key)
):
    """List SOPs with filtering"""
    if not SOP_GENERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SOP generator not available")

    try:
        generator = get_sop_generator()
        sops = await generator.list_sops(
            tenant_id=tenant_id,
            status=status,
            sop_type=sop_type,
            department=department,
            limit=limit,
            offset=offset
        )

        return {
            "sops": sops,
            "total": len(sops),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"List SOPs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def list_sop_types():
    """List available SOP types"""
    return {
        "sop_types": [
            {"id": "technical", "name": "Technical", "description": "IT and engineering procedures"},
            {"id": "operational", "name": "Operational", "description": "Day-to-day operations"},
            {"id": "customer_service", "name": "Customer Service", "description": "Customer support procedures"},
            {"id": "sales", "name": "Sales", "description": "Sales process documentation"},
            {"id": "hr", "name": "HR", "description": "Human resources procedures"},
            {"id": "security", "name": "Security", "description": "Security and compliance"},
            {"id": "emergency", "name": "Emergency", "description": "Emergency response procedures"},
            {"id": "onboarding", "name": "Onboarding", "description": "Employee onboarding"}
        ],
        "complexity_levels": [
            {"id": "simple", "name": "Simple", "description": "Basic procedures, few steps"},
            {"id": "standard", "name": "Standard", "description": "Moderate complexity"},
            {"id": "complex", "name": "Complex", "description": "Multi-step, cross-functional"}
        ]
    }


def _estimate_generation_time(complexity: str) -> str:
    """Estimate generation time based on complexity"""
    times = {
        "simple": "2-3 minutes",
        "standard": "5-8 minutes",
        "complex": "10-15 minutes"
    }
    return times.get(complexity, "5-8 minutes")

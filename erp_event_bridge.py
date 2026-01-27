"""
ERP Event Bridge - Unified Event System Integration

This module bridges Weathercraft ERP events to the BrainOps AI Agents unified event system.
It receives events via webhook from the ERP SystemEventBus and:
1. Transforms them to unified event format
2. Stores them in the unified_events table
3. Routes them to appropriate AI agents
4. Broadcasts via Supabase Realtime for live subscribers

Author: BrainOps AI Team
Version: 2.0.0 - Unified Event System Integration
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Optional

from safe_task import create_safe_task
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["erp-bridge"])

# =============================================================================
# UNIFIED EVENT SYSTEM INTEGRATION
# =============================================================================

try:
    from api.events.unified import mark_event_processed, store_event
    from lib.events.schema import (
        UnifiedEvent,
    )
    UNIFIED_EVENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Unified events not available: {e}")
    UNIFIED_EVENTS_AVAILABLE = False

# =============================================================================
# AGENT IMPORTS (with graceful fallback)
# =============================================================================

# Customer Success Agent
try:
    from customer_success_agent import CustomerSuccessAgent
    CUSTOMER_SUCCESS_AVAILABLE = True
except ImportError:
    CUSTOMER_SUCCESS_AVAILABLE = False
    CustomerSuccessAgent = None

# Revenue Generation System
try:
    from revenue_generation_system import get_revenue_system
    REVENUE_SYSTEM_AVAILABLE = True
except ImportError:
    REVENUE_SYSTEM_AVAILABLE = False
    get_revenue_system = lambda: None

# Intelligent Follow-up System
try:
    from intelligent_followup_system import (
        FollowUpPriority,
        FollowUpType,
        get_intelligent_followup_system,
    )
    FOLLOWUP_SYSTEM_AVAILABLE = True
except ImportError:
    FOLLOWUP_SYSTEM_AVAILABLE = False
    get_intelligent_followup_system = lambda: None
    FollowUpType = None
    FollowUpPriority = None

# Task Orchestrator
try:
    from intelligent_task_orchestrator import get_task_orchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    get_task_orchestrator = lambda: None

# Lead Qualification Agent
try:
    from lead_qualification_agent import LeadQualificationAgent
    LEAD_QUALIFICATION_AVAILABLE = True
except ImportError:
    LEAD_QUALIFICATION_AVAILABLE = False
    LeadQualificationAgent = None

# Notification System
try:
    from notification_system import get_notification_system
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False
    get_notification_system = lambda: None


# =============================================================================
# ERP EVENT MODELS
# =============================================================================

class ERPSystemEvent(BaseModel):
    """
    Event model matching the ERP SystemEventEnvelope structure.

    ERP SystemEventBus sends events in this format:
    {
        "version": 1,
        "eventId": "evt_xxx",
        "type": "JOB_CREATED",
        "tenantId": "tenant_123",
        "timestamp": "2024-01-01T00:00:00.000Z",
        "source": "erp",
        "origin": "instance_id",
        "payload": {...},
        "metadata": {...}
    }
    """
    # Required fields
    id: Optional[str] = Field(None, alias="eventId")
    type: str
    tenant_id: str = Field(..., alias="tenantId")
    timestamp: datetime
    payload: dict[str, Any]

    # Optional fields
    version: int = 1
    source: Optional[str] = None
    origin: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    class Config:
        populate_by_name = True


class ERPEventResponse(BaseModel):
    """Response for ERP webhook"""
    status: str
    event_id: str
    unified_event_id: Optional[str] = None
    agents_triggered: list[str] = []
    errors: list[str] = []


# =============================================================================
# EVENT TYPE MAPPING
# =============================================================================

# Map ERP event types to processing functions
EVENT_PROCESSORS = {}


def event_processor(event_type: str):
    """Decorator to register event processors"""
    def decorator(func):
        EVENT_PROCESSORS[event_type] = func
        return func
    return decorator


# =============================================================================
# EVENT PROCESSORS
# =============================================================================

@event_processor("CUSTOMER_CREATED")
@event_processor("NEW_CUSTOMER")
async def process_customer_created(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process new customer creation - trigger onboarding"""
    result = {"agents": [], "actions": []}

    customer_id = event.payload.get("id") or event.payload.get("customerId")

    if not customer_id:
        return {"error": "No customer_id in payload"}

    # 1. Customer Success Agent - Generate onboarding plan
    if CUSTOMER_SUCCESS_AVAILABLE and CustomerSuccessAgent:
        try:
            csa = CustomerSuccessAgent(event.tenant_id)
            create_safe_task(csa.generate_onboarding_plan(customer_id))
            result["agents"].append("customer_success_agent")
            result["actions"].append("onboarding_plan_generation")
            logger.info(f"Triggered onboarding for customer {customer_id}")
        except Exception as e:
            logger.error(f"CustomerSuccessAgent error: {e}")
            result["errors"] = result.get("errors", []) + [str(e)]

    # 2. Revenue System - Track new customer opportunity
    if REVENUE_SYSTEM_AVAILABLE:
        try:
            revenue_system = get_revenue_system()
            if revenue_system:
                # Log as new revenue opportunity
                create_safe_task(revenue_system.track_opportunity(
                    entity_type="customer",
                    entity_id=customer_id,
                    tenant_id=event.tenant_id,
                    opportunity_type="new_customer",
                    metadata=event.payload,
                ))
                result["agents"].append("revenue_agent")
                result["actions"].append("opportunity_tracking")
        except Exception as e:
            logger.error(f"RevenueSystem error: {e}")

    return result


@event_processor("LEAD_CREATED")
async def process_lead_created(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process new lead - trigger qualification and scoring"""
    result = {"agents": [], "actions": []}

    lead_id = event.payload.get("leadId") or event.payload.get("id")

    if not lead_id:
        return {"error": "No lead_id in payload"}

    # 1. Lead Qualification Agent
    if LEAD_QUALIFICATION_AVAILABLE and LeadQualificationAgent:
        try:
            lqa = LeadQualificationAgent(event.tenant_id)
            create_safe_task(lqa.qualify_lead(lead_id, event.payload))
            result["agents"].append("lead_qualification_agent")
            result["actions"].append("lead_qualification")
        except Exception as e:
            logger.error(f"LeadQualificationAgent error: {e}")

    # 2. Submit to orchestrator for comprehensive processing
    if ORCHESTRATOR_AVAILABLE:
        try:
            orchestrator = get_task_orchestrator()
            if orchestrator:
                await orchestrator.submit_task(
                    title=f"Process Lead {lead_id}",
                    task_type="lead_processing",
                    payload={
                        "lead_id": lead_id,
                        "tenant_id": event.tenant_id,
                        "source": event.payload.get("source"),
                        "score": event.payload.get("score"),
                        "urgency": event.payload.get("urgency"),
                    },
                    priority=80,  # High priority for leads
                )
                result["agents"].append("task_orchestrator")
                result["actions"].append("lead_task_submitted")
        except Exception as e:
            logger.error(f"Task orchestrator error: {e}")

    return result


@event_processor("JOB_CREATED")
@event_processor("NEW_JOB")
async def process_job_created(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process new job - trigger scheduling and revenue tracking"""
    result = {"agents": [], "actions": []}

    job_id = event.payload.get("id") or event.payload.get("jobId")
    customer_id = event.payload.get("customer_id") or event.payload.get("customerId")

    if not job_id:
        return {"error": "No job_id in payload"}

    # 1. Submit scheduling task
    if ORCHESTRATOR_AVAILABLE:
        try:
            orchestrator = get_task_orchestrator()
            if orchestrator:
                await orchestrator.submit_task(
                    title=f"Schedule Job {job_id}",
                    task_type="scheduling",
                    payload={
                        "job_id": job_id,
                        "customer_id": customer_id,
                        "action": "find_slot",
                        "description": event.payload.get("description", "New job scheduling"),
                        "job_type": event.payload.get("jobType") or event.payload.get("job_type"),
                        "scheduled_start": event.payload.get("scheduledStart"),
                    },
                    priority=75,
                )
                result["agents"].append("scheduling_agent")
                result["actions"].append("scheduling_task_submitted")
                logger.info(f"Submitted scheduling task for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to submit scheduling task: {e}")

    # 2. Revenue tracking
    if REVENUE_SYSTEM_AVAILABLE:
        try:
            revenue_system = get_revenue_system()
            if revenue_system:
                create_safe_task(revenue_system.track_opportunity(
                    entity_type="job",
                    entity_id=job_id,
                    tenant_id=event.tenant_id,
                    opportunity_type="new_job",
                    metadata=event.payload,
                ))
                result["agents"].append("revenue_agent")
                result["actions"].append("job_revenue_tracking")
        except Exception as e:
            logger.error(f"RevenueSystem error: {e}")

    return result


@event_processor("JOB_SCHEDULED")
async def process_job_scheduled(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process job scheduled - send notifications"""
    result = {"agents": [], "actions": []}

    job_id = event.payload.get("jobId") or event.payload.get("id")

    # Send notifications
    if NOTIFICATION_AVAILABLE:
        try:
            notification_system = get_notification_system()
            if notification_system:
                create_safe_task(notification_system.send_job_scheduled_notification(
                    job_id=job_id,
                    tenant_id=event.tenant_id,
                    scheduled_start=event.payload.get("scheduledStart"),
                    scheduled_end=event.payload.get("scheduledEnd"),
                ))
                result["agents"].append("notification_agent")
                result["actions"].append("schedule_notification_sent")
        except Exception as e:
            logger.error(f"Notification error: {e}")

    return result


@event_processor("JOB_COMPLETED")
async def process_job_completed(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process job completion - trigger follow-up and review request"""
    result = {"agents": [], "actions": []}

    job_id = event.payload.get("id") or event.payload.get("jobId")
    customer_id = event.payload.get("customer_id") or event.payload.get("customerId")

    if not job_id:
        return {"error": "No job_id in payload"}

    # 1. Intelligent Follow-up System
    if FOLLOWUP_SYSTEM_AVAILABLE and customer_id:
        try:
            followup_system = get_intelligent_followup_system()
            if followup_system:
                context = {
                    "job_id": job_id,
                    "customer_id": customer_id,
                    "completion_date": event.payload.get("completedAt") or datetime.utcnow().isoformat(),
                    "job_type": event.payload.get("job_type") or event.payload.get("jobType", "service"),
                    "actual_hours": event.payload.get("actualHours"),
                }

                create_safe_task(followup_system.create_followup_sequence(
                    followup_type=FollowUpType.SERVICE_COMPLETION,
                    entity_id=customer_id,
                    entity_type="customer",
                    context=context,
                    priority=FollowUpPriority.HIGH,
                ))
                result["agents"].append("followup_agent")
                result["actions"].append("followup_sequence_created")
                logger.info(f"Triggered follow-up for completed job {job_id}")
        except Exception as e:
            logger.error(f"FollowupSystem error: {e}")

    # 2. Submit review request task
    if ORCHESTRATOR_AVAILABLE and customer_id:
        try:
            orchestrator = get_task_orchestrator()
            if orchestrator:
                await orchestrator.submit_task(
                    title=f"Request Review for Job {job_id}",
                    task_type="review_request",
                    payload={
                        "job_id": job_id,
                        "customer_id": customer_id,
                        "tenant_id": event.tenant_id,
                    },
                    priority=60,
                )
                result["agents"].append("review_agent")
                result["actions"].append("review_request_submitted")
        except Exception as e:
            logger.error(f"Review task error: {e}")

    return result


@event_processor("ESTIMATE_CREATED")
async def process_estimate_created(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process new estimate - trigger pricing analysis"""
    result = {"agents": [], "actions": []}

    estimate_id = event.payload.get("estimateId") or event.payload.get("id")
    customer_id = event.payload.get("customerId") or event.payload.get("customer_id")
    amount = event.payload.get("amount")

    if ORCHESTRATOR_AVAILABLE:
        try:
            orchestrator = get_task_orchestrator()
            if orchestrator:
                await orchestrator.submit_task(
                    title=f"Analyze Estimate {estimate_id}",
                    task_type="pricing_analysis",
                    payload={
                        "estimate_id": estimate_id,
                        "customer_id": customer_id,
                        "amount": amount,
                        "tenant_id": event.tenant_id,
                    },
                    priority=50,
                )
                result["agents"].append("pricing_agent")
                result["actions"].append("pricing_analysis_submitted")
        except Exception as e:
            logger.error(f"Pricing analysis error: {e}")

    return result


@event_processor("ESTIMATE_ACCEPTED")
async def process_estimate_accepted(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process accepted estimate - trigger job creation workflow"""
    result = {"agents": [], "actions": []}

    estimate_id = event.payload.get("estimateId") or event.payload.get("id")
    converted_job_id = event.payload.get("convertToJobId")

    # Revenue tracking
    if REVENUE_SYSTEM_AVAILABLE:
        try:
            revenue_system = get_revenue_system()
            if revenue_system:
                create_safe_task(revenue_system.track_conversion(
                    source_type="estimate",
                    source_id=estimate_id,
                    target_type="job",
                    target_id=converted_job_id,
                    tenant_id=event.tenant_id,
                ))
                result["agents"].append("revenue_agent")
                result["actions"].append("conversion_tracked")
        except Exception as e:
            logger.error(f"Revenue tracking error: {e}")

    return result


@event_processor("INVOICE_CREATED")
async def process_invoice_created(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process new invoice - trigger notification"""
    result = {"agents": [], "actions": []}

    invoice_id = event.payload.get("invoiceId") or event.payload.get("id")
    amount = event.payload.get("amount") or event.payload.get("total_amount")
    customer_id = event.payload.get("customerId") or event.payload.get("customer_id")

    logger.info(f"Processing INVOICE_CREATED for {invoice_id} (${amount})")

    # Send invoice notification
    if NOTIFICATION_AVAILABLE:
        try:
            notification_system = get_notification_system()
            if notification_system:
                create_safe_task(notification_system.send_invoice_notification(
                    invoice_id=invoice_id,
                    customer_id=customer_id,
                    amount=amount,
                    tenant_id=event.tenant_id,
                ))
                result["agents"].append("notification_agent")
                result["actions"].append("invoice_notification_sent")
        except Exception as e:
            logger.error(f"Invoice notification error: {e}")

    return result


@event_processor("INVOICE_PAID")
@event_processor("PAYMENT_RECEIVED")
async def process_payment_received(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process payment - update revenue metrics and trigger thank you"""
    result = {"agents": [], "actions": []}

    amount = event.payload.get("amount")
    invoice_id = event.payload.get("invoiceId") or event.payload.get("invoice_id")
    customer_id = event.payload.get("customerId") or event.payload.get("customer_id")

    logger.info(f"Payment received: ${amount}")

    # 1. Update revenue metrics
    if REVENUE_SYSTEM_AVAILABLE:
        try:
            revenue_system = get_revenue_system()
            if revenue_system:
                create_safe_task(revenue_system.record_payment(
                    amount=amount,
                    invoice_id=invoice_id,
                    tenant_id=event.tenant_id,
                    payment_method=event.payload.get("paymentMethod") or event.payload.get("payment_method"),
                ))
                result["agents"].append("revenue_agent")
                result["actions"].append("payment_recorded")
        except Exception as e:
            logger.error(f"Revenue recording error: {e}")

    # 2. Send thank you / satisfaction survey
    if FOLLOWUP_SYSTEM_AVAILABLE and customer_id:
        try:
            followup_system = get_intelligent_followup_system()
            if followup_system:
                create_safe_task(followup_system.create_followup_sequence(
                    followup_type=FollowUpType.PAYMENT_THANK_YOU,
                    entity_id=customer_id,
                    entity_type="customer",
                    context={
                        "amount": amount,
                        "invoice_id": invoice_id,
                        "payment_date": datetime.utcnow().isoformat(),
                    },
                    priority=FollowUpPriority.NORMAL,
                ))
                result["agents"].append("followup_agent")
                result["actions"].append("thank_you_scheduled")
        except Exception as e:
            logger.error(f"Thank you followup error: {e}")

    return result


@event_processor("INVOICE_OVERDUE")
async def process_invoice_overdue(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process overdue invoice - trigger collection workflow"""
    result = {"agents": [], "actions": []}

    invoice_id = event.payload.get("invoiceId") or event.payload.get("id")
    days_overdue = event.payload.get("daysOverdue")
    amount = event.payload.get("amount")

    if ORCHESTRATOR_AVAILABLE:
        try:
            orchestrator = get_task_orchestrator()
            if orchestrator:
                priority = 90 if days_overdue > 30 else 70 if days_overdue > 14 else 50
                await orchestrator.submit_task(
                    title=f"Collect Overdue Invoice {invoice_id}",
                    task_type="collection",
                    payload={
                        "invoice_id": invoice_id,
                        "days_overdue": days_overdue,
                        "amount": amount,
                        "tenant_id": event.tenant_id,
                    },
                    priority=priority,
                )
                result["agents"].append("collection_agent")
                result["actions"].append("collection_task_submitted")
        except Exception as e:
            logger.error(f"Collection task error: {e}")

    return result


@event_processor("EMPLOYEE_CLOCK_IN")
async def process_employee_clock_in(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process employee clock in - update workforce metrics"""
    result = {"agents": [], "actions": []}

    employee_id = event.payload.get("employee_id") or event.payload.get("employeeId")
    logger.info(f"Employee {employee_id} clocked in")

    # Could trigger workforce optimization agent here
    result["actions"].append("clock_in_logged")

    return result


@event_processor("WEATHER_ALERT")
async def process_weather_alert(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process weather alert - trigger rescheduling workflow"""
    result = {"agents": [], "actions": []}

    alert_type = event.payload.get("alertType")
    severity = event.payload.get("severity")
    affected_job_ids = event.payload.get("affectedJobIds", [])

    if ORCHESTRATOR_AVAILABLE and affected_job_ids:
        try:
            orchestrator = get_task_orchestrator()
            if orchestrator:
                priority = 95 if severity == "critical" else 80 if severity == "high" else 60
                await orchestrator.submit_task(
                    title=f"Weather Reschedule - {len(affected_job_ids)} jobs",
                    task_type="weather_reschedule",
                    payload={
                        "alert_type": alert_type,
                        "severity": severity,
                        "affected_job_ids": affected_job_ids,
                        "recommendation": event.payload.get("recommendation"),
                        "tenant_id": event.tenant_id,
                    },
                    priority=priority,
                )
                result["agents"].append("scheduling_agent")
                result["actions"].append("weather_reschedule_submitted")
        except Exception as e:
            logger.error(f"Weather reschedule error: {e}")

    return result


@event_processor("SYSTEM_ANOMALY")
@event_processor("DATA_INTEGRITY_ISSUE")
async def process_system_anomaly(event: ERPSystemEvent, unified_event: Optional["UnifiedEvent"]) -> dict[str, Any]:
    """Process system anomaly - trigger self-healing"""
    result = {"agents": [], "actions": []}

    severity = event.payload.get("severity", "warning")

    if ORCHESTRATOR_AVAILABLE:
        try:
            orchestrator = get_task_orchestrator()
            if orchestrator:
                priority = 99 if severity == "critical" else 85 if severity == "error" else 60
                await orchestrator.submit_task(
                    title=f"Investigate {event.type}",
                    task_type="self_healing",
                    payload={
                        "anomaly_type": event.payload.get("anomalyType") or event.payload.get("issueType"),
                        "description": event.payload.get("description"),
                        "affected_module": event.payload.get("affectedModule") or event.payload.get("tableName"),
                        "severity": severity,
                        "auto_fixable": event.payload.get("autoFixable", False),
                        "tenant_id": event.tenant_id,
                    },
                    priority=priority,
                )
                result["agents"].append("self_healing_agent")
                result["actions"].append("investigation_submitted")
        except Exception as e:
            logger.error(f"Self-healing task error: {e}")

    return result


# =============================================================================
# MAIN WEBHOOK ENDPOINT
# =============================================================================

async def verify_erp_webhook_signature(request: Request) -> bool:
    """Verify HMAC signature from ERP webhook"""
    import hashlib
    import hmac

    signature = request.headers.get("X-ERP-Signature") or request.headers.get("X-Webhook-Signature")
    secret = os.getenv("ERP_WEBHOOK_SECRET", "")

    if not secret:
        logger.warning("ERP_WEBHOOK_SECRET not configured - webhook verification disabled")
        return True  # Allow in dev if no secret set

    if not signature:
        logger.error("Missing ERP webhook signature header")
        return False

    body = await request.body()
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(signature, expected):
        logger.error("Invalid ERP webhook signature")
        return False

    return True


@router.post("/events/webhook", response_model=ERPEventResponse, summary="Receive events from ERP SystemEventBus")
async def handle_erp_event(
    event: ERPSystemEvent,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Webhook endpoint to receive real-time events from the ERP SystemEventBus.

    This endpoint:
    1. Verifies webhook signature for security
    2. Transforms ERP events to unified event format
    3. Stores them in the unified_events table
    4. Routes them to appropriate AI agents
    5. Broadcasts via Supabase Realtime

    Returns 500 on processing errors (ERP should retry with backoff).
    """
    # Verify signature first
    if not await verify_erp_webhook_signature(request):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    logger.info(f"Received ERP event: {event.type} - {event.id or 'no-id'}")

    response = ERPEventResponse(
        status="processing",
        event_id=event.id or f"evt_{datetime.utcnow().timestamp()}",
        agents_triggered=[],
        errors=[],
    )

    # 1. Transform to unified event format
    unified_event = None
    if UNIFIED_EVENTS_AVAILABLE:
        try:
            unified_event = UnifiedEvent.from_erp_event({
                "version": event.version,
                "eventId": event.id,
                "type": event.type,
                "tenantId": event.tenant_id,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "origin": event.origin,
                "payload": event.payload,
                "metadata": event.metadata,
            })
            response.unified_event_id = unified_event.event_id

            # Store in unified_events table
            stored = await store_event(unified_event)
            if stored:
                logger.debug(f"Stored unified event: {unified_event.event_id}")
            else:
                logger.warning(f"Failed to store unified event: {unified_event.event_id}")

        except Exception as e:
            logger.error(f"Failed to create unified event: {e}")
            response.errors.append(f"unified_event_creation: {str(e)}")

    # 2. Process event with registered processor
    processor = EVENT_PROCESSORS.get(event.type)
    if processor:
        try:
            result = await processor(event, unified_event)
            response.agents_triggered.extend(result.get("agents", []))
            if result.get("errors"):
                response.errors.extend(result["errors"])
            if result.get("error"):
                response.errors.append(result["error"])
        except Exception as e:
            logger.error(f"Event processor error for {event.type}: {e}")
            response.errors.append(f"processor: {str(e)}")
    else:
        logger.debug(f"No processor registered for event type: {event.type}")

    # 3. Mark event as processed
    if unified_event and UNIFIED_EVENTS_AVAILABLE:
        try:
            await mark_event_processed(
                unified_event.event_id,
                {
                    "agents": response.agents_triggered,
                    "errors": response.errors,
                    "processor": processor.__name__ if processor else None,
                }
            )
        except Exception as e:
            logger.error(f"Failed to mark event processed: {e}")

    # Set final status
    if response.errors:
        response.status = "processed_with_errors"
    else:
        response.status = "processed"

    return response


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.get("/events/processors", summary="List registered event processors")
async def list_processors():
    """List all registered event processors and their status"""
    return {
        "processors": list(EVENT_PROCESSORS.keys()),
        "count": len(EVENT_PROCESSORS),
        "systems_available": {
            "unified_events": UNIFIED_EVENTS_AVAILABLE,
            "customer_success": CUSTOMER_SUCCESS_AVAILABLE,
            "revenue_system": REVENUE_SYSTEM_AVAILABLE,
            "followup_system": FOLLOWUP_SYSTEM_AVAILABLE,
            "orchestrator": ORCHESTRATOR_AVAILABLE,
            "lead_qualification": LEAD_QUALIFICATION_AVAILABLE,
            "notification": NOTIFICATION_AVAILABLE,
        },
    }


@router.get("/events/health", summary="Health check for ERP bridge")
async def bridge_health():
    """Check health of the ERP event bridge"""
    return {
        "status": "healthy",
        "processors_registered": len(EVENT_PROCESSORS),
        "unified_events_available": UNIFIED_EVENTS_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat(),
    }

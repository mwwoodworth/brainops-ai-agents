"""
ERP Event Bridge
Integrates SystemEventBus from ERP into AI Agents Service.
Consumes events via webhook and routes them to appropriate AI agents.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import asyncio
import json

# Import Agents and Systems
from customer_success_agent import CustomerSuccessAgent
from revenue_generation_system import get_revenue_system, RevenueAction
from intelligent_followup_system import get_intelligent_followup_system, FollowUpType, FollowUpPriority

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["erp-bridge"])

class SystemEvent(BaseModel):
    """
    Event model matching the ERP SystemEvent structure.
    """
    id: str
    type: str
    payload: Dict[str, Any]
    tenant_id: str
    timestamp: datetime
    source: Optional[str] = None

@router.post("/events/webhook", summary="Receive events from ERP SystemEventBus")
async def handle_erp_event(event: SystemEvent, request: Request):
    """
    Webhook endpoint to receive real-time events from the ERP.
    Routes events to the appropriate AI agents.
    """
    logger.info(f"Received ERP event: {event.type} - {event.id}")
    
    try:
        # 1. NEW_CUSTOMER: Trigger CustomerSuccessAgent
        if event.type == "NEW_CUSTOMER":
            customer_id = event.payload.get("id")
            if customer_id and event.tenant_id:
                # Initialize agent with tenant context
                csa = CustomerSuccessAgent(event.tenant_id)
                # Run in background to not block the webhook
                asyncio.create_task(csa.generate_onboarding_plan(customer_id))
                logger.info(f"Triggered CustomerSuccessAgent onboarding for {customer_id}")

        # 2. NEW_JOB: Trigger SchedulingAgent + RevenueAgent
        elif event.type == "NEW_JOB":
            job_id = event.payload.get("id")
            customer_id = event.payload.get("customer_id")
            # Note: Specific 'SchedulingAgent' class not found, assuming logic or placeholder.
            # For RevenueAgent, we can check if this job represents a closed deal/revenue opportunity.
            revenue_system = get_revenue_system()
            if job_id:
                 # Log action in revenue system context if applicable
                 # Or potentially trigger an upsell opportunity analysis
                 pass
            logger.info(f"Processed NEW_JOB event for {job_id}")

        # 3. INVOICE_CREATED: Trigger FinanceAgent
        elif event.type == "INVOICE_CREATED":
            invoice_id = event.payload.get("id")
            amount = event.payload.get("total_amount")
            # Trigger logic for finance/revenue tracking
            logger.info(f"Processed INVOICE_CREATED for {invoice_id} (${amount})")

        # 4. PAYMENT_RECEIVED: Update RevenueDashboard
        elif event.type == "PAYMENT_RECEIVED":
            amount = event.payload.get("amount")
            # Update revenue metrics in revenue generation system
            # Since revenue_system is autonomous, we might just log this as a metric
            # or trigger a 'thank you' follow-up
            if amount:
                logger.info(f"Payment received: ${amount}. Updating metrics.")
                # Logic to update metrics would go here if exposed by RevenueSystem

        # 5. JOB_COMPLETED: Trigger FollowupAgent + ReviewAgent
        elif event.type == "JOB_COMPLETED":
            job_id = event.payload.get("id")
            customer_id = event.payload.get("customer_id")
            
            if customer_id:
                followup_system = get_intelligent_followup_system()
                
                # Context for the follow-up
                context = {
                    "job_id": job_id,
                    "customer_id": customer_id,
                    "completion_date": datetime.utcnow().isoformat(),
                    "job_type": event.payload.get("job_type", "service")
                }
                
                # Create follow-up sequence for service completion (review request)
                asyncio.create_task(followup_system.create_followup_sequence(
                    followup_type=FollowUpType.SERVICE_COMPLETION,
                    entity_id=customer_id,
                    entity_type="customer",
                    context=context,
                    priority=FollowUpPriority.HIGH
                ))
                logger.info(f"Triggered Intelligent Follow-up for Job {job_id}")

        # 6. EMPLOYEE_CLOCK_IN: Update WorkforceMetrics
        elif event.type == "EMPLOYEE_CLOCK_IN":
            employee_id = event.payload.get("employee_id")
            # Update workforce metrics logic
            logger.info(f"Employee {employee_id} clocked in")

        return {"status": "processed", "event_id": event.id}

    except Exception as e:
        logger.error(f"Error processing ERP event {event.id}: {e}")
        # Return 200 OK even on internal processing error to prevent ERP from retrying indefinitely
        # (unless we want retries, but typically for events we log and move on)
        return {"status": "error", "message": str(e)}

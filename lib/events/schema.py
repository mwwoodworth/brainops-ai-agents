"""
Unified Event Schema for BrainOps AI OS

This module defines the canonical event types used across all systems:
- Weathercraft ERP
- MyRoofGenius
- BrainOps AI Agents
- Command Center

All events flow through a unified event bus stored in brainops_core.event_bus
and broadcast via Supabase Realtime for live subscribers.
"""

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT SOURCE SYSTEMS
# =============================================================================

class EventSource(str, Enum):
    """Systems that can emit events"""
    ERP = "erp"                    # Weathercraft ERP
    MRG = "mrg"                    # MyRoofGenius
    AI_AGENTS = "ai-agents"        # BrainOps AI Agents
    COMMAND_CENTER = "command-center"
    CLI = "cli"                    # BrainOps CLI
    MANUAL = "manual"              # Manual/Admin actions
    WEBHOOK = "webhook"            # External webhooks


class EventPriority(str, Enum):
    """Event processing priority"""
    CRITICAL = "critical"   # Immediate processing required
    HIGH = "high"           # Process within seconds
    NORMAL = "normal"       # Standard processing queue
    LOW = "low"             # Background processing OK
    BATCH = "batch"         # Can be batched with similar events


class EventCategory(str, Enum):
    """High-level event categories"""
    BUSINESS = "business"       # Core business events (jobs, invoices, etc.)
    SYSTEM = "system"           # System/infrastructure events
    AI = "ai"                   # AI agent/learning events
    USER = "user"               # User action events
    INTEGRATION = "integration" # Cross-system integration events
    PHYSICAL = "physical"       # Physical world events (drones, robots, sensors)


# =============================================================================
# ERP/BUSINESS EVENT TYPES
# =============================================================================

class ERPEventType(str, Enum):
    """Events originating from Weathercraft ERP"""
    # Customer lifecycle
    CUSTOMER_CREATED = "customer.created"
    CUSTOMER_UPDATED = "customer.updated"
    CUSTOMER_DELETED = "customer.deleted"

    # Lead/CRM events
    LEAD_CREATED = "lead.created"
    LEAD_UPDATED = "lead.updated"
    LEAD_CONVERTED = "lead.converted"
    LEAD_LOST = "lead.lost"

    # Job events
    JOB_CREATED = "job.created"
    JOB_UPDATED = "job.updated"
    JOB_SCHEDULED = "job.scheduled"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_CANCELLED = "job.cancelled"

    # Estimate events
    ESTIMATE_CREATED = "estimate.created"
    ESTIMATE_SENT = "estimate.sent"
    ESTIMATE_VIEWED = "estimate.viewed"
    ESTIMATE_ACCEPTED = "estimate.accepted"
    ESTIMATE_REJECTED = "estimate.rejected"
    ESTIMATE_EXPIRED = "estimate.expired"

    # Invoice events
    INVOICE_CREATED = "invoice.created"
    INVOICE_SENT = "invoice.sent"
    INVOICE_VIEWED = "invoice.viewed"
    INVOICE_PAID = "invoice.paid"
    INVOICE_PARTIAL_PAID = "invoice.partial_paid"
    INVOICE_OVERDUE = "invoice.overdue"
    INVOICE_VOIDED = "invoice.voided"

    # Payment events
    PAYMENT_RECEIVED = "payment.received"
    PAYMENT_FAILED = "payment.failed"
    PAYMENT_REFUNDED = "payment.refunded"

    # Employee events
    EMPLOYEE_CLOCK_IN = "employee.clock_in"
    EMPLOYEE_CLOCK_OUT = "employee.clock_out"
    EMPLOYEE_PTO_REQUESTED = "employee.pto_requested"
    EMPLOYEE_PTO_APPROVED = "employee.pto_approved"
    EMPLOYEE_PTO_DENIED = "employee.pto_denied"

    # Schedule events
    SCHEDULE_CREATED = "schedule.created"
    SCHEDULE_UPDATED = "schedule.updated"
    SCHEDULE_CONFLICT = "schedule.conflict"

    # Weather events
    WEATHER_ALERT = "weather.alert"
    WEATHER_IMPACT = "weather.impact"

    # System/anomaly events
    SYSTEM_ANOMALY = "system.anomaly"
    DATA_INTEGRITY_ISSUE = "data.integrity_issue"


class AIEventType(str, Enum):
    """Events from AI Agents system"""
    # Agent lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_TIMEOUT = "agent.timeout"

    # Decision events
    DECISION_MADE = "decision.made"
    DECISION_ESCALATED = "decision.escalated"
    DECISION_OVERRIDDEN = "decision.overridden"

    # Learning events
    INSIGHT_GENERATED = "insight.generated"
    PATTERN_DETECTED = "pattern.detected"
    MODEL_UPDATED = "model.updated"

    # Prediction events
    PREDICTION_CREATED = "prediction.created"
    PREDICTION_VALIDATED = "prediction.validated"
    PREDICTION_FAILED = "prediction.failed"

    # Self-healing events
    ANOMALY_DETECTED = "anomaly.detected"
    HEALING_STARTED = "healing.started"
    HEALING_COMPLETED = "healing.completed"
    HEALING_FAILED = "healing.failed"

    # Revenue/automation events
    OPPORTUNITY_IDENTIFIED = "opportunity.identified"
    OUTREACH_TRIGGERED = "outreach.triggered"
    FOLLOWUP_SCHEDULED = "followup.scheduled"


class SystemEventType(str, Enum):
    """System-level events"""
    SERVICE_STARTED = "service.started"
    SERVICE_STOPPED = "service.stopped"
    SERVICE_DEGRADED = "service.degraded"
    SERVICE_RECOVERED = "service.recovered"

    ERROR_OCCURRED = "error.occurred"
    ALERT_TRIGGERED = "alert.triggered"

    DEPLOYMENT_STARTED = "deployment.started"
    DEPLOYMENT_COMPLETED = "deployment.completed"
    DEPLOYMENT_FAILED = "deployment.failed"


class PhysicalEventType(str, Enum):
    """Events from Physical World (Robotics/IoT)"""
    # Drone events
    DRONE_MISSION_REQUESTED = "drone.mission_requested"
    DRONE_TRAJECTORY_PLANNED = "drone.trajectory_planned"
    DRONE_MISSION_STARTED = "drone.mission_started"
    DRONE_WAYPOINT_REACHED = "drone.waypoint_reached"
    DRONE_ANOMALY_DETECTED = "drone.anomaly_detected"
    DRONE_MISSION_COMPLETED = "drone.mission_completed"

    # Robot events
    ROBOT_TASK_ASSIGNED = "robot.task_assigned"
    ROBOT_PATH_PLANNED = "robot.path_planned"
    ROBOT_ACTION_EXECUTED = "robot.action_executed"
    ROBOT_STATE_CHANGED = "robot.state_changed"

    # Sensor events
    SENSOR_READING_RECEIVED = "sensor.reading_received"
    SENSOR_THRESHOLD_EXCEEDED = "sensor.threshold_exceeded"


# =============================================================================
# UNIFIED EVENT ENVELOPE
# =============================================================================

class UnifiedEventPayload(BaseModel):
    """Base payload that all events must include"""
    tenant_id: str = Field(..., description="Tenant/organization ID")
    entity_type: Optional[str] = Field(None, description="Type of entity affected (customer, job, invoice, etc.)")
    entity_id: Optional[str] = Field(None, description="ID of the affected entity")

    class Config:
        extra = "allow"  # Allow additional fields


class UnifiedEvent(BaseModel):
    """
    Canonical event envelope used across all BrainOps systems.

    This schema unifies:
    - ERP SystemEventEnvelope (TypeScript)
    - AI Agents SystemEvent (Python)
    - BrainOps event_bus table schema
    """
    # Identity
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:16]}")
    version: int = Field(default=1, description="Schema version for evolution")

    # Classification
    event_type: str = Field(..., description="Dot-notation event type (e.g., 'job.created')")
    category: EventCategory = Field(default=EventCategory.BUSINESS)
    priority: EventPriority = Field(default=EventPriority.NORMAL)

    # Source tracking
    source: EventSource = Field(..., description="System that generated the event")
    source_instance: Optional[str] = Field(None, description="Specific instance/server ID")

    # Tenant isolation
    tenant_id: str = Field(..., description="Tenant/organization ID for multi-tenancy")

    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    occurred_at: Optional[datetime] = Field(None, description="When the actual event occurred (may differ from timestamp)")

    # Content
    payload: dict[str, Any] = Field(default_factory=dict)

    # Routing metadata
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)
    correlation_id: Optional[str] = Field(None, description="For tracing related events")
    causation_id: Optional[str] = Field(None, description="Event that caused this event")

    # Actor tracking
    actor_type: Optional[str] = Field(None, description="Type of actor (user, agent, system, webhook)")
    actor_id: Optional[str] = Field(None, description="ID of the actor")

    # Processing state (set by event bus, not by publisher)
    processed: bool = Field(default=False)
    processed_at: Optional[datetime] = Field(None)
    processing_result: Optional[dict[str, Any]] = Field(None)
    retry_count: int = Field(default=0)

    @validator('event_type')
    def validate_event_type(cls, v):
        """Ensure event_type follows dot notation"""
        if not v or '.' not in v:
            # Allow legacy format but convert
            if v and v.upper() == v:
                # Convert SNAKE_CASE to dot.notation
                parts = v.lower().split('_')
                if len(parts) >= 2:
                    return f"{parts[0]}.{'_'.join(parts[1:])}"
        return v.lower()

    def to_db_record(self) -> dict[str, Any]:
        """Convert to database-friendly dict for insertion"""
        # Note: asyncpg requires datetime objects, not isoformat strings
        return {
            'event_id': self.event_id,
            'version': self.version,
            'event_type': self.event_type,
            'category': self.category.value,
            'priority': self.priority.value,
            'source': self.source.value,
            'source_instance': self.source_instance,
            'tenant_id': self.tenant_id,
            'timestamp': self.timestamp,  # Keep as datetime for asyncpg
            'occurred_at': self.occurred_at,  # Keep as datetime for asyncpg
            'payload': self.payload,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id,
            'actor_type': self.actor_type,
            'actor_id': self.actor_id,
            'processed': self.processed,
            'processed_at': self.processed_at,  # Keep as datetime for asyncpg
            'processing_result': self.processing_result,
            'retry_count': self.retry_count,
        }

    def to_broadcast_payload(self) -> dict[str, Any]:
        """Convert to Supabase Realtime broadcast payload"""
        return {
            'event_id': self.event_id,
            'version': self.version,
            'type': self.event_type,
            'category': self.category.value,
            'priority': self.priority.value,
            'source': self.source.value,
            'tenant_id': self.tenant_id,
            'timestamp': self.timestamp.isoformat(),
            'payload': self.payload,
            'metadata': self.metadata,
            'actor': {
                'type': self.actor_type,
                'id': self.actor_id,
            } if self.actor_type else None,
        }

    @classmethod
    def from_erp_event(cls, erp_event: dict[str, Any]) -> "UnifiedEvent":
        """
        Transform ERP SystemEventEnvelope to UnifiedEvent.

        ERP format:
        {
            "version": 1,
            "eventId": "...",
            "type": "JOB_CREATED",
            "tenantId": "...",
            "timestamp": "...",
            "source": "...",
            "payload": {...},
            "metadata": {...}
        }
        """
        event_type = erp_event.get('type', '').lower()
        # Convert SNAKE_CASE to dot.notation
        if '_' in event_type:
            parts = event_type.split('_')
            event_type = f"{parts[0]}.{'_'.join(parts[1:])}"

        return cls(
            event_id=erp_event.get('eventId', f"evt_{uuid.uuid4().hex[:16]}"),
            version=erp_event.get('version', 1),
            event_type=event_type,
            category=EventCategory.BUSINESS,
            priority=EventPriority.NORMAL,
            source=EventSource.ERP,
            source_instance=erp_event.get('origin'),
            tenant_id=erp_event.get('tenantId', ''),
            timestamp=datetime.fromisoformat(erp_event['timestamp'].replace('Z', '+00:00')) if erp_event.get('timestamp') else datetime.utcnow(),
            payload=erp_event.get('payload', {}),
            metadata=erp_event.get('metadata', {}),
        )

    @classmethod
    def from_legacy_event(cls, legacy: dict[str, Any]) -> "UnifiedEvent":
        """
        Transform legacy AI Agents SystemEvent to UnifiedEvent.

        Legacy format:
        {
            "id": "...",
            "type": "NEW_CUSTOMER",
            "payload": {...},
            "tenant_id": "...",
            "timestamp": "...",
            "source": "..."
        }
        """
        event_type = legacy.get('type', '').lower()
        # Convert SNAKE_CASE to dot.notation
        if '_' in event_type:
            parts = event_type.split('_')
            event_type = f"{parts[0]}.{'_'.join(parts[1:])}"

        timestamp = legacy.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = datetime.utcnow()

        return cls(
            event_id=legacy.get('id', f"evt_{uuid.uuid4().hex[:16]}"),
            event_type=event_type,
            category=EventCategory.BUSINESS,
            source=EventSource(legacy.get('source', 'erp')) if legacy.get('source') in [e.value for e in EventSource] else EventSource.ERP,
            tenant_id=legacy.get('tenant_id', ''),
            timestamp=timestamp,
            payload=legacy.get('payload', {}),
        )


# =============================================================================
# TYPED EVENT PAYLOADS
# =============================================================================

class LeadCreatedPayload(UnifiedEventPayload):
    """Payload for lead.created event"""
    lead_id: str
    source: str
    score: Optional[float] = None
    urgency: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None


class JobCreatedPayload(UnifiedEventPayload):
    """Payload for job.created event"""
    job_id: str
    customer_id: Optional[str] = None
    job_type: Optional[str] = None
    scheduled_start: Optional[str] = None
    scheduled_end: Optional[str] = None
    description: Optional[str] = None
    assigned_crew: Optional[str] = None


class JobCompletedPayload(UnifiedEventPayload):
    """Payload for job.completed event"""
    job_id: str
    customer_id: Optional[str] = None
    completed_at: str
    actual_hours: Optional[float] = None
    completion_notes: Optional[str] = None


class EstimateCreatedPayload(UnifiedEventPayload):
    """Payload for estimate.created event"""
    estimate_id: str
    customer_id: str
    amount: float
    description: Optional[str] = None
    valid_until: Optional[str] = None


class EstimateAcceptedPayload(UnifiedEventPayload):
    """Payload for estimate.accepted event"""
    estimate_id: str
    accepted_at: str
    converted_job_id: Optional[str] = None


class InvoiceCreatedPayload(UnifiedEventPayload):
    """Payload for invoice.created event"""
    invoice_id: str
    customer_id: Optional[str] = None
    job_id: Optional[str] = None
    amount: float
    due_date: Optional[str] = None


class InvoicePaidPayload(UnifiedEventPayload):
    """Payload for invoice.paid event"""
    invoice_id: str
    paid_at: str
    amount: float
    payment_method: Optional[str] = None
    transaction_id: Optional[str] = None


class PaymentReceivedPayload(UnifiedEventPayload):
    """Payload for payment.received event"""
    payment_id: str
    customer_id: str
    amount: float
    payment_method: str
    invoice_id: Optional[str] = None


class AgentExecutionPayload(UnifiedEventPayload):
    """Payload for agent execution events"""
    agent_id: str
    agent_name: str
    execution_id: Optional[str] = None
    input_data: Optional[dict[str, Any]] = None
    output_data: Optional[dict[str, Any]] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class AnomalyDetectedPayload(UnifiedEventPayload):
    """Payload for anomaly.detected event"""
    anomaly_type: str
    severity: str  # info, warning, error, critical
    affected_module: str
    description: str
    suggested_action: Optional[str] = None
    auto_healable: bool = False


class DroneMissionPayload(UnifiedEventPayload):
    """Payload for drone mission events"""
    mission_id: str
    drone_id: str
    mission_type: str  # inspection, surveillance, delivery
    coordinates: Optional[list[dict[str, float]]] = None  # Lat/Lon/Alt
    trajectory: Optional[list[dict[str, Any]]] = None  # Planned path
    status: str


class RobotTaskPayload(UnifiedEventPayload):
    """Payload for robot task events"""
    task_id: str
    robot_id: str
    action: str
    target_object: Optional[str] = None
    target_coordinates: Optional[list[float]] = None  # [x, y, z]
    confidence: Optional[float] = None


class SensorReadingPayload(UnifiedEventPayload):
    """Payload for sensor reading events"""
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    threshold: Optional[float] = None
    location: Optional[str] = None


# =============================================================================
# EVENT TYPE REGISTRY
# =============================================================================

# Maps event types to their expected payload schemas
EVENT_PAYLOAD_REGISTRY: dict[str, type] = {
    'lead.created': LeadCreatedPayload,
    'job.created': JobCreatedPayload,
    'job.completed': JobCompletedPayload,
    'estimate.created': EstimateCreatedPayload,
    'estimate.accepted': EstimateAcceptedPayload,
    'invoice.created': InvoiceCreatedPayload,
    'invoice.paid': InvoicePaidPayload,
    'payment.received': PaymentReceivedPayload,
    'agent.started': AgentExecutionPayload,
    'agent.completed': AgentExecutionPayload,
    'agent.failed': AgentExecutionPayload,
    'anomaly.detected': AnomalyDetectedPayload,
    'drone.mission_requested': DroneMissionPayload,
    'drone.mission_completed': DroneMissionPayload,
    'robot.task_assigned': RobotTaskPayload,
    'sensor.reading_received': SensorReadingPayload,
}

# Maps event types to their processing agents
EVENT_AGENT_ROUTING: dict[str, list[str]] = {
    # Customer lifecycle
    'customer.created': ['customer_success_agent', 'revenue_agent'],
    'lead.created': ['lead_qualification_agent', 'outreach_agent'],
    'lead.converted': ['customer_success_agent', 'revenue_agent'],

    # Job lifecycle
    'job.created': ['scheduling_agent', 'revenue_agent'],
    'job.scheduled': ['notification_agent', 'resource_agent'],
    'job.completed': ['followup_agent', 'review_agent', 'billing_agent'],

    # Estimate/Invoice lifecycle
    'estimate.created': ['pricing_agent'],
    'estimate.accepted': ['job_creation_agent', 'revenue_agent'],
    'invoice.created': ['notification_agent'],
    'invoice.paid': ['revenue_agent', 'customer_success_agent'],
    'invoice.overdue': ['collection_agent', 'notification_agent'],

    # Payment events
    'payment.received': ['revenue_agent', 'accounting_agent'],

    # System events
    'anomaly.detected': ['self_healing_agent', 'alert_agent'],
    'error.occurred': ['self_healing_agent', 'monitoring_agent'],
}


def get_agents_for_event(event_type: str) -> list[str]:
    """Get list of agents that should process a given event type"""
    return EVENT_AGENT_ROUTING.get(event_type.lower(), [])


def validate_payload(event_type: str, payload: dict[str, Any]) -> bool:
    """Validate payload against registered schema"""
    schema = EVENT_PAYLOAD_REGISTRY.get(event_type.lower())
    if not schema:
        return True  # No schema registered, allow anything

    try:
        schema(**payload)
        return True
    except Exception as exc:
        logger.debug("Event payload validation failed: %s", exc)
        return False

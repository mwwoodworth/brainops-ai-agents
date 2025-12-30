"""
Unified Events Module for BrainOps AI OS

This module provides the canonical event system that unifies:
- Weathercraft ERP events
- AI Agents internal events
- Command Center subscriptions
- Cross-system integrations
"""

from .schema import (
    # Enums
    EventSource,
    EventPriority,
    EventCategory,
    ERPEventType,
    AIEventType,
    SystemEventType,
    # Core models
    UnifiedEvent,
    UnifiedEventPayload,
    # Typed payloads
    LeadCreatedPayload,
    JobCreatedPayload,
    JobCompletedPayload,
    EstimateCreatedPayload,
    EstimateAcceptedPayload,
    InvoiceCreatedPayload,
    InvoicePaidPayload,
    PaymentReceivedPayload,
    AgentExecutionPayload,
    AnomalyDetectedPayload,
    # Utilities
    get_agents_for_event,
    validate_payload,
    EVENT_AGENT_ROUTING,
    EVENT_PAYLOAD_REGISTRY,
)

__all__ = [
    # Enums
    'EventSource',
    'EventPriority',
    'EventCategory',
    'ERPEventType',
    'AIEventType',
    'SystemEventType',
    # Core models
    'UnifiedEvent',
    'UnifiedEventPayload',
    # Typed payloads
    'LeadCreatedPayload',
    'JobCreatedPayload',
    'JobCompletedPayload',
    'EstimateCreatedPayload',
    'EstimateAcceptedPayload',
    'InvoiceCreatedPayload',
    'InvoicePaidPayload',
    'PaymentReceivedPayload',
    'AgentExecutionPayload',
    'AnomalyDetectedPayload',
    # Utilities
    'get_agents_for_event',
    'validate_payload',
    'EVENT_AGENT_ROUTING',
    'EVENT_PAYLOAD_REGISTRY',
]

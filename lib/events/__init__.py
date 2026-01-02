"""
Unified Events Module for BrainOps AI OS

This module provides the canonical event system that unifies:
- Weathercraft ERP events
- AI Agents internal events
- Command Center subscriptions
- Cross-system integrations
"""

from .schema import (
    EVENT_AGENT_ROUTING,
    EVENT_PAYLOAD_REGISTRY,
    AgentExecutionPayload,
    AIEventType,
    AnomalyDetectedPayload,
    ERPEventType,
    EstimateAcceptedPayload,
    EstimateCreatedPayload,
    EventCategory,
    EventPriority,
    # Enums
    EventSource,
    InvoiceCreatedPayload,
    InvoicePaidPayload,
    JobCompletedPayload,
    JobCreatedPayload,
    # Typed payloads
    LeadCreatedPayload,
    PaymentReceivedPayload,
    SystemEventType,
    # Core models
    UnifiedEvent,
    UnifiedEventPayload,
    # Utilities
    get_agents_for_event,
    validate_payload,
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

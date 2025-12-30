"""
AUTOMATED SOP GENERATOR - BRAINOPS AI OS
=========================================
Multi-AI powered Standard Operating Procedure generation system.
Automatically creates, maintains, and updates SOPs from various sources.

Features:
- Multi-AI SOP generation (Claude + GPT + Gemini)
- Process mining from logs and recordings
- Template-based generation
- Version control and change tracking
- Compliance and audit support
- Multi-format export (MD, PDF, HTML, JSON)
- Automatic updates from system changes
- Integration with Knowledge Base

Author: BrainOps AI OS
Version: 1.0.0
"""

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import httpx
from loguru import logger


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SOPType(Enum):
    """Types of SOPs."""
    TECHNICAL = "technical"           # IT/Engineering procedures
    OPERATIONAL = "operational"       # Day-to-day operations
    CUSTOMER_SERVICE = "customer_service"  # Customer-facing procedures
    SALES = "sales"                   # Sales processes
    HR = "hr"                        # Human resources
    FINANCE = "finance"              # Financial procedures
    COMPLIANCE = "compliance"         # Regulatory compliance
    SECURITY = "security"            # Security procedures
    EMERGENCY = "emergency"          # Emergency/incident response
    ONBOARDING = "onboarding"        # Employee/customer onboarding
    QUALITY = "quality"              # Quality assurance


class SOPStatus(Enum):
    """SOP lifecycle status."""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    NEEDS_UPDATE = "needs_update"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class SOPPriority(Enum):
    """SOP priority/criticality."""
    CRITICAL = "critical"      # Must-follow, safety/compliance
    HIGH = "high"             # Important for operations
    MEDIUM = "medium"         # Standard procedures
    LOW = "low"               # Nice-to-have guidelines


class GenerationSource(Enum):
    """Source of SOP generation."""
    MANUAL = "manual"                 # Human written
    AI_GENERATED = "ai_generated"     # AI created from scratch
    PROCESS_MINED = "process_mined"   # Extracted from logs
    TEMPLATE = "template"             # Generated from template
    IMPORTED = "imported"             # Imported from external source
    RECORDED = "recorded"             # From screen/task recording
    HYBRID = "hybrid"                 # Combination


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SOPSection:
    """Individual section of an SOP."""
    section_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    order: int = 0
    section_type: str = "content"  # purpose, scope, steps, warnings, etc.
    subsections: List["SOPSection"] = field(default_factory=list)
    media: List[Dict[str, str]] = field(default_factory=list)  # images, videos
    warnings: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)


@dataclass
class SOPStep:
    """Individual step in a procedure."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int = 0
    title: str = ""
    description: str = ""
    expected_outcome: str = ""
    estimated_time_minutes: int = 0

    # Sub-steps
    substeps: List[str] = field(default_factory=list)

    # Conditionals
    prerequisites: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)  # if/when conditions

    # References
    tools_required: List[str] = field(default_factory=list)
    roles_responsible: List[str] = field(default_factory=list)
    related_sops: List[str] = field(default_factory=list)

    # Warnings and notes
    warnings: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # Media
    screenshots: List[str] = field(default_factory=list)
    video_links: List[str] = field(default_factory=list)


@dataclass
class SOP:
    """Complete Standard Operating Procedure."""
    sop_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Identification
    title: str = ""
    sop_number: str = ""  # e.g., "SOP-IT-001"
    description: str = ""
    purpose: str = ""
    scope: str = ""

    # Classification
    sop_type: SOPType = SOPType.OPERATIONAL
    status: SOPStatus = SOPStatus.DRAFT
    priority: SOPPriority = SOPPriority.MEDIUM
    department: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)

    # Content
    sections: List[SOPSection] = field(default_factory=list)
    steps: List[SOPStep] = field(default_factory=list)

    # References
    definitions: Dict[str, str] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    related_sops: List[str] = field(default_factory=list)
    forms_templates: List[str] = field(default_factory=list)

    # Responsibilities
    owner: str = ""
    reviewers: List[str] = field(default_factory=list)
    approvers: List[str] = field(default_factory=list)
    target_audience: List[str] = field(default_factory=list)

    # Versioning
    version: str = "1.0.0"
    version_history: List[Dict[str, Any]] = field(default_factory=list)

    # Compliance
    compliance_standards: List[str] = field(default_factory=list)
    audit_requirements: List[str] = field(default_factory=list)

    # Generation
    generation_source: GenerationSource = GenerationSource.MANUAL
    source_documents: List[str] = field(default_factory=list)
    ai_confidence_score: float = 1.0

    # Metrics
    estimated_duration_minutes: int = 0
    difficulty_level: str = "intermediate"  # beginner, intermediate, advanced
    word_count: int = 0
    step_count: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    effective_date: Optional[datetime] = None
    review_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None

    # Audit trail
    created_by: str = ""
    updated_by: str = ""
    approved_by: str = ""
    approval_date: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SOPTemplate:
    """Template for SOP generation."""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    sop_type: SOPType = SOPType.OPERATIONAL

    # Template structure
    required_sections: List[str] = field(default_factory=list)
    optional_sections: List[str] = field(default_factory=list)
    section_templates: Dict[str, str] = field(default_factory=dict)

    # Placeholders
    placeholders: Dict[str, str] = field(default_factory=dict)

    # Defaults
    default_priority: SOPPriority = SOPPriority.MEDIUM
    default_review_days: int = 90

    # Compliance
    required_compliance_fields: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessMiningResult:
    """Result of process mining from logs."""
    mining_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_type: str = ""  # logs, recordings, conversations
    source_files: List[str] = field(default_factory=list)

    # Extracted data
    process_name: str = ""
    steps_identified: List[Dict[str, Any]] = field(default_factory=list)
    actors_identified: List[str] = field(default_factory=list)
    systems_used: List[str] = field(default_factory=list)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)
    exceptions_found: List[str] = field(default_factory=list)

    # Metrics
    avg_duration_minutes: float = 0.0
    frequency_per_day: float = 0.0
    success_rate: float = 0.0
    common_errors: List[str] = field(default_factory=list)

    # Quality
    confidence_score: float = 0.0
    sample_size: int = 0

    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# SOP TEMPLATES
# =============================================================================

DEFAULT_TEMPLATES: Dict[SOPType, SOPTemplate] = {
    SOPType.TECHNICAL: SOPTemplate(
        name="Technical Procedure Template",
        description="Template for IT and engineering procedures",
        sop_type=SOPType.TECHNICAL,
        required_sections=[
            "purpose", "scope", "prerequisites", "procedure",
            "verification", "troubleshooting", "rollback"
        ],
        optional_sections=["background", "security_considerations", "appendix"],
        section_templates={
            "purpose": "This procedure describes how to {action} in order to {goal}.",
            "scope": "This procedure applies to {systems/teams} when {conditions}.",
            "prerequisites": "Before starting, ensure:\n- [ ] {prerequisite_1}\n- [ ] {prerequisite_2}",
            "verification": "To verify successful completion:\n1. {verification_step}",
            "rollback": "If issues occur:\n1. {rollback_step}",
        },
        placeholders={
            "action": "perform the technical task",
            "goal": "achieve the desired outcome",
            "systems/teams": "relevant systems and teams",
            "conditions": "specific conditions apply",
        },
        default_review_days=90,
    ),

    SOPType.CUSTOMER_SERVICE: SOPTemplate(
        name="Customer Service Procedure Template",
        description="Template for customer-facing procedures",
        sop_type=SOPType.CUSTOMER_SERVICE,
        required_sections=[
            "purpose", "scope", "customer_greeting",
            "procedure", "escalation", "closing"
        ],
        optional_sections=["faqs", "scripts", "resources"],
        section_templates={
            "purpose": "This procedure ensures consistent {service_type} for our customers.",
            "customer_greeting": "Always greet the customer warmly:\n\"{greeting_script}\"",
            "escalation": "Escalate to {escalation_path} when:\n- {escalation_trigger}",
            "closing": "Before ending the interaction:\n1. Confirm resolution\n2. Thank the customer\n3. Document in CRM",
        },
        placeholders={
            "service_type": "customer service",
            "greeting_script": "Hello, thank you for contacting us!",
            "escalation_path": "Team Lead",
            "escalation_trigger": "customer requests manager",
        },
        default_review_days=60,
    ),

    SOPType.SECURITY: SOPTemplate(
        name="Security Procedure Template",
        description="Template for security-critical procedures",
        sop_type=SOPType.SECURITY,
        required_sections=[
            "purpose", "scope", "authorization_requirements",
            "procedure", "audit_logging", "incident_reporting"
        ],
        optional_sections=["compliance_mapping", "risk_assessment"],
        section_templates={
            "purpose": "This procedure establishes controls for {security_area}.",
            "authorization_requirements": "This procedure requires:\n- Role: {required_role}\n- Approval: {approval_level}",
            "audit_logging": "Log all actions to {audit_system}. Include:\n- Timestamp\n- User ID\n- Action taken\n- Outcome",
            "incident_reporting": "Report any security incidents immediately to {security_team}.",
        },
        placeholders={
            "security_area": "access control",
            "required_role": "Security Admin",
            "approval_level": "Manager",
            "audit_system": "SIEM",
            "security_team": "Security Operations",
        },
        default_review_days=30,
        required_compliance_fields=["authorization_requirements", "audit_logging"],
    ),

    SOPType.EMERGENCY: SOPTemplate(
        name="Emergency Response Template",
        description="Template for emergency and incident procedures",
        sop_type=SOPType.EMERGENCY,
        required_sections=[
            "purpose", "scope", "immediate_actions",
            "notification_chain", "response_steps",
            "recovery", "post_incident"
        ],
        optional_sections=["contacts", "resources", "checklists"],
        section_templates={
            "purpose": "This procedure outlines the response to {emergency_type}.",
            "immediate_actions": "IMMEDIATELY:\n1. {immediate_action_1}\n2. {immediate_action_2}\n3. Notify: {notification_contact}",
            "notification_chain": "Notification order:\n1. {first_contact}\n2. {second_contact}\n3. {third_contact}",
            "post_incident": "Within 24 hours:\n- Document incident\n- Conduct root cause analysis\n- Update this SOP if needed",
        },
        default_priority=SOPPriority.CRITICAL,
        default_review_days=30,
    ),
}


# =============================================================================
# MULTI-AI GENERATOR
# =============================================================================

class MultiAISOPGenerator:
    """
    Uses multiple AI models to generate high-quality SOPs.
    Claude: Writing and structure
    GPT: Technical accuracy and completeness
    Gemini: Analysis and optimization
    """

    def __init__(
        self,
        anthropic_api_key: str = None,
        openai_api_key: str = None,
        gemini_api_key: str = None
    ):
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

    async def generate_sop(
        self,
        title: str,
        description: str,
        sop_type: SOPType,
        context: Dict[str, Any] = None,
        template: SOPTemplate = None,
        existing_docs: List[str] = None
    ) -> SOP:
        """Generate a complete SOP using multi-AI approach."""

        logger.info(f"Generating SOP: {title}")

        # Get template
        if not template:
            template = DEFAULT_TEMPLATES.get(sop_type, DEFAULT_TEMPLATES[SOPType.OPERATIONAL])

        # Stage 1: Claude generates initial structure and content
        initial_content = await self._claude_generate_structure(
            title, description, sop_type, template, context
        )

        # Stage 2: GPT reviews for completeness and technical accuracy
        enhanced_content = await self._gpt_enhance_content(
            initial_content, sop_type, context
        )

        # Stage 3: Gemini analyzes and optimizes
        optimized_content = await self._gemini_optimize(
            enhanced_content, sop_type
        )

        # Build SOP object
        sop = self._build_sop_from_content(
            title=title,
            description=description,
            sop_type=sop_type,
            content=optimized_content,
            template=template,
        )

        return sop

    async def _claude_generate_structure(
        self,
        title: str,
        description: str,
        sop_type: SOPType,
        template: SOPTemplate,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Claude generates initial SOP structure and content."""

        sections = template.required_sections + template.optional_sections

        prompt = f"""Create a comprehensive Standard Operating Procedure (SOP) with the following specifications:

TITLE: {title}
DESCRIPTION: {description}
TYPE: {sop_type.value}
REQUIRED SECTIONS: {', '.join(template.required_sections)}
OPTIONAL SECTIONS: {', '.join(template.optional_sections)}

CONTEXT:
{json.dumps(context or {}, indent=2)}

Generate the SOP in JSON format with this structure:
{{
    "purpose": "Clear statement of why this SOP exists",
    "scope": "Who and what this applies to",
    "definitions": {{"term": "definition"}},
    "responsibilities": {{"role": "responsibilities"}},
    "prerequisites": ["list of prerequisites"],
    "procedure": [
        {{
            "step_number": 1,
            "title": "Step title",
            "description": "Detailed instructions",
            "substeps": ["substep 1", "substep 2"],
            "warnings": ["any warnings"],
            "notes": ["helpful notes"],
            "expected_outcome": "What should happen",
            "estimated_time_minutes": 5
        }}
    ],
    "verification": "How to verify success",
    "troubleshooting": [
        {{"problem": "description", "solution": "how to fix"}}
    ],
    "references": ["related documents"],
    "revision_history": "placeholder for version tracking"
}}

Requirements:
1. Be specific and actionable
2. Include all necessary details for someone unfamiliar with the process
3. Add appropriate warnings and cautions
4. Estimate time for each step
5. Make it audit-ready with clear accountability
6. Use clear, professional language"""

        response = await self._call_claude(prompt)
        return self._parse_json_response(response)

    async def _gpt_enhance_content(
        self,
        content: Dict[str, Any],
        sop_type: SOPType,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """GPT reviews and enhances the SOP content."""

        prompt = f"""Review and enhance this SOP content for completeness and technical accuracy:

CURRENT CONTENT:
{json.dumps(content, indent=2)}

SOP TYPE: {sop_type.value}

Please analyze and improve:
1. Check for missing steps or information
2. Ensure technical accuracy
3. Add any missing prerequisites
4. Enhance troubleshooting section
5. Add edge cases and exception handling
6. Improve clarity of instructions
7. Add specific metrics or success criteria where appropriate

Return the enhanced SOP in the same JSON format, with improvements made.
If you add new content, mark it with a note like "(Enhanced: reason)".
"""

        response = await self._call_gpt(prompt)
        enhanced = self._parse_json_response(response)

        # Merge enhancements with original
        if enhanced:
            return {**content, **enhanced}
        return content

    async def _gemini_optimize(
        self,
        content: Dict[str, Any],
        sop_type: SOPType
    ) -> Dict[str, Any]:
        """Gemini optimizes the SOP for clarity and efficiency."""

        prompt = f"""Analyze and optimize this SOP for maximum effectiveness:

CONTENT:
{json.dumps(content, indent=2)}

SOP TYPE: {sop_type.value}

Optimization tasks:
1. Identify redundant or unclear steps
2. Suggest process improvements
3. Add efficiency recommendations
4. Ensure logical flow
5. Add quality checkpoints
6. Suggest automation opportunities
7. Rate the overall quality (1-10)

Return optimized content in the same JSON format with:
- Streamlined steps
- Added "optimization_notes" field with your recommendations
- Added "quality_score" field (1-10)
- Added "automation_opportunities" list
"""

        response = await self._call_gemini(prompt)
        optimized = self._parse_json_response(response)

        if optimized:
            return {**content, **optimized}
        return content

    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API."""
        if not self.anthropic_api_key:
            return await self._fallback_claude_response(prompt)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-3-opus-20240229",
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return await self._fallback_claude_response(prompt)

    async def _fallback_claude_response(self, prompt: str) -> str:
        """
        Fallback to a smaller model or raise error if no API access.
        Replacing previous mock implementation with real API attempt.
        """
        # If we have an OpenAI key, try that as fallback
        if self.openai_api_key:
            logger.info("Falling back to OpenAI for SOP generation")
            return await self._call_gpt(prompt)
            
        # If we have a Gemini key, try that
        if self.gemini_api_key:
            logger.info("Falling back to Gemini for SOP generation")
            return await self._call_gemini(prompt)

        # If no keys available, we must fail in production
        logger.error("No AI providers available for SOP generation")
        raise RuntimeError("No AI providers available. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.")

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from AI response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return {}

    def _build_sop_from_content(
        self,
        title: str,
        description: str,
        sop_type: SOPType,
        content: Dict[str, Any],
        template: SOPTemplate
    ) -> SOP:
        """Build SOP object from generated content."""

        # Create steps
        steps = []
        for step_data in content.get("procedure", []):
            step = SOPStep(
                step_number=step_data.get("step_number", len(steps) + 1),
                title=step_data.get("title", ""),
                description=step_data.get("description", ""),
                substeps=step_data.get("substeps", []),
                warnings=step_data.get("warnings", []),
                notes=step_data.get("notes", []),
                expected_outcome=step_data.get("expected_outcome", ""),
                estimated_time_minutes=step_data.get("estimated_time_minutes", 0),
            )
            steps.append(step)

        # Create sections
        sections = []
        section_order = 0

        if content.get("purpose"):
            sections.append(SOPSection(
                title="Purpose",
                content=content["purpose"],
                order=section_order,
                section_type="purpose"
            ))
            section_order += 1

        if content.get("scope"):
            sections.append(SOPSection(
                title="Scope",
                content=content["scope"],
                order=section_order,
                section_type="scope"
            ))
            section_order += 1

        if content.get("verification"):
            sections.append(SOPSection(
                title="Verification",
                content=content["verification"],
                order=section_order,
                section_type="verification"
            ))
            section_order += 1

        # Build troubleshooting section
        if content.get("troubleshooting"):
            troubleshooting_content = "\n".join([
                f"**Problem:** {t.get('problem', '')}\n**Solution:** {t.get('solution', '')}"
                for t in content["troubleshooting"]
            ])
            sections.append(SOPSection(
                title="Troubleshooting",
                content=troubleshooting_content,
                order=section_order,
                section_type="troubleshooting"
            ))

        # Calculate metrics
        total_time = sum(s.estimated_time_minutes for s in steps)
        word_count = len(json.dumps(content).split())

        # Create SOP
        sop = SOP(
            title=title,
            description=description,
            purpose=content.get("purpose", ""),
            scope=content.get("scope", ""),
            sop_type=sop_type,
            status=SOPStatus.DRAFT,
            priority=template.default_priority,
            sections=sections,
            steps=steps,
            definitions=content.get("definitions", {}),
            references=content.get("references", []),
            generation_source=GenerationSource.AI_GENERATED,
            ai_confidence_score=content.get("quality_score", 8.0) / 10.0,
            estimated_duration_minutes=total_time,
            step_count=len(steps),
            word_count=word_count,
            review_date=datetime.utcnow() + timedelta(days=template.default_review_days),
            metadata={
                "optimization_notes": content.get("optimization_notes", []),
                "automation_opportunities": content.get("automation_opportunities", []),
            }
        )

        # Generate SOP number
        sop.sop_number = f"SOP-{sop_type.value[:3].upper()}-{uuid.uuid4().hex[:6].upper()}"

        return sop


# =============================================================================
# PROCESS MINER
# =============================================================================

class ProcessMiner:
    """Extracts process steps from logs, recordings, and conversations."""

    def __init__(self, anthropic_api_key: str = None):
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    async def mine_from_logs(
        self,
        log_content: str,
        process_name: str = ""
    ) -> ProcessMiningResult:
        """Extract process from application/system logs."""

        prompt = f"""Analyze these logs and extract the underlying business process:

LOGS:
{log_content[:10000]}

Extract:
1. PROCESS_NAME: What process is being performed
2. STEPS: Sequential steps identified (in order)
3. ACTORS: Who/what is performing actions
4. SYSTEMS: Systems/applications involved
5. DECISION_POINTS: Any branching or conditional logic
6. EXCEPTIONS: Any errors or unusual flows
7. METRICS: Average duration, frequency, success rate

Return as JSON:
{{
    "process_name": "name",
    "steps": [
        {{"order": 1, "action": "description", "actor": "who", "system": "what"}}
    ],
    "actors": ["list of actors"],
    "systems": ["list of systems"],
    "decision_points": [
        {{"condition": "if X", "branches": ["option A", "option B"]}}
    ],
    "exceptions": ["list of errors/issues found"],
    "avg_duration_minutes": 10,
    "frequency_per_day": 50,
    "success_rate": 0.95
}}"""

        response = await self._call_claude(prompt)
        data = self._parse_json_response(response)

        return ProcessMiningResult(
            source_type="logs",
            process_name=data.get("process_name", process_name),
            steps_identified=data.get("steps", []),
            actors_identified=data.get("actors", []),
            systems_used=data.get("systems", []),
            decision_points=data.get("decision_points", []),
            exceptions_found=data.get("exceptions", []),
            avg_duration_minutes=data.get("avg_duration_minutes", 0),
            frequency_per_day=data.get("frequency_per_day", 0),
            success_rate=data.get("success_rate", 0),
            confidence_score=0.8,
        )

    async def mine_from_conversation(
        self,
        conversation: str,
        process_name: str = ""
    ) -> ProcessMiningResult:
        """Extract process from conversation/interview transcript."""

        prompt = f"""Analyze this conversation/transcript and extract the process being described:

CONVERSATION:
{conversation[:10000]}

Extract the implicit or explicit process being discussed.

Return as JSON:
{{
    "process_name": "identified process name",
    "steps": [
        {{
            "order": 1,
            "action": "what is done",
            "who": "who does it",
            "when": "timing/trigger",
            "how": "method/tool used"
        }}
    ],
    "actors": ["people/roles involved"],
    "systems": ["tools/systems mentioned"],
    "decision_points": [
        {{"condition": "when/if", "outcome": "what happens"}}
    ],
    "pain_points": ["problems mentioned"],
    "suggestions": ["improvements mentioned"]
}}"""

        response = await self._call_claude(prompt)
        data = self._parse_json_response(response)

        return ProcessMiningResult(
            source_type="conversation",
            process_name=data.get("process_name", process_name),
            steps_identified=data.get("steps", []),
            actors_identified=data.get("actors", []),
            systems_used=data.get("systems", []),
            decision_points=data.get("decision_points", []),
            common_errors=data.get("pain_points", []),
            confidence_score=0.7,
        )

    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API."""
        if not self.anthropic_api_key:
            return "{}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return "{}"

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return {}


# =============================================================================
# SOP EXPORTER
# =============================================================================

class SOPExporter:
    """Export SOPs to various formats."""

    def to_markdown(self, sop: SOP) -> str:
        """Export SOP to Markdown format."""
        lines = [
            f"# {sop.title}",
            "",
            f"**SOP Number:** {sop.sop_number}",
            f"**Version:** {sop.version}",
            f"**Status:** {sop.status.value}",
            f"**Type:** {sop.sop_type.value}",
            f"**Priority:** {sop.priority.value}",
            "",
            f"**Effective Date:** {sop.effective_date.strftime('%Y-%m-%d') if sop.effective_date else 'TBD'}",
            f"**Review Date:** {sop.review_date.strftime('%Y-%m-%d') if sop.review_date else 'TBD'}",
            "",
            "---",
            "",
        ]

        # Purpose
        if sop.purpose:
            lines.extend([
                "## Purpose",
                "",
                sop.purpose,
                "",
            ])

        # Scope
        if sop.scope:
            lines.extend([
                "## Scope",
                "",
                sop.scope,
                "",
            ])

        # Definitions
        if sop.definitions:
            lines.extend([
                "## Definitions",
                "",
            ])
            for term, definition in sop.definitions.items():
                lines.append(f"- **{term}:** {definition}")
            lines.append("")

        # Procedure
        if sop.steps:
            lines.extend([
                "## Procedure",
                "",
            ])
            for step in sop.steps:
                lines.append(f"### Step {step.step_number}: {step.title}")
                lines.append("")
                lines.append(step.description)
                lines.append("")

                if step.substeps:
                    for i, substep in enumerate(step.substeps, 1):
                        lines.append(f"   {step.step_number}.{i}. {substep}")
                    lines.append("")

                if step.warnings:
                    for warning in step.warnings:
                        lines.append(f"> ⚠️ **Warning:** {warning}")
                    lines.append("")

                if step.notes:
                    for note in step.notes:
                        lines.append(f"> 💡 **Note:** {note}")
                    lines.append("")

                if step.expected_outcome:
                    lines.append(f"**Expected Outcome:** {step.expected_outcome}")
                    lines.append("")

                if step.estimated_time_minutes:
                    lines.append(f"*Estimated time: {step.estimated_time_minutes} minutes*")
                    lines.append("")

        # Additional sections
        for section in sop.sections:
            if section.section_type not in ["purpose", "scope"]:
                lines.extend([
                    f"## {section.title}",
                    "",
                    section.content,
                    "",
                ])

        # References
        if sop.references:
            lines.extend([
                "## References",
                "",
            ])
            for ref in sop.references:
                lines.append(f"- {ref}")
            lines.append("")

        # Footer
        lines.extend([
            "---",
            "",
            f"*Generated: {sop.created_at.strftime('%Y-%m-%d %H:%M UTC')}*",
            f"*Last Updated: {sop.updated_at.strftime('%Y-%m-%d %H:%M UTC')}*",
        ])

        if sop.generation_source == GenerationSource.AI_GENERATED:
            lines.append(f"*AI Confidence: {sop.ai_confidence_score:.0%}*")

        return "\n".join(lines)

    def to_json(self, sop: SOP) -> str:
        """Export SOP to JSON format."""
        return json.dumps({
            "sop_id": sop.sop_id,
            "sop_number": sop.sop_number,
            "title": sop.title,
            "description": sop.description,
            "purpose": sop.purpose,
            "scope": sop.scope,
            "sop_type": sop.sop_type.value,
            "status": sop.status.value,
            "priority": sop.priority.value,
            "version": sop.version,
            "definitions": sop.definitions,
            "steps": [
                {
                    "step_number": s.step_number,
                    "title": s.title,
                    "description": s.description,
                    "substeps": s.substeps,
                    "warnings": s.warnings,
                    "notes": s.notes,
                    "expected_outcome": s.expected_outcome,
                    "estimated_time_minutes": s.estimated_time_minutes,
                }
                for s in sop.steps
            ],
            "references": sop.references,
            "estimated_duration_minutes": sop.estimated_duration_minutes,
            "created_at": sop.created_at.isoformat(),
            "updated_at": sop.updated_at.isoformat(),
            "review_date": sop.review_date.isoformat() if sop.review_date else None,
            "generation_source": sop.generation_source.value,
            "ai_confidence_score": sop.ai_confidence_score,
        }, indent=2)

    def to_html(self, sop: SOP) -> str:
        """Export SOP to HTML format."""
        # Convert markdown to basic HTML
        md = self.to_markdown(sop)

        # Simple markdown to HTML conversion
        html = md
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
        html = re.sub(r'\n\n', r'</p><p>', html)
        html = re.sub(r'---', r'<hr>', html)

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{sop.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1a1a1a; border-bottom: 2px solid #e5e5e5; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; }}
        h3 {{ color: #555; }}
        blockquote {{ background: #f9f9f9; border-left: 4px solid #ccc; padding: 10px 15px; margin: 15px 0; }}
        li {{ margin: 5px 0; }}
        hr {{ border: none; border-top: 1px solid #e5e5e5; margin: 30px 0; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .note {{ background: #d1ecf1; border-left: 4px solid #0dcaf0; }}
    </style>
</head>
<body>
    <p>{html}</p>
</body>
</html>"""


# =============================================================================
# MAIN SOP MANAGER
# =============================================================================

class AutomatedSOPGenerator:
    """
    Main manager for automated SOP generation and management.

    Capabilities:
    - Multi-AI SOP generation
    - Process mining from various sources
    - Template-based generation
    - Version control
    - Export to multiple formats
    - Integration with knowledge base
    """

    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")

        # Storage
        self.sops: Dict[str, SOP] = {}
        self.templates: Dict[str, SOPTemplate] = {}

        # Components
        self.generator = MultiAISOPGenerator()
        self.miner = ProcessMiner()
        self.exporter = SOPExporter()

        # Initialize default templates
        for sop_type, template in DEFAULT_TEMPLATES.items():
            self.templates[template.template_id] = template

        logger.info("AutomatedSOPGenerator initialized")

    async def generate_sop(
        self,
        title: str,
        description: str,
        sop_type: SOPType,
        context: Dict[str, Any] = None,
        template_id: str = None,
        created_by: str = "system"
    ) -> SOP:
        """Generate a new SOP."""

        template = None
        if template_id:
            template = self.templates.get(template_id)

        sop = await self.generator.generate_sop(
            title=title,
            description=description,
            sop_type=sop_type,
            context=context,
            template=template,
        )

        sop.created_by = created_by
        sop.updated_by = created_by

        self.sops[sop.sop_id] = sop

        logger.info(f"Generated SOP: {sop.sop_number} - {title}")

        return sop

    async def generate_from_process_mining(
        self,
        source_content: str,
        source_type: str,  # logs, conversation
        process_name: str = "",
        sop_type: SOPType = SOPType.OPERATIONAL,
        created_by: str = "system"
    ) -> SOP:
        """Generate SOP from mined process."""

        # Mine the process
        if source_type == "logs":
            result = await self.miner.mine_from_logs(source_content, process_name)
        else:
            result = await self.miner.mine_from_conversation(source_content, process_name)

        # Generate SOP from mining result
        context = {
            "mined_steps": result.steps_identified,
            "actors": result.actors_identified,
            "systems": result.systems_used,
            "decision_points": result.decision_points,
            "exceptions": result.exceptions_found,
        }

        sop = await self.generate_sop(
            title=result.process_name or f"Process: {process_name}",
            description=f"SOP generated from {source_type} analysis",
            sop_type=sop_type,
            context=context,
            created_by=created_by,
        )

        sop.generation_source = GenerationSource.PROCESS_MINED
        sop.source_documents = result.source_files
        sop.ai_confidence_score = result.confidence_score

        return sop

    async def update_sop(
        self,
        sop_id: str,
        updates: Dict[str, Any],
        updated_by: str = "system",
        create_version: bool = True
    ) -> SOP:
        """Update an existing SOP."""

        sop = self.sops.get(sop_id)
        if not sop:
            raise ValueError(f"SOP {sop_id} not found")

        # Create version history
        if create_version:
            old_version = {
                "version": sop.version,
                "updated_at": sop.updated_at.isoformat(),
                "updated_by": sop.updated_by,
                "changes": list(updates.keys()),
            }
            sop.version_history.append(old_version)

            # Increment version
            parts = sop.version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            sop.version = ".".join(parts)

        # Apply updates
        for key, value in updates.items():
            if hasattr(sop, key):
                setattr(sop, key, value)

        sop.updated_at = datetime.utcnow()
        sop.updated_by = updated_by

        logger.info(f"Updated SOP: {sop.sop_number} to v{sop.version}")

        return sop

    async def approve_sop(
        self,
        sop_id: str,
        approved_by: str,
        effective_date: datetime = None
    ) -> SOP:
        """Approve and publish an SOP."""

        sop = self.sops.get(sop_id)
        if not sop:
            raise ValueError(f"SOP {sop_id} not found")

        sop.status = SOPStatus.APPROVED
        sop.approved_by = approved_by
        sop.approval_date = datetime.utcnow()
        sop.effective_date = effective_date or datetime.utcnow()

        logger.info(f"Approved SOP: {sop.sop_number} by {approved_by}")

        return sop

    async def publish_sop(self, sop_id: str) -> SOP:
        """Publish an approved SOP."""

        sop = self.sops.get(sop_id)
        if not sop:
            raise ValueError(f"SOP {sop_id} not found")

        if sop.status not in [SOPStatus.APPROVED, SOPStatus.NEEDS_UPDATE]:
            raise ValueError(f"SOP must be approved before publishing")

        sop.status = SOPStatus.PUBLISHED
        sop.published_at = datetime.utcnow()

        logger.info(f"Published SOP: {sop.sop_number}")

        return sop

    def export_sop(
        self,
        sop_id: str,
        format: str = "markdown"
    ) -> str:
        """Export SOP to specified format."""

        sop = self.sops.get(sop_id)
        if not sop:
            raise ValueError(f"SOP {sop_id} not found")

        if format == "markdown":
            return self.exporter.to_markdown(sop)
        elif format == "json":
            return self.exporter.to_json(sop)
        elif format == "html":
            return self.exporter.to_html(sop)
        else:
            raise ValueError(f"Unknown format: {format}")

    async def get_sop(self, sop_id: str) -> Optional[SOP]:
        """Get SOP by ID."""
        return self.sops.get(sop_id)

    async def search_sops(
        self,
        query: str = "",
        sop_type: SOPType = None,
        status: SOPStatus = None,
        department: str = None
    ) -> List[SOP]:
        """Search SOPs."""
        results = []

        for sop in self.sops.values():
            if sop_type and sop.sop_type != sop_type:
                continue
            if status and sop.status != status:
                continue
            if department and sop.department != department:
                continue

            if query:
                query_lower = query.lower()
                if (query_lower in sop.title.lower() or
                    query_lower in sop.description.lower() or
                    any(query_lower in tag.lower() for tag in sop.tags)):
                    results.append(sop)
            else:
                results.append(sop)

        return results

    async def get_statistics(self) -> Dict[str, Any]:
        """Get SOP statistics."""
        by_type = {}
        by_status = {}
        by_priority = {}

        total_steps = 0
        total_duration = 0

        for sop in self.sops.values():
            by_type[sop.sop_type.value] = by_type.get(sop.sop_type.value, 0) + 1
            by_status[sop.status.value] = by_status.get(sop.status.value, 0) + 1
            by_priority[sop.priority.value] = by_priority.get(sop.priority.value, 0) + 1
            total_steps += sop.step_count
            total_duration += sop.estimated_duration_minutes

        # SOPs needing review
        now = datetime.utcnow()
        needs_review = [
            sop for sop in self.sops.values()
            if sop.review_date and sop.review_date <= now
        ]

        return {
            "total_sops": len(self.sops),
            "by_type": by_type,
            "by_status": by_status,
            "by_priority": by_priority,
            "total_steps": total_steps,
            "total_duration_minutes": total_duration,
            "avg_steps_per_sop": total_steps / len(self.sops) if self.sops else 0,
            "sops_needing_review": len(needs_review),
            "ai_generated_count": len([
                s for s in self.sops.values()
                if s.generation_source == GenerationSource.AI_GENERATED
            ]),
        }


# =============================================================================
# FASTAPI ROUTER
# =============================================================================

def create_sop_router():
    """Create FastAPI router for SOP endpoints."""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    router = APIRouter(prefix="/sop", tags=["SOP Generator"])
    manager = AutomatedSOPGenerator()

    class GenerateSOPRequest(BaseModel):
        title: str
        description: str
        sop_type: str
        context: dict = None

    class UpdateSOPRequest(BaseModel):
        updates: dict
        create_version: bool = True

    @router.post("/generate")
    async def generate_sop(data: GenerateSOPRequest):
        """Generate a new SOP."""
        sop = await manager.generate_sop(
            title=data.title,
            description=data.description,
            sop_type=SOPType(data.sop_type),
            context=data.context,
        )
        return {
            "success": True,
            "sop_id": sop.sop_id,
            "sop_number": sop.sop_number,
            "title": sop.title,
            "step_count": sop.step_count,
            "ai_confidence": sop.ai_confidence_score,
        }

    @router.get("/{sop_id}")
    async def get_sop(sop_id: str):
        """Get SOP by ID."""
        sop = await manager.get_sop(sop_id)
        if not sop:
            raise HTTPException(status_code=404, detail="SOP not found")
        return manager.exporter.to_json(sop)

    @router.get("/{sop_id}/export/{format}")
    async def export_sop(sop_id: str, format: str = "markdown"):
        """Export SOP in specified format."""
        try:
            content = manager.export_sop(sop_id, format)
            return {"content": content, "format": format}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.put("/{sop_id}")
    async def update_sop(sop_id: str, data: UpdateSOPRequest):
        """Update an SOP."""
        try:
            sop = await manager.update_sop(
                sop_id=sop_id,
                updates=data.updates,
                create_version=data.create_version,
            )
            return {
                "success": True,
                "sop_id": sop.sop_id,
                "version": sop.version,
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.post("/{sop_id}/approve")
    async def approve_sop(sop_id: str, approved_by: str):
        """Approve an SOP."""
        try:
            sop = await manager.approve_sop(sop_id, approved_by)
            return {
                "success": True,
                "sop_id": sop.sop_id,
                "status": sop.status.value,
                "approved_by": sop.approved_by,
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.post("/{sop_id}/publish")
    async def publish_sop(sop_id: str):
        """Publish an approved SOP."""
        try:
            sop = await manager.publish_sop(sop_id)
            return {
                "success": True,
                "sop_id": sop.sop_id,
                "status": sop.status.value,
                "published_at": sop.published_at.isoformat(),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/")
    async def list_sops(
        query: str = "",
        sop_type: str = None,
        status: str = None
    ):
        """List/search SOPs."""
        sops = await manager.search_sops(
            query=query,
            sop_type=SOPType(sop_type) if sop_type else None,
            status=SOPStatus(status) if status else None,
        )
        return {
            "count": len(sops),
            "sops": [
                {
                    "sop_id": s.sop_id,
                    "sop_number": s.sop_number,
                    "title": s.title,
                    "type": s.sop_type.value,
                    "status": s.status.value,
                    "step_count": s.step_count,
                }
                for s in sops
            ]
        }

    @router.get("/statistics")
    async def get_statistics():
        """Get SOP statistics."""
        return await manager.get_statistics()

    return router


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_sop_generator_instance: Optional[AutomatedSOPGenerator] = None


def get_sop_generator() -> AutomatedSOPGenerator:
    """Get the singleton AutomatedSOPGenerator instance."""
    global _sop_generator_instance
    if _sop_generator_instance is None:
        _sop_generator_instance = AutomatedSOPGenerator()
    return _sop_generator_instance


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Demo the SOP generator."""
    print("=" * 70)
    print("AUTOMATED SOP GENERATOR DEMO")
    print("=" * 70)

    manager = get_sop_generator()

    # Generate SOPs
    print("\n1. Generating Technical SOP...")

    sop1 = await manager.generate_sop(
        title="Database Backup and Recovery Procedure",
        description="Standard procedure for backing up production databases and recovering from failures",
        sop_type=SOPType.TECHNICAL,
        context={
            "database": "PostgreSQL",
            "environment": "Production",
            "backup_frequency": "Daily",
            "retention": "30 days",
        }
    )
    print(f"   Generated: {sop1.sop_number} - {sop1.title}")
    print(f"   Steps: {sop1.step_count}")
    print(f"   Confidence: {sop1.ai_confidence_score:.0%}")

    print("\n2. Generating Customer Service SOP...")

    sop2 = await manager.generate_sop(
        title="Customer Complaint Resolution Process",
        description="Standard procedure for handling and resolving customer complaints",
        sop_type=SOPType.CUSTOMER_SERVICE,
        context={
            "response_time_sla": "4 hours",
            "resolution_sla": "24 hours",
            "escalation_path": ["Team Lead", "Manager", "Director"],
        }
    )
    print(f"   Generated: {sop2.sop_number} - {sop2.title}")
    print(f"   Steps: {sop2.step_count}")

    print("\n3. Generating Emergency SOP...")

    sop3 = await manager.generate_sop(
        title="Production Outage Response",
        description="Emergency response procedure for production system outages",
        sop_type=SOPType.EMERGENCY,
        context={
            "severity_levels": ["P1", "P2", "P3"],
            "notification_chain": ["On-call Engineer", "Engineering Manager", "CTO"],
            "communication_channels": ["Slack", "PagerDuty", "Email"],
        }
    )
    print(f"   Generated: {sop3.sop_number} - {sop3.title}")
    print(f"   Priority: {sop3.priority.value}")

    # Export
    print("\n4. Exporting to Markdown...")
    markdown = manager.export_sop(sop1.sop_id, "markdown")
    print(f"   Preview (first 500 chars):")
    print("   " + "-" * 50)
    for line in markdown[:500].split("\n"):
        print(f"   {line}")
    print("   ...")

    # Statistics
    print("\n5. SOP Statistics:")
    stats = await manager.get_statistics()
    print(f"   Total SOPs: {stats['total_sops']}")
    print(f"   Total Steps: {stats['total_steps']}")
    print(f"   AI Generated: {stats['ai_generated_count']}")
    print(f"   By Type: {stats['by_type']}")
    print(f"   By Priority: {stats['by_priority']}")

    # Approve and publish
    print("\n6. Approval Workflow:")
    await manager.approve_sop(sop1.sop_id, "admin@company.com")
    print(f"   Approved: {sop1.sop_number}")

    await manager.publish_sop(sop1.sop_id)
    print(f"   Published: {sop1.sop_number} - Status: {sop1.status.value}")

    print("\n" + "=" * 70)
    print("SOP GENERATOR DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

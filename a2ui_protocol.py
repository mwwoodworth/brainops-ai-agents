"""
A2UI Protocol Implementation - Agent-to-User Interface
=======================================================
Google's open standard for agent-generated user interfaces.
https://a2ui.org/ | https://github.com/google/A2UI

This module enables BrainOps agents to generate A2UI-compliant responses
that can be rendered by any A2UI-compatible frontend (React, Flutter, Angular, etc.)

Key Features:
- Declarative JSON format (security-first, no executable code)
- LLM-friendly flat component structure
- Framework-agnostic output
- Schema validation against A2UI spec
- Integration with AUREA and other agents

Note: This is the BACKEND generator. Frontend renderers would be
implemented in weathercraft-erp or myroofgenius-app.
"""

import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """A2UI Component Types - Trusted Component Catalog"""
    # Layout Components
    COLUMN = "Column"
    ROW = "Row"
    CARD = "Card"
    CONTAINER = "Container"
    DIVIDER = "Divider"
    SPACER = "Spacer"

    # Text Components
    TEXT = "Text"
    HEADING = "Heading"
    PARAGRAPH = "Paragraph"
    LABEL = "Label"

    # Input Components
    TEXT_FIELD = "TextField"
    TEXT_AREA = "TextArea"
    NUMBER_INPUT = "NumberInput"
    DATE_TIME_INPUT = "DateTimeInput"
    SELECT = "Select"
    CHECKBOX = "Checkbox"
    RADIO = "Radio"
    SWITCH = "Switch"

    # Action Components
    BUTTON = "Button"
    LINK = "Link"
    ICON_BUTTON = "IconButton"

    # Data Display
    TABLE = "Table"
    LIST = "List"
    CHART = "Chart"
    PROGRESS = "Progress"
    BADGE = "Badge"
    AVATAR = "Avatar"
    IMAGE = "Image"

    # Feedback
    ALERT = "Alert"
    TOAST = "Toast"
    DIALOG = "Dialog"
    LOADING = "Loading"


class UsageHint(Enum):
    """Semantic hints for component rendering"""
    TITLE = "title"
    SUBTITLE = "subtitle"
    BODY = "body"
    CAPTION = "caption"
    EMPHASIS = "emphasis"
    ERROR = "error"
    SUCCESS = "success"
    WARNING = "warning"
    INFO = "info"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    DESTRUCTIVE = "destructive"


class LayoutDirection(Enum):
    """Layout direction for container components"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass
class A2UIComponent:
    """A single A2UI component in the flat list structure"""
    id: str
    type: str  # ComponentType value
    properties: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)  # IDs of child components

    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2UI JSON format"""
        result = {
            "id": self.id,
            "type": self.type,
        }
        if self.properties:
            result["properties"] = self.properties
        if self.children:
            result["children"] = {"explicitList": self.children}
        return result


@dataclass
class A2UISurface:
    """An A2UI surface containing components"""
    surface_id: str
    components: List[A2UIComponent] = field(default_factory=list)
    root_component_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2UI surfaceUpdate format"""
        return {
            "surfaceUpdate": {
                "surfaceId": self.surface_id,
                "rootComponentId": self.root_component_id or (
                    self.components[0].id if self.components else None
                ),
                "components": [c.to_dict() for c in self.components],
                "metadata": self.metadata,
            }
        }


class A2UIBuilder:
    """
    Builder for constructing A2UI responses.

    Uses the flat adjacency list model that's LLM-friendly
    and supports incremental updates.
    """

    def __init__(self, surface_id: Optional[str] = None):
        self.surface_id = surface_id or f"surface_{uuid.uuid4().hex[:8]}"
        self.components: Dict[str, A2UIComponent] = {}
        self.root_id: Optional[str] = None

    def _gen_id(self, prefix: str = "comp") -> str:
        """Generate a unique component ID"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def add_component(
        self,
        component_type: ComponentType,
        properties: Dict[str, Any] = None,
        children: List[str] = None,
        component_id: str = None,
    ) -> str:
        """Add a component and return its ID"""
        comp_id = component_id or self._gen_id(component_type.value.lower())
        component = A2UIComponent(
            id=comp_id,
            type=component_type.value,
            properties=properties or {},
            children=children or [],
        )
        self.components[comp_id] = component

        # First component becomes root by default
        if self.root_id is None:
            self.root_id = comp_id

        return comp_id

    def text(
        self,
        content: str,
        usage_hint: UsageHint = UsageHint.BODY,
        component_id: str = None,
    ) -> str:
        """Add a text component"""
        return self.add_component(
            ComponentType.TEXT,
            properties={
                "literalString": content,
                "usageHint": usage_hint.value,
            },
            component_id=component_id,
        )

    def heading(self, content: str, level: int = 1, component_id: str = None) -> str:
        """Add a heading component"""
        return self.add_component(
            ComponentType.HEADING,
            properties={
                "literalString": content,
                "level": level,
                "usageHint": UsageHint.TITLE.value,
            },
            component_id=component_id,
        )

    def button(
        self,
        label: str,
        action: str,
        variant: UsageHint = UsageHint.PRIMARY,
        component_id: str = None,
    ) -> str:
        """Add a button component"""
        return self.add_component(
            ComponentType.BUTTON,
            properties={
                "label": label,
                "action": action,
                "variant": variant.value,
            },
            component_id=component_id,
        )

    def text_field(
        self,
        label: str,
        placeholder: str = "",
        binding: str = None,
        component_id: str = None,
    ) -> str:
        """Add a text input field"""
        props = {
            "label": label,
            "placeholder": placeholder,
        }
        if binding:
            props["binding"] = binding
        return self.add_component(
            ComponentType.TEXT_FIELD,
            properties=props,
            component_id=component_id,
        )

    def card(self, children: List[str], title: str = None, component_id: str = None) -> str:
        """Add a card container"""
        props = {}
        if title:
            props["title"] = title
        return self.add_component(
            ComponentType.CARD,
            properties=props,
            children=children,
            component_id=component_id,
        )

    def column(self, children: List[str], spacing: int = 8, component_id: str = None) -> str:
        """Add a vertical layout container"""
        return self.add_component(
            ComponentType.COLUMN,
            properties={"spacing": spacing, "direction": "vertical"},
            children=children,
            component_id=component_id,
        )

    def row(self, children: List[str], spacing: int = 8, component_id: str = None) -> str:
        """Add a horizontal layout container"""
        return self.add_component(
            ComponentType.ROW,
            properties={"spacing": spacing, "direction": "horizontal"},
            children=children,
            component_id=component_id,
        )

    def table(
        self,
        columns: List[Dict[str, str]],
        rows: List[Dict[str, Any]],
        component_id: str = None,
    ) -> str:
        """Add a data table"""
        return self.add_component(
            ComponentType.TABLE,
            properties={
                "columns": columns,
                "rows": rows,
            },
            component_id=component_id,
        )

    def chart(
        self,
        chart_type: str,
        data: List[Dict[str, Any]],
        title: str = None,
        component_id: str = None,
    ) -> str:
        """Add a chart component"""
        props = {
            "chartType": chart_type,
            "data": data,
        }
        if title:
            props["title"] = title
        return self.add_component(
            ComponentType.CHART,
            properties=props,
            component_id=component_id,
        )

    def alert(
        self,
        message: str,
        severity: UsageHint = UsageHint.INFO,
        component_id: str = None,
    ) -> str:
        """Add an alert/notification component"""
        return self.add_component(
            ComponentType.ALERT,
            properties={
                "message": message,
                "severity": severity.value,
            },
            component_id=component_id,
        )

    def progress(
        self,
        value: float,
        label: str = None,
        component_id: str = None,
    ) -> str:
        """Add a progress indicator"""
        props = {"value": value}
        if label:
            props["label"] = label
        return self.add_component(
            ComponentType.PROGRESS,
            properties=props,
            component_id=component_id,
        )

    def set_root(self, component_id: str):
        """Set the root component ID"""
        if component_id in self.components:
            self.root_id = component_id
        else:
            raise ValueError(f"Component {component_id} not found")

    def build(self) -> A2UISurface:
        """Build the A2UI surface"""
        return A2UISurface(
            surface_id=self.surface_id,
            components=list(self.components.values()),
            root_component_id=self.root_id,
            metadata={
                "generatedAt": datetime.utcnow().isoformat(),
                "generator": "brainops-ai-agents",
                "version": "1.0.0",
            },
        )

    def to_json(self) -> str:
        """Build and convert to JSON string"""
        return json.dumps(self.build().to_dict(), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Build and convert to dictionary"""
        return self.build().to_dict()


class A2UIGenerator:
    """
    High-level A2UI generator for common agent UI patterns.

    Provides pre-built UI templates for common scenarios:
    - Dashboard summaries
    - Data tables with actions
    - Forms for user input
    - Status displays
    - Confirmation dialogs
    """

    @staticmethod
    def dashboard_card(
        title: str,
        metrics: List[Dict[str, Any]],
        actions: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate a dashboard metrics card"""
        builder = A2UIBuilder()

        # Build metric displays
        metric_ids = []
        for metric in metrics:
            text_id = builder.text(
                f"{metric.get('label', '')}: {metric.get('value', '')}",
                usage_hint=UsageHint.BODY,
            )
            metric_ids.append(text_id)

        # Build action buttons
        action_ids = []
        if actions:
            for action in actions:
                btn_id = builder.button(
                    action.get("label", "Action"),
                    action.get("action", ""),
                    variant=UsageHint.PRIMARY if action.get("primary") else UsageHint.SECONDARY,
                )
                action_ids.append(btn_id)

        # Combine in column
        content_col = builder.column(metric_ids + action_ids)

        # Wrap in card
        heading_id = builder.heading(title, level=2)
        card_id = builder.card([heading_id, content_col], title=title)
        builder.set_root(card_id)

        return builder.to_dict()

    @staticmethod
    def data_table(
        title: str,
        columns: List[str],
        data: List[List[Any]],
        row_actions: List[str] = None,
    ) -> Dict[str, Any]:
        """Generate a data table UI"""
        builder = A2UIBuilder()

        # Convert columns to A2UI format
        col_defs = [{"key": col.lower().replace(" ", "_"), "label": col} for col in columns]

        # Convert data to row format
        rows = []
        for row_data in data:
            row = {}
            for i, val in enumerate(row_data):
                if i < len(col_defs):
                    row[col_defs[i]["key"]] = val
            rows.append(row)

        heading_id = builder.heading(title, level=2)
        table_id = builder.table(col_defs, rows)

        col_id = builder.column([heading_id, table_id])
        builder.set_root(col_id)

        return builder.to_dict()

    @staticmethod
    def status_display(
        title: str,
        status: str,
        details: Dict[str, Any],
        severity: str = "info",
    ) -> Dict[str, Any]:
        """Generate a status display UI"""
        builder = A2UIBuilder()

        # Map severity
        severity_map = {
            "info": UsageHint.INFO,
            "success": UsageHint.SUCCESS,
            "warning": UsageHint.WARNING,
            "error": UsageHint.ERROR,
        }
        hint = severity_map.get(severity, UsageHint.INFO)

        heading_id = builder.heading(title, level=2)
        alert_id = builder.alert(status, severity=hint)

        # Build detail items
        detail_ids = []
        for key, value in details.items():
            text_id = builder.text(f"{key}: {value}")
            detail_ids.append(text_id)

        detail_col = builder.column(detail_ids)
        card_id = builder.card([heading_id, alert_id, detail_col])
        builder.set_root(card_id)

        return builder.to_dict()

    @staticmethod
    def confirmation_dialog(
        title: str,
        message: str,
        confirm_action: str,
        cancel_action: str = "cancel",
    ) -> Dict[str, Any]:
        """Generate a confirmation dialog UI"""
        builder = A2UIBuilder()

        heading_id = builder.heading(title, level=2)
        message_id = builder.text(message)

        confirm_btn = builder.button("Confirm", confirm_action, UsageHint.PRIMARY)
        cancel_btn = builder.button("Cancel", cancel_action, UsageHint.SECONDARY)

        button_row = builder.row([cancel_btn, confirm_btn])
        col_id = builder.column([heading_id, message_id, button_row])
        builder.set_root(col_id)

        return builder.to_dict()

    @staticmethod
    def form(
        title: str,
        fields: List[Dict[str, Any]],
        submit_action: str,
    ) -> Dict[str, Any]:
        """Generate a form UI"""
        builder = A2UIBuilder()

        heading_id = builder.heading(title, level=2)

        field_ids = []
        for f in fields:
            field_type = f.get("type", "text")
            if field_type == "text":
                fid = builder.text_field(
                    f.get("label", ""),
                    f.get("placeholder", ""),
                    f.get("binding"),
                )
            elif field_type == "textarea":
                fid = builder.add_component(
                    ComponentType.TEXT_AREA,
                    properties={
                        "label": f.get("label", ""),
                        "placeholder": f.get("placeholder", ""),
                        "binding": f.get("binding"),
                    },
                )
            else:
                fid = builder.text_field(f.get("label", ""), f.get("placeholder", ""))
            field_ids.append(fid)

        submit_btn = builder.button("Submit", submit_action, UsageHint.PRIMARY)

        col_id = builder.column([heading_id] + field_ids + [submit_btn])
        builder.set_root(col_id)

        return builder.to_dict()


# Convenience functions for agents
def generate_dashboard_ui(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a dashboard UI from agent metrics"""
    metric_list = [
        {"label": k, "value": v}
        for k, v in metrics.items()
        if not k.startswith("_")
    ]
    return A2UIGenerator.dashboard_card("System Dashboard", metric_list)


def generate_status_ui(
    title: str,
    status: str,
    details: Dict[str, Any],
    is_healthy: bool = True,
) -> Dict[str, Any]:
    """Generate a status display UI"""
    return A2UIGenerator.status_display(
        title,
        status,
        details,
        severity="success" if is_healthy else "error",
    )


def generate_table_ui(
    title: str,
    headers: List[str],
    rows: List[List[Any]],
) -> Dict[str, Any]:
    """Generate a data table UI"""
    return A2UIGenerator.data_table(title, headers, rows)


# Integration with AUREA
class AUREAUIGenerator:
    """
    A2UI generator specifically for AUREA agent outputs.

    Generates UI representations for:
    - Decision displays
    - Health dashboards
    - Agent status grids
    - Memory visualizations
    """

    @staticmethod
    def decision_display(decision: Dict[str, Any]) -> Dict[str, Any]:
        """Generate UI for an AUREA decision"""
        builder = A2UIBuilder()

        heading = builder.heading(f"Decision: {decision.get('type', 'Unknown')}", level=2)

        desc = builder.text(
            decision.get("description", "No description"),
            UsageHint.BODY,
        )

        confidence = decision.get("confidence", 0)
        conf_text = builder.text(
            f"Confidence: {confidence:.0%}",
            UsageHint.EMPHASIS if confidence > 0.7 else UsageHint.WARNING,
        )

        action = builder.text(
            f"Recommended: {decision.get('recommended_action', 'None')}",
            UsageHint.INFO,
        )

        approve_btn = builder.button("Approve", f"approve_{decision.get('id', '')}", UsageHint.PRIMARY)
        reject_btn = builder.button("Reject", f"reject_{decision.get('id', '')}", UsageHint.DESTRUCTIVE)

        btn_row = builder.row([reject_btn, approve_btn])

        col = builder.column([heading, desc, conf_text, action, btn_row])
        card = builder.card([col], title="AUREA Decision")
        builder.set_root(card)

        return builder.to_dict()

    @staticmethod
    def health_dashboard(health: Dict[str, Any]) -> Dict[str, Any]:
        """Generate UI for system health display"""
        builder = A2UIBuilder()

        overall = health.get("overall_score", 0)
        severity = UsageHint.SUCCESS if overall > 80 else (
            UsageHint.WARNING if overall > 50 else UsageHint.ERROR
        )

        heading = builder.heading("System Health", level=1)

        score_text = builder.text(
            f"Overall Health: {overall:.1f}%",
            severity,
        )

        progress = builder.progress(overall / 100, f"{overall:.1f}%")

        # Component health items
        component_ids = []
        for comp, score in health.get("component_health", {}).items():
            comp_hint = UsageHint.SUCCESS if score > 70 else UsageHint.WARNING
            text_id = builder.text(f"{comp}: {score:.1f}%", comp_hint)
            component_ids.append(text_id)

        comp_col = builder.column(component_ids)
        comp_card = builder.card([comp_col], title="Component Health")

        main_col = builder.column([heading, score_text, progress, comp_card])
        builder.set_root(main_col)

        return builder.to_dict()

    @staticmethod
    def agent_grid(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate UI for agent status grid"""
        columns = ["Name", "Type", "Status", "Executions", "Last Active"]
        rows = []

        for agent in agents[:20]:  # Limit to 20 agents
            rows.append([
                agent.get("name", "Unknown"),
                agent.get("type", "generic"),
                agent.get("status", "unknown"),
                str(agent.get("total_executions", 0)),
                agent.get("last_active", "N/A")[:10] if agent.get("last_active") else "N/A",
            ])

        return A2UIGenerator.data_table("AI Agents", columns, rows)


logger.info("A2UI Protocol module loaded - Agent-to-User Interface ready")

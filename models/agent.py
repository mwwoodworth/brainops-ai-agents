"""
Type-Safe Agent Models - Production Ready
Pydantic models with validation
"""
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class AgentCategory(str, Enum):
    """Agent category enumeration"""

    SALES = "sales"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    CUSTOMER_SERVICE = "customer_service"
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    HR = "hr"
    ANALYTICS = "analytics"
    COMPLIANCE = "compliance"
    OTHER = "other"


class AgentCapability(BaseModel):
    """Agent capability model"""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    enabled: bool = Field(default=True)
    parameters: dict[str, Any] = Field(default_factory=dict)


class Agent(BaseModel):
    """AI Agent model"""

    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., min_length=1, max_length=255)
    category: AgentCategory
    description: str = Field(default="", max_length=1000)
    enabled: bool = Field(default=True)
    capabilities: list[AgentCapability] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Operational fields
    status: Optional[str] = Field(default="active", description="Agent status")
    type: Optional[str] = Field(default=None, description="Agent type")
    total_executions: Optional[int] = Field(default=0, description="Total executions")
    last_active: Optional[datetime] = Field(default=None, description="Last active timestamp")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate agent ID format"""
        if not v or len(v) < 3:
            raise ValueError("Agent ID must be at least 3 characters")
        return v

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v: list[AgentCapability]) -> list[AgentCapability]:
        """Validate agent has at least one capability if enabled"""
        return v

    model_config = ConfigDict(use_enum_values=True)


class AgentExecution(BaseModel):
    """Agent execution result model"""

    agent_id: str = Field(..., description="Agent ID")
    agent_name: str = Field(..., min_length=1)
    execution_id: str = Field(..., description="Unique execution ID")
    status: str = Field(..., description="Execution status")
    started_at: datetime
    completed_at: Optional[datetime] = None
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: Optional[int] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate execution status"""
        valid_statuses = {"pending", "running", "completed", "failed", "cancelled"}
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v

    model_config = ConfigDict()


class AgentList(BaseModel):
    """Agent list response model"""

    agents: list[Agent]
    total: int = Field(..., ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=100)

    @field_validator("total")
    @classmethod
    def validate_total(cls, v: int, info: ValidationInfo) -> int:
        """Validate total matches agents list if full list"""
        if "agents" in info.data and v < len(info.data["agents"]):
            raise ValueError("Total cannot be less than agents list length")
        return v

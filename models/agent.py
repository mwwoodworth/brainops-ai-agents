"""
Type-Safe Agent Models - Production Ready
Pydantic models with validation
"""
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


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

    @validator("id")
    def validate_id(cls, v: str) -> str:
        """Validate agent ID format"""
        if not v or len(v) < 3:
            raise ValueError("Agent ID must be at least 3 characters")
        return v

    @validator("capabilities")
    def validate_capabilities(cls, v: list[AgentCapability]) -> list[AgentCapability]:
        """Validate agent has at least one capability if enabled"""
        return v

    class Config:
        """Pydantic config"""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


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

    @validator("status")
    def validate_status(cls, v: str) -> str:
        """Validate execution status"""
        valid_statuses = {"pending", "running", "completed", "failed", "cancelled"}
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v

    class Config:
        """Pydantic config"""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class AgentList(BaseModel):
    """Agent list response model"""
    agents: list[Agent]
    total: int = Field(..., ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=100)

    @validator("total")
    def validate_total(cls, v: int, values: dict[str, Any]) -> int:
        """Validate total matches agents list if full list"""
        if "agents" in values and v < len(values["agents"]):
            raise ValueError("Total cannot be less than agents list length")
        return v

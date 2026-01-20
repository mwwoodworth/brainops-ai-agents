
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from logistics_solver import LogisticsSolver
import logging

router = APIRouter(prefix="/logistics", tags=["Logistics"])
logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    pending = "pending"
    scheduled = "scheduled"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"
    draft = "draft"
    on_hold = "on_hold"


class JobPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"
    normal = "normal"


PRIORITY_SCORE = {
    "low": 1,
    "normal": 2,
    "medium": 2,
    "high": 3,
    "urgent": 4,
}


class Job(BaseModel):
    id: str
    customer_id: Optional[str] = None
    estimate_id: Optional[str] = None
    invoice_id: Optional[str] = None
    job_number: Optional[str] = None
    title: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[JobStatus] = None
    priority: Optional[Union[JobPriority, int]] = None
    job_type: Optional[str] = None
    scheduled_date: Optional[str] = None
    scheduled_start: Optional[str] = None
    scheduled_end: Optional[str] = None
    completed_date: Optional[str] = None
    actual_start: Optional[str] = None
    actual_end: Optional[str] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    assigned_crew: Optional[str] = None
    assigned_to: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # Scheduling-specific fields (backwards compatible)
    duration: Optional[int] = None
    skills_required: List[str] = []
    deadline: Optional[int] = None

    def to_solver_payload(self) -> Dict[str, Any]:
        duration = self.duration
        if duration is None:
            if self.estimated_hours is not None:
                duration = max(1, int(round(self.estimated_hours)))
            else:
                duration = 1

        priority_value: int
        if isinstance(self.priority, JobPriority):
            priority_value = PRIORITY_SCORE.get(self.priority.value, 1)
        elif isinstance(self.priority, str):
            priority_value = PRIORITY_SCORE.get(self.priority.lower(), 1)
        elif isinstance(self.priority, int):
            priority_value = max(1, self.priority)
        else:
            priority_value = 1

        return {
            "id": self.id,
            "duration": duration,
            "priority": priority_value,
            "skills_required": self.skills_required,
            "deadline": self.deadline,
        }

class Crew(BaseModel):
    id: str
    skills: List[str] = []
    availability_start: int = 0
    availability_end: int = 24

class ScheduleRequest(BaseModel):
    jobs: List[Job]
    crews: List[Crew]
    time_slots: int = 24

@router.post("/optimize")
async def optimize_schedule(request: ScheduleRequest):
    """
    Neuro-Symbolic Endpoint: Uses OR-Tools to solve scheduling.
    """
    try:
        solver = LogisticsSolver()
        # Convert Pydantic models to dicts
        jobs_data = [j.to_solver_payload() for j in request.jobs]
        crews_data = [c.dict() for c in request.crews]
        
        result = solver.solve_schedule(jobs_data, crews_data, range(request.time_slots))
        
        return result
    except Exception as e:
        logger.error(f"Scheduling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

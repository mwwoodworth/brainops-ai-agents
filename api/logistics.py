
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from logistics_solver import LogisticsSolver
import logging

router = APIRouter(prefix="/logistics", tags=["Logistics"])
logger = logging.getLogger(__name__)

class Job(BaseModel):
    id: str
    duration: int
    priority: int = 1
    skills_required: List[str] = []
    deadline: Optional[int] = None

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
        jobs_data = [j.dict() for j in request.jobs]
        crews_data = [c.dict() for c in request.crews]
        
        result = solver.solve_schedule(jobs_data, crews_data, range(request.time_slots))
        
        return result
    except Exception as e:
        logger.error(f"Scheduling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

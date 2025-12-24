"""
Digital Twin API Router
========================
API endpoints for the Digital Twin System - virtual replicas of production systems.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/digital-twin", tags=["Digital Twin"])

# Lazy initialization
_engine = None


def _get_engine():
    """Lazy load the Digital Twin Engine"""
    global _engine
    if _engine is None:
        try:
            from digital_twin_system import DigitalTwinEngine
            _engine = DigitalTwinEngine()
        except Exception as e:
            logger.error(f"Failed to initialize Digital Twin Engine: {e}")
            raise HTTPException(status_code=503, detail="Digital Twin Engine not available")
    return _engine


class CreateTwinRequest(BaseModel):
    source_system: str
    system_type: str
    maturity_level: str = "status"
    sync_frequency_seconds: int = 60
    initial_state: Optional[Dict[str, Any]] = None


class SimulationRequest(BaseModel):
    scenario_type: str  # traffic_spike, failure_injection, resource_constraint, load_test
    parameters: Dict[str, Any]


class UpdateTestRequest(BaseModel):
    update_type: str
    changes: Dict[str, Any]


@router.get("/status")
async def get_twin_status():
    """Get Digital Twin system status"""
    engine = _get_engine()
    return {
        "system": "digital_twin",
        "status": "operational",
        "initialized": engine._initialized,
        "active_twins": len(engine.twins),
        "capabilities": [
            "real_time_sync",
            "failure_prediction",
            "safe_simulation",
            "update_testing",
            "performance_optimization"
        ]
    }


@router.post("/twins")
async def create_twin(request: CreateTwinRequest):
    """Create a new digital twin for a production system"""
    engine = _get_engine()
    if not engine._initialized:
        await engine.initialize()

    result = await engine.create_twin(
        source_system=request.source_system,
        system_type=request.system_type,
        maturity_level=request.maturity_level,
        sync_frequency=request.sync_frequency_seconds,
        initial_state=request.initial_state
    )
    return result


@router.get("/twins")
async def list_twins():
    """List all digital twins"""
    engine = _get_engine()
    if not engine._initialized:
        await engine.initialize()

    twins_list = []
    for twin_id, twin in engine.twins.items():
        twins_list.append({
            "twin_id": twin.twin_id,
            "source_system": twin.source_system,
            "system_type": twin.system_type.value if hasattr(twin.system_type, 'value') else twin.system_type,
            "maturity_level": twin.maturity_level.value if hasattr(twin.maturity_level, 'value') else twin.maturity_level,
            "health_score": twin.health_score,
            "last_sync": twin.last_sync,
            "drift_detected": twin.drift_detected
        })

    return {"twins": twins_list, "total": len(twins_list)}


@router.get("/twins/{twin_id}")
async def get_twin(twin_id: str):
    """Get details of a specific digital twin"""
    engine = _get_engine()
    if not engine._initialized:
        await engine.initialize()

    if twin_id not in engine.twins:
        raise HTTPException(status_code=404, detail=f"Twin {twin_id} not found")

    twin = engine.twins[twin_id]
    return {
        "twin_id": twin.twin_id,
        "source_system": twin.source_system,
        "system_type": twin.system_type.value if hasattr(twin.system_type, 'value') else twin.system_type,
        "maturity_level": twin.maturity_level.value if hasattr(twin.maturity_level, 'value') else twin.maturity_level,
        "created_at": twin.created_at,
        "last_sync": twin.last_sync,
        "health_score": twin.health_score,
        "drift_detected": twin.drift_detected,
        "drift_details": twin.drift_details,
        "state_snapshot": twin.state_snapshot,
        "recent_predictions": twin.failure_predictions[-5:] if twin.failure_predictions else [],
        "recent_simulations": twin.simulation_results[-5:] if twin.simulation_results else []
    }


@router.post("/twins/{twin_id}/sync")
async def sync_twin(twin_id: str):
    """Force synchronization of a digital twin with its source system"""
    engine = _get_engine()
    if not engine._initialized:
        await engine.initialize()

    result = await engine.sync_twin(twin_id)
    return result


@router.get("/twins/{twin_id}/predictions")
async def get_predictions(twin_id: str):
    """Get failure predictions for a digital twin"""
    engine = _get_engine()
    if not engine._initialized:
        await engine.initialize()

    if twin_id not in engine.twins:
        raise HTTPException(status_code=404, detail=f"Twin {twin_id} not found")

    predictions = await engine.predict_failures(twin_id)
    return {"twin_id": twin_id, "predictions": predictions}


@router.post("/twins/{twin_id}/simulate")
async def run_simulation(twin_id: str, request: SimulationRequest):
    """Run a simulation on the digital twin"""
    engine = _get_engine()
    if not engine._initialized:
        await engine.initialize()

    if twin_id not in engine.twins:
        raise HTTPException(status_code=404, detail=f"Twin {twin_id} not found")

    result = await engine.run_simulation(
        twin_id=twin_id,
        scenario_type=request.scenario_type,
        parameters=request.parameters
    )
    return result


@router.post("/twins/{twin_id}/test-update")
async def test_update(twin_id: str, request: UpdateTestRequest):
    """Test an update on the digital twin before deploying to production"""
    engine = _get_engine()
    if not engine._initialized:
        await engine.initialize()

    if twin_id not in engine.twins:
        raise HTTPException(status_code=404, detail=f"Twin {twin_id} not found")

    result = await engine.test_update(
        twin_id=twin_id,
        update_type=request.update_type,
        changes=request.changes
    )
    return result


@router.get("/dashboard")
async def get_twin_dashboard():
    """Get a dashboard view of all digital twins"""
    engine = _get_engine()
    if not engine._initialized:
        await engine.initialize()

    # Aggregate metrics
    total_twins = len(engine.twins)
    healthy_twins = sum(1 for t in engine.twins.values() if t.health_score >= 80)
    warning_twins = sum(1 for t in engine.twins.values() if 50 <= t.health_score < 80)
    critical_twins = sum(1 for t in engine.twins.values() if t.health_score < 50)
    drifted_twins = sum(1 for t in engine.twins.values() if t.drift_detected)

    # Get all active predictions
    all_predictions = []
    for twin in engine.twins.values():
        for pred in twin.failure_predictions:
            if hasattr(pred, '__dict__'):
                all_predictions.append({**pred.__dict__, "twin_id": twin.twin_id})
            else:
                all_predictions.append({**pred, "twin_id": twin.twin_id})

    return {
        "summary": {
            "total_twins": total_twins,
            "healthy": healthy_twins,
            "warning": warning_twins,
            "critical": critical_twins,
            "drifted": drifted_twins
        },
        "active_predictions": sorted(all_predictions, key=lambda x: x.get("probability", 0), reverse=True)[:10],
        "maturity_distribution": {
            level.value: sum(1 for t in engine.twins.values()
                           if (t.maturity_level.value if hasattr(t.maturity_level, 'value') else t.maturity_level) == level.value)
            for level in ["status", "informative", "predictive", "optimization", "autonomous"]
        } if engine.twins else {}
    }

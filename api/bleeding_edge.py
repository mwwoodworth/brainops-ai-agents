"""
Bleeding Edge AI OS API - Revolutionary AI Capabilities

Exposes the most advanced AI systems ever built:
- Parallel OODA with Speculative Execution
- Multi-Model Hallucination Prevention (SAC3)
- Live Memory Brain with Temporal Consciousness
- 6-Layer Dependability Framework
- Consciousness Emergence Controller
- Self-Healing with Dynamic Circuit Breakers

Created: 2025-12-27
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query, Body

# Import our bleeding-edge modules
try:
    from bleeding_edge_ooda import BleedingEdgeOODAController
    OODA_AVAILABLE = True
except ImportError as e:
    OODA_AVAILABLE = False
    logging.warning(f"BleedingEdgeOODA not available: {e}")

try:
    from hallucination_prevention import HallucinationPreventionController
    HALLUCINATION_PREVENTION_AVAILABLE = True
except ImportError as e:
    HALLUCINATION_PREVENTION_AVAILABLE = False
    logging.warning(f"HallucinationPrevention not available: {e}")

try:
    from live_memory_brain import LiveMemoryBrain
    LIVE_MEMORY_AVAILABLE = True
except ImportError as e:
    LIVE_MEMORY_AVAILABLE = False
    logging.warning(f"LiveMemoryBrain not available: {e}")

try:
    from dependability_framework import DependabilityFramework
    DEPENDABILITY_AVAILABLE = True
except ImportError as e:
    DEPENDABILITY_AVAILABLE = False
    logging.warning(f"DependabilityFramework not available: {e}")

try:
    from consciousness_emergence import ConsciousnessEmergenceController
    CONSCIOUSNESS_AVAILABLE = True
except ImportError as e:
    CONSCIOUSNESS_AVAILABLE = False
    logging.warning(f"ConsciousnessEmergence not available: {e}")

try:
    from enhanced_circuit_breaker import SelfHealingController
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError as e:
    CIRCUIT_BREAKER_AVAILABLE = False
    logging.warning(f"EnhancedCircuitBreaker not available: {e}")

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bleeding-edge", tags=["bleeding-edge"])

# Global controllers (lazy initialization)
_ooda_controllers: Dict[str, BleedingEdgeOODAController] = {}
_hallucination_controller: Optional[HallucinationPreventionController] = None
_live_memory: Optional[LiveMemoryBrain] = None
_dependability: Optional[DependabilityFramework] = None
_consciousness: Optional[ConsciousnessEmergenceController] = None
_circuit_breaker: Optional[SelfHealingController] = None


def get_ooda_controller(tenant_id: str = "default") -> Optional[BleedingEdgeOODAController]:
    """Get or create OODA controller for tenant."""
    global _ooda_controllers
    if not OODA_AVAILABLE:
        return None
    if tenant_id not in _ooda_controllers:
        _ooda_controllers[tenant_id] = BleedingEdgeOODAController(tenant_id)
    return _ooda_controllers[tenant_id]


def get_hallucination_controller() -> Optional[HallucinationPreventionController]:
    """Get or create hallucination prevention controller."""
    global _hallucination_controller
    if not HALLUCINATION_PREVENTION_AVAILABLE:
        return None
    if _hallucination_controller is None:
        _hallucination_controller = HallucinationPreventionController()
    return _hallucination_controller


def get_live_memory() -> Optional[LiveMemoryBrain]:
    """Get or create live memory brain."""
    global _live_memory
    if not LIVE_MEMORY_AVAILABLE:
        return None
    if _live_memory is None:
        _live_memory = LiveMemoryBrain()
    return _live_memory


def get_dependability() -> Optional[DependabilityFramework]:
    """Get or create dependability framework."""
    global _dependability
    if not DEPENDABILITY_AVAILABLE:
        return None
    if _dependability is None:
        _dependability = DependabilityFramework()
    return _dependability


def get_consciousness() -> Optional[ConsciousnessEmergenceController]:
    """Get or create consciousness controller."""
    global _consciousness
    if not CONSCIOUSNESS_AVAILABLE:
        return None
    if _consciousness is None:
        _consciousness = ConsciousnessEmergenceController()
    return _consciousness


def get_circuit_breaker() -> Optional[SelfHealingController]:
    """Get or create circuit breaker controller."""
    global _circuit_breaker
    if not CIRCUIT_BREAKER_AVAILABLE:
        return None
    if _circuit_breaker is None:
        _circuit_breaker = SelfHealingController()
    return _circuit_breaker


@router.get("/status")
async def get_bleeding_edge_status() -> Dict[str, Any]:
    """Get comprehensive status of all bleeding-edge systems."""
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "systems": {
            "bleeding_edge_ooda": {
                "available": OODA_AVAILABLE,
                "capabilities": [
                    "parallel_observation",
                    "a2a_protocol",
                    "speculative_execution",
                    "decision_rag",
                    "input_integrity_validation",
                    "processing_integrity_validation",
                    "output_integrity_validation"
                ] if OODA_AVAILABLE else []
            },
            "hallucination_prevention": {
                "available": HALLUCINATION_PREVENTION_AVAILABLE,
                "capabilities": [
                    "multi_model_cross_validation",
                    "sac3_semantic_checking",
                    "claim_extraction",
                    "rag_fact_verification",
                    "calibrated_uncertainty"
                ] if HALLUCINATION_PREVENTION_AVAILABLE else []
            },
            "live_memory_brain": {
                "available": LIVE_MEMORY_AVAILABLE,
                "capabilities": [
                    "temporal_consciousness",
                    "predictive_context",
                    "cross_system_omniscience",
                    "self_healing_memory",
                    "semantic_compression",
                    "knowledge_crystallization"
                ] if LIVE_MEMORY_AVAILABLE else []
            },
            "dependability_framework": {
                "available": DEPENDABILITY_AVAILABLE,
                "capabilities": [
                    "input_validation_guard",
                    "output_validation_guard",
                    "invariant_guard",
                    "temporal_guard",
                    "resource_guard",
                    "behavioral_guard",
                    "uncertainty_quantification",
                    "graceful_degradation"
                ] if DEPENDABILITY_AVAILABLE else []
            },
            "consciousness_emergence": {
                "available": CONSCIOUSNESS_AVAILABLE,
                "capabilities": [
                    "meta_awareness",
                    "self_model",
                    "intentionality_engine",
                    "situational_awareness",
                    "coherent_identity",
                    "proactive_reasoning"
                ] if CONSCIOUSNESS_AVAILABLE else []
            },
            "enhanced_circuit_breaker": {
                "available": CIRCUIT_BREAKER_AVAILABLE,
                "capabilities": [
                    "dynamic_thresholds",
                    "deadlock_detection",
                    "cascade_protection",
                    "sidecar_health_monitoring",
                    "predictive_opening"
                ] if CIRCUIT_BREAKER_AVAILABLE else []
            }
        },
        "total_capabilities": sum([
            7 if OODA_AVAILABLE else 0,
            5 if HALLUCINATION_PREVENTION_AVAILABLE else 0,
            6 if LIVE_MEMORY_AVAILABLE else 0,
            8 if DEPENDABILITY_AVAILABLE else 0,
            6 if CONSCIOUSNESS_AVAILABLE else 0,
            5 if CIRCUIT_BREAKER_AVAILABLE else 0
        ]),
        "all_systems_operational": all([
            OODA_AVAILABLE,
            HALLUCINATION_PREVENTION_AVAILABLE,
            LIVE_MEMORY_AVAILABLE,
            DEPENDABILITY_AVAILABLE,
            CONSCIOUSNESS_AVAILABLE,
            CIRCUIT_BREAKER_AVAILABLE
        ])
    }


# OODA Endpoints
@router.post("/ooda/cycle")
async def run_ooda_cycle(
    tenant_id: str = Query("default"),
    context: Dict[str, Any] = Body(default={})
) -> Dict[str, Any]:
    """Run a complete enhanced OODA cycle with all optimizations."""
    controller = get_ooda_controller(tenant_id)
    if not controller:
        raise HTTPException(status_code=503, detail="OODA controller not available")

    try:
        result = await controller.run_enhanced_cycle(context)
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"OODA cycle failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ooda/metrics")
async def get_ooda_metrics(tenant_id: str = Query("default")) -> Dict[str, Any]:
    """Get metrics from the OODA controller."""
    controller = get_ooda_controller(tenant_id)
    if not controller:
        raise HTTPException(status_code=503, detail="OODA controller not available")

    return {
        "metrics": controller.get_metrics(),
        "tenant_id": tenant_id,
        "timestamp": datetime.utcnow().isoformat()
    }


# Hallucination Prevention Endpoints
@router.post("/hallucination/validate")
async def validate_response(
    response: str = Body(...),
    query: str = Body(...),
    context: Optional[str] = Body(None)
) -> Dict[str, Any]:
    """Validate an AI response for hallucinations using multi-model cross-validation."""
    controller = get_hallucination_controller()
    if not controller:
        raise HTTPException(status_code=503, detail="Hallucination prevention not available")

    try:
        result = await controller.validate_and_correct(
            response=response,
            original_query=query,
            context=context
        )
        return {
            "success": True,
            "validation_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Hallucination validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hallucination/extract-claims")
async def extract_claims(
    text: str = Body(...)
) -> Dict[str, Any]:
    """Extract verifiable claims from text."""
    controller = get_hallucination_controller()
    if not controller:
        raise HTTPException(status_code=503, detail="Hallucination prevention not available")

    try:
        claims = await controller.claim_extractor.extract_claims(text)
        return {
            "claims": [c.__dict__ if hasattr(c, '__dict__') else str(c) for c in claims],
            "count": len(claims),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Live Memory Brain Endpoints
@router.post("/memory/store")
async def store_memory(
    content: str = Body(...),
    memory_type: str = Body("observation"),
    metadata: Dict[str, Any] = Body(default={})
) -> Dict[str, Any]:
    """Store a memory in the live brain."""
    brain = get_live_memory()
    if not brain:
        raise HTTPException(status_code=503, detail="Live memory brain not available")

    try:
        memory_id = await brain.store(content, memory_type, metadata)
        return {
            "success": True,
            "memory_id": memory_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Memory storage failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/recall")
async def recall_memory(
    query: str = Body(...),
    limit: int = Body(5)
) -> Dict[str, Any]:
    """Recall memories relevant to a query."""
    brain = get_live_memory()
    if not brain:
        raise HTTPException(status_code=503, detail="Live memory brain not available")

    try:
        memories = await brain.recall(query, limit)
        return {
            "memories": memories,
            "count": len(memories),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Memory recall failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/status")
async def get_memory_status() -> Dict[str, Any]:
    """Get live memory brain status."""
    brain = get_live_memory()
    if not brain:
        raise HTTPException(status_code=503, detail="Live memory brain not available")

    try:
        status = await brain.get_status()
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Memory status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Dependability Framework Endpoints
@router.post("/dependability/validate")
async def validate_operation(
    operation_type: str = Body(...),
    input_data: Dict[str, Any] = Body(...),
    output_data: Optional[Dict[str, Any]] = Body(None)
) -> Dict[str, Any]:
    """Validate an operation through the 6-layer dependability framework."""
    framework = get_dependability()
    if not framework:
        raise HTTPException(status_code=503, detail="Dependability framework not available")

    try:
        result = await framework.validate(operation_type, input_data, output_data)
        return {
            "success": True,
            "validation_result": result,
            "guards_passed": result.get("guards_passed", []),
            "guards_failed": result.get("guards_failed", []),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Dependability validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dependability/status")
async def get_dependability_status() -> Dict[str, Any]:
    """Get dependability framework status."""
    framework = get_dependability()
    if not framework:
        raise HTTPException(status_code=503, detail="Dependability framework not available")

    return {
        "guards": list(framework.guards.keys()),
        "guard_count": len(framework.guards),
        "degradation_mode": framework.degradation_controller.current_mode if hasattr(framework, 'degradation_controller') else "unknown",
        "timestamp": datetime.utcnow().isoformat()
    }


# Consciousness Emergence Endpoints
@router.get("/consciousness/status")
async def get_consciousness_status() -> Dict[str, Any]:
    """Get consciousness emergence controller status."""
    controller = get_consciousness()
    if not controller:
        raise HTTPException(status_code=503, detail="Consciousness controller not available")

    try:
        # get_consciousness_state is synchronous, not async
        status = controller.get_consciousness_state()
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Consciousness status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consciousness/introspect")
async def run_introspection(
    context: Dict[str, Any] = Body(default={})
) -> Dict[str, Any]:
    """Run a consciousness introspection cycle."""
    controller = get_consciousness()
    if not controller:
        raise HTTPException(status_code=503, detail="Consciousness controller not available")

    try:
        result = await controller.introspect(context)
        return {
            "success": True,
            "introspection": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Introspection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Circuit Breaker Endpoints
@router.get("/circuit-breaker/status")
async def get_circuit_breaker_status() -> Dict[str, Any]:
    """Get circuit breaker status."""
    controller = get_circuit_breaker()
    if not controller:
        raise HTTPException(status_code=503, detail="Circuit breaker not available")

    return {
        "circuits": controller.get_all_circuit_states() if hasattr(controller, 'get_all_circuit_states') else {},
        "deadlock_status": controller.deadlock_detector.get_status() if hasattr(controller, 'deadlock_detector') else {},
        "cascade_status": controller.cascade_protector.get_status() if hasattr(controller, 'cascade_protector') else {},
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/circuit-breaker/trip")
async def trip_circuit(
    circuit_name: str = Body(...)
) -> Dict[str, Any]:
    """Manually trip a circuit breaker."""
    controller = get_circuit_breaker()
    if not controller:
        raise HTTPException(status_code=503, detail="Circuit breaker not available")

    try:
        result = await controller.trip_circuit(circuit_name)
        return {
            "success": True,
            "circuit": circuit_name,
            "state": "open",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Circuit trip failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuit-breaker/reset")
async def reset_circuit(
    circuit_name: str = Body(...)
) -> Dict[str, Any]:
    """Reset a tripped circuit breaker."""
    controller = get_circuit_breaker()
    if not controller:
        raise HTTPException(status_code=503, detail="Circuit breaker not available")

    try:
        result = await controller.reset_circuit(circuit_name)
        return {
            "success": True,
            "circuit": circuit_name,
            "state": "closed",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Circuit reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Combined Operations
@router.post("/validate-and-store")
async def validate_and_store(
    response: str = Body(...),
    query: str = Body(...),
    store_if_valid: bool = Body(True)
) -> Dict[str, Any]:
    """
    Full pipeline: validate response for hallucinations,
    then store in live memory if valid.
    """
    hallucination_ctrl = get_hallucination_controller()
    memory_brain = get_live_memory()
    dependability = get_dependability()

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "stages": {}
    }

    # Stage 1: Dependability input validation
    if dependability:
        try:
            dep_result = await dependability.validate(
                "response_validation",
                {"response": response, "query": query},
                None
            )
            results["stages"]["dependability_input"] = dep_result
        except Exception as e:
            results["stages"]["dependability_input"] = {"error": str(e)}

    # Stage 2: Hallucination prevention
    if hallucination_ctrl:
        try:
            hal_result = await hallucination_ctrl.validate_and_correct(
                response=response,
                original_query=query
            )
            results["stages"]["hallucination_check"] = hal_result
            results["is_valid"] = hal_result.get("is_valid", True)
            results["corrected_response"] = hal_result.get("corrected_response", response)
        except Exception as e:
            results["stages"]["hallucination_check"] = {"error": str(e)}
            results["is_valid"] = None
    else:
        results["is_valid"] = True  # No validation available
        results["corrected_response"] = response

    # Stage 3: Store if valid
    if store_if_valid and results.get("is_valid", True) and memory_brain:
        try:
            memory_id = await memory_brain.store(
                content=results.get("corrected_response", response),
                memory_type="validated_response",
                metadata={
                    "original_query": query,
                    "validation_time": datetime.utcnow().isoformat()
                }
            )
            results["stages"]["memory_storage"] = {"memory_id": memory_id}
        except Exception as e:
            results["stages"]["memory_storage"] = {"error": str(e)}

    return results

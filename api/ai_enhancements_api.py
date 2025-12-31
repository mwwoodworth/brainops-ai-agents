#!/usr/bin/env python3
"""
AI System Enhancements API
===========================
Endpoints for health scoring, alerting, correlation, and WebSocket streaming.

Features:
- Module health scores and aggregate health
- Real-time alerting with threshold management
- Event correlation and chain tracking
- WebSocket streaming for live events
- Auto-recovery status and triggers
- Enhanced learning predictions
"""

import logging
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/enhanced", tags=["AI Enhanced Systems"])

# Lazy imports
_wiring_bridge = None
_websocket_clients: List[WebSocket] = []


def get_wiring_bridge():
    global _wiring_bridge
    if _wiring_bridge is None:
        try:
            from ai_system_enhancements import ModuleWiringBridge
            _wiring_bridge = ModuleWiringBridge.get_instance()
            _wiring_bridge.connect_observability()
            _wiring_bridge.connect_integration()
        except Exception as e:
            logger.error(f"Failed to load wiring bridge: {e}")
    return _wiring_bridge


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@router.get("/health")
async def get_enhanced_health():
    """Get comprehensive health status for all modules"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    return {
        "aggregate": bridge.health_scorer.get_aggregate_health(),
        "modules": {
            name: {
                "status": health.status.value,
                "score": health.score,
                "error_rate": health.error_rate,
                "latency_p95_ms": health.latency_p95_ms,
                "availability": health.availability,
                "last_check": health.last_check.isoformat(),
                "issues": health.issues
            }
            for name, health in bridge.health_scorer.get_all_health().items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/health/{module}")
async def get_module_health(module: str):
    """Get health for a specific module"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    health = bridge.health_scorer.get_module_health(module)
    if not health:
        return {"error": f"Module {module} not found"}

    return {
        "module": module,
        "status": health.status.value,
        "score": health.score,
        "error_rate": health.error_rate,
        "latency_p95_ms": health.latency_p95_ms,
        "availability": health.availability,
        "last_check": health.last_check.isoformat(),
        "issues": health.issues,
        "metrics": health.metrics
    }


@router.get("/health/{module}/history")
async def get_module_health_history(module: str, limit: int = Query(default=100, le=500)):
    """Get health history for a module"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    return {
        "module": module,
        "history": bridge.health_scorer.get_health_history(module, limit)
    }


@router.post("/health/{module}/update")
async def update_module_health(
    module: str,
    error_rate: float = Body(default=0.0),
    latency_p95_ms: float = Body(default=0.0),
    request_count: int = Body(default=0),
    success_count: int = Body(default=0),
    custom_metrics: Dict[str, float] = Body(default={})
):
    """Update health metrics for a module"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    health = bridge.update_module_metrics(
        module=module,
        error_rate=error_rate,
        latency_p95_ms=latency_p95_ms,
        request_count=request_count,
        success_count=success_count,
        custom_metrics=custom_metrics
    )

    # Broadcast update to WebSocket clients
    await broadcast_event({
        "type": "health_update",
        "module": module,
        "score": health.score,
        "status": health.status.value
    })

    return {
        "module": module,
        "status": health.status.value,
        "score": health.score
    }


# =============================================================================
# ALERTING ENDPOINTS
# =============================================================================

@router.get("/alerts")
async def get_alerts(severity: Optional[str] = None):
    """Get active alerts"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    from ai_system_enhancements import AlertSeverity

    sev = None
    if severity:
        try:
            sev = AlertSeverity(severity)
        except ValueError as exc:
            logger.debug("Invalid alert severity %s: %s", severity, exc)

    alerts = bridge.alerting.get_active_alerts(sev)
    return {
        "active_count": len(alerts),
        "alerts": [
            {
                "id": a.id,
                "type": a.alert_type,
                "severity": a.severity.value,
                "module": a.module,
                "metric": a.metric,
                "current_value": a.current_value,
                "threshold": a.threshold,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
                "acknowledged": a.acknowledged
            }
            for a in alerts
        ]
    }


@router.get("/alerts/history")
async def get_alert_history(limit: int = Query(default=100, le=500)):
    """Get alert history"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    return {
        "history": bridge.alerting.get_alert_history(limit)
    }


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    success = bridge.alerting.acknowledge_alert(alert_id)
    if success:
        await broadcast_event({
            "type": "alert_acknowledged",
            "alert_id": alert_id
        })
    return {"success": success, "alert_id": alert_id}


@router.post("/alerts/thresholds")
async def register_threshold(
    name: str = Body(...),
    module: str = Body(...),
    metric: str = Body(...),
    warning_threshold: Optional[float] = Body(default=None),
    error_threshold: Optional[float] = Body(default=None),
    critical_threshold: Optional[float] = Body(default=None),
    comparison: str = Body(default="gt")
):
    """Register a new alerting threshold"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    bridge.alerting.register_threshold(
        name=name,
        module=module,
        metric=metric,
        warning_threshold=warning_threshold,
        error_threshold=error_threshold,
        critical_threshold=critical_threshold,
        comparison=comparison
    )

    return {"success": True, "threshold": name}


# =============================================================================
# EVENT CORRELATION ENDPOINTS
# =============================================================================

@router.get("/correlation/chains")
async def get_event_chains():
    """Get active event chains"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    chains = bridge.correlator.get_active_chains()
    return {
        "active_count": len(chains),
        "chains": [
            {
                "chain_id": c.chain_id,
                "root_event": c.root_event_id,
                "event_count": len(c.events),
                "modules": list(c.modules_involved),
                "start_time": c.start_time.isoformat(),
                "status": c.status
            }
            for c in chains
        ]
    }


@router.get("/correlation/stats")
async def get_correlation_stats():
    """Get event correlation statistics"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    return bridge.correlator.get_chain_stats()


@router.post("/correlation/complete/{chain_id}")
async def complete_chain(chain_id: str, status: str = Body(default="completed")):
    """Mark an event chain as complete"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    bridge.correlator.complete_chain(chain_id, status)
    return {"success": True, "chain_id": chain_id, "status": status}


# =============================================================================
# AUTO-RECOVERY ENDPOINTS
# =============================================================================

@router.get("/recovery/history")
async def get_recovery_history(limit: int = Query(default=100, le=500)):
    """Get auto-recovery history"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    return {
        "history": bridge.auto_recovery.get_recovery_history(limit)
    }


@router.post("/recovery/trigger/{module}")
async def trigger_recovery_check(
    module: str,
    metrics: Dict[str, float] = Body(...)
):
    """Manually trigger recovery check for a module"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    result = await bridge.check_and_recover(module, metrics)

    if result:
        await broadcast_event({
            "type": "recovery_triggered",
            "module": module,
            "action": result.get("action"),
            "success": result.get("success")
        })

    return {
        "triggered": result is not None,
        "result": result
    }


@router.post("/recovery/rules")
async def register_recovery_rule(
    name: str = Body(...),
    module: str = Body(...),
    trigger_condition: str = Body(...),
    action: str = Body(...),
    cooldown_seconds: int = Body(default=300),
    max_attempts: int = Body(default=3)
):
    """Register a new recovery rule"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    from ai_system_enhancements import RecoveryRule, RecoveryAction

    try:
        recovery_action = RecoveryAction(action)
    except ValueError:
        return {"error": f"Invalid action: {action}"}

    bridge.auto_recovery.register_rule(RecoveryRule(
        name=name,
        trigger_condition=trigger_condition,
        action=recovery_action,
        module=module,
        cooldown_seconds=cooldown_seconds,
        max_attempts=max_attempts
    ))

    return {"success": True, "rule": name}


# =============================================================================
# ENHANCED LEARNING ENDPOINTS
# =============================================================================

@router.get("/learning/stats")
async def get_learning_stats():
    """Get enhanced learning statistics"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    return bridge.enhanced_learning.get_learning_stats()


@router.post("/learning/predict")
async def predict_outcome(
    operation_type: str = Body(...),
    context: Dict[str, Any] = Body(...)
):
    """Predict outcome for an operation"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    prediction, matches = bridge.enhanced_learning.predict_outcome(operation_type, context)

    return {
        "operation_type": operation_type,
        "prediction": prediction,
        "confidence": "high" if prediction > 0.7 or prediction < 0.3 else "medium",
        "similar_patterns": [
            {
                "pattern_id": m.pattern_id,
                "similarity": m.similarity,
                "outcome": m.outcome,
                "success_rate": m.success_rate
            }
            for m in matches
        ]
    }


@router.post("/learning/record")
async def record_learning(
    operation_type: str = Body(...),
    context: Dict[str, Any] = Body(...),
    outcome: str = Body(...),
    success: bool = Body(...)
):
    """Record a learning outcome"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    bridge.enhanced_learning.learn_pattern(operation_type, context, outcome, success)

    return {
        "success": True,
        "operation_type": operation_type,
        "recorded": True
    }


# =============================================================================
# SYSTEM STATUS ENDPOINT
# =============================================================================

@router.get("/status")
async def get_system_status():
    """Get comprehensive enhanced system status"""
    bridge = get_wiring_bridge()
    if not bridge:
        return {"error": "Enhancement system not available"}

    return bridge.get_system_status()


# =============================================================================
# WEBSOCKET STREAMING
# =============================================================================

async def broadcast_event(event: Dict[str, Any]):
    """Broadcast event to all connected WebSocket clients"""
    global _websocket_clients
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    message = json.dumps(event)

    disconnected = []
    for client in _websocket_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.append(client)

    # Clean up disconnected clients
    for client in disconnected:
        _websocket_clients.remove(client)


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.

    Events streamed:
    - health_update: Module health changes
    - alert_triggered: New alerts
    - alert_acknowledged: Alert acknowledgments
    - recovery_triggered: Auto-recovery actions
    - event_correlated: New event correlations
    """
    await websocket.accept()
    _websocket_clients.append(websocket)

    logger.info(f"WebSocket client connected. Total clients: {len(_websocket_clients)}")

    # Send initial status
    bridge = get_wiring_bridge()
    if bridge:
        await websocket.send_json({
            "type": "connected",
            "status": bridge.get_system_status(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    try:
        while True:
            # Keep connection alive with periodic pings
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle client messages if needed
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in _websocket_clients:
            _websocket_clients.remove(websocket)


# =============================================================================
# COMPREHENSIVE SMOKE TEST
# =============================================================================

@router.post("/smoke-test")
async def run_enhanced_smoke_test():
    """
    Run comprehensive smoke test for all enhanced systems.
    Tests health scoring, alerting, correlation, learning, and recovery.
    """
    import uuid as uuid_module

    bridge = get_wiring_bridge()
    if not bridge:
        return {
            "success": False,
            "error": "Enhancement system not available"
        }

    test_id = str(uuid_module.uuid4())[:8]
    results = {
        "test_id": test_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {}
    }

    # Test 1: Health Scoring
    try:
        health = bridge.update_module_metrics(
            module="smoke_test",
            error_rate=0.02,
            latency_p95_ms=150,
            request_count=100,
            success_count=98
        )
        results["tests"]["health_scoring"] = {
            "success": health.score > 0,
            "module": "smoke_test",
            "score": health.score,
            "status": health.status.value
        }
    except Exception as e:
        results["tests"]["health_scoring"] = {"success": False, "error": str(e)}

    # Test 2: Alerting
    try:
        bridge.alerting.register_threshold(
            f"smoke_test_{test_id}",
            module="smoke_test",
            metric="test_value",
            warning_threshold=50
        )
        alert = bridge.alerting.check_value(f"smoke_test_{test_id}", 75)
        results["tests"]["alerting"] = {
            "success": alert is not None,
            "alert_triggered": alert is not None,
            "severity": alert.severity.value if alert else None
        }
    except Exception as e:
        results["tests"]["alerting"] = {"success": False, "error": str(e)}

    # Test 3: Event Correlation
    try:
        chain = bridge.correlator.correlate_event(
            event_id=f"test_event_{test_id}",
            event_type="smoke_test.event",
            module="smoke_test",
            correlation_id=f"chain_{test_id}"
        )
        results["tests"]["correlation"] = {
            "success": chain is not None,
            "chain_id": chain.chain_id if chain else None,
            "event_count": len(chain.events) if chain else 0
        }
        # Complete the chain
        if chain:
            bridge.correlator.complete_chain(chain.chain_id)
    except Exception as e:
        results["tests"]["correlation"] = {"success": False, "error": str(e)}

    # Test 4: Enhanced Learning
    try:
        bridge.enhanced_learning.learn_pattern(
            operation_type=f"smoke_test_{test_id}",
            context={"test": True, "value": 42},
            outcome="success",
            success=True
        )
        prediction, matches = bridge.enhanced_learning.predict_outcome(
            f"smoke_test_{test_id}",
            {"test": True, "value": 42}
        )
        results["tests"]["learning"] = {
            "success": True,
            "prediction": prediction,
            "matches_found": len(matches)
        }
    except Exception as e:
        results["tests"]["learning"] = {"success": False, "error": str(e)}

    # Test 5: System Status
    try:
        status = bridge.get_system_status()
        results["tests"]["system_status"] = {
            "success": "health" in status,
            "modules_tracked": status.get("health", {}).get("modules", 0)
        }
    except Exception as e:
        results["tests"]["system_status"] = {"success": False, "error": str(e)}

    # Calculate overall success
    all_passed = all(t.get("success", False) for t in results["tests"].values())
    results["success"] = all_passed
    results["passed"] = sum(1 for t in results["tests"].values() if t.get("success", False))
    results["total"] = len(results["tests"])

    return results

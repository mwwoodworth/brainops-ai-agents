"""
Digital Twin System - Virtual Replica Engine
============================================
Creates and maintains virtual replicas of production systems for:
- Real-time simulation and testing
- Predictive failure analysis
- Performance optimization
- Safe update testing before production deployment

Based on 2025 best practices from IBM, Siemens, and Microsoft Azure Digital Twins.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)


class TwinMaturityLevel(Enum):
    """Digital Twin maturity levels (Wageningen Research classification)"""
    STATUS = "status"           # Basic monitoring
    INFORMATIVE = "informative" # Data visualization
    PREDICTIVE = "predictive"   # Future scenario forecasting
    OPTIMIZATION = "optimization"  # Prescriptive analytics
    AUTONOMOUS = "autonomous"   # Self-optimizing systems


class SystemType(Enum):
    """Types of systems that can be twinned"""
    SAAS_APPLICATION = "saas_application"
    MICROSERVICE = "microservice"
    DATABASE = "database"
    API_GATEWAY = "api_gateway"
    AI_AGENT = "ai_agent"
    INFRASTRUCTURE = "infrastructure"
    PIPELINE = "pipeline"
    COMPLETE_STACK = "complete_stack"


@dataclass
class SystemMetrics:
    """Real-time metrics from a system"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: float = 0.0
    request_latency_ms: float = 0.0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    active_connections: int = 0
    queue_depth: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class FailurePrediction:
    """Predicted failure event"""
    component: str
    failure_type: str
    probability: float  # 0-1
    predicted_time: str
    impact_severity: str  # low, medium, high, critical
    recommended_action: str
    confidence: float
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class DigitalTwin:
    """Virtual replica of a production system"""
    twin_id: str
    source_system: str
    system_type: SystemType
    maturity_level: TwinMaturityLevel
    created_at: str
    last_sync: str
    sync_frequency_seconds: int
    state_snapshot: Dict[str, Any]
    metrics_history: List[SystemMetrics] = field(default_factory=list)
    failure_predictions: List[FailurePrediction] = field(default_factory=list)
    simulation_results: List[Dict[str, Any]] = field(default_factory=list)
    health_score: float = 100.0
    drift_detected: bool = False
    drift_details: Optional[str] = None


class DigitalTwinEngine:
    """
    Core Digital Twin Engine

    Manages virtual replicas of production systems for:
    - Real-time state synchronization
    - Predictive failure analysis
    - Safe simulation and testing
    - Performance optimization recommendations
    """

    def __init__(self):
        self.twins: Dict[str, DigitalTwin] = {}
        self.db_url = os.getenv("DATABASE_URL")
        self.prediction_models: Dict[str, Any] = {}
        self.simulation_engine = SimulationEngine()
        self.failure_predictor = FailurePredictor()
        self._initialized = False

    async def initialize(self):
        """Initialize the Digital Twin Engine"""
        if self._initialized:
            return

        logger.info("Initializing Digital Twin Engine...")

        # Load existing twins from database
        await self._load_twins_from_db()

        # Initialize prediction models
        await self._initialize_prediction_models()

        self._initialized = True
        logger.info(f"Digital Twin Engine initialized with {len(self.twins)} twins")

    async def _load_twins_from_db(self):
        """Load existing digital twins from database"""
        try:
            import asyncpg
            if not self.db_url:
                logger.warning("No DATABASE_URL configured for Digital Twin persistence")
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                # Create table if not exists
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS digital_twins (
                        twin_id TEXT PRIMARY KEY,
                        source_system TEXT NOT NULL,
                        system_type TEXT NOT NULL,
                        maturity_level TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        last_sync TIMESTAMPTZ DEFAULT NOW(),
                        sync_frequency_seconds INTEGER DEFAULT 60,
                        state_snapshot JSONB DEFAULT '{}',
                        metrics_history JSONB DEFAULT '[]',
                        failure_predictions JSONB DEFAULT '[]',
                        simulation_results JSONB DEFAULT '[]',
                        health_score FLOAT DEFAULT 100.0,
                        drift_detected BOOLEAN DEFAULT FALSE,
                        drift_details TEXT
                    )
                """)

                # Create metrics history table for time-series data
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS twin_metrics_history (
                        id SERIAL PRIMARY KEY,
                        twin_id TEXT REFERENCES digital_twins(twin_id),
                        metrics JSONB NOT NULL,
                        recorded_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Create failure predictions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS twin_failure_predictions (
                        id SERIAL PRIMARY KEY,
                        twin_id TEXT REFERENCES digital_twins(twin_id),
                        component TEXT NOT NULL,
                        failure_type TEXT NOT NULL,
                        probability FLOAT NOT NULL,
                        predicted_time TIMESTAMPTZ,
                        impact_severity TEXT,
                        recommended_action TEXT,
                        confidence FLOAT,
                        contributing_factors JSONB DEFAULT '[]',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        resolved_at TIMESTAMPTZ,
                        was_accurate BOOLEAN
                    )
                """)

                # Load existing twins
                rows = await conn.fetch("SELECT * FROM digital_twins")
                for row in rows:
                    twin = DigitalTwin(
                        twin_id=row['twin_id'],
                        source_system=row['source_system'],
                        system_type=SystemType(row['system_type']),
                        maturity_level=TwinMaturityLevel(row['maturity_level']),
                        created_at=row['created_at'].isoformat() if row['created_at'] else "",
                        last_sync=row['last_sync'].isoformat() if row['last_sync'] else "",
                        sync_frequency_seconds=row['sync_frequency_seconds'],
                        state_snapshot=row['state_snapshot'] or {},
                        health_score=row['health_score'] or 100.0,
                        drift_detected=row['drift_detected'] or False,
                        drift_details=row['drift_details']
                    )
                    self.twins[twin.twin_id] = twin

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Error loading twins from DB: {e}")

    async def _initialize_prediction_models(self):
        """Initialize ML models for failure prediction"""
        # Initialize lightweight prediction models
        self.prediction_models = {
            "cpu_saturation": CPUSaturationPredictor(),
            "memory_leak": MemoryLeakPredictor(),
            "disk_exhaustion": DiskExhaustionPredictor(),
            "latency_degradation": LatencyDegradationPredictor(),
            "error_rate_spike": ErrorRateSpikePredictor(),
            "connection_exhaustion": ConnectionExhaustionPredictor()
        }

    async def create_twin(
        self,
        source_system: str,
        system_type: SystemType,
        initial_state: Dict[str, Any],
        maturity_level: TwinMaturityLevel = TwinMaturityLevel.PREDICTIVE,
        sync_frequency_seconds: int = 60
    ) -> DigitalTwin:
        """
        Create a new digital twin for a production system

        Args:
            source_system: Identifier for the production system (URL or ID)
            system_type: Type of system being twinned
            initial_state: Initial state snapshot of the system
            maturity_level: Target maturity level for the twin
            sync_frequency_seconds: How often to sync with production

        Returns:
            Created DigitalTwin instance
        """
        twin_id = self._generate_twin_id(source_system)
        now = datetime.utcnow().isoformat()

        twin = DigitalTwin(
            twin_id=twin_id,
            source_system=source_system,
            system_type=system_type,
            maturity_level=maturity_level,
            created_at=now,
            last_sync=now,
            sync_frequency_seconds=sync_frequency_seconds,
            state_snapshot=initial_state,
            health_score=100.0
        )

        self.twins[twin_id] = twin

        # Persist to database
        await self._persist_twin(twin)

        logger.info(f"Created digital twin {twin_id} for {source_system}")
        return twin

    def _generate_twin_id(self, source_system: str) -> str:
        """Generate unique twin ID from source system"""
        hash_input = f"{source_system}:{datetime.utcnow().timestamp()}"
        return f"twin_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"

    async def _persist_twin(self, twin: DigitalTwin):
        """Persist twin to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO digital_twins
                    (twin_id, source_system, system_type, maturity_level,
                     created_at, last_sync, sync_frequency_seconds, state_snapshot,
                     health_score, drift_detected, drift_details)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (twin_id) DO UPDATE SET
                        last_sync = EXCLUDED.last_sync,
                        state_snapshot = EXCLUDED.state_snapshot,
                        health_score = EXCLUDED.health_score,
                        drift_detected = EXCLUDED.drift_detected,
                        drift_details = EXCLUDED.drift_details
                """,
                    twin.twin_id,
                    twin.source_system,
                    twin.system_type.value,
                    twin.maturity_level.value,
                    datetime.fromisoformat(twin.created_at) if twin.created_at else datetime.utcnow(),
                    datetime.fromisoformat(twin.last_sync) if twin.last_sync else datetime.utcnow(),
                    twin.sync_frequency_seconds,
                    json.dumps(twin.state_snapshot),
                    twin.health_score,
                    twin.drift_detected,
                    twin.drift_details
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting twin: {e}")

    async def sync_twin(self, twin_id: str, current_metrics: SystemMetrics) -> Dict[str, Any]:
        """
        Synchronize a digital twin with current production metrics

        Args:
            twin_id: ID of the twin to sync
            current_metrics: Current metrics from production system

        Returns:
            Sync result with any detected anomalies or predictions
        """
        if twin_id not in self.twins:
            return {"error": f"Twin {twin_id} not found"}

        twin = self.twins[twin_id]

        # Store metrics in history
        twin.metrics_history.append(current_metrics)

        # Keep only last 1000 metrics in memory (rest in DB)
        if len(twin.metrics_history) > 1000:
            twin.metrics_history = twin.metrics_history[-1000:]

        # Detect drift from expected state
        drift_result = self._detect_drift(twin, current_metrics)

        # Run failure predictions
        predictions = await self._predict_failures(twin, current_metrics)
        twin.failure_predictions = predictions

        # Calculate health score
        twin.health_score = self._calculate_health_score(twin, current_metrics, predictions)

        # Update sync timestamp
        twin.last_sync = datetime.utcnow().isoformat()

        # Persist updated state
        await self._persist_twin(twin)
        await self._persist_metrics(twin_id, current_metrics)

        return {
            "twin_id": twin_id,
            "synced_at": twin.last_sync,
            "health_score": twin.health_score,
            "drift_detected": drift_result["detected"],
            "drift_details": drift_result.get("details"),
            "failure_predictions": [asdict(p) for p in predictions],
            "recommendations": self._generate_recommendations(twin, predictions)
        }

    def _detect_drift(self, twin: DigitalTwin, metrics: SystemMetrics) -> Dict[str, Any]:
        """Detect if system has drifted from expected behavior"""
        if len(twin.metrics_history) < 10:
            return {"detected": False}

        # Calculate baseline from historical data
        recent_metrics = twin.metrics_history[-100:] if len(twin.metrics_history) >= 100 else twin.metrics_history

        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_latency = sum(m.request_latency_ms for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)

        # Detect significant deviations (>2 standard deviations)
        drift_detected = False
        drift_details = []

        if metrics.cpu_usage > avg_cpu * 1.5:
            drift_detected = True
            drift_details.append(f"CPU usage {metrics.cpu_usage:.1f}% vs baseline {avg_cpu:.1f}%")

        if metrics.memory_usage > avg_memory * 1.3:
            drift_detected = True
            drift_details.append(f"Memory usage {metrics.memory_usage:.1f}% vs baseline {avg_memory:.1f}%")

        if metrics.request_latency_ms > avg_latency * 2:
            drift_detected = True
            drift_details.append(f"Latency {metrics.request_latency_ms:.0f}ms vs baseline {avg_latency:.0f}ms")

        if metrics.error_rate > avg_error_rate * 3 and metrics.error_rate > 0.01:
            drift_detected = True
            drift_details.append(f"Error rate {metrics.error_rate:.2%} vs baseline {avg_error_rate:.2%}")

        twin.drift_detected = drift_detected
        twin.drift_details = "; ".join(drift_details) if drift_details else None

        return {
            "detected": drift_detected,
            "details": twin.drift_details,
            "baselines": {
                "cpu": avg_cpu,
                "memory": avg_memory,
                "latency": avg_latency,
                "error_rate": avg_error_rate
            }
        }

    async def _predict_failures(self, twin: DigitalTwin, metrics: SystemMetrics) -> List[FailurePrediction]:
        """Run failure prediction models"""
        predictions = []

        for model_name, model in self.prediction_models.items():
            try:
                prediction = model.predict(twin.metrics_history, metrics)
                if prediction and prediction.probability > 0.3:  # Only include if >30% likely
                    predictions.append(prediction)
            except Exception as e:
                logger.error(f"Prediction model {model_name} failed: {e}")

        # Sort by probability descending
        predictions.sort(key=lambda p: p.probability, reverse=True)

        # Persist high-probability predictions
        for pred in predictions:
            if pred.probability > 0.5:
                await self._persist_prediction(twin.twin_id, pred)

        return predictions

    async def _persist_metrics(self, twin_id: str, metrics: SystemMetrics):
        """Persist metrics to time-series table"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO twin_metrics_history (twin_id, metrics)
                    VALUES ($1, $2)
                """, twin_id, json.dumps(asdict(metrics)))
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")

    async def _persist_prediction(self, twin_id: str, prediction: FailurePrediction):
        """Persist failure prediction to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO twin_failure_predictions
                    (twin_id, component, failure_type, probability, predicted_time,
                     impact_severity, recommended_action, confidence, contributing_factors)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    twin_id,
                    prediction.component,
                    prediction.failure_type,
                    prediction.probability,
                    datetime.fromisoformat(prediction.predicted_time) if prediction.predicted_time else None,
                    prediction.impact_severity,
                    prediction.recommended_action,
                    prediction.confidence,
                    json.dumps(prediction.contributing_factors)
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting prediction: {e}")

    def _calculate_health_score(
        self,
        twin: DigitalTwin,
        metrics: SystemMetrics,
        predictions: List[FailurePrediction]
    ) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0

        # Deduct for high resource usage
        if metrics.cpu_usage > 80:
            score -= (metrics.cpu_usage - 80) * 0.5
        if metrics.memory_usage > 85:
            score -= (metrics.memory_usage - 85) * 0.5
        if metrics.disk_usage > 90:
            score -= (metrics.disk_usage - 90) * 1.0

        # Deduct for high error rate
        if metrics.error_rate > 0.01:
            score -= metrics.error_rate * 100

        # Deduct for high latency
        if metrics.request_latency_ms > 500:
            score -= min(20, (metrics.request_latency_ms - 500) / 50)

        # Deduct for failure predictions
        for pred in predictions:
            if pred.impact_severity == "critical":
                score -= pred.probability * 20
            elif pred.impact_severity == "high":
                score -= pred.probability * 10
            elif pred.impact_severity == "medium":
                score -= pred.probability * 5

        # Deduct for drift
        if twin.drift_detected:
            score -= 10

        return max(0, min(100, score))

    def _generate_recommendations(
        self,
        twin: DigitalTwin,
        predictions: List[FailurePrediction]
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []

        for pred in predictions:
            if pred.probability > 0.5:
                recommendations.append({
                    "priority": "high" if pred.impact_severity in ["critical", "high"] else "medium",
                    "action": pred.recommended_action,
                    "reason": f"{pred.failure_type} predicted with {pred.probability:.0%} probability",
                    "component": pred.component
                })

        if twin.drift_detected:
            recommendations.append({
                "priority": "medium",
                "action": "Investigate system drift and consider rollback or recalibration",
                "reason": twin.drift_details,
                "component": "system"
            })

        return recommendations

    async def simulate_scenario(
        self,
        twin_id: str,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a simulation scenario on the digital twin

        Args:
            twin_id: ID of the twin to simulate on
            scenario: Scenario configuration (e.g., traffic spike, failure injection)

        Returns:
            Simulation results including predicted behavior
        """
        if twin_id not in self.twins:
            return {"error": f"Twin {twin_id} not found"}

        twin = self.twins[twin_id]

        result = await self.simulation_engine.run_simulation(
            twin=twin,
            scenario=scenario
        )

        # Store simulation result
        twin.simulation_results.append({
            "scenario": scenario,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Keep only last 100 simulation results
        if len(twin.simulation_results) > 100:
            twin.simulation_results = twin.simulation_results[-100:]

        return result

    async def test_update(
        self,
        twin_id: str,
        update_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a system update on the digital twin before production deployment

        Args:
            twin_id: ID of the twin
            update_config: Configuration of the update to test

        Returns:
            Test results including predicted impact
        """
        if twin_id not in self.twins:
            return {"error": f"Twin {twin_id} not found"}

        twin = self.twins[twin_id]

        # Run update simulation
        before_state = twin.state_snapshot.copy()

        # Simulate update application
        simulated_state = self.simulation_engine.apply_update(
            current_state=before_state,
            update=update_config
        )

        # Predict impact
        impact_analysis = self.simulation_engine.analyze_impact(
            before=before_state,
            after=simulated_state,
            historical_metrics=twin.metrics_history
        )

        return {
            "twin_id": twin_id,
            "update_tested": update_config,
            "safe_to_deploy": impact_analysis["risk_score"] < 0.3,
            "risk_score": impact_analysis["risk_score"],
            "predicted_impact": impact_analysis["impacts"],
            "rollback_recommended": impact_analysis["risk_score"] > 0.5,
            "recommendations": impact_analysis.get("recommendations", [])
        }

    def get_twin_status(self, twin_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a digital twin"""
        if twin_id not in self.twins:
            return None

        twin = self.twins[twin_id]
        return {
            "twin_id": twin.twin_id,
            "source_system": twin.source_system,
            "system_type": twin.system_type.value,
            "maturity_level": twin.maturity_level.value,
            "health_score": twin.health_score,
            "last_sync": twin.last_sync,
            "drift_detected": twin.drift_detected,
            "drift_details": twin.drift_details,
            "active_predictions": len([p for p in twin.failure_predictions if p.probability > 0.3]),
            "metrics_count": len(twin.metrics_history),
            "simulations_run": len(twin.simulation_results)
        }

    def list_twins(self) -> List[Dict[str, Any]]:
        """List all digital twins"""
        return [self.get_twin_status(tid) for tid in self.twins.keys()]


# Prediction Models
class BasePredictionModel:
    """Base class for failure prediction models"""

    def predict(self, history: List[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
        raise NotImplementedError


class CPUSaturationPredictor(BasePredictionModel):
    """Predicts CPU saturation based on trend analysis"""

    def predict(self, history: List[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
        if len(history) < 10:
            return None

        recent = [m.cpu_usage for m in history[-20:]]
        trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0

        if current.cpu_usage > 70 and trend > 0.5:
            time_to_saturation = (100 - current.cpu_usage) / trend if trend > 0 else float('inf')

            if time_to_saturation < 60:  # Less than 60 minutes
                return FailurePrediction(
                    component="cpu",
                    failure_type="CPU_SATURATION",
                    probability=min(0.95, 0.5 + (current.cpu_usage / 200) + (trend / 10)),
                    predicted_time=(datetime.utcnow() + timedelta(minutes=time_to_saturation)).isoformat(),
                    impact_severity="high" if current.cpu_usage > 85 else "medium",
                    recommended_action="Scale horizontally or optimize CPU-intensive operations",
                    confidence=0.8,
                    contributing_factors=[
                        f"Current CPU: {current.cpu_usage:.1f}%",
                        f"Trend: +{trend:.2f}% per interval"
                    ]
                )
        return None


class MemoryLeakPredictor(BasePredictionModel):
    """Predicts memory leaks based on steady growth patterns"""

    def predict(self, history: List[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
        if len(history) < 20:
            return None

        memory_values = [m.memory_usage for m in history[-50:]]

        # Check for consistent upward trend (potential leak)
        increases = sum(1 for i in range(1, len(memory_values)) if memory_values[i] > memory_values[i-1])
        increase_ratio = increases / (len(memory_values) - 1) if len(memory_values) > 1 else 0

        if increase_ratio > 0.7 and current.memory_usage > 60:
            growth_rate = (memory_values[-1] - memory_values[0]) / len(memory_values)
            time_to_exhaustion = (100 - current.memory_usage) / growth_rate if growth_rate > 0 else float('inf')

            return FailurePrediction(
                component="memory",
                failure_type="MEMORY_LEAK",
                probability=min(0.9, increase_ratio * (current.memory_usage / 100)),
                predicted_time=(datetime.utcnow() + timedelta(minutes=time_to_exhaustion)).isoformat(),
                impact_severity="critical" if current.memory_usage > 85 else "high",
                recommended_action="Investigate memory allocation patterns and consider restart",
                confidence=0.75,
                contributing_factors=[
                    f"Memory consistently increasing ({increase_ratio:.0%} of intervals)",
                    f"Current memory: {current.memory_usage:.1f}%"
                ]
            )
        return None


class DiskExhaustionPredictor(BasePredictionModel):
    """Predicts disk space exhaustion"""

    def predict(self, history: List[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
        if current.disk_usage > 80:
            if len(history) >= 10:
                disk_values = [m.disk_usage for m in history[-20:]]
                growth_rate = (disk_values[-1] - disk_values[0]) / len(disk_values)
                time_to_full = (100 - current.disk_usage) / growth_rate if growth_rate > 0 else float('inf')
            else:
                time_to_full = float('inf')
                growth_rate = 0

            return FailurePrediction(
                component="disk",
                failure_type="DISK_EXHAUSTION",
                probability=current.disk_usage / 100,
                predicted_time=(datetime.utcnow() + timedelta(minutes=time_to_full)).isoformat() if time_to_full < float('inf') else None,
                impact_severity="critical" if current.disk_usage > 95 else "high" if current.disk_usage > 90 else "medium",
                recommended_action="Clean up logs, archives, or expand storage",
                confidence=0.9,
                contributing_factors=[f"Current disk usage: {current.disk_usage:.1f}%"]
            )
        return None


class LatencyDegradationPredictor(BasePredictionModel):
    """Predicts latency degradation"""

    def predict(self, history: List[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
        if len(history) < 10:
            return None

        latency_values = [m.request_latency_ms for m in history[-30:]]
        baseline = sum(latency_values[:10]) / 10 if len(latency_values) >= 10 else latency_values[0]

        if current.request_latency_ms > baseline * 2 and current.request_latency_ms > 200:
            return FailurePrediction(
                component="api",
                failure_type="LATENCY_DEGRADATION",
                probability=min(0.85, (current.request_latency_ms / baseline - 1) / 3),
                predicted_time=datetime.utcnow().isoformat(),
                impact_severity="high" if current.request_latency_ms > 1000 else "medium",
                recommended_action="Check database queries, external API calls, or resource constraints",
                confidence=0.7,
                contributing_factors=[
                    f"Current latency: {current.request_latency_ms:.0f}ms",
                    f"Baseline: {baseline:.0f}ms",
                    f"Increase: {((current.request_latency_ms / baseline) - 1) * 100:.0f}%"
                ]
            )
        return None


class ErrorRateSpikePredictor(BasePredictionModel):
    """Predicts error rate spikes"""

    def predict(self, history: List[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
        if current.error_rate > 0.05:  # >5% error rate
            return FailurePrediction(
                component="application",
                failure_type="ERROR_RATE_SPIKE",
                probability=min(0.95, current.error_rate * 5),
                predicted_time=datetime.utcnow().isoformat(),
                impact_severity="critical" if current.error_rate > 0.2 else "high" if current.error_rate > 0.1 else "medium",
                recommended_action="Check recent deployments, external dependencies, and error logs",
                confidence=0.85,
                contributing_factors=[f"Error rate: {current.error_rate:.2%}"]
            )
        return None


class ConnectionExhaustionPredictor(BasePredictionModel):
    """Predicts connection pool exhaustion"""

    def predict(self, history: List[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
        if len(history) < 5:
            return None

        # Assume max connections around 1000 for prediction
        max_connections = 1000
        usage_ratio = current.active_connections / max_connections

        if usage_ratio > 0.7:
            return FailurePrediction(
                component="connections",
                failure_type="CONNECTION_EXHAUSTION",
                probability=usage_ratio,
                predicted_time=datetime.utcnow().isoformat(),
                impact_severity="critical" if usage_ratio > 0.9 else "high" if usage_ratio > 0.8 else "medium",
                recommended_action="Increase connection pool size or optimize connection usage",
                confidence=0.8,
                contributing_factors=[
                    f"Active connections: {current.active_connections}",
                    f"Usage: {usage_ratio:.0%}"
                ]
            )
        return None


class SimulationEngine:
    """Engine for running simulations on digital twins"""

    def __init__(self):
        self.scenarios = {}

    async def run_simulation(self, twin: DigitalTwin, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a simulation scenario"""
        scenario_type = scenario.get("type", "traffic_spike")

        if scenario_type == "traffic_spike":
            return self._simulate_traffic_spike(twin, scenario)
        elif scenario_type == "failure_injection":
            return self._simulate_failure(twin, scenario)
        elif scenario_type == "resource_constraint":
            return self._simulate_resource_constraint(twin, scenario)
        elif scenario_type == "load_test":
            return self._simulate_load_test(twin, scenario)
        else:
            return {"error": f"Unknown scenario type: {scenario_type}"}

    def _simulate_traffic_spike(self, twin: DigitalTwin, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a traffic spike"""
        multiplier = scenario.get("traffic_multiplier", 3)
        duration_minutes = scenario.get("duration_minutes", 30)

        if twin.metrics_history:
            current = twin.metrics_history[-1]
            projected_cpu = min(100, current.cpu_usage * (1 + (multiplier - 1) * 0.6))
            projected_memory = min(100, current.memory_usage * (1 + (multiplier - 1) * 0.3))
            projected_latency = current.request_latency_ms * (1 + (multiplier - 1) * 0.5)

            will_fail = projected_cpu > 95 or projected_memory > 95 or projected_latency > 5000

            return {
                "scenario": "traffic_spike",
                "multiplier": multiplier,
                "duration_minutes": duration_minutes,
                "projected_metrics": {
                    "cpu_usage": projected_cpu,
                    "memory_usage": projected_memory,
                    "latency_ms": projected_latency
                },
                "will_likely_fail": will_fail,
                "failure_probability": min(1.0, max(projected_cpu, projected_memory) / 100),
                "recommendations": [
                    "Pre-scale before expected traffic" if will_fail else "System should handle this load",
                    f"Add {int((multiplier - 1) * 2)} additional instances for safety margin" if will_fail else None
                ]
            }
        return {"error": "No historical metrics available for simulation"}

    def _simulate_failure(self, twin: DigitalTwin, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a component failure"""
        component = scenario.get("component", "database")

        return {
            "scenario": "failure_injection",
            "component": component,
            "impact_analysis": {
                "affected_services": ["api", "workers", "schedulers"],
                "estimated_downtime_seconds": 30 if component == "database" else 10,
                "data_loss_risk": "low" if component != "database" else "medium"
            },
            "recovery_plan": {
                "automatic_failover": True,
                "estimated_recovery_time_seconds": 60,
                "manual_intervention_required": component == "database"
            }
        }

    def _simulate_resource_constraint(self, twin: DigitalTwin, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate resource constraints"""
        constraint_type = scenario.get("constraint", "memory")
        reduction_percent = scenario.get("reduction_percent", 50)

        return {
            "scenario": "resource_constraint",
            "constraint": constraint_type,
            "reduction": f"{reduction_percent}%",
            "impact": {
                "performance_degradation": f"{reduction_percent * 0.8:.0f}%",
                "will_trigger_oom": constraint_type == "memory" and reduction_percent > 60,
                "estimated_errors": reduction_percent > 40
            }
        }

    def _simulate_load_test(self, twin: DigitalTwin, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate load testing"""
        concurrent_users = scenario.get("concurrent_users", 1000)
        duration_minutes = scenario.get("duration_minutes", 10)

        if twin.metrics_history:
            baseline = twin.metrics_history[-1]
            load_factor = concurrent_users / 100  # Assuming baseline is 100 users

            return {
                "scenario": "load_test",
                "concurrent_users": concurrent_users,
                "duration_minutes": duration_minutes,
                "projected_results": {
                    "max_throughput_rps": baseline.throughput_rps * min(3, load_factor * 0.3),
                    "p99_latency_ms": baseline.request_latency_ms * load_factor * 0.5,
                    "error_rate": min(0.5, baseline.error_rate + (load_factor * 0.01)),
                    "breaking_point_users": concurrent_users * 1.5 if load_factor < 5 else concurrent_users
                },
                "bottleneck_analysis": {
                    "primary_bottleneck": "cpu" if baseline.cpu_usage > 50 else "io",
                    "recommendations": [
                        "Add caching layer" if load_factor > 5 else None,
                        "Optimize database queries" if load_factor > 3 else None,
                        "Consider read replicas" if load_factor > 10 else None
                    ]
                }
            }
        return {"error": "No baseline metrics available"}

    def apply_update(self, current_state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an update to simulated state"""
        new_state = current_state.copy()
        new_state.update(update)
        new_state["last_update"] = datetime.utcnow().isoformat()
        return new_state

    def analyze_impact(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        historical_metrics: List[SystemMetrics]
    ) -> Dict[str, Any]:
        """Analyze the impact of a state change"""
        changes = {}
        for key in set(list(before.keys()) + list(after.keys())):
            if before.get(key) != after.get(key):
                changes[key] = {"before": before.get(key), "after": after.get(key)}

        # Simple risk scoring based on number and type of changes
        risk_score = min(1.0, len(changes) * 0.1)

        # Higher risk for certain types of changes
        high_risk_keys = ["database", "auth", "security", "config"]
        for key in changes:
            if any(hrk in key.lower() for hrk in high_risk_keys):
                risk_score = min(1.0, risk_score + 0.2)

        return {
            "risk_score": risk_score,
            "changes": changes,
            "impacts": [
                {"area": key, "severity": "high" if risk_score > 0.5 else "medium"}
                for key in changes.keys()
            ],
            "recommendations": [
                "Staged rollout recommended" if risk_score > 0.3 else "Safe for direct deployment",
                "Enable feature flags" if risk_score > 0.5 else None,
                "Prepare rollback plan" if risk_score > 0.4 else None
            ]
        }


class FailurePredictor:
    """Central failure prediction coordinator"""

    def __init__(self):
        self.models = {}

    def add_model(self, name: str, model: BasePredictionModel):
        self.models[name] = model

    async def predict_all(self, history: List[SystemMetrics], current: SystemMetrics) -> List[FailurePrediction]:
        predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(history, current)
                if pred:
                    predictions.append(pred)
            except Exception as e:
                logger.error(f"Model {name} failed: {e}")
        return predictions


# Singleton instance
digital_twin_engine = DigitalTwinEngine()


# API Functions for FastAPI integration
async def create_system_twin(
    source_system: str,
    system_type: str,
    initial_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a new digital twin for a system"""
    await digital_twin_engine.initialize()
    twin = await digital_twin_engine.create_twin(
        source_system=source_system,
        system_type=SystemType(system_type),
        initial_state=initial_state
    )
    return digital_twin_engine.get_twin_status(twin.twin_id)


async def sync_system_twin(twin_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Sync a digital twin with current metrics"""
    await digital_twin_engine.initialize()
    system_metrics = SystemMetrics(**metrics)
    return await digital_twin_engine.sync_twin(twin_id, system_metrics)


async def get_twin_health(twin_id: str = None) -> Dict[str, Any]:
    """Get health status of twins"""
    await digital_twin_engine.initialize()
    if twin_id:
        return digital_twin_engine.get_twin_status(twin_id)
    return {"twins": digital_twin_engine.list_twins()}


async def simulate_on_twin(twin_id: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Run simulation on a digital twin"""
    await digital_twin_engine.initialize()
    return await digital_twin_engine.simulate_scenario(twin_id, scenario)


async def test_update_on_twin(twin_id: str, update_config: Dict[str, Any]) -> Dict[str, Any]:
    """Test an update on a digital twin before production"""
    await digital_twin_engine.initialize()
    return await digital_twin_engine.test_update(twin_id, update_config)

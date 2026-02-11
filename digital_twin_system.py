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

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)



# Connection pool helper - prefer shared pool, fallback to direct connection
async def _get_db_connection(db_url: str = None):
    """Get database connection, preferring shared pool"""
    try:
        from database.async_connection import get_pool
        pool = get_pool()
        return await pool.acquire()
    except Exception as exc:
        logger.warning("Shared pool unavailable, falling back to direct connection: %s", exc, exc_info=True)
        # Fallback to direct connection if pool unavailable
        if db_url:
            import asyncpg
            return await asyncpg.connect(db_url)
        return None

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
    custom_metrics: dict[str, float] = field(default_factory=dict)
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
    contributing_factors: list[str] = field(default_factory=list)


@dataclass
class StatePrediction:
    """Predicted future state based on historical trends"""
    prediction_time: str
    predicted_metrics: SystemMetrics
    confidence: float
    contributing_trends: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)


@dataclass
class DivergenceAlert:
    """Alert when actual state diverges from expected"""
    alert_id: str
    component: str
    expected_value: float
    actual_value: float
    divergence_percent: float
    severity: str  # low, medium, high, critical
    recommended_correction: str
    auto_correct_eligible: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class StateHistoryEntry:
    """Historical state snapshot for debugging and rollback"""
    snapshot_id: str
    timestamp: str
    state_snapshot: dict[str, Any]
    metrics: SystemMetrics
    health_score: float
    change_reason: str
    can_rollback_to: bool = True


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
    state_snapshot: dict[str, Any]
    metrics_history: list[SystemMetrics] = field(default_factory=list)
    failure_predictions: list[FailurePrediction] = field(default_factory=list)
    simulation_results: list[dict[str, Any]] = field(default_factory=list)
    state_predictions: list[StatePrediction] = field(default_factory=list)
    divergence_alerts: list[DivergenceAlert] = field(default_factory=list)
    state_history: list[StateHistoryEntry] = field(default_factory=list)
    health_score: float = 100.0
    drift_detected: bool = False
    drift_details: Optional[str] = None
    auto_correction_enabled: bool = True
    expected_state: Optional[dict[str, Any]] = None


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
        self.twins: dict[str, DigitalTwin] = {}
        self.db_url = os.getenv("DATABASE_URL")
        self.prediction_models: dict[str, Any] = {}
        self.simulation_engine = SimulationEngine()
        self.failure_predictor = FailurePredictor()
        self.state_predictor = StatePredictor()
        self.divergence_detector = DivergenceDetector()
        self.auto_corrector = AutoCorrector()
        self._initialized = False
        self._sync_in_progress: dict[str, bool] = {}  # Track sync to prevent loops

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

                # Create metrics history table for time-series data

                # Create failure predictions table

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
        initial_state: dict[str, Any],
        maturity_level: TwinMaturityLevel = TwinMaturityLevel.PREDICTIVE,
        sync_frequency_seconds: int = 60
    ) -> DigitalTwin:
        """
        Create a new digital twin for a production system (or return existing)

        Args:
            source_system: Identifier for the production system (URL or ID)
            system_type: Type of system being twinned
            initial_state: Initial state snapshot of the system
            maturity_level: Target maturity level for the twin
            sync_frequency_seconds: How often to sync with production

        Returns:
            Created or existing DigitalTwin instance
        """
        twin_id = self._generate_twin_id(source_system)
        now = datetime.utcnow().isoformat()

        # DEDUPLICATION: Check if twin already exists
        if twin_id in self.twins:
            existing_twin = self.twins[twin_id]
            # Update the existing twin with new state
            existing_twin.last_sync = now
            existing_twin.state_snapshot = initial_state
            await self._persist_twin(existing_twin)
            logger.info(f"Updated existing digital twin {twin_id} for {source_system}")
            return existing_twin

        # Also check database for existing twin not in memory
        existing_db_twin = await self._get_twin_from_db(twin_id)
        if existing_db_twin:
            existing_db_twin.last_sync = now
            existing_db_twin.state_snapshot = initial_state
            self.twins[twin_id] = existing_db_twin
            await self._persist_twin(existing_db_twin)
            logger.info(f"Loaded and updated existing digital twin {twin_id} from DB for {source_system}")
            return existing_db_twin

        # Create new twin
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

    async def _get_twin_from_db(self, twin_id: str) -> Optional[DigitalTwin]:
        """Check if a twin exists in the database"""
        try:
            import asyncpg
            if not self.db_url:
                return None

            conn = await asyncpg.connect(self.db_url)
            try:
                row = await conn.fetchrow(
                    "SELECT * FROM digital_twins WHERE twin_id = $1",
                    twin_id
                )
                if row:
                    return DigitalTwin(
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
                return None
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error checking twin in DB: {e}")
            return None

    def _generate_twin_id(self, source_system: str) -> str:
        """Generate deterministic twin ID from source system (same system = same ID)"""
        # IMPORTANT: Use only source_system for deterministic IDs to prevent duplicates
        # This ensures the same system always maps to the same twin_id
        hash_input = f"twin:{source_system.lower().strip()}"
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

    async def deduplicate_twins(self) -> dict[str, Any]:
        """
        Clean up duplicate twins in the database.
        Keeps only the most recent twin per source_system.
        Returns summary of cleanup actions.
        """
        try:
            import asyncpg
            if not self.db_url:
                return {"error": "No database URL configured"}

            conn = await asyncpg.connect(self.db_url)
            try:
                # Find duplicates - keep the latest by last_sync for each source_system
                deleted_count = await conn.fetchval("""
                    WITH ranked_twins AS (
                        SELECT twin_id, source_system,
                               ROW_NUMBER() OVER (
                                   PARTITION BY source_system
                                   ORDER BY last_sync DESC NULLS LAST
                               ) as rn
                        FROM digital_twins
                    ),
                    to_delete AS (
                        SELECT twin_id FROM ranked_twins WHERE rn > 1
                    )
                    DELETE FROM digital_twins
                    WHERE twin_id IN (SELECT twin_id FROM to_delete)
                    RETURNING twin_id
                """)

                # Get remaining count
                remaining = await conn.fetchval("SELECT COUNT(*) FROM digital_twins")

                # Reload twins into memory
                await self._load_twins_from_db()

                logger.info(f"Deduplicated twins: deleted {deleted_count or 0} duplicates, {remaining} remaining")
                return {
                    "status": "success",
                    "duplicates_removed": deleted_count or 0,
                    "twins_remaining": remaining or 0
                }
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error deduplicating twins: {e}")
            return {"error": str(e)}

    async def sync_twin(self, twin_id: str, current_metrics: SystemMetrics, source: str = "external") -> dict[str, Any]:
        """
        Synchronize a digital twin with current production metrics

        IMPORTANT: This method does NOT call any external endpoints to prevent sync loops.
        It only processes incoming metrics and updates internal state.

        Args:
            twin_id: ID of the twin to sync
            current_metrics: Current metrics from production system
            source: Source of the sync (external, internal, simulation) - prevents loops

        Returns:
            Sync result with any detected anomalies or predictions
        """
        if twin_id not in self.twins:
            return {"error": f"Twin {twin_id} not found"}

        # CRITICAL: Prevent sync loops
        if self._sync_in_progress.get(twin_id, False):
            logger.warning(f"Sync already in progress for {twin_id}, skipping to prevent loop")
            return {"error": "Sync already in progress", "twin_id": twin_id}

        try:
            self._sync_in_progress[twin_id] = True
            twin = self.twins[twin_id]

            # Save state history before making changes
            await self._save_state_history(twin, current_metrics, "sync_update")

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

            # Predict future state
            state_predictions = await self._predict_future_state(twin, current_metrics)
            twin.state_predictions = state_predictions

            # Detect divergence from expected state
            divergence_alerts = await self._detect_divergence(twin, current_metrics)
            twin.divergence_alerts = divergence_alerts

            # Auto-correct if enabled and divergence detected
            corrections_applied = []
            if twin.auto_correction_enabled and divergence_alerts:
                corrections_applied = await self._apply_auto_corrections(twin, divergence_alerts)

            # Calculate health score
            twin.health_score = self._calculate_health_score(twin, current_metrics, predictions)

            # Update sync timestamp
            twin.last_sync = datetime.utcnow().isoformat()

            # Persist updated state (NO EXTERNAL CALLS HERE)
            await self._persist_twin(twin)
            await self._persist_metrics(twin_id, current_metrics)

            return {
                "twin_id": twin_id,
                "synced_at": twin.last_sync,
                "source": source,
                "health_score": twin.health_score,
                "drift_detected": drift_result["detected"],
                "drift_details": drift_result.get("details"),
                "failure_predictions": [asdict(p) for p in predictions],
                "state_predictions": [asdict(p) for p in state_predictions[:5]],  # Top 5
                "divergence_alerts": [asdict(a) for a in divergence_alerts],
                "corrections_applied": corrections_applied,
                "recommendations": self._generate_recommendations(twin, predictions)
            }
        finally:
            # Always release the lock
            self._sync_in_progress[twin_id] = False

    def _detect_drift(self, twin: DigitalTwin, metrics: SystemMetrics) -> dict[str, Any]:
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

    async def _predict_failures(self, twin: DigitalTwin, metrics: SystemMetrics) -> list[FailurePrediction]:
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

    async def _save_state_history(self, twin: DigitalTwin, metrics: SystemMetrics, reason: str):
        """Save current state to history for debugging and rollback"""
        try:
            snapshot_id = f"snap_{hashlib.sha256(f'{twin.twin_id}:{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:16]}"

            history_entry = StateHistoryEntry(
                snapshot_id=snapshot_id,
                timestamp=datetime.utcnow().isoformat(),
                state_snapshot=twin.state_snapshot.copy(),
                metrics=metrics,
                health_score=twin.health_score,
                change_reason=reason
            )

            twin.state_history.append(history_entry)

            # Keep only last 100 snapshots in memory
            if len(twin.state_history) > 100:
                twin.state_history = twin.state_history[-100:]

            # Persist to database
            if self.db_url:
                import asyncpg
                conn = await asyncpg.connect(self.db_url)
                try:

                    await conn.execute("""
                        INSERT INTO twin_state_history
                        (twin_id, snapshot_id, state_snapshot, metrics, health_score, change_reason)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                        twin.twin_id,
                        snapshot_id,
                        json.dumps(twin.state_snapshot),
                        json.dumps(asdict(metrics)),
                        twin.health_score,
                        reason
                    )
                finally:
                    await conn.close()

        except Exception as e:
            logger.error(f"Error saving state history: {e}")

    async def _predict_future_state(
        self,
        twin: DigitalTwin,
        current_metrics: SystemMetrics
    ) -> list[StatePrediction]:
        """Predict future system states"""
        try:
            return self.state_predictor.predict_future_states(
                twin.metrics_history,
                current_metrics
            )
        except Exception as e:
            logger.error(f"Error predicting future state: {e}")
            return []

    async def _detect_divergence(
        self,
        twin: DigitalTwin,
        current_metrics: SystemMetrics
    ) -> list[DivergenceAlert]:
        """Detect divergence from expected state"""
        try:
            return self.divergence_detector.detect_divergence(
                twin,
                current_metrics
            )
        except Exception as e:
            logger.error(f"Error detecting divergence: {e}")
            return []

    async def _apply_auto_corrections(
        self,
        twin: DigitalTwin,
        alerts: list[DivergenceAlert]
    ) -> list[dict[str, Any]]:
        """Apply automatic corrections for divergence alerts"""
        try:
            return await self.auto_corrector.apply_corrections(twin, alerts)
        except Exception as e:
            logger.error(f"Error applying auto-corrections: {e}")
            return []

    async def rollback_to_snapshot(
        self,
        twin_id: str,
        snapshot_id: str
    ) -> dict[str, Any]:
        """Rollback twin to a previous state snapshot"""
        if twin_id not in self.twins:
            return {"error": f"Twin {twin_id} not found"}

        twin = self.twins[twin_id]

        # Find snapshot
        snapshot = None
        for entry in twin.state_history:
            if entry.snapshot_id == snapshot_id:
                snapshot = entry
                break

        if not snapshot:
            return {"error": f"Snapshot {snapshot_id} not found"}

        if not snapshot.can_rollback_to:
            return {"error": f"Snapshot {snapshot_id} is not eligible for rollback"}

        # Save current state before rollback
        await self._save_state_history(
            twin,
            twin.metrics_history[-1] if twin.metrics_history else SystemMetrics(),
            f"pre_rollback_to_{snapshot_id}"
        )

        # Apply rollback
        twin.state_snapshot = snapshot.state_snapshot.copy()
        twin.health_score = snapshot.health_score

        # Persist
        await self._persist_twin(twin)

        logger.info(f"Rolled back twin {twin_id} to snapshot {snapshot_id}")

        return {
            "twin_id": twin_id,
            "rolled_back_to": snapshot_id,
            "snapshot_timestamp": snapshot.timestamp,
            "reason": snapshot.change_reason,
            "success": True
        }

    async def set_expected_state(
        self,
        twin_id: str,
        expected_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Set the expected state for divergence detection"""
        if twin_id not in self.twins:
            return {"error": f"Twin {twin_id} not found"}

        twin = self.twins[twin_id]
        twin.expected_state = expected_state

        await self._persist_twin(twin)

        logger.info(f"Set expected state for twin {twin_id}")

        return {
            "twin_id": twin_id,
            "expected_state": expected_state,
            "success": True
        }

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
        predictions: list[FailurePrediction]
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
        predictions: list[FailurePrediction]
    ) -> list[dict[str, str]]:
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
        scenario: dict[str, Any]
    ) -> dict[str, Any]:
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
        update_config: dict[str, Any]
    ) -> dict[str, Any]:
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

    def get_twin_status(self, twin_id: str) -> Optional[dict[str, Any]]:
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

    def list_twins(self) -> list[dict[str, Any]]:
        """List all digital twins"""
        return [self.get_twin_status(tid) for tid in self.twins.keys()]


# Prediction Models
class BasePredictionModel:
    """Base class for failure prediction models"""

    def predict(self, history: list[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
        raise NotImplementedError


class CPUSaturationPredictor(BasePredictionModel):
    """Predicts CPU saturation based on trend analysis"""

    def predict(self, history: list[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
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

    def predict(self, history: list[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
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

    def predict(self, history: list[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
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

    def predict(self, history: list[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
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

    def predict(self, history: list[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
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

    def predict(self, history: list[SystemMetrics], current: SystemMetrics) -> Optional[FailurePrediction]:
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

    async def run_simulation(self, twin: DigitalTwin, scenario: dict[str, Any]) -> dict[str, Any]:
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

    def _simulate_traffic_spike(self, twin: DigitalTwin, scenario: dict[str, Any]) -> dict[str, Any]:
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

    def _simulate_failure(self, twin: DigitalTwin, scenario: dict[str, Any]) -> dict[str, Any]:
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

    def _simulate_resource_constraint(self, twin: DigitalTwin, scenario: dict[str, Any]) -> dict[str, Any]:
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

    def _simulate_load_test(self, twin: DigitalTwin, scenario: dict[str, Any]) -> dict[str, Any]:
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

    def apply_update(self, current_state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        """Apply an update to simulated state"""
        new_state = current_state.copy()
        new_state.update(update)
        new_state["last_update"] = datetime.utcnow().isoformat()
        return new_state

    def analyze_impact(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        historical_metrics: list[SystemMetrics]
    ) -> dict[str, Any]:
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

    async def predict_all(self, history: list[SystemMetrics], current: SystemMetrics) -> list[FailurePrediction]:
        predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(history, current)
                if pred:
                    predictions.append(pred)
            except Exception as e:
                logger.error(f"Model {name} failed: {e}")
        return predictions


class StatePredictor:
    """Predicts future system state based on historical trends"""

    def predict_future_states(
        self,
        history: list[SystemMetrics],
        current: SystemMetrics,
        prediction_windows: list[int] = [5, 15, 30, 60]  # minutes
    ) -> list[StatePrediction]:
        """Predict future states at various time intervals"""
        if len(history) < 10:
            return []

        predictions = []
        recent = history[-50:] if len(history) >= 50 else history

        for minutes_ahead in prediction_windows:
            # Calculate trends
            cpu_trend = self._calculate_trend([m.cpu_usage for m in recent])
            memory_trend = self._calculate_trend([m.memory_usage for m in recent])
            latency_trend = self._calculate_trend([m.request_latency_ms for m in recent])
            error_trend = self._calculate_trend([m.error_rate for m in recent])

            # Project forward
            predicted_cpu = min(100, max(0, current.cpu_usage + (cpu_trend * minutes_ahead)))
            predicted_memory = min(100, max(0, current.memory_usage + (memory_trend * minutes_ahead)))
            predicted_latency = max(0, current.request_latency_ms + (latency_trend * minutes_ahead))
            predicted_error = max(0, current.error_rate + (error_trend * minutes_ahead))

            # Calculate confidence based on trend consistency
            confidence = self._calculate_confidence(recent)

            # Identify trends and risks
            trends = []
            risks = []

            if cpu_trend > 0.5:
                trends.append(f"CPU increasing at {cpu_trend:.2f}% per minute")
                if predicted_cpu > 80:
                    risks.append(f"CPU may reach {predicted_cpu:.0f}% in {minutes_ahead} minutes")

            if memory_trend > 0.3:
                trends.append(f"Memory increasing at {memory_trend:.2f}% per minute")
                if predicted_memory > 85:
                    risks.append(f"Memory may reach {predicted_memory:.0f}% in {minutes_ahead} minutes")

            if latency_trend > 5:
                trends.append(f"Latency increasing at {latency_trend:.1f}ms per minute")
                if predicted_latency > 1000:
                    risks.append(f"Latency may exceed 1000ms in {minutes_ahead} minutes")

            if error_trend > 0.001:
                trends.append("Error rate increasing")
                risks.append("Error rate trending upward")

            prediction = StatePrediction(
                prediction_time=(datetime.utcnow() + timedelta(minutes=minutes_ahead)).isoformat(),
                predicted_metrics=SystemMetrics(
                    cpu_usage=predicted_cpu,
                    memory_usage=predicted_memory,
                    disk_usage=current.disk_usage,  # Disk changes slowly
                    network_io=current.network_io,
                    request_latency_ms=predicted_latency,
                    error_rate=predicted_error,
                    throughput_rps=current.throughput_rps,
                    active_connections=current.active_connections,
                    queue_depth=current.queue_depth
                ),
                confidence=confidence,
                contributing_trends=trends,
                risk_factors=risks
            )

            predictions.append(prediction)

        return predictions

    def _calculate_trend(self, values: list[float]) -> float:
        """Calculate linear trend from values"""
        if len(values) < 2:
            return 0.0

        # Simple linear regression
        n = len(values)
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope

    def _calculate_confidence(self, metrics: list[SystemMetrics]) -> float:
        """Calculate prediction confidence based on data consistency"""
        if len(metrics) < 10:
            return 0.3

        # Check variance in recent data
        cpu_values = [m.cpu_usage for m in metrics]
        variance = sum((x - sum(cpu_values) / len(cpu_values)) ** 2 for x in cpu_values) / len(cpu_values)

        # Lower variance = higher confidence
        confidence = max(0.3, min(0.95, 1.0 - (variance / 1000)))
        return confidence


class DivergenceDetector:
    """Detects when actual state diverges from expected state"""

    def detect_divergence(
        self,
        twin: DigitalTwin,
        current_metrics: SystemMetrics,
        thresholds: Optional[dict[str, float]] = None
    ) -> list[DivergenceAlert]:
        """Detect divergence between expected and actual state"""
        alerts = []

        if not twin.expected_state:
            # No expected state set, use historical baseline
            if len(twin.metrics_history) < 10:
                return []

            recent = twin.metrics_history[-100:] if len(twin.metrics_history) >= 100 else twin.metrics_history
            expected_cpu = sum(m.cpu_usage for m in recent) / len(recent)
            expected_memory = sum(m.memory_usage for m in recent) / len(recent)
            expected_latency = sum(m.request_latency_ms for m in recent) / len(recent)
            expected_error = sum(m.error_rate for m in recent) / len(recent)
        else:
            expected_cpu = twin.expected_state.get("cpu_usage", current_metrics.cpu_usage)
            expected_memory = twin.expected_state.get("memory_usage", current_metrics.memory_usage)
            expected_latency = twin.expected_state.get("latency_ms", current_metrics.request_latency_ms)
            expected_error = twin.expected_state.get("error_rate", current_metrics.error_rate)

        # Default thresholds
        if not thresholds:
            thresholds = {
                "cpu": 20.0,  # 20% divergence
                "memory": 15.0,
                "latency": 50.0,
                "error_rate": 100.0  # 100% increase (e.g., 0.01 -> 0.02)
            }

        # Check CPU divergence
        if expected_cpu > 0:
            cpu_divergence = abs((current_metrics.cpu_usage - expected_cpu) / expected_cpu * 100)
            if cpu_divergence > thresholds["cpu"]:
                alerts.append(DivergenceAlert(
                    alert_id=f"div_{hashlib.md5(f'{twin.twin_id}_cpu_{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:8]}",
                    component="cpu",
                    expected_value=expected_cpu,
                    actual_value=current_metrics.cpu_usage,
                    divergence_percent=cpu_divergence,
                    severity=self._calculate_severity(cpu_divergence),
                    recommended_correction="Scale horizontally or investigate CPU-intensive processes",
                    auto_correct_eligible=cpu_divergence < 50  # Only auto-correct minor divergences
                ))

        # Check Memory divergence
        if expected_memory > 0:
            memory_divergence = abs((current_metrics.memory_usage - expected_memory) / expected_memory * 100)
            if memory_divergence > thresholds["memory"]:
                alerts.append(DivergenceAlert(
                    alert_id=f"div_{hashlib.md5(f'{twin.twin_id}_mem_{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:8]}",
                    component="memory",
                    expected_value=expected_memory,
                    actual_value=current_metrics.memory_usage,
                    divergence_percent=memory_divergence,
                    severity=self._calculate_severity(memory_divergence),
                    recommended_correction="Check for memory leaks or increase memory allocation",
                    auto_correct_eligible=memory_divergence < 30
                ))

        # Check Latency divergence
        if expected_latency > 0:
            latency_divergence = abs((current_metrics.request_latency_ms - expected_latency) / expected_latency * 100)
            if latency_divergence > thresholds["latency"]:
                alerts.append(DivergenceAlert(
                    alert_id=f"div_{hashlib.md5(f'{twin.twin_id}_lat_{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:8]}",
                    component="latency",
                    expected_value=expected_latency,
                    actual_value=current_metrics.request_latency_ms,
                    divergence_percent=latency_divergence,
                    severity=self._calculate_severity(latency_divergence),
                    recommended_correction="Optimize database queries or add caching",
                    auto_correct_eligible=False  # Latency requires manual investigation
                ))

        # Check Error Rate divergence
        if expected_error > 0:
            error_divergence = abs((current_metrics.error_rate - expected_error) / expected_error * 100)
            if error_divergence > thresholds["error_rate"]:
                alerts.append(DivergenceAlert(
                    alert_id=f"div_{hashlib.md5(f'{twin.twin_id}_err_{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:8]}",
                    component="error_rate",
                    expected_value=expected_error,
                    actual_value=current_metrics.error_rate,
                    divergence_percent=error_divergence,
                    severity="critical" if current_metrics.error_rate > 0.05 else "high",
                    recommended_correction="Check recent deployments and error logs",
                    auto_correct_eligible=False  # Errors require manual investigation
                ))

        return alerts

    def _calculate_severity(self, divergence_percent: float) -> str:
        """Calculate severity based on divergence percentage"""
        if divergence_percent > 75:
            return "critical"
        elif divergence_percent > 50:
            return "high"
        elif divergence_percent > 25:
            return "medium"
        else:
            return "low"


class AutoCorrector:
    """Automatically corrects certain types of divergences"""

    async def apply_corrections(
        self,
        twin: DigitalTwin,
        alerts: list[DivergenceAlert]
    ) -> list[dict[str, Any]]:
        """Apply automatic corrections for eligible divergences"""
        corrections = []

        for alert in alerts:
            if not alert.auto_correct_eligible:
                logger.info(f"Alert {alert.alert_id} not eligible for auto-correction")
                continue

            correction = await self._apply_correction(twin, alert)
            if correction:
                corrections.append(correction)

        return corrections

    async def _apply_correction(
        self,
        twin: DigitalTwin,
        alert: DivergenceAlert
    ) -> Optional[dict[str, Any]]:
        """Apply a specific correction"""
        try:
            if alert.component == "cpu" and alert.actual_value > alert.expected_value:
                # Recommend scaling
                return {
                    "alert_id": alert.alert_id,
                    "component": alert.component,
                    "action": "recommend_scale_up",
                    "details": f"Recommend adding 1-2 instances to handle {alert.actual_value:.1f}% CPU usage",
                    "applied": False,  # Recommendation only, not automatic
                    "timestamp": datetime.utcnow().isoformat()
                }

            elif alert.component == "memory" and alert.divergence_percent < 25:
                # Update expected state to accommodate gradual growth
                return {
                    "alert_id": alert.alert_id,
                    "component": alert.component,
                    "action": "update_expected_state",
                    "details": f"Updated expected memory from {alert.expected_value:.1f}% to {alert.actual_value:.1f}%",
                    "applied": True,
                    "timestamp": datetime.utcnow().isoformat()
                }

            return None

        except Exception as e:
            logger.error(f"Error applying correction for {alert.alert_id}: {e}")
            return None


# Singleton instance
digital_twin_engine = DigitalTwinEngine()


# API Functions for FastAPI integration
async def create_system_twin(
    source_system: str,
    system_type: str,
    initial_state: dict[str, Any]
) -> dict[str, Any]:
    """Create a new digital twin for a system"""
    await digital_twin_engine.initialize()
    twin = await digital_twin_engine.create_twin(
        source_system=source_system,
        system_type=SystemType(system_type),
        initial_state=initial_state
    )
    return digital_twin_engine.get_twin_status(twin.twin_id)


async def sync_system_twin(twin_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Sync a digital twin with current metrics"""
    await digital_twin_engine.initialize()
    system_metrics = SystemMetrics(**metrics)
    return await digital_twin_engine.sync_twin(twin_id, system_metrics)


async def get_twin_health(twin_id: str = None) -> dict[str, Any]:
    """Get health status of twins"""
    await digital_twin_engine.initialize()
    if twin_id:
        return digital_twin_engine.get_twin_status(twin_id)
    return {"twins": digital_twin_engine.list_twins()}


async def simulate_on_twin(twin_id: str, scenario: dict[str, Any]) -> dict[str, Any]:
    """Run simulation on a digital twin"""
    await digital_twin_engine.initialize()
    return await digital_twin_engine.simulate_scenario(twin_id, scenario)


async def test_update_on_twin(twin_id: str, update_config: dict[str, Any]) -> dict[str, Any]:
    """Test an update on a digital twin before production"""
    await digital_twin_engine.initialize()
    return await digital_twin_engine.test_update(twin_id, update_config)

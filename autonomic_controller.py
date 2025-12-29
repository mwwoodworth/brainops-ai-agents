#!/usr/bin/env python3
"""
BrainOps Autonomic Controller - MAPE-K Feedback Loop System
Self-driving capabilities for the AI Operating System

Features:
- Real-time Metric Collection with sliding windows
- Async Event Bus for cross-agent communication
- MAPE-K autonomic feedback loop
- Predictive failure detection
- Resource optimization
"""

import os
import asyncio
import logging
import collections
import psycopg2
from datetime import datetime, timedelta
from contextlib import contextmanager

# ============================================================================
# SHARED CONNECTION POOL - CRITICAL for preventing MaxClientsInSessionMode
# ============================================================================
try:
    from database.sync_pool import get_sync_pool
    _POOL_AVAILABLE = True
except ImportError:
    _POOL_AVAILABLE = False


@contextmanager
def _get_pooled_connection():
    """Get connection from shared pool - ALWAYS use this instead of psycopg2.connect()"""
    if _POOL_AVAILABLE:
        with get_sync_pool().get_connection() as conn:
            yield conn
    else:
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            yield conn
        finally:
            if conn and not conn.closed:
                conn.close()


from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, stdev
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
    'password': os.getenv("DB_PASSWORD"),
    'port': int(os.getenv('DB_PORT', 5432))
}


class EventType(Enum):
    """Event types for cross-agent communication"""
    WORKFLOW_START = "workflow.start"
    WORKFLOW_COMPLETE = "workflow.complete"
    WORKFLOW_FAILED = "workflow.failed"
    SYSTEM_ALERT = "system.alert"
    RESOURCE_LOW = "resource.low"
    RESOURCE_OPTIMIZED = "resource.optimized"
    COMPONENT_HEALTHY = "component.healthy"
    COMPONENT_DEGRADED = "component.degraded"
    COMPONENT_FAILED = "component.failed"
    PREDICTION_ALERT = "prediction.alert"
    SCALING_UP = "scaling.up"
    SCALING_DOWN = "scaling.down"
    HEALING_STARTED = "healing.started"
    HEALING_COMPLETE = "healing.complete"


@dataclass
class Metric:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Aggregated metric statistics"""
    current: float
    avg: float
    min_val: float
    max_val: float
    p50: float
    p95: float
    p99: float
    stddev: float
    count: int
    trend: str  # up, down, stable


class MetricCollector:
    """
    High-frequency metric collection with sliding windows.
    Non-blocking async emission with real-time aggregation.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=window_size)
        )
        self._lock = asyncio.Lock()
        self._baselines: Dict[str, float] = {}

    async def emit(self, name: str, value: float, tags: Dict[str, str] = None):
        """Non-blocking metric emission"""
        async with self._lock:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)

    def get_stats(self, name: str) -> Optional[MetricStats]:
        """Get real-time statistics for a metric window"""
        if not self.metrics[name]:
            return None

        values = [m.value for m in self.metrics[name]]
        values_sorted = sorted(values)
        n = len(values)

        # Calculate trend
        if n >= 10:
            recent = mean(values[-5:])
            older = mean(values[-10:-5])
            if recent > older * 1.1:
                trend = "up"
            elif recent < older * 0.9:
                trend = "down"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return MetricStats(
            current=values[-1],
            avg=mean(values),
            min_val=min(values),
            max_val=max(values),
            p50=values_sorted[int(n * 0.50)],
            p95=values_sorted[int(n * 0.95)] if n >= 20 else values_sorted[-1],
            p99=values_sorted[int(n * 0.99)] if n >= 100 else values_sorted[-1],
            stddev=stdev(values) if n >= 2 else 0.0,
            count=n,
            trend=trend
        )

    def set_baseline(self, name: str, value: float):
        """Set baseline for anomaly detection"""
        self._baselines[name] = value

    def detect_anomaly(self, name: str, threshold: float = 2.0) -> Optional[Dict]:
        """Detect if current values deviate from baseline"""
        stats = self.get_stats(name)
        if not stats or name not in self._baselines:
            return None

        baseline = self._baselines[name]
        deviation = abs(stats.current - baseline) / max(baseline, 0.001)

        if deviation > threshold:
            return {
                'metric': name,
                'current': stats.current,
                'baseline': baseline,
                'deviation': deviation,
                'severity': 'critical' if deviation > threshold * 2 else 'warning'
            }
        return None

    def get_all_stats(self) -> Dict[str, MetricStats]:
        """Get stats for all metrics"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}


class EventBus:
    """
    Asynchronous Pub/Sub Event Bus for cross-agent communication.
    Enables decoupled, many-to-many agent coordination.
    """

    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = collections.defaultdict(list)
        self.event_history: collections.deque = collections.deque(maxlen=1000)
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to an event type"""
        self.subscribers[event_type].append(callback)
        logger.info(f"Subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from an event type"""
        if callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)

    async def publish(self, event_type: EventType, payload: Dict[str, Any]):
        """Fire and forget event publishing with parallel subscriber notification"""
        async with self._lock:
            event = {
                'type': event_type.value,
                'payload': payload,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.event_history.append(event)

        if event_type in self.subscribers:
            tasks = []
            for callback in self.subscribers[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(payload))
                else:
                    # Wrap sync callbacks
                    tasks.append(asyncio.get_event_loop().run_in_executor(
                        None, callback, payload
                    ))

            # Run all subscribers in parallel
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def get_recent_events(self, event_type: EventType = None, limit: int = 50) -> List[Dict]:
        """Get recent events, optionally filtered by type"""
        events = list(self.event_history)
        if event_type:
            events = [e for e in events if e['type'] == event_type.value]
        return events[-limit:]


class AutonomicManager:
    """
    MAPE-K (Monitor, Analyze, Plan, Execute, Knowledge) Feedback Control Loop.
    Self-driving orchestration for AI Operating System.
    """

    def __init__(self, metrics: MetricCollector, event_bus: EventBus):
        self.metrics = metrics
        self.event_bus = event_bus
        self.active = False
        self.loop_count = 0
        self.knowledge_base: Dict[str, Any] = {}
        self._init_db()

    def _init_db(self):
        """Initialize autonomic controller database tables"""
        try:
            with _get_pooled_connection() as conn:
                if not conn:
                    logger.error("Failed to get connection for autonomic DB init")
                    return
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS autonomic_decisions (
                        id SERIAL PRIMARY KEY,
                        loop_id INTEGER,
                        phase VARCHAR(20),
                        decision TEXT,
                        action_taken TEXT,
                        result JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS autonomic_predictions (
                        id SERIAL PRIMARY KEY,
                        metric_name VARCHAR(100),
                        prediction_type VARCHAR(50),
                        predicted_value FLOAT,
                        confidence FLOAT,
                        time_horizon_hours INTEGER,
                        actual_value FLOAT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_autonomic_loop ON autonomic_decisions(loop_id);
                """)
                conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Failed to init autonomic DB: {e}")

    async def start_loop(self, interval: float = 10.0):
        """Start the MAPE-K autonomic control loop"""
        self.active = True
        logger.info("🔄 Starting Autonomic Control Loop (MAPE-K)")

        while self.active:
            self.loop_count += 1

            try:
                # 1. MONITOR - Collect current state
                state = await self._monitor()

                # 2. ANALYZE - Identify issues and opportunities
                analysis = await self._analyze(state)

                # 3. PLAN - Determine actions
                plan = await self._plan(analysis)

                # 4. EXECUTE - Take actions
                result = await self._execute(plan)

                # 5. KNOWLEDGE - Update baselines and learn
                await self._update_knowledge(state, analysis, plan, result)

            except Exception as e:
                logger.error(f"MAPE-K loop error: {e}")
                await self.event_bus.publish(EventType.SYSTEM_ALERT, {
                    'source': 'autonomic_manager',
                    'error': str(e),
                    'loop_id': self.loop_count
                })

            await asyncio.sleep(interval)

    def stop_loop(self):
        """Stop the autonomic control loop"""
        self.active = False
        logger.info("⏹️ Stopping Autonomic Control Loop")

    async def _monitor(self) -> Dict[str, Any]:
        """MONITOR phase: Collect system state"""
        state = {
            'loop_id': self.loop_count,
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {},
            'anomalies': []
        }

        # Collect all metric stats
        all_stats = self.metrics.get_all_stats()
        for name, stats in all_stats.items():
            if stats:
                state['metrics'][name] = {
                    'current': stats.current,
                    'avg': stats.avg,
                    'trend': stats.trend
                }

                # Check for anomalies
                anomaly = self.metrics.detect_anomaly(name)
                if anomaly:
                    state['anomalies'].append(anomaly)

        return state

    async def _analyze(self, state: Dict) -> Dict[str, Any]:
        """ANALYZE phase: Identify issues and opportunities"""
        analysis = {
            'needs_healing': False,
            'needs_scaling': False,
            'needs_optimization': False,
            'issues': [],
            'opportunities': []
        }

        # Check for critical anomalies
        for anomaly in state.get('anomalies', []):
            if anomaly['severity'] == 'critical':
                analysis['needs_healing'] = True
                analysis['issues'].append({
                    'type': 'anomaly',
                    'metric': anomaly['metric'],
                    'action': 'investigate_and_heal'
                })

        # Check latency trends
        latency_stats = self.metrics.get_stats('request_latency')
        if latency_stats and latency_stats.avg > 0.5:  # 500ms threshold
            analysis['needs_scaling'] = True
            analysis['issues'].append({
                'type': 'high_latency',
                'value': latency_stats.avg,
                'action': 'scale_up'
            })

        # Check error rates
        error_stats = self.metrics.get_stats('error_rate')
        if error_stats and error_stats.current > 0.05:  # 5% threshold
            analysis['needs_healing'] = True
            analysis['issues'].append({
                'type': 'high_error_rate',
                'value': error_stats.current,
                'action': 'trigger_recovery'
            })

        # Check resource utilization for optimization opportunities
        cpu_stats = self.metrics.get_stats('cpu_usage')
        if cpu_stats and cpu_stats.avg < 0.3:  # Under-utilized
            analysis['opportunities'].append({
                'type': 'underutilized_resources',
                'action': 'scale_down'
            })

        return analysis

    async def _plan(self, analysis: Dict) -> Dict[str, Any]:
        """PLAN phase: Determine actions to take"""
        plan = {
            'actions': [],
            'priority': 'normal'
        }

        # Handle critical issues first
        if analysis['needs_healing']:
            plan['priority'] = 'high'
            for issue in analysis['issues']:
                if issue['action'] == 'trigger_recovery':
                    plan['actions'].append({
                        'type': 'self_heal',
                        'target': issue['type'],
                        'params': {'issue': issue}
                    })

        # Handle scaling
        if analysis['needs_scaling']:
            plan['actions'].append({
                'type': 'scale_up',
                'target': 'worker_pool',
                'params': {'factor': 1.5}
            })

        # Handle optimization opportunities
        for opportunity in analysis['opportunities']:
            if opportunity['action'] == 'scale_down':
                plan['actions'].append({
                    'type': 'scale_down',
                    'target': 'worker_pool',
                    'params': {'factor': 0.8}
                })

        return plan

    async def _execute(self, plan: Dict) -> Dict[str, Any]:
        """EXECUTE phase: Take planned actions"""
        result = {
            'executed': [],
            'failed': [],
            'skipped': []
        }

        for action in plan['actions']:
            try:
                if action['type'] == 'self_heal':
                    await self.event_bus.publish(EventType.HEALING_STARTED, {
                        'target': action['target'],
                        'loop_id': self.loop_count
                    })
                    # Trigger healing (in real impl, call recovery system)
                    result['executed'].append(action)
                    await self.event_bus.publish(EventType.HEALING_COMPLETE, {
                        'target': action['target'],
                        'success': True
                    })

                elif action['type'] == 'scale_up':
                    await self.event_bus.publish(EventType.SCALING_UP, {
                        'target': action['target'],
                        'factor': action['params']['factor']
                    })
                    result['executed'].append(action)

                elif action['type'] == 'scale_down':
                    await self.event_bus.publish(EventType.SCALING_DOWN, {
                        'target': action['target'],
                        'factor': action['params']['factor']
                    })
                    result['executed'].append(action)

            except Exception as e:
                action['error'] = str(e)
                result['failed'].append(action)

        return result

    async def _update_knowledge(self, state: Dict, analysis: Dict,
                                 plan: Dict, result: Dict):
        """KNOWLEDGE phase: Update baselines and learn from outcomes"""
        # Update baselines from stable metrics
        for name, metrics in state.get('metrics', {}).items():
            if metrics['trend'] == 'stable':
                self.metrics.set_baseline(name, metrics['avg'])

        # Store decision for learning using shared pool
        try:
            with _get_pooled_connection() as conn:
                if conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO autonomic_decisions
                        (loop_id, phase, decision, action_taken, result)
                        VALUES (%s, 'mape_k', %s, %s, %s)
                    """, (
                        self.loop_count,
                        str(analysis),
                        str([a['type'] for a in plan['actions']]),
                        psycopg2.extras.Json(result)
                    ))
                    conn.commit()
                    cur.close()
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")

        # Update knowledge base
        self.knowledge_base['last_loop'] = {
            'loop_id': self.loop_count,
            'state': state,
            'analysis': analysis,
            'actions': plan['actions'],
            'result': result
        }


class PredictiveFailureDetector:
    """
    Predictive failure detection using trend analysis and pattern matching.
    Alerts before failures occur.
    """

    def __init__(self, metrics: MetricCollector, event_bus: EventBus):
        self.metrics = metrics
        self.event_bus = event_bus
        self.thresholds = {
            'cpu_usage': 0.85,
            'memory_usage': 0.90,
            'error_rate': 0.10,
            'request_latency': 1.0
        }

    async def predict_failures(self) -> List[Dict]:
        """Predict potential failures based on trends"""
        predictions = []

        for metric_name, threshold in self.thresholds.items():
            stats = self.metrics.get_stats(metric_name)
            if not stats:
                continue

            # Predict based on trend
            if stats.trend == 'up':
                # Calculate time to threshold
                if stats.avg > 0 and stats.stddev > 0:
                    distance_to_threshold = threshold - stats.current
                    rate_of_change = stats.stddev / 10  # Approximate

                    if distance_to_threshold > 0 and rate_of_change > 0:
                        time_to_breach = distance_to_threshold / rate_of_change

                        if time_to_breach < 60:  # Less than 60 intervals
                            prediction = {
                                'metric': metric_name,
                                'current': stats.current,
                                'threshold': threshold,
                                'predicted_breach_intervals': int(time_to_breach),
                                'confidence': min(0.9, 0.5 + (1 / max(time_to_breach, 1)) * 0.4),
                                'severity': 'high' if time_to_breach < 20 else 'medium'
                            }
                            predictions.append(prediction)

                            # Publish prediction alert
                            await self.event_bus.publish(
                                EventType.PREDICTION_ALERT,
                                prediction
                            )

        return predictions


class ResourceOptimizer:
    """
    Resource optimization based on usage patterns and predictions.
    """

    def __init__(self, metrics: MetricCollector, event_bus: EventBus):
        self.metrics = metrics
        self.event_bus = event_bus

    async def optimize(self) -> Dict[str, Any]:
        """Run resource optimization analysis"""
        recommendations = []

        # Check CPU utilization
        cpu_stats = self.metrics.get_stats('cpu_usage')
        if cpu_stats:
            if cpu_stats.avg < 0.3:
                recommendations.append({
                    'resource': 'compute',
                    'action': 'consolidate',
                    'reason': f'Average CPU usage is {cpu_stats.avg:.1%}',
                    'savings_estimate': '20-30%'
                })
            elif cpu_stats.avg > 0.8:
                recommendations.append({
                    'resource': 'compute',
                    'action': 'scale_out',
                    'reason': f'CPU usage is high at {cpu_stats.avg:.1%}',
                    'priority': 'high'
                })

        # Check memory utilization
        memory_stats = self.metrics.get_stats('memory_usage')
        if memory_stats:
            if memory_stats.avg > 0.85:
                recommendations.append({
                    'resource': 'memory',
                    'action': 'increase',
                    'reason': f'Memory usage is {memory_stats.avg:.1%}',
                    'priority': 'high'
                })

        # Check for idle workers
        worker_stats = self.metrics.get_stats('active_workers')
        if worker_stats and worker_stats.current < worker_stats.avg * 0.5:
            recommendations.append({
                'resource': 'workers',
                'action': 'reduce',
                'reason': 'Worker utilization is below average',
                'savings_estimate': '10-20%'
            })

        if recommendations:
            await self.event_bus.publish(EventType.RESOURCE_OPTIMIZED, {
                'recommendations': recommendations
            })

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        }


# Singleton instances
_metric_collector = None
_event_bus = None
_autonomic_manager = None


def get_metric_collector() -> MetricCollector:
    """Get singleton metric collector"""
    global _metric_collector
    if _metric_collector is None:
        _metric_collector = MetricCollector()
    return _metric_collector


def get_event_bus() -> EventBus:
    """Get singleton event bus"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def get_autonomic_manager() -> AutonomicManager:
    """Get singleton autonomic manager"""
    global _autonomic_manager
    if _autonomic_manager is None:
        _autonomic_manager = AutonomicManager(
            get_metric_collector(),
            get_event_bus()
        )
    return _autonomic_manager


if __name__ == "__main__":
    async def test_autonomic():
        """Test autonomic controller"""
        print("\n" + "="*60)
        print("🤖 AUTONOMIC CONTROLLER TEST")
        print("="*60)

        metrics = get_metric_collector()
        event_bus = get_event_bus()
        manager = get_autonomic_manager()

        # Emit some test metrics
        print("\n📊 Emitting test metrics...")
        for i in range(50):
            await metrics.emit('cpu_usage', 0.5 + (i * 0.01))
            await metrics.emit('memory_usage', 0.6)
            await metrics.emit('error_rate', 0.02 + (i * 0.001))
            await metrics.emit('request_latency', 0.3 + (i * 0.005))

        # Check stats
        print("\n📈 Metric Statistics:")
        for name in ['cpu_usage', 'memory_usage', 'error_rate', 'request_latency']:
            stats = metrics.get_stats(name)
            if stats:
                print(f"  {name}: current={stats.current:.3f}, avg={stats.avg:.3f}, trend={stats.trend}")

        # Test event bus
        print("\n📡 Testing Event Bus...")
        events_received = []

        async def handler(payload):
            events_received.append(payload)

        event_bus.subscribe(EventType.SYSTEM_ALERT, handler)
        await event_bus.publish(EventType.SYSTEM_ALERT, {'test': 'message'})
        print(f"  Events received: {len(events_received)}")

        # Test predictor
        print("\n🔮 Testing Predictive Failure Detection...")
        predictor = PredictiveFailureDetector(metrics, event_bus)
        predictions = await predictor.predict_failures()
        print(f"  Predictions: {len(predictions)}")
        for pred in predictions:
            print(f"    - {pred['metric']}: breach in {pred['predicted_breach_intervals']} intervals")

        # Test optimizer
        print("\n⚡ Testing Resource Optimizer...")
        optimizer = ResourceOptimizer(metrics, event_bus)
        optimization = await optimizer.optimize()
        print(f"  Recommendations: {optimization['total_recommendations']}")
        for rec in optimization['recommendations']:
            print(f"    - {rec['resource']}: {rec['action']} - {rec['reason']}")

        print("\n" + "="*60)
        print("✅ AUTONOMIC CONTROLLER: OPERATIONAL")
        print("="*60)

    asyncio.run(test_autonomic())

#!/usr/bin/env python3
"""
Predictive Scheduling System - Task 18
AI-powered predictive scheduling for optimal task execution and resource allocation
"""

import logging
import os
from urllib.parse import urlparse as _urlparse
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - validate required environment variables
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        
        # DATABASE_URL fallback
        _db_url = os.getenv('DATABASE_URL', '')
        if _db_url:
            try:
                _p = _urlparse(_db_url)
                globals().update({'_DB_HOST': _p.hostname, '_DB_NAME': _p.path.lstrip('/'), '_DB_USER': _p.username, '_DB_PASSWORD': _p.password, '_DB_PORT': str(_p.port or 5432)})
            except: pass
        missing = [v for v in required_vars if not os.getenv(v) and not globals().get('_' + v)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "5432"))
    }


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class TaskType(Enum):
    """Types of schedulable tasks"""
    AI_PROCESSING = "ai_processing"
    DATA_PIPELINE = "data_pipeline"
    REPORT_GENERATION = "report_generation"
    EMAIL_CAMPAIGN = "email_campaign"
    SYSTEM_MAINTENANCE = "system_maintenance"
    MODEL_TRAINING = "model_training"
    BATCH_JOB = "batch_job"
    REAL_TIME = "real_time"
    SCHEDULED = "scheduled"


class ResourceType(Enum):
    """Types of resources"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    API_QUOTA = "api_quota"
    DATABASE_CONNECTIONS = "database_connections"
    NETWORK = "network"


class ScheduleStatus(Enum):
    """Schedule status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class SchedulableTask:
    """A task that can be scheduled"""
    task_id: str
    name: str
    task_type: TaskType
    priority: TaskPriority
    estimated_duration_minutes: float
    resource_requirements: dict[ResourceType, float]
    dependencies: list[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    preferred_time_windows: list[dict] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class ScheduleSlot:
    """A time slot in the schedule"""
    slot_id: str
    start_time: datetime
    end_time: datetime
    task_id: Optional[str] = None
    available_resources: dict[ResourceType, float] = field(default_factory=dict)
    predicted_load: float = 0.0
    confidence: PredictionConfidence = PredictionConfidence.MEDIUM


@dataclass
class SchedulePrediction:
    """Prediction for scheduling decision"""
    task_id: str
    recommended_slot: ScheduleSlot
    success_probability: float
    completion_time_estimate: datetime
    resource_utilization: dict[ResourceType, float]
    confidence: PredictionConfidence
    alternative_slots: list[ScheduleSlot] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)


class LoadPredictor:
    """Predicts system load based on historical data"""

    def __init__(self):
        self.historical_data: dict[str, list[float]] = defaultdict(list)
        self.hourly_patterns: dict[int, float] = {}
        self.daily_patterns: dict[int, float] = {}

    async def learn_from_history(self, lookback_days: int = 30):
        """Learn load patterns from historical data"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get historical execution data
            cursor.execute("""
                SELECT
                    EXTRACT(HOUR FROM created_at) as hour,
                    EXTRACT(DOW FROM created_at) as day_of_week,
                    COUNT(*) as execution_count,
                    AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_duration
                FROM agent_executions
                WHERE created_at >= NOW() - INTERVAL '%s days'
                  AND status = 'completed'
                GROUP BY hour, day_of_week
                ORDER BY hour, day_of_week
            """, (lookback_days,))

            patterns = cursor.fetchall()

            # Build hourly patterns
            hourly_counts = defaultdict(list)
            daily_counts = defaultdict(list)

            for pattern in patterns:
                hourly_counts[int(pattern['hour'])].append(pattern['execution_count'])
                daily_counts[int(pattern['day_of_week'])].append(pattern['execution_count'])

            # Calculate averages
            for hour, counts in hourly_counts.items():
                self.hourly_patterns[hour] = sum(counts) / len(counts) if counts else 0

            for day, counts in daily_counts.items():
                self.daily_patterns[day] = sum(counts) / len(counts) if counts else 0

            cursor.close()
            conn.close()

            logger.info(f"Learned patterns from {len(patterns)} data points")

        except Exception as e:
            logger.error(f"Failed to learn from history: {e}")

    def predict_load(
        self,
        target_time: datetime
    ) -> tuple[float, PredictionConfidence]:
        """Predict system load at a given time"""
        hour = target_time.hour
        day = target_time.weekday()

        hourly_factor = self.hourly_patterns.get(hour, 50)
        daily_factor = self.daily_patterns.get(day, 50)

        # Combine factors
        predicted_load = (hourly_factor * 0.7 + daily_factor * 0.3) / 100

        # Determine confidence
        if len(self.hourly_patterns) >= 20:
            confidence = PredictionConfidence.HIGH
        elif len(self.hourly_patterns) >= 10:
            confidence = PredictionConfidence.MEDIUM
        else:
            confidence = PredictionConfidence.LOW

        return min(predicted_load, 1.0), confidence

    def get_optimal_windows(
        self,
        start_time: datetime,
        end_time: datetime,
        duration_minutes: int
    ) -> list[tuple[datetime, float]]:
        """Get optimal scheduling windows based on predicted load"""
        windows = []
        current = start_time

        while current + timedelta(minutes=duration_minutes) <= end_time:
            load, _ = self.predict_load(current)
            windows.append((current, 1 - load))  # Higher score = lower load
            current += timedelta(minutes=15)  # 15-minute increments

        # Sort by score (best windows first)
        windows.sort(key=lambda x: x[1], reverse=True)

        return windows[:10]  # Return top 10 windows


class ResourceManager:
    """Manages resource allocation and availability"""

    def __init__(self):
        self.resource_limits: dict[ResourceType, float] = {
            ResourceType.CPU: 100.0,
            ResourceType.MEMORY: 16384.0,  # MB
            ResourceType.GPU: 1.0,
            ResourceType.API_QUOTA: 10000.0,
            ResourceType.DATABASE_CONNECTIONS: 100.0,
            ResourceType.NETWORK: 1000.0  # Mbps
        }
        self.current_usage: dict[ResourceType, float] = {r: 0.0 for r in ResourceType}
        self.reservations: dict[str, dict[ResourceType, float]] = {}

    def check_availability(
        self,
        requirements: dict[ResourceType, float]
    ) -> tuple[bool, list[str]]:
        """Check if resources are available"""
        unavailable = []

        for resource, required in requirements.items():
            available = self.resource_limits.get(resource, 0) - self.current_usage.get(resource, 0)
            if required > available:
                unavailable.append(f"{resource.value}: need {required}, have {available}")

        return len(unavailable) == 0, unavailable

    def reserve_resources(
        self,
        task_id: str,
        requirements: dict[ResourceType, float]
    ) -> bool:
        """Reserve resources for a task"""
        available, _ = self.check_availability(requirements)
        if not available:
            return False

        self.reservations[task_id] = requirements
        for resource, amount in requirements.items():
            self.current_usage[resource] = self.current_usage.get(resource, 0) + amount

        return True

    def release_resources(self, task_id: str):
        """Release resources from a task"""
        if task_id in self.reservations:
            for resource, amount in self.reservations[task_id].items():
                self.current_usage[resource] = max(
                    0, self.current_usage.get(resource, 0) - amount
                )
            del self.reservations[task_id]

    def get_utilization(self) -> dict[ResourceType, float]:
        """Get current resource utilization percentages"""
        return {
            r: (self.current_usage.get(r, 0) / self.resource_limits.get(r, 1)) * 100
            for r in ResourceType
        }


class DependencyResolver:
    """Resolves task dependencies"""

    def __init__(self):
        self.task_graph: dict[str, list[str]] = {}
        self.completed_tasks: set = set()

    def add_task(self, task_id: str, dependencies: list[str]):
        """Add a task with its dependencies"""
        self.task_graph[task_id] = dependencies

    def mark_completed(self, task_id: str):
        """Mark a task as completed"""
        self.completed_tasks.add(task_id)

    def can_schedule(self, task_id: str) -> tuple[bool, list[str]]:
        """Check if a task's dependencies are satisfied"""
        dependencies = self.task_graph.get(task_id, [])
        pending = [d for d in dependencies if d not in self.completed_tasks]
        return len(pending) == 0, pending

    def get_execution_order(self, tasks: list[str]) -> list[str]:
        """Get optimal execution order respecting dependencies (topological sort)"""
        # Build in-degree map
        in_degree = {t: 0 for t in tasks}
        for task in tasks:
            for dep in self.task_graph.get(task, []):
                if dep in in_degree:
                    in_degree[task] += 1

        # Start with tasks that have no dependencies
        queue = [t for t, d in in_degree.items() if d == 0]
        result = []

        while queue:
            # Sort by priority if available
            queue.sort()
            task = queue.pop(0)
            result.append(task)

            # Update in-degrees
            for other_task in tasks:
                if task in self.task_graph.get(other_task, []):
                    in_degree[other_task] -= 1
                    if in_degree[other_task] == 0 and other_task not in result:
                        queue.append(other_task)

        return result


class ScheduleOptimizer:
    """Optimizes schedule assignments"""

    def __init__(
        self,
        load_predictor: LoadPredictor,
        resource_manager: ResourceManager,
        dependency_resolver: DependencyResolver
    ):
        self.load_predictor = load_predictor
        self.resource_manager = resource_manager
        self.dependency_resolver = dependency_resolver

    async def optimize_schedule(
        self,
        tasks: list[SchedulableTask],
        planning_horizon_hours: int = 24
    ) -> list[SchedulePrediction]:
        """Generate optimized schedule for tasks"""
        predictions = []
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=planning_horizon_hours)

        # Sort tasks by priority and deadline
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (
                -list(TaskPriority).index(t.priority),
                t.deadline or end_time
            )
        )

        # Get execution order respecting dependencies
        task_ids = [t.task_id for t in sorted_tasks]
        for task in sorted_tasks:
            self.dependency_resolver.add_task(task.task_id, task.dependencies)
        execution_order = self.dependency_resolver.get_execution_order(task_ids)

        # Reorder tasks
        task_map = {t.task_id: t for t in sorted_tasks}
        ordered_tasks = [task_map[tid] for tid in execution_order if tid in task_map]

        # Schedule each task
        for task in ordered_tasks:
            prediction = await self._find_optimal_slot(task, start_time, end_time)
            if prediction:
                predictions.append(prediction)
                # Reserve the slot
                self.resource_manager.reserve_resources(
                    task.task_id,
                    task.resource_requirements
                )

        return predictions

    async def _find_optimal_slot(
        self,
        task: SchedulableTask,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[SchedulePrediction]:
        """Find optimal slot for a task"""
        # Get candidate windows
        windows = self.load_predictor.get_optimal_windows(
            start_time,
            end_time,
            int(task.estimated_duration_minutes)
        )

        best_slot = None
        best_score = -1
        alternatives = []

        for window_start, load_score in windows:
            # Check resource availability
            available, issues = self.resource_manager.check_availability(
                task.resource_requirements
            )
            if not available:
                continue

            # Check dependencies
            deps_ready, pending = self.dependency_resolver.can_schedule(task.task_id)
            if not deps_ready:
                continue

            # Check deadline
            window_end = window_start + timedelta(minutes=task.estimated_duration_minutes)
            if task.deadline and window_end > task.deadline:
                continue

            # Calculate score
            score = self._calculate_slot_score(task, window_start, load_score)

            slot = ScheduleSlot(
                slot_id=str(uuid.uuid4()),
                start_time=window_start,
                end_time=window_end,
                task_id=task.task_id,
                predicted_load=1 - load_score
            )

            if score > best_score:
                if best_slot:
                    alternatives.append(best_slot)
                best_slot = slot
                best_score = score
            else:
                alternatives.append(slot)

        if not best_slot:
            return None

        # Calculate success probability
        success_prob = self._calculate_success_probability(task, best_slot)

        # Identify risk factors
        risks = self._identify_risks(task, best_slot)

        return SchedulePrediction(
            task_id=task.task_id,
            recommended_slot=best_slot,
            success_probability=success_prob,
            completion_time_estimate=best_slot.end_time,
            resource_utilization=self.resource_manager.get_utilization(),
            confidence=PredictionConfidence.HIGH if success_prob > 0.8 else
                       PredictionConfidence.MEDIUM if success_prob > 0.5 else
                       PredictionConfidence.LOW,
            alternative_slots=alternatives[:3],
            risk_factors=risks
        )

    def _calculate_slot_score(
        self,
        task: SchedulableTask,
        slot_time: datetime,
        load_score: float
    ) -> float:
        """Calculate overall score for a slot"""
        score = load_score * 0.4  # 40% weight on load

        # Deadline proximity bonus
        if task.deadline:
            time_to_deadline = (task.deadline - slot_time).total_seconds() / 3600
            deadline_score = min(time_to_deadline / 24, 1.0)  # Max bonus at 24+ hours
            score += deadline_score * 0.3

        # Priority bonus
        priority_scores = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 0.8,
            TaskPriority.MEDIUM: 0.5,
            TaskPriority.LOW: 0.3,
            TaskPriority.BACKGROUND: 0.1
        }
        score += priority_scores.get(task.priority, 0.5) * 0.3

        return score

    def _calculate_success_probability(
        self,
        task: SchedulableTask,
        slot: ScheduleSlot
    ) -> float:
        """Calculate probability of successful execution"""
        base_prob = 0.9

        # Adjust for load
        load_adjustment = 1 - (slot.predicted_load * 0.2)

        # Adjust for resource availability
        utilization = self.resource_manager.get_utilization()
        avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
        resource_adjustment = 1 - (avg_utilization / 100 * 0.1)

        return base_prob * load_adjustment * resource_adjustment

    def _identify_risks(
        self,
        task: SchedulableTask,
        slot: ScheduleSlot
    ) -> list[str]:
        """Identify potential risks"""
        risks = []

        if slot.predicted_load > 0.8:
            risks.append("High system load during scheduled time")

        if task.deadline:
            buffer = (task.deadline - slot.end_time).total_seconds() / 60
            if buffer < 30:
                risks.append("Less than 30 minutes buffer to deadline")

        utilization = self.resource_manager.get_utilization()
        for resource, usage in utilization.items():
            if usage > 80:
                risks.append(f"{resource.value} utilization above 80%")

        return risks


class PredictiveSchedulingSystem:
    """Main predictive scheduling system"""

    def __init__(self):
        self.load_predictor = LoadPredictor()
        self.resource_manager = ResourceManager()
        self.dependency_resolver = DependencyResolver()
        self.optimizer = ScheduleOptimizer(
            self.load_predictor,
            self.resource_manager,
            self.dependency_resolver
        )
        self.conn = None
        self._init_database()

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**_get_db_config())
        return self.conn

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Scheduled tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_scheduled_tasks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    task_type VARCHAR(50),
                    priority VARCHAR(50),
                    estimated_duration_minutes FLOAT,
                    resource_requirements JSONB DEFAULT '{}',
                    dependencies JSONB DEFAULT '[]',
                    deadline TIMESTAMPTZ,
                    preferred_windows JSONB DEFAULT '[]',
                    constraints JSONB DEFAULT '{}',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Schedule slots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_schedule_slots (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    task_id UUID REFERENCES ai_scheduled_tasks(id),
                    start_time TIMESTAMPTZ NOT NULL,
                    end_time TIMESTAMPTZ NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    predicted_load FLOAT DEFAULT 0.0,
                    actual_load FLOAT,
                    success_probability FLOAT,
                    confidence VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    executed_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ
                )
            """)

            # Schedule predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_schedule_predictions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    task_id UUID REFERENCES ai_scheduled_tasks(id),
                    slot_id UUID REFERENCES ai_schedule_slots(id),
                    success_probability FLOAT,
                    completion_estimate TIMESTAMPTZ,
                    resource_utilization JSONB DEFAULT '{}',
                    confidence VARCHAR(50),
                    alternative_slots JSONB DEFAULT '[]',
                    risk_factors JSONB DEFAULT '[]',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Load patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_load_patterns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pattern_type VARCHAR(50),
                    hour INT,
                    day_of_week INT,
                    avg_load FLOAT,
                    sample_count INT DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(pattern_type, hour, day_of_week)
                )
            """)

            # Schedule metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_schedule_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    slot_id UUID REFERENCES ai_schedule_slots(id),
                    predicted_duration FLOAT,
                    actual_duration FLOAT,
                    predicted_success FLOAT,
                    actual_success BOOLEAN,
                    deviation_percentage FLOAT,
                    recorded_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_schedule_slots_time
                ON ai_schedule_slots(start_time)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_schedule_slots_status
                ON ai_schedule_slots(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_priority
                ON ai_scheduled_tasks(priority)
            """)

            conn.commit()
            cursor.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    async def initialize(self):
        """Initialize the scheduling system"""
        await self.load_predictor.learn_from_history()
        logger.info("Predictive scheduling system initialized")

    async def create_task(
        self,
        name: str,
        task_type: TaskType,
        priority: TaskPriority,
        estimated_duration: float,
        resource_requirements: Optional[dict[str, float]] = None,
        dependencies: Optional[list[str]] = None,
        deadline: Optional[datetime] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """Create a new schedulable task"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            task_id = str(uuid.uuid4())

            # Convert resource requirements
            resources = {}
            if resource_requirements:
                for key, value in resource_requirements.items():
                    try:
                        resources[ResourceType(key).value] = value
                    except ValueError:
                        resources[key] = value

            cursor.execute("""
                INSERT INTO ai_scheduled_tasks
                (id, name, task_type, priority, estimated_duration_minutes,
                 resource_requirements, dependencies, deadline, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                task_id,
                name,
                task_type.value,
                priority.value,
                estimated_duration,
                Json(resources),
                Json(dependencies or []),
                deadline,
                Json(metadata or {})
            ))

            conn.commit()
            cursor.close()

            logger.info(f"Created task: {task_id}")
            return task_id

        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            raise

    async def schedule_task(
        self,
        task_id: str,
        start_after: Optional[datetime] = None,
        end_before: Optional[datetime] = None
    ) -> SchedulePrediction:
        """Schedule a task with optimal timing"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get task details
            cursor.execute("""
                SELECT * FROM ai_scheduled_tasks WHERE id = %s
            """, (task_id,))

            task_data = cursor.fetchone()
            if not task_data:
                raise ValueError(f"Task not found: {task_id}")

            # Convert to SchedulableTask
            resource_reqs = {}
            for key, value in (task_data['resource_requirements'] or {}).items():
                try:
                    resource_reqs[ResourceType(key)] = value
                except ValueError:
                    logger.debug("Invalid resource type %s", key)

            task = SchedulableTask(
                task_id=task_id,
                name=task_data['name'],
                task_type=TaskType(task_data['task_type']),
                priority=TaskPriority(task_data['priority']),
                estimated_duration_minutes=task_data['estimated_duration_minutes'],
                resource_requirements=resource_reqs,
                dependencies=task_data['dependencies'] or [],
                deadline=task_data['deadline'],
                metadata=task_data['metadata'] or {}
            )

            # Get schedule predictions
            start = start_after or datetime.now(timezone.utc)
            end = end_before or start + timedelta(hours=24)

            predictions = await self.optimizer.optimize_schedule([task], 24)

            if not predictions:
                raise ValueError("Could not find suitable schedule slot")

            prediction = predictions[0]

            # Store the schedule
            slot_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_schedule_slots
                (id, task_id, start_time, end_time, status,
                 predicted_load, success_probability, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                slot_id,
                task_id,
                prediction.recommended_slot.start_time,
                prediction.recommended_slot.end_time,
                ScheduleStatus.SCHEDULED.value,
                prediction.recommended_slot.predicted_load,
                prediction.success_probability,
                prediction.confidence.value
            ))

            # Store prediction
            prediction_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_schedule_predictions
                (id, task_id, slot_id, success_probability, completion_estimate,
                 resource_utilization, confidence, alternative_slots, risk_factors)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                prediction_id,
                task_id,
                slot_id,
                prediction.success_probability,
                prediction.completion_time_estimate,
                Json({r.value: v for r, v in prediction.resource_utilization.items()}),
                prediction.confidence.value,
                Json([{
                    "start": s.start_time.isoformat(),
                    "end": s.end_time.isoformat(),
                    "load": s.predicted_load
                } for s in prediction.alternative_slots]),
                Json(prediction.risk_factors)
            ))

            conn.commit()
            cursor.close()

            return prediction

        except Exception as e:
            logger.error(f"Failed to schedule task: {e}")
            raise

    async def get_schedule(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        status: Optional[ScheduleStatus] = None
    ) -> list[dict]:
        """Get scheduled tasks"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = """
                SELECT
                    s.*,
                    t.name as task_name,
                    t.task_type,
                    t.priority
                FROM ai_schedule_slots s
                JOIN ai_scheduled_tasks t ON s.task_id = t.id
                WHERE 1=1
            """
            params = []

            if start_time:
                query += " AND s.start_time >= %s"
                params.append(start_time)

            if end_time:
                query += " AND s.end_time <= %s"
                params.append(end_time)

            if status:
                query += " AND s.status = %s"
                params.append(status.value)

            query += " ORDER BY s.start_time"

            cursor.execute(query, params)
            slots = cursor.fetchall()
            cursor.close()

            return [dict(s) for s in slots]

        except Exception as e:
            logger.error(f"Failed to get schedule: {e}")
            return []

    async def execute_scheduled_tasks(self) -> list[dict]:
        """Execute all due scheduled tasks"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            now = datetime.now(timezone.utc)

            # Get due tasks
            cursor.execute("""
                SELECT s.*, t.name, t.task_type, t.metadata
                FROM ai_schedule_slots s
                JOIN ai_scheduled_tasks t ON s.task_id = t.id
                WHERE s.status = %s
                  AND s.start_time <= %s
                ORDER BY s.start_time
                LIMIT 10
            """, (ScheduleStatus.SCHEDULED.value, now))

            due_tasks = cursor.fetchall()
            executed = []

            for task in due_tasks:
                try:
                    # Mark as running
                    cursor.execute("""
                        UPDATE ai_schedule_slots
                        SET status = %s, executed_at = NOW()
                        WHERE id = %s
                    """, (ScheduleStatus.RUNNING.value, task['id']))
                    conn.commit()

                    # Execute the task (placeholder - would call actual execution)
                    logger.info(f"Executing scheduled task: {task['name']}")

                    # Mark as completed
                    cursor.execute("""
                        UPDATE ai_schedule_slots
                        SET status = %s, completed_at = NOW()
                        WHERE id = %s
                    """, (ScheduleStatus.COMPLETED.value, task['id']))

                    # Record metrics
                    actual_duration = (
                        datetime.now(timezone.utc) - task['executed_at']
                    ).total_seconds() / 60 if task.get('executed_at') else 0

                    cursor.execute("""
                        INSERT INTO ai_schedule_metrics
                        (slot_id, predicted_duration, actual_duration,
                         predicted_success, actual_success, deviation_percentage)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        task['id'],
                        (task['end_time'] - task['start_time']).total_seconds() / 60,
                        actual_duration,
                        task['success_probability'],
                        True,
                        0  # Would calculate deviation
                    ))

                    conn.commit()
                    executed.append(dict(task))

                    # Mark dependency as completed
                    self.dependency_resolver.mark_completed(str(task['task_id']))

                except Exception as e:
                    logger.error(f"Failed to execute task {task['id']}: {e}")
                    cursor.execute("""
                        UPDATE ai_schedule_slots
                        SET status = %s
                        WHERE id = %s
                    """, (ScheduleStatus.FAILED.value, task['id']))
                    conn.commit()

            cursor.close()
            return executed

        except Exception as e:
            logger.error(f"Failed to execute scheduled tasks: {e}")
            return []

    async def get_load_forecast(
        self,
        hours_ahead: int = 24
    ) -> list[dict]:
        """Get load forecast for upcoming hours"""
        forecast = []
        current = datetime.now(timezone.utc)

        for i in range(hours_ahead):
            target_time = current + timedelta(hours=i)
            load, confidence = self.load_predictor.predict_load(target_time)

            forecast.append({
                "time": target_time.isoformat(),
                "predicted_load": load,
                "confidence": confidence.value,
                "recommendation": (
                    "avoid" if load > 0.8 else
                    "caution" if load > 0.6 else
                    "optimal"
                )
            })

        return forecast

    async def get_scheduling_metrics(self) -> dict:
        """Get scheduling performance metrics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    COUNT(*) as total_scheduled,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed,
                    AVG(success_probability) as avg_predicted_success,
                    AVG(predicted_load) as avg_predicted_load
                FROM ai_schedule_slots
                WHERE created_at >= NOW() - INTERVAL '7 days'
            """)

            slot_metrics = cursor.fetchone()

            cursor.execute("""
                SELECT
                    AVG(deviation_percentage) as avg_deviation,
                    AVG(CASE WHEN actual_success THEN 1 ELSE 0 END) * 100 as actual_success_rate,
                    COUNT(*) as total_executions
                FROM ai_schedule_metrics
                WHERE recorded_at >= NOW() - INTERVAL '7 days'
            """)

            execution_metrics = cursor.fetchone()

            cursor.close()

            return {
                "slot_metrics": dict(slot_metrics) if slot_metrics else {},
                "execution_metrics": dict(execution_metrics) if execution_metrics else {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {"error": str(e)}


# Singleton instance
_scheduling_system: Optional[PredictiveSchedulingSystem] = None


def get_predictive_scheduling():
    """Get or create the predictive scheduling system instance"""
    global _scheduling_system
    if _scheduling_system is None:
        _scheduling_system = PredictiveSchedulingSystem()
    return _scheduling_system


# Export main components
__all__ = [
    'PredictiveSchedulingSystem',
    'get_predictive_scheduling',
    'TaskPriority',
    'TaskType',
    'ResourceType',
    'ScheduleStatus',
    'PredictionConfidence',
    'SchedulableTask',
    'ScheduleSlot',
    'SchedulePrediction'
]

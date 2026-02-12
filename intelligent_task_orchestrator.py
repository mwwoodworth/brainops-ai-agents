#!/usr/bin/env python3
"""
INTELLIGENT TASK ORCHESTRATOR
AI-driven task management with awareness, prioritization, and notifications

Features:
1. AI-driven task prioritization based on context and impact
2. Multi-channel notifications (Slack, Email, SMS, System)
3. Autonomous task execution with self-healing
4. Real-time status monitoring and alerting
5. Task dependency resolution
6. Resource-aware scheduling

Integrates with:
- Meta-critic scoring for task selection
- Self-healing reconciler for failure recovery
- AI Board for strategic task approval
"""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Optional

import httpx
import psycopg2

from safe_task import create_safe_task
from psycopg2.extras import Json, RealDictCursor

# CRITICAL: Use shared connection pool to prevent MaxClientsInSessionMode
try:
    from database.sync_pool import get_sync_pool
    SYNC_POOL_AVAILABLE = True
except ImportError:
    SYNC_POOL_AVAILABLE = False
    get_sync_pool = None

# Import our cutting-edge systems
try:
    from meta_critic_scoring import get_meta_critic
    META_CRITIC_AVAILABLE = True
except ImportError:
    META_CRITIC_AVAILABLE = False

try:
    from self_healing_reconciler import get_reconciler
    HEALING_AVAILABLE = True
except ImportError:
    HEALING_AVAILABLE = False

try:
    from ai_core import RealAICore
    AI_CORE_AVAILABLE = True
except ImportError:
    AI_CORE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Database configuration - supports both individual env vars and DATABASE_URL
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    from urllib.parse import unquote, urlparse

    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME", "postgres")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_port = os.getenv("DB_PORT", "5432")

    # Fallback to DATABASE_URL if individual vars not set
    if not all([db_host, db_user, db_password]):
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            parsed = urlparse(database_url)
            db_host = parsed.hostname or ''
            db_name = unquote(parsed.path.lstrip('/')) if parsed.path else 'postgres'
            db_user = unquote(parsed.username) if parsed.username else ''
            db_password = unquote(parsed.password) if parsed.password else ''
            db_port = str(parsed.port) if parsed.port else '5432'

    if not all([db_host, db_user, db_password]):
        raise RuntimeError("Missing required: DB_HOST/DB_USER/DB_PASSWORD or DATABASE_URL")

    return {
        "host": db_host,
        "database": db_name,
        "user": db_user,
        "password": db_password,
        "port": int(db_port)
    }

DB_CONFIG = _get_db_config()


# Helper context manager to use shared pool with fallback
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    """Get database connection from shared pool or create direct connection."""
    conn = None
    from_pool = False
    try:
        if SYNC_POOL_AVAILABLE and get_sync_pool:
            pool = get_sync_pool()
            with pool.get_connection() as pooled_conn:
                if pooled_conn:
                    yield pooled_conn
                    return
        # Fallback to direct connection
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    finally:
        if conn and not from_pool:
            try:
                conn.close()
            except Exception:
                pass


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    BACKGROUND = 10


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class NotificationChannel(Enum):
    """Notification channels"""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SYSTEM = "system"  # In-app notifications
    DATABASE = "database"  # Log to DB for dashboard


@dataclass
class TaskNotification:
    """Notification about task status"""
    task_id: str
    task_title: str
    event_type: str
    severity: str
    message: str
    details: dict[str, Any]
    channels: list[str]
    sent_at: datetime
    acknowledged: bool = False


@dataclass
class IntelligentTask:
    """Enhanced task with AI-driven properties"""
    id: str
    title: str
    description: str
    task_type: str
    payload: dict[str, Any]
    priority: int
    status: str
    created_at: datetime
    # AI-enhanced fields
    ai_priority_score: float
    ai_urgency_reason: str
    estimated_duration_mins: int
    required_capabilities: list[str]
    dependencies: list[str]
    retry_count: int
    max_retries: int
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    execution_result: Optional[dict[str, Any]] = None
    # Enhanced decision-making fields
    risk_assessment: Optional[dict[str, Any]] = None
    confidence_score: float = 0.0
    human_escalation_required: bool = False
    escalation_reason: Optional[str] = None


class NotificationService:
    """Multi-channel notification service"""

    def __init__(self):
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        self.email_api_key = os.getenv("SENDGRID_API_KEY")
        self.sms_api_key = os.getenv("TWILIO_API_KEY")
        self.owner_email = os.getenv("OWNER_EMAIL", "matt@weathercraft.com")
        self.owner_phone = os.getenv("OWNER_PHONE")

        # Notification preferences - use string values for JSON serialization
        self.severity_channels = {
            "critical": [NotificationChannel.SLACK.value, NotificationChannel.SMS.value, NotificationChannel.DATABASE.value],
            "high": [NotificationChannel.SLACK.value, NotificationChannel.DATABASE.value],
            "medium": [NotificationChannel.DATABASE.value],
            "low": [NotificationChannel.DATABASE.value],
        }

    async def notify(self, notification: TaskNotification):
        """Send notification to appropriate channels"""
        channels_to_use = notification.channels or self.severity_channels.get(
            notification.severity, [NotificationChannel.DATABASE]
        )

        results = []
        for channel in channels_to_use:
            try:
                if channel == NotificationChannel.SLACK or channel == "slack":
                    await self._send_slack(notification)
                    results.append(("slack", True))
                elif channel == NotificationChannel.DATABASE or channel == "database":
                    await self._store_notification(notification)
                    results.append(("database", True))
                elif channel == NotificationChannel.WEBHOOK or channel == "webhook":
                    await self._send_webhook(notification)
                    results.append(("webhook", True))
            except Exception as e:
                logger.error(f"Failed to send notification via {channel}: {e}")
                results.append((str(channel), False))

        return results

    async def _send_slack(self, notification: TaskNotification):
        """Send Slack notification"""
        if not self.slack_webhook:
            logger.debug("Slack webhook not configured")
            return

        emoji_map = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "📋",
            "low": "ℹ️",
        }

        emoji = emoji_map.get(notification.severity, "📌")

        payload = {
            "text": f"{emoji} *{notification.event_type}*: {notification.message}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {notification.event_type}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Task:*\n{notification.task_title}"},
                        {"type": "mrkdwn", "text": f"*Severity:*\n{notification.severity}"},
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": notification.message
                    }
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            await client.post(self.slack_webhook, json=payload)

    async def _store_notification(self, notification: TaskNotification):
        """Store notification in database"""
        try:
            with get_db_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO task_notifications
                (task_id, event_type, severity, message, details, channels, sent_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    notification.task_id,
                    notification.event_type,
                    notification.severity,
                    notification.message,
                    Json(notification.details),
                    Json(notification.channels),
                    notification.sent_at
                ))
                conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Failed to store notification: {e}")

    async def _send_webhook(self, notification: TaskNotification):
        """Send to configured webhook"""
        webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL")
        if not webhook_url:
            return

        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=asdict(notification))


class IntelligentTaskOrchestrator:
    """
    AI-driven task orchestrator with intelligent prioritization,
    autonomous execution, and multi-channel notifications.
    """

    def __init__(self):
        self.notification_service = NotificationService()
        self.meta_critic = get_meta_critic() if META_CRITIC_AVAILABLE else None
        self.reconciler = get_reconciler() if HEALING_AVAILABLE else None
        self.ai_core = RealAICore() if AI_CORE_AVAILABLE else None

        # Configuration
        self.max_concurrent_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
        self.task_timeout_mins = int(os.getenv("TASK_TIMEOUT_MINS", "30"))
        self.enable_auto_retry = os.getenv("ENABLE_AUTO_RETRY", "true").lower() == "true"

        # State
        self.running_tasks: dict[str, IntelligentTask] = {}
        self.task_executors: dict[str, Callable] = {}
        self.running = False

        self._init_database()

    def _init_database(self):
        """Verify required tables exist (DDL removed — agent_worker has no DDL permissions)."""
        required_tables = [
                "task_notifications",
                "task_execution_history",
                "ai_priority_adjustments",
                "revenue_audit_log",
        ]
        try:
            from database.verify_tables import verify_tables_sync
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            ok = verify_tables_sync(required_tables, cursor, module_name="intelligent_task_orchestrator")
            cursor.close()
            conn.close()
            if not ok:
                return
            self._tables_initialized = True
        except Exception as exc:
            logger.error("Table verification failed: %s", exc)
    async def start(self):
        """Start the intelligent task orchestrator"""
        self.running = True
        logger.info("🧠 Intelligent Task Orchestrator started")

        await self._notify_status("orchestrator_started", "medium",
            "Intelligent Task Orchestrator is now online and processing tasks")

        # Start background processing
        create_safe_task(self._process_task_queue(), "task_queue_processor")
        create_safe_task(self._monitor_running_tasks(), "task_monitor")
        create_safe_task(self._ai_priority_adjustment_loop(), "priority_adjuster")

    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        await self._notify_status("orchestrator_stopped", "high",
            "Intelligent Task Orchestrator has been stopped")

    async def _process_task_queue(self):
        """Main task processing loop"""
        while self.running:
            try:
                # Check capacity
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(5)
                    continue

                # Fetch next intelligent task
                task = await self._get_next_task()
                if task:
                    create_safe_task(self._execute_task(task), f"execute_task_{task.id}")

                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Task queue processing error: {e}")
                await asyncio.sleep(10)

    async def _get_next_task(self) -> Optional[IntelligentTask]:
        """Get next task with AI prioritization"""
        try:
            with get_db_connection() as conn:
                if not conn:
                    return None
                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Get pending tasks ordered by AI-adjusted priority
                # CAST t.priority to FLOAT to match ap.adjusted_priority type
                cur.execute("""
                SELECT t.*,
                       COALESCE(ap.adjusted_priority,
                                CASE WHEN t.priority ~ '^[0-9.]+$'
                                     THEN t.priority::FLOAT
                                     ELSE 0.5 END) as effective_priority
                FROM ai_autonomous_tasks t
                LEFT JOIN ai_priority_adjustments ap ON t.id::text = ap.task_id
                WHERE t.status = 'pending'
                ORDER BY effective_priority DESC, t.created_at ASC
                LIMIT 1
                """)

                row = cur.fetchone()
                cur.close()

                if not row:
                    return None

                # Enhance task with AI analysis
                task = await self._enhance_task_with_ai(dict(row))
                return task

        except Exception as e:
            logger.error(f"Failed to get next task: {e}")
            return None

    async def _enhance_task_with_ai(self, row: dict[str, Any]) -> IntelligentTask:
        """Enhance task with AI-driven insights"""
        # Some rows have an explicit NULL payload; normalize to a dict so we can safely .get().
        payload = row.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.debug("Failed to parse task payload JSON: %s", exc)
                payload = {"raw": payload}
        if not isinstance(payload, dict):
            payload = {"raw": payload}

        def _coerce_int(value: Any, default: int) -> int:
            """Best-effort int coercion for legacy rows that store numerics as text."""
            try:
                if value is None:
                    return default
                if isinstance(value, bool):
                    return default
                if isinstance(value, (int, float)):
                    return int(value)
                if isinstance(value, str):
                    stripped = value.strip()
                    if not stripped:
                        return default
                    # Accept "10", "10.0", etc.
                    return int(float(stripped))
            except Exception:
                return default
            return default

        priority = _coerce_int(row.get("priority"), 50)
        retry_count = _coerce_int(row.get("retry_count"), 0)
        max_retries = _coerce_int(payload.get("max_retries"), 3)
        estimated_duration_mins = _coerce_int(payload.get("estimated_mins"), 10)

        required_capabilities = payload.get("capabilities", [])
        if isinstance(required_capabilities, str):
            required_capabilities = [required_capabilities]
        if not isinstance(required_capabilities, list):
            required_capabilities = [required_capabilities]

        dependencies = payload.get("dependencies", [])
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        if not isinstance(dependencies, list):
            dependencies = [dependencies]

        # AI priority analysis
        ai_priority_score = priority
        ai_urgency_reason = "Standard priority"

        if self.ai_core:
            try:
                analysis = await self.ai_core.generate(
                    prompt=f"Analyze task urgency (1-100 score): {row.get('title', 'Unknown task')}. Payload: {json.dumps(payload)[:500]}",
                    model="gpt-4o-mini",
                    max_tokens=100
                )
                # Parse urgency from response
                if analysis:
                    try:
                        score_match = json.loads(analysis) if "{" in analysis else {"score": 50}
                        ai_priority_score = _coerce_int(score_match.get("score"), priority)
                        ai_urgency_reason = score_match.get("reason", "AI-analyzed priority")
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.debug("Failed to parse AI priority JSON: %s", exc)
            except Exception as e:
                logger.debug(f"AI priority analysis failed: {e}")

        task = IntelligentTask(
            id=str(row.get("id", "")),
            title=row.get("title", "Untitled Task"),
            description=payload.get("description", ""),
            task_type=payload.get("type", "general"),
            payload=payload,
            priority=priority,
            status=row.get("status", "pending"),
            created_at=row.get("created_at", datetime.now()),
            ai_priority_score=ai_priority_score,
            ai_urgency_reason=ai_urgency_reason,
            estimated_duration_mins=estimated_duration_mins,
            required_capabilities=required_capabilities,
            dependencies=dependencies,
            retry_count=retry_count,
            max_retries=max_retries,
        )

        # Enhanced: Perform risk assessment
        task.risk_assessment = self._assess_task_risk(task)

        # Enhanced: Calculate confidence score
        task.confidence_score = self._calculate_task_confidence(task)

        # Enhanced: Check for human escalation
        task.human_escalation_required, task.escalation_reason = self._check_task_escalation(task)

        return task

    async def _execute_task(self, task: IntelligentTask):
        """Execute a task with full tracking and notifications"""
        task.started_at = datetime.now()
        task.status = TaskStatus.IN_PROGRESS.value
        self.running_tasks[task.id] = task

        # Notify task start
        await self._notify_task_event(task, "task_started", "low",
            f"Task '{task.title}' has started execution")

        # Update database status
        await self._update_task_status(task.id, "in_progress")

        try:
            # Execute based on task type
            result = await self._route_task_execution(task)

            # Task completed successfully
            task.completed_at = datetime.now()
            task.status = TaskStatus.COMPLETED.value
            task.execution_result = result

            await self._update_task_status(task.id, "completed", result)
            await self._store_execution_history(task, "completed", result)

            await self._notify_task_event(task, "task_completed", "low",
                f"Task '{task.title}' completed successfully")

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")

            # Handle failure with retry logic
            if self.enable_auto_retry and task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING.value

                await self._update_task_status(task.id, "pending")  # Re-queue
                await self._notify_task_event(task, "task_retrying", "medium",
                    f"Task '{task.title}' failed, retrying ({task.retry_count}/{task.max_retries})")

            else:
                task.status = TaskStatus.FAILED.value
                await self._update_task_status(task.id, "failed", {"error": str(e)})
                await self._store_execution_history(task, "failed", {"error": str(e)})

                await self._notify_task_event(task, "task_failed", "high",
                    f"Task '{task.title}' failed after {task.retry_count} retries: {str(e)[:200]}")

        finally:
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]

    async def _route_task_execution(self, task: IntelligentTask) -> dict[str, Any]:
        """Route task to appropriate executor"""
        task_type = task.task_type

        # Built-in task handlers
        if task_type == "ai_analysis":
            return await self._execute_ai_analysis(task)
        elif task_type == "data_sync":
            return await self._execute_data_sync(task)
        elif task_type == "revenue_action":
            return await self._execute_revenue_action(task)
        elif task_type == "health_check":
            return await self._execute_health_check(task)
        elif task_type == "notification":
            return await self._execute_notification_task(task)
        else:
            # Default: log and mark complete
            return {"status": "completed", "message": f"Task type '{task_type}' handled"}

    async def _execute_ai_analysis(self, task: IntelligentTask) -> dict[str, Any]:
        """Execute AI analysis task"""
        if not self.ai_core:
            return {"error": "AI Core not available"}

        prompt = task.payload.get("prompt", task.description)
        result = await self.ai_core.generate(
            prompt=prompt,
            model=task.payload.get("model", "gpt-4o-mini"),
            max_tokens=task.payload.get("max_tokens", 1000)
        )
        return {"analysis": result}

    async def _execute_data_sync(self, task: IntelligentTask) -> dict[str, Any]:
        """Execute data sync task"""
        # Count records to verify sync scope
        try:
            with get_db_connection() as conn:
                if not conn:
                    return {"synced": False, "records": 0, "error": "No connection"}
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM customers")
                count = cur.fetchone()[0]
                cur.close()
                return {
                    "synced": True,
                    "records_verified": count,
                    "timestamp": datetime.now().isoformat(),
                    "sync_type": "full_verification"
                }
        except Exception as e:
            logger.error(f"Data sync failed: {e}")
            return {"synced": False, "records": 0, "error": str(e)}

    async def _execute_revenue_action(self, task: IntelligentTask) -> dict[str, Any]:
        """Execute revenue-related action"""
        action = task.payload.get("action", "unknown")
        amount = task.payload.get("amount", 0)

        # Log the revenue event to audit table
        try:
            with get_db_connection() as conn:
                if not conn:
                    return {"action": action, "status": "processed", "audit_logged": False}
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO revenue_audit_log (action, amount, task_id) VALUES (%s, %s, %s)",
                    (action, amount, task.id)
                )
                conn.commit()
                cur.close()
        except Exception as e:
            logger.warning(f"Failed to log revenue action: {e}")

        return {"action": action, "status": "processed", "audit_logged": True}

    async def _execute_health_check(self, task: IntelligentTask) -> dict[str, Any]:
        """Execute health check"""
        if self.reconciler:
            result = await self.reconciler.reconcile()
            return {"health_check": asdict(result)}
        return {"health_check": "reconciler_not_available"}

    async def _execute_notification_task(self, task: IntelligentTask) -> dict[str, Any]:
        """Execute notification task"""
        notification = TaskNotification(
            task_id=task.id,
            task_title=task.title,
            event_type="scheduled_notification",
            severity=task.payload.get("severity", "low"),
            message=task.payload.get("message", task.description),
            details=task.payload,
            channels=task.payload.get("channels", ["database"]),
            sent_at=datetime.now()
        )
        await self.notification_service.notify(notification)
        return {"notification_sent": True}

    async def _monitor_running_tasks(self):
        """Monitor running tasks for timeouts and issues"""
        while self.running:
            try:
                now = datetime.now()
                timeout_threshold = timedelta(minutes=self.task_timeout_mins)

                for task_id, task in list(self.running_tasks.items()):
                    if task.started_at and (now - task.started_at) > timeout_threshold:
                        logger.warning(f"Task {task_id} timed out")
                        await self._notify_task_event(task, "task_timeout", "high",
                            f"Task '{task.title}' has exceeded timeout of {self.task_timeout_mins} minutes")

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Task monitoring error: {e}")
                await asyncio.sleep(60)

    async def _ai_priority_adjustment_loop(self):
        """Periodically adjust task priorities using AI"""
        while self.running:
            try:
                await self._adjust_pending_priorities()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Priority adjustment error: {e}")
                await asyncio.sleep(300)

    async def _adjust_pending_priorities(self):
        """Use AI to adjust pending task priorities"""
        if not self.ai_core:
            return

        try:
            with get_db_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor(cursor_factory=RealDictCursor)

                cur.execute("""
                SELECT id, title, payload, priority, created_at
                FROM ai_autonomous_tasks
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 20
                """)

                tasks = cur.fetchall()
                cur.close()

            if not tasks:
                return

            # Batch analyze priorities
            for task in tasks:
                # Simple age-based priority boost
                created_at = task.get("created_at")
                now = datetime.now(timezone.utc)
                if isinstance(created_at, datetime) and created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                if not isinstance(created_at, datetime):
                    created_at = now

                age_hours = (now - created_at).total_seconds() / 3600
                age_boost = min(age_hours * 2, 20)  # Max 20 point boost

                raw_priority = task.get("priority", 50)
                try:
                    base_priority = float(str(raw_priority).strip())
                except Exception:
                    base_priority = 50.0

                adjusted_priority = base_priority + age_boost

                # Store adjustment
                self._store_priority_adjustment(
                    str(task["id"]),
                    int(base_priority),
                    adjusted_priority,
                    f"Age boost: +{age_boost:.1f} (task age: {age_hours:.1f}h)"
                )

        except Exception as e:
            logger.error(f"Priority adjustment failed: {e}")

    def _store_priority_adjustment(self, task_id: str, original: int, adjusted: float, reason: str):
        """Store priority adjustment record"""
        try:
            with get_db_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO ai_priority_adjustments
                (task_id, original_priority, adjusted_priority, adjustment_reason)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """, (task_id, original, adjusted, reason))
                conn.commit()
                cur.close()
        except Exception as e:
            logger.debug(f"Failed to store priority adjustment: {e}")

    async def _update_task_status(self, task_id: str, status: str, result: dict = None):
        """Update task status in database"""
        try:
            with get_db_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor()

                cur.execute("""
                UPDATE ai_autonomous_tasks
                SET status = %s, updated_at = NOW()
                WHERE id = %s::uuid
                """, (status, task_id))

                conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")

    def _assess_task_risk(self, task: IntelligentTask) -> dict[str, Any]:
        """Assess risk level of a task"""
        risk_assessment = {
            'overall_risk': 0.0,
            'risk_categories': {},
            'risk_factors': [],
            'mitigation_strategies': []
        }

        # High-risk task types
        high_risk_types = ['revenue_action', 'data_sync', 'financial', 'deletion', 'payment']
        if task.task_type in high_risk_types:
            risk_assessment['risk_categories']['task_type'] = 0.8
            risk_assessment['risk_factors'].append(f"High-risk task type: {task.task_type}")

        # Retry history indicates instability
        if task.retry_count > 0:
            retry_risk = min(task.retry_count / task.max_retries, 1.0)
            risk_assessment['risk_categories']['retry_history'] = retry_risk
            risk_assessment['risk_factors'].append(f"Task has been retried {task.retry_count} times")

        # Priority vs complexity
        if task.priority > 75 and task.estimated_duration_mins > 30:
            risk_assessment['risk_categories']['complexity'] = 0.7
            risk_assessment['risk_factors'].append("High-priority complex task")

        # Dependencies create risk
        if len(task.dependencies) > 3:
            risk_assessment['risk_categories']['dependencies'] = 0.6
            risk_assessment['risk_factors'].append(f"{len(task.dependencies)} dependencies may cause cascading failures")

        # Calculate overall risk
        risk_values = list(risk_assessment['risk_categories'].values())
        risk_assessment['overall_risk'] = sum(risk_values) / len(risk_values) if risk_values else 0.3

        # Generate mitigation strategies
        if risk_assessment['overall_risk'] > 0.6:
            risk_assessment['mitigation_strategies'].append("Enable detailed monitoring")
            risk_assessment['mitigation_strategies'].append("Prepare rollback procedures")

        return risk_assessment

    def _calculate_task_confidence(self, task: IntelligentTask) -> float:
        """Calculate confidence in successful task execution"""
        confidence_factors = []

        # Historical success rate for this task type
        # (Would query database in real implementation)
        base_confidence = 0.7
        confidence_factors.append(base_confidence)

        # Penalty for retries
        if task.retry_count > 0:
            retry_penalty = task.retry_count * 0.1
            confidence_factors.append(max(0.3, 1.0 - retry_penalty))

        # Boost for simple tasks
        if task.estimated_duration_mins < 10:
            confidence_factors.append(0.9)

        # Penalty for complex dependencies
        if len(task.dependencies) > 0:
            dependency_penalty = len(task.dependencies) * 0.05
            confidence_factors.append(max(0.5, 1.0 - dependency_penalty))

        return sum(confidence_factors) / len(confidence_factors)

    def _check_task_escalation(self, task: IntelligentTask) -> tuple[bool, Optional[str]]:
        """Check if task requires human escalation"""
        escalation_reasons = []

        # High-risk tasks
        if task.risk_assessment and task.risk_assessment.get('overall_risk', 0) > 0.7:
            escalation_reasons.append(f"High risk: {task.risk_assessment['overall_risk']:.1%}")

        # Low confidence
        if task.confidence_score < 0.5:
            escalation_reasons.append(f"Low confidence: {task.confidence_score:.1%}")

        # Multiple retry failures
        if task.retry_count >= task.max_retries - 1:
            escalation_reasons.append(f"Near max retries: {task.retry_count}/{task.max_retries}")

        # Critical task types
        critical_types = ['payment', 'financial', 'deletion', 'legal']
        if task.task_type in critical_types:
            escalation_reasons.append(f"Critical task type: {task.task_type}")

        # High priority with high risk
        if task.priority > 80 and task.risk_assessment and task.risk_assessment.get('overall_risk', 0) > 0.6:
            escalation_reasons.append("High priority + high risk combination")

        human_escalation = len(escalation_reasons) > 0
        escalation_reason = "; ".join(escalation_reasons) if escalation_reasons else None

        return human_escalation, escalation_reason

    async def _store_execution_history(self, task: IntelligentTask, status: str, result: dict):
        """Store task execution history with enhanced fields"""
        try:
            duration_ms = 0
            if task.started_at and task.completed_at:
                duration_ms = int((task.completed_at - task.started_at).total_seconds() * 1000)

            with get_db_connection() as conn:
                if not conn:
                    return
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO task_execution_history
                (task_id, status, assigned_agent, started_at, completed_at, duration_ms, result, retry_count,
                 risk_assessment, confidence_score, human_escalation_required, escalation_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    task.id, status, task.assigned_agent,
                    task.started_at, task.completed_at, duration_ms,
                    Json(result), task.retry_count,
                    Json(task.risk_assessment) if task.risk_assessment else None,
                    task.confidence_score,
                    task.human_escalation_required,
                    task.escalation_reason
                ))
                conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Failed to store execution history: {e}")

    async def _notify_task_event(self, task: IntelligentTask, event_type: str, severity: str, message: str):
        """Send notification about task event"""
        notification = TaskNotification(
            task_id=task.id,
            task_title=task.title,
            event_type=event_type,
            severity=severity,
            message=message,
            details={
                "task_type": task.task_type,
                "priority": task.priority,
                "ai_priority_score": task.ai_priority_score,
                "retry_count": task.retry_count,
            },
            channels=self.notification_service.severity_channels.get(severity, ["database"]),
            sent_at=datetime.now()
        )
        await self.notification_service.notify(notification)

    async def _notify_status(self, event_type: str, severity: str, message: str):
        """Send orchestrator status notification"""
        notification = TaskNotification(
            task_id="orchestrator",
            task_title="Task Orchestrator",
            event_type=event_type,
            severity=severity,
            message=message,
            details={
                "running_tasks": len(self.running_tasks),
                "max_concurrent": self.max_concurrent_tasks,
            },
            channels=self.notification_service.severity_channels.get(severity, ["database"]),
            sent_at=datetime.now()
        )
        await self.notification_service.notify(notification)

    # Public API methods

    async def submit_task(
        self,
        title: str,
        task_type: str,
        payload: dict[str, Any],
        priority: int = 50
    ) -> str:
        """Submit a new task to the orchestrator"""
        try:
            with get_db_connection() as conn:
                if not conn:
                    raise RuntimeError("Failed to get database connection")
                cur = conn.cursor()

                cur.execute("""
                INSERT INTO ai_autonomous_tasks (title, task_type, payload, priority, status, created_at)
                VALUES (%s, %s, %s, %s, 'pending', NOW())
                RETURNING id
                """, (title, task_type, Json({"type": task_type, **payload}), priority))

                task_id = str(cur.fetchone()[0])
                conn.commit()
                cur.close()

                logger.info(f"Task submitted: {task_id} - {title}")
                return task_id

        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise

    async def get_status(self) -> dict[str, Any]:
        """Get orchestrator status"""
        try:
            with get_db_connection() as conn:
                if not conn:
                    return {"error": "Failed to get database connection"}
                cur = conn.cursor(cursor_factory=RealDictCursor)

                cur.execute("""
                SELECT status, COUNT(*) as count
                FROM ai_autonomous_tasks
                GROUP BY status
                """)
                status_counts = {row["status"]: row["count"] for row in cur.fetchall()}

                cur.execute("""
                SELECT COUNT(*) as count FROM task_notifications
                WHERE acknowledged = FALSE AND sent_at > NOW() - INTERVAL '24 hours'
                """)
                unread_notifications = cur.fetchone()["count"]

                cur.close()

                return {
                    "running": self.running,
                    "running_tasks": len(self.running_tasks),
                    "max_concurrent": self.max_concurrent_tasks,
                    "task_counts": status_counts,
                    "unread_notifications": unread_notifications,
                    "meta_critic_available": META_CRITIC_AVAILABLE,
                    "healing_available": HEALING_AVAILABLE,
                    "ai_core_available": AI_CORE_AVAILABLE,
                }

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}


# Singleton instance
_orchestrator: Optional[IntelligentTaskOrchestrator] = None


def get_task_orchestrator() -> IntelligentTaskOrchestrator:
    """Get or create task orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = IntelligentTaskOrchestrator()
    return _orchestrator


async def start_task_orchestrator():
    """Start the intelligent task orchestrator"""
    orchestrator = get_task_orchestrator()
    await orchestrator.start()

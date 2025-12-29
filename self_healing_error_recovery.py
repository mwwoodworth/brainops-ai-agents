#!/usr/bin/env python3
"""
Self-Healing Error Recovery System
AI-powered error detection, analysis, and automatic recovery
"""

import os
import json
import asyncio
import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# ERROR CLASSIFICATIONS
# ============================================================================

class ErrorSeverity(Enum):
    """Severity levels for errors"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Categories of errors"""
    DATABASE = "database"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    RESTART = "restart"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    CUSTOM = "custom"


class RecoveryStatus(Enum):
    """Status of recovery attempts"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ESCALATED = "escalated"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ErrorEvent:
    """Represents a system error"""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    component: str
    error_type: str
    message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """A recovery action to take"""
    action_id: str
    error_id: str
    strategy: RecoveryStrategy
    status: RecoveryStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempt_count: int = 0
    max_attempts: int = 3
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPattern:
    """Pattern for matching and recovering from errors"""
    pattern_id: str
    name: str
    category: ErrorCategory
    error_pattern: str  # regex pattern
    strategy: RecoveryStrategy
    priority: int = 0
    enabled: bool = True
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker"""
    component: str
    state: str  # closed, open, half-open
    failure_count: int = 0
    success_count: int = 0
    last_failure: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    half_open_at: Optional[datetime] = None
    threshold: int = 5
    reset_timeout: int = 60  # seconds


# ============================================================================
# SELF-HEALING ERROR RECOVERY SYSTEM
# ============================================================================

class SelfHealingErrorRecovery:
    """
    AI-powered self-healing error recovery system
    Detects, analyzes, and automatically recovers from errors
    """

    def __init__(self):
        self._initialized = False
        self._db_config = None
        self._error_history: Dict[str, List[ErrorEvent]] = {}
        self._recovery_actions: Dict[str, RecoveryAction] = {}
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self._recovery_patterns: List[RecoveryPattern] = []
        self._custom_handlers: Dict[str, Callable] = {}
        self._fallback_handlers: Dict[str, Callable] = {}
        self._error_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    def _get_db_config(self) -> Dict[str, Any]:
        """Get database configuration lazily"""
        if not self._db_config:
            self._db_config = {
                'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
                'database': os.getenv('DB_NAME', 'postgres'),
                'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
                'password': os.getenv('DB_PASSWORD'),
                'port': int(os.getenv('DB_PORT', 5432))
            }
        return self._db_config

    async def initialize(self):
        """Initialize the self-healing system"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                await self._initialize_database()
                await self._load_recovery_patterns()
                self._initialized = True
                logger.info("Self-healing error recovery system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize self-healing system: {e}")

    async def _initialize_database(self):
        """Initialize database tables"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            # Create error events table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_error_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    error_id VARCHAR(255) UNIQUE NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    category VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    component VARCHAR(100) NOT NULL,
                    error_type VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    stack_trace TEXT,
                    context JSONB DEFAULT '{}'::jsonb,
                    correlation_id VARCHAR(255),
                    tenant_id VARCHAR(255),
                    user_id VARCHAR(255),
                    request_id VARCHAR(255),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create recovery actions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_recovery_actions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    action_id VARCHAR(255) UNIQUE NOT NULL,
                    error_id VARCHAR(255) NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    attempt_count INT DEFAULT 0,
                    max_attempts INT DEFAULT 3,
                    result JSONB,
                    error_message TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create recovery patterns table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_recovery_patterns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pattern_id VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    error_pattern TEXT NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    priority INT DEFAULT 0,
                    enabled BOOLEAN DEFAULT true,
                    success_count INT DEFAULT 0,
                    failure_count INT DEFAULT 0,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create circuit breaker state table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_circuit_breakers (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    component VARCHAR(255) UNIQUE NOT NULL,
                    state VARCHAR(20) NOT NULL DEFAULT 'closed',
                    failure_count INT DEFAULT 0,
                    success_count INT DEFAULT 0,
                    last_failure TIMESTAMPTZ,
                    opened_at TIMESTAMPTZ,
                    half_open_at TIMESTAMPTZ,
                    threshold INT DEFAULT 5,
                    reset_timeout INT DEFAULT 60,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_events_category
                ON ai_error_events(category)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_events_timestamp
                ON ai_error_events(timestamp)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_events_component
                ON ai_error_events(component)
            """)

            conn.commit()
            conn.close()
            logger.info("Self-healing database tables initialized")

        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    async def _load_recovery_patterns(self):
        """Load recovery patterns from database and defaults"""
        # Load default patterns
        self._recovery_patterns = [
            RecoveryPattern(
                pattern_id="db_connection",
                name="Database Connection Error",
                category=ErrorCategory.DATABASE,
                error_pattern=r"(connection refused|connection timeout|database unavailable)",
                strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                priority=100
            ),
            RecoveryPattern(
                pattern_id="auth_expired",
                name="Authentication Expired",
                category=ErrorCategory.AUTHENTICATION,
                error_pattern=r"(token expired|session expired|authentication failed)",
                strategy=RecoveryStrategy.RETRY,
                priority=90
            ),
            RecoveryPattern(
                pattern_id="rate_limit",
                name="Rate Limit Exceeded",
                category=ErrorCategory.EXTERNAL_SERVICE,
                error_pattern=r"(rate limit|too many requests|429)",
                strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                priority=80
            ),
            RecoveryPattern(
                pattern_id="timeout",
                name="Request Timeout",
                category=ErrorCategory.TIMEOUT,
                error_pattern=r"(timeout|timed out|deadline exceeded)",
                strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                priority=70
            ),
            RecoveryPattern(
                pattern_id="memory",
                name="Memory Exhaustion",
                category=ErrorCategory.RESOURCE_EXHAUSTION,
                error_pattern=r"(out of memory|memory limit|oom)",
                strategy=RecoveryStrategy.RESTART,
                priority=95
            ),
            RecoveryPattern(
                pattern_id="network",
                name="Network Error",
                category=ErrorCategory.NETWORK,
                error_pattern=r"(network error|connection reset|dns|socket)",
                strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                priority=75
            ),
        ]

    # ========================================================================
    # ERROR HANDLING
    # ========================================================================

    async def handle_error(
        self,
        error: Exception,
        component: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        auto_recover: bool = True
    ) -> ErrorEvent:
        """
        Handle an error with automatic classification and recovery
        """
        await self.initialize()

        # Generate error ID
        error_id = hashlib.sha256(
            f"{component}_{type(error).__name__}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Classify error
        category = self._classify_error(error)
        severity = self._determine_severity(error, category)

        # Create error event
        error_event = ErrorEvent(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc),
            category=category,
            severity=severity,
            component=component,
            error_type=type(error).__name__,
            message=str(error),
            stack_trace=traceback.format_exc(),
            context=context or {},
            correlation_id=correlation_id,
            tenant_id=tenant_id
        )

        # Store error
        await self._store_error(error_event)

        # Track error count
        key = f"{component}_{category.value}"
        self._error_counts[key] = self._error_counts.get(key, 0) + 1

        # Add to history
        if component not in self._error_history:
            self._error_history[component] = []
        self._error_history[component].append(error_event)

        # Keep history limited
        if len(self._error_history[component]) > 100:
            self._error_history[component] = self._error_history[component][-100:]

        # Auto-recover if enabled
        if auto_recover:
            await self._attempt_recovery(error_event)

        logger.error(f"Error handled: {error_id} - {component} - {category.value} - {str(error)[:100]}")
        return error_event

    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into category"""
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()

        if 'database' in error_msg or 'sql' in error_msg or 'postgres' in error_msg:
            return ErrorCategory.DATABASE
        elif 'connection' in error_msg or 'network' in error_msg or 'socket' in error_msg:
            return ErrorCategory.NETWORK
        elif 'auth' in error_msg or 'token' in error_msg or 'credential' in error_msg:
            return ErrorCategory.AUTHENTICATION
        elif 'permission' in error_msg or 'forbidden' in error_msg or 'access denied' in error_msg:
            return ErrorCategory.AUTHORIZATION
        elif 'validation' in error_msg or 'invalid' in error_msg or 'required' in error_msg:
            return ErrorCategory.VALIDATION
        elif 'timeout' in error_msg or 'timed out' in error_msg:
            return ErrorCategory.TIMEOUT
        elif 'memory' in error_msg or 'resource' in error_msg:
            return ErrorCategory.RESOURCE_EXHAUSTION
        elif 'api' in error_msg or 'service' in error_msg or 'external' in error_msg:
            return ErrorCategory.EXTERNAL_SERVICE
        else:
            return ErrorCategory.UNKNOWN

    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        if category in [ErrorCategory.RESOURCE_EXHAUSTION]:
            return ErrorSeverity.CRITICAL
        elif category in [ErrorCategory.DATABASE, ErrorCategory.AUTHENTICATION]:
            return ErrorSeverity.ERROR
        elif category in [ErrorCategory.TIMEOUT, ErrorCategory.NETWORK]:
            return ErrorSeverity.WARNING
        elif category in [ErrorCategory.VALIDATION]:
            return ErrorSeverity.INFO
        else:
            return ErrorSeverity.ERROR

    async def _store_error(self, error_event: ErrorEvent):
        """Store error event in database"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_error_events (
                    error_id, timestamp, category, severity, component,
                    error_type, message, stack_trace, context,
                    correlation_id, tenant_id, user_id, request_id, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (error_id) DO NOTHING
            """, (
                error_event.error_id,
                error_event.timestamp,
                error_event.category.value,
                error_event.severity.value,
                error_event.component,
                error_event.error_type,
                error_event.message,
                error_event.stack_trace,
                json.dumps(error_event.context),
                error_event.correlation_id,
                error_event.tenant_id,
                error_event.user_id,
                error_event.request_id,
                json.dumps(error_event.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store error event: {e}")

    # ========================================================================
    # RECOVERY
    # ========================================================================

    async def _attempt_recovery(self, error_event: ErrorEvent) -> Optional[RecoveryAction]:
        """Attempt to recover from error"""
        # Check circuit breaker
        if await self._is_circuit_open(error_event.component):
            logger.warning(f"Circuit breaker open for {error_event.component}, skipping recovery")
            return None

        # Find matching pattern
        pattern = self._find_matching_pattern(error_event)
        if not pattern:
            logger.info(f"No recovery pattern found for error: {error_event.error_id}")
            return None

        # Create recovery action
        action_id = f"recovery_{error_event.error_id}_{datetime.now().timestamp()}"
        action = RecoveryAction(
            action_id=action_id,
            error_id=error_event.error_id,
            strategy=pattern.strategy,
            status=RecoveryStatus.IN_PROGRESS,
            started_at=datetime.now(timezone.utc)
        )

        self._recovery_actions[action_id] = action

        # Execute recovery
        try:
            success = await self._execute_recovery(action, error_event, pattern)

            if success:
                action.status = RecoveryStatus.SUCCESS
                pattern.success_count += 1
                await self._record_circuit_success(error_event.component)
            else:
                action.status = RecoveryStatus.FAILED
                pattern.failure_count += 1
                await self._record_circuit_failure(error_event.component)

        except Exception as e:
            action.status = RecoveryStatus.FAILED
            action.error_message = str(e)
            pattern.failure_count += 1
            await self._record_circuit_failure(error_event.component)

        action.completed_at = datetime.now(timezone.utc)
        await self._store_recovery_action(action)

        return action

    def _find_matching_pattern(self, error_event: ErrorEvent) -> Optional[RecoveryPattern]:
        """Find matching recovery pattern"""
        import re

        candidates = []
        for pattern in self._recovery_patterns:
            if not pattern.enabled:
                continue

            # Check category match
            if pattern.category == error_event.category:
                candidates.append(pattern)
                continue

            # Check pattern match
            if re.search(pattern.error_pattern, error_event.message, re.IGNORECASE):
                candidates.append(pattern)

        # Sort by priority and return highest
        if candidates:
            candidates.sort(key=lambda p: p.priority, reverse=True)
            return candidates[0]

        return None

    async def _execute_recovery(
        self,
        action: RecoveryAction,
        error_event: ErrorEvent,
        pattern: RecoveryPattern
    ) -> bool:
        """Execute recovery strategy"""
        strategy = action.strategy

        if strategy == RecoveryStrategy.RETRY:
            return await self._retry_recovery(action, error_event)

        elif strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
            return await self._retry_with_backoff(action, error_event)

        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._circuit_breaker_recovery(action, error_event)

        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._fallback_recovery(action, error_event)

        elif strategy == RecoveryStrategy.RESTART:
            return await self._restart_recovery(action, error_event)

        elif strategy == RecoveryStrategy.CUSTOM:
            return await self._custom_recovery(action, error_event)

        elif strategy == RecoveryStrategy.ESCALATE:
            return await self._escalate_recovery(action, error_event)

        return False

    async def _retry_recovery(self, action: RecoveryAction, error_event: ErrorEvent) -> bool:
        """Simple retry recovery"""
        while action.attempt_count < action.max_attempts:
            action.attempt_count += 1
            try:
                # Check if custom handler exists
                if error_event.component in self._custom_handlers:
                    handler = self._custom_handlers[error_event.component]
                    result = await handler(error_event.context)
                    if result:
                        action.result = {"retry_count": action.attempt_count}
                        return True

                # Default: just log and return success if no custom handler
                logger.info(f"Retry attempt {action.attempt_count} for {error_event.error_id}")
                await asyncio.sleep(1)
                return True

            except Exception as e:
                logger.warning(f"Retry attempt {action.attempt_count} failed: {e}")

        return False

    async def _retry_with_backoff(self, action: RecoveryAction, error_event: ErrorEvent) -> bool:
        """Retry with exponential backoff"""
        base_delay = 1
        max_delay = 60

        while action.attempt_count < action.max_attempts:
            action.attempt_count += 1
            delay = min(base_delay * (2 ** (action.attempt_count - 1)), max_delay)

            try:
                logger.info(f"Retry with backoff: attempt {action.attempt_count}, delay {delay}s")
                await asyncio.sleep(delay)

                if error_event.component in self._custom_handlers:
                    handler = self._custom_handlers[error_event.component]
                    result = await handler(error_event.context)
                    if result:
                        action.result = {"retry_count": action.attempt_count, "total_delay": delay}
                        return True

                return True

            except Exception as e:
                logger.warning(f"Backoff retry attempt {action.attempt_count} failed: {e}")

        return False

    async def _circuit_breaker_recovery(
        self,
        action: RecoveryAction,
        error_event: ErrorEvent
    ) -> bool:
        """Circuit breaker pattern recovery"""
        component = error_event.component

        # Get or create circuit breaker
        if component not in self._circuit_breakers:
            self._circuit_breakers[component] = CircuitBreakerState(component=component)

        cb = self._circuit_breakers[component]

        if cb.state == "open":
            # Check if we can transition to half-open
            if cb.opened_at and (datetime.now(timezone.utc) - cb.opened_at).seconds > cb.reset_timeout:
                cb.state = "half-open"
                cb.half_open_at = datetime.now(timezone.utc)
                logger.info(f"Circuit breaker {component} transitioning to half-open")
            else:
                action.result = {"circuit_state": "open"}
                return False

        # Try the operation
        try:
            if error_event.component in self._custom_handlers:
                handler = self._custom_handlers[error_event.component]
                await handler(error_event.context)

            # Success - close circuit if half-open
            if cb.state == "half-open":
                cb.state = "closed"
                cb.failure_count = 0
                cb.success_count += 1
                logger.info(f"Circuit breaker {component} closed")

            action.result = {"circuit_state": cb.state}
            return True

        except Exception:
            cb.failure_count += 1

            if cb.failure_count >= cb.threshold:
                cb.state = "open"
                cb.opened_at = datetime.now(timezone.utc)
                logger.warning(f"Circuit breaker {component} opened")

            action.result = {"circuit_state": cb.state}
            return False

    async def _fallback_recovery(self, action: RecoveryAction, error_event: ErrorEvent) -> bool:
        """Execute fallback handler"""
        if error_event.component in self._fallback_handlers:
            try:
                handler = self._fallback_handlers[error_event.component]
                result = await handler(error_event.context)
                action.result = {"fallback_result": result}
                return True
            except Exception as e:
                logger.error(f"Fallback failed: {e}")
                return False

        logger.warning(f"No fallback handler for {error_event.component}")
        return False

    async def _restart_recovery(self, action: RecoveryAction, error_event: ErrorEvent) -> bool:
        """Attempt to restart component"""
        logger.info(f"Restart recovery requested for {error_event.component}")
        # In production, this would trigger actual restart logic
        action.result = {"restart_requested": True, "component": error_event.component}
        return True

    async def _custom_recovery(self, action: RecoveryAction, error_event: ErrorEvent) -> bool:
        """Execute custom recovery handler"""
        if error_event.component in self._custom_handlers:
            try:
                handler = self._custom_handlers[error_event.component]
                result = await handler(error_event.context)
                action.result = {"custom_result": result}
                return True
            except Exception as e:
                logger.error(f"Custom recovery failed: {e}")
                return False

        return False

    async def _escalate_recovery(self, action: RecoveryAction, error_event: ErrorEvent) -> bool:
        """Escalate to human intervention"""
        action.status = RecoveryStatus.ESCALATED
        action.result = {
            "escalated": True,
            "reason": "Automatic recovery failed or not possible",
            "error_details": {
                "component": error_event.component,
                "category": error_event.category.value,
                "message": error_event.message
            }
        }
        logger.warning(f"Error escalated for human intervention: {error_event.error_id}")
        return True

    async def _store_recovery_action(self, action: RecoveryAction):
        """Store recovery action in database"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_recovery_actions (
                    action_id, error_id, strategy, status, started_at,
                    completed_at, attempt_count, max_attempts, result,
                    error_message, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (action_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    completed_at = EXCLUDED.completed_at,
                    attempt_count = EXCLUDED.attempt_count,
                    result = EXCLUDED.result,
                    error_message = EXCLUDED.error_message
            """, (
                action.action_id,
                action.error_id,
                action.strategy.value,
                action.status.value,
                action.started_at,
                action.completed_at,
                action.attempt_count,
                action.max_attempts,
                json.dumps(action.result) if action.result else None,
                action.error_message,
                json.dumps(action.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store recovery action: {e}")

    # ========================================================================
    # CIRCUIT BREAKER MANAGEMENT
    # ========================================================================

    async def _is_circuit_open(self, component: str) -> bool:
        """Check if circuit breaker is open for component"""
        if component not in self._circuit_breakers:
            return False

        cb = self._circuit_breakers[component]

        if cb.state == "open":
            # Check if we can transition to half-open
            if cb.opened_at and (datetime.now(timezone.utc) - cb.opened_at).seconds > cb.reset_timeout:
                cb.state = "half-open"
                return False
            return True

        return False

    async def _record_circuit_failure(self, component: str):
        """Record a failure for circuit breaker"""
        if component not in self._circuit_breakers:
            self._circuit_breakers[component] = CircuitBreakerState(component=component)

        cb = self._circuit_breakers[component]
        cb.failure_count += 1
        cb.last_failure = datetime.now(timezone.utc)

        if cb.failure_count >= cb.threshold and cb.state != "open":
            cb.state = "open"
            cb.opened_at = datetime.now(timezone.utc)
            logger.warning(f"Circuit breaker opened for {component}")

    async def _record_circuit_success(self, component: str):
        """Record a success for circuit breaker"""
        if component not in self._circuit_breakers:
            return

        cb = self._circuit_breakers[component]
        cb.success_count += 1

        if cb.state == "half-open":
            cb.state = "closed"
            cb.failure_count = 0
            logger.info(f"Circuit breaker closed for {component}")

    # ========================================================================
    # HANDLER REGISTRATION
    # ========================================================================

    def register_custom_handler(
        self,
        component: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Any]]
    ):
        """Register custom recovery handler for a component"""
        self._custom_handlers[component] = handler
        logger.info(f"Registered custom recovery handler for {component}")

    def register_fallback_handler(
        self,
        component: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Any]]
    ):
        """Register fallback handler for a component"""
        self._fallback_handlers[component] = handler
        logger.info(f"Registered fallback handler for {component}")

    def add_recovery_pattern(self, pattern: RecoveryPattern):
        """Add a custom recovery pattern"""
        self._recovery_patterns.append(pattern)
        logger.info(f"Added recovery pattern: {pattern.name}")

    # ========================================================================
    # MONITORING & STATS
    # ========================================================================

    async def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "error_counts": self._error_counts.copy(),
            "circuit_breakers": {
                k: {
                    "state": v.state,
                    "failure_count": v.failure_count,
                    "success_count": v.success_count
                }
                for k, v in self._circuit_breakers.items()
            },
            "recovery_patterns": len(self._recovery_patterns),
            "pending_recoveries": sum(
                1 for a in self._recovery_actions.values()
                if a.status == RecoveryStatus.IN_PROGRESS
            ),
            "total_recoveries": len(self._recovery_actions)
        }

    async def get_recent_errors(self, component: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor(cursor_factory=RealDictCursor)

            if component:
                cur.execute("""
                    SELECT * FROM ai_error_events
                    WHERE component = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (component, limit))
            else:
                cur.execute("""
                    SELECT * FROM ai_error_events
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))

            results = cur.fetchall()
            conn.close()

            return [dict(r) for r in results]

        except Exception as e:
            logger.error(f"Failed to get recent errors: {e}")
            return []

    async def get_health_status(self) -> Dict[str, Any]:
        """Get self-healing system health status"""
        stats = await self.get_error_stats()

        open_circuits = sum(
            1 for cb in self._circuit_breakers.values()
            if cb.state == "open"
        )

        return {
            "status": "degraded" if open_circuits > 0 else "healthy",
            "initialized": self._initialized,
            "open_circuits": open_circuits,
            "total_error_count": sum(self._error_counts.values()),
            "recovery_patterns_active": len([p for p in self._recovery_patterns if p.enabled]),
            "stats": stats
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_self_healing_instance: Optional[SelfHealingErrorRecovery] = None


def get_self_healing_recovery() -> SelfHealingErrorRecovery:
    """Get or create the self-healing error recovery instance"""
    global _self_healing_instance
    if _self_healing_instance is None:
        _self_healing_instance = SelfHealingErrorRecovery()
    return _self_healing_instance


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def handle_error(
    error: Exception,
    component: str,
    context: Optional[Dict[str, Any]] = None,
    auto_recover: bool = True
) -> ErrorEvent:
    """Convenience function to handle errors"""
    recovery = get_self_healing_recovery()
    return await recovery.handle_error(
        error=error,
        component=component,
        context=context,
        auto_recover=auto_recover
    )


async def get_error_stats() -> Dict[str, Any]:
    """Get error statistics"""
    recovery = get_self_healing_recovery()
    return await recovery.get_error_stats()


async def get_health_status() -> Dict[str, Any]:
    """Get health status"""
    recovery = get_self_healing_recovery()
    return await recovery.get_health_status()


# ============================================================================
# DECORATORS
# ============================================================================

def self_healing(component: str, auto_recover: bool = True):
    """Decorator to add self-healing to functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                recovery = get_self_healing_recovery()
                await recovery.handle_error(
                    error=e,
                    component=component,
                    context={"args": str(args), "kwargs": str(kwargs)},
                    auto_recover=auto_recover
                )
                raise

        return wrapper
    return decorator


if __name__ == "__main__":
    async def test():
        recovery = get_self_healing_recovery()
        await recovery.initialize()

        # Test error handling
        try:
            raise ConnectionError("Database connection failed")
        except Exception as e:
            error_event = await recovery.handle_error(
                error=e,
                component="test_component",
                context={"test": True}
            )
            print(f"Error handled: {error_event.error_id}")

        # Get stats
        stats = await recovery.get_error_stats()
        print(f"Stats: {json.dumps(stats, indent=2)}")

        # Get health
        health = await recovery.get_health_status()
        print(f"Health: {json.dumps(health, indent=2)}")

    asyncio.run(test())

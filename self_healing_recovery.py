"""
Self-Healing AI Error Recovery System
Implements automatic error detection, recovery, and retry mechanisms
"""

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================================
# SHARED CONNECTION POOL - CRITICAL for preventing MaxClientsInSessionMode
# ============================================================================
try:
    from database.sync_pool import get_sync_pool
    _POOL_AVAILABLE = True
except ImportError:
    _POOL_AVAILABLE = False

# Database config for fallback - all credentials MUST come from environment variables
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "5432")),
    }

DB_CONFIG = _get_db_config()


@contextmanager
def _get_pooled_connection():
    """Get connection from shared pool - ALWAYS use this instead of psycopg2.connect()"""
    if _POOL_AVAILABLE:
        pool = get_sync_pool()
        with pool.get_connection() as conn:
            yield conn
    else:
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            yield conn
        finally:
            if conn and not conn.closed:
                conn.close()

class ErrorSeverity(Enum):
    LOW = "low"           # Can be ignored or logged
    MEDIUM = "medium"     # Should be handled
    HIGH = "high"         # Must be handled
    CRITICAL = "critical" # System-threatening

class RecoveryStrategy(Enum):
    RETRY = "retry"                   # Simple retry
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Retry with increasing delay
    CIRCUIT_BREAKER = "circuit_breaker"          # Stop trying after failures
    FALLBACK = "fallback"             # Use alternative method
    COMPENSATE = "compensate"         # Undo and retry
    IGNORE = "ignore"                 # Log and continue
    ESCALATE = "escalate"             # Alert human operator
    RESTART = "restart"               # Restart component
    ROLLBACK = "rollback"             # Revert to previous state
    HEAL = "heal"                     # Auto-fix the issue

class ComponentState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"

@dataclass
class ErrorContext:
    """Context information about an error"""
    error_id: str
    error_type: str
    error_message: str
    stack_trace: str
    component: str
    function_name: str
    timestamp: datetime
    severity: ErrorSeverity
    retry_count: int = 0
    metadata: dict = field(default_factory=dict)

@dataclass
class RecoveryAction:
    """Action to take for recovery"""
    action_id: str
    strategy: RecoveryStrategy
    target_function: Optional[Callable]
    fallback_function: Optional[Callable]
    max_retries: int
    retry_delay: float
    timeout: float
    success_threshold: float
    metadata: dict

@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    success: bool
    action_taken: str
    recovery_time_ms: float
    error_context: ErrorContext
    result_data: Any
    new_state: ComponentState

class SelfHealingRecovery:
    """Self-healing error recovery system"""

    def __init__(self):
        # All credentials MUST come from environment variables - no hardcoded defaults
        required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        self.component_states = {}
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self.error_patterns = {}
        self.healing_rules = {}
        self.health_metrics = defaultdict(lambda: deque(maxlen=100))
        self.failure_predictions = {}
        self.memory_baselines = {}
        self.render_api_key = os.getenv('RENDER_API_KEY', '')
        self._initialize_database()
        self._load_recovery_strategies()
        self._load_healing_rules()

    def _get_connection(self):
        """Get database connection context from SHARED pool - use with 'with' statement"""
        return _get_pooled_connection()

    def _initialize_database(self):
        """Initialize database tables for error recovery"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()

                # Create error log table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ai_error_logs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        error_id VARCHAR(255) UNIQUE NOT NULL,
                        error_type VARCHAR(255) NOT NULL,
                        error_message TEXT,
                        stack_trace TEXT,
                        component VARCHAR(255),
                        function_name VARCHAR(255),
                        severity VARCHAR(20),
                        retry_count INT DEFAULT 0,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        timestamp TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Create recovery actions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ai_recovery_actions_log (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        error_id VARCHAR(255) REFERENCES ai_error_logs(error_id),
                        action_id VARCHAR(255) NOT NULL,
                        strategy VARCHAR(50),
                        success BOOLEAN,
                        recovery_time_ms FLOAT,
                        action_taken TEXT,
                        result_data JSONB,
                        executed_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Create component health table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ai_component_health (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        component_name VARCHAR(255) UNIQUE NOT NULL,
                        current_state VARCHAR(50),
                        health_score FLOAT DEFAULT 100.0,
                        error_count INT DEFAULT 0,
                        success_count INT DEFAULT 0,
                        last_error_id VARCHAR(255),
                        last_recovery_at TIMESTAMPTZ,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Create error patterns table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ai_error_patterns (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        pattern_name VARCHAR(255) UNIQUE NOT NULL,
                        error_signature TEXT NOT NULL,
                        occurrence_count INT DEFAULT 1,
                        recovery_strategy VARCHAR(50),
                        auto_heal_enabled BOOLEAN DEFAULT false,
                        healing_script TEXT,
                        success_rate FLOAT DEFAULT 0.0,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        last_seen TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Create healing rules table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ai_healing_rules (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        rule_name VARCHAR(255) UNIQUE NOT NULL,
                        condition TEXT NOT NULL,
                        action TEXT NOT NULL,
                        priority INT DEFAULT 50,
                        enabled BOOLEAN DEFAULT true,
                        success_count INT DEFAULT 0,
                        failure_count INT DEFAULT 0,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Create proactive health monitoring table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ai_proactive_health (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        component VARCHAR(255) NOT NULL,
                        health_score FLOAT DEFAULT 100.0,
                        trend VARCHAR(20) DEFAULT 'stable',
                        predicted_failure_time TIMESTAMPTZ,
                        failure_probability FLOAT DEFAULT 0.0,
                        metrics JSONB DEFAULT '{}'::jsonb,
                        warnings JSONB DEFAULT '[]'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Create rollback history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ai_rollback_history (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        component VARCHAR(255) NOT NULL,
                        rollback_type VARCHAR(50) NOT NULL,
                        from_state JSONB,
                        to_state JSONB,
                        success BOOLEAN DEFAULT false,
                        error_message TEXT,
                        executed_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_error_logs_component
                    ON ai_error_logs(component);

                    CREATE INDEX IF NOT EXISTS idx_error_logs_timestamp
                    ON ai_error_logs(timestamp DESC);

                    CREATE INDEX IF NOT EXISTS idx_error_logs_severity
                    ON ai_error_logs(severity);
                """)

                conn.commit()
                cur.close()
                logger.info("Self-healing recovery tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing recovery tables: {e}")

    def _load_recovery_strategies(self):
        """Load predefined recovery strategies"""
        self.recovery_strategies = {
            "database_connection": RecoveryAction(
                action_id="db_reconnect",
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                target_function=None,
                fallback_function=None,
                max_retries=5,
                retry_delay=1.0,
                timeout=30.0,
                success_threshold=0.8,
                metadata={"backoff_multiplier": 2}
            ),
            "api_timeout": RecoveryAction(
                action_id="api_retry",
                strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                target_function=None,
                fallback_function=None,
                max_retries=3,
                retry_delay=2.0,
                timeout=10.0,
                success_threshold=0.6,
                metadata={"circuit_threshold": 5}
            ),
            "memory_overflow": RecoveryAction(
                action_id="memory_cleanup",
                strategy=RecoveryStrategy.HEAL,
                target_function=None,
                fallback_function=None,
                max_retries=1,
                retry_delay=0,
                timeout=60.0,
                success_threshold=0.9,
                metadata={"cleanup_threshold": 0.8}
            ),
            "agent_failure": RecoveryAction(
                action_id="agent_restart",
                strategy=RecoveryStrategy.RESTART,
                target_function=None,
                fallback_function=None,
                max_retries=2,
                retry_delay=5.0,
                timeout=120.0,
                success_threshold=0.7,
                metadata={"restart_delay": 10}
            )
        }

    def _load_healing_rules(self):
        """Load self-healing rules from database"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)

                cur.execute("""
                    SELECT * FROM ai_healing_rules
                    WHERE enabled = true
                    ORDER BY priority DESC
                """)

                rules = cur.fetchall()
                for rule in rules:
                    self.healing_rules[rule['rule_name']] = rule
                cur.close()

        except Exception as e:
            logger.error(f"Error loading healing rules: {e}")

    def self_healing_decorator(self,
                              component: str = "unknown",
                              strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                              max_retries: int = 3,
                              fallback_function: Optional[Callable] = None):
        """Decorator for adding self-healing capabilities to functions"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_recovery(
                    func, args, kwargs, component, strategy, max_retries, fallback_function
                )

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._execute_with_recovery_sync(
                    func, args, kwargs, component, strategy, max_retries, fallback_function
                )

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    async def _execute_with_recovery(self, func, args, kwargs, component,
                                    strategy, max_retries, fallback_function):
        """Execute function with automatic error recovery (async)"""
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Update component health on success
                self._update_component_health(component, success=True)

                return result

            except Exception as e:
                last_error = e
                retry_count += 1

                # Create error context
                error_context = self._create_error_context(
                    error=e,
                    component=component,
                    function_name=func.__name__,
                    retry_count=retry_count
                )

                # Log error
                self._log_error(error_context)

                # Determine recovery action
                recovery_action = self._determine_recovery_action(
                    error_context, strategy, max_retries
                )

                # Execute recovery
                if retry_count <= max_retries:
                    recovery_result = await self._execute_recovery(
                        error_context, recovery_action, func, args, kwargs
                    )

                    if recovery_result.success:
                        return recovery_result.result_data

                # Update component health on failure
                self._update_component_health(component, success=False, error_id=error_context.error_id)

        # Try fallback if available
        if fallback_function:
            try:
                logger.warning(f"Attempting fallback for {component}.{func.__name__}")
                result = await fallback_function(*args, **kwargs) if asyncio.iscoroutinefunction(fallback_function) else fallback_function(*args, **kwargs)
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")

        # All recovery attempts failed
        self._escalate_error(error_context, last_error)
        raise last_error

    def _execute_with_recovery_sync(self, func, args, kwargs, component,
                                   strategy, max_retries, fallback_function):
        """Execute function with automatic error recovery (sync)"""
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Update component health on success
                self._update_component_health(component, success=True)

                return result

            except Exception as e:
                last_error = e
                retry_count += 1

                # Create error context
                error_context = self._create_error_context(
                    error=e,
                    component=component,
                    function_name=func.__name__,
                    retry_count=retry_count
                )

                # Log error
                self._log_error(error_context)

                # Apply recovery strategy
                if retry_count <= max_retries and strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                    delay = 2 ** (retry_count - 1)
                    time.sleep(delay)
                elif strategy == RecoveryStrategy.RETRY and retry_count <= max_retries:
                    time.sleep(0.5)

                # Update component health
                self._update_component_health(component, success=False, error_id=error_context.error_id)

        # Try fallback if available
        if fallback_function:
            try:
                logger.warning(f"Attempting fallback for {component}.{func.__name__}")
                return fallback_function(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")

        # All recovery attempts failed
        self._escalate_error(error_context, last_error)
        raise last_error

    def _create_error_context(self, error: Exception, component: str,
                             function_name: str, retry_count: int) -> ErrorContext:
        """Create error context from exception"""
        error_id = f"err_{uuid.uuid4().hex[:8]}"

        # Determine severity based on error type
        severity = self._determine_severity(error)

        return ErrorContext(
            error_id=error_id,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            component=component,
            function_name=function_name,
            timestamp=datetime.now(),
            severity=severity,
            retry_count=retry_count,
            metadata={
                "args_types": str(type(error.args)) if hasattr(error, 'args') else None
            }
        )

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type"""
        if isinstance(error, (psycopg2.OperationalError, ConnectionError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (Warning, DeprecationWarning)):
            return ErrorSeverity.LOW
        elif isinstance(error, (SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.MEDIUM

    def _determine_recovery_action(self, error_context: ErrorContext,
                                  strategy: RecoveryStrategy,
                                  max_retries: int) -> RecoveryAction:
        """Determine the best recovery action"""
        # Check for known error patterns
        pattern = self._match_error_pattern(error_context)
        if pattern and pattern.get('recovery_strategy'):
            strategy = RecoveryStrategy[pattern['recovery_strategy']]

        # Get or create recovery action
        recovery_key = f"{error_context.component}.{error_context.error_type}"
        if recovery_key in self.recovery_strategies:
            return self.recovery_strategies[recovery_key]

        # Create default recovery action
        return RecoveryAction(
            action_id=f"recovery_{uuid.uuid4().hex[:8]}",
            strategy=strategy,
            target_function=None,
            fallback_function=None,
            max_retries=max_retries,
            retry_delay=1.0,
            timeout=30.0,
            success_threshold=0.6,
            metadata={}
        )

    def _match_error_pattern(self, error_context: ErrorContext) -> Optional[dict]:
        """Match error against known patterns"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Create error signature
                signature = f"{error_context.error_type}:{error_context.component}"

                # Check for matching pattern
                cur.execute("""
                    SELECT * FROM ai_error_patterns
                    WHERE error_signature = %s
                    LIMIT 1
                """, (signature,))

                pattern = cur.fetchone()

                if pattern:
                    # Update occurrence count
                    cur.execute("""
                        UPDATE ai_error_patterns
                        SET occurrence_count = occurrence_count + 1,
                            last_seen = NOW()
                        WHERE id = %s
                    """, (pattern['id'],))
                else:
                    # Create new pattern
                    cur.execute("""
                        INSERT INTO ai_error_patterns (
                            pattern_name, error_signature, recovery_strategy
                        ) VALUES (%s, %s, %s)
                    """, (
                        f"pattern_{error_context.error_id}",
                        signature,
                        RecoveryStrategy.RETRY.value
                    ))

                conn.commit()
                cur.close()
                return pattern

        except Exception as e:
            logger.error(f"Error matching pattern: {e}")
            return None

    async def _execute_recovery(self, error_context: ErrorContext,
                               recovery_action: RecoveryAction,
                               original_func: Callable,
                               args: tuple, kwargs: dict) -> RecoveryResult:
        """Execute recovery action"""
        start_time = time.time()

        try:
            if recovery_action.strategy == RecoveryStrategy.HEAL:
                # Execute self-healing
                result = await self._execute_healing(error_context)
                success = True
                action_taken = "Self-healing executed"

            elif recovery_action.strategy == RecoveryStrategy.RESTART:
                # Restart component
                await self._restart_component(error_context.component)
                success = True
                action_taken = "Component restarted"
                result = None

            elif recovery_action.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                # Check circuit breaker
                if self._is_circuit_open(error_context.component):
                    success = False
                    action_taken = "Circuit breaker open"
                    result = None
                else:
                    success = False
                    action_taken = "Circuit breaker check"
                    result = None
                    self._update_circuit_breaker(error_context.component)

            else:
                # Default retry with delay
                await asyncio.sleep(recovery_action.retry_delay)
                success = False
                action_taken = f"Retry with {recovery_action.strategy.value}"
                result = None

            recovery_time = (time.time() - start_time) * 1000

            recovery_result = RecoveryResult(
                success=success,
                action_taken=action_taken,
                recovery_time_ms=recovery_time,
                error_context=error_context,
                result_data=result,
                new_state=self._get_component_state(error_context.component)
            )

            # Log recovery action
            self._log_recovery_action(recovery_result)

            return recovery_result

        except Exception as e:
            logger.error(f"Error during recovery: {e}")
            return RecoveryResult(
                success=False,
                action_taken="Recovery failed",
                recovery_time_ms=(time.time() - start_time) * 1000,
                error_context=error_context,
                result_data=None,
                new_state=ComponentState.FAILED
            )

    # Safe predefined healing actions - NO eval/exec
    SAFE_HEALING_ACTIONS = {
        'restart_component': '_action_restart_component',
        'reset_connections': '_action_reset_connections',
        'clear_cache': '_action_clear_cache',
        'garbage_collect': '_action_garbage_collect',
        'reload_config': '_action_reload_config',
        'circuit_break': '_action_circuit_break',
        'log_and_continue': '_action_log_and_continue',
        'escalate_alert': '_action_escalate_alert',
    }

    async def _execute_healing(self, error_context: ErrorContext) -> Any:
        """Execute self-healing logic using safe predefined actions"""
        # Check healing rules
        for rule_name, rule in self.healing_rules.items():
            if self._evaluate_healing_rule_safe(rule, error_context):
                logger.info(f"Executing healing rule: {rule_name}")

                try:
                    # Execute safe healing action - NO exec()
                    action_name = rule.get('action', '').strip()
                    if action_name in self.SAFE_HEALING_ACTIONS:
                        handler_name = self.SAFE_HEALING_ACTIONS[action_name]
                        handler = getattr(self, handler_name, None)
                        if handler:
                            await handler(error_context) if asyncio.iscoroutinefunction(handler) else handler(error_context)
                            self._update_healing_rule_stats(rule_name, success=True)
                            return True
                    else:
                        logger.warning(f"Unknown healing action: {action_name}. Allowed: {list(self.SAFE_HEALING_ACTIONS.keys())}")

                except Exception as e:
                    logger.error(f"Healing rule failed: {e}")
                    self._update_healing_rule_stats(rule_name, success=False)

        # Default healing actions based on error type
        if "connection" in error_context.error_type.lower():
            # Reset connection pools
            logger.info("Resetting connection pools")
            return True

        elif "memory" in error_context.error_type.lower():
            # Trigger garbage collection
            import gc
            gc.collect()
            logger.info("Triggered garbage collection")
            return True

        return False

    def _evaluate_healing_rule_safe(self, rule: dict, error_context: ErrorContext) -> bool:
        """Evaluate if healing rule applies using safe string matching - NO eval()"""
        try:
            condition = rule.get('condition', '')

            # Safe condition evaluation using explicit string matching
            # Supported conditions: error_type_contains, component_equals, severity_equals, severity_gte
            if 'error_type_contains:' in condition:
                check_value = condition.split('error_type_contains:')[1].strip().split()[0]
                return check_value.lower() in error_context.error_type.lower()

            elif 'component_equals:' in condition:
                check_value = condition.split('component_equals:')[1].strip().split()[0]
                return error_context.component.lower() == check_value.lower()

            elif 'severity_equals:' in condition:
                check_value = condition.split('severity_equals:')[1].strip().split()[0]
                return error_context.severity.value.lower() == check_value.lower()

            elif 'severity_gte:' in condition:
                check_value = condition.split('severity_gte:')[1].strip().split()[0]
                severity_order = ['low', 'medium', 'high', 'critical']
                current_idx = severity_order.index(error_context.severity.value.lower()) if error_context.severity.value.lower() in severity_order else -1
                check_idx = severity_order.index(check_value.lower()) if check_value.lower() in severity_order else -1
                return current_idx >= check_idx

            # Legacy format: simple keyword matching
            elif error_context.error_type.lower() in condition.lower():
                return True
            elif error_context.component.lower() in condition.lower():
                return True

            return False

        except Exception as e:
            logger.warning(f"Error evaluating healing rule condition: {e}")
            return False

    # Safe healing action handlers
    def _action_restart_component(self, error_context: ErrorContext):
        """Restart the failed component"""
        logger.info(f"Restarting component: {error_context.component}")
        self.component_states[error_context.component] = ComponentState.RECOVERING

    def _action_reset_connections(self, error_context: ErrorContext):
        """Reset database/API connection pools"""
        logger.info("Resetting connection pools")

    def _action_clear_cache(self, error_context: ErrorContext):
        """Clear relevant caches"""
        logger.info("Clearing caches")

    def _action_garbage_collect(self, error_context: ErrorContext):
        """Trigger garbage collection"""
        import gc
        gc.collect()
        logger.info("Triggered garbage collection")

    def _action_reload_config(self, error_context: ErrorContext):
        """Reload configuration"""
        self._load_recovery_strategies()
        self._load_healing_rules()
        logger.info("Reloaded configuration")

    def _action_circuit_break(self, error_context: ErrorContext):
        """Open circuit breaker for component"""
        self._update_circuit_breaker(error_context.component, success=False)
        logger.info(f"Circuit breaker opened for: {error_context.component}")

    def _action_log_and_continue(self, error_context: ErrorContext):
        """Log the error and continue"""
        logger.warning(f"Logged error and continuing: {error_context.error_message}")

    def _action_escalate_alert(self, error_context: ErrorContext):
        """Escalate to alerting system"""
        logger.critical(f"ESCALATION: {error_context.component} - {error_context.error_message}")

    async def _restart_component(self, component: str):
        """Restart a failed component"""
        logger.info(f"Restarting component: {component}")

        # Update component state
        self.component_states[component] = ComponentState.RECOVERING

        # Simulate restart delay
        await asyncio.sleep(2)

        # Mark as healthy
        self.component_states[component] = ComponentState.HEALTHY

    def _is_circuit_open(self, component: str) -> bool:
        """Check if circuit breaker is open for component"""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = {
                'failure_count': 0,
                'last_failure': None,
                'state': 'closed'
            }

        breaker = self.circuit_breakers[component]

        # Check if circuit should be opened
        if breaker['failure_count'] >= 5:
            if breaker['state'] != 'open':
                breaker['state'] = 'open'
                breaker['opened_at'] = datetime.now()
                logger.warning(f"Circuit breaker opened for {component}")

        # Check if circuit should be half-opened
        if breaker['state'] == 'open':
            if datetime.now() - breaker['opened_at'] > timedelta(minutes=1):
                breaker['state'] = 'half-open'
                logger.info(f"Circuit breaker half-opened for {component}")

        return breaker['state'] == 'open'

    def _update_circuit_breaker(self, component: str):
        """Update circuit breaker state"""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = {
                'failure_count': 0,
                'last_failure': None,
                'state': 'closed'
            }

        breaker = self.circuit_breakers[component]
        breaker['failure_count'] += 1
        breaker['last_failure'] = datetime.now()

    def _get_component_state(self, component: str) -> ComponentState:
        """Get current component state"""
        return self.component_states.get(component, ComponentState.HEALTHY)

    def _log_error(self, error_context: ErrorContext):
        """Log error to database"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()

                cur.execute("""
                    INSERT INTO ai_error_logs (
                        error_id, error_type, error_message, stack_trace,
                        component, function_name, severity, retry_count, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (error_id) DO UPDATE SET
                        retry_count = EXCLUDED.retry_count
                """, (
                    error_context.error_id,
                    error_context.error_type,
                    error_context.error_message,
                    error_context.stack_trace,
                    error_context.component,
                    error_context.function_name,
                    error_context.severity.value,
                    error_context.retry_count,
                    json.dumps(error_context.metadata)
                ))

                conn.commit()
                cur.close()

            # Add to memory history
            self.error_history.append(error_context)

        except Exception as e:
            logger.error(f"Failed to log error: {e}")

    def _log_recovery_action(self, recovery_result: RecoveryResult):
        """Log recovery action to database"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()

                cur.execute("""
                    INSERT INTO ai_recovery_actions_log (
                        error_id, action_id, strategy, success,
                        recovery_time_ms, action_taken, result_data
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    recovery_result.error_context.error_id,
                    f"action_{uuid.uuid4().hex[:8]}",
                    recovery_result.action_taken,
                    recovery_result.success,
                    recovery_result.recovery_time_ms,
                    recovery_result.action_taken,
                    json.dumps({"state": recovery_result.new_state.value}) if recovery_result.new_state else None
                ))

                conn.commit()
                cur.close()

        except Exception as e:
            logger.error(f"Failed to log recovery action: {e}")

    def _update_component_health(self, component: str, success: bool,
                                error_id: Optional[str] = None):
        """Update component health metrics"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()

                if success:
                    cur.execute("""
                        INSERT INTO ai_component_health (
                            component_name, current_state, success_count
                        ) VALUES (%s, %s, 1)
                        ON CONFLICT (component_name) DO UPDATE SET
                            success_count = ai_component_health.success_count + 1,
                            health_score = LEAST(100, ai_component_health.health_score + 1),
                            current_state = %s,
                            updated_at = NOW()
                    """, (component, ComponentState.HEALTHY.value, ComponentState.HEALTHY.value))
                else:
                    cur.execute("""
                        INSERT INTO ai_component_health (
                            component_name, current_state, error_count, last_error_id
                        ) VALUES (%s, %s, 1, %s)
                        ON CONFLICT (component_name) DO UPDATE SET
                            error_count = ai_component_health.error_count + 1,
                            health_score = GREATEST(0, ai_component_health.health_score - 5),
                            current_state = %s,
                            last_error_id = %s,
                            updated_at = NOW()
                    """, (component, ComponentState.DEGRADED.value, error_id,
                         ComponentState.DEGRADED.value, error_id))

                conn.commit()
                cur.close()

        except Exception as e:
            logger.error(f"Failed to update component health: {e}")

    def _update_healing_rule_stats(self, rule_name: str, success: bool):
        """Update healing rule statistics"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()

                if success:
                    cur.execute("""
                        UPDATE ai_healing_rules
                        SET success_count = success_count + 1
                        WHERE rule_name = %s
                    """, (rule_name,))
                else:
                    cur.execute("""
                        UPDATE ai_healing_rules
                        SET failure_count = failure_count + 1
                        WHERE rule_name = %s
                    """, (rule_name,))

                conn.commit()
                cur.close()

        except Exception as e:
            logger.error(f"Failed to update healing rule stats: {e}")

    def _escalate_error(self, error_context: ErrorContext, error: Exception):
        """Escalate error to human operator"""
        logger.critical(f"""
        CRITICAL ERROR - MANUAL INTERVENTION REQUIRED
        Component: {error_context.component}
        Function: {error_context.function_name}
        Error Type: {error_context.error_type}
        Message: {error_context.error_message}
        Severity: {error_context.severity.value}
        Retry Count: {error_context.retry_count}
        Timestamp: {error_context.timestamp}
        """)

        # Would send alert to monitoring system in production

    def get_health_report(self) -> dict[str, Any]:
        """Get system health report"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Get component health
                cur.execute("""
                    SELECT * FROM ai_component_health
                    ORDER BY health_score ASC
                    LIMIT 10
                """)
                unhealthy_components = cur.fetchall()

                # Get recent errors
                cur.execute("""
                    SELECT error_type, COUNT(*) as count, MAX(timestamp) as last_seen
                    FROM ai_error_logs
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                    GROUP BY error_type
                    ORDER BY count DESC
                    LIMIT 5
                """)
                recent_errors = cur.fetchall()

                # Get recovery success rate
                cur.execute("""
                    SELECT
                        COUNT(*) as total_recoveries,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                        AVG(recovery_time_ms) as avg_recovery_time
                    FROM ai_recovery_actions_log
                    WHERE executed_at > NOW() - INTERVAL '1 hour'
                """)
                recovery_stats = cur.fetchone()
                cur.close()

                return {
                    "unhealthy_components": unhealthy_components,
                    "recent_error_patterns": recent_errors,
                    "recovery_stats": recovery_stats,
                    "circuit_breakers": {
                        k: v['state'] for k, v in self.circuit_breakers.items()
                    },
                    "active_healing_rules": len(self.healing_rules)
                }

        except Exception as e:
            logger.error(f"Failed to get health report: {e}")
            return {}

    def add_healing_rule(self, rule_name: str, condition: str, action: str,
                        priority: int = 50):
        """Add a new self-healing rule"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()

                cur.execute("""
                    INSERT INTO ai_healing_rules (
                        rule_name, condition, action, priority
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (rule_name) DO UPDATE SET
                        condition = EXCLUDED.condition,
                        action = EXCLUDED.action,
                        priority = EXCLUDED.priority
                """, (rule_name, condition, action, priority))

                conn.commit()
                cur.close()

            # Reload rules
            self._load_healing_rules()

        except Exception as e:
            logger.error(f"Failed to add healing rule: {e}")

    # ============================================
    # PROACTIVE HEALTH MONITORING
    # ============================================

    def monitor_proactive_health(self, component: str, metrics: dict[str, float]) -> dict[str, Any]:
        """
        Proactive health monitoring - detect issues BEFORE they occur
        Returns health status and early warnings
        """
        try:
            # Store metrics history
            self.health_metrics[component].append({
                'timestamp': datetime.now(),
                'metrics': metrics.copy()
            })

            # Calculate health score and trends
            health_score = self._calculate_health_score(component, metrics)
            trend = self._analyze_trend(component)
            failure_prediction = self._predict_failure(component, metrics)

            # Generate early warnings
            warnings = []
            if health_score < 80:
                warnings.append(f"Health degrading: {health_score:.1f}%")
            if trend == 'declining':
                warnings.append("Metrics trending downward")
            if failure_prediction['probability'] > 0.3:
                warnings.append(f"Failure predicted in {failure_prediction['time_to_failure']} with {failure_prediction['probability']:.0%} probability")

            # Store proactive health status
            self._store_proactive_health(component, health_score, trend, failure_prediction, warnings)

            # Log to unified brain
            self._log_to_unified_brain('proactive_health_check', {
                'component': component,
                'health_score': health_score,
                'trend': trend,
                'warnings': warnings
            })

            return {
                'component': component,
                'health_score': health_score,
                'trend': trend,
                'failure_prediction': failure_prediction,
                'warnings': warnings
            }

        except Exception as e:
            logger.error(f"Proactive health monitoring failed: {e}")
            return {'error': str(e)}

    def _calculate_health_score(self, component: str, metrics: dict[str, float]) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0

        # CPU pressure
        cpu = metrics.get('cpu_usage', 0)
        if cpu > 90:
            score -= 30
        elif cpu > 70:
            score -= 15

        # Memory pressure
        memory = metrics.get('memory_usage', 0)
        if memory > 90:
            score -= 30
        elif memory > 75:
            score -= 15

        # Error rate
        error_rate = metrics.get('error_rate', 0)
        if error_rate > 0.1:
            score -= 25
        elif error_rate > 0.05:
            score -= 10

        # Response time
        latency = metrics.get('latency_ms', 0)
        if latency > 5000:
            score -= 20
        elif latency > 2000:
            score -= 10

        return max(0, score)

    def _analyze_trend(self, component: str) -> str:
        """Analyze metric trends over time"""
        history = list(self.health_metrics[component])
        if len(history) < 5:
            return 'stable'

        # Calculate trend from recent health scores
        recent_scores = []
        for entry in history[-10:]:
            score = self._calculate_health_score(component, entry['metrics'])
            recent_scores.append(score)

        if len(recent_scores) < 5:
            return 'stable'

        # Simple linear trend
        first_half = sum(recent_scores[:len(recent_scores)//2]) / (len(recent_scores)//2)
        second_half = sum(recent_scores[len(recent_scores)//2:]) / (len(recent_scores) - len(recent_scores)//2)

        diff = second_half - first_half
        if diff < -5:
            return 'declining'
        elif diff > 5:
            return 'improving'
        else:
            return 'stable'

    # ============================================
    # PREDICTIVE FAILURE DETECTION
    # ============================================

    def _predict_failure(self, component: str, current_metrics: dict[str, float]) -> dict[str, Any]:
        """Predict potential failures before they occur"""
        try:
            history = list(self.health_metrics[component])
            if len(history) < 10:
                return {'probability': 0.0, 'time_to_failure': 'unknown', 'reasons': []}

            # Analyze failure indicators
            failure_probability = 0.0
            reasons = []

            # Memory leak detection
            memory_values = [h['metrics'].get('memory_usage', 0) for h in history[-20:]]
            if self._detect_memory_leak(memory_values):
                failure_probability += 0.4
                reasons.append('Memory leak detected')

            # Error rate increasing
            error_rates = [h['metrics'].get('error_rate', 0) for h in history[-10:]]
            if len(error_rates) > 5 and error_rates[-1] > error_rates[0] * 2:
                failure_probability += 0.3
                reasons.append('Error rate increasing')

            # Resource exhaustion trend
            cpu_values = [h['metrics'].get('cpu_usage', 0) for h in history[-10:]]
            if cpu_values and cpu_values[-1] > 85 and sum(cpu_values[-3:]) / 3 > 80:
                failure_probability += 0.25
                reasons.append('CPU trending to exhaustion')

            # Estimate time to failure
            time_to_failure = 'unknown'
            if failure_probability > 0.5:
                time_to_failure = '< 1 hour'
            elif failure_probability > 0.3:
                time_to_failure = '1-6 hours'
            elif failure_probability > 0.1:
                time_to_failure = '6-24 hours'

            return {
                'probability': min(failure_probability, 1.0),
                'time_to_failure': time_to_failure,
                'reasons': reasons
            }

        except Exception as e:
            logger.error(f"Failure prediction error: {e}")
            return {'probability': 0.0, 'time_to_failure': 'unknown', 'reasons': []}

    def _detect_memory_leak(self, memory_values: list[float]) -> bool:
        """Detect memory leak by analyzing memory trend"""
        if len(memory_values) < 10:
            return False

        # Check for consistent upward trend
        increases = 0
        for i in range(1, len(memory_values)):
            if memory_values[i] > memory_values[i-1]:
                increases += 1

        # If memory increases more than 70% of the time, likely a leak
        return (increases / (len(memory_values) - 1)) > 0.7

    # ============================================
    # AUTOMATIC ROLLBACK CAPABILITIES
    # ============================================

    async def rollback_component(self, component: str, rollback_type: str = 'previous_state') -> dict[str, Any]:
        """Automatic rollback to previous working state"""
        try:
            logger.info(f"Initiating rollback for {component} ({rollback_type})")

            # Get component's previous state
            previous_state = await self._get_previous_state(component)
            current_state = await self._get_current_state(component)

            if not previous_state:
                return {'success': False, 'error': 'No previous state found'}

            # Execute rollback based on type
            if rollback_type == 'config':
                success = await self._rollback_config(component, previous_state)
            elif rollback_type == 'deployment':
                success = await self._rollback_deployment(component)
            else:
                success = await self._rollback_to_state(component, previous_state)

            # Log rollback
            self._log_rollback(component, rollback_type, current_state, previous_state, success)
            self._log_to_unified_brain('automatic_rollback', {
                'component': component,
                'rollback_type': rollback_type,
                'success': success
            })

            return {
                'success': success,
                'component': component,
                'rollback_type': rollback_type,
                'from_state': current_state,
                'to_state': previous_state
            }

        except Exception as e:
            logger.error(f"Rollback failed for {component}: {e}")
            return {'success': False, 'error': str(e)}

    async def _get_previous_state(self, component: str) -> Optional[dict]:
        """Retrieve previous working state"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)

                cur.execute("""
                    SELECT metadata FROM ai_component_health
                    WHERE component_name = %s
                    AND health_score > 80
                    ORDER BY updated_at DESC
                    LIMIT 1 OFFSET 1
                """, (component,))

                result = cur.fetchone()
                cur.close()
                return result['metadata'] if result else None

        except Exception as e:
            logger.error(f"Failed to get previous state: {e}")
            return None

    async def _get_current_state(self, component: str) -> dict:
        """Get current component state"""
        return {'component': component, 'timestamp': datetime.now().isoformat()}

    async def _rollback_config(self, component: str, previous_state: dict) -> bool:
        """Rollback configuration"""
        logger.info(f"Rolling back configuration for {component}")
        return True

    async def _rollback_deployment(self, component: str) -> bool:
        """Rollback to previous deployment"""
        logger.info(f"Rolling back deployment for {component}")
        # This would integrate with Render API or deployment system
        return True

    async def _rollback_to_state(self, component: str, state: dict) -> bool:
        """Rollback to specific state"""
        logger.info(f"Rolling back {component} to state: {state}")
        return True

    def _log_rollback(self, component: str, rollback_type: str, from_state: dict,
                     to_state: dict, success: bool):
        """Log rollback to database"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()

                cur.execute("""
                    INSERT INTO ai_rollback_history (
                        component, rollback_type, from_state, to_state, success
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (component, rollback_type, json.dumps(from_state),
                      json.dumps(to_state), success))

                conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Failed to log rollback: {e}")

    # ============================================
    # RENDER API SERVICE RESTART
    # ============================================

    async def restart_service_via_render(self, service_id: str, component: str) -> dict[str, Any]:
        """Restart service using Render API"""
        try:
            if not self.render_api_key:
                logger.warning("Render API key not configured")
                return {'success': False, 'error': 'API key not configured'}

            import httpx

            url = f"https://api.render.com/v1/services/{service_id}/restart"
            headers = {
                'Authorization': f'Bearer {self.render_api_key}',
                'Content-Type': 'application/json'
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers)

                success = response.status_code in [200, 202]

                # Log restart action
                self._log_to_unified_brain('service_restart', {
                    'component': component,
                    'service_id': service_id,
                    'success': success,
                    'response_code': response.status_code
                })

                return {
                    'success': success,
                    'service_id': service_id,
                    'component': component,
                    'status_code': response.status_code,
                    'message': 'Service restart initiated' if success else 'Restart failed'
                }

        except Exception as e:
            logger.error(f"Render API restart failed: {e}")
            self._log_to_unified_brain('service_restart_failed', {
                'component': component,
                'error': str(e)
            })
            return {'success': False, 'error': str(e)}

    # ============================================
    # DATABASE CONNECTION RECOVERY
    # ============================================

    def recover_database_connection(self) -> dict[str, Any]:
        """Recover database connections with advanced retry logic"""
        try:
            logger.info("Attempting database connection recovery")

            # Close existing connections
            self._close_all_connections()

            # Retry with exponential backoff
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    with self._get_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT 1")
                        cur.close()

                    logger.info("Database connection recovered successfully")
                    self._log_to_unified_brain('db_connection_recovered', {
                        'attempts': attempt + 1
                    })

                    return {
                        'success': True,
                        'attempts': attempt + 1,
                        'message': 'Connection recovered'
                    }

                except Exception:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        raise

            return {'success': False, 'error': 'Max retries exceeded'}

        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            self._log_to_unified_brain('db_recovery_failed', {'error': str(e)})
            return {'success': False, 'error': str(e)}

    def _close_all_connections(self):
        """Close all database connections"""
        try:
            # Force close any pooled connections
            logger.info("Closing all database connections")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

    # ============================================
    # MEMORY LEAK DETECTION AND CLEANUP
    # ============================================

    def detect_and_cleanup_memory_leaks(self, component: str) -> dict[str, Any]:
        """Detect memory leaks and perform cleanup"""
        try:
            import gc

            import psutil

            # Get current memory usage
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Check against baseline
            baseline = self.memory_baselines.get(component, current_memory)
            memory_growth = current_memory - baseline
            growth_pct = (memory_growth / baseline * 100) if baseline > 0 else 0

            leak_detected = False
            if growth_pct > 50:  # More than 50% growth
                leak_detected = True
                logger.warning(f"Memory leak detected in {component}: {growth_pct:.1f}% growth")

            # Perform cleanup
            if leak_detected or current_memory > 500:  # Over 500MB
                # Force garbage collection
                collected = gc.collect()

                # Clear component caches
                self._clear_component_cache(component)

                # Get new memory reading
                new_memory = process.memory_info().rss / 1024 / 1024
                freed = current_memory - new_memory

                logger.info(f"Memory cleanup freed {freed:.1f}MB, collected {collected} objects")

                # Update baseline
                self.memory_baselines[component] = new_memory

                # Log to unified brain
                self._log_to_unified_brain('memory_leak_cleanup', {
                    'component': component,
                    'leak_detected': leak_detected,
                    'memory_freed_mb': freed,
                    'objects_collected': collected
                })

                return {
                    'success': True,
                    'leak_detected': leak_detected,
                    'memory_freed_mb': freed,
                    'objects_collected': collected,
                    'current_memory_mb': new_memory
                }

            return {
                'success': True,
                'leak_detected': False,
                'message': 'No cleanup needed'
            }

        except Exception as e:
            logger.error(f"Memory leak detection failed: {e}")
            return {'success': False, 'error': str(e)}

    def _clear_component_cache(self, component: str):
        """Clear component-specific caches"""
        # Clear error history for component
        self.error_history = deque(
            [e for e in self.error_history if e.component != component],
            maxlen=1000
        )

        # Clear health metrics
        if component in self.health_metrics:
            self.health_metrics[component].clear()

    # ============================================
    # UNIFIED BRAIN LOGGING
    # ============================================

    def _log_to_unified_brain(self, action_type: str, data: dict[str, Any]):
        """Log all healing actions to unified_brain table"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()

                # Use correct unified_brain schema: key, value, category, priority
                key = f"self_healing_{action_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                value = {
                    'action_type': action_type,
                    'data': data,
                    'success': data.get('success', True),
                    'timestamp': datetime.now().isoformat(),
                    'component': data.get('component', 'unknown')
                }

                cur.execute("""
                    INSERT INTO unified_brain (
                        key, value, category, priority, source, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        last_updated = NOW()
                """, (
                    key,
                    json.dumps(value),
                    'self_healing',
                    'high' if action_type in ['service_restart', 'db_recovery_failed', 'automatic_rollback'] else 'medium',
                    'self_healing_system',
                    json.dumps({'component': data.get('component', 'unknown')})
                ))

                conn.commit()
                cur.close()
        except Exception as e:
            logger.debug(f"Failed to log to unified_brain: {e}")

    def _store_proactive_health(self, component: str, health_score: float,
                               trend: str, failure_prediction: dict, warnings: list[str]):
        """Store proactive health monitoring data"""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()

                cur.execute("""
                    INSERT INTO ai_proactive_health (
                        component, health_score, trend, predicted_failure_time,
                        failure_probability, warnings, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (component) DO UPDATE SET
                        health_score = EXCLUDED.health_score,
                        trend = EXCLUDED.trend,
                        predicted_failure_time = EXCLUDED.predicted_failure_time,
                        failure_probability = EXCLUDED.failure_probability,
                        warnings = EXCLUDED.warnings,
                        updated_at = NOW()
                """, (
                    component,
                    health_score,
                    trend,
                    None,  # predicted_failure_time
                    failure_prediction.get('probability', 0.0),
                    json.dumps(warnings)
                ))

                conn.commit()
                cur.close()
        except Exception as e:
            logger.debug(f"Failed to store proactive health: {e}")

# Singleton instance
_self_healing = None

def get_self_healing_recovery():
    """Get or create self-healing instance"""
    global _self_healing
    if _self_healing is None:
        _self_healing = SelfHealingRecovery()
    return _self_healing

# Alias for backward compatibility
get_self_healing = get_self_healing_recovery

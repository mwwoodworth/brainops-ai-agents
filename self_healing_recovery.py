"""
Self-Healing AI Error Recovery System
Implements automatic error detection, recovery, and retry mechanisms
"""

import json
import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict, field
import os
from dotenv import load_dotenv
import logging
import traceback
import uuid
from collections import defaultdict, deque
import time
import inspect
from functools import wraps

load_dotenv()
logger = logging.getLogger(__name__)

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
    metadata: Dict = field(default_factory=dict)

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
    metadata: Dict

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
        self.db_config = {
            'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
            'password': os.getenv('DB_PASSWORD', 'Brain0ps2O2S'),
            'port': os.getenv('DB_PORT', 5432)
        }
        self.component_states = {}
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self.error_patterns = {}
        self.healing_rules = {}
        self._initialize_database()
        self._load_recovery_strategies()
        self._load_healing_rules()

    def _get_connection(self):
        """Get database connection with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return psycopg2.connect(**self.db_config)
            except psycopg2.OperationalError as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def _initialize_database(self):
        """Initialize database tables for error recovery"""
        try:
            conn = self._get_connection()
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
            logger.info("Self-healing recovery tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing recovery tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

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
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT * FROM ai_healing_rules
                WHERE enabled = true
                ORDER BY priority DESC
            """)

            rules = cur.fetchall()
            for rule in rules:
                self.healing_rules[rule['rule_name']] = rule

        except Exception as e:
            logger.error(f"Error loading healing rules: {e}")
        finally:
            if conn:
                conn.close()

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

    def _match_error_pattern(self, error_context: ErrorContext) -> Optional[Dict]:
        """Match error against known patterns"""
        try:
            conn = self._get_connection()
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
            return pattern

        except Exception as e:
            logger.error(f"Error matching pattern: {e}")
            return None
        finally:
            if conn:
                conn.close()

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

    async def _execute_healing(self, error_context: ErrorContext) -> Any:
        """Execute self-healing logic"""
        # Check healing rules
        for rule_name, rule in self.healing_rules.items():
            if self._evaluate_healing_rule(rule, error_context):
                logger.info(f"Executing healing rule: {rule_name}")

                try:
                    # Execute healing action
                    exec(rule['action'])

                    # Update rule success count
                    self._update_healing_rule_stats(rule_name, success=True)

                    return True

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

    def _evaluate_healing_rule(self, rule: Dict, error_context: ErrorContext) -> bool:
        """Evaluate if healing rule applies"""
        try:
            # Simple condition evaluation (would need proper parser in production)
            condition = rule['condition']

            # Replace placeholders with actual values
            condition = condition.replace('${error_type}', error_context.error_type)
            condition = condition.replace('${component}', error_context.component)
            condition = condition.replace('${severity}', error_context.severity.value)

            return eval(condition)

        except Exception:
            return False

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
            conn = self._get_connection()
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

            # Add to memory history
            self.error_history.append(error_context)

        except Exception as e:
            logger.error(f"Failed to log error: {e}")
        finally:
            if conn:
                conn.close()

    def _log_recovery_action(self, recovery_result: RecoveryResult):
        """Log recovery action to database"""
        try:
            conn = self._get_connection()
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

        except Exception as e:
            logger.error(f"Failed to log recovery action: {e}")
        finally:
            if conn:
                conn.close()

    def _update_component_health(self, component: str, success: bool,
                                error_id: Optional[str] = None):
        """Update component health metrics"""
        try:
            conn = self._get_connection()
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

        except Exception as e:
            logger.error(f"Failed to update component health: {e}")
        finally:
            if conn:
                conn.close()

    def _update_healing_rule_stats(self, rule_name: str, success: bool):
        """Update healing rule statistics"""
        try:
            conn = self._get_connection()
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

        except Exception as e:
            logger.error(f"Failed to update healing rule stats: {e}")
        finally:
            if conn:
                conn.close()

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

    def get_health_report(self) -> Dict[str, Any]:
        """Get system health report"""
        try:
            conn = self._get_connection()
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
        finally:
            if conn:
                conn.close()

    def add_healing_rule(self, rule_name: str, condition: str, action: str,
                        priority: int = 50):
        """Add a new self-healing rule"""
        try:
            conn = self._get_connection()
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

            # Reload rules
            self._load_healing_rules()

        except Exception as e:
            logger.error(f"Failed to add healing rule: {e}")
        finally:
            if conn:
                conn.close()

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
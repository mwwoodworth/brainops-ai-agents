"""
Supabase Realtime Monitor for Live AI Updates
Provides real-time subscriptions and monitoring of AI activity
"""

import json
import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import os
from dotenv import load_dotenv
import logging
import uuid
from collections import defaultdict
import websocket
import threading
import time

load_dotenv()
logger = logging.getLogger(__name__)

class EventType(Enum):
    AGENT_EXECUTION = "agent_execution"
    DECISION_MADE = "decision_made"
    MEMORY_STORED = "memory_stored"
    SYSTEM_ALERT = "system_alert"
    TASK_COMPLETED = "task_completed"
    ERROR_OCCURRED = "error_occurred"
    LEARNING_EVENT = "learning_event"
    STATE_CHANGE = "state_change"
    CONVERSATION_MESSAGE = "conversation_message"
    REVENUE_EVENT = "revenue_event"

class SubscriptionType(Enum):
    ALL = "all"
    AGENTS = "agents"
    DECISIONS = "decisions"
    MEMORY = "memory"
    ALERTS = "alerts"
    TASKS = "tasks"
    ERRORS = "errors"
    LEARNING = "learning"
    CONVERSATIONS = "conversations"
    REVENUE = "revenue"

@dataclass
class RealtimeEvent:
    """Represents a real-time event"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    priority: int = 5

@dataclass
class Subscription:
    """Represents a subscription to real-time events"""
    subscription_id: str
    client_id: str
    subscription_type: SubscriptionType
    filters: Dict[str, Any]
    callback: Optional[Callable]
    created_at: datetime
    is_active: bool = True

class RealtimeMonitor:
    """Monitors and broadcasts AI system events in real-time"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
            'password': os.getenv('DB_PASSWORD', 'Brain0ps2O2S'),
            'port': os.getenv('DB_PORT', 5432)
        }
        self.subscriptions = {}
        self.event_queue = asyncio.Queue()
        self.event_history = []
        self.listeners = defaultdict(list)
        self.is_running = False
        self.poll_interval = 1  # seconds
        self._initialize_database()
        self._setup_triggers()

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def _initialize_database(self):
        """Initialize database tables for real-time monitoring"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Create real-time events table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_realtime_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_id VARCHAR(255) UNIQUE NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    source VARCHAR(255) NOT NULL,
                    data JSONB NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    priority INT DEFAULT 5,
                    processed BOOLEAN DEFAULT false,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create subscriptions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_realtime_subscriptions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    subscription_id VARCHAR(255) UNIQUE NOT NULL,
                    client_id VARCHAR(255) NOT NULL,
                    subscription_type VARCHAR(50) NOT NULL,
                    filters JSONB DEFAULT '{}'::jsonb,
                    is_active BOOLEAN DEFAULT true,
                    last_event_id VARCHAR(255),
                    events_received INT DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create event broadcasts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_event_broadcasts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_id VARCHAR(255) REFERENCES ai_realtime_events(event_id),
                    subscription_id VARCHAR(255) REFERENCES ai_realtime_subscriptions(subscription_id),
                    delivered BOOLEAN DEFAULT false,
                    delivered_at TIMESTAMPTZ,
                    retry_count INT DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create activity feed table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_activity_feed (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    activity_type VARCHAR(50) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    description TEXT,
                    source VARCHAR(255),
                    severity VARCHAR(20) DEFAULT 'info',
                    data JSONB DEFAULT '{}'::jsonb,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    is_read BOOLEAN DEFAULT false
                )
            """)

            # Create index for performance
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON ai_realtime_events(timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_events_type
                ON ai_realtime_events(event_type);

                CREATE INDEX IF NOT EXISTS idx_events_processed
                ON ai_realtime_events(processed);

                CREATE INDEX IF NOT EXISTS idx_activity_timestamp
                ON ai_activity_feed(timestamp DESC);
            """)

            conn.commit()
            logger.info("Realtime monitoring tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing realtime tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _setup_triggers(self):
        """Setup database triggers for real-time events"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Create trigger function for agent executions
            cur.execute("""
                CREATE OR REPLACE FUNCTION notify_agent_execution()
                RETURNS trigger AS $$
                BEGIN
                    PERFORM pg_notify(
                        'agent_execution',
                        json_build_object(
                            'id', NEW.id,
                            'agent_type', NEW.agent_type,
                            'status', NEW.status,
                            'created_at', NEW.created_at
                        )::text
                    );
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)

            # Create trigger for agent executions
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger
                        WHERE tgname = 'agent_execution_notify'
                    ) THEN
                        CREATE TRIGGER agent_execution_notify
                        AFTER INSERT OR UPDATE ON agent_executions
                        FOR EACH ROW EXECUTE FUNCTION notify_agent_execution();
                    END IF;
                END $$;
            """)

            # Create trigger function for system alerts
            cur.execute("""
                CREATE OR REPLACE FUNCTION notify_system_alert()
                RETURNS trigger AS $$
                BEGIN
                    PERFORM pg_notify(
                        'system_alert',
                        json_build_object(
                            'id', NEW.id,
                            'alert_type', NEW.alert_type,
                            'severity', NEW.severity,
                            'message', NEW.message,
                            'created_at', NEW.created_at
                        )::text
                    );
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)

            # Create trigger for system alerts
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger
                        WHERE tgname = 'system_alert_notify'
                    ) THEN
                        CREATE TRIGGER system_alert_notify
                        AFTER INSERT ON ai_system_alerts
                        FOR EACH ROW EXECUTE FUNCTION notify_system_alert();
                    END IF;
                END $$;
            """)

            conn.commit()
            logger.info("Database triggers setup successfully")

        except Exception as e:
            logger.error(f"Error setting up triggers: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    async def start(self):
        """Start the real-time monitoring service"""
        if self.is_running:
            logger.warning("Realtime monitor is already running")
            return

        self.is_running = True

        # Start background tasks
        asyncio.create_task(self._event_processor())
        asyncio.create_task(self._database_listener())
        asyncio.create_task(self._subscription_manager())
        asyncio.create_task(self._activity_aggregator())

        logger.info("Realtime monitoring started")

    async def stop(self):
        """Stop the real-time monitoring service"""
        self.is_running = False
        logger.info("Realtime monitoring stopped")

    async def _event_processor(self):
        """Process events from the queue"""
        while self.is_running:
            try:
                # Get event from queue
                if not self.event_queue.empty():
                    event = await self.event_queue.get()

                    # Store event
                    self._store_event(event)

                    # Broadcast to subscribers
                    await self._broadcast_event(event)

                    # Update activity feed
                    self._update_activity_feed(event)

                    # Add to history
                    self.event_history.append(event)
                    if len(self.event_history) > 1000:
                        self.event_history = self.event_history[-500:]

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error processing event: {e}")

    async def _database_listener(self):
        """Listen for database notifications"""
        while self.is_running:
            try:
                conn = self._get_connection()
                conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
                cur = conn.cursor()

                # Listen to channels
                cur.execute("LISTEN agent_execution;")
                cur.execute("LISTEN system_alert;")
                cur.execute("LISTEN decision_made;")
                cur.execute("LISTEN memory_update;")

                logger.info("Listening for database notifications...")

                while self.is_running:
                    # Wait for notifications
                    if conn.poll() > 0:
                        conn.consume_input()
                        while conn.notifies:
                            notify = conn.notifies.pop(0)
                            await self._handle_notification(notify)

                    await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Database listener error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
            finally:
                if conn:
                    conn.close()

    async def _handle_notification(self, notify):
        """Handle database notification"""
        try:
            event_data = json.loads(notify.payload)

            # Create event based on channel
            event_type_map = {
                'agent_execution': EventType.AGENT_EXECUTION,
                'system_alert': EventType.SYSTEM_ALERT,
                'decision_made': EventType.DECISION_MADE,
                'memory_update': EventType.MEMORY_STORED
            }

            event = RealtimeEvent(
                event_id=f"evt_{uuid.uuid4().hex[:8]}",
                event_type=event_type_map.get(notify.channel, EventType.STATE_CHANGE),
                timestamp=datetime.now(),
                source=notify.channel,
                data=event_data,
                metadata={'pid': notify.pid},
                priority=self._calculate_priority(event_data)
            )

            await self.event_queue.put(event)

        except Exception as e:
            logger.error(f"Error handling notification: {e}")

    def _calculate_priority(self, event_data: Dict) -> int:
        """Calculate event priority"""
        # High priority for errors and critical alerts
        if event_data.get('severity') == 'critical':
            return 10
        elif event_data.get('status') == 'failed':
            return 8
        elif event_data.get('alert_type') == 'error':
            return 7
        # Medium priority for completions and warnings
        elif event_data.get('status') == 'completed':
            return 5
        elif event_data.get('severity') == 'warning':
            return 5
        # Low priority for routine events
        else:
            return 3

    async def _broadcast_event(self, event: RealtimeEvent):
        """Broadcast event to subscribers"""
        for subscription_id, subscription in self.subscriptions.items():
            if not subscription.is_active:
                continue

            # Check if subscription matches event
            if self._matches_subscription(event, subscription):
                try:
                    # Record broadcast
                    self._record_broadcast(event.event_id, subscription.subscription_id)

                    # Execute callback if provided
                    if subscription.callback:
                        await subscription.callback(event)

                    # Update subscription stats
                    self._update_subscription_stats(subscription.subscription_id)

                except Exception as e:
                    logger.error(f"Error broadcasting to {subscription_id}: {e}")

    def _matches_subscription(self, event: RealtimeEvent,
                            subscription: Subscription) -> bool:
        """Check if event matches subscription filters"""
        # Check subscription type
        if subscription.subscription_type == SubscriptionType.ALL:
            return True

        type_map = {
            SubscriptionType.AGENTS: EventType.AGENT_EXECUTION,
            SubscriptionType.DECISIONS: EventType.DECISION_MADE,
            SubscriptionType.MEMORY: EventType.MEMORY_STORED,
            SubscriptionType.ALERTS: EventType.SYSTEM_ALERT,
            SubscriptionType.TASKS: EventType.TASK_COMPLETED,
            SubscriptionType.ERRORS: EventType.ERROR_OCCURRED,
            SubscriptionType.LEARNING: EventType.LEARNING_EVENT,
            SubscriptionType.CONVERSATIONS: EventType.CONVERSATION_MESSAGE,
            SubscriptionType.REVENUE: EventType.REVENUE_EVENT
        }

        if type_map.get(subscription.subscription_type) != event.event_type:
            return False

        # Check additional filters
        for key, value in subscription.filters.items():
            if key in event.data and event.data[key] != value:
                return False

        return True

    async def _subscription_manager(self):
        """Manage subscriptions and clean up inactive ones"""
        while self.is_running:
            try:
                # Clean up inactive subscriptions
                inactive_ids = []
                for sub_id, subscription in self.subscriptions.items():
                    if not subscription.is_active:
                        inactive_ids.append(sub_id)

                for sub_id in inactive_ids:
                    del self.subscriptions[sub_id]

                # Update database
                self._sync_subscriptions()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Subscription manager error: {e}")

    async def _activity_aggregator(self):
        """Aggregate events into activity feed"""
        while self.is_running:
            try:
                # Get recent events
                recent_events = await self._get_recent_events(minutes=5)

                # Aggregate by type
                aggregated = self._aggregate_events(recent_events)

                # Update activity feed
                for activity in aggregated:
                    self._add_to_activity_feed(activity)

                await asyncio.sleep(60)  # Aggregate every minute

            except Exception as e:
                logger.error(f"Activity aggregator error: {e}")

    def _store_event(self, event: RealtimeEvent):
        """Store event in database"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_realtime_events (
                    event_id, event_type, timestamp, source, data, metadata, priority
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
            """, (
                event.event_id,
                event.event_type.value,
                event.timestamp,
                event.source,
                json.dumps(event.data),
                json.dumps(event.metadata),
                event.priority
            ))

            conn.commit()

        except Exception as e:
            logger.error(f"Error storing event: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _update_activity_feed(self, event: RealtimeEvent):
        """Update activity feed with event"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Create activity title and description
            title, description, severity = self._format_activity(event)

            cur.execute("""
                INSERT INTO ai_activity_feed (
                    activity_type, title, description, source, severity, data
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                event.event_type.value,
                title,
                description,
                event.source,
                severity,
                json.dumps(event.data)
            ))

            conn.commit()

        except Exception as e:
            logger.error(f"Error updating activity feed: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _format_activity(self, event: RealtimeEvent) -> Tuple[str, str, str]:
        """Format event for activity feed"""
        if event.event_type == EventType.AGENT_EXECUTION:
            title = f"Agent {event.data.get('agent_type', 'Unknown')} executed"
            description = f"Status: {event.data.get('status', 'Unknown')}"
            severity = 'error' if event.data.get('status') == 'failed' else 'info'

        elif event.event_type == EventType.DECISION_MADE:
            title = "Decision made"
            description = f"Type: {event.data.get('decision_type', 'Unknown')}"
            severity = 'info'

        elif event.event_type == EventType.SYSTEM_ALERT:
            title = f"System alert: {event.data.get('alert_type', 'Unknown')}"
            description = event.data.get('message', '')
            severity = event.data.get('severity', 'warning')

        elif event.event_type == EventType.TASK_COMPLETED:
            title = f"Task completed"
            description = f"Task: {event.data.get('task_name', 'Unknown')}"
            severity = 'success'

        else:
            title = f"{event.event_type.value.replace('_', ' ').title()}"
            description = ""
            severity = 'info'

        return title, description, severity

    def _record_broadcast(self, event_id: str, subscription_id: str):
        """Record event broadcast"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_event_broadcasts (
                    event_id, subscription_id, delivered, delivered_at
                ) VALUES (%s, %s, true, NOW())
            """, (event_id, subscription_id))

            conn.commit()

        except Exception as e:
            logger.error(f"Error recording broadcast: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _update_subscription_stats(self, subscription_id: str):
        """Update subscription statistics"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE ai_realtime_subscriptions
                SET events_received = events_received + 1,
                    updated_at = NOW()
                WHERE subscription_id = %s
            """, (subscription_id,))

            conn.commit()

        except Exception as e:
            logger.error(f"Error updating subscription stats: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _sync_subscriptions(self):
        """Sync subscriptions with database"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            for subscription in self.subscriptions.values():
                cur.execute("""
                    INSERT INTO ai_realtime_subscriptions (
                        subscription_id, client_id, subscription_type,
                        filters, is_active
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (subscription_id) DO UPDATE SET
                        is_active = EXCLUDED.is_active,
                        updated_at = NOW()
                """, (
                    subscription.subscription_id,
                    subscription.client_id,
                    subscription.subscription_type.value,
                    json.dumps(subscription.filters),
                    subscription.is_active
                ))

            conn.commit()

        except Exception as e:
            logger.error(f"Error syncing subscriptions: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    async def _get_recent_events(self, minutes: int = 5) -> List[Dict]:
        """Get recent events from database"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT * FROM ai_realtime_events
                WHERE timestamp > NOW() - INTERVAL '%s minutes'
                ORDER BY timestamp DESC
            """, (minutes,))

            events = cur.fetchall()
            return events

        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def _aggregate_events(self, events: List[Dict]) -> List[Dict]:
        """Aggregate events for activity feed"""
        aggregated = defaultdict(lambda: {'count': 0, 'events': []})

        for event in events:
            key = event['event_type']
            aggregated[key]['count'] += 1
            aggregated[key]['events'].append(event)

        # Create aggregated activities
        activities = []
        for event_type, data in aggregated.items():
            if data['count'] > 1:
                activities.append({
                    'type': event_type,
                    'title': f"{data['count']} {event_type} events",
                    'description': f"In the last 5 minutes",
                    'severity': 'info'
                })

        return activities

    def _add_to_activity_feed(self, activity: Dict):
        """Add aggregated activity to feed"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_activity_feed (
                    activity_type, title, description, severity
                ) VALUES (%s, %s, %s, %s)
            """, (
                activity['type'],
                activity['title'],
                activity['description'],
                activity['severity']
            ))

            conn.commit()

        except Exception as e:
            logger.error(f"Error adding to activity feed: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    # Public methods for managing subscriptions
    def subscribe(self, client_id: str, subscription_type: SubscriptionType,
                 filters: Optional[Dict] = None,
                 callback: Optional[Callable] = None) -> str:
        """Subscribe to real-time events"""
        subscription_id = f"sub_{uuid.uuid4().hex[:8]}"

        subscription = Subscription(
            subscription_id=subscription_id,
            client_id=client_id,
            subscription_type=subscription_type,
            filters=filters or {},
            callback=callback,
            created_at=datetime.now()
        )

        self.subscriptions[subscription_id] = subscription

        logger.info(f"Created subscription {subscription_id} for client {client_id}")

        return subscription_id

    def unsubscribe(self, subscription_id: str):
        """Unsubscribe from real-time events"""
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].is_active = False
            logger.info(f"Deactivated subscription {subscription_id}")

    def emit_event(self, event_type: EventType, source: str,
                  data: Dict[str, Any], metadata: Optional[Dict] = None):
        """Manually emit an event"""
        event = RealtimeEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            timestamp=datetime.now(),
            source=source,
            data=data,
            metadata=metadata or {},
            priority=self._calculate_priority(data)
        )

        # Add to queue
        asyncio.create_task(self.event_queue.put(event))

    def get_activity_feed(self, limit: int = 50) -> List[Dict]:
        """Get recent activity feed"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT * FROM ai_activity_feed
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))

            activities = cur.fetchall()

            # Convert to list of dicts
            return [dict(activity) for activity in activities]

        except Exception as e:
            logger.error(f"Error getting activity feed: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_event_history(self, event_type: Optional[EventType] = None,
                         limit: int = 100) -> List[Dict]:
        """Get event history"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            if event_type:
                cur.execute("""
                    SELECT * FROM ai_realtime_events
                    WHERE event_type = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (event_type.value, limit))
            else:
                cur.execute("""
                    SELECT * FROM ai_realtime_events
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))

            events = cur.fetchall()

            return [dict(event) for event in events]

        except Exception as e:
            logger.error(f"Error getting event history: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get subscription statistics"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT
                    COUNT(*) as total_subscriptions,
                    SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active_subscriptions,
                    SUM(events_received) as total_events_delivered,
                    AVG(events_received) as avg_events_per_subscription
                FROM ai_realtime_subscriptions
            """)

            stats = cur.fetchone()

            return dict(stats) if stats else {}

        except Exception as e:
            logger.error(f"Error getting subscription stats: {e}")
            return {}
        finally:
            if conn:
                conn.close()

# Singleton instance
_realtime_monitor = None

def get_realtime_monitor():
    """Get or create realtime monitor instance"""
    global _realtime_monitor
    if _realtime_monitor is None:
        _realtime_monitor = RealtimeMonitor()
    return _realtime_monitor
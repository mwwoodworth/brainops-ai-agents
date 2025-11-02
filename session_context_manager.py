#!/usr/bin/env python3
"""
SESSION CONTEXT MANAGER - Perfect Session-Level Memory Management
Ensures complete context preservation across conversations and agent handoffs
"""

import os
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import hashlib
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SESSION STRUCTURES
# ============================================================================

@dataclass
class SessionState:
    """Complete state of a session"""
    session_id: str
    tenant_id: Optional[str]
    user_id: Optional[str]
    start_time: datetime
    last_activity: datetime
    status: str  # active, paused, completed, expired

    # Context tracking
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    active_agents: List[str] = field(default_factory=list)
    completed_tasks: List[Dict[str, Any]] = field(default_factory=list)
    pending_tasks: List[Dict[str, Any]] = field(default_factory=list)

    # Memory tracking
    critical_facts: Dict[str, Any] = field(default_factory=dict)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    long_term_refs: List[str] = field(default_factory=list)

    # Handoff tracking
    handoff_history: List[Dict[str, Any]] = field(default_factory=list)
    current_agent: Optional[str] = None
    previous_agent: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    snapshot_version: int = 1


@dataclass
class AgentHandoff:
    """Information for agent handoffs"""
    from_agent: str
    to_agent: str
    timestamp: datetime
    context_snapshot: Dict[str, Any]
    handoff_reason: str
    critical_info: Dict[str, Any]
    continuation_instructions: str


# ============================================================================
# SESSION CONTEXT MANAGER
# ============================================================================

class SessionContextManager:
    """
    Manages perfect context preservation for AI sessions
    Ensures seamless handoffs and zero context loss
    """

    def __init__(self, memory_coordinator):
        self.coordinator = memory_coordinator
        self.active_sessions: Dict[str, SessionState] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}

    # ========================================================================
    # SESSION LIFECYCLE
    # ========================================================================

    async def start_session(
        self,
        session_id: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> SessionState:
        """
        Start a new session with full context initialization
        """
        logger.info(f"🚀 Starting session: {session_id}")

        # Create session state
        session = SessionState(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            start_time=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            status="active",
            metadata=initial_context or {}
        )

        # Load any existing context from previous sessions
        await self._load_historical_context(session)

        # Store in active sessions
        self.active_sessions[session_id] = session
        self.session_locks[session_id] = asyncio.Lock()

        # Persist to database
        await self._persist_session(session)

        logger.info(f"✅ Session started: {session_id}")
        return session

    async def resume_session(self, session_id: str) -> Optional[SessionState]:
        """
        Resume an existing session with full context restoration
        """
        logger.info(f"🔄 Resuming session: {session_id}")

        # Check if already active
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.last_activity = datetime.now(timezone.utc)
            session.status = "active"
            return session

        # Load from database
        session = await self._load_session(session_id)

        if session:
            session.status = "active"
            session.last_activity = datetime.now(timezone.utc)
            self.active_sessions[session_id] = session
            self.session_locks[session_id] = asyncio.Lock()

            await self._persist_session(session)
            logger.info(f"✅ Session resumed: {session_id}")
            return session

        logger.warning(f"⚠️ Session not found: {session_id}")
        return None

    async def end_session(self, session_id: str, reason: str = "completed"):
        """
        End a session with full context preservation
        """
        if session_id not in self.active_sessions:
            logger.warning(f"⚠️ Session not active: {session_id}")
            return

        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]
            session.status = reason
            session.last_activity = datetime.now(timezone.utc)

            # Create final snapshot
            await self._create_session_snapshot(session, f"session_ended_{reason}")

            # Archive critical context to long-term memory
            await self._archive_session_context(session)

            # Remove from active
            del self.active_sessions[session_id]
            del self.session_locks[session_id]

            logger.info(f"✅ Session ended: {session_id} (reason: {reason})")

    # ========================================================================
    # CONTEXT OPERATIONS
    # ========================================================================

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to conversation history"""
        if session_id not in self.active_sessions:
            logger.error(f"❌ Session not active: {session_id}")
            return

        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]

            message = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'role': role,
                'content': content,
                'metadata': metadata or {}
            }

            session.conversation_history.append(message)
            session.last_activity = datetime.now(timezone.utc)

            # Auto-snapshot every 50 messages
            if len(session.conversation_history) % 50 == 0:
                await self._create_session_snapshot(session, "auto_snapshot")

            # Extract and store critical facts
            await self._extract_critical_facts(session, content)

    async def add_task(
        self,
        session_id: str,
        task: Dict[str, Any],
        status: str = "pending"
    ):
        """Add a task to session tracking"""
        if session_id not in self.active_sessions:
            return

        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]

            task['added_at'] = datetime.now(timezone.utc).isoformat()
            task['status'] = status

            if status == "pending":
                session.pending_tasks.append(task)
            elif status == "completed":
                session.completed_tasks.append(task)

            session.last_activity = datetime.now(timezone.utc)

    async def complete_task(self, session_id: str, task_id: str, result: Any):
        """Mark a task as completed"""
        if session_id not in self.active_sessions:
            return

        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]

            # Find and move task
            for i, task in enumerate(session.pending_tasks):
                if task.get('id') == task_id:
                    completed_task = session.pending_tasks.pop(i)
                    completed_task['status'] = 'completed'
                    completed_task['completed_at'] = datetime.now(timezone.utc).isoformat()
                    completed_task['result'] = result
                    session.completed_tasks.append(completed_task)
                    break

            session.last_activity = datetime.now(timezone.utc)

    async def update_critical_fact(
        self,
        session_id: str,
        key: str,
        value: Any,
        persist: bool = True
    ):
        """Update a critical fact (persistent across session)"""
        if session_id not in self.active_sessions:
            return

        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]
            session.critical_facts[key] = value
            session.last_activity = datetime.now(timezone.utc)

            if persist:
                # Store in long-term memory
                from memory_coordination_system import ContextEntry, MemoryLayer, ContextScope

                entry = ContextEntry(
                    key=f"critical_fact_{session.tenant_id}_{key}",
                    value=value,
                    layer=MemoryLayer.LONG_TERM,
                    scope=ContextScope.TENANT if session.tenant_id else ContextScope.GLOBAL,
                    priority="critical",
                    category="critical_fact",
                    source=f"session_{session_id}",
                    tenant_id=session.tenant_id,
                    session_id=session_id,
                    metadata={'original_session': session_id}
                )

                await self.coordinator.store_context(entry)

    async def update_working_memory(
        self,
        session_id: str,
        key: str,
        value: Any
    ):
        """Update working memory (ephemeral, session-only)"""
        if session_id not in self.active_sessions:
            return

        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]
            session.working_memory[key] = value
            session.last_activity = datetime.now(timezone.utc)

    # ========================================================================
    # AGENT HANDOFFS
    # ========================================================================

    async def handoff_to_agent(
        self,
        session_id: str,
        to_agent: str,
        handoff_reason: str,
        critical_info: Dict[str, Any],
        continuation_instructions: str
    ) -> AgentHandoff:
        """
        Hand off session to another agent with perfect context transfer
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not active: {session_id}")

        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]

            from_agent = session.current_agent or "system"

            # Create context snapshot
            context_snapshot = {
                'conversation_summary': self._summarize_conversation(session),
                'critical_facts': session.critical_facts.copy(),
                'working_memory': session.working_memory.copy(),
                'pending_tasks': session.pending_tasks.copy(),
                'completed_tasks_count': len(session.completed_tasks),
                'session_duration': (datetime.now(timezone.utc) - session.start_time).total_seconds(),
                'last_messages': session.conversation_history[-10:] if len(session.conversation_history) > 0 else []
            }

            # Create handoff record
            handoff = AgentHandoff(
                from_agent=from_agent,
                to_agent=to_agent,
                timestamp=datetime.now(timezone.utc),
                context_snapshot=context_snapshot,
                handoff_reason=handoff_reason,
                critical_info=critical_info,
                continuation_instructions=continuation_instructions
            )

            # Update session
            session.previous_agent = session.current_agent
            session.current_agent = to_agent
            session.handoff_history.append(asdict(handoff))

            if to_agent not in session.active_agents:
                session.active_agents.append(to_agent)

            session.last_activity = datetime.now(timezone.utc)

            # Persist handoff
            await self._persist_handoff(session_id, handoff)

            logger.info(f"✅ Handoff: {from_agent} → {to_agent} (session: {session_id})")
            return handoff

    async def get_handoff_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest handoff context for an agent"""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        if not session.handoff_history:
            return None

        latest_handoff = session.handoff_history[-1]
        return {
            'from_agent': latest_handoff['from_agent'],
            'to_agent': latest_handoff['to_agent'],
            'handoff_reason': latest_handoff['handoff_reason'],
            'critical_info': latest_handoff['critical_info'],
            'continuation_instructions': latest_handoff['continuation_instructions'],
            'context_snapshot': latest_handoff['context_snapshot'],
            'timestamp': latest_handoff['timestamp']
        }

    # ========================================================================
    # CONTEXT RETRIEVAL
    # ========================================================================

    async def get_full_context(self, session_id: str) -> Dict[str, Any]:
        """Get complete session context"""
        if session_id not in self.active_sessions:
            # Try to load from database
            session = await self._load_session(session_id)
            if not session:
                return {}
        else:
            session = self.active_sessions[session_id]

        return {
            'session_id': session.session_id,
            'tenant_id': session.tenant_id,
            'user_id': session.user_id,
            'status': session.status,
            'start_time': session.start_time.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'duration_seconds': (session.last_activity - session.start_time).total_seconds(),

            'conversation': {
                'message_count': len(session.conversation_history),
                'recent_messages': session.conversation_history[-20:],
                'summary': self._summarize_conversation(session)
            },

            'agents': {
                'current': session.current_agent,
                'previous': session.previous_agent,
                'active': session.active_agents,
                'handoff_count': len(session.handoff_history)
            },

            'tasks': {
                'pending': session.pending_tasks,
                'completed': session.completed_tasks,
                'completion_rate': len(session.completed_tasks) / max(len(session.completed_tasks) + len(session.pending_tasks), 1)
            },

            'memory': {
                'critical_facts': session.critical_facts,
                'working_memory': session.working_memory,
                'long_term_refs': session.long_term_refs
            },

            'metadata': session.metadata
        }

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    async def _persist_session(self, session: SessionState):
        """Persist session to database"""
        from memory_coordination_system import ContextEntry, MemoryLayer, ContextScope

        # Convert session to dict and handle datetime serialization
        session_dict = asdict(session)
        if 'start_time' in session_dict and hasattr(session_dict['start_time'], 'isoformat'):
            session_dict['start_time'] = session_dict['start_time'].isoformat()
        if 'last_activity' in session_dict and hasattr(session_dict['last_activity'], 'isoformat'):
            session_dict['last_activity'] = session_dict['last_activity'].isoformat()

        entry = ContextEntry(
            key=f"session_state_{session.session_id}",
            value=session_dict,
            layer=MemoryLayer.SESSION,
            scope=ContextScope.SESSION,
            priority="high",
            category="session_state",
            source="session_manager",
            tenant_id=session.tenant_id,
            user_id=session.user_id,
            session_id=session.session_id,
            metadata={'snapshot_version': session.snapshot_version}
        )

        await self.coordinator.store_context(entry)

    async def _load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session from database"""
        from memory_coordination_system import ContextScope

        entry = await self.coordinator.retrieve_context(
            key=f"session_state_{session_id}",
            scope=ContextScope.SESSION,
            session_id=session_id
        )

        if entry and entry.value:
            data = entry.value

            # Convert datetime strings back to datetime objects
            if isinstance(data.get('start_time'), str):
                data['start_time'] = datetime.fromisoformat(data['start_time'])
            if isinstance(data.get('last_activity'), str):
                data['last_activity'] = datetime.fromisoformat(data['last_activity'])

            return SessionState(**data)

        return None

    async def _create_session_snapshot(self, session: SessionState, snapshot_type: str):
        """Create a point-in-time snapshot"""
        from memory_coordination_system import ContextEntry, MemoryLayer, ContextScope

        session.snapshot_version += 1

        snapshot = {
            'session_id': session.session_id,
            'snapshot_type': snapshot_type,
            'version': session.snapshot_version,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'conversation_count': len(session.conversation_history),
            'completed_tasks': len(session.completed_tasks),
            'critical_facts': session.critical_facts.copy(),
            'active_agents': session.active_agents.copy()
        }

        entry = ContextEntry(
            key=f"session_snapshot_{session.session_id}_{session.snapshot_version}",
            value=snapshot,
            layer=MemoryLayer.SHORT_TERM,
            scope=ContextScope.SESSION,
            priority="medium",
            category="session_snapshot",
            source="session_manager",
            tenant_id=session.tenant_id,
            session_id=session.session_id,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30)
        )

        await self.coordinator.store_context(entry)

    async def _archive_session_context(self, session: SessionState):
        """Archive important session context to long-term memory"""
        from memory_coordination_system import ContextEntry, MemoryLayer, ContextScope

        # Archive critical facts
        for key, value in session.critical_facts.items():
            entry = ContextEntry(
                key=f"archived_fact_{session.tenant_id}_{key}",
                value=value,
                layer=MemoryLayer.LONG_TERM,
                scope=ContextScope.TENANT if session.tenant_id else ContextScope.GLOBAL,
                priority="high",
                category="archived_fact",
                source=f"session_{session.session_id}",
                tenant_id=session.tenant_id,
                metadata={'original_session': session.session_id, 'archived_at': datetime.now(timezone.utc).isoformat()}
            )
            await self.coordinator.store_context(entry)

        # Archive session summary
        summary = {
            'session_id': session.session_id,
            'duration': (session.last_activity - session.start_time).total_seconds(),
            'message_count': len(session.conversation_history),
            'tasks_completed': len(session.completed_tasks),
            'agents_used': session.active_agents,
            'handoff_count': len(session.handoff_history),
            'summary': self._summarize_conversation(session)
        }

        entry = ContextEntry(
            key=f"session_summary_{session.session_id}",
            value=summary,
            layer=MemoryLayer.LONG_TERM,
            scope=ContextScope.TENANT if session.tenant_id else ContextScope.GLOBAL,
            priority="medium",
            category="session_summary",
            source="session_manager",
            tenant_id=session.tenant_id,
            metadata={'session_end_time': session.last_activity.isoformat()}
        )

        await self.coordinator.store_context(entry)

    async def _load_historical_context(self, session: SessionState):
        """Load historical context from previous sessions"""
        if not session.tenant_id:
            return

        # Get recent session summaries
        from memory_coordination_system import ContextScope, MemoryLayer

        results = await self.coordinator.search_context(
            query=f"tenant:{session.tenant_id}",
            scope=ContextScope.TENANT,
            layer=MemoryLayer.LONG_TERM,
            category="session_summary",
            tenant_id=session.tenant_id,
            limit=5
        )

        if results:
            session.long_term_refs = [r.key for r in results]

            # Load critical facts from recent sessions
            for result in results:
                if result.value.get('critical_facts'):
                    session.critical_facts.update(result.value['critical_facts'])

    async def _persist_handoff(self, session_id: str, handoff: AgentHandoff):
        """Persist handoff for audit trail"""
        from memory_coordination_system import ContextEntry, MemoryLayer, ContextScope

        entry = ContextEntry(
            key=f"handoff_{session_id}_{handoff.timestamp.isoformat()}",
            value=asdict(handoff),
            layer=MemoryLayer.SHORT_TERM,
            scope=ContextScope.SESSION,
            priority="high",
            category="agent_handoff",
            source="session_manager",
            session_id=session_id,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7)
        )

        await self.coordinator.store_context(entry)

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _summarize_conversation(self, session: SessionState) -> str:
        """Create a summary of the conversation"""
        if not session.conversation_history:
            return "No conversation yet"

        message_count = len(session.conversation_history)
        user_messages = sum(1 for m in session.conversation_history if m.get('role') == 'user')
        assistant_messages = sum(1 for m in session.conversation_history if m.get('role') == 'assistant')

        return f"{message_count} messages ({user_messages} from user, {assistant_messages} from assistant)"

    async def _extract_critical_facts(self, session: SessionState, content: str):
        """Extract critical facts from message content"""
        # Simple keyword-based extraction
        # In production, use LLM to extract facts

        keywords = {
            'customer_name': ['customer is', 'client is', 'working with'],
            'project_name': ['project called', 'project is', 'working on'],
            'deadline': ['deadline', 'due by', 'needs to be done by'],
            'budget': ['budget', 'cost', 'price']
        }

        content_lower = content.lower()

        for fact_type, triggers in keywords.items():
            for trigger in triggers:
                if trigger in content_lower:
                    # Extract the value (simplified)
                    idx = content_lower.find(trigger)
                    value_snippet = content[idx:idx+100]

                    if fact_type not in session.critical_facts:
                        session.critical_facts[fact_type] = value_snippet


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

async def get_session_manager(coordinator):
    """Get session manager instance"""
    return SessionContextManager(coordinator)


if __name__ == "__main__":
    # Test
    from memory_coordination_system import get_memory_coordinator

    async def test():
        coordinator = get_memory_coordinator()
        manager = await get_session_manager(coordinator)

        # Start session
        session = await manager.start_session(
            session_id="test_session_456",
            tenant_id="test_tenant",
            user_id="test_user"
        )
        print(f"✅ Started: {session.session_id}")

        # Add messages
        await manager.add_message(session.session_id, "user", "I need help with a project")
        await manager.add_message(session.session_id, "assistant", "I'd be happy to help!")

        # Add task
        await manager.add_task(session.session_id, {'id': 'task_1', 'description': 'Test task'})

        # Get full context
        ctx = await manager.get_full_context(session.session_id)
        print(f"✅ Context: {ctx['conversation']['message_count']} messages")

        # End session
        await manager.end_session(session.session_id)
        print("✅ Session ended")

    asyncio.run(test())

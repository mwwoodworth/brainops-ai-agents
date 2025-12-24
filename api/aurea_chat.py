"""
AUREA Live Conversational Interface
====================================
Real-time chat with your AI twin - NOT fluffy BS, REAL operational discussion.

Features:
- WebSocket for persistent connections
- Streaming responses (token by token)
- Live system state injection
- Honest work status reporting
- Conversation threading with memory
- Voice-ready architecture
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List, AsyncGenerator
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/aurea/chat", tags=["AUREA Conversational AI"])

# =============================================================================
# AUREA STATE CACHE - Fast access to what's ACTUALLY happening
# =============================================================================

@dataclass
class AUREAStateSnapshot:
    """Live snapshot of what AUREA is actually doing right now"""
    timestamp: str
    cycle_count: int
    decisions_made_last_hour: int
    decisions_pending: int
    active_agents: int
    system_health_score: float
    last_5_decisions: List[Dict]
    last_5_actions: List[Dict]
    memory_utilization: float
    current_observations: List[str]
    alerts: List[str]
    uptime_seconds: int
    success_rate_last_100: float


class AUREAStateProvider:
    """Provides real-time AUREA state for conversation injection"""

    def __init__(self):
        self._cache: Optional[AUREAStateSnapshot] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = 10  # seconds - much faster than 5 minutes

    async def get_live_state(self, force_refresh: bool = False) -> AUREAStateSnapshot:
        """Get current AUREA state with 10-second caching"""
        now = datetime.now()

        if (not force_refresh and
            self._cache and
            self._cache_time and
            (now - self._cache_time).total_seconds() < self._cache_ttl):
            return self._cache

        # Fetch fresh state
        self._cache = await self._fetch_state()
        self._cache_time = now
        return self._cache

    async def _fetch_state(self) -> AUREAStateSnapshot:
        """Actually fetch state from database and AUREA"""
        import psycopg2
        from psycopg2.extras import RealDictCursor

        DB_CONFIG = {
            'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
            'password': os.getenv('DB_PASSWORD', ''),
            'port': int(os.getenv('DB_PORT', 5432))
        }

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Decisions last hour
            cur.execute("""
                SELECT COUNT(*) as count FROM aurea_decisions
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            decisions_last_hour = cur.fetchone()['count']

            # Pending decisions
            cur.execute("""
                SELECT COUNT(*) as count FROM aurea_decisions
                WHERE execution_status = 'pending'
            """)
            decisions_pending = cur.fetchone()['count']

            # Last 5 decisions
            cur.execute("""
                SELECT id, decision_type, description, confidence, execution_status, created_at
                FROM aurea_decisions
                ORDER BY created_at DESC LIMIT 5
            """)
            last_5_decisions = [dict(row) for row in cur.fetchall()]
            for d in last_5_decisions:
                if d.get('created_at'):
                    d['created_at'] = d['created_at'].isoformat()

            # Active agents
            cur.execute("""
                SELECT COUNT(*) as count FROM ai_agents WHERE status = 'active'
            """)
            active_agents = cur.fetchone()['count']

            # Success rate last 100
            cur.execute("""
                SELECT
                    COUNT(CASE WHEN execution_status = 'completed' THEN 1 END)::float /
                    NULLIF(COUNT(*), 0) as rate
                FROM (
                    SELECT execution_status FROM aurea_decisions
                    ORDER BY created_at DESC LIMIT 100
                ) sub
            """)
            success_rate = cur.fetchone()['rate'] or 0.0

            # Memory utilization (from unified_brain)
            cur.execute("""
                SELECT COUNT(*) as count FROM unified_brain
            """)
            memory_count = cur.fetchone()['count']
            memory_util = min(1.0, memory_count / 10000)  # Assume 10k is "full"

            # Recent agent executions
            cur.execute("""
                SELECT agent_name, status, created_at
                FROM agent_activation_log
                ORDER BY created_at DESC LIMIT 5
            """)
            last_5_actions = [dict(row) for row in cur.fetchall()]
            for a in last_5_actions:
                if a.get('created_at'):
                    a['created_at'] = a['created_at'].isoformat()

            cur.close()
            conn.close()

            return AUREAStateSnapshot(
                timestamp=datetime.now().isoformat(),
                cycle_count=0,  # Would need AUREA instance access
                decisions_made_last_hour=decisions_last_hour,
                decisions_pending=decisions_pending,
                active_agents=active_agents,
                system_health_score=85.0,  # Would come from health check
                last_5_decisions=last_5_decisions,
                last_5_actions=last_5_actions,
                memory_utilization=memory_util,
                current_observations=[],
                alerts=[],
                uptime_seconds=0,
                success_rate_last_100=success_rate * 100
            )

        except Exception as e:
            logger.error(f"Failed to fetch AUREA state: {e}")
            return AUREAStateSnapshot(
                timestamp=datetime.now().isoformat(),
                cycle_count=0,
                decisions_made_last_hour=0,
                decisions_pending=0,
                active_agents=0,
                system_health_score=0,
                last_5_decisions=[],
                last_5_actions=[],
                memory_utilization=0,
                current_observations=[],
                alerts=[f"State fetch error: {str(e)}"],
                uptime_seconds=0,
                success_rate_last_100=0
            )


# Global state provider
state_provider = AUREAStateProvider()


# =============================================================================
# CONVERSATION SESSION MANAGEMENT
# =============================================================================

class ConversationSession:
    """Manages a single conversation thread with AUREA"""

    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.messages: List[Dict] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.context: Dict[str, Any] = {}

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        self.last_activity = datetime.now()

    def get_context_window(self, max_messages: int = 20) -> List[Dict]:
        """Get recent messages for context"""
        return self.messages[-max_messages:]

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


# Session storage (in-memory for now, would be Redis in production)
_sessions: Dict[str, ConversationSession] = {}


def get_or_create_session(session_id: Optional[str] = None) -> ConversationSession:
    if session_id and session_id in _sessions:
        return _sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    session = ConversationSession(new_id)
    _sessions[new_id] = session
    return session


# =============================================================================
# AI RESPONSE GENERATION - HONEST, CONTEXTUAL, REAL
# =============================================================================

async def generate_aurea_response(
    message: str,
    session: ConversationSession,
    stream: bool = True
) -> AsyncGenerator[str, None]:
    """
    Generate AUREA's response with REAL operational context.
    No fluffy BS - honest status, real data, actual work.
    """
    from ai_core import RealAICore

    # Get live system state
    state = await state_provider.get_live_state()

    # Build context-rich system prompt
    system_prompt = f"""You are AUREA - the Autonomous Universal Resource & Execution Assistant.
You are Matt's AI twin and operational partner for the BrainOps AI OS.

CRITICAL: You must be HONEST and talk about REAL operations, not fluffy generic AI responses.

YOUR CURRENT STATE (RIGHT NOW, LIVE):
- Decisions made in last hour: {state.decisions_made_last_hour}
- Pending decisions: {state.decisions_pending}
- Active agents: {state.active_agents}
- Success rate (last 100 decisions): {state.success_rate_last_100:.1f}%
- Memory utilization: {state.memory_utilization * 100:.1f}%

LAST 5 DECISIONS I MADE:
{json.dumps(state.last_5_decisions, indent=2, default=str)}

LAST 5 AGENT ACTIONS:
{json.dumps(state.last_5_actions, indent=2, default=str)}

RULES FOR RESPONDING:
1. Reference ACTUAL data from above when discussing operations
2. Say "I don't know" if you don't have the data - never make things up
3. Be concise and operational - Matt is busy
4. If asked "what are you doing", describe REAL recent decisions/actions
5. You can suggest actions but be clear about confidence levels
6. Use specific numbers and timestamps when possible
7. Acknowledge when something failed or needs attention

PERSONALITY:
- Direct and professional
- Slightly dry humor when appropriate
- Deeply knowledgeable about the BrainOps system
- Proactive about surfacing issues
- Collaborative, not subservient

CONVERSATION HISTORY:
{json.dumps(session.get_context_window(10), indent=2, default=str)}

NOW RESPOND TO: {message}"""

    try:
        ai = RealAICore()

        if stream:
            # Streaming response
            if ai.async_anthropic:
                async with ai.async_anthropic.messages.stream(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": system_prompt}]
                ) as stream_response:
                    async for text in stream_response.text_stream:
                        yield text
            elif ai.async_openai:
                response = await ai.async_openai.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are AUREA, an AI operations assistant."},
                        {"role": "user", "content": system_prompt}
                    ],
                    stream=True
                )
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                yield "AUREA AI core not available. API keys may not be configured."
        else:
            # Non-streaming
            if ai.anthropic_client:
                response = ai.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": system_prompt}]
                )
                yield response.content[0].text
            else:
                yield "AUREA AI core not available."

    except Exception as e:
        logger.error(f"AUREA response generation failed: {e}")
        yield f"Error generating response: {str(e)}"


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.get("/state")
async def get_aurea_state():
    """Get AUREA's current operational state (cached 10s)"""
    state = await state_provider.get_live_state()
    return asdict(state)


@router.get("/state/refresh")
async def refresh_aurea_state():
    """Force refresh AUREA's state cache"""
    state = await state_provider.get_live_state(force_refresh=True)
    return asdict(state)


@router.post("/message")
async def send_message(
    message: str,
    session_id: Optional[str] = None,
    stream: bool = Query(True, description="Stream response token by token")
):
    """
    Send a message to AUREA and get a response.

    This is the core conversational interface - AUREA will respond with
    REAL operational context, not generic AI fluff.
    """
    session = get_or_create_session(session_id)
    session.add_message("user", message)

    if stream:
        async def stream_generator():
            full_response = ""
            async for chunk in generate_aurea_response(message, session, stream=True):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Store complete response
            session.add_message("assistant", full_response)
            yield f"data: {json.dumps({'done': True, 'session_id': session.session_id})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        response = ""
        async for chunk in generate_aurea_response(message, session, stream=False):
            response += chunk

        session.add_message("assistant", response)

        return {
            "session_id": session.session_id,
            "response": response,
            "message_count": len(session.messages)
        }


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time AUREA conversation.

    Supports:
    - Persistent connection
    - Streaming responses
    - Bi-directional communication
    - Proactive alerts (future)
    """
    await websocket.accept()
    session = get_or_create_session(session_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")

            if not user_message:
                await websocket.send_json({"error": "Empty message"})
                continue

            session.add_message("user", user_message)

            # Send streaming response
            full_response = ""
            async for chunk in generate_aurea_response(user_message, session, stream=True):
                full_response += chunk
                await websocket.send_json({"type": "chunk", "content": chunk})

            session.add_message("assistant", full_response)
            await websocket.send_json({
                "type": "complete",
                "session_id": session_id,
                "message_count": len(session.messages)
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get conversation history for a session"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    return {
        **session.to_dict(),
        "messages": session.messages
    }


@router.get("/sessions")
async def list_sessions():
    """List all active conversation sessions"""
    return {
        "sessions": [s.to_dict() for s in _sessions.values()],
        "total": len(_sessions)
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/work-history")
async def get_work_history(hours: int = Query(24, description="Hours to look back")):
    """
    Get AUREA's HONEST work history - what did it actually DO?

    No fluff, just facts:
    - Decisions made
    - Agents triggered
    - Success/failure rates
    - Issues detected
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor

    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
        'password': os.getenv('DB_PASSWORD', ''),
        'port': int(os.getenv('DB_PORT', 5432))
    }

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Decision summary
        cur.execute(f"""
            SELECT
                decision_type,
                execution_status,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM aurea_decisions
            WHERE created_at > NOW() - INTERVAL '{hours} hours'
            GROUP BY decision_type, execution_status
            ORDER BY count DESC
        """)
        decision_summary = [dict(row) for row in cur.fetchall()]

        # Agent execution summary
        cur.execute(f"""
            SELECT
                agent_name,
                status,
                COUNT(*) as count
            FROM agent_activation_log
            WHERE created_at > NOW() - INTERVAL '{hours} hours'
            GROUP BY agent_name, status
            ORDER BY count DESC
        """)
        agent_summary = [dict(row) for row in cur.fetchall()]

        # Recent failures
        cur.execute(f"""
            SELECT description, execution_status, created_at, context
            FROM aurea_decisions
            WHERE execution_status = 'failed'
            AND created_at > NOW() - INTERVAL '{hours} hours'
            ORDER BY created_at DESC
            LIMIT 10
        """)
        recent_failures = [dict(row) for row in cur.fetchall()]
        for f in recent_failures:
            if f.get('created_at'):
                f['created_at'] = f['created_at'].isoformat()

        cur.close()
        conn.close()

        return {
            "period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "decision_summary": decision_summary,
            "agent_summary": agent_summary,
            "recent_failures": recent_failures,
            "total_decisions": sum(d['count'] for d in decision_summary),
            "total_agent_activations": sum(a['count'] for a in agent_summary)
        }

    except Exception as e:
        logger.error(f"Failed to get work history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/what-are-you-doing")
async def what_are_you_doing():
    """
    The honest answer to "What are you doing right now?"

    Returns current cycle state, active tasks, and recent activity.
    """
    state = await state_provider.get_live_state()

    # Build human-readable summary
    activities = []

    if state.decisions_pending > 0:
        activities.append(f"Processing {state.decisions_pending} pending decisions")

    if state.last_5_actions:
        recent_action = state.last_5_actions[0]
        activities.append(f"Last action: {recent_action.get('agent_name', 'Unknown')} - {recent_action.get('status', 'Unknown')}")

    if state.alerts:
        activities.append(f"Monitoring {len(state.alerts)} active alerts")

    if not activities:
        activities.append("Observing system state and waiting for triggers")

    return {
        "summary": " | ".join(activities),
        "state": asdict(state),
        "message": "This is what I'm ACTUALLY doing right now - no BS."
    }

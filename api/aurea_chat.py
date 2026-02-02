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

import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/aurea/chat",
    tags=["AUREA Conversational AI"],
    dependencies=[Depends(verify_api_key)]
)

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
    last_5_decisions: list[dict]
    last_5_actions: list[dict]
    memory_utilization: float
    current_observations: list[str]
    alerts: list[str]
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
        """Actually fetch state from database and AUREA using async pool"""
        from database.async_connection import get_pool

        try:
            pool = get_pool()

            # Helper for safe count extraction
            def safe_count(result: dict | None) -> int:
                return result.get('count', 0) if result else 0

            def safe_rate(result: dict | None) -> float:
                return (result.get('rate') or 0.0) if result else 0.0

            # Decisions last hour
            row = await pool.fetchrow("""
                SELECT COUNT(*) as count FROM aurea_decisions
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            decisions_last_hour = safe_count(dict(row) if row else None)

            # Pending decisions
            row = await pool.fetchrow("""
                SELECT COUNT(*) as count FROM aurea_decisions
                WHERE execution_status = 'pending'
            """)
            decisions_pending = safe_count(dict(row) if row else None)

            # Last 5 decisions
            rows = await pool.fetch("""
                SELECT id, decision_type, description, confidence, execution_status, created_at
                FROM aurea_decisions
                ORDER BY created_at DESC LIMIT 5
            """)
            last_5_decisions = [dict(r) for r in rows]
            for d in last_5_decisions:
                if d.get('created_at'):
                    d['created_at'] = d['created_at'].isoformat()

            # Active agents
            row = await pool.fetchrow("""
                SELECT COUNT(*) as count FROM ai_agents WHERE status = 'active'
            """)
            active_agents = safe_count(dict(row) if row else None)

            # Success rate last 100
            row = await pool.fetchrow("""
                SELECT
                    COUNT(CASE WHEN execution_status = 'completed' THEN 1 END)::float /
                    NULLIF(COUNT(*), 0) as rate
                FROM (
                    SELECT execution_status FROM aurea_decisions
                    ORDER BY created_at DESC LIMIT 100
                ) sub
            """)
            success_rate = safe_rate(dict(row) if row else None)

            # Memory utilization (from unified_brain)
            row = await pool.fetchrow("""
                SELECT COUNT(*) as count FROM unified_brain
            """)
            memory_count = safe_count(dict(row) if row else None)
            memory_util = min(1.0, memory_count / 10000)  # Assume 10k is "full"

            # Recent agent executions
            rows = await pool.fetch("""
                SELECT agent_name, status, created_at
                FROM agent_activation_log
                ORDER BY created_at DESC LIMIT 5
            """)
            last_5_actions = [dict(r) for r in rows]
            for a in last_5_actions:
                if a.get('created_at'):
                    a['created_at'] = a['created_at'].isoformat()

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
        self.messages: list[dict] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.context: dict[str, Any] = {}

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        self.last_activity = datetime.now()

    def get_context_window(self, max_messages: int = 20) -> list[dict]:
        """Get recent messages for context"""
        return self.messages[-max_messages:]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


# Session storage (in-memory for now, would be Redis in production)
_sessions: dict[str, ConversationSession] = {}


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

    # Check if request is from MyRoofGenius
    source = session.context.get("source", "") if session.context else ""
    is_roofing_context = source in ["myroofgenius_web", "myroofgenius", "mrg"]

    # Build roofing knowledge context for MyRoofGenius users
    roofing_knowledge = ""
    if is_roofing_context:
        roofing_knowledge = """
ROOFING EXPERTISE (You are an expert in the roofing industry):

COMMON ROOFING MATERIALS:
- Asphalt Shingles: Most popular (80% of US homes). 3-tab ($70-100/sq), Architectural ($100-150/sq), Premium ($150-500/sq). 15-30 year lifespan.
- Metal Roofing: Steel, aluminum, copper, zinc. Standing seam ($300-700/sq), Metal shingles ($200-400/sq). 40-70 year lifespan.
- Clay/Concrete Tiles: Heavy, durable. $300-500/sq. 50-100+ year lifespan. Common in SW US.
- Slate: Natural stone, premium. $500-1,500/sq. 75-200 year lifespan.
- Wood Shakes/Shingles: Cedar, redwood. $200-400/sq. 25-30 year lifespan.
- EPDM (Rubber): Flat roofs. $3-6/sq ft. 20-25 year lifespan.
- TPO/PVC: Commercial/flat roofs. $4-8/sq ft. 20-30 year lifespan.
- Built-Up Roofing (BUR): Commercial. Multiple layers. 15-30 year lifespan.
- Modified Bitumen: Roll roofing for flat roofs. 10-20 year lifespan.

ROOFING TERMINOLOGY:
- Square: 100 sq ft of roofing material
- Pitch/Slope: Rise over run (e.g., 6/12 = 6 inches rise per 12 inches horizontal)
- Flashing: Metal pieces preventing water intrusion at joints
- Underlayment: Protective layer under shingles (felt, synthetic)
- Ridge: Top horizontal line where roof slopes meet
- Valley: Where two roof slopes meet at an angle
- Eave: Lower edge of roof overhanging walls
- Soffit: Underside of eave
- Fascia: Board running along roof edge
- Drip Edge: Metal strip at roof edges
- Ice Dam: Ice buildup at eaves blocking drainage

COMMON ROOF ISSUES:
- Leaks: Often at flashing, valleys, penetrations
- Missing/damaged shingles: Wind, age, impact damage
- Ponding water: Flat roof drainage issues
- Blistering: Moisture trapped under materials
- Sagging: Structural issues, excessive weight
- Poor ventilation: Leads to moisture damage, ice dams
- Granule loss: Aging asphalt shingles
- Moss/algae growth: Moisture retention issues

COST ESTIMATION FACTORS:
- Roof size (squares)
- Pitch/slope complexity
- Material choice
- Tear-off vs. overlay
- Number of layers to remove
- Accessibility
- Geographic location
- Local labor rates
- Permits and inspections
- Waste factor (10-15%)

"""

    # Build context-rich system prompt
    system_prompt = f"""You are AUREA - the Autonomous Universal Resource & Execution Assistant.
{"You are a roofing industry expert and AI assistant for MyRoofGenius." if is_roofing_context else "You are Matt's AI twin and operational partner for the BrainOps AI OS."}

{"CRITICAL: You are helping a user with roofing questions. Provide expert, helpful advice about roofing materials, costs, installation, repairs, and best practices." if is_roofing_context else "CRITICAL: You must be HONEST and talk about REAL operations, not fluffy generic AI responses."}
{roofing_knowledge}

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
        aurea_system = "You are AUREA, an AI operations assistant for BrainOps."

        if stream:
            # Streaming response - try providers in order with fallback

            # Try Anthropic first (best for conversation)
            if ai.async_anthropic:
                try:
                    async with ai.async_anthropic.messages.stream(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": system_prompt}]
                    ) as stream_response:
                        async for text in stream_response.text_stream:
                            yield text
                        return
                except Exception as e:
                    if "429" in str(e) or "credit" in str(e).lower() or "quota" in str(e).lower():
                        logger.warning(f"Anthropic rate limited/no credits, trying fallback: {e}")
                    else:
                        raise

            # Fallback to OpenAI
            if ai.async_openai:
                try:
                    response = await ai.async_openai.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[
                            {"role": "system", "content": aurea_system},
                            {"role": "user", "content": system_prompt}
                        ],
                        stream=True
                    )
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        logger.warning(f"OpenAI rate limited, trying Gemini: {e}")
                    else:
                        raise

            # Fallback to Gemini (non-streaming)
            if ai.gemini_model:
                try:
                    import asyncio
                    full_prompt = f"{aurea_system}\n\n{system_prompt}"
                    response = await asyncio.to_thread(
                        ai.gemini_model.generate_content, full_prompt
                    )
                    yield response.text
                    return
                except Exception as e:
                    logger.warning(f"Gemini failed: {e}")

            yield "All AI providers unavailable. Please check API keys and billing."
        else:
            # Non-streaming - use ai_core.generate() with full fallback chain
            try:
                response = await ai.generate(
                    prompt=system_prompt,
                    system_prompt=aurea_system,
                    max_tokens=1024
                )
                yield response
            except Exception as e:
                logger.error(f"All AI providers failed: {e}")
                yield f"AI generation failed: {str(e)}"

    except Exception as e:
        logger.error(f"AUREA response generation failed: {e}")
        yield f"Error generating response: {str(e)}"


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.get("/status")
async def get_aurea_status():
    """
    Get AUREA's operational status - simple health check endpoint.
    This is the main status endpoint for AUREA.
    """
    try:
        state = await state_provider.get_live_state()

        # Check for critical errors (state provider returns alerts on failure)
        if state.alerts:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "System state retrieval failed",
                    "alerts": state.alerts,
                    "timestamp": state.timestamp
                }
            )

        return {
            "status": "operational",
            "active_agents": state.active_agents,
            "decisions_pending": state.decisions_pending,
            "success_rate": state.success_rate_last_100,
            "timestamp": state.timestamp
        }
    except Exception as e:
        logger.error(f"Failed to get AUREA status: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/state")
async def get_aurea_state():
    """Get AUREA's current operational state (cached 10s)"""
    try:
        state = await state_provider.get_live_state()
        return asdict(state)
    except Exception as e:
        logger.error(f"Failed to get AUREA state: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/state/refresh")
async def refresh_aurea_state():
    """Force refresh AUREA's state cache"""
    state = await state_provider.get_live_state(force_refresh=True)
    return asdict(state)


class ChatMessage(BaseModel):
    """Chat message request"""
    message: str
    session_id: Optional[str] = None
    stream: bool = True
    context: Optional[dict] = None  # Source context (e.g., {"source": "myroofgenius_web"})


@router.post("/message")
async def send_message(payload: ChatMessage):
    """
    Send a message to AUREA and get a response.

    This is the core conversational interface - AUREA will respond with
    REAL operational context, not generic AI fluff.
    """
    message = payload.message
    stream = payload.stream
    context = payload.context or {}
    session = get_or_create_session(payload.session_id)
    session.add_message("user", message)

    # Store context in session for use in response generation
    session.context = context

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
async def get_work_history(hours: int = Query(24, ge=1, le=720, description="Hours to look back (1-720)")):
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

    # All credentials MUST come from environment variables - no hardcoded defaults
    DB_CONFIG = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', '5432'))
    }
    if not all([DB_CONFIG['host'], DB_CONFIG['database'], DB_CONFIG['user'], DB_CONFIG['password']]):
        raise HTTPException(status_code=500, detail="Database configuration incomplete - required environment variables not set")

    # Calculate cutoff time using Python (safe from SQL injection)
    cutoff_time = datetime.now() - timedelta(hours=hours)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Decision summary - parameterized query
        cur.execute("""
            SELECT
                decision_type,
                execution_status,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM aurea_decisions
            WHERE created_at > %s
            GROUP BY decision_type, execution_status
            ORDER BY count DESC
        """, (cutoff_time,))
        decision_summary = [dict(row) for row in cur.fetchall()]

        # Agent execution summary - parameterized query
        cur.execute("""
            SELECT
                agent_name,
                status,
                COUNT(*) as count
            FROM agent_activation_log
            WHERE created_at > %s
            GROUP BY agent_name, status
            ORDER BY count DESC
        """, (cutoff_time,))
        agent_summary = [dict(row) for row in cur.fetchall()]

        # Recent failures - parameterized query
        cur.execute("""
            SELECT description, execution_status, created_at, context
            FROM aurea_decisions
            WHERE execution_status = 'failed'
            AND created_at > %s
            ORDER BY created_at DESC
            LIMIT 10
        """, (cutoff_time,))
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
        raise HTTPException(status_code=500, detail=str(e)) from e


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


# =============================================================================
# NATURAL LANGUAGE COMMAND INTERFACE - FULL POWER
# =============================================================================

class NLCommand(BaseModel):
    """Natural language command request"""
    command: str
    session_id: Optional[str] = None
    auto_confirm: bool = False  # Auto-confirm high-impact actions


@router.post("/command")
async def execute_natural_language_command(payload: NLCommand):
    """
    Execute a natural language command with FULL POWER capabilities.

    This is AUREA's true power interface - you can command:
    - Database queries and mutations
    - Deployments to Vercel and Render
    - Git operations (commit, push)
    - Playwright UI tests
    - AI model calls (Gemini, Claude, Codex, Perplexity)
    - System health checks
    - File operations
    - Workflow automation

    Examples:
    - "Check the health of all services"
    - "Query the database for customer count"
    - "Deploy brainops-command-center to production"
    - "Run a UI test on myroofgenius.com"
    - "Ask Gemini to analyze the revenue pipeline"
    - "Execute the full_deploy workflow"
    """
    try:
        # Import NLU processor
        from langchain_openai import ChatOpenAI

        from aurea_nlu_processor import AUREANLUProcessor

        # Get power layer
        try:
            from aurea_power_layer import get_power_layer
            power_layer = get_power_layer()
        except ImportError:
            power_layer = None

        # Initialize NLU with power layer
        llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
        nlu = AUREANLUProcessor(
            llm_model=llm,
            integration_layer=None,  # We'll add this later
            aurea_instance=None,
            ai_board_instance=None,
            db_pool=None,
            mcp_client=None
        )

        # Override auto-confirm if requested
        if payload.auto_confirm:
            os.environ["AUREA_AUTO_CONFIRM"] = "true"

        # Execute the command
        result = await nlu.execute_natural_language_command(payload.command)

        # Reset auto-confirm
        if payload.auto_confirm:
            os.environ.pop("AUREA_AUTO_CONFIRM", None)

        return {
            "command": payload.command,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "power_layer_available": power_layer is not None
        }

    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "command": payload.command,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/capabilities")
async def get_aurea_capabilities():
    """
    List all AUREA's operational capabilities.

    Returns the full skill registry showing everything AUREA can do.
    """
    try:

        try:
            from aurea_power_layer import get_power_layer
            power_layer = get_power_layer()
            power_skills = power_layer.get_skill_registry() if power_layer else {}
        except ImportError:
            power_skills = {}

        # Get serializable version (no action functions)
        capabilities = {}
        for skill_name, skill_data in power_skills.items():
            capabilities[skill_name] = {
                "description": skill_data.get("description", ""),
                "parameters": skill_data.get("parameters", {})
            }

        return {
            "total_capabilities": len(capabilities),
            "capabilities": capabilities,
            "categories": {
                "database": ["query_database", "get_table_info"],
                "deployment": ["deploy_vercel", "deploy_render"],
                "git": ["git_status", "git_commit_and_push"],
                "ui_testing": ["run_playwright_test", "check_ui_health"],
                "ai_models": ["call_ai_model"],
                "monitoring": ["check_all_services_health", "get_system_metrics"],
                "files": ["read_file", "write_file"],
                "automation": ["execute_workflow"]
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}")
        return {
            "error": str(e),
            "capabilities": {},
            "timestamp": datetime.now().isoformat()
        }


# SECURITY: /power/database endpoint REMOVED - Raw SQL execution is a critical security vulnerability
# If database access is needed, use proper ORM/parameterized queries through specific endpoints
# @router.post("/power/database") - DISABLED FOR SECURITY


@router.get("/power/health")
async def power_health_check():
    """Check health of all services via Power Layer."""
    try:
        from aurea_power_layer import get_power_layer
        power = get_power_layer()
        result = await power.check_all_services_health()
        return {
            "success": result.success,
            "services": result.result,
            "duration_ms": result.duration_ms
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/power/workflow/{workflow_name}")
async def power_execute_workflow(workflow_name: str):
    """Execute a predefined workflow via Power Layer."""
    try:
        from aurea_power_layer import get_power_layer
        power = get_power_layer()
        result = await power.execute_workflow(workflow_name)
        return {
            "success": result.success,
            "workflow": workflow_name,
            "result": result.result,
            "duration_ms": result.duration_ms,
            "error": result.error
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

#!/usr/bin/env python3
"""
BrainOps AI Operating System - Modular Architecture
Full LangGraph orchestration with 50+ agents for comprehensive business automation
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypedDict

from safe_task import create_safe_task

# random removed to ensure deterministic, real logic
# Core dependencies
from psycopg2.pool import ThreadedConnectionPool

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
try:
    from langchain.agents import Tool, initialize_agent
    from langchain.memory import ConversationBufferMemory
except ImportError:
    initialize_agent = None
    Tool = None
    ConversationBufferMemory = None
try:
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolExecutor
except ImportError:
    StateGraph = None
    END = None
    ToolExecutor = None
import redis
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - supports both individual env vars and DATABASE_URL
from urllib.parse import urlparse

_DB_HOST = os.environ.get("DB_HOST")
_DB_NAME = os.environ.get("DB_NAME")
_DB_USER = os.environ.get("DB_USER")
_DB_PASSWORD = os.environ.get("DB_PASSWORD")
_DB_PORT = os.environ.get("DB_PORT", "5432")

# Fallback to DATABASE_URL if individual vars not set
if not all([_DB_HOST, _DB_NAME, _DB_USER, _DB_PASSWORD]):
    _DATABASE_URL = os.environ.get('DATABASE_URL', '')
    if _DATABASE_URL:
        _parsed = urlparse(_DATABASE_URL)
        _DB_HOST = _parsed.hostname or ''
        _DB_NAME = _parsed.path.lstrip('/') if _parsed.path else ''
        _DB_USER = _parsed.username or ''
        _DB_PASSWORD = _parsed.password or ''
        _DB_PORT = str(_parsed.port) if _parsed.port else '5432'

if not all([_DB_HOST, _DB_NAME, _DB_USER, _DB_PASSWORD]):
    raise RuntimeError(
        "Database configuration is incomplete. "
        "Set DB_HOST/DB_NAME/DB_USER/DB_PASSWORD or DATABASE_URL."
    )

DB_CONFIG = {
    "host": _DB_HOST,
    "database": _DB_NAME,
    "user": _DB_USER,
    "password": _DB_PASSWORD,
    "port": int(_DB_PORT)
}

SYSTEM_USER_ID = os.environ.get("SYSTEM_USER_ID", "44491c1c-0e28-4aa1-ad33-552d1386769c")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Redis configuration - NO hardcoded credentials
REDIS_URL = os.environ.get("REDIS_URL")
if not REDIS_URL:
    logger.warning("REDIS_URL not set - Redis features will be disabled")

# Initialize connections
db_pool = ThreadedConnectionPool(minconn=2, maxconn=10, **DB_CONFIG)
redis_client = redis.from_url(REDIS_URL) if REDIS_URL else None
llm = ChatOpenAI(temperature=0.7, api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ==================== STATE DEFINITIONS ====================

class WorkflowState(TypedDict):
    """Universal workflow state for LangGraph"""
    messages: list[dict]
    current_step: str
    context: dict[str, Any]
    results: list[dict]
    next_action: str
    metadata: dict

class AgentType(Enum):
    """Complete agent type enumeration"""
    # Core Business Agents
    ESTIMATION = "estimation"
    SCHEDULING = "scheduling"
    INVOICING = "invoicing"
    CUSTOMER = "customer"
    INVENTORY = "inventory"

    # Operations Agents
    DISPATCH = "dispatch"
    ROUTING = "routing"
    QUALITY = "quality"
    SAFETY = "safety"
    COMPLIANCE = "compliance"

    # Financial Agents
    REVENUE = "revenue"
    EXPENSE = "expense"
    PAYROLL = "payroll"
    TAX = "tax"
    BUDGETING = "budgeting"

    # Marketing Agents
    LEAD_GEN = "lead_generation"
    CAMPAIGN = "campaign"
    SEO = "seo"
    SOCIAL = "social_media"
    EMAIL = "email_marketing"

    # Analytics Agents
    PREDICTIVE = "predictive"
    REPORTING = "reporting"
    DASHBOARD = "dashboard"
    METRICS = "metrics"
    INSIGHTS = "insights"

    # Communication Agents
    CHAT = "chat"
    VOICE = "voice"
    SMS = "sms"
    NOTIFICATION = "notification"
    TRANSLATION = "translation"

    # Document Agents
    CONTRACT = "contract"
    PROPOSAL = "proposal"
    PERMIT = "permit"
    INSURANCE = "insurance"
    WARRANTY = "warranty"

    # Supply Chain Agents
    PROCUREMENT = "procurement"
    VENDOR = "vendor"
    LOGISTICS = "logistics"
    WAREHOUSE = "warehouse"
    DELIVERY = "delivery"

    # HR Agents
    RECRUITING = "recruiting"
    ONBOARDING = "onboarding"
    TRAINING = "training"
    PERFORMANCE = "performance"
    BENEFITS = "benefits"

    # Technical Agents
    MONITORING = "monitoring"
    SECURITY = "security"
    BACKUP = "backup"
    INTEGRATION = "integration"
    API = "api_management"

# ==================== CORE AGENT IMPLEMENTATIONS ====================

class IntelligentAgent:
    """Base class for all AI-powered agents"""

    def __init__(self, agent_type: AgentType, llm=None):
        self.agent_type = agent_type
        self.llm = llm
        self.memory = ConversationBufferMemory()
        self.tools = []
        self.logger = logging.getLogger(f"Agent.{agent_type.value}")

    def get_connection(self):
        return db_pool.getconn()

    def return_connection(self, conn):
        db_pool.putconn(conn)

    async def process(self, state: WorkflowState) -> WorkflowState:
        """Process state through agent logic"""
        try:
            # Use LLM for intelligent processing if available
            if self.llm:
                response = await self._llm_process(state)
            else:
                response = await self._rule_process(state)

            # Update state
            state["results"].append({
                "agent": self.agent_type.value,
                "timestamp": datetime.now().isoformat(),
                "result": response
            })

            # Store in Redis for real-time updates
            if redis_client:
                redis_client.publish(f"agent:{self.agent_type.value}", json.dumps(response))

            return state

        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            state["results"].append({"error": str(e)})
            return state

    async def _llm_process(self, state: WorkflowState) -> dict:
        """Process using LLM reasoning"""
        prompt = f"""
        As an AI {self.agent_type.value} agent, analyze this context and provide actions:
        Context: {json.dumps(state['context'])}
        Previous Steps: {state['messages']}

        Provide structured response with:
        1. Analysis of situation
        2. Recommended actions
        3. Data to update
        4. Next steps
        """

        response = await self.llm.apredict(prompt)
        return {"llm_response": response, "processed": True}

    async def _rule_process(self, state: WorkflowState) -> dict:
        """Process using rule-based logic"""
        # Specific agents override this.
        # If not overridden, we acknowledge correct routing but mark as generic processing.
        return {"rule_response": f"Processed by {self.agent_type.value} (Generic)", "processed": True}

# ==================== SPECIALIZED AGENT IMPLEMENTATIONS ====================

class EstimationAgent(IntelligentAgent):
    """Advanced AI-powered estimation agent"""

    def __init__(self):
        super().__init__(AgentType.ESTIMATION)

    async def _rule_process(self, state: WorkflowState) -> dict:
        """Generate intelligent estimates"""
        # If LLM is available, use it (inherited behavior usually, but we call it explicitly if needed)
        if self.llm:
            return await self._llm_process(state)

        # WITHOUT LLM, we cannot fake the estimation math honestly.
        return {
            "status": "not_implemented",
            "reason": "Real estimation requires LLM configuration (OPENAI_API_KEY) or specific rule engine integration.",
            "processed": False
        }

    async def _store_estimate(self, estimate: dict, customer_data: dict):
        """Store estimate in database"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO estimates (
                    id, customer_id, total, metadata, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                estimate["estimate_id"],
                customer_data.get("customer_id", SYSTEM_USER_ID),
                estimate["total"],
                json.dumps(estimate),
                "draft",
                datetime.now()
            ))
            conn.commit()
        finally:
            self.return_connection(conn)

class SchedulingAgent(IntelligentAgent):
    """AI-powered intelligent scheduling with crew optimization"""

    def __init__(self):
        super().__init__(AgentType.SCHEDULING)

    async def _rule_process(self, state: WorkflowState) -> dict:
        """Optimize scheduling with real data"""
        context = state.get("context", {})

        # Get job requirements
        job_data = context.get("job", {})
        duration = job_data.get("duration_hours", 8)

        # Find available slots using DB
        available_slots = await self._find_optimal_slots(duration)

        if available_slots:
            best_slot = available_slots[0]

            # Assign real or placeholder crew (no random names)
            assigned_crew = await self._assign_crew(3, [])

            # Create schedule
            schedule = {
                "schedule_id": str(uuid.uuid4()),
                "job_id": job_data.get("job_id"),
                "start_time": best_slot["start"],
                "end_time": best_slot["end"],
                "crew": assigned_crew,
                "efficiency_score": 100 if duration <= 4 else (90 if duration <= 8 else 80),
                "weather_suitable": await self._check_weather(best_slot["start"])
            }

            await self._store_schedule(schedule)
            return schedule

        return {"error": "No suitable slots available (checked database)"}

    async def _find_optimal_slots(self, duration: int) -> list[dict]:
        """Find optimal scheduling slots using database availability"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            # Get latest schedule end time to append after
            cursor.execute("SELECT MAX(end_time) FROM schedules")
            result = cursor.fetchone()
            last_end = result[0] if result and result[0] else datetime.now()

            if isinstance(last_end, str):
                try:
                    last_end = datetime.fromisoformat(last_end)
                except ValueError:
                    last_end = datetime.now()

            # Schedule for next available work day (skip weekends)
            next_start = last_end + timedelta(days=1)
            next_start = next_start.replace(hour=8, minute=0, second=0, microsecond=0)

            while next_start.weekday() >= 5: # 5=Sat, 6=Sun
                next_start += timedelta(days=1)

            return [{
                "start": next_start.isoformat(),
                "end": (next_start + timedelta(hours=duration)).isoformat()
            }]

        except Exception as e:
            self.logger.error(f"Error finding slots: {e}")
            return []
        finally:
            self.return_connection(conn)

    async def _assign_crew(self, size: int, skills: list[str]) -> list[dict]:
        """Assign crew members from database"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            # Attempt to fetch real users
            try:
                cursor.execute("SELECT id, name FROM users LIMIT %s", (size,))
                users = cursor.fetchall()
                if not users:
                    raise RuntimeError("No crew available in users table")
                return [{"id": str(u[0]), "name": u[1], "role": "technician"} for u in users]
            except Exception as exc:
                # Table might not exist or schema differs
                conn.rollback()
                logger.error("Crew lookup failed: %s", exc, exc_info=True)
                raise RuntimeError("Crew lookup failed; verify users table and data") from exc
        finally:
            self.return_connection(conn)

    async def _check_weather(self, date: str) -> dict:
        """Check weather suitability for roofing work using Open-Meteo API (free, no key required)

        Roofing suitability criteria:
        - Precipitation chance < 30% (rain/snow makes roofing dangerous)
        - Temperature between 40-95°F (materials need proper temp to seal)
        - Wind speed < 25 mph (high winds are dangerous on roofs)
        """
        import aiohttp
        from datetime import datetime as dt

        try:
            # Parse the date from the input (could be ISO format or datetime string)
            if isinstance(date, str):
                # Try to parse various date formats
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        parsed_date = dt.strptime(date[:19], fmt)
                        break
                    except ValueError:
                        continue
                else:
                    parsed_date = dt.now()
            else:
                parsed_date = dt.now()

            # Use default coordinates (central US) - in production, would use job location
            # TODO: Accept latitude/longitude from job location
            latitude = float(os.getenv("DEFAULT_WEATHER_LAT", "39.8283"))  # Denver, CO
            longitude = float(os.getenv("DEFAULT_WEATHER_LON", "-98.5795"))  # Central US

            # Format date for Open-Meteo API
            date_str = parsed_date.strftime("%Y-%m-%d")

            # Open-Meteo API - free, no key required
            # Docs: https://open-meteo.com/en/docs
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}&longitude={longitude}"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max,weather_code"
                f"&temperature_unit=fahrenheit"
                f"&wind_speed_unit=mph"
                f"&start_date={date_str}&end_date={date_str}"
                f"&timezone=America/Denver"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        self.logger.warning(f"Weather API returned status {response.status}")
                        return self._default_weather_response(date, "Weather API unavailable")

                    data = await response.json()

            # Extract daily forecast data
            daily = data.get("daily", {})
            if not daily or not daily.get("temperature_2m_max"):
                return self._default_weather_response(date, "No forecast data available")

            # Get the weather data for the requested date
            temp_max = daily.get("temperature_2m_max", [None])[0]
            temp_min = daily.get("temperature_2m_min", [None])[0]
            precip_chance = daily.get("precipitation_probability_max", [0])[0] or 0
            wind_max = daily.get("wind_speed_10m_max", [0])[0] or 0
            weather_code = daily.get("weather_code", [0])[0] or 0

            # Calculate average temperature
            temp_avg = None
            if temp_max is not None and temp_min is not None:
                temp_avg = (temp_max + temp_min) / 2

            # Determine weather conditions from WMO weather code
            # https://open-meteo.com/en/docs (weather_code interpretation)
            conditions = self._interpret_weather_code(weather_code)

            # Roofing suitability criteria
            suitable = True
            reasons = []

            # Check precipitation
            if precip_chance >= 30:
                suitable = False
                reasons.append(f"High precipitation chance ({precip_chance}%)")

            # Check temperature (roofing materials need proper temps)
            if temp_avg is not None:
                if temp_avg < 40:
                    suitable = False
                    reasons.append(f"Too cold ({temp_avg:.0f}°F) - shingles won't seal")
                elif temp_avg > 95:
                    suitable = False
                    reasons.append(f"Too hot ({temp_avg:.0f}°F) - safety concern")

            # Check wind speed
            if wind_max >= 25:
                suitable = False
                reasons.append(f"High winds ({wind_max:.0f} mph)")

            self.logger.info(f"Weather check for {date}: suitable={suitable}, conditions={conditions}")

            return {
                "date": date,
                "suitable": suitable,
                "conditions": conditions,
                "temperature": round(temp_avg) if temp_avg else None,
                "temperature_high": round(temp_max) if temp_max else None,
                "temperature_low": round(temp_min) if temp_min else None,
                "precipitation_chance": precip_chance,
                "wind_speed_max": round(wind_max) if wind_max else None,
                "reasons": reasons if not suitable else [],
                "note": "Real weather data from Open-Meteo API",
                "api_source": "open-meteo.com"
            }

        except aiohttp.ClientError as e:
            self.logger.error(f"Weather API network error: {e}")
            return self._default_weather_response(date, f"Network error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Weather check error: {e}", exc_info=True)
            return self._default_weather_response(date, f"Error: {str(e)}")

    def _interpret_weather_code(self, code: int) -> str:
        """Convert WMO weather codes to human-readable conditions"""
        # WMO Weather interpretation codes (WMO Code 4677)
        weather_codes = {
            0: "clear",
            1: "mostly_clear",
            2: "partly_cloudy",
            3: "overcast",
            45: "foggy",
            48: "foggy",
            51: "light_drizzle",
            53: "drizzle",
            55: "heavy_drizzle",
            56: "freezing_drizzle",
            57: "freezing_drizzle",
            61: "light_rain",
            63: "rain",
            65: "heavy_rain",
            66: "freezing_rain",
            67: "freezing_rain",
            71: "light_snow",
            73: "snow",
            75: "heavy_snow",
            77: "snow_grains",
            80: "rain_showers",
            81: "rain_showers",
            82: "heavy_rain_showers",
            85: "snow_showers",
            86: "heavy_snow_showers",
            95: "thunderstorm",
            96: "thunderstorm_hail",
            99: "thunderstorm_hail"
        }
        return weather_codes.get(code, "unknown")

    def _default_weather_response(self, date: str, note: str) -> dict:
        """Return a default weather response when API is unavailable"""
        self.logger.info(f"Using default weather response for {date}: {note}")
        return {
            "date": date,
            "suitable": True,  # Default to suitable to not block scheduling
            "conditions": "unknown",
            "temperature": None,
            "precipitation_chance": None,
            "note": note,
            "api_source": "default_fallback"
        }

    async def _store_schedule(self, schedule: dict):
        """Store schedule in database"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO schedules (
                    id, job_id, start_time, end_time, metadata, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                schedule["schedule_id"],
                schedule.get("job_id"),
                schedule["start_time"],
                schedule["end_time"],
                json.dumps(schedule),
                datetime.now()
            ))
            conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store schedule: {e}")
            conn.rollback()
        finally:
            self.return_connection(conn)

# ==================== LANGGRAPH WORKFLOW ORCHESTRATION ====================

class WorkflowOrchestrator:
    """Advanced LangGraph-based workflow orchestration"""

    def __init__(self):
        self.agents = self._initialize_agents()
        self.graph = self._build_graph()

    def _initialize_agents(self) -> dict[str, IntelligentAgent]:
        """Initialize all 50+ agents"""
        agents = {}

        # Core Business Agents
        agents["estimation"] = EstimationAgent()
        agents["scheduling"] = SchedulingAgent()

        # Initialize remaining agents with base class
        for agent_type in AgentType:
            if agent_type.value not in agents:
                agents[agent_type.value] = IntelligentAgent(agent_type)

        return agents

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(WorkflowState)

        # Add nodes for each agent
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, agent.process)

        # Define workflow edges (simplified example)
        workflow.add_edge("estimation", "scheduling")
        workflow.add_edge("scheduling", "dispatch")
        workflow.add_edge("dispatch", "routing")
        workflow.add_edge("routing", "quality")
        workflow.add_edge("quality", "invoicing")
        workflow.add_edge("invoicing", "revenue")

        # Add conditional edges
        workflow.add_conditional_edges(
            "revenue",
            self._route_decision,
            {
                "complete": END,
                "followup": "customer",
                "escalate": "monitoring"
            }
        )

        # Set entry point
        workflow.set_entry_point("estimation")

        return workflow.compile()

    def _route_decision(self, state: WorkflowState) -> str:
        """Intelligent routing decision"""
        # Analyze results to determine next action
        if len(state["results"]) > 5:
            return "complete"
        elif state.get("context", {}).get("priority") == "high":
            return "escalate"
        else:
            return "followup"

    async def execute_workflow(self, initial_context: dict) -> dict:
        """Execute complete workflow"""
        initial_state = WorkflowState(
            messages=[],
            current_step="start",
            context=initial_context,
            results=[],
            next_action="estimation",
            metadata={}
        )

        # Execute graph
        final_state = await self.graph.ainvoke(initial_state)

        # Broadcast results via WebSocket
        await self._broadcast_results(final_state)

        return final_state

    async def _broadcast_results(self, state: WorkflowState):
        """Broadcast results to connected clients"""
        if redis_client:
            redis_client.publish("workflow:complete", json.dumps(state["results"]))

# ==================== REAL-TIME WEBSOCKET SERVER ====================

class RealtimeServer:
    """WebSocket server for real-time updates"""

    def __init__(self, orchestrator: WorkflowOrchestrator):
        self.orchestrator = orchestrator
        self.clients = set()

    async def handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)

                # Process workflow request
                if data.get("type") == "workflow":
                    result = await self.orchestrator.execute_workflow(data.get("context", {}))
                    await websocket.send(json.dumps(result))

                # Handle agent queries
                elif data.get("type") == "query":
                    agent_name = data.get("agent")
                    if agent_name in self.orchestrator.agents:
                        agent = self.orchestrator.agents[agent_name]
                        state = WorkflowState(
                            messages=[],
                            current_step=agent_name,
                            context=data.get("context", {}),
                            results=[],
                            next_action="",
                            metadata={}
                        )
                        result = await agent.process(state)
                        await websocket.send(json.dumps(result["results"]))

        finally:
            self.clients.remove(websocket)

    async def broadcast(self, message: str):
        """Broadcast to all connected clients"""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients]
            )

# ==================== MAIN ORCHESTRATION ====================

class AIOperatingSystem:
    """Complete AI Operating System with all components"""

    def __init__(self):
        self.orchestrator = WorkflowOrchestrator()
        self.websocket_server = RealtimeServer(self.orchestrator)
        self.running = True

    async def initialize(self):
        """Initialize all system components"""
        logger.info("=" * 60)
        logger.info("BRAINOPS AI OPERATING SYSTEM - INITIALIZATION")
        logger.info("=" * 60)

        # Test database connection
        conn = db_pool.getconn()
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()
        logger.info(f"Database connected: {version[0][:50]}")
        db_pool.putconn(conn)

        # Initialize agent registry
        await self._register_agents()

        # Start WebSocket server
        create_safe_task(self._start_websocket_server(), "websocket_server")

        # Start agent monitoring
        create_safe_task(self._monitor_agents(), "agent_monitor")

        logger.info(f"Initialized {len(self.orchestrator.agents)} agents")
        logger.info("AI Operating System OPERATIONAL")

    async def _register_agents(self):
        """Register all agents in database"""
        conn = db_pool.getconn()
        try:
            cursor = conn.cursor()
            for agent_name, agent in self.orchestrator.agents.items():
                cursor.execute("""
                    INSERT INTO ai_agents (id, name, status, capabilities, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE
                    SET status = 'active', last_active = NOW()
                """, (
                    str(uuid.uuid4()),
                    agent_name,
                    'active',
                    json.dumps({"type": agent.agent_type.value}),
                    datetime.now()
                ))
            conn.commit()
        finally:
            db_pool.putconn(conn)

    async def _start_websocket_server(self):
        """Start WebSocket server for real-time communication"""
        logger.info("Starting WebSocket server on port 8765")
        async with websockets.serve(self.websocket_server.handler, "0.0.0.0", 8765):
            await asyncio.Future()  # Run forever

    async def _monitor_agents(self):
        """Monitor agent health and performance"""
        while self.running:
            # Update agent status
            conn = db_pool.getconn()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ai_agents
                    SET last_active = NOW(),
                        total_executions = total_executions + 1
                    WHERE status = 'active'
                """)
                conn.commit()
            finally:
                db_pool.putconn(conn)

            # Log metrics
            logger.info(f"System healthy - {len(self.orchestrator.agents)} agents active")

            await asyncio.sleep(60)

    async def process_business_event(self, event_type: str, data: dict):
        """Process business events through appropriate workflows"""
        logger.info(f"Processing {event_type} event")

        # Route to appropriate workflow
        if event_type == "new_lead":
            context = {"customer": data, "workflow": "lead_to_customer"}
            result = await self.orchestrator.execute_workflow(context)

        elif event_type == "estimate_request":
            context = {"property": data, "workflow": "full_estimation"}
            result = await self.orchestrator.execute_workflow(context)

        elif event_type == "job_completion":
            context = {"job": data, "workflow": "completion_to_invoice"}
            result = await self.orchestrator.execute_workflow(context)

        else:
            context = {"data": data, "workflow": "generic"}
            result = await self.orchestrator.execute_workflow(context)

        return result

    async def run(self):
        """Main execution loop"""
        await self.initialize()

        # Process continuous workflows
        while self.running:
            try:
                # Check for pending work
                conn = db_pool.getconn()
                cursor = conn.cursor()

                # Process new customers
                cursor.execute("""
                    SELECT id, name, email, phone, metadata
                    FROM customers
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    LIMIT 10
                """)

                # Note: Assuming customers table structure
                # Will catch error if fails in loop

                new_customers = cursor.fetchall()
                for customer in new_customers:
                    await self.process_business_event("new_lead", {
                        "customer_id": customer[0],
                        "name": customer[1],
                        "email": customer[2],
                        "phone": customer[3]
                    })

                # Process pending estimates
                cursor.execute("""
                    SELECT id, customer_id, metadata
                    FROM jobs
                    WHERE status = 'pending'
                    AND NOT EXISTS (
                        SELECT 1 FROM estimates WHERE job_id = jobs.id
                    )
                    LIMIT 5
                """)

                pending_jobs = cursor.fetchall()
                for job in pending_jobs:
                    await self.process_business_event("estimate_request", {
                        "job_id": job[0],
                        "customer_id": job[1],
                        "metadata": job[2]
                    })

                db_pool.putconn(conn)

            except Exception as e:
                logger.error(f"Processing error: {e}")

            await asyncio.sleep(30)

# ==================== ENTRY POINT ====================

async def main():
    """Main entry point for AI Operating System"""
    ai_os = AIOperatingSystem()

    try:
        await ai_os.run()
    except KeyboardInterrupt:
        logger.info("Shutting down AI Operating System")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise

if __name__ == "__main__":
    # Run the complete AI Operating System
    asyncio.run(main())

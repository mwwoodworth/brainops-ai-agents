#!/usr/bin/env python3
"""
Production Hotfix for BrainOps AI Agents Service
Fixes all critical errors identified in production logs
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)

# Fix 1: Custom JSON Encoder for datetime and Enum types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

# Fix 2: Database schema fixes
async def fix_database_schema():
    """Fix missing columns and type mismatches in database"""

    # Validate required environment variables - NO hardcoded fallbacks
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_port = os.getenv("DB_PORT", "5432")

    missing = []
    if not db_host:
        missing.append("DB_HOST")
    if not db_name:
        missing.append("DB_NAME")
    if not db_user:
        missing.append("DB_USER")
    if not db_password:
        missing.append("DB_PASSWORD")

    if missing:
        raise RuntimeError(
            f"Required environment variables not set: {', '.join(missing)}. "
            "Set these variables before running this hotfix."
        )

    db_config = {
        "host": db_host,
        "database": db_name,
        "user": db_user,
        "password": db_password,
        "port": int(db_port)
    }

    conn = await asyncpg.connect(**db_config)

    try:
        # Check and add missing columns
        print("🔧 Checking database schema...")

        # Fix 1: Add last_job_date column if missing
        try:
            await conn.execute("""
                ALTER TABLE customers
                ADD COLUMN IF NOT EXISTS last_job_date TIMESTAMP
            """)
            print("✅ Added last_job_date column to customers")
        except Exception as e:
            print(f"⚠️  Column may already exist: {e}")

        # Fix 2: Create index on last_job_date for performance
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_customers_last_job_date
                ON customers(last_job_date)
            """)
            print("✅ Created index on last_job_date")
        except Exception as e:
            print(f"⚠️  Index may already exist: {e}")

        # Fix 3: Update last_job_date from jobs table
        await conn.execute("""
            UPDATE customers c
            SET last_job_date = (
                SELECT MAX(scheduled_start)
                FROM jobs j
                WHERE j.customer_id = c.id
            )
            WHERE EXISTS (
                SELECT 1 FROM jobs j2 WHERE j2.customer_id = c.id
            )
        """)
        print("✅ Updated last_job_date from jobs data")

        # Fix 4: Check unified_ai_memory table structure (CANONICAL memory table)
        result = await conn.fetchval("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'unified_ai_memory'
            AND column_name = 'related_memories'
        """)

        if result:
            # Fix type of related_memories column
            await conn.execute("""
                ALTER TABLE unified_ai_memory
                ALTER COLUMN related_memories
                TYPE uuid[] USING related_memories::uuid[]
            """)
            print("✅ Fixed related_memories column type")

        print("✅ Database schema fixes completed")

    except Exception as e:
        print(f"❌ Database fix error: {e}")
    finally:
        await conn.close()

# Fix 3: Updated AUREA Orchestrator observation method
AUREA_OBSERVE_FIX = '''
async def observe(self) -> Dict:
    """Fixed observation gathering from environment"""
    observations = {
        "timestamp": datetime.utcnow().isoformat(),  # Convert to string
        "metrics": {},
        "alerts": [],
        "opportunities": []
    }

    if not self.db_pool:
        return observations

    try:
        async with self.db_pool.acquire() as conn:
            # Get customers who haven't had jobs in 90 days (FIXED QUERY)
            inactive_customers = await conn.fetch("""
                SELECT c.id, c.name, c.email,
                       COALESCE(c.last_job_date,
                               (SELECT MAX(scheduled_start)
                                FROM jobs WHERE customer_id = c.id),
                               c.created_at) as last_activity
                FROM customers c
                WHERE COALESCE(c.last_job_date,
                              (SELECT MAX(scheduled_start)
                               FROM jobs WHERE customer_id = c.id),
                              c.created_at) < NOW() - INTERVAL \'90 days\'
                LIMIT 10
            """)

            if inactive_customers:
                observations["opportunities"].append({
                    "type": "customer_retention",
                    "count": len(inactive_customers),
                    "customers": [dict(c) for c in inactive_customers]
                })

            # Get overdue invoices (USING COMPUTED FIELD)
            overdue_invoices = await conn.fetch("""
                SELECT id, customer_id,
                       (COALESCE(total_amount::numeric/100, 0) -
                        COALESCE(paid_amount, 0)) as amount_due,
                       due_date
                FROM invoices
                WHERE status != \'paid\'
                AND due_date < NOW()
                LIMIT 20
            """)

            if overdue_invoices:
                observations["alerts"].append({
                    "type": "overdue_invoices",
                    "count": len(overdue_invoices),
                    "total_due": sum(float(inv["amount_due"]) for inv in overdue_invoices)
                })

    except Exception as e:
        logger.error(f"Observation error (handled): {e}")
        # Return partial observations rather than failing
        observations["error"] = str(e)

    return observations
'''

# Fix 4: Updated Memory Manager with proper JSON handling
MEMORY_STORE_FIX = '''
async def store(self, key: str, value: Any, context_type: str = "system",
               context_id: str = None, importance: int = 5,
               expires_in_hours: Optional[int] = None,
               tags: List[str] = None) -> bool:
    """Store memory with proper JSON serialization"""

    if not self.pool:
        await self.initialize()

    try:
        # Ensure context_id is a string
        if context_id is None:
            context_id = "default"
        else:
            context_id = str(context_id)[:255]

        # Serialize value with custom encoder
        json_value = json.dumps(value, cls=CustomJSONEncoder)

        # Calculate expiration
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO unified_ai_memory (
                    memory_type, content, source_system, source_agent,
                    created_by, importance_score, tags, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (content_hash)
                DO UPDATE SET
                    content = $2,
                    importance_score = $6,
                    tags = $7,
                    expires_at = $8,
                    updated_at = NOW(),
                    access_count = unified_ai_memory.access_count + 1
            """, 'semantic', json_value, context_type, context_id, key,
                importance / 10.0, tags or [], expires_at)

        logger.info(f"✅ Stored memory: {key}")
        return True

    except Exception as e:
        logger.error(f"❌ Memory store error: {e}")
        return False
'''

# Fix 5: API Route Registration fixes
API_ROUTES_FIX = '''
# Add these routes to main_integration.py

# Memory endpoints (were missing)
@app.post("/memory/store")
async def store_memory(request: Dict[str, Any]):
    """Store a memory in the unified system"""
    try:
        success = await memory_manager.store(
            key=request.get("key"),
            value=request.get("value"),
            context_type=request.get("context_type", "system"),
            context_id=request.get("context_id"),
            importance=request.get("importance", 5),
            tags=request.get("tags", [])
        )
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/recall")
async def recall_memory(
    query: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    limit: int = 10
):
    """Recall memories from the unified system"""
    try:
        memories = await memory_manager.recall(
            query=query,
            context_type=context_type,
            context_id=context_id,
            limit=limit
        )
        return {"memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Board endpoints
@app.get("/board/members")
async def get_board_members():
    """Get AI board members"""
    if not ai_board:
        raise HTTPException(status_code=503, detail="AI Board not initialized")
    return {"members": ai_board.get_members()}

# Agent categories endpoint
@app.get("/agents/categories")
async def get_agent_categories():
    """Get agent categories and counts"""
    categories = {}
    for agent_name, agent_info in AGENT_REGISTRY.items():
        category = agent_info.get("category", "general")
        if category not in categories:
            categories[category] = []
        categories[category].append(agent_name)

    return {
        "categories": categories,
        "counts": {cat: len(agents) for cat, agents in categories.items()}
    }

# Business-specific endpoints
@app.post("/weathercraft/enhance")
async def weathercraft_enhance(request: Dict[str, Any]):
    """Enhance Weathercraft operations with AI"""
    task = request.get("task")
    data = request.get("data", {})

    # Route to appropriate agent based on task
    agent_mapping = {
        "optimize_schedule": "SchedulerAgent",
        "generate_invoice": "InvoicingAgent",
        "forecast_demand": "ForecastingAgent",
        "analyze_customer": "CustomerAnalysisAgent"
    }

    agent_name = agent_mapping.get(task, "GeneralAgent")
    result = await activation_system.activate_agent(
        agent_name=agent_name,
        event_data=data
    )

    return {"task": task, "result": result}

@app.post("/myroofgenius/automate")
async def myroofgenius_automate(request: Dict[str, Any]):
    """Automate MyRoofGenius workflows"""
    workflow = request.get("workflow")
    data = request.get("data", {})

    # Trigger autonomous workflow
    if workflow == "full_autonomy":
        # Start AUREA in full auto mode
        aurea.set_autonomy_level(100)
        await aurea.start()

    return {"workflow": workflow, "status": "initiated", "data": data}
'''

# Fix 6: Agent timeout configuration
AGENT_TIMEOUT_FIX = '''
# In agent_activation_system.py, increase timeout and add retry logic

class AgentActivationSystem:
    def __init__(self):
        self.timeout = 120  # Increase from 30s to 120s
        self.max_retries = 3
        self.retry_delay = 5

    async def activate_agent_with_retry(self, agent_name: str, event_data: Dict):
        """Activate agent with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Create timeout with longer duration
                result = await asyncio.wait_for(
                    self._execute_agent(agent_name, event_data),
                    timeout=self.timeout
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ {agent_name} timeout attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"❌ {agent_name} failed after {self.max_retries} attempts")
                    return {"error": "Agent timeout", "agent": agent_name}
'''

# Fix 7: Validation schemas
VALIDATION_SCHEMAS_FIX = '''
# Add proper Pydantic models for validation

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AgentActivationRequest(BaseModel):
    event_type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    priority: Optional[int] = Field(default=5, ge=1, le=10)
    async_mode: Optional[bool] = False

class BoardProposalRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    category: str = Field(default="strategic")
    urgency: Optional[int] = Field(default=5, ge=1, le=10)
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)

# Update endpoints to use these models
@app.post("/agents/activate")
async def activate_agents(request: AgentActivationRequest):
    """Activate agents with proper validation"""
    result = await activation_system.handle_event(
        event_type=request.event_type,
        data=request.data,
        priority=request.priority
    )
    return result

@app.post("/board/proposal")
async def submit_proposal(request: BoardProposalRequest):
    """Submit proposal to AI Board"""
    if not ai_board:
        raise HTTPException(status_code=503, detail="AI Board not initialized")

    result = await ai_board.submit_proposal(
        title=request.title,
        description=request.description,
        category=request.category,
        urgency=request.urgency,
        data=request.data
    )
    return result
'''

async def apply_all_fixes():
    """Apply all production fixes"""
    print("=" * 80)
    print("🚀 APPLYING PRODUCTION HOTFIXES")
    print("=" * 80)

    # Fix database schema
    await fix_database_schema()

    print("""
✅ FIXES TO APPLY IN CODE:

1. AUREA Observation Method - Replace in aurea_orchestrator.py
2. Memory Store Method - Replace in unified_memory_manager.py
3. Add Missing API Routes - Add to main_integration.py
4. Fix Agent Timeouts - Update agent_activation_system.py
5. Add Validation Schemas - Add to main_integration.py

After applying these fixes:
1. Commit changes
2. Push to GitHub
3. Render will auto-deploy
4. Monitor logs for improvements
""")

if __name__ == "__main__":
    asyncio.run(apply_all_fixes())

# AUREA Event Endpoint Integration Instructions

## Quick Integration (5 minutes)

Add this code to `app.py` to enable Event Router integration.

---

## Step 1: Add Request Model (after line 1674)

Add this after `class AgentActivateRequest`:

```python
class AUREAEventRequest(BaseModel):
    """Request model for AUREA event execution"""
    event_id: str
    topic: str
    source: str
    payload: Dict[str, Any]
    target_agent: Dict[str, Any]  # {name, role, capabilities}
    routing_metadata: Optional[Dict[str, Any]] = None
```

---

## Step 2: Add Endpoint (after /api/v1/agents/activate endpoint, around line 2100)

```python
@app.post("/api/v1/aurea/execute-event")
async def execute_aurea_event(
    request: AUREAEventRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Execute event with specified AI agent via AUREA orchestration.
    Called by Event Router daemon to process events from brainops_core.event_bus.
    """
    logger.info(f"🎯 AUREA Event: {request.event_id} ({request.topic}) -> {request.target_agent['name']}")

    pool = get_pool()

    try:
        # Find target agent by name
        agent_row = await pool.fetchrow(
            """
            SELECT id, name, category, enabled
            FROM agents
            WHERE name = $1 AND enabled = TRUE
            """,
            request.target_agent['name']
        )

        if not agent_row:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{request.target_agent['name']}' not found or disabled"
            )

        agent_id = str(agent_row['id'])
        agent_name = agent_row['name']

        # Prepare agent execution payload
        agent_payload = {
            "event_id": request.event_id,
            "topic": request.topic,
            "source": request.source,
            **request.payload
        }

        # Execute agent (simple acknowledgment for now)
        # Can be expanded with topic-specific handlers
        result = {
            "status": "acknowledged",
            "agent": agent_name,
            "event_id": request.event_id,
            "topic": request.topic,
            "action": "processed"
        }

        # Update agent last_active_at in brainops_core.agents (if table exists)
        try:
            await pool.execute(
                """
                UPDATE brainops_core.agents
                SET last_active_at = NOW()
                WHERE name = $1
                """,
                agent_name
            )
        except Exception:
            pass  # Table might not exist

        # Store in embedded memory if available
        embedded_memory = getattr(app.state, "embedded_memory", None)
        if embedded_memory:
            try:
                embedded_memory.store_memory(
                    memory_id=str(uuid.uuid4()),
                    memory_type="episodic",
                    source_agent=agent_name,
                    content=f"Processed event: {request.topic}",
                    metadata={
                        "event_id": request.event_id,
                        "topic": request.topic,
                        "source": request.source
                    },
                    importance_score=0.7
                )
            except Exception as e:
                logger.warning(f"Could not store in embedded memory: {e}")

        logger.info(f"✅ AUREA Event {request.event_id} executed by {agent_name}")

        return {
            "success": True,
            "event_id": request.event_id,
            "agent": agent_name,
            "topic": request.topic,
            "result": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ AUREA Event {request.event_id} failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Event execution failed: {str(e)}"
        )
```

---

## Step 3: Test the Endpoint

```bash
# Test with curl
curl -X POST https://brainops-ai-agents.onrender.com/api/v1/aurea/execute-event \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_BRAINOPS_API_KEY>" \
  -d '{
    "event_id": "test-123",
    "topic": "task.created",
    "source": "manual_test",
    "payload": {"task_id": "task-456", "title": "Test Task"},
    "target_agent": {"name": "Claude", "role": "Chief of Staff", "capabilities": []},
    "routing_metadata": {}
  }'
```

Expected response:
```json
{
  "success": true,
  "event_id": "test-123",
  "agent": "Claude",
  "topic": "task.created",
  "result": {
    "status": "acknowledged",
    "agent": "Claude",
    "event_id": "test-123",
    "topic": "task.created",
    "action": "processed"
  }
}
```

---

## Step 4: Deploy

```bash
# Commit changes
git add app.py
git commit -m "feat: Add AUREA event execution endpoint for Event Router integration"
git push origin main

# Render will auto-deploy
# Check deployment: https://dashboard.render.com
```

---

## Advanced: Topic-Specific Handlers (Optional)

To add specialized handling for different event types, you can expand the endpoint with a handler dispatcher:

```python
async def handle_event_by_topic(topic: str, agent_name: str, payload: Dict[str, Any]):
    """Route to topic-specific handler"""
    handlers = {
        'roof.inspection_requested': handle_roof_inspection,
        'task.created': handle_task_created,
        'job.created': handle_job_created,
        # Add more as needed
    }

    handler = handlers.get(topic, handle_generic_event)
    return await handler(agent_name, payload)
```

See `aurea_event_endpoint.py` for full handler implementations.

---

## Verification Checklist

- [ ] Request model added to app.py
- [ ] Endpoint added to app.py
- [ ] Code committed and pushed
- [ ] Render deployment completed
- [ ] Endpoint tested with curl
- [ ] Event Router can call endpoint
- [ ] Events are being processed

---

**Integration Time**: ~5 minutes
**Testing Time**: ~2 minutes
**Total**: ~7 minutes to full Event Router integration

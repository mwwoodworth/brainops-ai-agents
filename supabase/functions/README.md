# Supabase Edge Functions for AI Endpoints

This directory contains serverless Edge Functions that provide AI capabilities through Supabase's edge runtime.

## Functions Overview

### 1. ai-chat
**Purpose:** Conversational AI endpoint for chat interactions

**Endpoint:** `POST https://yomagoqdmxszqtdwuhab.supabase.co/functions/v1/ai-chat`

**Request Body:**
```json
{
  "message": "Your message here",
  "conversation_id": "optional-uuid",
  "user_id": "optional-user-id",
  "context": {}
}
```

**Response:**
```json
{
  "success": true,
  "conversation_id": "uuid",
  "response": "AI response",
  "message_id": "uuid"
}
```

### 2. ai-analyze
**Purpose:** Data analysis and insights generation

**Endpoint:** `POST https://yomagoqdmxszqtdwuhab.supabase.co/functions/v1/ai-analyze`

**Request Body:**
```json
{
  "data_type": "customer|revenue|performance|general",
  "entity_id": "optional-id",
  "time_period": "30d",
  "metrics": ["metric1", "metric2"]
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "key_metrics": {},
    "trends": {}
  },
  "insights": ["insight1", "insight2"],
  "timestamp": "2025-09-17T16:00:00Z"
}
```

### 3. ai-execute
**Purpose:** Execute AI agent tasks

**Endpoint:** `POST https://yomagoqdmxszqtdwuhab.supabase.co/functions/v1/ai-execute`

**Request Body:**
```json
{
  "agent_type": "revenue_generator|customer_acquisition|data_analyzer|automation",
  "task": "Task description",
  "parameters": {},
  "priority": "normal"
}
```

**Response:**
```json
{
  "success": true,
  "execution_id": "uuid",
  "result": {},
  "status": "completed"
}
```

## Deployment

1. **Install Supabase CLI:**
   ```bash
   npm install -g supabase
   ```

2. **Login to Supabase:**
   ```bash
   supabase login
   ```

3. **Link project:**
   ```bash
   supabase link --project-ref yomagoqdmxszqtdwuhab
   ```

4. **Deploy functions:**
   ```bash
   ./deploy-edge-functions.sh
   ```

## Environment Variables

Required environment variables (set in Supabase dashboard):
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `OPENAI_API_KEY`

## Testing

### Test ai-chat:
```bash
curl -X POST https://yomagoqdmxszqtdwuhab.supabase.co/functions/v1/ai-chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ANON_KEY" \
  -d '{"message": "Hello AI"}'
```

### Test ai-analyze:
```bash
curl -X POST https://yomagoqdmxszqtdwuhab.supabase.co/functions/v1/ai-analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ANON_KEY" \
  -d '{"data_type": "customer"}'
```

### Test ai-execute:
```bash
curl -X POST https://yomagoqdmxszqtdwuhab.supabase.co/functions/v1/ai-execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ANON_KEY" \
  -d '{"agent_type": "revenue_generator", "task": "identify leads"}'
```

## Features

- **Serverless:** Runs on Supabase Edge Runtime (Deno)
- **Global Distribution:** Deployed to edge locations worldwide
- **Auto-scaling:** Handles traffic spikes automatically
- **Integrated Auth:** Works with Supabase Auth
- **Direct DB Access:** Can query Supabase database directly
- **CORS Enabled:** Accessible from web applications
- **OpenAI Integration:** Leverages GPT-4 for intelligence

## Architecture

```
┌─────────────────────────────────────┐
│         Client Application          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│    Supabase Edge Functions (CDN)    │
├─────────────────────────────────────┤
│  • ai-chat   (Conversational AI)    │
│  • ai-analyze (Data Analysis)       │
│  • ai-execute (Task Execution)      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      Supabase PostgreSQL + AI       │
├─────────────────────────────────────┤
│  • ai_messages                      │
│  • ai_customer_interactions         │
│  • agent_executions                 │
│  • ai_learning_insights             │
└─────────────────────────────────────┘
```

## Benefits

1. **No Infrastructure Management:** Serverless deployment
2. **Cost Effective:** Pay only for execution time
3. **Low Latency:** Edge deployment near users
4. **Secure:** Built-in auth and RLS
5. **Scalable:** Auto-scales with demand
6. **Integrated:** Direct database and storage access

## Monitoring

View function logs in Supabase Dashboard:
1. Go to Functions section
2. Select function name
3. View logs and metrics

## Security

- Functions use JWT verification (can be disabled)
- Service role key for admin operations
- Row Level Security (RLS) on database tables
- CORS configuration for web access
- Environment variables for sensitive data
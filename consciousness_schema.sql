-- Consciousness Loop Database Schema

-- 1. Consciousness State: Snapshot of what the AI knows/feels
CREATE TABLE IF NOT EXISTS ai_consciousness_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    awareness_level FLOAT DEFAULT 0.0, -- 0.0 to 1.0
    active_systems JSONB DEFAULT '{}'::jsonb, -- Snapshot of healthy/active components
    current_context TEXT, -- Summary of current situation
    short_term_memory_load FLOAT, -- Percentage of context window used
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_consciousness_timestamp ON ai_consciousness_state(timestamp);

-- 2. Thought Stream: The internal monologue
CREATE TABLE IF NOT EXISTS ai_thought_stream (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    thought_content TEXT NOT NULL,
    thought_type VARCHAR(50) NOT NULL, -- 'observation', 'analysis', 'decision', 'dream', 'alert'
    related_entities JSONB DEFAULT '[]'::jsonb, -- IDs or names of things being thought about
    intensity FLOAT DEFAULT 0.5, -- 0.0 to 1.0 (emotional weight / urgency)
    metadata JSONB DEFAULT '{}'::jsonb,
    priority INTEGER,
    confidence FLOAT,
    related_thoughts TEXT[],
    thought_id VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_thought_stream_timestamp ON ai_thought_stream(timestamp);
CREATE INDEX IF NOT EXISTS idx_thought_stream_type ON ai_thought_stream(thought_type);

-- 3. Attention Focus: What the AI is prioritizing
CREATE TABLE IF NOT EXISTS ai_attention_focus (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    focus_target VARCHAR(255) NOT NULL, -- System, User, or Problem ID
    reason TEXT,
    priority INTEGER DEFAULT 1, -- 1 (low) to 10 (critical)
    status VARCHAR(50) DEFAULT 'active', -- 'active', 'shifted', 'resolved'
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_attention_focus_target ON ai_attention_focus(focus_target);
CREATE INDEX IF NOT EXISTS idx_attention_focus_status ON ai_attention_focus(status);

-- 4. Vital Signs: System health metrics
CREATE TABLE IF NOT EXISTS ai_vital_signs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    cpu_usage FLOAT,
    memory_usage FLOAT,
    request_rate FLOAT,
    error_rate FLOAT,
    active_connections INTEGER,
    system_load FLOAT,
    component_health_score FLOAT -- Aggregate score of all components
);

CREATE INDEX IF NOT EXISTS idx_vital_signs_timestamp ON ai_vital_signs(timestamp);

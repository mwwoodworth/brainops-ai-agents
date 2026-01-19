-- =============================================================================
-- UNIFIED AGENT STATE TABLE - Single source of truth for all agent state
-- =============================================================================
-- This migration creates a unified agent state tracking system that consolidates
-- fragmented state tracking across multiple systems (AgentScheduler, AgentExecutor,
-- TaskQueueConsumer, etc.) into a single coherent view.
--
-- Part of BrainOps AI OS Total Completion Protocol
-- Created: 2026-01-19
-- =============================================================================

-- Create the unified agent state table
CREATE TABLE IF NOT EXISTS unified_agent_state (
    -- Primary key
    agent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Agent identification
    agent_name TEXT NOT NULL UNIQUE,
    agent_type TEXT NOT NULL DEFAULT 'general',

    -- Execution tracking
    last_execution TIMESTAMPTZ,
    total_executions INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    failed_executions INTEGER DEFAULT 0,

    -- Health and status
    health_score FLOAT DEFAULT 1.0 CHECK (health_score >= 0 AND health_score <= 1),
    current_status TEXT DEFAULT 'idle' CHECK (current_status IN ('idle', 'running', 'error', 'disabled', 'warming_up')),
    last_health_check TIMESTAMPTZ,

    -- Current task tracking
    current_task_id UUID,
    current_task_started_at TIMESTAMPTZ,

    -- Performance metrics
    avg_execution_time_ms FLOAT DEFAULT 0,
    p95_execution_time_ms FLOAT DEFAULT 0,
    error_rate FLOAT DEFAULT 0,

    -- Memory and context
    memory_context_size INTEGER DEFAULT 0,
    last_memory_retrieval TIMESTAMPTZ,

    -- Configuration
    config JSONB DEFAULT '{}'::jsonb,
    capabilities TEXT[] DEFAULT ARRAY[]::TEXT[],

    -- Learning and adaptation
    learning_score FLOAT DEFAULT 0.5,
    adaptation_count INTEGER DEFAULT 0,
    last_learned_at TIMESTAMPTZ,

    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Source tracking
    last_updated_by TEXT DEFAULT 'system'
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_unified_agent_state_name ON unified_agent_state(agent_name);
CREATE INDEX IF NOT EXISTS idx_unified_agent_state_type ON unified_agent_state(agent_type);
CREATE INDEX IF NOT EXISTS idx_unified_agent_state_status ON unified_agent_state(current_status);
CREATE INDEX IF NOT EXISTS idx_unified_agent_state_health ON unified_agent_state(health_score DESC);
CREATE INDEX IF NOT EXISTS idx_unified_agent_state_last_execution ON unified_agent_state(last_execution DESC);

-- Create agent execution history table for detailed tracking
CREATE TABLE IF NOT EXISTS agent_execution_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES unified_agent_state(agent_id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,

    -- Execution details
    task_id UUID,
    task_type TEXT,
    execution_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    execution_completed_at TIMESTAMPTZ,
    execution_time_ms INTEGER,

    -- Result tracking
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'timeout', 'cancelled')),
    result JSONB,
    error_message TEXT,

    -- Memory integration
    memory_context_used JSONB,
    memory_written JSONB,

    -- Performance metrics
    tokens_used INTEGER,
    model_used TEXT,

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for execution history
CREATE INDEX IF NOT EXISTS idx_agent_exec_history_agent ON agent_execution_history(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_exec_history_name ON agent_execution_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_exec_history_status ON agent_execution_history(status);
CREATE INDEX IF NOT EXISTS idx_agent_exec_history_created ON agent_execution_history(created_at DESC);

-- Create agent health metrics table for time-series health data
CREATE TABLE IF NOT EXISTS agent_health_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES unified_agent_state(agent_id) ON DELETE CASCADE,

    -- Timestamp
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Health indicators
    health_score FLOAT NOT NULL,
    error_rate FLOAT,
    avg_latency_ms FLOAT,
    queue_depth INTEGER,
    memory_usage_mb FLOAT,

    -- Context
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create index for time-series queries
CREATE INDEX IF NOT EXISTS idx_agent_health_metrics_agent_time
    ON agent_health_metrics(agent_id, recorded_at DESC);

-- Create function to update unified agent state on execution
CREATE OR REPLACE FUNCTION update_agent_state_on_execution()
RETURNS TRIGGER AS $$
BEGIN
    -- Update or insert unified agent state
    INSERT INTO unified_agent_state (
        agent_name,
        last_execution,
        total_executions,
        successful_executions,
        failed_executions,
        avg_execution_time_ms,
        current_status,
        updated_at,
        last_updated_by
    ) VALUES (
        NEW.agent_name,
        NEW.execution_completed_at,
        1,
        CASE WHEN NEW.status = 'completed' THEN 1 ELSE 0 END,
        CASE WHEN NEW.status = 'failed' THEN 1 ELSE 0 END,
        COALESCE(NEW.execution_time_ms, 0),
        'idle',
        NOW(),
        'execution_trigger'
    )
    ON CONFLICT (agent_name) DO UPDATE SET
        last_execution = COALESCE(NEW.execution_completed_at, NOW()),
        total_executions = unified_agent_state.total_executions + 1,
        successful_executions = unified_agent_state.successful_executions +
            CASE WHEN NEW.status = 'completed' THEN 1 ELSE 0 END,
        failed_executions = unified_agent_state.failed_executions +
            CASE WHEN NEW.status = 'failed' THEN 1 ELSE 0 END,
        avg_execution_time_ms = (
            (unified_agent_state.avg_execution_time_ms * unified_agent_state.total_executions) +
            COALESCE(NEW.execution_time_ms, 0)
        ) / (unified_agent_state.total_executions + 1),
        error_rate = (unified_agent_state.failed_executions +
            CASE WHEN NEW.status = 'failed' THEN 1 ELSE 0 END)::float /
            (unified_agent_state.total_executions + 1),
        health_score = CASE
            WHEN NEW.status = 'completed' THEN LEAST(1.0, unified_agent_state.health_score + 0.01)
            WHEN NEW.status = 'failed' THEN GREATEST(0.0, unified_agent_state.health_score - 0.05)
            ELSE unified_agent_state.health_score
        END,
        current_status = 'idle',
        updated_at = NOW(),
        last_updated_by = 'execution_trigger';

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update agent state on execution completion
DROP TRIGGER IF EXISTS trg_update_agent_state ON agent_execution_history;
CREATE TRIGGER trg_update_agent_state
AFTER INSERT OR UPDATE OF status ON agent_execution_history
FOR EACH ROW
WHEN (NEW.status IN ('completed', 'failed', 'timeout'))
EXECUTE FUNCTION update_agent_state_on_execution();

-- Create view for agent dashboard
CREATE OR REPLACE VIEW agent_dashboard AS
SELECT
    uas.agent_id,
    uas.agent_name,
    uas.agent_type,
    uas.current_status,
    uas.health_score,
    uas.total_executions,
    uas.successful_executions,
    uas.failed_executions,
    ROUND((uas.successful_executions::numeric / NULLIF(uas.total_executions, 0) * 100), 2) as success_rate,
    ROUND(uas.avg_execution_time_ms::numeric, 2) as avg_execution_time_ms,
    uas.last_execution,
    uas.current_task_id,
    uas.learning_score,
    uas.capabilities,
    -- Recent execution summary
    (
        SELECT COUNT(*)
        FROM agent_execution_history aeh
        WHERE aeh.agent_id = uas.agent_id
        AND aeh.created_at > NOW() - INTERVAL '24 hours'
    ) as executions_last_24h,
    (
        SELECT COUNT(*)
        FROM agent_execution_history aeh
        WHERE aeh.agent_id = uas.agent_id
        AND aeh.status = 'failed'
        AND aeh.created_at > NOW() - INTERVAL '24 hours'
    ) as failures_last_24h
FROM unified_agent_state uas
ORDER BY uas.health_score DESC, uas.last_execution DESC;

-- Grant permissions
GRANT SELECT ON agent_dashboard TO PUBLIC;
GRANT ALL ON unified_agent_state TO PUBLIC;
GRANT ALL ON agent_execution_history TO PUBLIC;
GRANT ALL ON agent_health_metrics TO PUBLIC;

-- Add comment for documentation
COMMENT ON TABLE unified_agent_state IS 'Unified agent state tracking - single source of truth for all agent state across BrainOps AI OS';
COMMENT ON VIEW agent_dashboard IS 'Real-time agent dashboard view with execution statistics';

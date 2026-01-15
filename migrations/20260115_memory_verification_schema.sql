-- BrainOps Memory Verification Schema Enhancement
-- Generated: 2026-01-15
-- Purpose: Add verification, proof, and enforcement fields to unified_ai_memory
-- Part of Total Completion Protocol

-- =============================================================================
-- STEP 1: Create verification state ENUM
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'verification_state') THEN
        CREATE TYPE verification_state AS ENUM (
            'UNVERIFIED',     -- No verification performed
            'VERIFIED',       -- Verified with proof artifacts (E2+)
            'DEGRADED',       -- Was verified but stale or confidence dropped
            'BROKEN'          -- Verification failed or proof invalid
        );
    END IF;
END$$;

-- =============================================================================
-- STEP 2: Create evidence level ENUM
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'evidence_level') THEN
        CREATE TYPE evidence_level AS ENUM (
            'E0_UNVERIFIED',  -- No artifacts, just claims
            'E1_RECORDED',    -- Logs captured + linked
            'E2_TESTED',      -- Automated test results + env + timestamp + commit
            'E3_OBSERVED',    -- Synthetic monitor / real usage observation + trend
            'E4_AUDITED'      -- Cross-checked evidence + rollback + owner signoff
        );
    END IF;
END$$;

-- =============================================================================
-- STEP 3: Create memory object type ENUM (BrainOps OS schema requirement)
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'memory_object_type') THEN
        CREATE TYPE memory_object_type AS ENUM (
            'decision',       -- AI decision records
            'sop',            -- Standard operating procedures
            'proof',          -- Verification proof artifacts
            'task',           -- Task records
            'incident',       -- Incident reports
            'kpi',            -- KPI definitions
            'architecture',   -- Architecture documentation
            'integration',    -- Integration records
            'experiment',     -- Experiment results
            'runbook'         -- Operational runbooks
        );
    END IF;
END$$;

-- =============================================================================
-- STEP 4: Add verification columns to unified_ai_memory
-- =============================================================================

-- Add verification_state column
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS verification_state verification_state DEFAULT 'UNVERIFIED';

-- Add evidence_level column
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS evidence_level evidence_level DEFAULT 'E0_UNVERIFIED';

-- Add last_verified_at timestamp
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS last_verified_at TIMESTAMPTZ;

-- Add verified_by (agent/tool/human identifier)
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS verified_by TEXT;

-- Add artifact_refs (list of artifact IDs/URLs/hashes)
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS artifact_refs TEXT[] DEFAULT ARRAY[]::TEXT[];

-- Add confidence_score (0-1 trust level)
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS confidence_score FLOAT DEFAULT 0.0 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0);

-- Add verification_expires_at (when re-verification is required)
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS verification_expires_at TIMESTAMPTZ;

-- Add supersedes (IDs of memories this supersedes)
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS supersedes UUID[] DEFAULT ARRAY[]::UUID[];

-- Add object_type (BrainOps OS memory object classification)
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS object_type memory_object_type;

-- Add owner (responsible party for this memory)
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS owner TEXT;

-- Add project (project association)
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS project TEXT;

-- Add retrieval_policy_version (for RAG versioning)
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS retrieval_policy_version TEXT DEFAULT 'v1.0';

-- Add chunking_strategy (how content was chunked for embedding)
ALTER TABLE unified_ai_memory
ADD COLUMN IF NOT EXISTS chunking_strategy TEXT DEFAULT 'none';

-- =============================================================================
-- STEP 5: Create verification table for storing proof artifacts
-- =============================================================================

CREATE TABLE IF NOT EXISTS memory_verification_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES unified_ai_memory(id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,  -- log, test_result, screenshot, response, commit_hash
    artifact_url TEXT,            -- URL or path to artifact
    artifact_hash TEXT,           -- SHA256 hash of artifact content
    artifact_content JSONB,       -- Inline artifact content (if small)
    evidence_level evidence_level DEFAULT 'E1_RECORDED',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT,
    expires_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX IF NOT EXISTS idx_verification_artifacts_memory_id
ON memory_verification_artifacts(memory_id);

CREATE INDEX IF NOT EXISTS idx_verification_artifacts_type
ON memory_verification_artifacts(artifact_type);

-- =============================================================================
-- STEP 6: Create memory operation audit log
-- =============================================================================

CREATE TABLE IF NOT EXISTS memory_operation_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation TEXT NOT NULL,      -- read, write, update, delete, verify
    memory_id UUID,               -- Target memory (NULL for failed lookups)
    agent_id TEXT,                -- Agent that performed operation
    tool_id TEXT,                 -- Tool/MCP tool used
    correlation_id UUID,          -- Request correlation ID
    tenant_id UUID,
    operation_context JSONB,      -- Query params, filters, etc.
    operation_result TEXT,        -- success, failed, blocked
    error_message TEXT,
    duration_ms INTEGER,
    rba_enforced BOOLEAN DEFAULT FALSE,  -- Read-before-act was enforced
    wba_enforced BOOLEAN DEFAULT FALSE,  -- Write-back-after was enforced
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX IF NOT EXISTS idx_memory_audit_memory_id
ON memory_operation_audit(memory_id);

CREATE INDEX IF NOT EXISTS idx_memory_audit_agent_id
ON memory_operation_audit(agent_id);

CREATE INDEX IF NOT EXISTS idx_memory_audit_correlation_id
ON memory_operation_audit(correlation_id);

CREATE INDEX IF NOT EXISTS idx_memory_audit_created_at
ON memory_operation_audit(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_audit_operation
ON memory_operation_audit(operation);

-- =============================================================================
-- STEP 7: Create memory conflicts table for detecting contradictions
-- =============================================================================

CREATE TABLE IF NOT EXISTS memory_conflicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id_a UUID NOT NULL REFERENCES unified_ai_memory(id) ON DELETE CASCADE,
    memory_id_b UUID NOT NULL REFERENCES unified_ai_memory(id) ON DELETE CASCADE,
    conflict_type TEXT NOT NULL,  -- contradictory, superseded, stale, duplicate
    severity TEXT DEFAULT 'medium',  -- low, medium, high, critical
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    detected_by TEXT,             -- Agent/job that detected
    resolution_status TEXT DEFAULT 'open',  -- open, investigating, resolved, ignored
    resolved_at TIMESTAMPTZ,
    resolved_by TEXT,
    resolution_notes TEXT,
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX IF NOT EXISTS idx_memory_conflicts_status
ON memory_conflicts(resolution_status);

CREATE INDEX IF NOT EXISTS idx_memory_conflicts_memory_a
ON memory_conflicts(memory_id_a);

CREATE INDEX IF NOT EXISTS idx_memory_conflicts_memory_b
ON memory_conflicts(memory_id_b);

-- =============================================================================
-- STEP 8: Create indexes for verification queries
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_unified_ai_memory_verification_state
ON unified_ai_memory(verification_state);

CREATE INDEX IF NOT EXISTS idx_unified_ai_memory_evidence_level
ON unified_ai_memory(evidence_level);

CREATE INDEX IF NOT EXISTS idx_unified_ai_memory_confidence_score
ON unified_ai_memory(confidence_score);

CREATE INDEX IF NOT EXISTS idx_unified_ai_memory_verification_expires
ON unified_ai_memory(verification_expires_at)
WHERE verification_expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_unified_ai_memory_object_type
ON unified_ai_memory(object_type)
WHERE object_type IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_unified_ai_memory_owner
ON unified_ai_memory(owner)
WHERE owner IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_unified_ai_memory_project
ON unified_ai_memory(project)
WHERE project IS NOT NULL;

-- =============================================================================
-- STEP 9: Create function to update verification state based on age
-- =============================================================================

CREATE OR REPLACE FUNCTION update_verification_degradation()
RETURNS void AS $$
BEGIN
    -- Mark verified memories as degraded if past expiration
    UPDATE unified_ai_memory
    SET verification_state = 'DEGRADED',
        confidence_score = GREATEST(confidence_score - 0.1, 0.0),
        updated_at = NOW()
    WHERE verification_state = 'VERIFIED'
      AND verification_expires_at IS NOT NULL
      AND verification_expires_at < NOW();

    -- Reduce confidence for old unverified memories
    UPDATE unified_ai_memory
    SET confidence_score = GREATEST(confidence_score - 0.05, 0.0),
        updated_at = NOW()
    WHERE verification_state = 'UNVERIFIED'
      AND created_at < NOW() - INTERVAL '30 days'
      AND confidence_score > 0.0;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- STEP 10: Create function to check for memory conflicts
-- =============================================================================

CREATE OR REPLACE FUNCTION detect_memory_conflicts()
RETURNS INTEGER AS $$
DECLARE
    conflict_count INTEGER := 0;
BEGIN
    -- Detect duplicate content (same content_hash, both active)
    INSERT INTO memory_conflicts (memory_id_a, memory_id_b, conflict_type, severity, detected_by)
    SELECT DISTINCT ON (LEAST(a.id, b.id), GREATEST(a.id, b.id))
        LEAST(a.id, b.id),
        GREATEST(a.id, b.id),
        'duplicate',
        'medium',
        'detect_memory_conflicts'
    FROM unified_ai_memory a
    JOIN unified_ai_memory b ON a.content_hash = b.content_hash AND a.id < b.id
    WHERE a.verification_state IN ('UNVERIFIED', 'VERIFIED')
      AND b.verification_state IN ('UNVERIFIED', 'VERIFIED')
      AND (a.expires_at IS NULL OR a.expires_at > NOW())
      AND (b.expires_at IS NULL OR b.expires_at > NOW())
      AND NOT EXISTS (
          SELECT 1 FROM memory_conflicts mc
          WHERE (mc.memory_id_a = LEAST(a.id, b.id) AND mc.memory_id_b = GREATEST(a.id, b.id))
            AND mc.resolution_status = 'open'
      );

    GET DIAGNOSTICS conflict_count = ROW_COUNT;
    RETURN conflict_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- STEP 11: Create view for truth backlog (unverified/degraded/conflicting)
-- =============================================================================

CREATE OR REPLACE VIEW memory_truth_backlog AS
SELECT
    m.id,
    m.memory_type,
    m.object_type,
    m.verification_state,
    m.evidence_level,
    m.confidence_score,
    m.verification_expires_at,
    m.last_verified_at,
    m.owner,
    m.project,
    m.created_at,
    m.updated_at,
    CASE
        WHEN m.verification_state = 'BROKEN' THEN 'critical'
        WHEN m.verification_state = 'DEGRADED' THEN 'high'
        WHEN m.verification_state = 'UNVERIFIED' AND m.importance_score > 0.7 THEN 'high'
        WHEN m.verification_state = 'UNVERIFIED' THEN 'medium'
        ELSE 'low'
    END AS priority,
    CASE
        WHEN m.verification_expires_at < NOW() THEN 're-verification required'
        WHEN m.verification_state = 'UNVERIFIED' THEN 'initial verification needed'
        WHEN m.verification_state = 'DEGRADED' THEN 'confidence restoration needed'
        WHEN m.verification_state = 'BROKEN' THEN 'proof repair required'
        ELSE 'no action needed'
    END AS recommended_action,
    (SELECT COUNT(*) FROM memory_conflicts mc
     WHERE (mc.memory_id_a = m.id OR mc.memory_id_b = m.id)
       AND mc.resolution_status = 'open') AS open_conflicts
FROM unified_ai_memory m
WHERE m.verification_state != 'VERIFIED'
   OR m.verification_expires_at < NOW()
   OR m.confidence_score < 0.5
   OR (m.owner IS NULL AND m.importance_score > 0.5)
ORDER BY
    CASE m.verification_state
        WHEN 'BROKEN' THEN 1
        WHEN 'DEGRADED' THEN 2
        WHEN 'UNVERIFIED' THEN 3
        ELSE 4
    END,
    m.importance_score DESC,
    m.created_at DESC;

-- =============================================================================
-- STEP 12: Migrate existing data to UNVERIFIED state
-- =============================================================================

UPDATE unified_ai_memory
SET verification_state = 'UNVERIFIED',
    evidence_level = 'E0_UNVERIFIED',
    confidence_score = 0.3,
    retrieval_policy_version = 'v1.0'
WHERE verification_state IS NULL;

-- =============================================================================
-- STEP 13: Verification summary
-- =============================================================================

DO $$
DECLARE
    total_count INTEGER;
    unverified_count INTEGER;
    verified_count INTEGER;
    degraded_count INTEGER;
    broken_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_count FROM unified_ai_memory;
    SELECT COUNT(*) INTO unverified_count FROM unified_ai_memory WHERE verification_state = 'UNVERIFIED';
    SELECT COUNT(*) INTO verified_count FROM unified_ai_memory WHERE verification_state = 'VERIFIED';
    SELECT COUNT(*) INTO degraded_count FROM unified_ai_memory WHERE verification_state = 'DEGRADED';
    SELECT COUNT(*) INTO broken_count FROM unified_ai_memory WHERE verification_state = 'BROKEN';

    RAISE NOTICE '========================================';
    RAISE NOTICE 'Memory Verification Schema Migration Complete';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Total Memories: %', total_count;
    RAISE NOTICE '  UNVERIFIED: %', unverified_count;
    RAISE NOTICE '  VERIFIED: %', verified_count;
    RAISE NOTICE '  DEGRADED: %', degraded_count;
    RAISE NOTICE '  BROKEN: %', broken_count;
    RAISE NOTICE '';
    RAISE NOTICE 'New Tables Created:';
    RAISE NOTICE '  - memory_verification_artifacts';
    RAISE NOTICE '  - memory_operation_audit';
    RAISE NOTICE '  - memory_conflicts';
    RAISE NOTICE '';
    RAISE NOTICE 'New Views Created:';
    RAISE NOTICE '  - memory_truth_backlog';
    RAISE NOTICE '';
    RAISE NOTICE 'All memories initialized to UNVERIFIED state';
    RAISE NOTICE 'Run verification jobs to establish trust baseline';
END $$;

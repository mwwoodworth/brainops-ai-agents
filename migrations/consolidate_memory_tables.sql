-- Memory Table Consolidation Migration
-- CANONICAL TABLE: unified_ai_memory (23,948+ rows)
-- This migration consolidates data from legacy tables into the canonical table
--
-- Legacy tables:
--   - unified_memory: 3 rows
--   - ai_memory: 1 row
--   - ai_memories: 3 rows
--   - ai_memory_store: 3 rows
--
-- IMPORTANT: This does NOT drop the legacy tables - only migrates data
--
-- Valid memory_type values: 'episodic', 'semantic', 'procedural', 'working', 'meta'

-- Step 1: Migrate from unified_memory (3 rows)
INSERT INTO unified_ai_memory (
    memory_type,
    content,
    source_system,
    source_agent,
    created_by,
    importance_score,
    tags,
    metadata,
    created_at,
    updated_at
)
SELECT
    'semantic'::text as memory_type,
    value as content,
    context_type as source_system,
    context_id as source_agent,
    key as created_by,
    COALESCE(importance::float / 10.0, 0.5) as importance_score,
    COALESCE(tags, ARRAY[]::text[]) as tags,
    jsonb_build_object(
        'migrated_from', 'unified_memory',
        'original_id', id::text,
        'migration_date', NOW()
    ) as metadata,
    created_at,
    updated_at
FROM unified_memory
WHERE NOT EXISTS (
    SELECT 1 FROM unified_ai_memory uam
    WHERE uam.metadata->>'original_id' = unified_memory.id::text
    AND uam.metadata->>'migrated_from' = 'unified_memory'
);

-- Step 2: Migrate from ai_memory (1 row)
-- Map memory_type enum to valid values
INSERT INTO unified_ai_memory (
    memory_type,
    content,
    source_system,
    source_agent,
    created_by,
    importance_score,
    tags,
    metadata,
    created_at,
    updated_at
)
SELECT
    CASE
        WHEN type::text IN ('episodic', 'semantic', 'procedural', 'working', 'meta') THEN type::text
        ELSE 'semantic'
    END as memory_type,
    content,
    COALESCE(session_id, 'legacy') as source_system,
    COALESCE(agent_id, 'migrated') as source_agent,
    'migration' as created_by,
    0.5 as importance_score,
    ARRAY['migrated', 'ai_memory']::text[] as tags,
    jsonb_build_object(
        'migrated_from', 'ai_memory',
        'original_id', id::text,
        'original_metadata', metadata,
        'migration_date', NOW()
    ) as metadata,
    created_at,
    updated_at
FROM ai_memory
WHERE NOT EXISTS (
    SELECT 1 FROM unified_ai_memory uam
    WHERE uam.metadata->>'original_id' = ai_memory.id::text
    AND uam.metadata->>'migrated_from' = 'ai_memory'
);

-- Step 3: Migrate from ai_memories (3 rows)
-- Map memory_type to valid values
INSERT INTO unified_ai_memory (
    memory_type,
    content,
    source_system,
    source_agent,
    created_by,
    importance_score,
    tags,
    metadata,
    expires_at,
    created_at,
    updated_at
)
SELECT
    CASE
        WHEN memory_type IN ('episodic', 'semantic', 'procedural', 'working', 'meta') THEN memory_type
        ELSE 'semantic'
    END as memory_type,
    COALESCE(
        CASE
            WHEN content IS NOT NULL THEN jsonb_build_object('text', content, 'value', value)
            ELSE value
        END,
        '{}'::jsonb
    ) as content,
    'ai_memories' as source_system,
    agent_id as source_agent,
    'migration' as created_by,
    COALESCE(importance::float, 0.5) as importance_score,
    COALESCE(tags, ARRAY[]::text[]) as tags,
    jsonb_build_object(
        'migrated_from', 'ai_memories',
        'original_id', id::text,
        'original_key', key,
        'category', category,
        'original_metadata', metadata,
        'migration_date', NOW()
    ) as metadata,
    expires_at,
    created_at,
    updated_at
FROM ai_memories
WHERE NOT EXISTS (
    SELECT 1 FROM unified_ai_memory uam
    WHERE uam.metadata->>'original_id' = ai_memories.id::text
    AND uam.metadata->>'migrated_from' = 'ai_memories'
);

-- Step 4: Migrate from ai_memory_store (3 rows)
INSERT INTO unified_ai_memory (
    memory_type,
    content,
    source_system,
    source_agent,
    created_by,
    importance_score,
    tags,
    metadata,
    created_at,
    updated_at
)
SELECT
    'semantic'::text as memory_type,
    CASE
        WHEN value ~ '^[\{\[].*[\}\]]$' THEN value::jsonb
        ELSE jsonb_build_object('text', value, 'key', key)
    END as content,
    'ai_memory_store' as source_system,
    'migrated' as source_agent,
    'migration' as created_by,
    0.5 as importance_score,
    ARRAY['migrated', 'ai_memory_store']::text[] as tags,
    jsonb_build_object(
        'migrated_from', 'ai_memory_store',
        'original_id', id::text,
        'original_key', key,
        'original_metadata', metadata,
        'migration_date', NOW()
    ) as metadata,
    created_at,
    updated_at
FROM ai_memory_store
WHERE NOT EXISTS (
    SELECT 1 FROM unified_ai_memory uam
    WHERE uam.metadata->>'original_id' = ai_memory_store.id::text
    AND uam.metadata->>'migrated_from' = 'ai_memory_store'
);

-- Step 5: Create a view for backward compatibility (optional)
-- This allows old code to query unified_memory and get data from unified_ai_memory
CREATE OR REPLACE VIEW unified_memory_view AS
SELECT
    id,
    source_system as context_type,
    source_agent as context_id,
    created_by as key,
    content as value,
    (importance_score * 10)::int as importance,
    tags,
    expires_at,
    created_at,
    updated_at,
    last_accessed as accessed_at,
    access_count
FROM unified_ai_memory;

-- Step 6: Verify migration
DO $$
DECLARE
    unified_count INT;
    ai_mem_count INT;
    ai_memories_count INT;
    ai_store_count INT;
    canonical_count INT;
BEGIN
    SELECT COUNT(*) INTO unified_count FROM unified_memory;
    SELECT COUNT(*) INTO ai_mem_count FROM ai_memory;
    SELECT COUNT(*) INTO ai_memories_count FROM ai_memories;
    SELECT COUNT(*) INTO ai_store_count FROM ai_memory_store;
    SELECT COUNT(*) INTO canonical_count FROM unified_ai_memory;

    RAISE NOTICE 'Migration Summary:';
    RAISE NOTICE '  unified_memory: % rows (migrated)', unified_count;
    RAISE NOTICE '  ai_memory: % rows (migrated)', ai_mem_count;
    RAISE NOTICE '  ai_memories: % rows (migrated)', ai_memories_count;
    RAISE NOTICE '  ai_memory_store: % rows (migrated)', ai_store_count;
    RAISE NOTICE '  CANONICAL unified_ai_memory: % rows (total)', canonical_count;
    RAISE NOTICE '';
    RAISE NOTICE 'NOTE: Legacy tables are preserved but should no longer be used.';
    RAISE NOTICE 'All code should reference unified_ai_memory as the CANONICAL source.';
END $$;

-- Create tables for Codebase Graph System

CREATE TABLE IF NOT EXISTS codebase_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    type TEXT NOT NULL, -- 'file', 'class', 'function', 'endpoint', 'variable'
    file_path TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    line_number INTEGER,
    content_hash TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(repo_name, file_path, name, type)
);

CREATE TABLE IF NOT EXISTS codebase_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES codebase_nodes(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES codebase_nodes(id) ON DELETE CASCADE,
    type TEXT NOT NULL, -- 'imports', 'calls', 'inherits', 'defines', 'contains'
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_id, target_id, type)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_codebase_nodes_repo ON codebase_nodes(repo_name);
CREATE INDEX IF NOT EXISTS idx_codebase_nodes_type ON codebase_nodes(type);
CREATE INDEX IF NOT EXISTS idx_codebase_nodes_file_path ON codebase_nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_codebase_edges_source ON codebase_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_codebase_edges_target ON codebase_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_codebase_edges_type ON codebase_edges(type);

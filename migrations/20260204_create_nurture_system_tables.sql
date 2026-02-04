-- 2026-02-04: Create missing Lead Nurture tables required by CampaignAgent/EmailMarketingAgent.
-- Safe/idempotent for production.

BEGIN;

-- ============================================================================
-- Core tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_lead_enrollments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id UUID REFERENCES revenue_leads(id) ON DELETE CASCADE,
    sequence_id UUID REFERENCES ai_nurture_sequences(id) ON DELETE CASCADE,
    enrollment_date TIMESTAMPTZ DEFAULT NOW(),
    current_touchpoint INT DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    completion_date TIMESTAMPTZ,
    opt_out_date TIMESTAMPTZ,
    engagement_score DOUBLE PRECISION DEFAULT 0.0,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS ai_nurture_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id UUID REFERENCES ai_nurture_sequences(id) ON DELETE CASCADE,
    metric_date DATE NOT NULL,
    enrollments INT DEFAULT 0,
    completions INT DEFAULT 0,
    opt_outs INT DEFAULT 0,
    total_touches INT DEFAULT 0,
    opens INT DEFAULT 0,
    clicks INT DEFAULT 0,
    responses INT DEFAULT 0,
    conversions INT DEFAULT 0,
    revenue_generated NUMERIC(12,2) DEFAULT 0,
    avg_engagement_score DOUBLE PRECISION DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(sequence_id, metric_date)
);

CREATE TABLE IF NOT EXISTS ai_nurture_engagement (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID REFERENCES ai_touchpoint_executions(id) ON DELETE CASCADE,
    engagement_type VARCHAR(50),
    engagement_timestamp TIMESTAMPTZ DEFAULT NOW(),
    engagement_data JSONB DEFAULT '{}'::jsonb,
    lead_score_impact DOUBLE PRECISION DEFAULT 0.0,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS ai_nurture_ab_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id UUID REFERENCES ai_nurture_sequences(id) ON DELETE CASCADE,
    test_name VARCHAR(255),
    variant_a JSONB,
    variant_b JSONB,
    test_metric VARCHAR(50),
    sample_size INT,
    variant_a_results JSONB DEFAULT '{}'::jsonb,
    variant_b_results JSONB DEFAULT '{}'::jsonb,
    winner VARCHAR(1),
    confidence_level DOUBLE PRECISION,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS ai_nurture_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_name VARCHAR(255),
    content_type VARCHAR(50),
    category VARCHAR(100),
    subject_line TEXT,
    body_content TEXT,
    html_content TEXT,
    personalization_fields JSONB DEFAULT '[]'::jsonb,
    performance_score DOUBLE PRECISION DEFAULT 0.5,
    tags JSONB DEFAULT '[]'::jsonb,
    active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_ai_lead_enrollments_lead_id
  ON ai_lead_enrollments(lead_id);
CREATE INDEX IF NOT EXISTS idx_ai_lead_enrollments_sequence_id
  ON ai_lead_enrollments(sequence_id);
CREATE INDEX IF NOT EXISTS idx_ai_lead_enrollments_status
  ON ai_lead_enrollments(status);

CREATE INDEX IF NOT EXISTS idx_ai_nurture_metrics_sequence_date
  ON ai_nurture_metrics(sequence_id, metric_date DESC);

CREATE INDEX IF NOT EXISTS idx_ai_nurture_engagement_timestamp_desc
  ON ai_nurture_engagement(engagement_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_nurture_engagement_type
  ON ai_nurture_engagement(engagement_type);

-- ============================================================================
-- Foreign keys (best-effort / idempotent)
-- ============================================================================

DO $$
BEGIN
  IF NOT EXISTS (
      SELECT 1 FROM pg_constraint WHERE conname = 'ai_touchpoint_executions_enrollment_id_fkey'
  ) THEN
      ALTER TABLE ai_touchpoint_executions
      ADD CONSTRAINT ai_touchpoint_executions_enrollment_id_fkey
      FOREIGN KEY (enrollment_id) REFERENCES ai_lead_enrollments(id) ON DELETE CASCADE;
  END IF;
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
      SELECT 1 FROM pg_constraint WHERE conname = 'ai_sequence_touchpoints_sequence_id_fkey'
  ) THEN
      ALTER TABLE ai_sequence_touchpoints
      ADD CONSTRAINT ai_sequence_touchpoints_sequence_id_fkey
      FOREIGN KEY (sequence_id) REFERENCES ai_nurture_sequences(id) ON DELETE CASCADE;
  END IF;
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

-- ============================================================================
-- RLS (service_role only)
-- ============================================================================

ALTER TABLE ai_lead_enrollments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_nurture_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_nurture_engagement ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_nurture_ab_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_nurture_content ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
      SELECT 1 FROM pg_policies
      WHERE schemaname = 'public' AND tablename = 'ai_lead_enrollments' AND policyname = 'service_role_all'
  ) THEN
    CREATE POLICY service_role_all ON ai_lead_enrollments
      FOR ALL TO service_role USING (true) WITH CHECK (true);
  END IF;

  IF NOT EXISTS (
      SELECT 1 FROM pg_policies
      WHERE schemaname = 'public' AND tablename = 'ai_nurture_metrics' AND policyname = 'service_role_all'
  ) THEN
    CREATE POLICY service_role_all ON ai_nurture_metrics
      FOR ALL TO service_role USING (true) WITH CHECK (true);
  END IF;

  IF NOT EXISTS (
      SELECT 1 FROM pg_policies
      WHERE schemaname = 'public' AND tablename = 'ai_nurture_engagement' AND policyname = 'service_role_all'
  ) THEN
    CREATE POLICY service_role_all ON ai_nurture_engagement
      FOR ALL TO service_role USING (true) WITH CHECK (true);
  END IF;

  IF NOT EXISTS (
      SELECT 1 FROM pg_policies
      WHERE schemaname = 'public' AND tablename = 'ai_nurture_ab_tests' AND policyname = 'service_role_all'
  ) THEN
    CREATE POLICY service_role_all ON ai_nurture_ab_tests
      FOR ALL TO service_role USING (true) WITH CHECK (true);
  END IF;

  IF NOT EXISTS (
      SELECT 1 FROM pg_policies
      WHERE schemaname = 'public' AND tablename = 'ai_nurture_content' AND policyname = 'service_role_all'
  ) THEN
    CREATE POLICY service_role_all ON ai_nurture_content
      FOR ALL TO service_role USING (true) WITH CHECK (true);
  END IF;
END $$;

COMMIT;


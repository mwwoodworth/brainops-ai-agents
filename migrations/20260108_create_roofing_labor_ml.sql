-- Roofing Labor ML (synthetic-friendly) training dataset
-- Stores labeled examples for predicting labor hours / productivity.

CREATE TABLE IF NOT EXISTS public.ml_roofing_labor_samples (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  source TEXT NOT NULL DEFAULT 'manual',
  is_synthetic BOOLEAN NOT NULL DEFAULT FALSE,

  -- Features (inspired by industry research + internal schema)
  roof_type TEXT,
  roof_size_sqft NUMERIC,
  wet_ratio NUMERIC,
  detail_ratio NUMERIC,
  building_height_ft NUMERIC,
  month INTEGER,
  crew_size INTEGER,
  management_style TEXT,
  application_method TEXT,
  roof_life_years NUMERIC,

  -- Target
  labor_hours NUMERIC NOT NULL,

  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

  CONSTRAINT chk_ml_roofing_labor_month CHECK (month IS NULL OR (month >= 1 AND month <= 12)),
  CONSTRAINT chk_ml_roofing_labor_wet_ratio CHECK (wet_ratio IS NULL OR (wet_ratio >= 0 AND wet_ratio <= 1)),
  CONSTRAINT chk_ml_roofing_labor_detail_ratio CHECK (detail_ratio IS NULL OR (detail_ratio >= 0 AND detail_ratio <= 1)),
  CONSTRAINT chk_ml_roofing_labor_roof_size CHECK (roof_size_sqft IS NULL OR roof_size_sqft > 0),
  CONSTRAINT chk_ml_roofing_labor_crew_size CHECK (crew_size IS NULL OR crew_size > 0),
  CONSTRAINT chk_ml_roofing_labor_roof_life CHECK (roof_life_years IS NULL OR roof_life_years >= 0),
  CONSTRAINT chk_ml_roofing_labor_hours CHECK (labor_hours > 0)
);

CREATE INDEX IF NOT EXISTS idx_ml_roofing_labor_samples_created_at
  ON public.ml_roofing_labor_samples (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ml_roofing_labor_samples_tenant
  ON public.ml_roofing_labor_samples (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ml_roofing_labor_samples_is_synthetic
  ON public.ml_roofing_labor_samples (is_synthetic, created_at DESC);

-- RLS: Tenant isolation for PostgREST / client access
ALTER TABLE public.ml_roofing_labor_samples ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ml_roofing_labor_samples FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS ml_roofing_labor_samples_tenant_policy ON public.ml_roofing_labor_samples;
CREATE POLICY ml_roofing_labor_samples_tenant_policy
  ON public.ml_roofing_labor_samples
  FOR ALL
  USING (
    length(current_setting('request.jwt.claim.tenant_id', true)) > 0
    AND tenant_id::text = current_setting('request.jwt.claim.tenant_id', true)
  )
  WITH CHECK (
    length(current_setting('request.jwt.claim.tenant_id', true)) > 0
    AND tenant_id::text = current_setting('request.jwt.claim.tenant_id', true)
  );

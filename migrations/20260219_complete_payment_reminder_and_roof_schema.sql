BEGIN;

-- Ensure reminder tables exist with tenant-safe shape.
CREATE TABLE IF NOT EXISTS public.customer_reminder_settings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id uuid NOT NULL REFERENCES public.customers(id) ON DELETE CASCADE,
    enable_reminders boolean DEFAULT true,
    preferred_channel varchar(20) DEFAULT 'email',
    email_address varchar(255),
    phone_number varchar(20),
    reminder_frequency varchar(20) DEFAULT 'weekly',
    quiet_hours_start time,
    quiet_hours_end time,
    language varchar(10) DEFAULT 'en',
    timezone varchar(50) DEFAULT 'America/New_York',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    tenant_id uuid NOT NULL,
    UNIQUE(customer_id, tenant_id)
);

CREATE TABLE IF NOT EXISTS public.campaign_enrollments (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id uuid NOT NULL REFERENCES public.reminder_campaigns(id) ON DELETE CASCADE,
    invoice_id uuid NOT NULL REFERENCES public.invoices(id) ON DELETE CASCADE,
    enrolled_at timestamptz DEFAULT now(),
    current_step integer DEFAULT 0,
    completed_at timestamptz,
    status varchar(20) DEFAULT 'active',
    tenant_id uuid NOT NULL,
    UNIQUE(campaign_id, invoice_id, tenant_id)
);

-- Create missing payment reminder / tracking tables using production baseline-compatible columns.
CREATE TABLE IF NOT EXISTS public.payment_reminders (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    invoice_id uuid,
    days_offset integer,
    type varchar(50),
    reminder_type varchar(50),
    scheduled_time timestamptz,
    sent_at timestamptz,
    delivered_at timestamptz,
    opened_at timestamptz,
    clicked_at timestamptz,
    channels jsonb,
    template_id uuid,
    custom_message text,
    error_message text,
    metadata jsonb DEFAULT '{}'::jsonb,
    status varchar(50) DEFAULT 'scheduled',
    created_at timestamptz DEFAULT now(),
    tenant_id uuid
);

CREATE TABLE IF NOT EXISTS public.payment_failures (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    invoice_id varchar(255),
    customer_id varchar(255),
    subscription_id varchar(255),
    error_message text,
    created_at timestamptz DEFAULT now(),
    tenant_id uuid
);

CREATE TABLE IF NOT EXISTS public.payment_refunds (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    payment_id uuid,
    refund_date date DEFAULT CURRENT_DATE,
    amount numeric(12,2),
    reason varchar(50),
    notes text,
    transaction_id varchar(100),
    status varchar(20) DEFAULT 'completed',
    gateway_response jsonb,
    created_at timestamptz DEFAULT now(),
    created_by uuid,
    tenant_id uuid
);

CREATE TABLE IF NOT EXISTS public.payment_transactions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id uuid,
    amount numeric(10,2) NOT NULL DEFAULT 0,
    currency varchar(3) DEFAULT 'USD',
    payment_method varchar(50),
    stripe_payment_id varchar(255),
    status varchar(20) DEFAULT 'pending',
    description text,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz DEFAULT now(),
    tenant_id uuid
);

CREATE TABLE IF NOT EXISTS public.payment_gateway_logs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    payment_id uuid,
    gateway varchar(50) NOT NULL DEFAULT 'unknown',
    event_type varchar(100),
    request_data jsonb,
    response_data jsonb,
    status_code integer,
    error_message text,
    created_at timestamptz DEFAULT now(),
    tenant_id uuid
);

-- Add any missing columns for partially-provisioned environments.
ALTER TABLE public.payment_reminders
  ADD COLUMN IF NOT EXISTS reminder_type varchar(50),
  ADD COLUMN IF NOT EXISTS scheduled_time timestamptz,
  ADD COLUMN IF NOT EXISTS sent_at timestamptz,
  ADD COLUMN IF NOT EXISTS delivered_at timestamptz,
  ADD COLUMN IF NOT EXISTS opened_at timestamptz,
  ADD COLUMN IF NOT EXISTS clicked_at timestamptz,
  ADD COLUMN IF NOT EXISTS channels jsonb,
  ADD COLUMN IF NOT EXISTS template_id uuid,
  ADD COLUMN IF NOT EXISTS custom_message text,
  ADD COLUMN IF NOT EXISTS error_message text,
  ADD COLUMN IF NOT EXISTS metadata jsonb,
  ADD COLUMN IF NOT EXISTS tenant_id uuid;

ALTER TABLE public.customer_reminder_settings
  ADD COLUMN IF NOT EXISTS tenant_id uuid;
ALTER TABLE public.campaign_enrollments
  ADD COLUMN IF NOT EXISTS tenant_id uuid;
ALTER TABLE public.payment_failures
  ADD COLUMN IF NOT EXISTS tenant_id uuid;
ALTER TABLE public.payment_refunds
  ADD COLUMN IF NOT EXISTS tenant_id uuid;
ALTER TABLE public.payment_transactions
  ADD COLUMN IF NOT EXISTS tenant_id uuid;
ALTER TABLE public.payment_gateway_logs
  ADD COLUMN IF NOT EXISTS tenant_id uuid;

-- Keep indexes aligned with operational query paths.
CREATE INDEX IF NOT EXISTS idx_customer_reminder_settings_tenant_id ON public.customer_reminder_settings(tenant_id);
CREATE INDEX IF NOT EXISTS idx_campaign_enrollments_tenant_id ON public.campaign_enrollments(tenant_id);
CREATE INDEX IF NOT EXISTS idx_payment_reminders_tenant_id ON public.payment_reminders(tenant_id);
CREATE INDEX IF NOT EXISTS idx_payment_reminders_scheduled_time ON public.payment_reminders(scheduled_time);
CREATE INDEX IF NOT EXISTS idx_payment_reminders_sent_at ON public.payment_reminders(sent_at);
CREATE INDEX IF NOT EXISTS idx_payment_failures_tenant_id ON public.payment_failures(tenant_id);
CREATE INDEX IF NOT EXISTS idx_payment_refunds_tenant_id ON public.payment_refunds(tenant_id);
CREATE INDEX IF NOT EXISTS idx_payment_transactions_tenant_id ON public.payment_transactions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_payment_gateway_logs_tenant_id ON public.payment_gateway_logs(tenant_id);

-- Idempotent FKs to tenants and reminder templates.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'customer_reminder_settings_tenant_id_fkey') THEN
    ALTER TABLE public.customer_reminder_settings
      ADD CONSTRAINT customer_reminder_settings_tenant_id_fkey
      FOREIGN KEY (tenant_id) REFERENCES public.tenants(id) ON DELETE CASCADE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'campaign_enrollments_tenant_id_fkey') THEN
    ALTER TABLE public.campaign_enrollments
      ADD CONSTRAINT campaign_enrollments_tenant_id_fkey
      FOREIGN KEY (tenant_id) REFERENCES public.tenants(id) ON DELETE CASCADE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'payment_reminders_template_id_fkey') THEN
    ALTER TABLE public.payment_reminders
      ADD CONSTRAINT payment_reminders_template_id_fkey
      FOREIGN KEY (template_id) REFERENCES public.reminder_templates(id) ON DELETE SET NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'payment_reminders_tenant_id_fkey') THEN
    ALTER TABLE public.payment_reminders
      ADD CONSTRAINT payment_reminders_tenant_id_fkey
      FOREIGN KEY (tenant_id) REFERENCES public.tenants(id) ON DELETE CASCADE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'payment_failures_tenant_id_fkey') THEN
    ALTER TABLE public.payment_failures
      ADD CONSTRAINT payment_failures_tenant_id_fkey
      FOREIGN KEY (tenant_id) REFERENCES public.tenants(id) ON DELETE CASCADE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'payment_refunds_tenant_id_fkey') THEN
    ALTER TABLE public.payment_refunds
      ADD CONSTRAINT payment_refunds_tenant_id_fkey
      FOREIGN KEY (tenant_id) REFERENCES public.tenants(id) ON DELETE CASCADE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'payment_transactions_tenant_id_fkey') THEN
    ALTER TABLE public.payment_transactions
      ADD CONSTRAINT payment_transactions_tenant_id_fkey
      FOREIGN KEY (tenant_id) REFERENCES public.tenants(id) ON DELETE CASCADE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'payment_gateway_logs_tenant_id_fkey') THEN
    ALTER TABLE public.payment_gateway_logs
      ADD CONSTRAINT payment_gateway_logs_tenant_id_fkey
      FOREIGN KEY (tenant_id) REFERENCES public.tenants(id) ON DELETE CASCADE;
  END IF;
END $$;

-- Backfill tenant_id deterministically from relational links, then fallback to a primary tenant.
DO $$
DECLARE
  primary_tenant uuid;
BEGIN
  SELECT tenant_id INTO primary_tenant
  FROM public.customers
  WHERE tenant_id IS NOT NULL
  GROUP BY tenant_id
  ORDER BY COUNT(*) DESC
  LIMIT 1;

  IF primary_tenant IS NULL THEN
    SELECT id INTO primary_tenant FROM public.tenants ORDER BY created_at ASC LIMIT 1;
  END IF;

  UPDATE public.payment_reminders pr
  SET reminder_type = COALESCE(pr.reminder_type, pr.type)
  WHERE pr.reminder_type IS NULL AND pr.type IS NOT NULL;

  UPDATE public.payment_reminders pr
  SET tenant_id = i.tenant_id
  FROM public.invoices i
  WHERE pr.tenant_id IS NULL AND pr.invoice_id = i.id;

  UPDATE public.customer_reminder_settings crs
  SET tenant_id = c.tenant_id
  FROM public.customers c
  WHERE crs.tenant_id IS NULL AND crs.customer_id = c.id;

  UPDATE public.campaign_enrollments ce
  SET tenant_id = i.tenant_id
  FROM public.invoices i
  WHERE ce.tenant_id IS NULL AND ce.invoice_id = i.id;

  UPDATE public.campaign_enrollments ce
  SET tenant_id = rc.tenant_id
  FROM public.reminder_campaigns rc
  WHERE ce.tenant_id IS NULL AND ce.campaign_id = rc.id;

  UPDATE public.payment_failures pf
  SET tenant_id = i.tenant_id
  FROM public.invoices i
  WHERE pf.tenant_id IS NULL AND pf.invoice_id = i.id::text;

  UPDATE public.payment_failures pf
  SET tenant_id = c.tenant_id
  FROM public.customers c
  WHERE pf.tenant_id IS NULL AND pf.customer_id = c.id::text;

  UPDATE public.payment_transactions pt
  SET tenant_id = c.tenant_id
  FROM public.customers c
  WHERE pt.tenant_id IS NULL AND pt.customer_id = c.id;

  IF to_regclass('public.invoice_payments') IS NOT NULL THEN
    UPDATE public.payment_refunds r
    SET tenant_id = i.tenant_id
    FROM public.invoice_payments p
    JOIN public.invoices i ON i.id = p.invoice_id
    WHERE r.tenant_id IS NULL
      AND r.payment_id = p.id;

    UPDATE public.payment_gateway_logs l
    SET tenant_id = i.tenant_id
    FROM public.invoice_payments p
    JOIN public.invoices i ON i.id = p.invoice_id
    WHERE l.tenant_id IS NULL
      AND l.payment_id = p.id;
  END IF;

  IF primary_tenant IS NOT NULL THEN
    UPDATE public.payment_reminders SET tenant_id = primary_tenant WHERE tenant_id IS NULL;
    UPDATE public.customer_reminder_settings SET tenant_id = primary_tenant WHERE tenant_id IS NULL;
    UPDATE public.campaign_enrollments SET tenant_id = primary_tenant WHERE tenant_id IS NULL;
    UPDATE public.payment_failures SET tenant_id = primary_tenant WHERE tenant_id IS NULL;
    UPDATE public.payment_refunds SET tenant_id = primary_tenant WHERE tenant_id IS NULL;
    UPDATE public.payment_transactions SET tenant_id = primary_tenant WHERE tenant_id IS NULL;
    UPDATE public.payment_gateway_logs SET tenant_id = primary_tenant WHERE tenant_id IS NULL;
  END IF;
END $$;

-- Enforce RLS tenant isolation and service-role operational access.
DO $$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'customer_reminder_settings',
    'campaign_enrollments',
    'payment_reminders',
    'payment_failures',
    'payment_refunds',
    'payment_transactions',
    'payment_gateway_logs'
  ]
  LOOP
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS tenant_isolation_policy ON public.%I', t);
    EXECUTE format(
      'CREATE POLICY tenant_isolation_policy ON public.%I TO authenticated USING (tenant_id = public.current_tenant_id()) WITH CHECK (tenant_id = public.current_tenant_id())',
      t
    );
    EXECUTE format('DROP POLICY IF EXISTS service_role_all ON public.%I', t);
    EXECUTE format(
      'CREATE POLICY service_role_all ON public.%I FOR ALL TO service_role USING (true) WITH CHECK (true)',
      t
    );
  END LOOP;
END $$;

-- Restore missing roof analysis results table.
CREATE TABLE IF NOT EXISTS public.roof_analysis_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    image_hash varchar(64),
    analysis_data jsonb NOT NULL,
    user_id varchar(255),
    ip_address varchar(45),
    is_free_tier boolean DEFAULT false,
    confidence_score double precision,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analysis_image_hash ON public.roof_analysis_results(image_hash);
CREATE INDEX IF NOT EXISTS idx_analysis_user_id ON public.roof_analysis_results(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_created_at ON public.roof_analysis_results(created_at DESC);

ALTER TABLE public.roof_analysis_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS roof_analysis_results_insert_own ON public.roof_analysis_results;
CREATE POLICY roof_analysis_results_insert_own ON public.roof_analysis_results
  FOR INSERT
  WITH CHECK ((auth.uid() IS NOT NULL) AND (auth.uid()::text = user_id::text));
DROP POLICY IF EXISTS roof_analysis_results_select_own ON public.roof_analysis_results;
CREATE POLICY roof_analysis_results_select_own ON public.roof_analysis_results
  FOR SELECT
  USING (((auth.uid() IS NOT NULL) AND (auth.uid()::text = user_id::text)) OR (is_free_tier = true));
DROP POLICY IF EXISTS service_role_all ON public.roof_analysis_results;
CREATE POLICY service_role_all ON public.roof_analysis_results
  FOR ALL TO service_role USING (true) WITH CHECK (true);

COMMIT;

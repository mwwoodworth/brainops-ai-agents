-- Packet 3: Durable awareness state in Supabase/Postgres
-- Moves awareness persistence off ephemeral local files.

CREATE TABLE IF NOT EXISTS public.system_awareness_state (
    component_id text PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('healthy', 'degraded', 'down', 'unknown')),
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    last_heartbeat timestamptz NOT NULL DEFAULT now(),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION public.touch_system_awareness_state_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_touch_system_awareness_state_updated_at ON public.system_awareness_state;
CREATE TRIGGER trg_touch_system_awareness_state_updated_at
BEFORE UPDATE ON public.system_awareness_state
FOR EACH ROW
EXECUTE FUNCTION public.touch_system_awareness_state_updated_at();

CREATE INDEX IF NOT EXISTS idx_system_awareness_state_status
    ON public.system_awareness_state (status);
CREATE INDEX IF NOT EXISTS idx_system_awareness_state_last_heartbeat
    ON public.system_awareness_state (last_heartbeat DESC);

ALTER TABLE public.system_awareness_state ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'system_awareness_state'
          AND policyname = 'system_awareness_state_read_authenticated'
    ) THEN
        CREATE POLICY system_awareness_state_read_authenticated
            ON public.system_awareness_state
            FOR SELECT
            TO authenticated, service_role
            USING (true);
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'system_awareness_state'
          AND policyname = 'system_awareness_state_write_service_role'
    ) THEN
        CREATE POLICY system_awareness_state_write_service_role
            ON public.system_awareness_state
            FOR ALL
            TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END
$$;


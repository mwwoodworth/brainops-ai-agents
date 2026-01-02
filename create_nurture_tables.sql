-- Create missing tables for Lead Nurturing System

-- Core Nurture Sequences (shared schema for all agents)
CREATE TABLE IF NOT EXISTS ai_nurture_sequences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    sequence_name VARCHAR(255),
    sequence_type VARCHAR(50),
    target_segment VARCHAR(100),
    touchpoint_count INT DEFAULT 0,
    days_duration INT,
    success_criteria JSONB DEFAULT '{}'::jsonb,
    configuration JSONB DEFAULT '{}'::jsonb,
    effectiveness_score FLOAT DEFAULT 0.5,
    is_active BOOLEAN DEFAULT TRUE,
    active BOOLEAN DEFAULT TRUE,
    status VARCHAR(50) DEFAULT 'active',
    trigger_type VARCHAR(50) DEFAULT 'manual',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE ai_nurture_sequences
    ADD COLUMN IF NOT EXISTS name TEXT,
    ADD COLUMN IF NOT EXISTS sequence_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS sequence_type VARCHAR(50),
    ADD COLUMN IF NOT EXISTS target_segment VARCHAR(100),
    ADD COLUMN IF NOT EXISTS touchpoint_count INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS days_duration INT,
    ADD COLUMN IF NOT EXISTS success_criteria JSONB DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS configuration JSONB DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS effectiveness_score FLOAT DEFAULT 0.5,
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS trigger_type VARCHAR(50) DEFAULT 'manual',
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

UPDATE ai_nurture_sequences
SET name = COALESCE(name, sequence_name),
    sequence_name = COALESCE(sequence_name, name),
    is_active = COALESCE(is_active, active, TRUE),
    active = COALESCE(active, is_active, TRUE),
    status = COALESCE(status, CASE WHEN COALESCE(is_active, active, TRUE) THEN 'active' ELSE 'inactive' END);

-- Sequence Touchpoints
CREATE TABLE IF NOT EXISTS ai_sequence_touchpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id UUID REFERENCES ai_nurture_sequences(id),
    touchpoint_number INT,
    touchpoint_type VARCHAR(50),
    days_after_trigger INT,
    time_of_day TIME,
    subject_line TEXT,
    content_template TEXT,
    personalization_tokens JSONB DEFAULT '[]',
    call_to_action VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Lead Enrollments
CREATE TABLE IF NOT EXISTS ai_lead_enrollments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id UUID REFERENCES revenue_leads(id), -- Changed to UUID reference
    sequence_id UUID REFERENCES ai_nurture_sequences(id),
    enrollment_date TIMESTAMPTZ DEFAULT NOW(),
    current_touchpoint INT DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    completion_date TIMESTAMPTZ,
    opt_out_date TIMESTAMPTZ,
    engagement_score FLOAT DEFAULT 0.0,
    metadata JSONB DEFAULT '{}'
);

-- Touchpoint Executions
CREATE TABLE IF NOT EXISTS ai_touchpoint_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    enrollment_id UUID REFERENCES ai_lead_enrollments(id),
    touchpoint_id UUID REFERENCES ai_sequence_touchpoints(id),
    scheduled_for TIMESTAMPTZ,
    executed_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'scheduled',
    delivery_channel VARCHAR(50),
    personalized_content TEXT,
    response_data JSONB DEFAULT '{}',
    engagement_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Engagement Tracking
CREATE TABLE IF NOT EXISTS ai_nurture_engagement (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID REFERENCES ai_touchpoint_executions(id),
    engagement_type VARCHAR(50),
    engagement_timestamp TIMESTAMPTZ DEFAULT NOW(),
    engagement_data JSONB DEFAULT '{}',
    lead_score_impact FLOAT DEFAULT 0.0,
    metadata JSONB DEFAULT '{}'
);

-- A/B Tests
CREATE TABLE IF NOT EXISTS ai_nurture_ab_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id UUID REFERENCES ai_nurture_sequences(id),
    test_name VARCHAR(255),
    variant_a JSONB,
    variant_b JSONB,
    test_metric VARCHAR(50),
    sample_size INT,
    variant_a_results JSONB DEFAULT '{}',
    variant_b_results JSONB DEFAULT '{}',
    winner VARCHAR(1),
    confidence_level FLOAT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Nurture Metrics
CREATE TABLE IF NOT EXISTS ai_nurture_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id UUID REFERENCES ai_nurture_sequences(id),
    metric_date DATE,
    enrollments INT DEFAULT 0,
    completions INT DEFAULT 0,
    opt_outs INT DEFAULT 0,
    total_touches INT DEFAULT 0,
    opens INT DEFAULT 0,
    clicks INT DEFAULT 0,
    responses INT DEFAULT 0,
    conversions INT DEFAULT 0,
    revenue_generated DECIMAL(10,2) DEFAULT 0,
    avg_engagement_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(sequence_id, metric_date)
);

ALTER TABLE ai_nurture_metrics
    ADD COLUMN IF NOT EXISTS metric_date DATE,
    ADD COLUMN IF NOT EXISTS enrollments INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS completions INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS opt_outs INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS total_touches INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS opens INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS clicks INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS responses INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS conversions INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS revenue_generated DECIMAL(10,2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_engagement_score FLOAT DEFAULT 0.0;

UPDATE ai_nurture_metrics
SET metric_date = COALESCE(metric_date, date_recorded);

-- Content Library
CREATE TABLE IF NOT EXISTS ai_nurture_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_name VARCHAR(255),
    content_type VARCHAR(50),
    category VARCHAR(100),
    subject_line TEXT,
    body_content TEXT,
    html_content TEXT,
    personalization_fields JSONB DEFAULT '[]',
    performance_score FLOAT DEFAULT 0.5,
    tags JSONB DEFAULT '[]',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Email Sequences (Simple version from revenue_generation_system)
CREATE TABLE IF NOT EXISTS ai_email_sequences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id UUID REFERENCES revenue_leads(id),
    sequence_type VARCHAR(50),
    emails JSONB DEFAULT '[]'::jsonb,
    status VARCHAR(50) DEFAULT 'draft',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    executed_at TIMESTAMPTZ
);

-- Competitor Analysis
CREATE TABLE IF NOT EXISTS ai_competitor_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id UUID REFERENCES revenue_leads(id),
    competitors JSONB DEFAULT '[]'::jsonb,
    analysis JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Churn Predictions
CREATE TABLE IF NOT EXISTS ai_churn_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id UUID REFERENCES revenue_leads(id),
    churn_probability FLOAT DEFAULT 0.0,
    risk_level VARCHAR(20),
    prediction_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Upsell Recommendations
CREATE TABLE IF NOT EXISTS ai_upsell_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id UUID REFERENCES revenue_leads(id),
    recommendations JSONB DEFAULT '[]'::jsonb,
    total_potential FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Revenue Forecasts
CREATE TABLE IF NOT EXISTS ai_revenue_forecasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    months_ahead INT,
    forecast_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Unified Brain Logs
CREATE TABLE IF NOT EXISTS unified_brain_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system VARCHAR(100),
    action VARCHAR(100),
    data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Advanced Lead Metrics
CREATE TABLE IF NOT EXISTS advanced_lead_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id UUID NOT NULL,
    behavioral_score FLOAT DEFAULT 0,
    firmographic_score FLOAT DEFAULT 0,
    intent_score FLOAT DEFAULT 0,
    velocity_score FLOAT DEFAULT 0,
    financial_score FLOAT DEFAULT 0,
    composite_score FLOAT DEFAULT 0,
    tier VARCHAR(20),
    probability_conversion_30d FLOAT,
    expected_deal_size FLOAT,
    next_best_action VARCHAR(100),
    recommended_touch_frequency INT,
    scoring_factors JSONB DEFAULT '{}',
    last_calculated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(lead_id)
);

-- Lead Engagement History
CREATE TABLE IF NOT EXISTS lead_engagement_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB DEFAULT '{}',
    engagement_value FLOAT DEFAULT 0,
    channel VARCHAR(50),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Scoring Model Performance
CREATE TABLE IF NOT EXISTS scoring_model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(50),
    predictions_made INT DEFAULT 0,
    conversions_predicted INT DEFAULT 0,
    actual_conversions INT DEFAULT 0,
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    period_start TIMESTAMPTZ,
    period_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_sequences_type ON ai_nurture_sequences(sequence_type);
CREATE INDEX IF NOT EXISTS idx_enrollments_lead ON ai_lead_enrollments(lead_id);
CREATE INDEX IF NOT EXISTS idx_enrollments_status ON ai_lead_enrollments(status);
CREATE INDEX IF NOT EXISTS idx_executions_scheduled ON ai_touchpoint_executions(scheduled_for);
CREATE INDEX IF NOT EXISTS idx_executions_status ON ai_touchpoint_executions(status);
CREATE INDEX IF NOT EXISTS idx_metrics_date ON ai_nurture_metrics(metric_date);

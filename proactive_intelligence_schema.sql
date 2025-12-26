CREATE TABLE IF NOT EXISTS proactive_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    prediction_type TEXT NOT NULL, -- 'anomaly', 'trend', 'churn_risk'
    predicted_value JSONB,
    confidence_score FLOAT,
    outcome_value JSONB,
    accuracy_score FLOAT,
    model_version TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_proactive_predictions_ts ON proactive_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_proactive_predictions_type ON proactive_predictions(prediction_type);

"""
Roofing Labor ML API
====================
Predict labor hours/productivity using a lightweight RandomForest model trained from
`ml_roofing_labor_samples` (supports both real + synthetic labeled data).
"""

from __future__ import annotations

import pickle
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from database.async_connection import get_pool

router = APIRouter(prefix="/roofing/labor-ml", tags=["Roofing Labor ML"])

MODEL_TYPE = "roofing_labor_hours_regressor"


class LaborSampleIn(BaseModel):
    roof_type: Optional[str] = None
    roof_size_sqft: Optional[float] = Field(None, gt=0)
    wet_ratio: Optional[float] = Field(None, ge=0, le=1)
    detail_ratio: Optional[float] = Field(None, ge=0, le=1)
    building_height_ft: Optional[float] = Field(None, ge=0)
    month: Optional[int] = Field(None, ge=1, le=12)
    crew_size: Optional[int] = Field(None, ge=1)
    management_style: Optional[str] = None
    application_method: Optional[str] = None
    roof_life_years: Optional[float] = Field(None, ge=0)

    labor_hours: float = Field(..., gt=0)

    tenant_id: Optional[str] = None
    source: str = "manual"
    is_synthetic: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LaborPredictIn(BaseModel):
    roof_type: Optional[str] = None
    roof_size_sqft: Optional[float] = Field(None, gt=0)
    wet_ratio: Optional[float] = Field(None, ge=0, le=1)
    detail_ratio: Optional[float] = Field(None, ge=0, le=1)
    building_height_ft: Optional[float] = Field(None, ge=0)
    month: Optional[int] = Field(None, ge=1, le=12)
    crew_size: Optional[int] = Field(None, ge=1)
    management_style: Optional[str] = None
    application_method: Optional[str] = None
    roof_life_years: Optional[float] = Field(None, ge=0)


def _features_dict(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "roof_type": sample.get("roof_type"),
        "roof_size_sqft": sample.get("roof_size_sqft"),
        "wet_ratio": sample.get("wet_ratio"),
        "detail_ratio": sample.get("detail_ratio"),
        "building_height_ft": sample.get("building_height_ft"),
        "month": sample.get("month"),
        "crew_size": sample.get("crew_size"),
        "management_style": sample.get("management_style"),
        "application_method": sample.get("application_method"),
        "roof_life_years": sample.get("roof_life_years"),
    }


async def _get_latest_model_row() -> Optional[dict[str, Any]]:
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT id, model_version, training_data_count, accuracy, parameters, model_data, created_at
        FROM ai_trained_models
        WHERE model_type = $1 AND status = 'completed'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        MODEL_TYPE,
    )
    return dict(row) if row else None


@router.get("/status")
async def get_status() -> dict[str, Any]:
    pool = get_pool()
    sample_counts = await pool.fetchrow(
        """
        SELECT
          COUNT(*)::int AS total,
          SUM(CASE WHEN is_synthetic THEN 1 ELSE 0 END)::int AS synthetic,
          SUM(CASE WHEN NOT is_synthetic THEN 1 ELSE 0 END)::int AS real
        FROM ml_roofing_labor_samples
        """
    )
    latest = await _get_latest_model_row()
    return {
        "system": "roofing_labor_ml",
        "status": "operational",
        "samples": dict(sample_counts) if sample_counts else {"total": 0, "synthetic": 0, "real": 0},
        "latest_model": {
            "id": latest.get("id"),
            "version": latest.get("model_version"),
            "training_data_count": latest.get("training_data_count"),
            "r2": latest.get("accuracy"),
            "created_at": latest.get("created_at").isoformat() if latest and latest.get("created_at") else None,
        }
        if latest
        else None,
    }


@router.post("/samples")
async def create_sample(sample: LaborSampleIn) -> dict[str, Any]:
    pool = get_pool()
    sample_id = str(uuid.uuid4())
    await pool.execute(
        """
        INSERT INTO ml_roofing_labor_samples (
          id, tenant_id, source, is_synthetic,
          roof_type, roof_size_sqft, wet_ratio, detail_ratio, building_height_ft,
          month, crew_size, management_style, application_method, roof_life_years,
          labor_hours, metadata
        )
        VALUES (
          $1, $2, $3, $4,
          $5, $6, $7, $8, $9,
          $10, $11, $12, $13, $14,
          $15, $16
        )
        """,
        sample_id,
        sample.tenant_id,
        sample.source,
        sample.is_synthetic,
        sample.roof_type,
        sample.roof_size_sqft,
        sample.wet_ratio,
        sample.detail_ratio,
        sample.building_height_ft,
        sample.month,
        sample.crew_size,
        sample.management_style,
        sample.application_method,
        sample.roof_life_years,
        sample.labor_hours,
        sample.metadata,
    )
    return {"status": "created", "id": sample_id}


@router.post("/train")
async def train_model(
    include_synthetic: bool = Query(True, description="Include synthetic samples in training"),
    min_samples: int = Query(100, ge=10, le=100000),
    n_estimators: int = Query(300, ge=50, le=2000),
) -> dict[str, Any]:
    """
    Train a RandomForestRegressor and store the model in `ai_trained_models`.

    - Uses `DictVectorizer` for mixed categorical + numeric features.
    - Stores R^2 in `ai_trained_models.accuracy` (regression metric).
    """
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT
          roof_type, roof_size_sqft, wet_ratio, detail_ratio, building_height_ft,
          month, crew_size, management_style, application_method, roof_life_years,
          labor_hours
        FROM ml_roofing_labor_samples
        WHERE ($1::bool OR NOT is_synthetic)
        ORDER BY created_at DESC
        """,
        include_synthetic,
    )
    if not rows or len(rows) < min_samples:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient samples: {len(rows) if rows else 0} (min_samples={min_samples})",
        )

    # Lazy imports (keep app startup light)
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    samples = [dict(r) for r in rows]
    X_dict = [_features_dict(s) for s in samples]
    y = np.array([float(s["labor_hours"]) for s in samples], dtype=float)

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(X_dict)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    # scikit-learn >= 1.8 removed `squared=`; compute RMSE manually for compatibility.
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(mse**0.5)

    # Feature importance (top 25)
    feature_names = list(vec.get_feature_names_out())
    importances = list(model.feature_importances_)
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top_features = [{"feature": n, "importance": float(v)} for n, v in ranked[:25]]

    model_package = {
        "type": MODEL_TYPE,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse, "samples": len(samples), "features": X.shape[1]},
        "vectorizer": vec,
        "model": model,
    }
    model_bytes = pickle.dumps(model_package)

    model_id = str(uuid.uuid4())
    model_version = f"v1.{datetime.now(timezone.utc).strftime('%Y%m%d.%H%M%S')}"
    await pool.execute(
        """
        INSERT INTO ai_trained_models
        (id, model_type, model_version, training_data_count,
         accuracy, precision_score, recall_score, f1_score,
         parameters, model_data, status, deployed)
        VALUES ($1, $2, $3, $4,
                $5, $6, $7, $8,
                $9, $10, 'completed', FALSE)
        """,
        model_id,
        MODEL_TYPE,
        model_version,
        len(samples),
        r2,  # stored in `accuracy` for backwards-compat with existing schema
        0.0,
        0.0,
        0.0,
        {
            "regression": True,
            "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
            "n_estimators": n_estimators,
            "include_synthetic": include_synthetic,
            "top_features": top_features,
        },
        model_bytes,
    )

    return {
        "status": "trained",
        "model_id": model_id,
        "model_version": model_version,
        "samples_used": len(samples),
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "top_features": top_features,
    }


@router.post("/predict")
async def predict(payload: LaborPredictIn) -> dict[str, Any]:
    latest = await _get_latest_model_row()
    if not latest:
        raise HTTPException(status_code=404, detail="No trained model available; run POST /roofing/labor-ml/train first")

    try:
        model_package = pickle.loads(latest["model_data"])
        vec = model_package["vectorizer"]
        model = model_package["model"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}") from e

    import numpy as np

    X = vec.transform([_features_dict(payload.model_dump())])
    mean_hours = float(model.predict(X)[0])

    # Uncertainty estimate via per-tree predictions
    try:
        tree_preds = np.array([est.predict(X)[0] for est in model.estimators_], dtype=float)
        std_hours = float(tree_preds.std())
        p10 = float(np.percentile(tree_preds, 10))
        p90 = float(np.percentile(tree_preds, 90))
    except Exception:
        std_hours = 0.0
        p10 = mean_hours
        p90 = mean_hours

    crew_size = payload.crew_size or None
    crew_days = float(mean_hours / (crew_size * 8)) if crew_size else None

    return {
        "model": {
            "id": latest["id"],
            "version": latest["model_version"],
            "trained_at": model_package.get("trained_at"),
            "r2": latest.get("accuracy"),
        },
        "prediction": {
            "labor_hours": mean_hours,
            "labor_hours_p10": p10,
            "labor_hours_p90": p90,
            "labor_hours_std": std_hours,
            "crew_days": crew_days,
        },
    }

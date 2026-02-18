#!/usr/bin/env python3
"""
Seed synthetic training data for Roofing Labor ML.

Writes to: public.ml_roofing_labor_samples
All rows are flagged `is_synthetic=true` to avoid contaminating real ops data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

import psycopg2
from psycopg2.extras import execute_values


def _db_config() -> dict:
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        parsed = urlparse(database_url)
        return {
            "host": parsed.hostname,
            "database": (parsed.path or "").lstrip("/") or "postgres",
            "user": parsed.username,
            "password": parsed.password,
            "port": parsed.port or 5432,
            "sslmode": "require",
        }

    required = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing DB env vars: {', '.join(missing)} (or set DATABASE_URL)")

    return {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "sslmode": "require" if os.getenv("DB_SSL", "true").lower() not in ("false", "0", "no") else "prefer",
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _synthetic_row(rng: random.Random) -> dict:
    roof_types = ["TPO", "EPDM", "asphalt_shingle", "metal", "modified_bitumen", "built_up"]
    management_styles = ["self_managed", "traditional"]
    application_methods = ["spray", "roll", "mechanically_fastened"]

    roof_type = rng.choice(roof_types)
    roof_size_sqft = rng.uniform(800, 200_000)
    wet_ratio = _clamp(rng.betavariate(2, 6), 0.0, 1.0)  # skew low
    detail_ratio = _clamp(rng.betavariate(2, 4), 0.0, 1.0)
    building_height_ft = rng.uniform(10, 120)
    month = rng.randint(1, 12)
    crew_size = rng.randint(2, 10)
    management_style = rng.choice(management_styles)
    application_method = rng.choice(application_methods)
    roof_life_years = rng.uniform(0, 35)

    base_prod = {
        "asphalt_shingle": 360.0,
        "metal": 290.0,
        "TPO": 310.0,
        "EPDM": 320.0,
        "modified_bitumen": 280.0,
        "built_up": 260.0,
    }.get(roof_type, 300.0)  # sqft per worker-day

    # Seasonality (article: June/July/September positive; Feb/Mar/Nov negative)
    if month in (6, 7, 9):
        month_factor = 1.10
    elif month in (2, 3, 11):
        month_factor = 0.90
    else:
        month_factor = 1.00

    # Wet ratio dominates productivity reduction (article: strong negative corr)
    wet_factor = _clamp(1.0 - 0.70 * wet_ratio, 0.15, 1.05)
    detail_factor = _clamp(1.0 - 0.35 * detail_ratio, 0.20, 1.05)
    height_factor = _clamp(1.0 - 0.06 * (building_height_ft / 80.0), 0.70, 1.02)

    # Roof life: mild positive correlation with productivity (small effect)
    roof_life_factor = _clamp(1.0 + 0.06 * (roof_life_years / 35.0), 0.95, 1.08)

    application_factor = {"spray": 1.06, "roll": 0.98, "mechanically_fastened": 1.00}.get(application_method, 1.0)

    # Management style x crew size interaction (nonlinear)
    crew_eff = 1.0
    if management_style == "self_managed":
        if crew_size <= 4:
            crew_eff *= 1.10
        elif crew_size >= 8:
            crew_eff *= 0.90
        if wet_ratio > 0.5 or detail_ratio > 0.5:
            crew_eff *= 0.92
    else:
        if crew_size >= 6:
            crew_eff *= 1.05
        else:
            crew_eff *= 0.96

    productivity = base_prod * month_factor * wet_factor * detail_factor * height_factor * roof_life_factor * application_factor * crew_eff
    productivity = max(60.0, productivity)  # avoid infinite hours

    # Total labor-hours (crew scaling imperfectly captured via crew_eff)
    labor_hours = (roof_size_sqft / productivity) * 8.0
    labor_hours *= _clamp(1.0 + 0.20 * wet_ratio + 0.10 * detail_ratio, 1.0, 1.6)

    # Noise
    labor_hours *= _clamp(rng.gauss(1.0, 0.10), 0.75, 1.35)

    return {
        "roof_type": roof_type,
        "roof_size_sqft": float(round(roof_size_sqft, 2)),
        "wet_ratio": float(round(wet_ratio, 4)),
        "detail_ratio": float(round(detail_ratio, 4)),
        "building_height_ft": float(round(building_height_ft, 2)),
        "month": int(month),
        "crew_size": int(crew_size),
        "management_style": management_style,
        "application_method": application_method,
        "roof_life_years": float(round(roof_life_years, 2)),
        "labor_hours": float(round(max(1.0, labor_hours), 2)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1500, help="Rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--tenant-id", type=str, default=None, help="Tenant UUID (optional)")
    parser.add_argument("--source", type=str, default="synthetic_v1", help="source label")
    parser.add_argument("--reset-synthetic", action="store_true", help="Delete existing synthetic rows for this source")
    args = parser.parse_args()

    if args.count < 1:
        print("count must be >= 1", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rows = [_synthetic_row(rng) for _ in range(args.count)]

    cfg = _db_config()
    conn = psycopg2.connect(**cfg)
    try:
        with conn, conn.cursor() as cur:
            if args.reset_synthetic:
                cur.execute(
                    "DELETE FROM ml_roofing_labor_samples WHERE is_synthetic = TRUE AND source = %s",
                    (args.source,),
                )

            now = datetime.now(timezone.utc).isoformat()
            values = [
                (
                    args.tenant_id,
                    args.source,
                    True,
                    r["roof_type"],
                    r["roof_size_sqft"],
                    r["wet_ratio"],
                    r["detail_ratio"],
                    r["building_height_ft"],
                    r["month"],
                    r["crew_size"],
                    r["management_style"],
                    r["application_method"],
                    r["roof_life_years"],
                    r["labor_hours"],
                    json.dumps({"generated_at": now, "seed": args.seed, "generator": "seed_roofing_labor_ml_samples.py"}),
                )
                for r in rows
            ]

            execute_values(
                cur,
                """
                INSERT INTO ml_roofing_labor_samples (
                  tenant_id, source, is_synthetic,
                  roof_type, roof_size_sqft, wet_ratio, detail_ratio, building_height_ft,
                  month, crew_size, management_style, application_method, roof_life_years,
                  labor_hours, metadata
                )
                VALUES %s
                """,
                values,
                page_size=500,
            )

        print(f"Seeded {args.count} synthetic rows into ml_roofing_labor_samples (source={args.source})")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())


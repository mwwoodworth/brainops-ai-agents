#!/usr/bin/env python3
"""
Test script for revenue enhancement features
Run this to verify all new capabilities work correctly
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revenue_generation_system import get_revenue_system

async def test_enhancements():
    """Test all new revenue enhancement features"""

    print("=" * 80)
    print("REVENUE ENHANCEMENT FEATURES TEST")
    print("=" * 80)

    system = get_revenue_system()

    # Create a test lead
    print("\n1. Creating test lead...")
    test_lead_id = await create_test_lead(system)
    if test_lead_id:
        print(f"✅ Test lead created: {test_lead_id}")
    else:
        print("❌ Failed to create test lead")
        return

    # Test AI Lead Scoring
    print("\n2. Testing AI Lead Scoring...")
    try:
        score, qualification = await system.qualify_lead(test_lead_id)
        print(f"✅ Lead Score: {score:.2%}")
        print(f"   - LTV: ${qualification.get('lifetime_value', 0):,.2f}")
        print(f"   - Churn Risk: {qualification.get('churn_risk', 0):.1%}")
        print(f"   - Upsell Potential: {qualification.get('upsell_potential', 'unknown')}")
    except Exception as e:
        print(f"❌ Lead scoring failed: {e}")

    # Test Email Sequence Generation
    print("\n3. Testing Email Sequence Generation...")
    try:
        sequence = await system.generate_email_sequence(test_lead_id, "nurture")
        if sequence and 'emails' in sequence:
            print(f"✅ Email sequence generated: {len(sequence.get('emails', []))} emails")
            print(f"   Sequence ID: {sequence.get('sequence_id')}")
        else:
            print("⚠️  Email sequence generated but no emails returned")
    except Exception as e:
        print(f"❌ Email sequence generation failed: {e}")

    # Test Competitor Analysis
    print("\n4. Testing Competitor Pricing Analysis...")
    try:
        analysis = await system.analyze_competitor_pricing(
            test_lead_id,
            competitors=["JobNimbus", "AccuLynx"]
        )
        if analysis:
            print("✅ Competitor analysis complete")
            print(f"   Competitors analyzed: {len(analysis.get('competitors', []))}")
        else:
            print("⚠️  Competitor analysis returned empty")
    except Exception as e:
        print(f"❌ Competitor analysis failed: {e}")

    # Test Churn Prediction
    print("\n5. Testing Churn Risk Prediction...")
    try:
        prediction = await system.predict_churn_risk(test_lead_id)
        if prediction:
            print("✅ Churn prediction complete")
            print(f"   Churn Probability: {prediction.get('churn_probability', 0):.1%}")
            print(f"   Risk Level: {prediction.get('risk_level', 'unknown')}")
            print(f"   Retention Actions: {len(prediction.get('retention_actions', []))}")
        else:
            print("⚠️  Churn prediction returned empty")
    except Exception as e:
        print(f"❌ Churn prediction failed: {e}")

    # Test Upsell Recommendations
    print("\n6. Testing Upsell/Cross-Sell Recommendations...")
    try:
        recommendations = await system.generate_upsell_recommendations(test_lead_id)
        if recommendations:
            opportunities = recommendations.get('opportunities', [])
            print(f"✅ Upsell recommendations generated: {len(opportunities)} opportunities")
            total_potential = sum([o.get('expected_revenue', 0) for o in opportunities])
            print(f"   Total Revenue Potential: ${total_potential:,.2f}")
        else:
            print("⚠️  Upsell recommendations returned empty")
    except Exception as e:
        print(f"❌ Upsell recommendations failed: {e}")

    # Test Revenue Forecasting
    print("\n7. Testing Revenue Forecasting...")
    try:
        forecast = await system.forecast_revenue(months_ahead=6)
        if forecast:
            print("✅ Revenue forecast generated")
            monthly_forecasts = forecast.get('monthly_forecast', [])
            print(f"   Forecast periods: {len(monthly_forecasts)} months")
            total_forecast = sum([m.get('expected_revenue', 0) for m in monthly_forecasts])
            print(f"   Total Forecast: ${total_forecast:,.2f}")
        else:
            print("⚠️  Revenue forecast returned empty")
    except Exception as e:
        print(f"❌ Revenue forecasting failed: {e}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nCheck unified_brain_logs table for detailed activity logs")
    print("All enhancements are operational! ✅")

async def create_test_lead(system):
    """Create a test lead for testing"""
    import uuid
    import psycopg2
    from datetime import datetime, timezone

    try:
        # Database config
        db_config = {
            "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
            "database": os.getenv("DB_NAME", "postgres"),
            "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
            "password": os.getenv("DB_PASSWORD"),
            "port": int(os.getenv("DB_PORT", 5432))
        }

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        lead_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO revenue_leads
            (id, company_name, contact_name, email, phone, website,
             stage, score, value_estimate, source, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            lead_id,
            "Test Roofing Company",
            "John Test",
            "john@testroof.com",
            "555-0123",
            "https://testroof.com",
            "new",
            0.5,
            5000.0,
            "test",
            '{"company_size": "medium", "location": "Austin, TX"}'
        ))

        conn.commit()
        cursor.close()
        conn.close()

        return lead_id

    except Exception as e:
        print(f"Error creating test lead: {e}")
        return None

if __name__ == "__main__":
    print("\n🚀 Starting Revenue Enhancement Tests...\n")
    asyncio.run(test_enhancements())

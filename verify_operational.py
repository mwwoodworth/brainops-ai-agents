#!/usr/bin/env python3
"""
Verify complete operational status - final test
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import json
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

# Database config
DB_CONFIG = {
    'host': 'aws-0-us-east-2.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.yomagoqdmxszqtdwuhab',
    'password': 'Brain0ps2O2S',
    'port': 5432
}

def test_database() -> Tuple[bool, str]:
    """Test database connectivity and operations"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Test connection
        cursor.execute("SELECT 1")
        cursor.fetchone()

        # Test insert with fixed constraint
        test_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO ai_master_context
            (id, context_type, context_key, context_value, importance, is_critical, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (context_type, context_key)
            DO UPDATE SET
                context_value = EXCLUDED.context_value,
                updated_at = NOW()
            RETURNING id
        """, (test_id, 'operational_test', f'test_{datetime.now().isoformat()}',
              json.dumps({'test': True, 'timestamp': datetime.now().isoformat()}), 5, False))

        inserted_id = cursor.fetchone()['id']
        conn.commit()

        # Verify data
        cursor.execute("SELECT * FROM ai_master_context WHERE id = %s", (inserted_id,))
        data = cursor.fetchone()

        # Clean up
        cursor.execute("DELETE FROM ai_master_context WHERE id = %s", (inserted_id,))
        conn.commit()

        cursor.close()
        conn.close()

        return True, "Database fully operational"

    except Exception as e:
        return False, f"Database error: {str(e)}"

def test_ai_agents() -> Tuple[bool, str]:
    """Test AI agents service"""
    try:
        base_url = "https://brainops-ai-agents.onrender.com"

        # Test health
        health_resp = requests.get(f"{base_url}/health", timeout=10)
        health = health_resp.json()

        if health.get('database') != 'connected':
            return False, "AI Agents: Database not connected"

        # Test agents list
        agents_resp = requests.get(f"{base_url}/agents", timeout=10)
        agents = agents_resp.json()

        if agents.get('count', 0) < 1:
            return False, "AI Agents: No agents found"

        # Test memory store
        memory_data = {
            'type': 'test',
            'key': f'test_{uuid.uuid4().hex[:8]}',
            'value': {'operational': True},
            'importance': 5
        }

        store_resp = requests.post(
            f"{base_url}/memory/store",
            json=memory_data,
            timeout=10
        )

        if store_resp.status_code != 200:
            return False, f"AI Agents: Memory store failed ({store_resp.status_code})"

        return True, f"AI Agents operational (v{health.get('version')}, {agents.get('count')} agents)"

    except Exception as e:
        return False, f"AI Agents error: {str(e)}"

def test_erp() -> Tuple[bool, str]:
    """Test ERP frontend"""
    try:
        # Test main site
        main_resp = requests.get("https://myroofgenius.com", timeout=10)

        if main_resp.status_code != 200:
            return False, f"ERP: Main site error ({main_resp.status_code})"

        # Check for client-side errors
        if "Application error" in main_resp.text or "client-side exception" in main_resp.text:
            return False, "ERP: Client-side errors detected"

        # Test API
        api_resp = requests.get("https://myroofgenius.com/api/health", timeout=10)

        if api_resp.status_code >= 500:
            return False, f"ERP API error ({api_resp.status_code})"

        return True, "ERP frontend operational"

    except Exception as e:
        return False, f"ERP error: {str(e)}"

def test_integration() -> Tuple[bool, str]:
    """Test system integration"""
    try:
        # Test AI analysis endpoint
        test_data = {
            'prompt': 'System integration test',
            'context': {'test': True}
        }

        response = requests.post(
            "https://brainops-ai-agents.onrender.com/ai/analyze",
            json=test_data,
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('analysis_id'):
                return True, "Integration working"

        return False, f"Integration failed ({response.status_code})"

    except requests.exceptions.Timeout:
        return False, "Integration timeout (service may be slow on free tier)"
    except Exception as e:
        return False, f"Integration error: {str(e)}"

def main():
    """Run all operational tests"""
    print("\n" + "="*80)
    print("OPERATIONAL STATUS VERIFICATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    tests = [
        ("DATABASE", test_database),
        ("AI AGENTS", test_ai_agents),
        ("ERP FRONTEND", test_erp),
        ("INTEGRATION", test_integration)
    ]

    results = []
    all_operational = True

    for name, test_func in tests:
        print(f"Testing {name}...")
        success, message = test_func()

        if success:
            print(f"  ✅ {message}")
            results.append((name, True, message))
        else:
            print(f"  ❌ {message}")
            results.append((name, False, message))
            all_operational = False

    # Summary
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)

    operational_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    percentage = (operational_count / total_count * 100) if total_count > 0 else 0

    for name, success, message in results:
        icon = "✅" if success else "❌"
        print(f"{icon} {name:15} {message}")

    print("\n" + "="*80)

    if all_operational:
        print("🎉 SYSTEM STATUS: 100% OPERATIONAL")
        print("All systems are functioning correctly!")
    elif percentage >= 75:
        print(f"⚠️  SYSTEM STATUS: {percentage:.0f}% OPERATIONAL")
        print("Most systems working, minor issues present")
    else:
        print(f"❌ SYSTEM STATUS: {percentage:.0f}% OPERATIONAL")
        print("Critical issues detected, immediate attention required")

    print("="*80 + "\n")

    return 0 if all_operational else 1

if __name__ == "__main__":
    sys.exit(main())
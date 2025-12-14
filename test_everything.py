#!/usr/bin/env python3
"""
Complete system test - assume nothing, test everything directly
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import json
import time
from datetime import datetime
import uuid
import subprocess

# Set up environment
os.environ['DB_HOST'] = 'aws-0-us-east-2.pooler.supabase.com'
os.environ['DB_NAME'] = 'postgres'
os.environ['DB_USER'] = 'postgres.yomagoqdmxszqtdwuhab'
os.environ['DB_PASSWORD'] = '<DB_PASSWORD_REDACTED>'
os.environ['DB_PORT'] = '5432'

print("="*80)
print("COMPLETE SYSTEM TEST - ASSUMING NOTHING")
print("="*80)
print(f"Timestamp: {datetime.now().isoformat()}\n")

# Test results
results = {
    'database': {},
    'ai_agents': {},
    'erp': {},
    'integrations': {},
    'data': {}
}

# 1. TEST DATABASE CONNECTION DIRECTLY
print("1. TESTING DATABASE CONNECTION DIRECTLY")
print("-"*40)

try:
    print("   Attempting connection...")
    conn = psycopg2.connect(
        host='aws-0-us-east-2.pooler.supabase.com',
        database='postgres',
        user='postgres.yomagoqdmxszqtdwuhab',
        password='<DB_PASSWORD_REDACTED>',
        port=5432,
        connect_timeout=10
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    print("   ✅ Database connection successful")
    results['database']['connection'] = True

    # Test basic query
    cursor.execute("SELECT version()")
    version = cursor.fetchone()
    print(f"   PostgreSQL version: {version['version'][:50]}...")

    # Count tables
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    table_count = cursor.fetchone()['count']
    print(f"   Total tables: {table_count}")
    results['database']['table_count'] = table_count

    # Check AI tables
    cursor.execute("""
        SELECT COUNT(*) as ai_tables
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name LIKE 'ai_%'
    """)
    ai_tables = cursor.fetchone()['ai_tables']
    print(f"   AI tables: {ai_tables}")
    results['database']['ai_tables'] = ai_tables

    # Check active agents
    cursor.execute("""
        SELECT COUNT(*) as active_agents
        FROM ai_agents
        WHERE status = 'active'
    """)
    agents = cursor.fetchone()['active_agents']
    print(f"   Active AI agents: {agents}")
    results['database']['active_agents'] = agents

    # Check recent executions
    cursor.execute("""
        SELECT COUNT(*) as recent
        FROM agent_executions
        WHERE created_at > NOW() - INTERVAL '24 hours'
    """)
    recent = cursor.fetchone()['recent']
    print(f"   Recent executions (24h): {recent}")
    results['database']['recent_executions'] = recent

    cursor.close()
    conn.close()

except Exception as e:
    print(f"   ❌ Database connection failed: {e}")
    results['database']['connection'] = False
    results['database']['error'] = str(e)

# 2. TEST AI AGENTS SERVICE
print("\n2. TESTING AI AGENTS SERVICE")
print("-"*40)

ai_base_url = "https://brainops-ai-agents.onrender.com"

# Test health endpoint
try:
    print("   Testing /health endpoint...")
    response = requests.get(f"{ai_base_url}/health", timeout=10)
    health_data = response.json()

    if response.status_code == 200:
        print(f"   ✅ Health check passed")
        print(f"      Version: {health_data.get('version')}")
        print(f"      Database: {health_data.get('database')}")
        print(f"      Status: {health_data.get('status')}")
        results['ai_agents']['health'] = True
        results['ai_agents']['version'] = health_data.get('version')
        results['ai_agents']['db_status'] = health_data.get('database')
    else:
        print(f"   ❌ Health check failed: {response.status_code}")
        results['ai_agents']['health'] = False
except Exception as e:
    print(f"   ❌ Health endpoint error: {e}")
    results['ai_agents']['health'] = False
    results['ai_agents']['error'] = str(e)

# Test agents endpoint
try:
    print("   Testing /agents endpoint...")
    response = requests.get(f"{ai_base_url}/agents", timeout=10)

    if response.status_code == 200:
        agents_data = response.json()
        agent_count = agents_data.get('count', 0)
        print(f"   ✅ Agents endpoint working")
        print(f"      Agents returned: {agent_count}")
        results['ai_agents']['agents_endpoint'] = True
        results['ai_agents']['agent_count'] = agent_count
    elif response.status_code == 503:
        print(f"   ⚠️  Agents endpoint: Database unavailable")
        results['ai_agents']['agents_endpoint'] = False
        results['ai_agents']['agents_error'] = 'Database unavailable'
    else:
        print(f"   ❌ Agents endpoint failed: {response.status_code}")
        results['ai_agents']['agents_endpoint'] = False
except Exception as e:
    print(f"   ❌ Agents endpoint error: {e}")
    results['ai_agents']['agents_endpoint'] = False

# Test other endpoints
endpoints_to_test = [
    '/ai/status',
    '/memory/retrieve',
    '/performance/metrics',
    '/ab-test/experiments'
]

for endpoint in endpoints_to_test:
    try:
        response = requests.get(f"{ai_base_url}{endpoint}", timeout=5)
        if response.status_code < 400:
            print(f"   ✅ {endpoint}: {response.status_code}")
            results['ai_agents'][endpoint] = True
        else:
            print(f"   ❌ {endpoint}: {response.status_code}")
            results['ai_agents'][endpoint] = False
    except Exception as e:
        print(f"   ❌ {endpoint}: Error - {str(e)[:50]}")
        results['ai_agents'][endpoint] = False

# 3. TEST ERP FRONTEND
print("\n3. TESTING ERP FRONTEND")
print("-"*40)

erp_url = "https://myroofgenius.com"

try:
    print("   Testing main site...")
    response = requests.get(erp_url, timeout=10)

    if response.status_code == 200:
        print(f"   ✅ Main site responding: {response.status_code}")
        print(f"      Content length: {len(response.content)} bytes")

        # Check for error indicators
        if "Application error" in response.text:
            print(f"   ⚠️  Found 'Application error' in response")
            results['erp']['has_errors'] = True
        else:
            results['erp']['has_errors'] = False

        results['erp']['main_site'] = True
    else:
        print(f"   ❌ Main site error: {response.status_code}")
        results['erp']['main_site'] = False
except Exception as e:
    print(f"   ❌ Main site error: {e}")
    results['erp']['main_site'] = False

# Test API endpoints
api_endpoints = [
    '/api/health',
    '/api/auth/session',
    '/api/customers',
    '/api/ai/status'
]

for endpoint in api_endpoints:
    try:
        response = requests.get(f"{erp_url}{endpoint}", timeout=5)
        if response.status_code < 500:
            print(f"   ✅ {endpoint}: {response.status_code}")
            results['erp'][endpoint] = True
        else:
            print(f"   ❌ {endpoint}: {response.status_code}")
            results['erp'][endpoint] = False
    except Exception as e:
        print(f"   ❌ {endpoint}: Error")
        results['erp'][endpoint] = False

# 4. TEST DATA OPERATIONS
print("\n4. TESTING DATA OPERATIONS")
print("-"*40)

if results['database'].get('connection'):
    try:
        conn = psycopg2.connect(
            host='aws-0-us-east-2.pooler.supabase.com',
            database='postgres',
            user='postgres.yomagoqdmxszqtdwuhab',
            password='<DB_PASSWORD_REDACTED>',
            port=5432
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Test insert
        test_id = str(uuid.uuid4())
        print(f"   Testing data insert with ID: {test_id[:8]}...")

        cursor.execute("""
            INSERT INTO ai_master_context
            (id, context_type, context_key, context_value, importance, is_critical, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (context_type, context_key)
            DO UPDATE SET updated_at = NOW()
            RETURNING id
        """, (test_id, 'test', f'test_{test_id[:8]}', json.dumps({'test': True}), 5, False))

        result_id = cursor.fetchone()['id']
        conn.commit()
        print(f"   ✅ Insert successful: {result_id}")
        results['data']['insert'] = True

        # Test select
        cursor.execute("""
            SELECT * FROM ai_master_context
            WHERE id = %s
        """, (result_id,))

        data = cursor.fetchone()
        if data:
            print(f"   ✅ Select successful")
            results['data']['select'] = True
        else:
            print(f"   ❌ Select failed")
            results['data']['select'] = False

        # Test delete
        cursor.execute("""
            DELETE FROM ai_master_context
            WHERE id = %s
        """, (result_id,))
        conn.commit()
        print(f"   ✅ Delete successful")
        results['data']['delete'] = True

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"   ❌ Data operation failed: {e}")
        results['data']['error'] = str(e)
else:
    print("   ⚠️  Skipping - database not connected")

# 5. TEST AGENT EXECUTION
print("\n5. TESTING AGENT EXECUTION")
print("-"*40)

try:
    print("   Submitting test agent execution...")

    test_data = {
        'prompt': 'Test execution',
        'type': 'test',
        'timestamp': datetime.now().isoformat()
    }

    response = requests.post(
        f"{ai_base_url}/ai/analyze",
        json=test_data,
        timeout=10
    )

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Agent execution successful")
        print(f"      Analysis ID: {result.get('analysis_id', 'N/A')}")
        results['integrations']['agent_execution'] = True
    else:
        print(f"   ❌ Agent execution failed: {response.status_code}")
        results['integrations']['agent_execution'] = False

except Exception as e:
    print(f"   ❌ Agent execution error: {e}")
    results['integrations']['agent_execution'] = False

# 6. TEST LOCAL SERVICES
print("\n6. TESTING LOCAL SERVICES")
print("-"*40)

# Check for running processes
try:
    result = subprocess.run(
        "ps aux | grep -E 'python.*brainops|myroofgenius' | grep -v grep | wc -l",
        shell=True,
        capture_output=True,
        text=True
    )
    process_count = int(result.stdout.strip())
    print(f"   Running processes: {process_count}")
    results['local']['processes'] = process_count
except:
    print("   ❌ Could not check processes")

# Check systemd service
try:
    result = subprocess.run(
        "systemctl status myroofgenius-automation --no-pager 2>/dev/null | head -3",
        shell=True,
        capture_output=True,
        text=True
    )
    if "Active: active" in result.stdout:
        print("   ✅ Automation service: Active")
        results['local']['automation_service'] = True
    else:
        print("   ⚠️  Automation service: Not active")
        results['local']['automation_service'] = False
except:
    print("   ⚠️  Could not check automation service")

# FINAL SUMMARY
print("\n" + "="*80)
print("FINAL TEST SUMMARY")
print("="*80)

# Calculate scores
total_tests = 0
passed_tests = 0

for category, tests in results.items():
    category_passed = 0
    category_total = 0

    for test, result in tests.items():
        if test not in ['error', 'version', 'db_status', 'agent_count']:
            category_total += 1
            total_tests += 1
            if result is True:
                category_passed += 1
                passed_tests += 1

    if category_total > 0:
        percentage = (category_passed / category_total) * 100
        status = "✅" if percentage >= 80 else "⚠️" if percentage >= 50 else "❌"
        print(f"{status} {category.upper():15} {category_passed}/{category_total} ({percentage:.0f}%)")

print("-"*80)
overall_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({overall_percentage:.0f}%)")

# Determine overall status
if overall_percentage >= 90:
    print("\n✅ SYSTEM STATUS: FULLY OPERATIONAL")
elif overall_percentage >= 70:
    print("\n⚠️  SYSTEM STATUS: PARTIALLY OPERATIONAL")
else:
    print("\n❌ SYSTEM STATUS: CRITICAL ISSUES")

# Critical issues
print("\nCRITICAL ISSUES:")
if not results['database'].get('connection'):
    print("  ❌ Database connection failed")
if results['ai_agents'].get('db_status') == 'disconnected':
    print("  ❌ AI Agents service cannot connect to database")
if not results['ai_agents'].get('agents_endpoint'):
    print("  ❌ Agents endpoint not working")
if results['erp'].get('has_errors'):
    print("  ⚠️  ERP frontend showing error messages")

print("\n" + "="*80)

# Exit with appropriate code
sys.exit(0 if overall_percentage >= 90 else 1)
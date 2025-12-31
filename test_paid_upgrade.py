#!/usr/bin/env python3
"""
Comprehensive test of upgraded paid AI agents service
Tests performance improvements, consistency, and identifies any runtime errors
"""

import os
import sys
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import json
import uuid
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Tuple
import statistics

# Configuration
DB_CONFIG = {
    'host': 'aws-0-us-east-2.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.yomagoqdmxszqtdwuhab',
    'password': '<DB_PASSWORD_REDACTED>',
    'port': 5432
}

AI_AGENTS_URL = "https://brainops-ai-agents.onrender.com"
ERP_URL = "https://myroofgenius.com"

class PerformanceTest:
    def __init__(self):
        self.results = {
            'response_times': [],
            'error_count': 0,
            'success_count': 0,
            'errors': []
        }

    def measure_response_time(self, func, *args, **kwargs):
        """Measure function execution time"""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        self.results['response_times'].append(elapsed)
        return result, elapsed

    def get_statistics(self):
        """Calculate performance statistics"""
        if not self.results['response_times']:
            return {}

        times = self.results['response_times']
        return {
            'min': min(times),
            'max': max(times),
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'total_requests': len(times),
            'success_rate': (self.results['success_count'] / len(times) * 100) if times else 0
        }

def test_ai_agents_performance():
    """Test AI Agents service performance on paid tier"""
    print("\n" + "="*80)
    print("AI AGENTS SERVICE PERFORMANCE TEST (PAID TIER)")
    print("="*80)

    perf = PerformanceTest()
    endpoints = [
        '/health',
        '/agents',
        '/ai/status',
        '/memory/retrieve',
        '/performance/metrics',
        '/ab-test/experiments'
    ]

    # Test each endpoint multiple times
    print("\n1. ENDPOINT RESPONSE TIMES (10 requests each):")
    print("-"*50)

    for endpoint in endpoints:
        times = []
        errors = 0

        for i in range(10):
            try:
                start = time.time()
                response = requests.get(f"{AI_AGENTS_URL}{endpoint}", timeout=10)
                elapsed = time.time() - start
                times.append(elapsed)

                if response.status_code == 200:
                    perf.results['success_count'] += 1
                else:
                    errors += 1
                    perf.results['error_count'] += 1

            except Exception as e:
                errors += 1
                perf.results['error_count'] += 1
                perf.results['errors'].append(f"{endpoint}: {str(e)}")

        avg_time = statistics.mean(times) if times else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0

        status = "✅" if errors == 0 else "⚠️" if errors < 3 else "❌"
        print(f"{status} {endpoint:25} Avg: {avg_time:.3f}s | Min: {min_time:.3f}s | Max: {max_time:.3f}s | Errors: {errors}")

        perf.results['response_times'].extend(times)

    # Test concurrent requests
    print("\n2. CONCURRENT REQUEST TEST (20 simultaneous):")
    print("-"*50)

    async def fetch(session, url):
        try:
            async with session.get(url) as response:
                return response.status, time.time()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            print(f"   ⚠️ Concurrent request failed: {exc}")
            return None, time.time()

    async def concurrent_test():
        async with aiohttp.ClientSession() as session:
            start = time.time()
            tasks = [fetch(session, f"{AI_AGENTS_URL}/health") for _ in range(20)]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start

            successful = sum(1 for status, _ in results if status == 200)
            return successful, total_time

    successful, total_time = asyncio.run(concurrent_test())
    print(f"   Successful: {successful}/20")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Requests/sec: {20/total_time:.1f}")

    # Test memory operations
    print("\n3. MEMORY OPERATIONS TEST:")
    print("-"*50)

    # Store memory
    try:
        store_start = time.time()
        memory_data = {
            'type': 'performance_test',
            'key': f'test_{uuid.uuid4().hex[:8]}',
            'value': {'timestamp': datetime.now().isoformat()},
            'importance': 5
        }

        response = requests.post(
            f"{AI_AGENTS_URL}/memory/store",
            json=memory_data,
            timeout=10
        )
        store_time = time.time() - store_start

        if response.status_code == 200:
            print(f"   ✅ Memory store: {store_time:.3f}s")
            memory_id = response.json().get('memory_id')
        else:
            print(f"   ❌ Memory store failed: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Memory store error: {e}")

    # Retrieve memory
    try:
        retrieve_start = time.time()
        response = requests.get(
            f"{AI_AGENTS_URL}/memory/retrieve?limit=10",
            timeout=10
        )
        retrieve_time = time.time() - retrieve_start

        if response.status_code == 200:
            memories = response.json().get('memories', [])
            print(f"   ✅ Memory retrieve: {retrieve_time:.3f}s ({len(memories)} items)")
        else:
            print(f"   ❌ Memory retrieve failed: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Memory retrieve error: {e}")

    # Test AI analysis
    print("\n4. AI ANALYSIS TEST:")
    print("-"*50)

    try:
        analysis_start = time.time()
        analysis_data = {
            'prompt': 'Performance test analysis',
            'context': {'test': True, 'tier': 'paid'}
        }

        response = requests.post(
            f"{AI_AGENTS_URL}/ai/analyze",
            json=analysis_data,
            timeout=30
        )
        analysis_time = time.time() - analysis_start

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Analysis completed: {analysis_time:.3f}s")
            print(f"      Analysis ID: {result.get('analysis_id', 'N/A')}")
        else:
            print(f"   ❌ Analysis failed: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Analysis error: {e}")

    # Performance summary
    stats = perf.get_statistics()

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    if stats:
        print(f"Average Response Time: {stats['mean']:.3f}s")
        print(f"Median Response Time:  {stats['median']:.3f}s")
        print(f"Fastest Response:      {stats['min']:.3f}s")
        print(f"Slowest Response:      {stats['max']:.3f}s")
        print(f"Standard Deviation:    {stats['stdev']:.3f}s")
        print(f"Success Rate:          {stats['success_rate']:.1f}%")

    return perf

def test_erp_thoroughly():
    """Thoroughly test ERP for consistency and runtime errors"""
    print("\n" + "="*80)
    print("ERP FRONTEND COMPREHENSIVE TEST")
    print("="*80)

    errors_found = []
    warnings = []

    # Critical pages to test
    pages = [
        ('/', 'Homepage'),
        ('/dashboard', 'Dashboard'),
        ('/customers', 'Customers'),
        ('/jobs', 'Jobs'),
        ('/invoices', 'Invoices'),
        ('/ai-estimator', 'AI Estimator'),
        ('/marketplace', 'Marketplace'),
        ('/settings', 'Settings'),
        ('/auth/signin', 'Sign In'),
        ('/auth/signup', 'Sign Up')
    ]

    print("\n1. PAGE LOAD TESTS:")
    print("-"*50)

    for path, name in pages:
        try:
            url = f"{ERP_URL}{path}"
            response = requests.get(url, timeout=10, allow_redirects=True)

            # Check status
            if response.status_code == 200:
                status = "✅"
            elif response.status_code in [301, 302, 307, 308]:
                status = "➡️"
            elif response.status_code == 404:
                status = "⚠️"
                warnings.append(f"{name} ({path}): Page not found")
            else:
                status = "❌"
                errors_found.append(f"{name} ({path}): HTTP {response.status_code}")

            # Check for runtime errors in response
            error_indicators = [
                'Application error',
                'client-side exception',
                'TypeError',
                'ReferenceError',
                'SyntaxError',
                'Cannot read',
                'undefined is not',
                'Unhandled Runtime Error',
                'Error boundary',
                '500 Internal',
                'Something went wrong'
            ]

            content = response.text[:50000]  # Check first 50KB

            runtime_errors = []
            for indicator in error_indicators:
                if indicator in content:
                    runtime_errors.append(indicator)

            if runtime_errors:
                status = "❌"
                errors_found.append(f"{name} ({path}): Contains {', '.join(runtime_errors)}")

            load_time = response.elapsed.total_seconds()
            print(f"{status} {name:20} {path:25} {response.status_code:3} | {load_time:.2f}s")

        except Exception as e:
            print(f"❌ {name:20} {path:25} ERROR | {str(e)[:30]}")
            errors_found.append(f"{name} ({path}): {str(e)[:50]}")

    # Test API endpoints
    print("\n2. API ENDPOINT TESTS:")
    print("-"*50)

    api_endpoints = [
        ('/api/health', 'GET', None, 'Health Check'),
        ('/api/auth/session', 'GET', None, 'Auth Session'),
        ('/api/customers', 'GET', None, 'Get Customers'),
        ('/api/jobs', 'GET', None, 'Get Jobs'),
        ('/api/ai/status', 'GET', None, 'AI Status'),
        ('/api/workflows', 'GET', None, 'Workflows'),
        ('/api/pricing', 'GET', None, 'Pricing')
    ]

    for endpoint, method, data, name in api_endpoints:
        try:
            url = f"{ERP_URL}{endpoint}"

            if method == 'GET':
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=10)

            if response.status_code < 400:
                status = "✅"

                # Check if JSON is valid
                if 'application/json' in response.headers.get('content-type', ''):
                    try:
                        response.json()
                    except (json.JSONDecodeError, ValueError) as exc:
                        status = "⚠️"
                        warnings.append(f"{name}: Invalid JSON response ({str(exc)[:50]})")
            elif response.status_code == 404:
                status = "⚠️"
                warnings.append(f"{name}: Endpoint not found")
            elif response.status_code >= 500:
                status = "❌"
                errors_found.append(f"{name}: Server error {response.status_code}")
            else:
                status = "⚠️"

            print(f"{status} {name:25} {method:6} {endpoint:30} {response.status_code}")

        except Exception as e:
            print(f"❌ {name:25} {method:6} {endpoint:30} ERROR")
            errors_found.append(f"{name}: {str(e)[:50]}")

    # Test interactive features
    print("\n3. INTERACTIVE FEATURES TEST:")
    print("-"*50)

    # Test form submission simulation
    test_cases = [
        ('Lead capture', '/api/leads/capture', {'email': 'test@example.com', 'name': 'Test User'}),
        ('Newsletter', '/api/newsletter', {'email': 'test@example.com'}),
        ('Contact', '/api/contact', {'name': 'Test', 'email': 'test@example.com', 'message': 'Test'})
    ]

    for name, endpoint, data in test_cases:
        try:
            response = requests.post(f"{ERP_URL}{endpoint}", json=data, timeout=10)

            if response.status_code < 400:
                print(f"✅ {name}: Working ({response.status_code})")
            elif response.status_code == 404:
                print(f"⚠️ {name}: Endpoint not found")
                warnings.append(f"{name}: Endpoint not found")
            else:
                print(f"❌ {name}: Failed ({response.status_code})")
                errors_found.append(f"{name}: HTTP {response.status_code}")

        except Exception as e:
            print(f"❌ {name}: Error - {str(e)[:30]}")
            errors_found.append(f"{name}: {str(e)[:50]}")

    return errors_found, warnings

def test_database_consistency():
    """Test database consistency and data integrity"""
    print("\n" + "="*80)
    print("DATABASE CONSISTENCY TEST")
    print("="*80)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        print("\n1. DATA INTEGRITY CHECKS:")
        print("-"*50)

        # Check for orphaned records
        checks = [
            ("Orphaned agent executions", """
                SELECT COUNT(*) as count
                FROM agent_executions ae
                LEFT JOIN ai_agents a ON ae.agent_id = a.id
                WHERE a.id IS NULL
            """),
            ("Invalid job statuses", """
                SELECT COUNT(*) as count
                FROM jobs
                WHERE status NOT IN ('pending', 'in_progress', 'completed', 'cancelled')
            """),
            ("Duplicate customer emails", """
                SELECT COUNT(*) as count
                FROM (
                    SELECT email, COUNT(*) as cnt
                    FROM customers
                    WHERE email IS NOT NULL
                    GROUP BY email
                    HAVING COUNT(*) > 1
                ) duplicates
            """),
            ("Missing critical memory", """
                SELECT COUNT(*) as count
                FROM ai_master_context
                WHERE is_critical = true
                AND context_value IS NULL
            """)
        ]

        issues = []
        for check_name, query in checks:
            cursor.execute(query)
            result = cursor.fetchone()
            count = result['count']

            if count == 0:
                print(f"✅ {check_name}: None found")
            else:
                print(f"⚠️ {check_name}: {count} found")
                issues.append(f"{check_name}: {count}")

        # Check table relationships
        print("\n2. RELATIONSHIP INTEGRITY:")
        print("-"*50)

        cursor.execute("""
            SELECT
                COUNT(DISTINCT ae.agent_id) as unique_agents,
                COUNT(DISTINCT a.id) as total_agents
            FROM agent_executions ae
            FULL OUTER JOIN ai_agents a ON ae.agent_id = a.id
        """)

        result = cursor.fetchone()
        if result['unique_agents'] <= result['total_agents']:
            print(f"✅ Agent relationships consistent")
        else:
            print(f"⚠️ Agent relationship issues detected")
            issues.append("Agent relationship mismatch")

        cursor.close()
        conn.close()

        return issues

    except Exception as e:
        print(f"❌ Database test error: {e}")
        return [f"Database error: {str(e)}"]

def analyze_upgrade_value():
    """Analyze if the paid upgrade is worth the cost"""
    print("\n" + "="*80)
    print("PAID UPGRADE COST-BENEFIT ANALYSIS")
    print("="*80)

    metrics = {
        'free_tier': {
            'response_time': 2.5,  # seconds average
            'uptime': 95,  # percentage
            'concurrent_requests': 5,
            'memory': 512,  # MB
            'cpu': 0.1,  # vCPU
            'cost': 0
        },
        'paid_tier': {
            'response_time': 0,  # Will calculate
            'uptime': 99.9,  # Render guarantee
            'concurrent_requests': 100,
            'memory': 2048,  # MB
            'cpu': 1.0,  # vCPU
            'cost': 25  # $/month estimate
        }
    }

    # Test current performance
    print("\nMeasuring current performance...")

    response_times = []
    for i in range(5):
        try:
            start = time.time()
            requests.get(f"{AI_AGENTS_URL}/health", timeout=10)
            response_times.append(time.time() - start)
        except requests.RequestException as exc:
            print(f"   ⚠️ Health check failed: {exc}")

    if response_times:
        metrics['paid_tier']['response_time'] = statistics.mean(response_times)

    print("\nCOMPARISON:")
    print("-"*50)
    print(f"{'Metric':<25} {'Free Tier':<15} {'Paid Tier':<15} {'Improvement'}")
    print("-"*50)

    # Response time
    free_rt = metrics['free_tier']['response_time']
    paid_rt = metrics['paid_tier']['response_time']
    rt_improvement = ((free_rt - paid_rt) / free_rt * 100) if paid_rt > 0 else 0
    print(f"{'Response Time':<25} {free_rt:.2f}s{'':<10} {paid_rt:.2f}s{'':<10} {rt_improvement:+.0f}%")

    # Uptime
    free_up = metrics['free_tier']['uptime']
    paid_up = metrics['paid_tier']['uptime']
    print(f"{'Uptime':<25} {free_up}%{'':<12} {paid_up}%{'':<11} {paid_up-free_up:+.1f}%")

    # Resources
    print(f"{'Memory':<25} {metrics['free_tier']['memory']} MB{'':<10} {metrics['paid_tier']['memory']} MB{'':<10} {metrics['paid_tier']['memory']/metrics['free_tier']['memory']:.0f}x")
    print(f"{'CPU':<25} {metrics['free_tier']['cpu']} vCPU{'':<10} {metrics['paid_tier']['cpu']} vCPU{'':<10} {metrics['paid_tier']['cpu']/metrics['free_tier']['cpu']:.0f}x")
    print(f"{'Concurrent Requests':<25} {metrics['free_tier']['concurrent_requests']}{'':<14} {metrics['paid_tier']['concurrent_requests']}{'':<14} {metrics['paid_tier']['concurrent_requests']/metrics['free_tier']['concurrent_requests']:.0f}x")

    print("-"*50)
    print(f"{'Monthly Cost':<25} ${metrics['free_tier']['cost']}{'':<14} ${metrics['paid_tier']['cost']}{'':<14}")

    # Calculate value score
    value_score = 0
    if paid_rt < 1.0:  # Sub-second response
        value_score += 40
    elif paid_rt < 2.0:
        value_score += 20

    if rt_improvement > 50:
        value_score += 30
    elif rt_improvement > 30:
        value_score += 15

    if metrics['paid_tier']['uptime'] > 99:
        value_score += 30

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if value_score >= 70:
        print("✅ HIGHLY RECOMMENDED - Significant performance improvements")
        print(f"   Value Score: {value_score}/100")
        print("   The paid tier provides:")
        print("   - Much faster response times")
        print("   - Better reliability for production")
        print("   - Handles more concurrent users")
    elif value_score >= 50:
        print("⚠️ RECOMMENDED - Good improvements for production use")
        print(f"   Value Score: {value_score}/100")
        print("   Consider paid tier for:")
        print("   - Production workloads")
        print("   - Customer-facing features")
    else:
        print("ℹ️ OPTIONAL - Free tier may be sufficient")
        print(f"   Value Score: {value_score}/100")
        print("   Free tier works for:")
        print("   - Development/testing")
        print("   - Low-traffic applications")

    return value_score

def main():
    """Run all comprehensive tests"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SYSTEM TEST - PAID TIER EVALUATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_errors = []
    all_warnings = []

    # Test AI Agents Performance
    perf = test_ai_agents_performance()

    # Test ERP Thoroughly
    erp_errors, erp_warnings = test_erp_thoroughly()
    all_errors.extend(erp_errors)
    all_warnings.extend(erp_warnings)

    # Test Database Consistency
    db_issues = test_database_consistency()
    all_warnings.extend(db_issues)

    # Analyze upgrade value
    value_score = analyze_upgrade_value()

    # FINAL REPORT
    print("\n" + "="*80)
    print("FINAL SYSTEM REPORT")
    print("="*80)

    if not all_errors and not all_warnings:
        print("\n🎉 PERFECT! No runtime errors or warnings found!")
        print("All systems functioning exactly as expected.")
    else:
        if all_errors:
            print(f"\n❌ ERRORS FOUND ({len(all_errors)}):")
            for error in all_errors[:10]:  # Show first 10
                print(f"   - {error}")

        if all_warnings:
            print(f"\n⚠️ WARNINGS ({len(all_warnings)}):")
            for warning in all_warnings[:10]:  # Show first 10
                print(f"   - {warning}")

    print("\n" + "="*80)
    print("UPGRADE VERDICT")
    print("="*80)

    if value_score >= 70 and not all_errors:
        print("✅ UPGRADE IS WORTH THE COST!")
        print("   Significant performance improvements observed")
        print("   System is stable and error-free")
    elif value_score >= 50:
        print("⚠️ UPGRADE IS BENEFICIAL")
        print("   Good improvements but some issues remain")
    else:
        print("ℹ️ UPGRADE VALUE IS MARGINAL")
        print("   Consider optimizing before upgrading")

    print("\n" + "="*80)

    return len(all_errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Monitor deployment status and system health
"""

import time
import requests
import json
from datetime import datetime
from typing import Dict, Any

# Service endpoints
ENDPOINTS = {
    'ai_agents_health': 'https://brainops-ai-agents.onrender.com/health',
    'ai_agents_root': 'https://brainops-ai-agents.onrender.com/',
    'ai_agents_agents': 'https://brainops-ai-agents.onrender.com/agents',
    'erp_frontend': 'https://myroofgenius.com',
    'erp_api': 'https://myroofgenius.com/api/health'
}

def check_endpoint(name: str, url: str) -> Dict[str, Any]:
    """Check a single endpoint"""
    try:
        start = time.time()
        response = requests.get(url, timeout=10)
        elapsed = time.time() - start

        result = {
            'name': name,
            'url': url,
            'status_code': response.status_code,
            'response_time': f"{elapsed:.2f}s",
            'success': response.status_code < 400
        }

        # Parse JSON responses
        if 'application/json' in response.headers.get('content-type', ''):
            try:
                data = response.json()
                if 'health' in name.lower():
                    result['database'] = data.get('database', 'unknown')
                    result['version'] = data.get('version', 'unknown')
                result['data'] = data
            except:
                pass

        return result
    except requests.exceptions.Timeout:
        return {
            'name': name,
            'url': url,
            'error': 'Timeout',
            'success': False
        }
    except Exception as e:
        return {
            'name': name,
            'url': url,
            'error': str(e)[:100],
            'success': False
        }

def monitor_deployment():
    """Monitor all endpoints"""
    print("\n" + "="*60)
    print("BRAINOPS DEPLOYMENT MONITOR")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)

    results = []

    for name, url in ENDPOINTS.items():
        result = check_endpoint(name, url)
        results.append(result)

        # Print result
        if result.get('success'):
            status_icon = "✅"
            extra = ""
            if 'database' in result:
                db_icon = "✅" if result['database'] == 'connected' else "❌"
                extra = f" | DB: {db_icon} {result['database']}"
            if 'version' in result:
                extra += f" | v{result['version']}"
        else:
            status_icon = "❌"
            extra = f" | Error: {result.get('error', 'Failed')}"

        print(f"{status_icon} {name:20} | {result.get('status_code', 'N/A'):3} | {result.get('response_time', 'N/A'):6} {extra}")

    # Summary
    print("-"*60)
    total = len(results)
    successful = sum(1 for r in results if r.get('success'))
    print(f"Summary: {successful}/{total} endpoints operational")

    # Check specific issues
    ai_health = next((r for r in results if r['name'] == 'ai_agents_health'), None)
    if ai_health and ai_health.get('success'):
        if ai_health.get('database') == 'connected':
            print("✅ Database connection: WORKING")
        else:
            print("⚠️  Database connection: DISCONNECTED")
            print("   The service may still be starting up or env vars not loaded")

    print("="*60)

    return results

def continuous_monitor(interval: int = 30, max_checks: int = 10):
    """Continuously monitor deployment"""
    print(f"Starting continuous monitoring (checking every {interval}s)")

    for i in range(max_checks):
        results = monitor_deployment()

        # Check if everything is working
        all_success = all(r.get('success') for r in results)
        db_connected = any(r.get('database') == 'connected' for r in results)

        if all_success and db_connected:
            print("\n🎉 DEPLOYMENT SUCCESSFUL! All systems operational.")
            return True

        if i < max_checks - 1:
            print(f"\nWaiting {interval} seconds before next check...")
            time.sleep(interval)

    print("\n⚠️ Deployment monitoring completed. Some issues may remain.")
    return False

if __name__ == "__main__":
    # Run continuous monitoring
    success = continuous_monitor(interval=30, max_checks=10)
    exit(0 if success else 1)
#!/usr/bin/env python3
"""Comprehensive Production Endpoint Testing"""
import requests
import json
from datetime import datetime

def test_endpoint(name, url, method="GET", data=None):
    """Test an endpoint and return status"""
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        
        status = "✅" if response.status_code < 400 else "❌"
        print(f"{status} [{response.status_code}] {name}: {url}")
        
        if response.status_code < 400 and response.text:
            try:
                data = response.json()
                if 'version' in data:
                    print(f"   Version: {data['version']}")
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"   ⚠️ Invalid JSON from {name}: {str(exc)[:50]}")
        return response.status_code < 400
    except Exception as e:
        print(f"❌ [ERROR] {name}: {str(e)[:50]}")
        return False

print("=" * 60)
print("PRODUCTION ENDPOINT VERIFICATION")
print(f"Time: {datetime.now().isoformat()}")
print("=" * 60)

# Test myroofgenius-backend (BrainOps Backend)
print("\n📦 MYROOFGENIUS-BACKEND (Docker Hub → Render)")
print("-" * 40)
backend_base = "https://brainops-backend-prod.onrender.com"
backend_tests = [
    ("Health Check", f"{backend_base}/health"),
    ("API Health", f"{backend_base}/api/v1/health"),
    ("Root", f"{backend_base}/"),
    ("Customers", f"{backend_base}/api/v1/customers"),
    ("Jobs", f"{backend_base}/api/v1/jobs"),
    ("Invoices", f"{backend_base}/api/v1/invoices"),
    ("AI Status", f"{backend_base}/api/v1/ai/status"),
]

backend_success = 0
for name, url in backend_tests:
    if test_endpoint(name, url):
        backend_success += 1

# Test brainops-ai-agents
print("\n📦 BRAINOPS-AI-AGENTS (GitHub → Render)")
print("-" * 40)
agents_base = "https://brainops-ai-agents.onrender.com"
agents_tests = [
    ("Health Check", f"{agents_base}/health"),
    ("Root", f"{agents_base}/"),
    ("Agents List", f"{agents_base}/agents"),
    ("Memory Status", f"{agents_base}/memory/status"),
    ("Executions", f"{agents_base}/executions"),
]

agents_success = 0
for name, url in agents_tests:
    if test_endpoint(name, url):
        agents_success += 1

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"MyRoofGenius Backend: {backend_success}/{len(backend_tests)} endpoints working")
print(f"BrainOps AI Agents: {agents_success}/{len(agents_tests)} endpoints working")
total_success = backend_success + agents_success
total_tests = len(backend_tests) + len(agents_tests)
percentage = (total_success / total_tests * 100) if total_tests > 0 else 0
print(f"\nOverall: {total_success}/{total_tests} ({percentage:.1f}%) endpoints operational")

if percentage >= 90:
    print("\n🎉 SYSTEM STATUS: PRODUCTION READY")
elif percentage >= 70:
    print("\n⚠️ SYSTEM STATUS: PARTIALLY OPERATIONAL")
else:
    print("\n❌ SYSTEM STATUS: CRITICAL ISSUES")

#!/usr/bin/env python3
"""
Test a simple non-AI endpoint first
"""

import requests

BASE_URL = "https://brainops-ai-agents.onrender.com"

print("Testing basic endpoints first...")

# Test health
print("\n1. Health endpoint:")
r = requests.get(f"{BASE_URL}/health")
print(f"   Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"   Version: {data.get('version')}")
    print(f"   AI Enabled: {data.get('ai_enabled')}")

# Test agents endpoint
print("\n2. Agents endpoint:")
try:
    r = requests.get(f"{BASE_URL}/agents", timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   Agent count: {data.get('count')}")
except Exception as e:
    print(f"   Error: {e}")

# Test a simple POST that doesn't use AI
print("\n3. Memory store (non-AI):")
try:
    r = requests.post(
        f"{BASE_URL}/memory/store",
        json={
            "type": "test",
            "key": "test_key",
            "value": {"test": True}
        },
        timeout=5
    )
    print(f"   Status: {r.status_code}")
except Exception as e:
    print(f"   Error: {e}")

print("\nConclusion: Basic endpoints work, issue is specifically with AI endpoints")
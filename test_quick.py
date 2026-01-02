#!/usr/bin/env python3
"""Quick test of production status"""

import requests

url = "https://brainops-ai-agents.onrender.com"

print("Testing production status...")

# Test health
r = requests.get(f"{url}/health")
print(f"Health: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"  Version: {data.get('version')}")
    print(f"  Status: {data.get('status')}")

# Test AI analyze endpoint (should work even with old version)
print("\nTesting AI analyze...")
try:
    r = requests.post(
        f"{url}/ai/analyze",
        json={"prompt": "test"},
        timeout=5
    )
    print(f"AI Analyze: {r.status_code}")
    if r.status_code == 200:
        result = r.json()
        print(f"  Result: {result.get('result', 'No result')[:100]}...")
except Exception as e:
    print(f"AI Analyze: Error - {e}")

# Test new AI endpoints
new_endpoints = ["/ai/chat", "/ai/generate", "/ai/embeddings"]
print("\nTesting new endpoints...")
for endpoint in new_endpoints:
    try:
        if endpoint == "/ai/embeddings":
            r = requests.post(f"{url}{endpoint}", json={"text": "test"}, timeout=3)
        else:
            r = requests.post(f"{url}{endpoint}", json={"prompt": "test", "messages": [{"role": "user", "content": "test"}]}, timeout=3)
        print(f"{endpoint}: {r.status_code}")
    except Exception:
        print(f"{endpoint}: Timeout/Error")

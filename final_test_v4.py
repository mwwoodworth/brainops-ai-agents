#!/usr/bin/env python3
"""
Final test of v4.0.4 with synchronous AI
"""

import requests
import time

BASE_URL = "https://brainops-ai-agents.onrender.com"

print("="*60)
print("FINAL TEST - v4.0.4 SYNCHRONOUS AI")
print("="*60)

# Wait for deployment
print("\nWaiting for v4.0.4...")
for i in range(30):
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            version = data.get('version')
            print(f"Current version: {version}")
            if version == "4.0.4":
                print("✅ v4.0.4 is live!")
                break
    except requests.RequestException as exc:
        print(f"⚠️ Health check failed: {exc}")
    time.sleep(10)

print("\nTesting AI endpoints...")
print("-"*60)

# Test 1: Simple test endpoint
print("\n1. Testing /ai/test endpoint...")
try:
    r = requests.post(
        f"{BASE_URL}/ai/test",
        json={"prompt": "What is 2+2? Answer with just the number."},
        timeout=20
    )
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if data.get('success'):
            print(f"   ✅ SUCCESS! Result: {data.get('result')}")
            print(f"   This is REAL AI working!")
        else:
            print(f"   ❌ Failed: {data.get('error')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Main analyze endpoint
print("\n2. Testing /ai/analyze endpoint...")
try:
    r = requests.post(
        f"{BASE_URL}/ai/analyze",
        json={
            "prompt": "What is the best roofing material for Florida? One sentence.",
            "model": "gpt-3.5-turbo"
        },
        timeout=20
    )
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        result = data.get('result', '')
        if len(result) > 10:
            print(f"   ✅ SUCCESS! AI responded with real answer")
            print(f"   Response: {result[:150]}...")
        else:
            print(f"   ⚠️ Response: {result}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: GPT-4
print("\n3. Testing GPT-4...")
try:
    r = requests.post(
        f"{BASE_URL}/ai/analyze",
        json={
            "prompt": "List 3 signs a roof needs replacement. Be brief.",
            "model": "gpt-4"
        },
        timeout=25
    )
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        result = data.get('result', '')
        if len(result) > 50:
            print(f"   ✅ GPT-4 WORKS!")
            print(f"   Response preview: {result[:200]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)

print("""
If tests pass:
   🎉 REAL AI IS WORKING IN PRODUCTION!
   ✨ GPT-4 and GPT-3.5 are operational
   ✨ Synchronous implementation fixed the timeout issues
   ✨ Your system has genuine AI power!

If tests fail:
   - Check Render logs for errors
   - Ensure API keys are set in environment
   - May need to wait for deployment to complete
""")

print("="*60)

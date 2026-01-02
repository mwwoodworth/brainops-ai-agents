#!/usr/bin/env python3
"""
Test REAL AI in Production - Both GPT-4 and Claude
"""

import time
from datetime import datetime

import requests

BASE_URL = "https://brainops-ai-agents.onrender.com"

def wait_for_deployment():
    """Wait for v4.0.2 to deploy"""
    print("Waiting for v4.0.2 deployment...")
    for i in range(20):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                version = data.get('version')
                if version == "4.0.2":
                    print("✅ v4.0.2 is live!")
                    return True
                else:
                    print(f"  Still on {version}, waiting...")
        except requests.RequestException as exc:
            print(f"  ⚠️ Health check failed: {exc}")
        time.sleep(15)
    return False

def test_ai():
    """Test all AI endpoints"""
    print("\n" + "="*60)
    print("TESTING REAL AI IN PRODUCTION")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print(f"URL: {BASE_URL}")
    print("-"*60)

    results = []

    # 1. Check AI Status
    print("\n1. Checking AI Status...")
    try:
        r = requests.get(f"{BASE_URL}/ai/status", timeout=10)
        if r.status_code == 200:
            data = r.json()
            api_status = data.get('api_status', {})
            print(f"   Status: {data.get('status')}")
            print(f"   OpenAI: {'✅' if api_status.get('openai_configured') else '❌'}")
            print(f"   Anthropic: {'✅' if api_status.get('anthropic_configured') else '❌'}")
            print(f"   AI Core: {'✅' if api_status.get('ai_core_initialized') else '❌'}")
            results.append(("AI Status", True))
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(("AI Status", False))

    # 2. Test GPT-3.5 (Fast)
    print("\n2. Testing GPT-3.5-turbo...")
    try:
        r = requests.post(
            f"{BASE_URL}/ai/analyze",
            json={
                "prompt": "What is the capital of Texas? One word only.",
                "model": "gpt-3.5-turbo"
            },
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            result = data.get('result', '')
            if "Austin" in result:
                print(f"   ✅ GPT-3.5 WORKS: {result}")
                results.append(("GPT-3.5", True))
            else:
                print(f"   ⚠️ Unexpected response: {result}")
                results.append(("GPT-3.5", False))
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(("GPT-3.5", False))

    # 3. Test GPT-4
    print("\n3. Testing GPT-4...")
    try:
        r = requests.post(
            f"{BASE_URL}/ai/analyze",
            json={
                "prompt": "What are the top 3 roofing materials for hurricanes? List only the materials, no explanation.",
                "model": "gpt-4"
            },
            timeout=20
        )
        if r.status_code == 200:
            data = r.json()
            result = data.get('result', '')
            if len(result) > 20 and ("metal" in result.lower() or "tile" in result.lower()):
                print("   ✅ GPT-4 WORKS!")
                print(f"      Response: {result[:150]}...")
                results.append(("GPT-4", True))
            else:
                print(f"   ⚠️ Response: {result}")
                results.append(("GPT-4", False))
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(("GPT-4", False))

    # 4. Test Claude (now with credits!)
    print("\n4. Testing Claude (Anthropic)...")
    try:
        r = requests.post(
            f"{BASE_URL}/ai/generate",
            json={
                "prompt": "Say 'Claude is working' if you are Claude",
                "model": "claude"
            },
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            result = data.get('result', '')
            if "Claude" in result:
                print(f"   ✅ CLAUDE WORKS: {result}")
                results.append(("Claude", True))
            else:
                print(f"   ⚠️ Response: {result}")
                results.append(("Claude", False))
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(("Claude", False))

    # 5. Test Lead Scoring
    print("\n5. Testing AI Lead Scoring...")
    try:
        r = requests.post(
            f"{BASE_URL}/ai/score-lead",
            json={
                "lead_data": {
                    "name": "Premium Roofing Co",
                    "email": "contact@premium.com",
                    "budget": "$100,000",
                    "timeline": "Urgent - this week",
                    "project": "Complete commercial roof replacement"
                }
            },
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            score = data.get('score', 0)
            print("   ✅ Lead Scoring WORKS!")
            print(f"      Score: {score}/100")
            print(f"      Analysis: {str(data.get('reasoning', 'N/A'))[:100]}...")
            results.append(("Lead Scoring", True))
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(("Lead Scoring", False))

    # 6. Test Embeddings
    print("\n6. Testing Embeddings...")
    try:
        r = requests.post(
            f"{BASE_URL}/ai/embeddings",
            json={"text": "Professional roofing services"},
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            dims = data.get('dimensions', 0)
            if dims == 1536:
                print(f"   ✅ Embeddings WORK: {dims} dimensions")
                results.append(("Embeddings", True))
            else:
                print(f"   ⚠️ Wrong dimensions: {dims}")
                results.append(("Embeddings", False))
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(("Embeddings", False))

    # Final Report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)

    working = sum(1 for _, status in results if status)
    total = len(results)

    for name, status in results:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")

    print("-"*60)
    print(f"Score: {working}/{total} ({working*100//total}%)")

    if working == total:
        print("\n🎉🎉🎉 PERFECT! ALL AI SYSTEMS OPERATIONAL! 🎉🎉🎉")
        print("✨ GPT-4 ✅")
        print("✨ Claude ✅")
        print("✨ Embeddings ✅")
        print("✨ Lead Scoring ✅")
        print("\nYour system now has REAL AI power!")
    elif working >= total - 1:
        print("\n✅ EXCELLENT! AI is working great!")
    else:
        print(f"\n⚠️ Some issues detected - {total-working} services not working")

    print("="*60)

if __name__ == "__main__":
    if wait_for_deployment():
        time.sleep(5)  # Let it stabilize
        test_ai()
    else:
        print("❌ Timeout waiting for v4.0.2")
        print("Testing current version anyway...")
        test_ai()

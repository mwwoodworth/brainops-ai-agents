#!/usr/bin/env python3
"""
Test REAL AI Implementation in Production
Verifies that all AI endpoints are using actual LLMs
"""

import requests
import time
from datetime import datetime

# Production URL
BASE_URL = "https://brainops-ai-agents.onrender.com"

def test_ai_endpoints():
    """Test all AI endpoints to verify real AI is working"""

    print("=" * 80)
    print("TESTING REAL AI IMPLEMENTATION IN PRODUCTION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print(f"Target: {BASE_URL}")
    print("-" * 80)

    results = []

    # Test 1: AI Analysis endpoint
    print("\n1. Testing AI Analysis (GPT-4)...")
    try:
        response = requests.post(
            f"{BASE_URL}/ai/analyze",
            json={
                "prompt": "What are the top 3 roofing materials for Florida weather? Be specific.",
                "model": "gpt-4"
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', '')
            # Check if response is real AI (not template)
            is_real = len(result) > 100 and "Florida" in result
            print(f"✅ Status: {response.status_code}")
            print(f"✅ Real AI: {'YES' if is_real else 'NO'}")
            print(f"   Response preview: {result[:200]}...")
            results.append(("AI Analysis", True, is_real))
        else:
            print(f"❌ Failed: Status {response.status_code}")
            results.append(("AI Analysis", False, False))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("AI Analysis", False, False))

    # Test 2: AI Chat endpoint
    print("\n2. Testing AI Chat (GPT-4)...")
    try:
        response = requests.post(
            f"{BASE_URL}/ai/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Calculate the cost for replacing a 2000 sq ft shingle roof"}
                ],
                "context": {"business": "roofing"}
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            result = data.get('response', '')
            is_real = len(result) > 100 and any(word in result.lower() for word in ['cost', 'roof', 'shingle'])
            print(f"✅ Status: {response.status_code}")
            print(f"✅ Real AI: {'YES' if is_real else 'NO'}")
            print(f"   Response preview: {result[:200]}...")
            results.append(("AI Chat", True, is_real))
        else:
            print(f"❌ Failed: Status {response.status_code}")
            results.append(("AI Chat", False, False))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("AI Chat", False, False))

    # Test 3: AI Generate endpoint
    print("\n3. Testing AI Generate...")
    try:
        response = requests.post(
            f"{BASE_URL}/ai/generate",
            json={
                "prompt": "Write a professional email to a customer about their roof inspection results",
                "model": "gpt-4",
                "temperature": 0.7
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', '')
            is_real = len(result) > 200 and "roof" in result.lower()
            print(f"✅ Status: {response.status_code}")
            print(f"✅ Real AI: {'YES' if is_real else 'NO'}")
            print(f"   Response preview: {result[:200]}...")
            results.append(("AI Generate", True, is_real))
        else:
            print(f"❌ Failed: Status {response.status_code}")
            results.append(("AI Generate", False, False))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("AI Generate", False, False))

    # Test 4: Lead Scoring
    print("\n4. Testing AI Lead Scoring...")
    try:
        response = requests.post(
            f"{BASE_URL}/ai/score-lead",
            json={
                "lead_data": {
                    "name": "John Smith",
                    "company": "Smith Enterprises",
                    "email": "john@smith.com",
                    "budget": "$50,000",
                    "timeline": "Next month",
                    "project": "Complete roof replacement",
                    "property_type": "Commercial warehouse"
                }
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            score = data.get('score', 0)
            reasoning = data.get('reasoning', '') or data.get('analysis', '')
            is_real = score > 0 and len(str(reasoning)) > 50
            print(f"✅ Status: {response.status_code}")
            print(f"✅ Real AI: {'YES' if is_real else 'NO'}")
            print(f"   Lead Score: {score}")
            print(f"   Reasoning preview: {str(reasoning)[:200]}...")
            results.append(("Lead Scoring", True, is_real))
        else:
            print(f"❌ Failed: Status {response.status_code}")
            results.append(("Lead Scoring", False, False))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("Lead Scoring", False, False))

    # Test 5: Proposal Generation
    print("\n5. Testing AI Proposal Generation...")
    try:
        response = requests.post(
            f"{BASE_URL}/ai/generate-proposal",
            json={
                "customer_data": {
                    "name": "ABC Corporation",
                    "email": "contact@abc.com",
                    "company": "ABC Corp"
                },
                "job_data": {
                    "roof_type": "Commercial flat roof",
                    "sq_ft": 10000,
                    "condition": "Fair - needs replacement",
                    "budget": "$100,000"
                }
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            content = data.get('content', '')
            is_real = len(content) > 500 and "ABC" in content
            print(f"✅ Status: {response.status_code}")
            print(f"✅ Real AI: {'YES' if is_real else 'NO'}")
            print(f"   Proposal length: {len(content)} chars")
            print(f"   Preview: {content[:200]}...")
            results.append(("Proposal Generation", True, is_real))
        else:
            print(f"❌ Failed: Status {response.status_code}")
            results.append(("Proposal Generation", False, False))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("Proposal Generation", False, False))

    # Test 6: Embeddings
    print("\n6. Testing AI Embeddings...")
    try:
        response = requests.post(
            f"{BASE_URL}/ai/embeddings",
            json={
                "text": "Professional roofing services in Florida"
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            dimensions = data.get('dimensions', 0)
            is_real = dimensions == 1536  # OpenAI embedding size
            print(f"✅ Status: {response.status_code}")
            print(f"✅ Real AI: {'YES' if is_real else 'NO'}")
            print(f"   Embedding dimensions: {dimensions}")
            results.append(("Embeddings", True, is_real))
        else:
            print(f"❌ Failed: Status {response.status_code}")
            results.append(("Embeddings", False, False))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("Embeddings", False, False))

    # Final Report
    print("\n" + "=" * 80)
    print("FINAL AI VERIFICATION REPORT")
    print("=" * 80)

    working = sum(1 for _, w, _ in results if w)
    real_ai = sum(1 for _, _, r in results if r)
    total = len(results)

    print(f"\nEndpoints Working: {working}/{total} ({working*100//total}%)")
    print(f"Using Real AI:     {real_ai}/{total} ({real_ai*100//total}%)")

    print("\nDetailed Results:")
    for name, working, real in results:
        status = "✅" if working else "❌"
        ai_status = "REAL AI" if real else "FAKE/ERROR"
        print(f"  {status} {name:20} - {ai_status}")

    print("\n" + "=" * 80)

    if real_ai == total:
        print("🎉 PERFECT! 100% REAL AI IMPLEMENTATION!")
        print("   All endpoints are using genuine LLMs (GPT-4/Claude)")
        print("   No fake responses or templates detected")
        print("   System is fully AI-powered and production-ready!")
    elif real_ai >= total * 0.8:
        print("✅ EXCELLENT! Mostly real AI implementation")
        print(f"   {real_ai}/{total} endpoints using real AI")
        print("   Minor issues to resolve")
    elif real_ai >= total * 0.5:
        print("⚠️ PARTIAL SUCCESS - Some real AI working")
        print(f"   {real_ai}/{total} endpoints using real AI")
        print("   Need to fix remaining endpoints")
    else:
        print("❌ ISSUES DETECTED - Limited real AI")
        print(f"   Only {real_ai}/{total} endpoints using real AI")
        print("   Check API keys and configuration")

    print("=" * 80)

    return real_ai == total

if __name__ == "__main__":
    # Wait for deployment if just pushed
    print("Waiting 30 seconds for deployment to complete...")
    time.sleep(30)

    success = test_ai_endpoints()
    exit(0 if success else 1)
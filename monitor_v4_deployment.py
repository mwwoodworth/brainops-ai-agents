#!/usr/bin/env python3
"""
Monitor v4.0.1 deployment with REAL AI
"""

import requests
import time
from datetime import datetime

BASE_URL = "https://brainops-ai-agents.onrender.com"

def check_deployment():
    """Check if v4.0.1 is deployed"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            version = data.get('version', 'unknown')
            ai_enabled = data.get('ai_enabled', False)
            build = data.get('build', 'unknown')

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
            print(f"  Version: {version}")
            print(f"  Build: {build}")
            print(f"  AI Enabled: {'✅ YES' if ai_enabled else '❌ NO'}")

            if data.get('features', {}).get('gpt4'):
                print(f"  GPT-4: ✅ Available")
            else:
                print(f"  GPT-4: ❌ Not configured")

            if data.get('features', {}).get('claude'):
                print(f"  Claude: ✅ Available")
            else:
                print(f"  Claude: ❌ Not configured")

            if version == "4.0.1" and ai_enabled:
                return True
            return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection error: {e}")
        return False

def test_real_ai():
    """Test if AI is really working"""
    print("\n" + "="*60)
    print("TESTING REAL AI...")
    print("="*60)

    try:
        # Test AI status endpoint first
        response = requests.get(f"{BASE_URL}/ai/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\nAI Status:")
            print(f"  Status: {data.get('status')}")
            print(f"  Version: {data.get('version')}")
            api_status = data.get('api_status', {})
            print(f"  OpenAI Key: {'✅' if api_status.get('openai_configured') else '❌'}")
            print(f"  Anthropic Key: {'✅' if api_status.get('anthropic_configured') else '❌'}")
            print(f"  AI Core: {'✅' if api_status.get('ai_core_initialized') else '❌'}")

        # Quick AI test
        print("\nTesting AI Analysis...")
        response = requests.post(
            f"{BASE_URL}/ai/analyze",
            json={"prompt": "What is 2+2?"},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            result = data.get('result', '')
            status = data.get('status', '')

            if status == "degraded":
                print("⚠️ AI is degraded - check API keys")
                print(f"   Message: {result}")
                return False
            elif len(result) > 10 and "4" in str(result):
                print("✅ REAL AI IS WORKING!")
                print(f"   Response: {result[:100]}...")
                return True
            else:
                print("❌ Response doesn't look like real AI")
                print(f"   Response: {result}")
                return False
        else:
            print(f"❌ AI endpoint returned status {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error testing AI: {e}")
        return False

def main():
    print("="*60)
    print("MONITORING v4.0.1 DEPLOYMENT - REAL AI")
    print("="*60)
    print(f"Target: {BASE_URL}")
    print(f"Started: {datetime.now()}")
    print("-"*60)

    # Wait for deployment
    max_attempts = 40  # 10 minutes
    for i in range(max_attempts):
        if check_deployment():
            print("\n" + "="*60)
            print("🎉 v4.0.1 DEPLOYED!")
            print("="*60)

            # Test AI
            time.sleep(5)  # Give it a moment to stabilize
            if test_real_ai():
                print("\n" + "🎉"*20)
                print("SUCCESS! REAL AI IS LIVE IN PRODUCTION!")
                print("🎉"*20)
            else:
                print("\n⚠️ Deployment successful but AI needs configuration")
                print("Check that API keys are set in Render environment")
            return

        time.sleep(15)

    print("\n⏰ Timeout waiting for deployment")
    print("Check Render dashboard for build status")

if __name__ == "__main__":
    main()
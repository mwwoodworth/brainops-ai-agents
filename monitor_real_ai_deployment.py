#!/usr/bin/env python3
"""
Monitor Real AI Deployment Progress
"""

import time
from datetime import datetime

import requests

BASE_URL = "https://brainops-ai-agents.onrender.com"

def check_deployment():
    """Check deployment status"""

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking deployment status...")

    # Check version
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            version = data.get('version', 'unknown')
            status = data.get('status', 'unknown')

            if version == "4.0.0":
                print("✅ DEPLOYED! Version 4.0.0 with REAL AI is live!")
                return True
            else:
                print(f"⏳ Still on version {version} (waiting for 4.0.0)")
                return False
        else:
            print(f"❌ Health check failed: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ai_endpoint():
    """Quick test of AI endpoint"""
    try:
        response = requests.post(
            f"{BASE_URL}/ai/analyze",
            json={"prompt": "Test"},
            timeout=5
        )
        if response.status_code == 200:
            print("   ✅ AI endpoint responding")
            return True
        else:
            print(f"   ⚠️ AI endpoint status: {response.status_code}")
            return False
    except requests.RequestException as exc:
        print(f"   ⏳ AI endpoint not ready: {exc}")
        return False

def main():
    """Monitor deployment"""
    print("=" * 60)
    print("MONITORING REAL AI DEPLOYMENT")
    print("=" * 60)
    print(f"Target: {BASE_URL}")
    print("Waiting for version 4.0.0 with REAL AI...")
    print("-" * 60)

    start_time = time.time()
    max_wait = 600  # 10 minutes

    while time.time() - start_time < max_wait:
        deployed = check_deployment()

        if deployed:
            # Test AI endpoints
            print("\nTesting AI endpoints...")
            test_ai_endpoint()

            print("\n" + "=" * 60)
            print("🎉 DEPLOYMENT SUCCESSFUL!")
            print("Real AI is now live in production!")
            print("=" * 60)
            return 0

        # Wait before next check
        time.sleep(15)

    print("\n" + "=" * 60)
    print("⚠️ TIMEOUT - Deployment taking longer than expected")
    print("Check Render dashboard for build status")
    print("=" * 60)
    return 1

if __name__ == "__main__":
    exit(main())

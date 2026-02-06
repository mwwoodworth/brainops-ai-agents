#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment
# Try explicit paths
env_paths = [
    '.env',
    '.env.local',
    '/home/matt-woodworth/dev/_secure/BrainOps.env',
    '/home/matt-woodworth/dev/weathercraft-erp/.env.local',
    '/home/matt-woodworth/dev/brainops-command-center/.env.local'
]

env_loaded = False
for path in env_paths:
    if os.path.exists(path):
        print(f"Loading env from: {path}")
        load_dotenv(path)
        env_loaded = True

if not env_loaded:
    print("⚠️  Warning: No .env file found in standard locations.")

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
# Try multiple keys
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: Supabase credentials not found in environment.")
    print("Checked paths:", env_paths)
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configuration: Target Systems
# Running against PRODUCTION as per "Live Prod" mandate
SYSTEMS = [
    {
        "id": "weathercraft-erp",
        "name": "Weathercraft ERP (PROD)",
        "path": "/home/matt-woodworth/dev/weathercraft-erp",
        "command": "PLAYWRIGHT_TEST_BASE_URL=https://weathercraft-erp.vercel.app PLAYWRIGHT_ALLOW_PROD_E2E=1 PLAYWRIGHT_BYPASS_AUTH=false npx playwright test tests/e2e/critical-smoke.spec.ts --reporter=json",
        "url": "https://weathercraft-erp.vercel.app"
    },
    {
        "id": "myroofgenius-app",
        "name": "MyRoofGenius (PROD)",
        "path": "/home/matt-woodworth/dev/myroofgenius-app",
        "command": "PLAYWRIGHT_BASE_URL=https://myroofgenius.com npx playwright test e2e/production-readiness.spec.ts --reporter=json",
        "url": "https://myroofgenius.com"
    },
    {
        "id": "brainops-command-center",
        "name": "Command Center (PROD)",
        "path": "/home/matt-woodworth/dev/brainops-command-center",
        "command": "PLAYWRIGHT_TEST_BASE_URL=https://brainops-command-center.vercel.app RUN_LIVE_NAV=true npx playwright test e2e/live-nav.spec.ts --reporter=json",
        "url": "https://brainops-command-center.vercel.app"
    },
    {
        "id": "brainstack-studio",
        "name": "BrainStack Studio (Worldwide Tech)",
        "path": "/home/matt-woodworth/dev/brainstack-studio",
        "command": "curl -I -f https://brainstackstudio.com || exit 1",
        "url": "https://brainstackstudio.com"
    }
]

def log_to_brain(system, status, details):
    """Logs the QA result to the unified_brain_logs table."""
    try:
        payload = {
            "system": system,
            "action": "qa_audit",
            "data": {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "details": details
            }
        }
        supabase.table("unified_brain_logs").insert(payload).execute()
        print(f"🧠 Logged to Brain: {system} -> {status}")
    except Exception as e:
        print(f"⚠️ Failed to log to Brain: {e}")

def run_test(system):
    print(f"\n🚀 Starting QA Audit for: {system['name']}...")
    print(f"📂 Path: {system['path']}")
    
    try:
        # Run Playwright command
        result = subprocess.run(
            system['command'],
            shell=True,
            cwd=system['path'],
            capture_output=True,
            text=True
        )
        
        output = result.stdout
        error_out = result.stderr
        
        # Parse JSON reporter output if possible
        status = "failed"
        details = {"raw_output": output[-500:], "error": error_out[-500:]}
        
        if result.returncode == 0:
            status = "passed"
            print(f"✅ {system['name']}: PASSED")
        else:
            print(f"❌ {system['name']}: FAILED")
            print(f"Errors:\n{error_out[:200]}...")

        # Try to parse JSON output for better details
        try:
            # Playwright JSON reporter prints JSON to stdout. 
            # We need to find the JSON blob if there's other noise.
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = output[json_start:json_end]
                test_results = json.loads(json_str)
                details["stats"] = test_results.get("stats", {})
        except json.JSONDecodeError:
            pass

        log_to_brain(system['id'], status, details)
        return status == "passed"

    except Exception as e:
        print(f"🔥 Critical Execution Error: {e}")
        log_to_brain(system['id'], "execution_error", {"error": str(e)})
        return False

def main():
    print("🤖 Master QA Orchestrator Initialized")
    print("=======================================")
    
    results = {}
    all_passed = True
    
    for system in SYSTEMS:
        passed = run_test(system)
        results[system['id']] = "passed" if passed else "failed"
        if not passed:
            all_passed = False
            
    print("\n=======================================")
    print("📊 Final Report:")
    print(json.dumps(results, indent=2))
    
    if all_passed:
        print("\n✅ GLOBAL STATUS: OPERATIONAL")
        sys.exit(0)
    else:
        print("\n❌ GLOBAL STATUS: DEGRADED")
        sys.exit(1)

if __name__ == "__main__":
    main()

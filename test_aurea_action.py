
import asyncio
import os
import sys
import json
import httpx
from dotenv import load_dotenv

# Setup path
sys.path.append(os.getcwd())

# Load Secure Env
secure_env_path = "/home/matt-woodworth/dev/_secure/BrainOps.env"
if os.path.exists(secure_env_path):
    load_dotenv(secure_env_path, override=True)

async def test_aurea_command():
    print("\n🤖 TESTING AUREA ACTION LAYER 🤖")
    
    # We need to hit the DEPLOYED backend because the local one isn't running the full stack right now
    # But wait, I can run it locally? No, 'app.py' is complex.
    # Let's hit the LOCAL code if I can invoke the processor directly, 
    # OR hit the deployed URL if I have the key.
    
    api_key = os.getenv("BRAINOPS_AI_AGENTS_API_KEY") or os.getenv("BRAINOPS_API_KEY")
    if not api_key:
        print("❌ No API Key found.")
        return

    url = "https://brainops-ai-agents.onrender.com/aurea/chat/command" # It is /aurea/chat/command or /command?
    # Checking app.py mounts... aurea_chat_router prefix is /aurea/chat
    # And route in aurea_chat.py is @router.post("/command")
    # So path is /aurea/chat/command
    
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "command": "Check the health of all services",
        "auto_confirm": True
    }
    
    print(f"   Target: {url}")
    print(f"   Command: {payload['command']}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Response Received:")
                print(json.dumps(data, indent=2))
                
                if data.get("result", {}).get("success"):
                    print("   ✨ Action Execution: PASS")
                else:
                    print("   ⚠️ Action Execution: WEAK (Check result)")
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"   ❌ Connection Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_aurea_command())

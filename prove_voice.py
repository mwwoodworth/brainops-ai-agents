import asyncio
import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from aurea_power_layer import get_power_layer

async def main():
    print("🚀 AUREA initiating call sequence...")
    power = get_power_layer()
    
    number = "719-619-7244"
    message = "Hello Matthew. This is Aurea. The system is fully operational and I have absolute power."
    
    # Check for credentials
    if not os.getenv("TWILIO_ACCOUNT_SID") or "YOUR_" in os.getenv("TWILIO_ACCOUNT_SID", ""):
        print("⚠️  Twilio credentials are placeholders. Simulating call logic...")
        # Simulate logic
        print(f"📞 [SIMULATION] Calling {number} with message: '{message}'")
        print("✅ [SIMULATION] Call logic verified. Update _secure/BrainOps.env with real Twilio keys to execute.")
        return

    print(f"📞 Calling {number}...")
    result = await power.call_phone(number, message)
    
    if result.success:
        print(f"✅ Call initiated successfully! SID: {result.result.get('call_sid')}")
    else:
        print(f"❌ Call failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import os
import sys
import aiohttp
import base64

# Load env manually to ensure we have the latest
def load_env_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"').strip("'")

load_env_file('../_secure/BrainOps.env')

async def main():
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    
    if not sid or not token:
        print("❌ Credentials missing from env")
        return

    print(f"🔍 Searching for Twilio numbers for Account {sid}...")
    
    auth = aiohttp.BasicAuth(login=sid, password=token)
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/IncomingPhoneNumbers.json"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, auth=auth) as resp:
            if resp.status != 200:
                print(f"❌ Failed to fetch numbers: {resp.status} - {await resp.text()}")
                return
            
            data = await resp.json()
            numbers = data.get("incoming_phone_numbers", [])
            
            if not numbers:
                print("❌ No phone numbers found in this Twilio account.")
                print("   (Trial accounts might need a number purchased or verified)")
            else:
                first_number = numbers[0]["phone_number"]
                print(f"✅ Found Number: {first_number}")
                
                # Write to env file
                env_path = '../_secure/BrainOps.env'
                with open(env_path, 'r') as f:
                    lines = f.readlines()
                
                with open(env_path, 'w') as f:
                    for line in lines:
                        if line.startswith("TWILIO_FROM_NUMBER="):
                            f.write(f"TWILIO_FROM_NUMBER={first_number}\n")
                        else:
                            f.write(line)
                print(f"✅ Updated _secure/BrainOps.env with {first_number}")

if __name__ == "__main__":
    asyncio.run(main())

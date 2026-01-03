#!/usr/bin/env python3
"""
Diagnose AI Issues
"""

import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

print("="*60)
print("AI DIAGNOSTICS")
print("="*60)

# Check environment
openai_key = os.getenv("OPENAI_API_KEY", "")
anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
print("\n1. LOCAL ENVIRONMENT:")
print(f"   OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not set'}")
print(f"   ANTHROPIC_API_KEY: {'✅ Set' if anthropic_key else '❌ Not set'}")

if not openai_key and not anthropic_key:
    print("\n❌ No API keys found in local environment")
    print("   This is expected - keys should be in Render environment")
    print("\n2. RENDER ENVIRONMENT:")
    print("   Check https://dashboard.render.com")
    print("   Go to your service → Environment")
    print("   Ensure these are set:")
    print("   - OPENAI_API_KEY")
    print("   - ANTHROPIC_API_KEY")
    print("\n3. COMMON ISSUES:")
    print("   - Keys might be truncated (should be 50+ chars)")
    print("   - Keys might have extra quotes")
    print("   - Keys might be from free trial (expired)")
    print("   - OpenAI account might need credits")
else:
    # Test OpenAI if available
    if openai_key:
        print("\n2. TESTING OPENAI LOCALLY:")
        if OpenAI is None:
            print("   ❌ OpenAI SDK not installed")
        else:
            try:
                client = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Say 'AI works'"}],
                    max_tokens=10
                )
                print(f"   ✅ OpenAI works: {response.choices[0].message.content}")
            except Exception as e:
                print(f"   ❌ OpenAI error: {e}")

print("\n" + "="*60)
print("NEXT STEPS:")
print("1. Check Render dashboard for environment variables")
print("2. Ensure API keys are complete and valid")
print("3. Check OpenAI account has credits")
print("4. Redeploy after fixing environment variables")
print("="*60)

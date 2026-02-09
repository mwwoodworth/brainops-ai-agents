#!/usr/bin/env python3
"""
One-time script to get a Google Keep master token.

Run this LOCALLY (not on the server) - it requires interactive browser login.

Usage:
    python3 scripts/get_keep_master_token.py

It will:
1. Ask for your Google email
2. Ask for your Google app password (NOT your regular password)
3. Authenticate with Google and print the master token
4. You then set GOOGLE_KEEP_MASTER_TOKEN in Render env vars

IMPORTANT: You need an App Password, not your regular Google password.
Go to https://myaccount.google.com/apppasswords to create one.
"""

import getpass
import sys

try:
    import gkeepapi
except ImportError:
    print("ERROR: gkeepapi not installed. Run: pip install gkeepapi")
    sys.exit(1)


def main():
    print("=" * 60)
    print("Google Keep Master Token Generator")
    print("=" * 60)
    print()
    print("This generates a master token for gkeepapi authentication.")
    print("You need a Google App Password (NOT your regular password).")
    print("Create one at: https://myaccount.google.com/apppasswords")
    print()

    email = input("Google email (e.g., matthew@brainstackstudio.com): ").strip()
    if not email:
        print("ERROR: Email required")
        sys.exit(1)

    password = getpass.getpass("App Password (16 chars, no spaces): ").strip()
    if not password:
        print("ERROR: Password required")
        sys.exit(1)

    print(f"\nAuthenticating as {email}...")
    keep = gkeepapi.Keep()

    try:
        keep.login(email, password)
        master_token = keep.getMasterToken()

        print("\n" + "=" * 60)
        print("SUCCESS! Master token obtained.")
        print("=" * 60)
        print(f"\nGOOGLE_KEEP_EMAIL={email}")
        print(f"GOOGLE_KEEP_MASTER_TOKEN={master_token}")
        print()
        print("Add these to your Render environment variables.")
        print("The master token does NOT expire (unless you revoke app access).")

    except gkeepapi.exception.LoginException as e:
        print(f"\nLOGIN FAILED: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're using an App Password, not your regular password")
        print("2. Create one at: https://myaccount.google.com/apppasswords")
        print("3. If 2FA is off, enable it first (App Passwords require 2FA)")
        sys.exit(1)


if __name__ == "__main__":
    main()

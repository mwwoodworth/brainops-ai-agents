#!/usr/bin/env python3
"""
Main entry point for BrainOps AI Agent Service
This file ensures the correct service runs in production
"""

import os
import sys

# Force app.py to be used
print("="*50)
print("BRAINOPS AI SERVICE LAUNCHER")
print("Forcing app.py execution")
print("="*50)

# Import and run the correct app
from app import app
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting on port {port} with app.py v2.0.0")
    uvicorn.run(app, host="0.0.0.0", port=port)
#!/usr/bin/env python3
"""
Production startup script for BrainOps AI Agents
Ensures the new integrated system runs
"""

import os
import uvicorn

# Set environment variables
os.environ['PYTHONUNBUFFERED'] = '1'

# Import and run the main integration app
from main_integration import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = "0.0.0.0"

    print(f"🚀 Starting BrainOps AI OS - Integrated System")
    print(f"Running on http://{host}:{port}")
    print(f"AUREA, Memory, Board, and all 59 agents activated")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

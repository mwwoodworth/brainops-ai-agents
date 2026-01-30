import os
import sys

if __name__ != "__main__":
    # Manual script for local experimentation (requires API keys/DB). Avoid
    # breaking `pytest` collection in CI/dev environments.
    import pytest

    pytest.skip("manual local agent runner (not collected as a unit test)", allow_module_level=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio

from agent_executor import AgentExecutor
from ai_core import test_ai_providers


async def test_local_execution():
    print("\n=== TESTING LOCAL AI AGENT EXECUTION ===\n")

    # Test AI providers
    print("1. Testing AI Providers...")
    test_ai_providers()

    # Test agent executor
    print("\n2. Testing Agent Executor...")
    executor = AgentExecutor()

    # Initialize agents
    await executor.initialize_agents()
    print(f"  ✅ Initialized {len(executor.agents)} agents")

    # Test simple execution
    test_task = {
        "type": "general",
        "description": "Calculate 2+2",
        "priority": 1
    }

    result = await executor.execute_task(test_task)
    print(f"  Task result: {result}")

if __name__ == "__main__":
    asyncio.run(test_local_execution())

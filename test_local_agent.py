import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_core import test_ai_providers
from agent_executor import AgentExecutor
import asyncio

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

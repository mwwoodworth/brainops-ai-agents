import asyncio
import os
import sys

# Add parent directory to path so we can import modules
sys.path.append('/home/matt-woodworth/dev/brainops-ai-agents')

from market_analyzer import MarketAnalyzerAgent

async def test_market_analyzer():
    print("Testing MarketAnalyzerAgent locally...")
    
    agent = MarketAnalyzerAgent()
    
    task = {
        "action": "get_current_pricing",
        "location": "Denver, CO",
        "materials": ["Architectural Shingles"]
    }
    
    print(f"Executing task: {task}")
    try:
        result = await agent.execute(task)
        print("\n--- Result ---")
        print(result)
        
        if result.get('status') == 'success' or (result.get('status') == 'partial_success' and result.get('local_rates')):
            print("\n✅ Verification Successful: Agent returned structured data.")
        else:
            print("\n⚠️ Verification Warning: Agent returned error or empty data.")
            
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_market_analyzer())
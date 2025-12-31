import asyncio
import os
import json
import logging
from datetime import datetime

# SECURITY: Load credentials from environment or .env file
# DO NOT hardcode credentials - they must be set in environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use existing environment

# Verify required environment variables are set
required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    raise EnvironmentError(f"Required environment variables not set: {', '.join(missing)}. "
                          "Set them in .env file or environment.")

# Set Auth Bypass for Testing (only for local test runs)
os.environ.setdefault("ALLOW_TEST_KEY", "true")
os.environ.setdefault("API_KEYS", '["test-key"]')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of 59 Agents
AGENTS = [
    "InventoryAgent", "APIManagementAgent", "DispatchAgent", "TranslationProcessor",
    "SchedulingAgent", "InvoicingAgent", "DashboardMonitor", "BudgetingAgent",
    "PayrollAgent", "MetricsCalculator", "Elena", "NotificationAgent",
    "PermitWorkflow", "OnboardingAgent", "LeadScorer", "ChatInterface",
    "WorkflowEngine", "QualityAgent", "CustomerIntelligence", "RoutingAgent",
    "LogisticsOptimizer", "SocialMediaAgent", "ComplianceAgent", "ExpenseMonitor",
    "VoiceInterface", "SafetyAgent", "ProcurementAgent", "SEOOptimizer",
    "TaxCalculator", "LeadGenerationAgent", "WarrantyAgent", "IntegrationAgent",
    "Monitor", "Scheduler", "InsightsAnalyzer", "TrainingAgent", "Invoicer",
    "ReportingAgent", "ContractGenerator", "SMSInterface", "CustomerAgent",
    "CampaignAgent", "RevenueOptimizer", "SystemMonitor", "EstimationAgent",
    "WorkflowAutomation", "WarehouseMonitor", "InsuranceAgent", "IntelligentScheduler",
    "DeliveryAgent", "ProposalGenerator", "RecruitingAgent", "PredictiveAnalyzer",
    "SecurityMonitor", "PerformanceMonitor", "BenefitsAgent", "BackupAgent",
    "VendorAgent", "EmailMarketingAgent"
]

async def verify_memory_system():
    print("\n=== Verifying Memory System ===")
    try:
        from memory_system import AIMemorySystem
        memory = AIMemorySystem()
        
        # Test 1: Store Context
        key = f"verification_test_{int(datetime.now().timestamp())}"
        value = {"status": "testing", "timestamp": str(datetime.now())}
        try:
            memory.store_context("test", key, value)
            print(f"✅ Memory Store: Success (Key: {key})")
        except Exception as e:
            print(f"❌ Memory Store: Failed ({str(e)})")
        
        # Test 2: Get Context (Try to get system info if test key failed)
        try:
            # Try getting 'system' context which usually exists
            retrieved = memory.get_context("system")
            if retrieved:
                 print(f"✅ Memory Retrieve (System): Success")
            else:
                 # Try the key we just stored (if it worked)
                 retrieved = memory.get_context(key)
                 if retrieved:
                     print(f"✅ Memory Retrieve (Test Key): Success")
                 else:
                     print(f"❌ Memory Retrieve: Failed")
        except Exception as e:
            print(f"❌ Memory Retrieve Exception: {e}")

        # Test 3: System Overview
        try:
            overview = memory.get_system_overview()
            if overview and 'statistics' in overview:
                 print(f"✅ System Overview: Success (Stats found)")
                 print(f"   Stats: {json.dumps(overview['statistics'], default=str)}")
            else:
                 print(f"❌ System Overview: Failed")
        except Exception as e:
            print(f"❌ System Overview Exception: {e}")

    except Exception as e:
        print(f"❌ Memory System Verification Infrastructure Failed: {e}")

async def verify_agents():
    print("\n=== Verifying AI Agents ===")
    try:
        # Mock config if needed, but we rely on imports
        from agent_executor import AgentExecutor
        
        executor = AgentExecutor()
        
        results = {"success": [], "failed": []}
        
        for agent_name in AGENTS:
            print(f"Testing Agent: {agent_name}...", end=" ", flush=True)
            try:
                # Define a harmless task
                task = {
                    "action": "health_check", 
                    "type": "verification",
                    "description": "Verification ping"
                }
                
                # Specific tasks for known implemented agents
                if agent_name == "Monitor":
                    task = {"action": "backend_check"}
                elif agent_name == "CustomerAgent":
                    task = {"action": "analyze"} # Read-only
                elif agent_name == "InvoicingAgent":
                    task = {"action": "report"} # Read-only
                elif agent_name == "SystemMonitor":
                    task = {"action": "health_check"} # Might trigger full check
                
                # We use a short timeout wrapper if possible, but AgentExecutor calls are async
                # Just call it
                result = await executor.execute(agent_name, task)
                
                if result and result.get("status") != "failed":
                    print("✅ OK")
                    results["success"].append(agent_name)
                else:
                    print(f"❌ Failed (Status: {result.get('status')})")
                    results["failed"].append(agent_name)
                    
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                results["failed"].append(agent_name)
        
        print("\n=== Agent Verification Summary ===")
        print(f"Total Agents: {len(AGENTS)}")
        print(f"Success: {len(results['success'])}")
        print(f"Failed: {len(results['failed'])}")
        
        if results['failed']:
            print("Failed Agents:")
            for a in results['failed']:
                print(f"- {a}")

    except Exception as e:
        print(f"❌ Agent Verification Infrastructure Failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await verify_memory_system()
    await verify_agents()

if __name__ == "__main__":
    asyncio.run(main())

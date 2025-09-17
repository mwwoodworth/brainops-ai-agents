#!/usr/bin/env python3
"""
Test script to verify all modules can be imported
"""

import sys

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        print(f"✗ {description}: {e}")
        return False
    except Exception as e:
        print(f"✗ {description}: Unexpected error - {e}")
        return False

def main():
    """Test all modules"""
    print("Testing module imports...\n")

    modules = [
        ("memory_system", "Core memory system"),
        ("orchestrator", "Core orchestrator"),
        ("langgraph_orchestrator", "LangGraph orchestrator (optional)"),
        ("vector_memory_system", "Vector memory system"),
        ("revenue_generation_system", "Revenue generation"),
        ("customer_acquisition_agents", "Customer acquisition"),
        ("ai_pricing_engine", "AI pricing engine"),
        ("notebook_lm_plus", "Notebook LM+"),
        ("conversation_memory", "Conversation memory"),
        ("system_state_manager", "System state manager"),
        ("ai_decision_tree", "AI decision tree"),
        ("realtime_monitor", "Realtime monitor"),
        ("app", "Main FastAPI app")
    ]

    success_count = 0
    for module, description in modules:
        if test_import(module, description):
            success_count += 1

    print(f"\n{success_count}/{len(modules)} modules imported successfully")

    # Test database connection
    print("\nTesting database connection...")
    try:
        import psycopg2
        import os
        from dotenv import load_dotenv

        load_dotenv()

        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
            database=os.getenv('DB_NAME', 'postgres'),
            user=os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
            password=os.getenv('DB_PASSWORD', 'Brain0ps2O2S'),
            port=os.getenv('DB_PORT', 5432)
        )

        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()

        if result and result[0] == 1:
            print("✓ Database connection successful")
        else:
            print("✗ Database connection failed - unexpected result")

        conn.close()

    except Exception as e:
        print(f"✗ Database connection failed: {e}")

    return success_count == len(modules)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
#!/usr/bin/env python3
"""
Test script to verify all modules can be imported
"""

import sys

if __name__ != "__main__":
    # Manual script; not a unit test. Avoid pytest collection errors.
    import pytest

    pytest.skip("manual module import script (not collected as a unit test)", allow_module_level=True)


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
        import os

        import psycopg2
        from dotenv import load_dotenv

        load_dotenv()

        # Validate required environment variables - NO hardcoded fallbacks
        db_host = os.getenv('DB_HOST')
        db_name = os.getenv('DB_NAME')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        db_port = os.getenv('DB_PORT', '5432')

        missing = []
        if not db_host:
            missing.append('DB_HOST')
        if not db_name:
            missing.append('DB_NAME')
        if not db_user:
            missing.append('DB_USER')
        if not db_password:
            missing.append('DB_PASSWORD')

        if missing:
            raise RuntimeError(
                f"Required environment variables not set: {', '.join(missing)}"
            )

        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=int(db_port)
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

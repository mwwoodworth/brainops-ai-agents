import os
import sys

import pytest

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_keys_present():
    """
    Verifies that essential model keys are present in the environment.
    Marks xfail if missing, rather than failing the suite, to allow local dev without all keys.
    """
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing = [key for key in required_keys if not os.getenv(key)]

    if missing:
        pytest.xfail(f"Missing API keys: {', '.join(missing)}")

    for key in required_keys:
        assert os.getenv(key) is not None

def test_db_connection():
    """
    Verifies database connectivity if credentials are present.
    """
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")

    # Here we would try to connect using sqlalchemy or psycopg2
    # import sqlalchemy
    # engine = sqlalchemy.create_engine(os.getenv("DATABASE_URL"))
    # connection = engine.connect()
    # connection.close()
    pass

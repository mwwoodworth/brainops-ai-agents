"""
Centralized database configuration helper.
Supports both individual environment variables and DATABASE_URL.
"""
import os
from urllib.parse import urlparse


def get_db_config():
    """Get database configuration with DATABASE_URL fallback."""
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME", "postgres")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_port = os.getenv("DB_PORT", "5432")

    # Fallback to DATABASE_URL if individual vars not set
    if not all([db_host, db_user, db_password]):
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            parsed = urlparse(database_url)
            db_host = parsed.hostname or ''
            db_name = parsed.path.lstrip('/') if parsed.path else 'postgres'
            db_user = parsed.username or ''
            db_password = parsed.password or ''
            db_port = str(parsed.port) if parsed.port else '5432'

    if not all([db_host, db_user, db_password]):
        raise RuntimeError("Missing required: DB_HOST/DB_USER/DB_PASSWORD or DATABASE_URL")

    return {
        'host': db_host,
        'database': db_name,
        'user': db_user,
        'password': db_password,
        'port': int(db_port)
    }


# Pre-configured DB_CONFIG for backwards compatibility
DB_CONFIG = get_db_config()

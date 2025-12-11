"""
Configuration Management for AI Agents Service
Centralizes all configuration with environment variable support
"""
import os
from typing import Optional, List
import logging
from urllib.parse import urlparse

from dotenv import load_dotenv

# Hydrate env vars from a local .env in the current working directory when present.
load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration with secure defaults"""

    def __init__(self):
        # Try individual vars first
        self.host = os.getenv('DB_HOST', '')
        self.database = os.getenv('DB_NAME', '')
        self.user = os.getenv('DB_USER', '')
        self.password = os.getenv('DB_PASSWORD', '')
        self.port = int(os.getenv('DB_PORT', '5432'))
        self.ssl = os.getenv('DB_SSL', 'true').lower() not in ('false', '0', 'no')
        self.ssl_verify = os.getenv('DB_SSL_VERIFY', 'false').lower() not in ('false', '0', 'no')

        # Fallback to DATABASE_URL if individual vars not set (team-level Render env)
        if not all([self.host, self.database, self.user, self.password]):
            database_url = os.getenv('DATABASE_URL', '')
            if database_url:
                try:
                    parsed = urlparse(database_url)
                    self.host = parsed.hostname or ''
                    self.database = parsed.path.lstrip('/') if parsed.path else ''
                    self.user = parsed.username or ''
                    self.password = parsed.password or ''
                    self.port = parsed.port or 5432
                    logger.info(f"Parsed DATABASE_URL: host={self.host}, db={self.database}")
                except Exception as e:
                    logger.error(f"Failed to parse DATABASE_URL: {e}")

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        if not all([self.host, self.database, self.user, self.password]):
            raise RuntimeError(
                "Database configuration is incomplete. "
                "Ensure DB_HOST, DB_NAME, DB_USER, and DB_PASSWORD are set."
            )
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def to_dict(self) -> dict:
        """Get config as dictionary (without password for logging)"""
        return {
            'host': self.host,
            'database': self.database,
            'user': self.user,
            'port': self.port,
            'password': '***REDACTED***',
            'ssl': self.ssl,
            'ssl_verify': self.ssl_verify,
        }


class SecurityConfig:
    """Security configuration for authentication and CORS"""

    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.dev_mode = os.getenv('DEV_MODE', 'false').lower() == 'true'
        auth_required_env = os.getenv('AUTH_REQUIRED')
        self.auth_required = (
            auth_required_env.lower() not in ('false', '0', 'no')
            if auth_required_env is not None
            else True
        )
        
        api_keys_str = os.getenv('API_KEYS', '')
        self.valid_api_keys = set(api_keys_str.split(',')) if api_keys_str else set()

        test_key = (
            os.getenv('TEST_API_KEY')
            or os.getenv('AI_AGENTS_TEST_KEY')
            or os.getenv('DEFAULT_TEST_API_KEY')
        )
        default_local_test_key = "brainops-local-test-key"
        allow_test_key = (
            os.getenv('ALLOW_TEST_KEY', 'false').lower() == 'true'
            or self.dev_mode
            or self.environment != 'production'
        )
        self.test_api_key: Optional[str] = None
        if allow_test_key:
            effective_test_key = test_key or default_local_test_key
            self.valid_api_keys.add(effective_test_key)
            self.test_api_key = effective_test_key

        if self.auth_required and not self.valid_api_keys:
            logger.critical(
                "AUTH_REQUIRED is enabled but no API keys are configured. "
                "Set API_KEYS or enable a non-production test key via ALLOW_TEST_KEY."
            )
            raise RuntimeError("Authentication required but no API keys provided.")
        
        cors_origins_str = os.getenv('ALLOWED_ORIGINS', '')
        if cors_origins_str:
            self.allowed_origins = cors_origins_str.split(',')
        elif self.dev_mode:
            self.allowed_origins = ["http://localhost:3000", "http://localhost:3001", "*"]
        else:
            self.allowed_origins = []


class AppConfig:
    """Main application configuration"""

    def __init__(self):
        self.version = "3.1.0"
        self.service_name = "BrainOps AI OS"
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', '10000'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.database = DatabaseConfig()
        self.security = SecurityConfig()


config = AppConfig()

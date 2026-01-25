"""
Configuration Management for AI Agents Service
Centralizes all configuration with environment variable support
"""
import logging
import os
from typing import Optional
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
        # SSL verification: Supabase pooler (PgBouncer) uses self-signed certs, so disable by default
        # This is safe because we're connecting over TLS, just not verifying the cert chain
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
        return f"postgresql://{self.user}:<DB_PASSWORD_REDACTED>@{self.host}:{self.port}/{self.database}"

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

        # Build valid API keys from multiple sources
        api_keys_str = os.getenv('API_KEYS', '')
        self.valid_api_keys = set(k.strip() for k in api_keys_str.split(',') if k.strip())

        # Also accept individual API key environment variables
        for key_name in ['BRAINOPS_API_KEY', 'AGENTS_API_KEY', 'MCP_API_KEY']:
            key_value = os.getenv(key_name)
            if key_value:
                self.valid_api_keys.add(key_value)

        test_key = (
            os.getenv('TEST_API_KEY')
            or os.getenv('AI_AGENTS_TEST_KEY')
            or os.getenv('DEFAULT_TEST_API_KEY')
        )
        default_local_test_key = "brainops-local-test-key"
        allow_test_key_flag = os.getenv('ALLOW_TEST_KEY', 'false').lower() == 'true'
        allow_test_key = allow_test_key_flag and self.environment != 'production'
        if allow_test_key_flag and self.environment == 'production':
            logger.critical(
                "ALLOW_TEST_KEY is set in production; test API keys are disabled."
            )
        self.test_api_key: Optional[str] = None
        if allow_test_key:
            effective_test_key = test_key or default_local_test_key
            self.valid_api_keys.add(effective_test_key)
            self.test_api_key = effective_test_key

        self.auth_configured = bool(self.valid_api_keys)
        if self.auth_required and not self.auth_configured:
            logger.critical(
                "AUTH_REQUIRED is enabled but no API keys are configured. "
                "Set API_KEYS or enable a non-production test key via ALLOW_TEST_KEY. "
                "Service will start in lockdown mode (all secured endpoints return 503)."
            )

        cors_origins_str = os.getenv('ALLOWED_ORIGINS', '')
        if cors_origins_str:
            self.allowed_origins = [o.strip() for o in cors_origins_str.split(',') if o.strip()]
        elif self.dev_mode:
            self.allowed_origins = ["http://localhost:3000", "http://localhost:3001"]
        else:
            # Secure production defaults - only known frontends
            self.allowed_origins = [
                "https://weathercraft-erp.vercel.app",
                "https://myroofgenius.com",
                "https://www.myroofgenius.com",
                "https://brainops-command-center.vercel.app",
                "https://brainops-mcp-bridge.onrender.com",
            ]
        
        self.supabase_jwt_secret = os.getenv('SUPABASE_JWT_SECRET', '')


class TenantConfig:
    """Tenant configuration for multi-tenancy support"""

    def __init__(self):
        # Default tenant ID from environment (required for production)
        self.default_tenant_id = os.getenv('DEFAULT_TENANT_ID') or os.getenv('TENANT_ID')
        if not self.default_tenant_id:
            logger.warning("No DEFAULT_TENANT_ID set - multi-tenancy may not work correctly")

        # Per-request tenant resolution is handled via X-Tenant-ID header
        self.header_name = os.getenv('TENANT_HEADER', 'X-Tenant-ID')


class AppConfig:
    """Main application configuration"""

    def __init__(self):
        self.version = os.getenv('VERSION', 'v3.7.31')  # Fix orchestrator numeric coercion + LiveMemoryBrain schema bootstraps + consciousness cross-loop DB safety
        self.service_name = "BrainOps AI OS"
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', '10000'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.tenant = TenantConfig()

        # Autonomic Systems Feature Flags
        self.enable_nerve_center = os.getenv('ENABLE_NERVE_CENTER', 'false').lower() == 'true'
        self.enable_autonomic_loop = os.getenv('ENABLE_AUTONOMIC_LOOP', 'false').lower() == 'true'
        self.autonomic_loop_interval = float(os.getenv('AUTONOMIC_LOOP_INTERVAL', '30'))


config = AppConfig()

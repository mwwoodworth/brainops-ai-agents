"""
Configuration Management for AI Agents Service
Centralizes all configuration with environment variable support
"""
import os
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration with secure defaults"""

    def __init__(self):
        self.host = os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com')
        self.database = os.getenv('DB_NAME', 'postgres')
        self.user = os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab')
        self.password = os.getenv('DB_PASSWORD', 'Brain0ps2O2S')
        self.port = int(os.getenv('DB_PORT', '5432'))

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def to_dict(self) -> dict:
        """Get config as dictionary (without password for logging)"""
        return {
            'host': self.host,
            'database': self.database,
            'user': self.user,
            'port': self.port,
            'password': '***REDACTED***'
        }


class SecurityConfig:
    """Security configuration for authentication and CORS"""

    def __init__(self):
        self.dev_mode = os.getenv('DEV_MODE', 'false').lower() == 'true'
        self.auth_required = os.getenv('AUTH_REQUIRED', 'false' if self.dev_mode else 'true').lower() == 'true'
        
        api_keys_str = os.getenv('API_KEYS', '')
        self.valid_api_keys = set(api_keys_str.split(',')) if api_keys_str else set()

        if self.auth_required and not self.valid_api_keys:
            logger.warning(
                "AUTH_REQUIRED enabled but no API keys configured. "
                "Falling back to unauthenticated mode."
            )
            self.auth_required = False
        
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

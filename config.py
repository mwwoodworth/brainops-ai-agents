"""
Centralized configuration management
All sensitive data should come from environment variables
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""

    # Database Configuration
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD'),  # No default!
        'port': int(os.getenv('DB_PORT', '5432'))
    }

    # Validate critical environment variables
    @classmethod
    def validate(cls):
        """Validate that all required environment variables are set"""
        required_vars = [
            'DB_PASSWORD',
            'DB_HOST',
            'DB_USER'
        ]

        missing = []
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

    # Service URLs
    BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')
    FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')

    # Email Service
    SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')

    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(32).hex())

    # Feature Flags
    ENABLE_AI_AGENTS = os.getenv('ENABLE_AI_AGENTS', 'true').lower() == 'true'
    ENABLE_MONITORING = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please set up your .env file with the required variables")
    print("Copy .env.example to .env and fill in your values")
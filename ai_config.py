#!/usr/bin/env python3
"""
Complete AI Configuration with all API keys from production .env
All keys verified from BrainOps (4).env downloaded from Render
"""

import os

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Anthropic Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Perplexity Configuration
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Hugging Face Configuration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# ElevenLabs Configuration (for voice)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Database Configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
    "port": os.getenv("DB_PORT", "5432")
}

# Supabase Configuration
SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "https://yomagoqdmxszqtdwuhab.supabase.co"
)

SUPABASE_SERVICE_KEY = os.getenv(
    "SUPABASE_SERVICE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlvbWFnb3FkbXhzenF0ZHd1aGFiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0OTgzMzI3NiwiZXhwIjoyMDY1NDA5Mjc2fQ.7C3guJ_0moYGkdyeFmJ9cd2BmduB5NnU00erIIxH3gQ"
)

# AI Model Preferences
AI_MODELS = {
    "fast": "gpt-3.5-turbo",  # Quick responses
    "smart": "gpt-4-0125-preview",  # Complex reasoning
    "creative": "claude-3-opus-20240229",  # Creative tasks
    "research": "perplexity-70b-online",  # Real-time web data
    "analysis": "gemini-2.5-flash",  # Deep analysis (updated Oct 2025 - Gemini 1.5 retired)
    "fallback": "mixtral-8x7b"  # Hugging Face fallback
}

# Feature Flags
AI_FEATURES = {
    "openai_enabled": bool(OPENAI_API_KEY),
    "anthropic_enabled": bool(ANTHROPIC_API_KEY),
    "gemini_enabled": bool(GOOGLE_API_KEY),
    "perplexity_enabled": bool(PERPLEXITY_API_KEY),
    "huggingface_enabled": bool(HUGGINGFACE_API_TOKEN),
    "voice_enabled": bool(ELEVENLABS_API_KEY),
    "realtime_search": bool(PERPLEXITY_API_KEY),
    "multi_model_consensus": True,
    "smart_fallback": True
}

def get_api_status():
    """Get status of all API keys"""
    return {
        "openai": "✅ Configured" if OPENAI_API_KEY else "❌ Missing",
        "anthropic": "✅ Configured" if ANTHROPIC_API_KEY else "❌ Missing",
        "gemini": "✅ Configured" if GOOGLE_API_KEY else "❌ Missing",
        "perplexity": "✅ Configured" if PERPLEXITY_API_KEY else "❌ Missing",
        "huggingface": "✅ Configured" if HUGGINGFACE_API_TOKEN else "❌ Missing",
        "elevenlabs": "✅ Configured" if ELEVENLABS_API_KEY else "❌ Missing",
        "database": "✅ Configured" if DATABASE_CONFIG["password"] else "❌ Missing",
        "supabase": "✅ Configured" if SUPABASE_SERVICE_KEY else "❌ Missing"
    }

def validate_all_keys():
    """Validate all API keys are present"""
    status = get_api_status()
    all_configured = all("✅" in v for v in status.values())

    if all_configured:
        print("🚀 All AI systems configured and ready!")
    else:
        print("⚠️ Some API keys missing:")
        for service, status in status.items():
            if "❌" in status:
                print(f"  - {service}: {status}")

    return all_configured

# Export all configuration
__all__ = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "PERPLEXITY_API_KEY",
    "HUGGINGFACE_API_TOKEN",
    "ELEVENLABS_API_KEY",
    "DATABASE_CONFIG",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY",
    "AI_MODELS",
    "AI_FEATURES",
    "get_api_status",
    "validate_all_keys"
]

#!/usr/bin/env python3
"""
BrainOps AI Agents - environment-backed configuration.

This module must not embed credentials. It only reads environment variables and
exposes structured config + simple diagnostics for local ops.
"""

from __future__ import annotations

import os

# AI Provider API Keys (server-side only)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Supabase / Postgres
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_SERVICE_KEY", "")

DATABASE_CONFIG = {
    "host": os.getenv("SUPABASE_DB_HOST", "") or os.getenv("DB_HOST", ""),
    "database": os.getenv("SUPABASE_DB_NAME", "") or os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("SUPABASE_DB_USER", "") or os.getenv("DB_USER", ""),
    "password": os.getenv("SUPABASE_DB_PASSWORD", "") or os.getenv("DB_PASSWORD", ""),
    "port": os.getenv("SUPABASE_DB_PORT", "") or os.getenv("DB_PORT", "5432"),
}

# AI Model Preferences (non-secret)
AI_MODELS = {
    "fast": os.getenv("BRAINOPS_MODEL_FAST", "gpt-4o-mini"),
    "smart": os.getenv("BRAINOPS_MODEL_SMART", "gpt-5"),
    "creative": os.getenv("BRAINOPS_MODEL_CREATIVE", "claude-sonnet-4-5-20250929"),
    "research": os.getenv("BRAINOPS_MODEL_RESEARCH", "perplexity-70b-online"),
    "analysis": os.getenv("BRAINOPS_MODEL_ANALYSIS", "gemini-2.5-flash"),
    "fallback": os.getenv("BRAINOPS_MODEL_FALLBACK", "mixtral-8x7b"),
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
    "multi_model_consensus": os.getenv("BRAINOPS_MULTI_MODEL_CONSENSUS", "true").lower() == "true",
    "smart_fallback": os.getenv("BRAINOPS_SMART_FALLBACK", "true").lower() == "true",
}


def get_api_status() -> dict[str, str]:
    return {
        "openai": "configured" if OPENAI_API_KEY else "missing",
        "anthropic": "configured" if ANTHROPIC_API_KEY else "missing",
        "gemini": "configured" if GOOGLE_API_KEY else "missing",
        "perplexity": "configured" if PERPLEXITY_API_KEY else "missing",
        "huggingface": "configured" if HUGGINGFACE_API_TOKEN else "missing",
        "elevenlabs": "configured" if ELEVENLABS_API_KEY else "missing",
        "database_password": "configured" if DATABASE_CONFIG.get("password") else "missing",
        "supabase_service_key": "configured" if SUPABASE_SERVICE_KEY else "missing",
    }


def validate_all_keys(*, require_db: bool = False, require_supabase: bool = False) -> bool:
    status = get_api_status()
    required = {"openai", "anthropic"}
    if require_db:
        required.add("database_password")
    if require_supabase:
        required.add("supabase_service_key")
    return all(status.get(k) == "configured" for k in required)


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
    "validate_all_keys",
]

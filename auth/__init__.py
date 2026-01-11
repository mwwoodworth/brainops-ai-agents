"""
Authentication Module for BrainOps AI Agents
============================================
Centralizes API key verification to avoid duplication across routers.

Usage:
    from auth import verify_api_key, API_KEY_HEADER

    router = APIRouter(
        prefix="/my-endpoint",
        dependencies=[Depends(verify_api_key)]
    )
"""

from auth.api_key import API_KEY_HEADER, verify_api_key, get_optional_api_key

__all__ = ["verify_api_key", "API_KEY_HEADER", "get_optional_api_key"]

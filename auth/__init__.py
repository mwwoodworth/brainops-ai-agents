"""
Authentication Module for BrainOps AI Agents
============================================
Authentication primitives live in this package.

Notes:
- API key verification is currently implemented in `app.py` (FastAPI dependency),
  because it needs access to the active runtime config and request context.
- JWT verification is implemented in `auth/jwt.py`.
"""

from .jwt import verify_jwt

__all__ = ["verify_jwt"]

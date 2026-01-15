"""
JWT Authentication Handler for Supabase
Verifies Bearer tokens and extracts user/tenant context.
"""
import logging
import jwt
import os
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import config

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

async def verify_jwt(
    request: Request,
    token: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[Dict[str, Any]]:
    """
    Verify Supabase JWT token.
    Returns decoded token dict if valid, None if not provided (optional auth),
    or raises HTTPException if invalid.
    """
    if not token:
        return None

    if not config.security.supabase_jwt_secret:
        logger.warning("SUPABASE_JWT_SECRET not configured, cannot verify JWT")
        return None

    try:
        payload = jwt.decode(
            token.credentials,
            config.security.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated"
        )
        
        # Extract user info
        user_id = payload.get("sub")
        app_metadata = payload.get("app_metadata", {})
        user_metadata = payload.get("user_metadata", {})
        tenant_id = app_metadata.get("tenant_id")
        
        # Store in request state for downstream use
        request.state.user = {
            "id": user_id,
            "email": payload.get("email"),
            "tenant_id": tenant_id,
            "role": payload.get("role")
        }
        request.state.tenant_id = tenant_id
        
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"JWT verification error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

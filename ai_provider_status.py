#!/usr/bin/env python3
"""
AI Provider Status Introspection

Runtime checks for OpenAI, Anthropic, Gemini, Perplexity, and Hugging Face.
This module does NOT modify any configuration or credentials; it only reports
configuration and basic liveness so AUREA and the Command Center can see what
is actually working.

Design goals:
- No secret printing: never return keys, only booleans + sanitized errors.
- "No assumptions": surface the *real* reason a provider is down (quota, auth, etc.).
- Low overhead: small probe requests + short caching TTL.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_CACHE: tuple[dict[str, Any], float] | None = None
_CACHE_TTL_S = float(os.getenv("AI_PROVIDER_STATUS_CACHE_TTL_S", "30"))


def _truncate_error(message: str, limit: int = 220) -> str:
    if not message:
        return ""
    message = " ".join(message.split())
    return message[:limit]


def _status_payload(*, configured: bool, reachable: bool, last_error: Optional[str], meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "configured": configured,
        "reachable": reachable,
        "last_error": last_error,
    }
    if meta:
        payload.update(meta)
    return payload


def _check_openai() -> dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY", "")
    configured = bool(key)
    model = os.getenv("BRAINOPS_MODEL_FAST", "gpt-4o-mini")

    try:
        from openai import OpenAI
        from openai import AuthenticationError, RateLimitError  # type: ignore
    except Exception as exc:
        return _status_payload(
            configured=configured,
            reachable=False,
            last_error=f"OpenAI SDK unavailable: {_truncate_error(str(exc))}" if configured else None,
            meta={"model": model},
        )

    if not configured:
        return _status_payload(configured=False, reachable=False, last_error=None, meta={"model": model})

    client = OpenAI(api_key=key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=8,
            temperature=0,
            timeout=12,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _status_payload(configured=True, reachable=bool(text), last_error=None if text else "Empty response", meta={"model": model})
    except RateLimitError as exc:  # quota/rate limiting
        msg = str(exc).lower()
        if "quota" in msg or "billing" in msg:
            err = "Quota exceeded (billing/credits)"
        else:
            err = "Rate limited"
        return _status_payload(configured=True, reachable=False, last_error=err, meta={"model": model})
    except AuthenticationError:
        return _status_payload(configured=True, reachable=False, last_error="Invalid API key", meta={"model": model})
    except Exception as exc:
        return _status_payload(configured=True, reachable=False, last_error=_truncate_error(str(exc)), meta={"model": model})


def _check_anthropic() -> dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    configured = bool(key)
    model = os.getenv("BRAINOPS_ANTHROPIC_FAST_MODEL", "claude-3-haiku-20240307")

    try:
        from anthropic import Anthropic
        from anthropic import AuthenticationError  # type: ignore
    except Exception as exc:
        return _status_payload(
            configured=configured,
            reachable=False,
            last_error=f"Anthropic SDK unavailable: {_truncate_error(str(exc))}" if configured else None,
            meta={"model": model},
        )

    if not configured:
        return _status_payload(configured=False, reachable=False, last_error=None, meta={"model": model})

    client = Anthropic(api_key=key)
    try:
        resp = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=8,
            temperature=0,
            timeout=12,
        )
        text = (resp.content[0].text if resp.content else "").strip()
        return _status_payload(configured=True, reachable=bool(text), last_error=None if text else "Empty response", meta={"model": model})
    except AuthenticationError:
        return _status_payload(configured=True, reachable=False, last_error="Invalid API key", meta={"model": model})
    except Exception as exc:
        msg = str(exc).lower()
        if "credit balance is too low" in msg or ("plans" in msg and "billing" in msg and "too low" in msg):
            return _status_payload(configured=True, reachable=False, last_error="Credit balance too low", meta={"model": model})
        if "rate limit" in msg or "429" in msg:
            return _status_payload(configured=True, reachable=False, last_error="Rate limited", meta={"model": model})
        return _status_payload(configured=True, reachable=False, last_error=_truncate_error(str(exc)), meta={"model": model})


def _check_gemini() -> dict[str, Any]:
    key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    configured = bool(key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if not configured:
        return _status_payload(configured=False, reachable=False, last_error=None, meta={"model": model})

    try:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore
    except Exception as exc:
        return _status_payload(
            configured=True,
            reachable=False,
            last_error=f"google-genai not available: {_truncate_error(str(exc))}",
            meta={"model": model},
        )

    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=model,
            contents="ping",
            config=genai_types.GenerateContentConfig(
                max_output_tokens=6,
                temperature=0,
                top_p=0.9,
            ),
        )
        text = (getattr(resp, "text", None) or "").strip()
        return _status_payload(configured=True, reachable=bool(text), last_error=None if text else "Empty response", meta={"model": model})
    except Exception as exc:
        return _status_payload(configured=True, reachable=False, last_error=_truncate_error(str(exc)), meta={"model": model})


def _check_perplexity() -> dict[str, Any]:
    key = os.getenv("PERPLEXITY_API_KEY", "")
    configured = bool(key)
    model = os.getenv("PERPLEXITY_MODEL", "sonar")
    if not configured:
        return _status_payload(configured=False, reachable=False, last_error=None, meta={"model": model})

    try:
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=12,
        )
        if r.status_code != 200:
            return _status_payload(
                configured=True,
                reachable=False,
                last_error=f"HTTP {r.status_code}",
                meta={"model": model},
            )
        data = r.json()
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        text = str(text).strip()
        return _status_payload(configured=True, reachable=bool(text), last_error=None if text else "Empty response", meta={"model": model})
    except Exception as exc:
        return _status_payload(configured=True, reachable=False, last_error=_truncate_error(str(exc)), meta={"model": model})


def _check_huggingface() -> dict[str, Any]:
    token = os.getenv("HUGGINGFACE_API_TOKEN", "")
    configured = bool(token)
    model = os.getenv("HUGGINGFACE_MODEL", "gpt2")

    headers: dict[str, str] = {}
    if configured:
        headers["Authorization"] = f"Bearer {token}"

    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json={"inputs": "ping", "parameters": {"max_new_tokens": 8, "return_full_text": False, "do_sample": False}},
            timeout=12,
        )
        if r.status_code != 200:
            # Common: 503 while model is loading.
            return _status_payload(
                configured=configured,
                reachable=False,
                last_error=f"HTTP {r.status_code}",
                meta={"model": model},
            )
        data = r.json()
        text = ""
        if isinstance(data, list) and data:
            text = str(data[0].get("generated_text") or "")
        elif isinstance(data, dict):
            text = str(data.get("generated_text") or "")
        return _status_payload(
            configured=configured,
            reachable=bool(text.strip()),
            last_error=None if text.strip() else "Empty response",
            meta={"model": model},
        )
    except Exception as exc:
        return _status_payload(configured=configured, reachable=False, last_error=_truncate_error(str(exc)), meta={"model": model})


def _compute_provider_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "openai": _check_openai(),
        "anthropic": _check_anthropic(),
        "gemini": _check_gemini(),
        "perplexity": _check_perplexity(),
        "huggingface": _check_huggingface(),
    }

    configured = sorted([name for name, details in status.items() if details.get("configured")])
    reachable = sorted([name for name, details in status.items() if details.get("configured") and details.get("reachable")])

    status["_summary"] = {
        "configured_providers": configured,
        "reachable_providers": reachable,
        "any_provider_healthy": bool(reachable),
        "all_providers_healthy": bool(configured) and len(reachable) == len(configured),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return status


def get_provider_status() -> dict[str, Any]:
    """Return a cached snapshot of AI provider status."""
    global _CACHE
    now = time.monotonic()
    if _CACHE is not None:
        cached_payload, expires_at = _CACHE
        if expires_at > now:
            return {**cached_payload, "_summary": {**cached_payload.get("_summary", {}), "cached": True}}

    payload = _compute_provider_status()
    _CACHE = (payload, now + _CACHE_TTL_S)
    return {**payload, "_summary": {**payload.get("_summary", {}), "cached": False}}


__all__ = ["get_provider_status"]

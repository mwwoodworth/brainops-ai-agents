#!/usr/bin/env python3
"""
AI Provider Status Introspection

Runtime checks for OpenAI, Anthropic, Gemini, Perplexity, and Hugging Face.
This module does NOT modify any configuration or credentials; it only reports
configuration and basic liveness so AUREA and the Command Center can see what
is actually working.
"""

from __future__ import annotations

import logging
from typing import Any

from ai_advanced_providers import advanced_ai
from ai_smart_fallback import SmartAISystem

logger = logging.getLogger(__name__)


def _check_openai(system: SmartAISystem) -> dict[str, Any]:
    configured = bool(system.openai_key)
    if not system.openai_client:
        return {"configured": configured, "reachable": False, "last_error": None}

    try:
        text = system._try_openai("ping", max_tokens=4, timeout=3)
        return {
            "configured": configured,
            "reachable": bool(text),
            "last_error": None if text else "No response from OpenAI with current key/model",
        }
    except Exception as exc:  # defensive
        logger.warning("OpenAI provider check failed: %s", exc)
        return {"configured": configured, "reachable": False, "last_error": str(exc)}


def _check_anthropic(system: SmartAISystem) -> dict[str, Any]:
    configured = bool(system.anthropic_key)
    if not system.anthropic_client:
        return {"configured": configured, "reachable": False, "last_error": None}

    try:
        text = system._try_anthropic("ping", max_tokens=4, timeout=3)
        return {
            "configured": configured,
            "reachable": bool(text),
            "last_error": None if text else "No response from Anthropic with current key/model",
        }
    except Exception as exc:  # defensive
        logger.warning("Anthropic provider check failed: %s", exc)
        return {"configured": configured, "reachable": False, "last_error": str(exc)}


def _check_huggingface(system: SmartAISystem) -> dict[str, Any]:
    configured = bool(system.hf_token)
    try:
        text = system._try_huggingface("ping", max_tokens=8)
        return {
            "configured": configured,
            "reachable": bool(text),
            "last_error": None if text else "No response from Hugging Face with current token/models",
        }
    except Exception as exc:  # defensive
        logger.warning("Hugging Face provider check failed: %s", exc)
        return {"configured": configured, "reachable": False, "last_error": str(exc)}


def _check_gemini() -> dict[str, Any]:
    configured = bool(getattr(advanced_ai, "gemini_key", None))
    model_loaded = bool(getattr(advanced_ai, "gemini_model", None))
    if not (configured and model_loaded):
        return {"configured": configured, "reachable": False, "last_error": None}

    try:
        text = advanced_ai.generate_with_gemini("ping", max_tokens=4)
        return {
            "configured": configured,
            "reachable": bool(text),
            "last_error": None if text else "No response from Gemini with current key/model",
        }
    except Exception as exc:  # defensive
        logger.warning("Gemini provider check failed: %s", exc)
        return {"configured": configured, "reachable": False, "last_error": str(exc)}


def _check_perplexity() -> dict[str, Any]:
    configured = bool(getattr(advanced_ai, "perplexity_key", None))
    try:
        result = advanced_ai.search_with_perplexity("ping", citations=False)
        return {
            "configured": configured,
            "reachable": bool(result),
            "last_error": None if result else "No response from Perplexity with current key/model",
        }
    except Exception as exc:  # defensive
        logger.warning("Perplexity provider check failed: %s", exc)
        return {"configured": configured, "reachable": False, "last_error": str(exc)}


def get_provider_status() -> dict[str, Any]:
    """
    Return a snapshot of AI provider status.
    """
    system = SmartAISystem()

    status: dict[str, Any] = {
        "openai": _check_openai(system),
        "anthropic": _check_anthropic(system),
        "huggingface": _check_huggingface(system),
        "gemini": _check_gemini(),
        "perplexity": _check_perplexity(),
    }

    healthy = {
        name: details
        for name, details in status.items()
        if details.get("configured") and details.get("reachable")
    }

    status["_summary"] = {
        "configured_providers": sorted(
            [name for name, details in status.items() if not name.startswith("_") and details.get("configured")]
        ),
        "reachable_providers": sorted(healthy.keys()),
        "all_providers_healthy": bool(healthy),
    }

    return status


__all__ = ["get_provider_status"]


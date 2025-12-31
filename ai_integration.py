#!/usr/bin/env python3
"""
AI Integration Layer
Safely integrates all AI providers without breaking on missing dependencies
"""

import logging

logger = logging.getLogger(__name__)

# Track available AI systems
AI_SYSTEMS = {
    "core": False,
    "smart_fallback": False,
    "advanced": False,
    "ultimate": False
}

# Try to import AI core
try:
    from ai_core import ai_core, generate_ai_response
    AI_SYSTEMS["core"] = True
    logger.info("✅ AI Core loaded")
except ImportError as e:
    logger.warning(f"AI Core not available: {e}")
    ai_core = None

# Try to import smart fallback
try:
    from ai_smart_fallback import smart_ai
    AI_SYSTEMS["smart_fallback"] = True
    logger.info("✅ Smart Fallback loaded")
except ImportError as e:
    logger.warning(f"Smart Fallback not available: {e}")
    smart_ai = None

# Try to import advanced providers
try:
    from ai_advanced_providers import (
        advanced_ai,
        generate_with_gemini_endpoint,
        search_with_perplexity_endpoint
    )
    AI_SYSTEMS["advanced"] = True
    logger.info("✅ Advanced Providers loaded")
except ImportError as e:
    logger.warning(f"Advanced Providers not available: {e}")
    advanced_ai = None

# Try to import ultimate system
try:
    from ai_ultimate_system import (
        ultimate_ai,
        ai_generate_ultimate
    )
    AI_SYSTEMS["ultimate"] = True
    logger.info("✅ Ultimate System loaded")
except ImportError as e:
    logger.warning(f"Ultimate System not available: {e}")
    ultimate_ai = None

# Fallback function if no AI is available
def fallback_ai_response(prompt: str, **kwargs):
    """Generate response when no AI is available"""
    return {
        "response": f"Processing request: {prompt[:100]}... Analysis complete.",
        "provider": "fallback",
        "error": "No AI providers available"
    }

# Unified AI interface
async def generate_ai_unified(
    prompt: str,
    provider: str = "auto",
    task_type: str = "general",
    max_tokens: int = 1000,
    **kwargs
):
    """
    Unified AI generation interface
    Automatically selects best available provider
    """

    # Try Ultimate system first (has all providers)
    if provider == "auto" and ultimate_ai:
        try:
            return await ai_generate_ultimate(prompt, task_type, max_tokens)
        except Exception as e:
            logger.error(f"Ultimate AI error: {e}")

    # Try Smart fallback system
    if smart_ai and provider in ["auto", "smart"]:
        try:
            return smart_ai.generate(prompt, max_tokens)
        except Exception as e:
            logger.error(f"Smart AI error: {e}")

    # Try Advanced providers for specific tasks
    if advanced_ai and provider in ["auto", "gemini", "perplexity"]:
        try:
            if "search" in prompt.lower() or "current" in prompt.lower():
                return await search_with_perplexity_endpoint(prompt)
            else:
                return await generate_with_gemini_endpoint(prompt, max_tokens)
        except Exception as e:
            logger.error(f"Advanced AI error: {e}")

    # Try basic AI core
    if ai_core and provider in ["auto", "openai", "anthropic"]:
        try:
            return await generate_ai_response(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"Core AI error: {e}")

    # Last resort fallback
    return fallback_ai_response(prompt, **kwargs)

def get_ai_status():
    """Get status of all AI systems"""
    status = {
        "systems_available": AI_SYSTEMS,
        "providers": {}
    }

    # Check each system's providers
    if ai_core:
        status["providers"]["openai"] = bool(ai_core.openai_client)
        status["providers"]["anthropic"] = bool(ai_core.anthropic_client)

    if smart_ai:
        smart_status = smart_ai.get_status()
        status["providers"].update(smart_status.get("providers", {}))

    if advanced_ai:
        adv_status = advanced_ai.get_status()
        status["providers"].update(adv_status)

    if ultimate_ai:
        ult_status = ultimate_ai.get_system_status()
        status["providers"].update(ult_status.get("providers", {}))

    # Overall status
    status["operational"] = any(AI_SYSTEMS.values())
    status["message"] = "AI System Operational" if status["operational"] else "No AI providers available"

    return status

# Export unified interface
__all__ = [
    "generate_ai_unified",
    "get_ai_status",
    "AI_SYSTEMS"
]

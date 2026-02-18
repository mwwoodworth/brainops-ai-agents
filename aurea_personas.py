#!/usr/bin/env python3
"""AUREA persona and capability-scope resolution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

VALID_SCOPES = {"read_only", "operator", "admin"}


@dataclass(frozen=True)
class PersonaDefinition:
    persona_id: str
    display_name: str
    prompt_file: str
    default_scopes: tuple[str, ...]
    description: str


PERSONAS: dict[str, PersonaDefinition] = {
    "core": PersonaDefinition(
        persona_id="core",
        display_name="AUREA Core",
        prompt_file="core.md",
        default_scopes=("admin",),
        description="Founder/operator control-plane assistant.",
    ),
    "erp_operator": PersonaDefinition(
        persona_id="erp_operator",
        display_name="ERP Operator",
        prompt_file="erp.md",
        default_scopes=("operator",),
        description="Weathercraft ERP operations assistant.",
    ),
    "mrg_advisor": PersonaDefinition(
        persona_id="mrg_advisor",
        display_name="MRG Advisor",
        prompt_file="mrg.md",
        default_scopes=("read_only",),
        description="MyRoofGenius roofing advisor.",
    ),
    "bss_architect": PersonaDefinition(
        persona_id="bss_architect",
        display_name="BSS Architect",
        prompt_file="bss.md",
        default_scopes=("admin",),
        description="BrainStack Studio engineering copilot.",
    ),
}


def _normalize_scope(scope: Any) -> str | None:
    if scope is None:
        return None
    candidate = str(scope).strip().lower()
    if candidate in VALID_SCOPES:
        return candidate
    return None


def _expand_scopes(scopes: set[str]) -> set[str]:
    expanded = {scope for scope in scopes if scope in VALID_SCOPES}
    if "admin" in expanded:
        expanded.update({"operator", "read_only"})
    elif "operator" in expanded:
        expanded.add("read_only")
    if not expanded:
        expanded = {"read_only"}
    return expanded


def _scopes_from_context(context: dict[str, Any], persona: PersonaDefinition) -> set[str]:
    raw = context.get("allowed_scopes")
    parsed: set[str] = set()

    if isinstance(raw, str):
        for part in raw.split(","):
            normalized = _normalize_scope(part)
            if normalized:
                parsed.add(normalized)
    elif isinstance(raw, (list, tuple, set)):
        for value in raw:
            normalized = _normalize_scope(value)
            if normalized:
                parsed.add(normalized)

    if not parsed:
        parsed = set(persona.default_scopes)
    return _expand_scopes(parsed)


def _restrict_scopes_for_role(scopes: set[str], user_role: Any) -> set[str]:
    role = (str(user_role or "").strip().lower())
    if role in {"viewer", "read_only", "readonly", "customer"}:
        return {"read_only"}
    if role in {"operator", "manager", "staff"}:
        return {"read_only", "operator"}
    if role in {"admin", "owner", "founder", "super_admin", "superadmin"}:
        return _expand_scopes(scopes | {"admin"})
    return _expand_scopes(scopes)


def _resolve_persona_id(context: dict[str, Any]) -> str:
    requested = (str(context.get("persona") or "").strip().lower())
    if requested in PERSONAS:
        return requested

    source = (str(context.get("source") or "").strip().lower())

    if any(token in source for token in ("myroofgenius", "mrg", "roof")):
        return "mrg_advisor"
    if any(token in source for token in ("weathercraft", "erp")):
        return "erp_operator"
    if any(token in source for token in ("brainstack", "bss", "studio")):
        return "bss_architect"

    return "core"


def build_execution_profile(context: dict[str, Any] | None) -> dict[str, Any]:
    context = context or {}
    persona_id = _resolve_persona_id(context)
    persona = PERSONAS.get(persona_id, PERSONAS["core"])

    scopes = _scopes_from_context(context, persona)
    scopes = _restrict_scopes_for_role(scopes, context.get("user_role"))

    return {
        "persona_id": persona.persona_id,
        "persona_name": persona.display_name,
        "persona_description": persona.description,
        "source": context.get("source"),
        "user_role": context.get("user_role"),
        "tenant_id": context.get("tenant_id"),
        "allowed_scopes": sorted(_expand_scopes(scopes)),
    }


@lru_cache(maxsize=16)
def load_persona_prompt(persona_id: str) -> str:
    persona = PERSONAS.get(persona_id, PERSONAS["core"])
    prompt_path = Path(__file__).resolve().parent / "prompts" / "personas" / persona.prompt_file
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        return "You are AUREA. Stay factual, safe, and operationally useful."

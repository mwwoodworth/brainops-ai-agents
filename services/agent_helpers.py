"""
Agent data conversion helpers.

Extracted from app.py during Phase 2 Wave 2B.
Provides row-to-model conversion and JSON normalization for agent data.
"""

import json
from typing import Any

from models.agent import Agent, AgentCategory


def _parse_capabilities(raw: Any) -> list[dict[str, Any]]:
    """Normalize capabilities payload into the Pydantic-friendly format."""
    if raw is None:
        return []

    data: Any = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [
                {
                    "name": raw,
                    "description": "",
                    "enabled": True,
                    "parameters": {},
                }
            ]

    if isinstance(data, dict):
        data = [data]

    capabilities: list[dict[str, Any]] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, str):
            capabilities.append(
                {
                    "name": item,
                    "description": "",
                    "enabled": True,
                    "parameters": {},
                }
            )
        elif isinstance(item, dict):
            capabilities.append(
                {
                    "name": item.get("name")
                    or item.get("capability")
                    or item.get("id")
                    or "capability",
                    "description": item.get("description", ""),
                    "enabled": bool(item.get("enabled", True)),
                    "parameters": item.get("parameters", {}),
                }
            )
    return capabilities


def _parse_configuration(raw: Any) -> dict[str, Any]:
    """Normalize configuration payload."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def row_to_agent(row: dict[str, Any]) -> Agent:
    """Convert a database row (asyncpg or fallback dict) to an Agent model."""
    category_value = row.get("category") or AgentCategory.OTHER.value
    if category_value not in {c.value for c in AgentCategory}:
        category_value = AgentCategory.OTHER.value

    return Agent(
        id=str(row["id"]),
        name=row["name"],
        category=category_value,
        description=row.get("description") or "",
        enabled=bool(row.get("enabled", True)),
        capabilities=_parse_capabilities(row.get("capabilities")),
        configuration=_parse_configuration(row.get("configuration")),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        status=row.get("status") or ("active" if row.get("enabled", True) else "inactive"),
        type=row.get("type") or row.get("agent_type"),
        total_executions=row.get("total_executions") or row.get("execution_count") or 0,
        last_active=row.get("last_active") or row.get("last_active_at") or row.get("updated_at"),
    )

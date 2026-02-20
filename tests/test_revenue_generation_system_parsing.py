import pytest

import revenue_generation_system as rgs


def test_parse_ai_json_payload_accepts_fenced_object():
    payload = """```json
    {"location":"Texas","company_size":"small","indicators":["growth"]}
    ```"""
    parsed = rgs._parse_ai_json_payload(payload, expected_type="object")
    assert isinstance(parsed, dict)
    assert parsed["location"] == "Texas"


def test_parse_ai_json_payload_extracts_array_from_mixed_text():
    payload = "Here are the leads:\n```json\n[{\"company_name\":\"Acme Roofing\"}]\n```\nUse these first."
    parsed = rgs._parse_ai_json_payload(payload, expected_type="array")
    assert isinstance(parsed, list)
    assert parsed[0]["company_name"] == "Acme Roofing"


@pytest.mark.asyncio
async def test_identify_new_leads_uses_fenced_json_search_params(monkeypatch):
    async def _fake_generate(*_args, **_kwargs):
        return """```json
        {"location":"Texas","company_size":"small","indicators":["growth"]}
        ```"""

    captured: dict[str, object] = {}

    async def _fake_discover(search_params):
        captured["search_params"] = search_params
        return [{"company_name": "Acme Roofing"}]

    async def _fake_store(_lead):
        return "lead-1"

    async def _fake_log(*_args, **_kwargs):
        return None

    monkeypatch.setattr(rgs, "_generate_text", _fake_generate)

    system = rgs.AutonomousRevenueSystem.__new__(rgs.AutonomousRevenueSystem)
    system._discover_leads = _fake_discover  # type: ignore[attr-defined]
    system._store_lead = _fake_store  # type: ignore[attr-defined]
    system._log_action = _fake_log  # type: ignore[attr-defined]

    lead_ids = await system.identify_new_leads({"location": "ignored"})

    assert lead_ids == ["lead-1"]
    assert captured["search_params"] == {
        "location": "Texas",
        "company_size": "small",
        "indicators": ["growth"],
    }


@pytest.mark.asyncio
async def test_identify_new_leads_skips_discovery_on_invalid_json(monkeypatch):
    async def _fake_generate(*_args, **_kwargs):
        return "lead_generation_strategy: Roofing Contractor Lead Generation - US SMB"

    calls = {"discover": 0}

    async def _fake_discover(_search_params):
        calls["discover"] += 1
        return []

    async def _fake_store(_lead):
        return "lead-1"

    async def _fake_log(*_args, **_kwargs):
        return None

    monkeypatch.setattr(rgs, "_generate_text", _fake_generate)

    system = rgs.AutonomousRevenueSystem.__new__(rgs.AutonomousRevenueSystem)
    system._discover_leads = _fake_discover  # type: ignore[attr-defined]
    system._store_lead = _fake_store  # type: ignore[attr-defined]
    system._log_action = _fake_log  # type: ignore[attr-defined]

    lead_ids = await system.identify_new_leads({"location": "ignored"})

    assert lead_ids == []
    assert calls["discover"] == 0

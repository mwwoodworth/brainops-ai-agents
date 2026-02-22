"""
Tests for api/pipeline.py — Pipeline State Machine API.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.pipeline as pipeline_api
from pipeline_state_machine import PipelineState, VALID_TRANSITIONS


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(pipeline_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /pipeline/states
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_valid_states(client):
    resp = await client.get("/pipeline/states")
    assert resp.status_code == 200
    data = resp.json()
    assert "states" in data
    assert "transitions" in data
    assert "terminal_states" in data
    assert "paid" in data["terminal_states"]
    assert "lost" in data["terminal_states"]
    assert len(data["states"]) > 0


@pytest.mark.asyncio
async def test_states_match_enum(client):
    resp = await client.get("/pipeline/states")
    data = resp.json()
    enum_values = {s.value for s in PipelineState}
    api_values = set(data["states"])
    assert api_values == enum_values


@pytest.mark.asyncio
async def test_transitions_present(client):
    resp = await client.get("/pipeline/states")
    data = resp.json()
    transitions = data["transitions"]
    # Every non-terminal state should have at least one transition target
    for state_val, targets in transitions.items():
        if state_val not in ("paid", "lost"):
            assert len(targets) > 0, f"State {state_val} has no transitions"


# ---------------------------------------------------------------------------
# POST /pipeline/{lead_id}/transition — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transition_validation_missing_fields(client):
    resp = await client.post(
        "/pipeline/lead/lead-123/transition",
        json={},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_transition_validation_missing_trigger(client):
    resp = await client.post(
        "/pipeline/lead/lead-123/transition",
        json={"to_state": "enriched"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# PipelineState enum
# ---------------------------------------------------------------------------


class TestPipelineState:
    def test_all_states_have_string_values(self):
        for state in PipelineState:
            assert isinstance(state.value, str)
            assert len(state.value) > 0

    def test_terminal_states_exist(self):
        values = {s.value for s in PipelineState}
        assert "paid" in values
        assert "lost" in values


# ---------------------------------------------------------------------------
# VALID_TRANSITIONS structure
# ---------------------------------------------------------------------------


class TestValidTransitions:
    def test_transitions_dict_not_empty(self):
        assert len(VALID_TRANSITIONS) > 0

    def test_all_keys_are_pipeline_states(self):
        for key in VALID_TRANSITIONS:
            assert isinstance(key, PipelineState)

    def test_all_targets_are_pipeline_states(self):
        for key, targets in VALID_TRANSITIONS.items():
            for t in targets:
                assert isinstance(t, PipelineState)

    def test_terminal_states_have_no_outgoing(self):
        # Terminal states (paid, lost) should not have outgoing transitions
        for state in (PipelineState("paid"), PipelineState("lost")):
            if state in VALID_TRANSITIONS:
                assert len(VALID_TRANSITIONS[state]) == 0

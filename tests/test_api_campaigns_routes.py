"""
Tests for api/campaigns.py — Campaign management route handlers.
"""
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.campaigns as campaigns_api
from campaign_manager import CAMPAIGNS


@pytest_asyncio.fixture
async def client(monkeypatch):
    app = FastAPI()
    app.include_router(campaigns_api.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /campaigns/
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_campaigns_success(client):
    resp = await client.get("/campaigns/")
    assert resp.status_code == 200
    data = resp.json()
    assert "campaigns" in data
    assert "total" in data
    assert data["total"] >= 0


@pytest.mark.asyncio
async def test_list_campaigns_include_inactive(client):
    resp = await client.get("/campaigns/?active_only=false")
    assert resp.status_code == 200
    data = resp.json()
    assert "campaigns" in data


# ---------------------------------------------------------------------------
# GET /campaigns/{campaign_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_campaign_success(client):
    if not CAMPAIGNS:
        pytest.skip("No campaigns configured")

    campaign_id = list(CAMPAIGNS.keys())[0]
    resp = await client.get(f"/campaigns/{campaign_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == campaign_id


@pytest.mark.asyncio
async def test_get_campaign_not_found(client):
    resp = await client.get("/campaigns/nonexistent-campaign-xyz")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


class TestProspectInput:
    def test_valid_minimal(self):
        p = campaigns_api.ProspectInput(
            company_name="Acme",
            email="acme@test.com",
        )
        assert p.company_name == "Acme"
        assert p.state == "CO"  # default

    def test_valid_full(self):
        p = campaigns_api.ProspectInput(
            company_name="Acme",
            contact_name="Jordan",
            email="jordan@acme.com",
            phone="555-0100",
            website="https://acme.com",
            building_type="warehouse",
            city="Denver",
            state="CO",
            estimated_sqft=50000,
            roof_system="TPO",
        )
        assert p.estimated_sqft == 50000


class TestEnrollLeadInput:
    def test_valid(self):
        inp = campaigns_api.EnrollLeadInput(
            campaign_id="roofing-campaign",
            lead_id=str(uuid.uuid4()),
        )
        assert inp.campaign_id == "roofing-campaign"


class TestBatchEnrollInput:
    def test_valid(self):
        inp = campaigns_api.BatchEnrollInput(
            campaign_id="roofing",
            limit=50,
        )
        assert inp.limit == 50

    def test_limit_min_boundary(self):
        inp = campaigns_api.BatchEnrollInput(
            campaign_id="roofing",
            limit=1,
        )
        assert inp.limit == 1

    def test_limit_max_boundary(self):
        inp = campaigns_api.BatchEnrollInput(
            campaign_id="roofing",
            limit=200,
        )
        assert inp.limit == 200

    def test_limit_above_max_raises(self):
        with pytest.raises(Exception):
            campaigns_api.BatchEnrollInput(
                campaign_id="roofing",
                limit=201,
            )

    def test_limit_below_min_raises(self):
        with pytest.raises(Exception):
            campaigns_api.BatchEnrollInput(
                campaign_id="roofing",
                limit=0,
            )


class TestWebsiteDiscoveryInput:
    def test_valid(self):
        inp = campaigns_api.WebsiteDiscoveryInput(
            websites=["https://acme.com", "https://bigbox.com"],
        )
        assert len(inp.websites) == 2

    def test_empty_websites(self):
        inp = campaigns_api.WebsiteDiscoveryInput(websites=[])
        assert inp.websites == []

    def test_with_filters(self):
        inp = campaigns_api.WebsiteDiscoveryInput(
            websites=["https://acme.com"],
            building_type="warehouse",
            city="Denver",
        )
        assert inp.building_type == "warehouse"


class TestBatchProspectsInput:
    def test_valid(self):
        inp = campaigns_api.BatchProspectsInput(
            prospects=[
                campaigns_api.ProspectInput(company_name="A", email="a@a.com"),
                campaigns_api.ProspectInput(company_name="B", email="b@b.com"),
            ]
        )
        assert len(inp.prospects) == 2

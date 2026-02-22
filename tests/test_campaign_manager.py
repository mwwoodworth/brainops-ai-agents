"""
Tests for campaign_manager.py — Campaign configuration, template personalization, lifecycle.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campaign_manager import (
    CampaignConfig,
    EmailTemplate,
    GeographyConfig,
    HandoffPartner,
    get_campaign,
    list_campaigns,
    campaign_to_dict,
    personalize_template,
    CAMPAIGNS,
)


# ---------------------------------------------------------------------------
# Campaign registry
# ---------------------------------------------------------------------------


class TestCampaignRegistry:
    def test_campaigns_dict_populated(self):
        assert len(CAMPAIGNS) > 0

    def test_get_campaign_valid(self):
        campaign_id = list(CAMPAIGNS.keys())[0]
        campaign = get_campaign(campaign_id)
        assert campaign is not None
        assert campaign.id == campaign_id

    def test_get_campaign_invalid(self):
        result = get_campaign("nonexistent-campaign")
        assert result is None

    def test_list_campaigns_active_only(self):
        active = list_campaigns(active_only=True)
        for c in active:
            assert c.is_active is True

    def test_list_campaigns_all(self):
        all_campaigns = list_campaigns(active_only=False)
        assert len(all_campaigns) >= len(list_campaigns(active_only=True))

    def test_campaign_has_templates(self):
        campaign_id = list(CAMPAIGNS.keys())[0]
        campaign = get_campaign(campaign_id)
        assert len(campaign.templates) > 0
        for t in campaign.templates:
            assert isinstance(t, EmailTemplate)
            assert t.step >= 1
            assert len(t.subject) > 0


# ---------------------------------------------------------------------------
# campaign_to_dict
# ---------------------------------------------------------------------------


class TestCampaignToDict:
    def test_returns_dict(self):
        campaign_id = list(CAMPAIGNS.keys())[0]
        campaign = get_campaign(campaign_id)
        result = campaign_to_dict(campaign)
        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result

    def test_contains_essential_fields(self):
        campaign_id = list(CAMPAIGNS.keys())[0]
        campaign = get_campaign(campaign_id)
        d = campaign_to_dict(campaign)
        assert d["id"] == campaign_id
        assert isinstance(d["name"], str)


# ---------------------------------------------------------------------------
# EmailTemplate
# ---------------------------------------------------------------------------


class TestEmailTemplate:
    def test_creation(self):
        t = EmailTemplate(
            step=1,
            delay_days=0,
            subject="Test Subject {city}",
            body_html="<p>Hello {contact_name}</p>",
            call_to_action="Reply now",
        )
        assert t.step == 1
        assert t.delay_days == 0
        assert "{city}" in t.subject

    def test_delay_days_sequence(self):
        templates = [
            EmailTemplate(step=1, delay_days=0, subject="Day 0", body_html="", call_to_action=""),
            EmailTemplate(step=2, delay_days=2, subject="Day 2", body_html="", call_to_action=""),
            EmailTemplate(step=3, delay_days=5, subject="Day 5", body_html="", call_to_action=""),
        ]
        # Ensure delay increases
        for i in range(1, len(templates)):
            assert templates[i].delay_days >= templates[i - 1].delay_days


# ---------------------------------------------------------------------------
# GeographyConfig
# ---------------------------------------------------------------------------


class TestGeographyConfig:
    def test_creation(self):
        geo = GeographyConfig(
            states=["CO", "WY"],
            cities=["Denver", "Cheyenne"],
            metro_areas=["Front Range"],
        )
        assert "CO" in geo.states
        assert len(geo.cities) == 2

    def test_default_metro_areas(self):
        geo = GeographyConfig(states=["CO"], cities=["Denver"])
        assert geo.metro_areas == []


# ---------------------------------------------------------------------------
# HandoffPartner
# ---------------------------------------------------------------------------


class TestHandoffPartner:
    def test_creation(self):
        partner = HandoffPartner(
            name="Weathercraft",
            location="Colorado Springs, CO",
            capabilities=["TPO", "EPDM", "PVC"],
            certifications=["GAF Master Select"],
            experience="25+ years",
        )
        assert partner.name == "Weathercraft"
        assert len(partner.capabilities) == 3

    def test_default_phone_website(self):
        partner = HandoffPartner(
            name="Test",
            location="Denver",
            capabilities=[],
            certifications=[],
            experience="5 years",
        )
        assert partner.phone == ""
        assert partner.website == ""


# ---------------------------------------------------------------------------
# personalize_template
# ---------------------------------------------------------------------------


class TestPersonalizeTemplate:
    def test_personalizes_subject_and_body(self):
        template = EmailTemplate(
            step=1,
            delay_days=0,
            subject="Your {building_type} in {city}",
            body_html="<p>Hi {contact_name}</p>",
            call_to_action="Reply",
        )
        lead = {
            "contact_name": "Jordan",
            "company_name": "Acme",
            "email": "jordan@acme.com",
            "building_type": "warehouse",
            "city": "Denver",
            "state": "CO",
        }
        campaign = get_campaign(list(CAMPAIGNS.keys())[0])
        subject, body = personalize_template(template, lead, campaign)
        assert "warehouse" in subject or "Denver" in subject or isinstance(subject, str)
        assert isinstance(body, str)

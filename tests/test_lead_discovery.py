#!/usr/bin/env python3
"""
Tests for the Lead Discovery Engine

Tests cover:
- Lead qualification logic
- Scoring algorithms
- Deduplication
- Source-specific discovery
- ERP sync functionality
"""

import json
import sys
import types

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from lead_discovery_engine import (
    DiscoveredLead,
    LeadDiscoveryEngine,
    LeadQualificationCriteria,
    LeadQualificationStatus,
    LeadSource,
    LeadTier,
    get_discovery_engine,
)


class TestDiscoveredLead:
    """Tests for the DiscoveredLead dataclass"""

    def test_default_values(self):
        """Test that default values are set correctly"""
        lead = DiscoveredLead(company_name="Test Co")

        assert lead.company_name == "Test Co"
        assert lead.source == LeadSource.WEB_SEARCH
        assert lead.score == 50.0
        assert lead.tier == LeadTier.COOL
        assert lead.qualification_status == LeadQualificationStatus.UNQUALIFIED
        assert lead.estimated_value == 5000.0
        assert isinstance(lead.discovered_at, datetime)

    def test_to_dict(self):
        """Test serialization to dictionary"""
        lead = DiscoveredLead(
            company_name="Test Company",
            contact_name="John Doe",
            email="john@test.com",
            source=LeadSource.ERP_REACTIVATION,
            score=75.0,
            signals=["past_customer", "high_value"]
        )

        result = lead.to_dict()

        assert result["company_name"] == "Test Company"
        assert result["email"] == "john@test.com"
        assert result["source"] == "erp_reactivation"
        assert result["score"] == 75.0
        assert "past_customer" in result["signals"]


class TestLeadQualificationCriteria:
    """Tests for qualification criteria"""

    def test_default_criteria(self):
        """Test default qualification criteria"""
        criteria = LeadQualificationCriteria()

        assert criteria.min_score == 40.0
        assert criteria.require_email is True
        assert criteria.require_phone is False
        assert "test.com" in criteria.excluded_domains

    def test_custom_criteria(self):
        """Test custom qualification criteria"""
        criteria = LeadQualificationCriteria(
            min_score=60.0,
            require_phone=True,
            excluded_domains=["custom.com"]
        )

        assert criteria.min_score == 60.0
        assert criteria.require_phone is True
        assert "custom.com" in criteria.excluded_domains


class TestLeadDiscoveryEngine:
    """Tests for the LeadDiscoveryEngine"""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance"""
        return LeadDiscoveryEngine(tenant_id="test-tenant-123")

    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.tenant_id == "test-tenant-123"
        assert isinstance(engine.criteria, LeadQualificationCriteria)

    def test_get_tier_from_score(self, engine):
        """Test tier classification based on score"""
        assert engine._get_tier_from_score(90) == LeadTier.HOT
        assert engine._get_tier_from_score(80) == LeadTier.HOT
        assert engine._get_tier_from_score(70) == LeadTier.WARM
        assert engine._get_tier_from_score(60) == LeadTier.WARM
        assert engine._get_tier_from_score(50) == LeadTier.COOL
        assert engine._get_tier_from_score(40) == LeadTier.COOL
        assert engine._get_tier_from_score(30) == LeadTier.COLD
        assert engine._get_tier_from_score(0) == LeadTier.COLD

    def test_is_excluded_email(self, engine):
        """Test email exclusion logic"""
        assert engine._is_excluded_email("test@test.com") is True
        assert engine._is_excluded_email("test@example.com") is True
        assert engine._is_excluded_email("test@mailinator.com") is True
        assert engine._is_excluded_email("test@realcompany.com") is False
        assert engine._is_excluded_email("") is True
        assert engine._is_excluded_email(None) is True


class TestLeadQualification:
    """Tests for lead qualification logic"""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance"""
        return LeadDiscoveryEngine()

    @pytest.mark.asyncio
    async def test_qualify_lead_with_email(self, engine):
        """Test qualification of lead with valid email"""
        # Mock database operations
        with patch('lead_discovery_engine.get_pool') as mock_pool:
            mock_pool.return_value.execute = AsyncMock()

            lead = DiscoveredLead(
                company_name="Good Company",
                email="valid@realcompany.com",
                source=LeadSource.ERP_UPSELL,
                estimated_value=10000
            )

            result = await engine.qualify_lead(lead)

            # Should be qualified - has email, good source
            assert result.score > 40
            assert result.qualification_status == LeadQualificationStatus.QUALIFIED

    @pytest.mark.asyncio
    async def test_disqualify_lead_without_email(self, engine):
        """Test disqualification of lead without email"""
        with patch('lead_discovery_engine.get_pool') as mock_pool:
            mock_pool.return_value.execute = AsyncMock()

            lead = DiscoveredLead(
                company_name="No Email Company",
                email=None,
                source=LeadSource.WEB_SEARCH
            )

            result = await engine.qualify_lead(lead)

            assert result.qualification_status == LeadQualificationStatus.DISQUALIFIED
            assert "missing_email" in result.metadata.get("disqualification_reasons", [])

    @pytest.mark.asyncio
    async def test_disqualify_lead_with_excluded_email(self, engine):
        """Test disqualification of lead with excluded email domain"""
        with patch('lead_discovery_engine.get_pool') as mock_pool:
            mock_pool.return_value.execute = AsyncMock()

            lead = DiscoveredLead(
                company_name="Test Company",
                email="user@test.com",  # Excluded domain
                source=LeadSource.WEB_SEARCH
            )

            result = await engine.qualify_lead(lead)

            assert result.qualification_status == LeadQualificationStatus.DISQUALIFIED
            assert "excluded_email_domain" in result.metadata.get("disqualification_reasons", [])


class TestLeadScoring:
    """Tests for lead scoring algorithm"""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance"""
        return LeadDiscoveryEngine()

    @pytest.mark.asyncio
    async def test_score_contact_completeness(self, engine):
        """Test scoring based on contact completeness"""
        # Lead with all contact info
        complete_lead = DiscoveredLead(
            company_name="Complete Corp",
            contact_name="John Doe",
            email="john@complete.com",
            phone="555-1234",
            location="Austin, TX"
        )

        # Lead with minimal info
        minimal_lead = DiscoveredLead(
            company_name="Unknown"
        )

        complete_score = await engine._calculate_lead_score(complete_lead)
        minimal_score = await engine._calculate_lead_score(minimal_lead)

        # Complete lead should score higher
        assert complete_score > minimal_score

    @pytest.mark.asyncio
    async def test_score_source_quality(self, engine):
        """Test scoring based on lead source quality"""
        # High-quality source
        upsell_lead = DiscoveredLead(
            company_name="Premium Co",
            email="premium@company.com",
            source=LeadSource.ERP_UPSELL
        )

        # Lower-quality source
        web_lead = DiscoveredLead(
            company_name="Web Co",
            email="web@company.com",
            source=LeadSource.WEB_SEARCH
        )

        upsell_score = await engine._calculate_lead_score(upsell_lead)
        web_score = await engine._calculate_lead_score(web_lead)

        # ERP upsell should score higher than web search
        assert upsell_score > web_score

    @pytest.mark.asyncio
    async def test_score_high_value_signals(self, engine):
        """Test scoring based on buying signals"""
        # Lead with strong signals
        hot_lead = DiscoveredLead(
            company_name="Hot Lead Inc",
            email="hot@lead.com",
            signals=["high_value_customer", "premium_customer", "repeat_customer"]
        )

        # Lead with weak signals
        cold_lead = DiscoveredLead(
            company_name="Cold Lead Inc",
            email="cold@lead.com",
            signals=["basic_signal"]
        )

        hot_score = await engine._calculate_lead_score(hot_lead)
        cold_score = await engine._calculate_lead_score(cold_lead)

        assert hot_score > cold_score

    @pytest.mark.asyncio
    async def test_score_estimated_value(self, engine):
        """Test scoring based on estimated deal value"""
        high_value = DiscoveredLead(
            company_name="Big Corp",
            email="big@corp.com",
            estimated_value=50000
        )

        low_value = DiscoveredLead(
            company_name="Small Corp",
            email="small@corp.com",
            estimated_value=1000
        )

        high_score = await engine._calculate_lead_score(high_value)
        low_score = await engine._calculate_lead_score(low_value)

        assert high_score > low_score


class TestLeadValueEstimation:
    """Tests for explicit value parsing and deterministic estimation."""

    @pytest.fixture
    def engine(self):
        return LeadDiscoveryEngine()

    def test_extract_currency_values(self, engine):
        values = engine._extract_currency_values("RFP is between $250k and 4 million this year")
        assert 250000.0 in values
        assert 4000000.0 in values

    def test_estimate_value_prefers_explicit_amount(self, engine):
        value = engine._estimate_discovery_value(
            primary_text="Commercial upgrade budget approved at $1.2M",
            signals=["intent_high"],
            metadata={"estimated_size": "small"},
        )
        assert value == pytest.approx(1200000.0)

    def test_estimate_value_uses_size_and_intent_fallback(self, engine):
        value = engine._estimate_discovery_value(
            primary_text="Needs operations workflow support",
            signals=["intent_high"],
            metadata={"estimated_size": "medium", "intent_level": "high"},
        )
        assert value > 18000

    @pytest.mark.asyncio
    async def test_web_search_uses_parsed_amount_for_estimated_value(self, engine, monkeypatch):
        payload = [
            {
                "company_name": "Acme Roofing",
                "location": "Denver, CO",
                "website": "https://acme.example",
                "buying_signals": ["Approved modernization budget of $4M for 2026"],
                "estimated_size": "large",
            }
        ]

        class _FakeAdvancedAI:
            @staticmethod
            def search_with_perplexity(_prompt: str):
                return {"answer": json.dumps(payload)}

        monkeypatch.setitem(
            sys.modules,
            "ai_advanced_providers",
            types.SimpleNamespace(advanced_ai=_FakeAdvancedAI),
        )

        leads = await engine._discover_web_search(limit=1)
        assert len(leads) == 1
        assert leads[0].estimated_value == pytest.approx(4000000.0)


class TestLeadDeduplication:
    """Tests for lead deduplication"""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance"""
        return LeadDiscoveryEngine()

    @pytest.mark.asyncio
    async def test_deduplicate_by_email(self, engine):
        """Test deduplication by email address"""
        with patch('lead_discovery_engine.get_pool') as mock_pool:
            # Mock empty existing emails
            mock_pool.return_value.fetch = AsyncMock(return_value=[])

            leads = [
                DiscoveredLead(company_name="Company A", email="same@email.com"),
                DiscoveredLead(company_name="Company B", email="same@email.com"),
                DiscoveredLead(company_name="Company C", email="different@email.com"),
            ]

            result = await engine._deduplicate_leads(leads)

            # Should have 2 unique leads
            assert len(result) == 2
            emails = [l.email for l in result]
            assert "same@email.com" in emails
            assert "different@email.com" in emails

    @pytest.mark.asyncio
    async def test_deduplicate_preserves_first(self, engine):
        """Test that deduplication preserves the first occurrence"""
        with patch('lead_discovery_engine.get_pool') as mock_pool:
            mock_pool.return_value.fetch = AsyncMock(return_value=[])

            leads = [
                DiscoveredLead(company_name="First Company", email="test@example.org"),
                DiscoveredLead(company_name="Second Company", email="test@example.org"),
            ]

            result = await engine._deduplicate_leads(leads)

            assert len(result) == 1
            assert result[0].company_name == "First Company"


class TestSingletonPattern:
    """Tests for the singleton factory function"""

    def test_get_discovery_engine_creates_instance(self):
        """Test that get_discovery_engine creates an instance"""
        engine = get_discovery_engine()
        assert isinstance(engine, LeadDiscoveryEngine)

    def test_get_discovery_engine_returns_same_instance(self):
        """Test that get_discovery_engine returns the same instance"""
        engine1 = get_discovery_engine()
        engine2 = get_discovery_engine()
        assert engine1 is engine2

    def test_get_discovery_engine_with_different_tenant(self):
        """Test that different tenant_id creates new instance"""
        engine1 = get_discovery_engine(tenant_id="tenant-1")
        engine2 = get_discovery_engine(tenant_id="tenant-2")
        # With different tenant, should create new instance
        assert engine2.tenant_id == "tenant-2"


class TestLeadSourceEnum:
    """Tests for LeadSource enumeration"""

    def test_all_sources_have_values(self):
        """Test that all sources have string values"""
        for source in LeadSource:
            assert isinstance(source.value, str)
            assert len(source.value) > 0

    def test_source_values_are_unique(self):
        """Test that all source values are unique"""
        values = [s.value for s in LeadSource]
        assert len(values) == len(set(values))


class TestLeadTierEnum:
    """Tests for LeadTier enumeration"""

    def test_tier_order(self):
        """Test that tier values make sense"""
        tiers = list(LeadTier)
        assert LeadTier.HOT in tiers
        assert LeadTier.WARM in tiers
        assert LeadTier.COOL in tiers
        assert LeadTier.COLD in tiers


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

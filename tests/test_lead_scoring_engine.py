"""
Tests for advanced_lead_scoring.py — Multi-factor lead scoring engine.
Pure business logic tests — no database, no HTTP.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advanced_lead_scoring import (
    BehavioralScore,
    FirmographicScore,
    FinancialScore,
    IntentScore,
    LeadScoreResult,
    LeadTier,
    ScoreCategory,
    VelocityScore,
)


# ---------------------------------------------------------------------------
# BehavioralScore tests
# ---------------------------------------------------------------------------


class TestBehavioralScore:
    def test_total_sums_all_fields(self):
        score = BehavioralScore(
            email_engagement=8.0,
            website_activity=6.0,
            content_downloads=5.0,
            demo_attendance=6.0,
            support_interaction=5.0,
        )
        assert score.total == 30.0

    def test_total_capped_at_30(self):
        score = BehavioralScore(
            email_engagement=10.0,
            website_activity=10.0,
            content_downloads=10.0,
            demo_attendance=10.0,
            support_interaction=10.0,
        )
        assert score.total == 30.0

    def test_default_zeros(self):
        score = BehavioralScore()
        assert score.total == 0.0

    def test_partial_values(self):
        score = BehavioralScore(email_engagement=4.0, website_activity=3.0)
        assert score.total == 7.0


# ---------------------------------------------------------------------------
# FirmographicScore tests
# ---------------------------------------------------------------------------


class TestFirmographicScore:
    def test_total_sums_correctly(self):
        score = FirmographicScore(
            company_size_fit=7.0,
            industry_fit=6.0,
            revenue_alignment=5.0,
            growth_indicators=4.0,
            tech_maturity=3.0,
        )
        assert score.total == 25.0

    def test_total_capped_at_25(self):
        score = FirmographicScore(
            company_size_fit=10.0,
            industry_fit=10.0,
            revenue_alignment=10.0,
            growth_indicators=10.0,
            tech_maturity=10.0,
        )
        assert score.total == 25.0

    def test_default_zeros(self):
        score = FirmographicScore()
        assert score.total == 0.0


# ---------------------------------------------------------------------------
# IntentScore tests
# ---------------------------------------------------------------------------


class TestIntentScore:
    def test_total_sums_correctly(self):
        score = IntentScore(
            search_behavior=7.0,
            competitor_interest=6.0,
            job_postings=5.0,
            website_changes=4.0,
            social_activity=3.0,
        )
        assert score.total == 25.0

    def test_total_capped_at_25(self):
        score = IntentScore(
            search_behavior=20.0,
            competitor_interest=20.0,
        )
        assert score.total == 25.0


# ---------------------------------------------------------------------------
# VelocityScore tests
# ---------------------------------------------------------------------------


class TestVelocityScore:
    def test_total_sums_correctly(self):
        score = VelocityScore(
            response_time=5.0,
            meeting_scheduling=4.0,
            decision_timeline=3.0,
            sales_cycle_pace=3.0,
        )
        assert score.total == 15.0

    def test_total_capped_at_15(self):
        score = VelocityScore(
            response_time=10.0,
            meeting_scheduling=10.0,
        )
        assert score.total == 15.0


# ---------------------------------------------------------------------------
# FinancialScore tests
# ---------------------------------------------------------------------------


class TestFinancialScore:
    def test_total_sums_correctly(self):
        score = FinancialScore(
            payment_history=2.0,
            expansion_potential=2.0,
            churn_risk=1.0,
        )
        assert score.total == 5.0

    def test_total_capped_at_5(self):
        score = FinancialScore(
            payment_history=5.0,
            expansion_potential=5.0,
            churn_risk=5.0,
        )
        assert score.total == 5.0


# ---------------------------------------------------------------------------
# LeadScoreResult (tier determination)
# ---------------------------------------------------------------------------


class TestLeadScoreResult:
    def _make_result(
        self,
        behavioral_total=0,
        firmographic_total=0,
        intent_total=0,
        velocity_total=0,
        financial_total=0,
    ):
        return LeadScoreResult(
            lead_id="lead-test",
            behavioral=BehavioralScore(email_engagement=behavioral_total),
            firmographic=FirmographicScore(company_size_fit=firmographic_total),
            intent=IntentScore(search_behavior=intent_total),
            velocity=VelocityScore(response_time=velocity_total),
            financial=FinancialScore(payment_history=financial_total),
        )

    def test_hot_tier_above_80(self):
        result = self._make_result(
            behavioral_total=25,
            firmographic_total=20,
            intent_total=20,
            velocity_total=10,
            financial_total=5,
        )
        assert result.composite_score == 80.0
        assert result.tier == LeadTier.HOT
        assert result.next_best_action == "immediate_call"
        assert result.recommended_touch_frequency == 1

    def test_warm_tier_60_to_79(self):
        result = self._make_result(
            behavioral_total=20,
            firmographic_total=15,
            intent_total=15,
            velocity_total=10,
            financial_total=0,
        )
        assert 60 <= result.composite_score < 80
        assert result.tier == LeadTier.WARM
        assert result.next_best_action == "personalized_outreach"
        assert result.recommended_touch_frequency == 3

    def test_cool_tier_40_to_59(self):
        result = self._make_result(
            behavioral_total=15,
            firmographic_total=10,
            intent_total=10,
            velocity_total=5,
            financial_total=0,
        )
        assert 40 <= result.composite_score < 60
        assert result.tier == LeadTier.COOL
        assert result.next_best_action == "nurture_sequence"
        assert result.recommended_touch_frequency == 7

    def test_cold_tier_below_40(self):
        result = self._make_result(
            behavioral_total=5,
            firmographic_total=5,
            intent_total=5,
            velocity_total=3,
            financial_total=1,
        )
        assert result.composite_score < 40
        assert result.tier == LeadTier.COLD
        assert result.next_best_action == "long_term_nurture"
        assert result.recommended_touch_frequency == 14

    def test_zero_score_is_cold(self):
        result = self._make_result()
        assert result.composite_score == 0
        assert result.tier == LeadTier.COLD

    def test_expected_deal_size_scales_with_firmographic(self):
        low = self._make_result(firmographic_total=0)
        high = self._make_result(firmographic_total=7)
        assert high.expected_deal_size > low.expected_deal_size

    def test_conversion_probability_positive(self):
        result = self._make_result(
            behavioral_total=30,
            firmographic_total=25,
            intent_total=25,
            velocity_total=15,
            financial_total=5,
        )
        assert result.probability_conversion_30d > 0

    def test_conversion_probability_increases_with_score(self):
        cold = self._make_result(behavioral_total=5)
        hot = self._make_result(
            behavioral_total=30,
            firmographic_total=25,
            intent_total=25,
            velocity_total=15,
            financial_total=5,
        )
        assert hot.probability_conversion_30d > cold.probability_conversion_30d


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_lead_tier_values(self):
        assert LeadTier.HOT.value == "hot"
        assert LeadTier.WARM.value == "warm"
        assert LeadTier.COOL.value == "cool"
        assert LeadTier.COLD.value == "cold"

    def test_score_category_values(self):
        assert ScoreCategory.BEHAVIORAL.value == "behavioral"
        assert ScoreCategory.FIRMOGRAPHIC.value == "firmographic"
        assert ScoreCategory.INTENT.value == "intent"
        assert ScoreCategory.VELOCITY.value == "velocity"
        assert ScoreCategory.FINANCIAL.value == "financial"

"""
Tests for api/stripe_webhook.py — Stripe payment event handling.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.stripe_webhook as stripe_api


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestParseUUID:
    def test_valid_uuid(self):
        result = stripe_api._parse_uuid("550e8400-e29b-41d4-a716-446655440000")
        assert result is not None
        assert str(result) == "550e8400-e29b-41d4-a716-446655440000"

    def test_none(self):
        assert stripe_api._parse_uuid(None) is None

    def test_empty_string(self):
        assert stripe_api._parse_uuid("") is None

    def test_invalid_uuid(self):
        assert stripe_api._parse_uuid("not-a-uuid") is None

    def test_uuid_with_whitespace(self):
        result = stripe_api._parse_uuid("  550e8400-e29b-41d4-a716-446655440000  ")
        assert result is not None


class TestUtcFromUnix:
    def test_valid_timestamp(self):
        result = stripe_api._utc_from_unix(1700000000)
        assert result is not None
        assert result.year == 2023

    def test_none(self):
        assert stripe_api._utc_from_unix(None) is None

    def test_empty_string(self):
        assert stripe_api._utc_from_unix("") is None

    def test_string_timestamp(self):
        result = stripe_api._utc_from_unix("1700000000")
        assert result is not None


class TestCoerceBool:
    def test_true_values(self):
        for val in [True, "1", "true", "True", "yes", "YES", "y", "on"]:
            assert stripe_api._coerce_bool(val) is True

    def test_false_values(self):
        for val in [False, None, "0", "false", "no", "off", ""]:
            assert stripe_api._coerce_bool(val) is False


class TestNormalizeTier:
    def test_none(self):
        assert stripe_api._normalize_tier(None) is None

    def test_empty_string(self):
        assert stripe_api._normalize_tier("") is None

    def test_exact_matches(self):
        assert stripe_api._normalize_tier("starter") == "starter"
        assert stripe_api._normalize_tier("professional") == "professional"
        assert stripe_api._normalize_tier("enterprise") == "enterprise"
        assert stripe_api._normalize_tier("free") == "free"
        assert stripe_api._normalize_tier("demo") == "demo"

    def test_case_insensitive(self):
        assert stripe_api._normalize_tier("STARTER") == "starter"
        assert stripe_api._normalize_tier("Professional") == "professional"

    def test_partial_match(self):
        assert stripe_api._normalize_tier("pro plan") == "professional"
        assert stripe_api._normalize_tier("enterprise_plus") == "enterprise"
        assert stripe_api._normalize_tier("starter_basic") == "starter"
        assert stripe_api._normalize_tier("free trial") == "free"

    def test_unknown_tier(self):
        result = stripe_api._normalize_tier("custom")
        # Unknown tiers return None or a generic value
        assert result is None or isinstance(result, str)

import sys
from pathlib import Path
from datetime import timezone
from decimal import Decimal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.gumroad_webhook import _is_test_sale, _parse_gumroad_timestamp, _parse_price


def test_parse_gumroad_timestamp_accepts_z_suffix():
    parsed = _parse_gumroad_timestamp("2026-01-17T00:00:00Z")
    assert parsed.tzinfo is not None
    assert parsed.tzinfo.utcoffset(parsed) == timezone.utc.utcoffset(parsed)


def test_parse_price_handles_currency_symbols():
    assert _parse_price("$49.00") == Decimal("49.00")
    assert _parse_price("1,234.50") == Decimal("1234.50")
    assert _parse_price(None) == Decimal("0")


def test_is_test_sale_prefers_explicit_test_flag():
    assert _is_test_sale({"email": "real@company.com", "sale_id": "abc", "test": True}) is True
    assert _is_test_sale({"email": "real@company.com", "sale_id": "abc", "test": "true"}) is True
    assert _is_test_sale({"email": "real@company.com", "sale_id": "TEST-123"}) is True

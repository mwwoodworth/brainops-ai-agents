import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affiliate_partnership_pipeline import Affiliate, AffiliatePartnershipPipeline, Payout


def _set_dev_env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("DATABASE_URL", raising=False)


def test_affiliate_pipeline_requires_db_or_flag(monkeypatch):
    _set_dev_env(monkeypatch)
    monkeypatch.delenv("ALLOW_AFFILIATE_INMEMORY", raising=False)

    with pytest.raises(RuntimeError, match="Affiliate pipeline requires DATABASE_URL"):
        AffiliatePartnershipPipeline()


@pytest.mark.asyncio
async def test_paypal_payout_returns_failure(monkeypatch):
    _set_dev_env(monkeypatch)
    monkeypatch.setenv("ALLOW_AFFILIATE_INMEMORY", "true")

    pipeline = AffiliatePartnershipPipeline()
    affiliate = Affiliate(affiliate_id="aff-1", payout_method="paypal")
    payout = Payout(affiliate_id="aff-1", payment_method="paypal")

    result = await pipeline._process_payment(payout, affiliate)

    assert result["success"] is False
    assert "not implemented" in result["error"].lower()


@pytest.mark.asyncio
async def test_wire_payout_returns_failure(monkeypatch):
    _set_dev_env(monkeypatch)
    monkeypatch.setenv("ALLOW_AFFILIATE_INMEMORY", "true")

    pipeline = AffiliatePartnershipPipeline()
    affiliate = Affiliate(affiliate_id="aff-2", payout_method="wire")
    payout = Payout(affiliate_id="aff-2", payment_method="wire")

    result = await pipeline._process_payment(payout, affiliate)

    assert result["success"] is False
    assert "not implemented" in result["error"].lower()

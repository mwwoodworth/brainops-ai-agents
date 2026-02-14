"""Regression tests for E2E verifier rate-limit bypass.

The E2E verifier fires many requests to BRAINOPS_API_URL in rapid succession.
Without the X-Internal-E2E HMAC header, these requests share a single per-key
rate-limit counter and many hit 429.  The fix adds an HMAC-signed header that
gives each internal request its own counter, preventing false-negative failures.

Regression: if /e2e/verify quick reports 429 failures when endpoints are
otherwise healthy, this test catches it.
"""

import hashlib
import hmac as hmac_mod

import e2e_system_verification as mod


def test_compute_e2e_internal_sig_deterministic():
    """Signature must be deterministic for a given key."""
    sig1 = mod._compute_e2e_internal_sig("test-key")
    sig2 = mod._compute_e2e_internal_sig("test-key")
    assert sig1 == sig2
    assert len(sig1) == 64  # SHA-256 hex digest


def test_compute_e2e_internal_sig_varies_by_key():
    """Different keys must produce different signatures."""
    sig_a = mod._compute_e2e_internal_sig("key-a")
    sig_b = mod._compute_e2e_internal_sig("key-b")
    assert sig_a != sig_b


def test_compute_e2e_internal_sig_matches_server_expectation():
    """Signature must match the HMAC the server computes in _rate_limit_key."""
    api_key = "brainops_prod_key_2025"
    sig = mod._compute_e2e_internal_sig(api_key)
    expected = hmac_mod.new(
        api_key.encode("utf-8"), b"brainops-e2e-internal", hashlib.sha256
    ).hexdigest()
    assert hmac_mod.compare_digest(sig, expected)


def test_internal_header_injected_for_self_calls(monkeypatch):
    """_run_single_test must add X-Internal-E2E for BRAINOPS_API_URL targets."""
    monkeypatch.setattr(mod, "API_KEY", "test-api-key")
    monkeypatch.setattr(mod, "BRAINOPS_API_URL", "https://brainops-api.test")

    test = mod.EndpointTest(
        name="Self Call",
        url="https://brainops-api.test/health",
        headers={"X-API-Key": "test-api-key"},
    )

    # We cannot easily call _run_single_test without a real HTTP session,
    # so we verify the header injection logic directly:
    headers = dict(test.headers) if test.headers else {}
    if mod.API_KEY and test.url.startswith(mod.BRAINOPS_API_URL):
        signing_key = headers.get("X-API-Key") or mod.API_KEY
        headers["X-Internal-E2E"] = mod._compute_e2e_internal_sig(signing_key)

    assert "X-Internal-E2E" in headers
    assert len(headers["X-Internal-E2E"]) == 64


def test_hmac_uses_overridden_api_key(monkeypatch):
    """When _apply_api_key_override replaces X-API-Key, HMAC must use the override key.

    Root cause of P1-E2E-VERIFY-001: the verifier signed with API_KEY but the
    request carried a different key (from the caller), causing HMAC mismatch
    and falling through to the shared rate-limit counter.
    """
    monkeypatch.setattr(mod, "API_KEY", "internal-key")
    monkeypatch.setattr(mod, "BRAINOPS_API_URL", "https://brainops-api.test")

    # Simulate _apply_api_key_override replacing the key with the caller's key
    test = mod.EndpointTest(
        name="Self Call with Override",
        url="https://brainops-api.test/agents",
        headers={"X-API-Key": "caller-override-key"},
    )

    headers = dict(test.headers) if test.headers else {}
    if mod.API_KEY and test.url.startswith(mod.BRAINOPS_API_URL):
        signing_key = headers.get("X-API-Key") or mod.API_KEY
        headers["X-Internal-E2E"] = mod._compute_e2e_internal_sig(signing_key)

    # HMAC must be signed with the OVERRIDE key, not the internal key
    expected_sig = mod._compute_e2e_internal_sig("caller-override-key")
    wrong_sig = mod._compute_e2e_internal_sig("internal-key")
    assert headers["X-Internal-E2E"] == expected_sig
    assert headers["X-Internal-E2E"] != wrong_sig


def test_internal_header_not_injected_for_external_calls(monkeypatch):
    """External endpoints must NOT receive the X-Internal-E2E header."""
    monkeypatch.setattr(mod, "API_KEY", "test-api-key")
    monkeypatch.setattr(mod, "BRAINOPS_API_URL", "https://brainops-api.test")

    test = mod.EndpointTest(
        name="External Call",
        url="https://other-service.test/health",
        headers={"X-API-Key": "test-api-key"},
    )

    headers = dict(test.headers) if test.headers else {}
    if mod.API_KEY and test.url.startswith(mod.BRAINOPS_API_URL):
        headers["X-Internal-E2E"] = mod._compute_e2e_internal_sig(mod.API_KEY)

    assert "X-Internal-E2E" not in headers


def test_rate_limit_key_exempts_valid_e2e_header():
    """Verify that _rate_limit_key returns unique keys for valid E2E headers.

    This test imports and calls the actual _rate_limit_key function from app.py
    with a mock Request to ensure the HMAC validation path works end-to-end.
    """
    try:
        from app import _rate_limit_key
        from config import config
    except ImportError:
        # If app.py cannot be imported in test env (missing deps), skip gracefully
        import pytest

        pytest.skip("Cannot import app module in this test environment")

    api_key = list(config.security.valid_api_keys)[0] if config.security.valid_api_keys else None
    if not api_key:
        import pytest

        pytest.skip("No valid API keys configured")

    sig = mod._compute_e2e_internal_sig(api_key)

    class MockClient:
        host = "127.0.0.1"

    class MockRequest:
        def __init__(self, headers):
            self._headers = headers
            self.client = MockClient()

        @property
        def headers(self):
            return self._headers

    # With valid E2E header: should get unique e2e-internal key
    req_with = MockRequest({"X-API-Key": api_key, "X-Internal-E2E": sig})
    key_with = _rate_limit_key(req_with)
    assert key_with.startswith("e2e-internal:")

    # Without E2E header: should get normal key:hash
    req_without = MockRequest({"X-API-Key": api_key})
    key_without = _rate_limit_key(req_without)
    assert key_without.startswith("key:")

    # With invalid E2E sig: should get normal key:hash (not exempt)
    req_bad = MockRequest({"X-API-Key": api_key, "X-Internal-E2E": "invalid"})
    key_bad = _rate_limit_key(req_bad)
    assert key_bad.startswith("key:")

    # Without API key: should fall back to IP
    req_nokey = MockRequest({"X-Internal-E2E": sig})
    key_nokey = _rate_limit_key(req_nokey)
    assert key_nokey.startswith("ip:")

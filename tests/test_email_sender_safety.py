#!/usr/bin/env python3
"""
Safety-rail tests for outbound email behavior.
"""

import importlib
import sys


def _load_email_sender_module(
    monkeypatch,
    *,
    mode: str,
    allowlist: str = "",
    allowlist_domains: str = "",
    require_allowlist_in_live: bool | None = None,
):
    for key in (
        "OUTBOUND_EMAIL_MODE",
        "OUTBOUND_EMAIL_ALLOWLIST",
        "OUTBOUND_EMAIL_ALLOWLIST_DOMAINS",
        "OUTBOUND_EMAIL_REQUIRE_ALLOWLIST_IN_LIVE",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("OUTBOUND_EMAIL_MODE", mode)
    if allowlist:
        monkeypatch.setenv("OUTBOUND_EMAIL_ALLOWLIST", allowlist)
    if allowlist_domains:
        monkeypatch.setenv("OUTBOUND_EMAIL_ALLOWLIST_DOMAINS", allowlist_domains)
    if require_allowlist_in_live is not None:
        monkeypatch.setenv(
            "OUTBOUND_EMAIL_REQUIRE_ALLOWLIST_IN_LIVE",
            "true" if require_allowlist_in_live else "false",
        )

    sys.modules.pop("email_sender", None)
    return importlib.import_module("email_sender")


def test_disabled_mode_fails_closed(monkeypatch):
    module = _load_email_sender_module(monkeypatch, mode="disabled")
    assert module._outbound_block_reason("owner@weathercraft.net", None) == "outbound_disabled"
    assert module._outbound_delivery_mode("owner@weathercraft.net") == "restricted"


def test_allowlist_mode_allows_explicit_recipient(monkeypatch):
    module = _load_email_sender_module(
        monkeypatch,
        mode="allowlist",
        allowlist="owner@weathercraft.net",
    )
    assert module._outbound_block_reason("owner@weathercraft.net", None) is None
    assert module._outbound_delivery_mode("owner@weathercraft.net") == "allowlisted"


def test_allowlist_mode_blocks_non_allowlisted_recipient(monkeypatch):
    module = _load_email_sender_module(
        monkeypatch,
        mode="allowlist",
        allowlist="owner@weathercraft.net",
        allowlist_domains="weathercraft.net",
    )
    assert module._outbound_block_reason("outside@example.com", None) == "not_allowlisted"


def test_live_mode_requires_allowlist_by_default(monkeypatch):
    module = _load_email_sender_module(
        monkeypatch,
        mode="live",
        allowlist="owner@weathercraft.net",
    )
    assert module._outbound_block_reason("outside@example.org", None) == "not_allowlisted"
    assert module._outbound_delivery_mode("outside@example.org") == "restricted"


def test_live_mode_can_be_unrestricted_with_explicit_override(monkeypatch):
    module = _load_email_sender_module(
        monkeypatch,
        mode="live",
        require_allowlist_in_live=False,
    )
    assert module._outbound_block_reason("lead@example.org", None) is None
    assert module._outbound_delivery_mode("lead@example.org") == "live"


def test_unknown_mode_fails_closed(monkeypatch):
    module = _load_email_sender_module(monkeypatch, mode="unexpected_mode")
    assert module._outbound_block_reason("lead@example.org", None) == "outbound_disabled"

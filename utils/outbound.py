from __future__ import annotations

import os
import re
from typing import Any

_CSV_SPLIT_RE = re.compile(r"\s*,\s*")


def _parse_csv_env(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in _CSV_SPLIT_RE.split(value) if item.strip()}


def _normalize_email(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower()


_OUTBOUND_EMAIL_ALLOWLIST_RECIPIENTS = _parse_csv_env(os.getenv("OUTBOUND_EMAIL_ALLOWLIST", ""))
_OUTBOUND_EMAIL_ALLOWLIST_DOMAINS = _parse_csv_env(os.getenv("OUTBOUND_EMAIL_ALLOWLIST_DOMAINS", ""))


def is_allowlisted_email(recipient: str | None) -> bool:
    lowered = _normalize_email(recipient)
    if not lowered:
        return False

    if lowered in _OUTBOUND_EMAIL_ALLOWLIST_RECIPIENTS:
        return True

    if "@" not in lowered:
        return False

    domain = lowered.split("@", 1)[1]
    if domain in _OUTBOUND_EMAIL_ALLOWLIST_DOMAINS:
        return True

    return any(domain.endswith(f".{allowed}") for allowed in _OUTBOUND_EMAIL_ALLOWLIST_DOMAINS)


def email_block_reason(recipient: str | None, metadata: dict[str, Any] | None = None) -> str | None:
    """
    Centralized outbound email safety rail.

    Modes:
      - disabled (default): nothing sends.
      - allowlist: only allowlisted recipients/domains send.
      - live: sends.

    This must be checked BEFORE attempting provider delivery.
    """
    mode = os.getenv("OUTBOUND_EMAIL_MODE", "disabled").strip().lower()

    if mode == "live":
        return None

    if is_allowlisted_email(recipient):
        return None

    if mode == "allowlist":
        return "not_allowlisted"

    # disabled (or unknown): fail closed
    if metadata and metadata.get("allow_outbound") is True:
        # Explicit override is ignored unless allowlisted (keeps strict safety default).
        return "outbound_disabled"
    return "outbound_disabled"


def _normalize_phone(value: str | None) -> str:
    if not value:
        return ""
    stripped = value.strip()
    if not stripped:
        return ""

    # Keep leading + for E.164; remove spaces, dashes, parens, dots.
    plus = "+" if stripped.startswith("+") else ""
    digits = re.sub(r"[^\d]", "", stripped)
    return f"{plus}{digits}" if digits else ""


_OUTBOUND_SMS_ALLOWLIST_NUMBERS = {
    _normalize_phone(v) for v in _parse_csv_env(os.getenv("OUTBOUND_SMS_ALLOWLIST", "")) if _normalize_phone(v)
}
_OUTBOUND_SMS_ALLOWLIST_PREFIXES = {
    _normalize_phone(v)
    for v in _parse_csv_env(os.getenv("OUTBOUND_SMS_ALLOWLIST_PREFIXES", ""))
    if _normalize_phone(v)
}


def is_allowlisted_phone(number: str | None) -> bool:
    normalized = _normalize_phone(number)
    if not normalized:
        return False

    if normalized in _OUTBOUND_SMS_ALLOWLIST_NUMBERS:
        return True

    return any(normalized.startswith(prefix) for prefix in _OUTBOUND_SMS_ALLOWLIST_PREFIXES)


def sms_block_reason(number: str | None, metadata: dict[str, Any] | None = None) -> str | None:
    """
    Centralized outbound SMS safety rail.

    Modes:
      - disabled (default): nothing sends.
      - allowlist: only allowlisted numbers/prefixes send.
      - live: sends.
    """
    mode = os.getenv("OUTBOUND_SMS_MODE", "disabled").strip().lower()

    if mode == "live":
        return None

    if is_allowlisted_phone(number):
        return None

    if mode == "allowlist":
        return "not_allowlisted"

    if metadata and metadata.get("allow_outbound") is True:
        return "outbound_disabled"
    return "outbound_disabled"


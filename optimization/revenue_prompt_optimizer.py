from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from optimization.prompt_compiler import RevenuePromptCompiler
from optimization.revenue_training_data import build_revenue_training_samples

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_subject_body(text: str) -> tuple[str | None, str | None]:
    """
    Parse a combined email in the format:
      Subject: ...

      <body>
    Returns (subject, body). Missing parts return None.
    """
    if not isinstance(text, str):
        return None, None
    raw = text.replace("\r", "").strip()
    if not raw:
        return None, None

    lines = raw.split("\n")
    first = lines[0].strip()
    subject = None
    body_start_idx = 0

    if first.lower().startswith("subject:"):
        subject = first.split(":", 1)[1].strip()
        body_start_idx = 1
        # Skip blank lines after subject.
        while body_start_idx < len(lines) and not lines[body_start_idx].strip():
            body_start_idx += 1

    body = "\n".join(lines[body_start_idx:]).strip() if body_start_idx < len(lines) else ""
    return (subject or None), (body or None if body else None)


@dataclass
class OptimizeResult:
    subject: str
    body: str
    used_optimizer: bool
    compiler_enabled: bool
    compiled: bool
    compiled_at: Optional[str]
    compile_stats: dict[str, Any]


class RevenuePromptOptimizer:
    """
    Phase 2: Revenue reinforcement.

    Uses REAL conversion/revenue signals to compile a DSPy program that rewrites emails.
    Safe defaults:
    - Disabled unless ENABLE_DSPY_OPTIMIZATION=true
    - Compilation is rate-limited
    - If anything fails, returns the original draft unchanged
    """

    def __init__(self) -> None:
        self._compiler = RevenuePromptCompiler()
        self._compiled_at: Optional[datetime] = None
        self._compile_lock = asyncio.Lock()
        self._last_compile_attempt_ts: float = 0.0
        self._last_compile_stats: dict[str, Any] = {}

    def enabled(self) -> bool:
        if not _env_bool("ENABLE_DSPY_OPTIMIZATION", default=False):
            return False
        return self._compiler.enabled()

    def status(self) -> dict[str, Any]:
        """Return a small, import-safe status snapshot for observability endpoints."""
        return {
            "enabled": self.enabled(),
            "compiler_available": self._compiler.enabled(),
            "compiled": self._compiler.compiled(),
            "compiled_at": self._compiled_at.isoformat() if self._compiled_at else None,
            "model": self._compiler.model,
            "auto_recompile": _env_bool("DSPY_REVENUE_AUTO_RECOMPILE", default=False),
            "compile_min_interval_seconds": float(os.getenv("DSPY_REVENUE_COMPILE_MIN_INTERVAL_SECONDS", "3600")),
            "training_lookback_days": _coerce_int(os.getenv("DSPY_REVENUE_TRAINING_LOOKBACK_DAYS"), 180),
            "training_email_limit": _coerce_int(os.getenv("DSPY_REVENUE_TRAINING_EMAIL_LIMIT"), 300),
            "training_min_samples": _coerce_int(os.getenv("DSPY_REVENUE_MIN_TRAIN_SAMPLES"), 25),
            "reward_cap_amount": float(os.getenv("DSPY_REWARD_CAP_AMOUNT") or 5000.0),
            "training_source_allowlist": os.getenv("DSPY_REVENUE_TRAINING_SOURCE_ALLOWLIST") or "",
            "last_compile_stats": self._last_compile_stats,
        }

    async def ensure_compiled(self, *, pool: Any | None = None, force: bool = False) -> dict[str, Any]:
        if not self.enabled():
            return {"status": "skipped", "reason": "dspy_disabled_or_unavailable"}

        now = time.time()
        min_interval = float(os.getenv("DSPY_REVENUE_COMPILE_MIN_INTERVAL_SECONDS", "3600"))
        if not force:
            if self._compiled_at and (now - self._compiled_at.timestamp()) < min_interval:
                return {
                    "status": "skipped",
                    "reason": "compile_recent",
                    "compiled_at": self._compiled_at.isoformat(),
                }
            if (now - self._last_compile_attempt_ts) < min_interval:
                return {"status": "skipped", "reason": "compile_attempt_rate_limited"}

        async with self._compile_lock:
            now = time.time()
            if not force and self._compiled_at and (now - self._compiled_at.timestamp()) < min_interval:
                return {
                    "status": "skipped",
                    "reason": "compile_recent",
                    "compiled_at": self._compiled_at.isoformat(),
                }

            self._last_compile_attempt_ts = now

            if pool is None:
                from database.async_connection import get_pool

                pool = get_pool()

            lookback_days = _coerce_int(os.getenv("DSPY_REVENUE_TRAINING_LOOKBACK_DAYS"), 180)
            limit = _coerce_int(os.getenv("DSPY_REVENUE_TRAINING_EMAIL_LIMIT"), 300)
            min_samples = _coerce_int(os.getenv("DSPY_REVENUE_MIN_TRAIN_SAMPLES"), 25)

            try:
                training_samples, stats = await build_revenue_training_samples(
                    pool=pool,
                    lookback_days=lookback_days,
                    limit=limit,
                )
            except Exception as exc:
                logger.warning("DSPy training sample build failed: %s", exc)
                return {"status": "error", "error": str(exc)}

            self._last_compile_stats = stats or {}

            if len(training_samples) < min_samples:
                return {
                    "status": "skipped",
                    "reason": "insufficient_training_samples",
                    "samples": len(training_samples),
                    "min_samples": min_samples,
                    "stats": stats,
                }

            compiled = self._compiler.compile(training_samples)
            if compiled is None:
                return {"status": "error", "error": "dspy_compile_failed", "stats": stats}

            self._compiled_at = datetime.now(timezone.utc)
            return {
                "status": "compiled",
                "compiled_at": self._compiled_at.isoformat(),
                "samples": len(training_samples),
                "stats": stats,
            }

    async def optimize(
        self,
        *,
        leads: str,
        revenue_metrics: str,
        subject: str,
        body: str,
        pool: Any | None = None,
    ) -> OptimizeResult:
        compiler_enabled = self._compiler.enabled()
        if not self.enabled():
            return OptimizeResult(
                subject=subject,
                body=body,
                used_optimizer=False,
                compiler_enabled=compiler_enabled,
                compiled=self._compiler.compiled(),
                compiled_at=self._compiled_at.isoformat() if self._compiled_at else None,
                compile_stats=self._last_compile_stats,
            )

        # Compile opportunistically if not compiled yet (rate-limited).
        if not self._compiler.compiled():
            await self.ensure_compiled(pool=pool, force=False)

        draft = "\n".join([f"Subject: {subject}".strip(), "", body or ""]).strip()
        optimized_text = self._compiler.optimize_email(
            leads=leads,
            revenue_metrics=revenue_metrics,
            draft_email=draft,
        )

        opt_subject, opt_body = _parse_subject_body(str(optimized_text or ""))
        final_subject = opt_subject or subject
        final_body = opt_body or body

        # Guardrails: keep outputs sane and non-empty.
        if not final_body.strip():
            final_subject = subject
            final_body = body

        if len(final_subject) > 180:
            final_subject = final_subject[:180].rstrip()

        max_body = _coerce_int(os.getenv("DSPY_REVENUE_MAX_BODY_CHARS"), 12000)
        if max_body > 0 and len(final_body) > max_body:
            final_body = final_body[:max_body].rstrip()

        return OptimizeResult(
            subject=final_subject,
            body=final_body,
            used_optimizer=final_subject != subject or final_body != body,
            compiler_enabled=compiler_enabled,
            compiled=self._compiler.compiled(),
            compiled_at=self._compiled_at.isoformat() if self._compiled_at else None,
            compile_stats=self._last_compile_stats,
        )


_optimizer: Optional[RevenuePromptOptimizer] = None


def get_revenue_prompt_optimizer() -> RevenuePromptOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = RevenuePromptOptimizer()
    return _optimizer

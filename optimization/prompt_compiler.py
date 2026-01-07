"""
DSPy Prompt Compiler
====================
Provides a *safe* DSPy integration that never breaks service startup.

`agent_executor.py` imports `RevenuePromptCompiler` unconditionally, so this
module must be import-safe even when DSPy isn't installed or isn't configured.

Current usage:
  compiler = RevenuePromptCompiler()
  compiled = compiler.compile(training_samples)

The compiler returns a compiled DSPy program when possible, otherwise `None`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _try_import_dspy():
    try:
        import dspy  # type: ignore

        return dspy
    except Exception as exc:  # ImportError + runtime errors
        logger.info("DSPy unavailable: %s", exc)
        return None


class RevenuePromptCompiler:
    """
    Revenue prompt compiler using DSPy.

    Notes:
    - Import-safe: all DSPy classes are defined lazily.
    - Metric-driven compilation requires labeled outcomes; today we use a
      lightweight heuristic and treat compilation as best-effort.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._dspy = _try_import_dspy()
        self._lm_configured = False
        self._compiled_program: Any | None = None
        self.model = (
            model
            or os.getenv("DSPY_MODEL")
            or os.getenv("DSPY_OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )

    def enabled(self) -> bool:
        return self._dspy is not None

    def compile(self, training_samples: list[dict[str, Any]]) -> Any | None:
        """
        Compile (optimize) a revenue email program from training samples.

        Expected training sample shape (best-effort):
          {
            "leads": "<json or text>",
            "revenue_metrics": "<json or text>",
            "email": "<draft email>",
            "revenue_generated": <number>,
          }
        """
        if not self._dspy:
            return None

        self._configure_lm()
        if not self._dspy:
            return None

        dspy = self._dspy

        try:
            signature = self._build_signature(dspy)
            program = dspy.ChainOfThought(signature)

            trainset = self._to_trainset(dspy, training_samples)
            if not trainset:
                logger.info("DSPy compile skipped: no usable training samples")
                self._compiled_program = None
                return None

            try:
                from dspy.teleprompt import BootstrapFewShot  # type: ignore

                teleprompter = BootstrapFewShot(metric=self._revenue_metric)
                self._compiled_program = teleprompter.compile(program, trainset=trainset)
            except Exception as exc:
                # Teleprompt APIs can change between DSPy versions; fall back to raw program.
                logger.warning("DSPy teleprompt compile failed; using uncompiled program: %s", exc)
                self._compiled_program = program

            return self._compiled_program
        except Exception as exc:
            logger.warning("DSPy compilation failed: %s", exc, exc_info=True)
            self._compiled_program = None
            return None

    def optimize_email(self, *, leads: str, revenue_metrics: str, draft_email: str) -> str:
        """Run the compiled program (if available) and return an optimized email."""
        if not self._compiled_program:
            return draft_email

        try:
            result = self._compiled_program(leads=leads, revenue_metrics=revenue_metrics, email=draft_email)
            optimized = getattr(result, "optimized_email", None)
            if isinstance(optimized, str) and optimized.strip():
                return optimized
        except Exception:
            logger.debug("DSPy optimize_email failed; returning draft", exc_info=True)

        return draft_email

    def _configure_lm(self) -> None:
        if not self._dspy or self._lm_configured:
            return

        dspy = self._dspy
        try:
            if not os.getenv("OPENAI_API_KEY"):
                logger.info("DSPy disabled: OPENAI_API_KEY not configured")
                self._dspy = None
                return

            # DSPy defaults to OpenAI backend when configured with dspy.OpenAI.
            # Model is configurable so prod can pin it without code changes.
            dspy.settings.configure(lm=dspy.OpenAI(model=self.model))  # type: ignore[attr-defined]
            self._lm_configured = True
        except Exception as exc:
            logger.warning("DSPy LM configuration failed: %s", exc)
            self._dspy = None

    @staticmethod
    def _build_signature(dspy: Any) -> Any:
        class RevenueOptimization(dspy.Signature):  # type: ignore[attr-defined]
            leads = dspy.InputField(desc="Lead context and history")  # type: ignore[attr-defined]
            revenue_metrics = dspy.InputField(desc="Revenue signals/metrics")  # type: ignore[attr-defined]
            email = dspy.InputField(desc="Draft outreach email")  # type: ignore[attr-defined]
            optimized_email = dspy.OutputField(desc="Optimized outreach email")  # type: ignore[attr-defined]

        return RevenueOptimization

    @staticmethod
    def _to_trainset(dspy: Any, samples: list[dict[str, Any]]) -> list[Any]:
        trainset: list[Any] = []
        for sample in samples or []:
            leads = str(sample.get("leads") or "").strip()
            revenue_metrics = str(sample.get("revenue_metrics") or "").strip()
            email = str(sample.get("email") or "").strip()
            if not (leads and email):
                continue

            # We generally don't have ground-truth optimized emails yet.
            # Use the draft as a placeholder label so DSPy can still bootstrap.
            example = dspy.Example(  # type: ignore[attr-defined]
                leads=leads,
                revenue_metrics=revenue_metrics,
                email=email,
                optimized_email=email,
                revenue_generated=float(sample.get("revenue_generated") or 0),
            ).with_inputs("leads", "revenue_metrics", "email")
            trainset.append(example)
        return trainset

    @staticmethod
    def _revenue_metric(example: Any, pred: Any, trace: Any = None) -> float:
        """
        Best-effort metric placeholder.

        When real outcomes are available, replace with a metric that correlates
        generated email with conversions/revenue (e.g., from `revenue_leads`).
        """
        try:
            value = float(getattr(example, "revenue_generated", 0) or 0)
            # Normalize to [0, 1] with a soft cap.
            return max(0.0, min(1.0, value / 1000.0))
        except Exception:
            return 0.0

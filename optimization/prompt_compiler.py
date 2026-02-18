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

import difflib
import logging
import os
import re
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

    def compiled(self) -> bool:
        return bool(self._compiled_program)

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
            optimized_email = str(sample.get("optimized_email") or email or "").strip()
            if not (leads and email):
                continue

            example = dspy.Example(  # type: ignore[attr-defined]
                leads=leads,
                revenue_metrics=revenue_metrics,
                email=email,
                optimized_email=optimized_email,
                revenue_generated=float(sample.get("revenue_generated") or 0),
                reward_score=float(sample.get("reward_score") or 0),
                outcome=str(sample.get("outcome") or ""),
            ).with_inputs("leads", "revenue_metrics", "email")
            trainset.append(example)
        return trainset

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
        # Remove non-content wrappers that can vary without meaning.
        cleaned = cleaned.replace("\r", "")
        return cleaned

    @classmethod
    def _text_similarity(cls, a: str, b: str) -> float:
        a_norm = cls._normalize_text(a)
        b_norm = cls._normalize_text(b)
        if not a_norm or not b_norm:
            return 0.0
        # Token overlap gives a stable signal even when formatting differs.
        a_tokens = set(re.findall(r"[a-z0-9']{2,}", a_norm))
        b_tokens = set(re.findall(r"[a-z0-9']{2,}", b_norm))
        if a_tokens and b_tokens:
            overlap = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
        else:
            overlap = 0.0
        ratio = difflib.SequenceMatcher(a=a_norm, b=b_norm).ratio()
        return max(0.0, min(1.0, 0.6 * ratio + 0.4 * overlap))

    @classmethod
    def _revenue_metric(cls, example: Any, pred: Any, trace: Any = None) -> float:
        """
        Conversion-weighted imitation metric.

        DSPy teleprompting needs a metric that depends on the prediction. We don't
        have counterfactual "would this email have closed the deal?" data yet, so we
        use a pragmatic proxy:
        - Similarity(pred.optimized_email, example.optimized_email)
        - Weighted by example.reward_score (0..1) and revenue_generated
        """
        try:
            target = str(getattr(example, "optimized_email", "") or "")
            predicted = str(getattr(pred, "optimized_email", "") or "")
            similarity = cls._text_similarity(target, predicted)

            reward = float(getattr(example, "reward_score", 0) or 0)
            if reward <= 0:
                # Neutral sample: only enforce non-degenerate output.
                return max(0.0, min(0.2, similarity))

            # Ensure reward is in [0, 1].
            reward = max(0.0, min(1.0, reward))

            # Soft-scale revenue so large deals matter more without dominating.
            value = float(getattr(example, "revenue_generated", 0) or 0)
            cap = float(os.getenv("DSPY_REWARD_CAP_AMOUNT") or 5000.0)
            cap = max(1.0, cap)
            value_scale = max(0.0, min(1.0, (value / cap)))

            score = similarity * (0.65 * reward + 0.35 * value_scale)
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

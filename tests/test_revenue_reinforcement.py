import os
import sys

import pytest

# Add parent directory to path for imports (repo uses flat module layout).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.prompt_compiler import RevenuePromptCompiler
from optimization.revenue_prompt_compile_queue import enqueue_revenue_prompt_compile_task
from optimization.revenue_prompt_optimizer import _parse_subject_body
from optimization.revenue_training_data import _state_reward


def test_parse_subject_body_subject_and_body():
    subject, body = _parse_subject_body("Subject: Hello there\n\nLine 1\nLine 2")
    assert subject == "Hello there"
    assert body == "Line 1\nLine 2"


def test_parse_subject_body_no_subject_header():
    subject, body = _parse_subject_body("Hello\n\nWorld")
    assert subject is None
    assert body == "Hello\n\nWorld"


def test_state_reward_known_and_unknown():
    assert _state_reward("paid") == 1.0
    assert _state_reward("won") > 0.0  # legacy stages supported
    assert _state_reward("proposal_sent") > _state_reward("contacted")
    assert _state_reward("totally_unknown_state") == 0.0


def test_revenue_metric_depends_on_prediction(monkeypatch):
    class Example:
        optimized_email = "Subject: A\n\nHello"
        reward_score = 1.0
        revenue_generated = 5000.0

    class Pred:
        def __init__(self, text: str):
            self.optimized_email = text

    compiler = RevenuePromptCompiler()
    same = compiler._revenue_metric(Example(), Pred("Subject: A\n\nHello"))
    different = compiler._revenue_metric(Example(), Pred("Subject: Z\n\nCompletely different"))

    assert 0.0 <= same <= 1.0
    assert 0.0 <= different <= 1.0
    assert same > different

    # Neutral samples should never exceed the low cap.
    class NeutralExample:
        optimized_email = "Subject: A\n\nHello"
        reward_score = 0.0
        revenue_generated = 0.0

    neutral = compiler._revenue_metric(NeutralExample(), Pred("Subject: A\n\nHello"))
    assert 0.0 <= neutral <= 0.2


@pytest.mark.asyncio
async def test_enqueue_compile_task_respects_env_gates(monkeypatch):
    class Pool:
        async def fetchrow(self, *_args, **_kwargs):
            raise AssertionError("pool should not be queried when disabled")

    monkeypatch.delenv("DSPY_REVENUE_AUTO_RECOMPILE", raising=False)
    monkeypatch.delenv("ENABLE_DSPY_OPTIMIZATION", raising=False)

    task_id = await enqueue_revenue_prompt_compile_task(
        pool=Pool(),
        tenant_id=None,
        lead_id="123",
        reason="test",
        priority=80,
        force=True,
    )
    assert task_id is None


@pytest.mark.asyncio
async def test_enqueue_compile_task_dedupe_and_insert(monkeypatch):
    monkeypatch.setenv("DSPY_REVENUE_AUTO_RECOMPILE", "true")
    monkeypatch.setenv("ENABLE_DSPY_OPTIMIZATION", "true")
    monkeypatch.setenv("TENANT_ID", "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457")
    monkeypatch.setenv("DSPY_REVENUE_COMPILE_TASK_DEDUPE_SECONDS", "600")

    class Pool:
        def __init__(self):
            self.calls = 0

        async def fetchrow(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                # Dedupe query: no existing task
                return None
            return {"id": "00000000-0000-0000-0000-000000000000"}

    pool = Pool()
    task_id = await enqueue_revenue_prompt_compile_task(
        pool=pool,
        tenant_id=None,
        lead_id="123",
        reason="test",
        priority=80,
        force=True,
    )
    assert task_id == "00000000-0000-0000-0000-000000000000"

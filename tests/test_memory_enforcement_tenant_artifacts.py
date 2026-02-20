from __future__ import annotations

import pytest

import memory_enforcement as me


class _FakePool:
    def __init__(self) -> None:
        self.last_query = ""
        self.last_args = ()

    async def fetchrow(self, query, *args):
        self.last_query = query
        self.last_args = args
        if "INSERT INTO unified_ai_memory" in query:
            return {"id": "00000000-0000-0000-0000-000000000111"}
        return {"id": "artifact-1"}


@pytest.mark.asyncio
async def test_store_proof_artifact_writes_tenant_id() -> None:
    pool = _FakePool()
    engine = me.MemoryEnforcementEngine(pool=pool)

    proof = me.VerificationProof(
        artifact_type="log",
        artifact_url="https://example.com/artifact",
        evidence_level=me.EvidenceLevel.E1_RECORDED,
        created_by="test",
    )

    artifact_id = await engine._store_proof_artifact(
        "00000000-0000-0000-0000-000000000001",
        proof,
        tenant_id="51e728c5-94e8-4ae0-8a0a-6a08d1fb3457",
    )

    assert artifact_id == "artifact-1"
    assert "tenant_id" in pool.last_query
    assert pool.last_args[-1] == "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"


@pytest.mark.asyncio
async def test_store_memory_keeps_primary_write_when_proof_storage_fails(monkeypatch):
    pool = _FakePool()
    engine = me.MemoryEnforcementEngine(pool=pool)

    async def _fake_generate_embedding(_content: str):
        return [0.1, 0.2]

    import api.memory as api_memory

    monkeypatch.setattr(api_memory, "generate_embedding", _fake_generate_embedding)

    async def _fail_proof(*_args, **_kwargs):
        raise RuntimeError("proof write denied")

    monkeypatch.setattr(engine, "_store_proof_artifact", _fail_proof)

    contract = me.MemoryContract(
        type=me.MemoryObjectType.DECISION,
        title="Decision",
        content={"summary": "ok"},
        source="unit-test",
    )
    proof = me.VerificationProof(
        artifact_type="log",
        evidence_level=me.EvidenceLevel.E1_RECORDED,
    )

    memory_id = await engine._store_memory(
        contract=contract,
        agent_id="tester",
        tenant_id="51e728c5-94e8-4ae0-8a0a-6a08d1fb3457",
        proof=proof,
    )

    assert memory_id == "00000000-0000-0000-0000-000000000111"

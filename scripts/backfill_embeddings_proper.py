import asyncio
import json
import os
import sys
import time
from typing import Optional, Tuple

from dotenv import load_dotenv

SECURE_ENV_PATH = "/home/matt-woodworth/dev/_secure/BrainOps.env"


# Ensure repo root on path
sys.path.append(os.getcwd())


def _load_env() -> None:
    if os.path.exists(SECURE_ENV_PATH):
        print(f"🔐 Loading Secure Env: {SECURE_ENV_PATH}")
        load_dotenv(SECURE_ENV_PATH, override=True)


# Load env before importing config (config reads env on import)
_load_env()

from config import config as app_config
from database.async_connection import init_pool, close_pool, get_pool, PoolConfig


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(content)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("429", "rate limit", "quota", "too many requests", "service unavailable", "503"))


def _truncate(text: str, limit: int = 30000) -> str:
    return text[:limit] if len(text) > limit else text


def _provider_order() -> list[str]:
    order = os.getenv("EMBEDDING_PROVIDER_ORDER", "openai,gemini")
    return [p.strip().lower() for p in order.split(",") if p.strip()]


async def _embed_openai(text: str, max_retries: int = 5, base_backoff: float = 1.0) -> Optional[list[float]]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        import openai  # type: ignore
    except Exception:
        return None

    for attempt in range(max_retries):
        try:
            client = openai.OpenAI(api_key=key)
            response = client.embeddings.create(
                input=_truncate(text),
                model="text-embedding-3-small",
            )
            return response.data[0].embedding
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < max_retries - 1:
                sleep_for = base_backoff * (2 ** attempt)
                print(f"⚠️ OpenAI rate-limited; retrying in {sleep_for:.1f}s")
                await asyncio.sleep(sleep_for)
                continue
            print(f"⚠️ OpenAI embedding failed: {exc}")
            return None
    return None


async def _embed_gemini(text: str, max_retries: int = 5, base_backoff: float = 1.0) -> Optional[list[float]]:
    key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai  # type: ignore
    except Exception as exc:
        print(f"⚠️ Gemini client unavailable: {exc}")
        return None

    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=key)
            result = client.models.embed_content(
                model="text-embedding-004",
                contents=_truncate(text),
            )
            if result and result.embeddings:
                embedding = list(result.embeddings[0].values)
                # Normalize to 1536 dims for DB compatibility
                if len(embedding) > 1536:
                    embedding = embedding[:1536]
                elif len(embedding) < 1536:
                    embedding = embedding + [0.0] * (1536 - len(embedding))
                return embedding
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < max_retries - 1:
                sleep_for = base_backoff * (2 ** attempt)
                print(f"⚠️ Gemini rate-limited; retrying in {sleep_for:.1f}s")
                await asyncio.sleep(sleep_for)
                continue
            print(f"⚠️ Gemini embedding failed: {exc}")
            return None
    return None


async def _generate_embedding(text: str) -> Tuple[Optional[list[float]], Optional[str]]:
    for provider in _provider_order():
        if provider == "openai":
            emb = await _embed_openai(text)
            if emb:
                return emb, "openai"
        elif provider == "gemini":
            emb = await _embed_gemini(text)
            if emb:
                return emb, "gemini"
    return None, None


async def backfill_embeddings() -> None:
    pool_config = PoolConfig(
        host=app_config.database.host,
        port=app_config.database.port,
        user=app_config.database.user,
        password=app_config.database.password,
        database=app_config.database.database,
        ssl=app_config.database.ssl,
        ssl_verify=app_config.database.ssl_verify,
    )

    await init_pool(pool_config)
    pool = get_pool()

    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "25"))
    sleep_between_batches = float(os.getenv("EMBEDDING_BATCH_SLEEP", "0.2"))

    stats = {"processed": 0, "updated": 0, "failed": 0, "openai": 0, "gemini": 0}

    try:
        total_missing = await pool.fetchval(
            "SELECT COUNT(*) FROM unified_ai_memory WHERE embedding IS NULL AND content IS NOT NULL"
        )
        print(f"📉 Memories missing embeddings: {total_missing}")
        if not total_missing:
            print("✅ Nothing to backfill.")
            return

        while True:
            rows = await pool.fetch(
                """
                SELECT id, content
                FROM unified_ai_memory
                WHERE embedding IS NULL AND content IS NOT NULL
                ORDER BY created_at ASC
                LIMIT $1
                """,
                batch_size,
            )
            if not rows:
                break

            updates = []
            for row in rows:
                mem_id = row["id"]
                content = row["content"]
                text = _content_to_text(content)
                if not text:
                    stats["failed"] += 1
                    continue

                embedding, provider = await _generate_embedding(text)
                stats["processed"] += 1
                if embedding:
                    updates.append((json.dumps(embedding), mem_id))
                    if provider:
                        stats[provider] += 1
                else:
                    stats["failed"] += 1

            if updates:
                try:
                    await pool.executemany(
                        "UPDATE unified_ai_memory SET embedding = $1::vector WHERE id = $2",
                        updates,
                    )
                except asyncio.TimeoutError:
                    print("⚠️ Pool release timed out; reinitializing pool and retrying batch")
                    await close_pool()
                    await init_pool(pool_config)
                    pool = get_pool()
                    await pool.executemany(
                        "UPDATE unified_ai_memory SET embedding = $1::vector WHERE id = $2",
                        updates,
                    )
                stats["updated"] += len(updates)
                print(
                    f"✅ Batch healed: {len(updates)} | total updated: {stats['updated']} | openai: {stats['openai']} | gemini: {stats['gemini']}"
                )

            if sleep_between_batches:
                await asyncio.sleep(sleep_between_batches)

        remaining = await pool.fetchval(
            "SELECT COUNT(*) FROM unified_ai_memory WHERE embedding IS NULL AND content IS NOT NULL"
        )
        print(f"✅ Backfill complete. Remaining missing embeddings: {remaining}")
        print(f"📊 Stats: {stats}")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(backfill_embeddings())

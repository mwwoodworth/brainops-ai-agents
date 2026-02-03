import asyncio
import hashlib
import logging
import os
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)
_OPENAI_CLIENT = None
_GEMINI_CLIENT = None
_LOCAL_MODEL = None


def get_embedding_dimension() -> int:
    raw = os.getenv("EMBEDDING_DIMENSION", "1536").strip()
    try:
        return max(1, int(raw))
    except Exception:
        logger.warning("Invalid EMBEDDING_DIMENSION=%r; defaulting to 1536", raw)
        return 1536


def get_embedding_max_chars() -> int:
    raw = os.getenv("EMBEDDING_MAX_CHARS", "30000").strip()
    try:
        return max(1, int(raw))
    except Exception:
        logger.warning("Invalid EMBEDDING_MAX_CHARS=%r; defaulting to 30000", raw)
        return 30000


def get_openai_model() -> str:
    return os.getenv("EMBEDDING_OPENAI_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"


def get_gemini_model() -> str:
    return os.getenv("EMBEDDING_GEMINI_MODEL", "text-embedding-004").strip() or "text-embedding-004"


def get_local_model_name() -> str:
    return os.getenv("EMBEDDING_LOCAL_MODEL", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"


def is_strict_provider() -> bool:
    return os.getenv("EMBEDDING_PROVIDER_STRICT", "").strip().lower() in {"1", "true", "yes"}


def get_provider_order() -> List[str]:
    explicit_provider = os.getenv("EMBEDDING_PROVIDER", "").strip()
    order_env = os.getenv("EMBEDDING_PROVIDER_ORDER", "openai,gemini").strip()
    order = explicit_provider or order_env
    providers = [p.strip().lower() for p in order.split(",") if p.strip()]
    if not providers:
        providers = ["openai"]
    if len(providers) > 1 and not is_strict_provider():
        logger.warning(
            "Multiple embedding providers configured (%s). "
            "Mixed vector spaces can degrade similarity. "
            "Set EMBEDDING_PROVIDER_STRICT=true to lock to the first provider.",
            providers,
        )
    return providers


def iter_providers(providers: Iterable[str]) -> List[str]:
    providers = list(providers)
    if is_strict_provider() and providers:
        return providers[:1]
    return providers


def allow_local_fallback(providers: Iterable[str]) -> bool:
    return "local" in providers


def allow_hash_fallback(providers: Iterable[str]) -> bool:
    return "hash" in providers


def allow_zero_fallback() -> bool:
    return os.getenv("EMBEDDING_ALLOW_ZERO", "").strip().lower() in {"1", "true", "yes"}


def normalize_embedding(embedding, dimension: int | None = None):
    if embedding is None:
        return None
    dim = dimension or get_embedding_dimension()
    try:
        emb = list(embedding)
    except Exception:
        return None
    if len(emb) > dim:
        return emb[:dim]
    if len(emb) < dim:
        return emb + [0.0] * (dim - len(emb))
    return emb


def _truncate(text: str) -> str:
    if text is None:
        return ""
    max_chars = get_embedding_max_chars()
    return text[:max_chars] if len(text) > max_chars else text


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        try:
            import openai  # type: ignore
            OpenAI = getattr(openai, "OpenAI", None)
        except Exception:
            OpenAI = None
    if OpenAI is None:
        return None
    _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT
    api_key = (
        os.getenv("GOOGLE_AI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if not api_key:
        return None
    try:
        from google import genai  # type: ignore
    except Exception:
        return None
    _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def _get_local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is not None:
        return _LOCAL_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    _LOCAL_MODEL = SentenceTransformer(get_local_model_name())
    return _LOCAL_MODEL


def _embed_openai(text: str, log: Optional[logging.Logger] = None) -> Optional[list[float]]:
    client = _get_openai_client()
    if not client:
        return None
    try:
        response = client.embeddings.create(
            input=_truncate(text),
            model=get_openai_model(),
        )
        return normalize_embedding(response.data[0].embedding)
    except Exception as exc:
        if log:
            log.warning("OpenAI embedding failed: %s", exc)
        return None


def _embed_gemini(text: str, log: Optional[logging.Logger] = None) -> Optional[list[float]]:
    client = _get_gemini_client()
    if not client:
        return None
    try:
        result = client.models.embed_content(
            model=get_gemini_model(),
            contents=_truncate(text),
        )
        if result and result.embeddings:
            embedding = list(result.embeddings[0].values)
            return normalize_embedding(embedding)
    except Exception as exc:
        if log:
            log.warning("Gemini embedding failed: %s", exc)
    return None


def _embed_local(text: str, log: Optional[logging.Logger] = None) -> Optional[list[float]]:
    model = _get_local_model()
    if model is None:
        return None
    try:
        embedding = model.encode(_truncate(text)).tolist()
        return normalize_embedding(embedding)
    except Exception as exc:
        if log:
            log.warning("Local embedding failed: %s", exc)
    return None


def _hash_embedding(text: str, dimension: int) -> list[float]:
    text_lower = (text or "").lower()
    embedding: list[float] = []
    for i in range(dimension):
        h = hashlib.sha256(f"{text_lower}:{i}".encode()).digest()
        val = (int.from_bytes(h[:4], "big") / (2**32)) * 2 - 1
        embedding.append(val)
    return embedding


def generate_embedding_sync(text: str, log: Optional[logging.Logger] = None) -> Optional[list[float]]:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    providers = iter_providers(get_provider_order())
    dimension = get_embedding_dimension()

    for provider in providers:
        if provider in {"openai", "oai"}:
            embedding = _embed_openai(text, log=log)
        elif provider in {"gemini", "google"}:
            embedding = _embed_gemini(text, log=log)
        elif provider == "local":
            embedding = _embed_local(text, log=log)
        elif provider == "hash":
            embedding = _hash_embedding(text, dimension)
        elif provider == "zero":
            embedding = [0.0] * dimension
        else:
            if log:
                log.warning("Unknown embedding provider %r", provider)
            continue

        if embedding is not None:
            return normalize_embedding(embedding, dimension)

    if allow_zero_fallback():
        if log:
            log.warning("All embedding providers failed; using zero embedding (EMBEDDING_ALLOW_ZERO=true)")
        return [0.0] * dimension
    return None


async def generate_embedding_async(text: str, log: Optional[logging.Logger] = None) -> Optional[list[float]]:
    return await asyncio.to_thread(generate_embedding_sync, text, log)

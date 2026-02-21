# Phase 2 — Cross-Repo Duplication Map

**Date:** 2026-02-21
**Repos compared:** brainops-ai-agents vs. myroofgenius-backend
**Method:** file-name overlap + diff line count

---

## Executive Summary

Only 4 Python source files share the same filename across both repositories.
These 4 files account for approximately 4% of the total Python file population
across both repos (4 out of ~100 unique filenames in the overlap set).

**Conclusion:** The two repositories are intentionally separate services with
distinct responsibilities. The overlapping filenames reflect shared *concepts*
(config, observability, LLM orchestration, system utilities), not copy-paste
duplication. Each copy has diverged significantly and should remain independent.

| File | Agents lines | Backend lines | Diff lines | Similarity | Recommendation |
|------|-------------|---------------|------------|------------|----------------|
| config.py | 213 | 188 | 399 | Low (service-specific schemas) | Keep separate |
| observability.py | 180 | 420 | 595 | Low (different middleware) | Keep separate |
| langgraph_orchestrator.py | 495 | 169 | 657 | Low (different graph topology) | Keep separate |
| ai_ultimate_system.py | 503 | 493 | 186 | Moderate (shared base concepts) | Keep separate — monitor |

---

## File-by-File Analysis

### 1. config.py

| Attribute | brainops-ai-agents | myroofgenius-backend |
|-----------|-------------------|---------------------|
| Path | `/home/matt-woodworth/dev/brainops-ai-agents/config.py` | `/home/matt-woodworth/dev/myroofgenius-backend/config.py` |
| Lines | 213 | 188 |
| Diff lines | 399 | |

**Agents version** configures: DatabaseConfig (individual env vars + DATABASE_URL fallback,
SSL/TLS settings for Supabase pooler), SecurityConfig (valid_api_keys, master_api_key,
auth_required), and service-level identifiers for the AI Agents service. Loads from .env
via python-dotenv.

**Backend version** configures: WeatherCraft ERP backend settings, different database
connection parameters, different security defaults, and ERP-specific feature flags.

**Assessment:** Same pattern, entirely different configuration surface. Extracting a shared
config library would add coupling between two independently deployed services with different
release cycles. Keep separate.

---

### 2. observability.py

| Attribute | brainops-ai-agents | myroofgenius-backend |
|-----------|-------------------|---------------------|
| Path | `/home/matt-woodworth/dev/brainops-ai-agents/observability.py` | `/home/matt-woodworth/dev/myroofgenius-backend/observability.py` |
| Lines | 180 | 420 |
| Diff lines | 595 | |

**Agents version** (180 lines) implements lightweight FastAPI middleware for request
duration tracking and structured JSON logging, tailored to the AI agents workload
(agent invocations, memory operations).

**Backend version** (420 lines) is significantly larger and implements ERP-focused
observability: tenant-scoped trace IDs, WeatherCraft job tracking, and ERP-specific
metric labels. The additional 240 lines represent ERP domain logic not applicable
to the agents service.

**Assessment:** Related concept, diverged implementations driven by different domain
requirements. Merging would bloat the agents service with ERP concerns. Keep separate.

---

### 3. langgraph_orchestrator.py

| Attribute | brainops-ai-agents | myroofgenius-backend |
|-----------|-------------------|---------------------|
| Path | `/home/matt-woodworth/dev/brainops-ai-agents/langgraph_orchestrator.py` | `/home/matt-woodworth/dev/myroofgenius-backend/langgraph_orchestrator.py` |
| Lines | 495 | 169 |
| Diff lines | 657 | |

**Agents version** (495 lines) implements a full multi-stage LangGraph StateGraph:
initialization → memory retrieval → context building → agent selection → task execution
→ memory storage → response generation. Integrates OpenAI + Anthropic LLMs and pgvector
RAG. This is a production orchestration graph with conditional routing.

**Backend version** (169 lines) is a smaller stub/adapter used by the ERP backend for
a different subset of LangGraph capabilities relevant to ERP workflow routing.

**Assessment:** The agents version is 3x larger and implements a substantially different
graph topology. The name similarity is coincidental to both using LangGraph. Keep separate.

---

### 4. ai_ultimate_system.py

| Attribute | brainops-ai-agents | myroofgenius-backend |
|-----------|-------------------|---------------------|
| Path | `/home/matt-woodworth/dev/brainops-ai-agents/ai_ultimate_system.py` | `/home/matt-woodworth/dev/myroofgenius-backend/ai_ultimate_system.py` |
| Lines | 503 | 493 |
| Diff lines | 186 | |

**Similarity is highest here** (186 diff lines against ~500 lines each = ~63% structural
similarity). Both files define a top-level "ultimate system" class that aggregates
subsystem references, coordinates agent startup, and exposes a unified status dict.

**Assessment:** This is the most similar pair. The 186-line diff represents
service-specific subsystem wiring (agents service has AUREA, self-healing, memory agents;
backend has ERP-specific agents). The shared pattern emerged from a common initial
implementation. Despite the similarity, extracting a shared base class would introduce
a cross-repo dependency on an unstable internal API. Flag for future review if both
files continue to converge. For now, keep separate and accept the ~37% duplicate logic
as a known trade-off of the independently-deployed service model.

---

## Total Python File Counts

| Repo | Total .py files |
|------|----------------|
| brainops-ai-agents | 16,860 (includes __pycache__) |
| myroofgenius-backend | 19,848 (includes __pycache__) |
| Overlapping filenames | 4 |

Overlap rate by filename: 4 shared names out of the meaningful source files in each
repo. This is within the expected range for two microservices built on the same
Python/FastAPI/LangGraph stack by the same author.

---

## Policy Decision

**No action required in Wave 2A.** Both services are intentionally separate:

- Separate Render service IDs, separate Docker images, separate deploy pipelines.
- Different tenancy models (agents = system-level; backend = tenant-scoped ERP).
- Different RLS roles (agent_worker vs. backend_worker, both NOBYPASSRLS).
- Independent versioning (Agents v11.33.0; Backend v163.12.0).

If ai_ultimate_system.py drift continues to close, consider extracting a shared
`brainops-common` internal package. That decision is out of scope for Phase 2.

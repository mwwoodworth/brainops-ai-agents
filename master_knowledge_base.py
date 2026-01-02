"""
MASTER KNOWLEDGE BASE - BRAINOPS AI OS
======================================
Central knowledge repository with vector search, semantic understanding,
and multi-AI integration for comprehensive organizational intelligence.

Features:
- Vector-based semantic search
- Hierarchical knowledge organization
- Multi-format document ingestion
- AI-powered knowledge extraction
- Automatic relationship discovery
- Version control and audit trails
- Real-time knowledge updates
- Cross-system knowledge sync

Author: BrainOps AI OS
Version: 1.0.0
"""

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx
from loguru import logger

# =============================================================================
# ENUMERATIONS
# =============================================================================

class KnowledgeType(Enum):
    """Types of knowledge entries."""
    # Core Documentation
    SOP = "sop"                           # Standard Operating Procedures
    POLICY = "policy"                     # Company policies
    PROCESS = "process"                   # Business processes
    GUIDE = "guide"                       # How-to guides
    TUTORIAL = "tutorial"                 # Step-by-step tutorials

    # Technical
    API_DOCUMENTATION = "api_documentation"
    CODE_REFERENCE = "code_reference"
    ARCHITECTURE = "architecture"
    INTEGRATION = "integration"
    TROUBLESHOOTING = "troubleshooting"

    # Business
    PRODUCT = "product"                   # Product information
    PRICING = "pricing"                   # Pricing details
    FAQ = "faq"                          # Frequently asked questions
    CASE_STUDY = "case_study"            # Customer success stories
    COMPETITOR = "competitor"             # Competitive intelligence

    # AI/Agent
    PROMPT_TEMPLATE = "prompt_template"   # AI prompt templates
    AGENT_CONFIG = "agent_config"         # Agent configurations
    MODEL_GUIDE = "model_guide"           # AI model usage guides
    TRAINING_DATA = "training_data"       # Training examples

    # Operations
    RUNBOOK = "runbook"                   # Operational runbooks
    INCIDENT = "incident"                 # Incident reports
    LESSON_LEARNED = "lesson_learned"     # Post-mortems
    BEST_PRACTICE = "best_practice"       # Best practices

    # Customer
    CUSTOMER_PROFILE = "customer_profile"
    USE_CASE = "use_case"
    TESTIMONIAL = "testimonial"
    OBJECTION_HANDLER = "objection_handler"


class AccessLevel(Enum):
    """Access control levels."""
    PUBLIC = "public"           # Anyone can access
    INTERNAL = "internal"       # All internal users
    TEAM = "team"              # Specific team access
    RESTRICTED = "restricted"   # Limited access
    CONFIDENTIAL = "confidential"  # Executive only
    AI_ONLY = "ai_only"        # Only AI agents can access


class KnowledgeStatus(Enum):
    """Knowledge entry status."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class RelationType(Enum):
    """Types of knowledge relationships."""
    PARENT = "parent"           # Hierarchical parent
    CHILD = "child"             # Hierarchical child
    RELATED = "related"         # Related topic
    PREREQUISITE = "prerequisite"  # Required before
    SUPERSEDES = "supersedes"   # Replaces older version
    IMPLEMENTS = "implements"   # Implements spec/design
    REFERENCES = "references"   # References another
    CONTRADICTS = "contradicts"  # Conflicting information


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class KnowledgeEntry:
    """Individual knowledge entry."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Core content
    title: str = ""
    content: str = ""
    summary: str = ""
    knowledge_type: KnowledgeType = KnowledgeType.GUIDE

    # Organization
    category: str = ""
    subcategory: str = ""
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # Access control
    access_level: AccessLevel = AccessLevel.INTERNAL
    allowed_teams: list[str] = field(default_factory=list)
    allowed_users: list[str] = field(default_factory=list)
    allowed_agents: list[str] = field(default_factory=list)

    # Status
    status: KnowledgeStatus = KnowledgeStatus.DRAFT
    version: int = 1
    is_current: bool = True

    # Relationships
    parent_id: Optional[str] = None
    relationships: list[dict[str, str]] = field(default_factory=list)

    # Source
    source_type: str = "manual"  # manual, import, ai_generated, extracted
    source_url: str = ""
    source_file: str = ""
    original_author: str = ""

    # Vector embedding
    embedding: list[float] = field(default_factory=list)
    embedding_model: str = ""

    # Metadata
    word_count: int = 0
    reading_time_minutes: int = 0
    language: str = "en"
    confidence_score: float = 1.0  # For AI-generated content

    # Engagement
    view_count: int = 0
    helpful_count: int = 0
    not_helpful_count: int = 0
    citation_count: int = 0  # How often AI cites this

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_reviewed_at: Optional[datetime] = None
    next_review_at: Optional[datetime] = None

    # Audit
    created_by: str = ""
    updated_by: str = ""
    review_history: list[dict[str, Any]] = field(default_factory=list)

    # Custom data
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeCategory:
    """Knowledge category/taxonomy node."""
    category_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    slug: str = ""
    description: str = ""
    parent_id: Optional[str] = None
    icon: str = ""
    color: str = ""
    order: int = 0
    entry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeQuery:
    """Knowledge search query."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    user_id: str = ""
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Results
    results_count: int = 0
    top_result_score: float = 0.0
    selected_result_id: Optional[str] = None

    # Feedback
    was_helpful: Optional[bool] = None
    feedback_text: str = ""


@dataclass
class KnowledgeExtraction:
    """AI-extracted knowledge from documents."""
    extraction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_document: str = ""
    source_type: str = ""  # pdf, webpage, email, conversation

    # Extracted content
    extracted_entries: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, str]] = field(default_factory=list)
    key_phrases: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    # Metadata
    extraction_model: str = ""
    confidence_score: float = 0.0
    processing_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# VECTOR STORE (SIMPLIFIED IN-MEMORY)
# =============================================================================

class SimpleVectorStore:
    """
    Simple in-memory vector store for semantic search.
    In production, replace with Pinecone, Weaviate, or Qdrant.
    """

    def __init__(self):
        self.vectors: dict[str, list[float]] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

    def add(self, entry_id: str, vector: list[float], metadata: dict[str, Any] = None):
        """Add a vector to the store."""
        self.vectors[entry_id] = vector
        self.metadata[entry_id] = metadata or {}

    def delete(self, entry_id: str):
        """Delete a vector from the store."""
        self.vectors.pop(entry_id, None)
        self.metadata.pop(entry_id, None)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] = None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar vectors."""
        if not self.vectors:
            return []

        results = []
        for entry_id, vector in self.vectors.items():
            # Apply filters
            if filters:
                meta = self.metadata.get(entry_id, {})
                if not self._matches_filters(meta, filters):
                    continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_vector, vector)
            results.append((entry_id, similarity, self.metadata.get(entry_id, {})))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            meta_value = metadata.get(key)
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            elif meta_value != value:
                return False
        return True


# =============================================================================
# KNOWLEDGE PROCESSOR
# =============================================================================

class KnowledgeProcessor:
    """Processes and extracts knowledge from various sources."""

    def __init__(self, openai_api_key: str = None, anthropic_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    async def process_document(
        self,
        content: str,
        source_type: str,
        source_name: str
    ) -> KnowledgeExtraction:
        """Process a document and extract knowledge."""
        extraction = KnowledgeExtraction(
            source_document=source_name,
            source_type=source_type,
        )

        # Extract key information
        prompt = f"""Analyze this document and extract structured knowledge:

Document Type: {source_type}
Document Name: {source_name}
Content:
{content[:8000]}  # Truncate for context window

Extract:
1. KEY_FACTS: List of important facts/information
2. ENTITIES: Named entities (people, products, companies, systems)
3. PROCEDURES: Any step-by-step procedures or processes
4. DECISIONS: Any decisions or conclusions mentioned
5. ACTION_ITEMS: Any action items or tasks
6. RELATIONSHIPS: Connections between concepts
7. SUMMARY: Brief summary (2-3 sentences)

Format as JSON with these exact keys."""

        try:
            response = await self._call_claude(prompt)
            parsed = self._parse_json_response(response)

            extraction.extracted_entries = parsed.get("KEY_FACTS", [])
            extraction.entities = [
                {"name": e, "type": "entity"} for e in parsed.get("ENTITIES", [])
            ]
            extraction.key_phrases = parsed.get("PROCEDURES", [])
            extraction.action_items = parsed.get("ACTION_ITEMS", [])
            extraction.decisions = parsed.get("DECISIONS", [])
            extraction.confidence_score = 0.85

        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            extraction.confidence_score = 0.0

        return extraction

    async def generate_embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small"
    ) -> list[list[float]]:
        """Generate embeddings for texts using OpenAI."""
        if not self.openai_api_key:
            # Try to get from environment if not set
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            logger.error("OpenAI API Key missing for embeddings")
            raise ValueError("OPENAI_API_KEY is required for knowledge base embeddings")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "input": texts,
                    }
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # If API fails, we cannot return mock data in production
            raise

    async def generate_summary(self, content: str, max_words: int = 100) -> str:
        """Generate a summary of content."""
        prompt = f"""Summarize this content in {max_words} words or less:

{content[:4000]}

Write a clear, concise summary that captures the key points."""

        return await self._call_claude(prompt)

    async def extract_keywords(self, content: str, max_keywords: int = 10) -> list[str]:
        """Extract keywords from content."""
        prompt = f"""Extract the {max_keywords} most important keywords/phrases from this content:

{content[:4000]}

Return as a simple comma-separated list."""

        response = await self._call_claude(prompt)
        return [kw.strip() for kw in response.split(",")][:max_keywords]

    async def categorize_content(
        self,
        content: str,
        categories: list[str]
    ) -> tuple[str, float]:
        """Categorize content into predefined categories."""
        prompt = f"""Categorize this content into one of these categories:
{', '.join(categories)}

Content:
{content[:2000]}

Return ONLY the category name that best matches, nothing else."""

        category = await self._call_claude(prompt)
        category = category.strip()

        # Find best match
        for cat in categories:
            if cat.lower() in category.lower() or category.lower() in cat.lower():
                return cat, 0.9

        return categories[0], 0.5  # Default to first with low confidence

    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API."""
        if not self.anthropic_api_key:
             self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.anthropic_api_key:
            logger.error("Anthropic API Key missing")
            raise ValueError("ANTHROPIC_API_KEY is required for knowledge processing")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-3-haiku-20240307",  # Use Haiku for faster processing
                        "max_tokens": 2048,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from AI response."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.debug("Failed to parse knowledge base JSON response")
        return {}


# =============================================================================
# MASTER KNOWLEDGE BASE
# =============================================================================

class MasterKnowledgeBase:
    """
    Central knowledge repository for the entire AI OS.

    Capabilities:
    - CRUD operations on knowledge entries
    - Semantic vector search
    - Hierarchical organization
    - Access control
    - Version history
    - AI-powered extraction and processing
    - Cross-reference management
    - Analytics and insights
    """

    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")

        # In-memory storage (would be database in production)
        self.entries: dict[str, KnowledgeEntry] = {}
        self.categories: dict[str, KnowledgeCategory] = {}
        self.queries: dict[str, KnowledgeQuery] = {}

        # Vector store
        self.vector_store = SimpleVectorStore()

        # Processor
        self.processor = KnowledgeProcessor()

        # Indexes
        self.by_type: dict[KnowledgeType, set[str]] = {t: set() for t in KnowledgeType}
        self.by_category: dict[str, set[str]] = {}
        self.by_tag: dict[str, set[str]] = {}

        # Initialize default categories
        self._initialize_categories()

        logger.info("MasterKnowledgeBase initialized")

    def _initialize_categories(self):
        """Initialize default knowledge categories."""
        default_categories = [
            ("documentation", "Documentation", "Technical and user documentation"),
            ("operations", "Operations", "Operational procedures and runbooks"),
            ("product", "Product", "Product information and guides"),
            ("sales", "Sales", "Sales materials and processes"),
            ("support", "Support", "Customer support knowledge"),
            ("engineering", "Engineering", "Engineering and development"),
            ("ai-agents", "AI Agents", "AI agent configurations and prompts"),
            ("training", "Training", "Training materials and courses"),
            ("compliance", "Compliance", "Policies and compliance"),
            ("analytics", "Analytics", "Reports and analytics"),
        ]

        for slug, name, description in default_categories:
            category = KnowledgeCategory(
                slug=slug,
                name=name,
                description=description,
            )
            self.categories[category.category_id] = category

    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================

    async def create_entry(
        self,
        title: str,
        content: str,
        knowledge_type: KnowledgeType,
        category: str = "",
        tags: list[str] = None,
        access_level: AccessLevel = AccessLevel.INTERNAL,
        source_type: str = "manual",
        created_by: str = "system",
        metadata: dict[str, Any] = None,
        auto_process: bool = True
    ) -> KnowledgeEntry:
        """Create a new knowledge entry."""

        entry = KnowledgeEntry(
            title=title,
            content=content,
            knowledge_type=knowledge_type,
            category=category,
            tags=tags or [],
            access_level=access_level,
            source_type=source_type,
            created_by=created_by,
            updated_by=created_by,
            metadata=metadata or {},
        )

        # Auto-process
        if auto_process:
            # Generate summary
            if len(content) > 500:
                entry.summary = await self.processor.generate_summary(content)

            # Extract keywords
            entry.keywords = await self.processor.extract_keywords(content)

            # Calculate metrics
            entry.word_count = len(content.split())
            entry.reading_time_minutes = max(1, entry.word_count // 200)

            # Generate embedding
            embedding_text = f"{title}\n{entry.summary}\n{content[:2000]}"
            embeddings = await self.processor.generate_embeddings([embedding_text])
            if embeddings:
                entry.embedding = embeddings[0]
                entry.embedding_model = "text-embedding-3-small"

        # Store entry
        self.entries[entry.entry_id] = entry

        # Update indexes
        self.by_type[knowledge_type].add(entry.entry_id)

        if category:
            if category not in self.by_category:
                self.by_category[category] = set()
            self.by_category[category].add(entry.entry_id)

        for tag in entry.tags:
            if tag not in self.by_tag:
                self.by_tag[tag] = set()
            self.by_tag[tag].add(entry.entry_id)

        # Add to vector store
        if entry.embedding:
            self.vector_store.add(
                entry.entry_id,
                entry.embedding,
                {
                    "type": knowledge_type.value,
                    "category": category,
                    "access_level": access_level.value,
                    "status": entry.status.value,
                }
            )

        logger.info(f"Created knowledge entry: {entry.entry_id} ({title})")

        return entry

    async def update_entry(
        self,
        entry_id: str,
        updates: dict[str, Any],
        updated_by: str = "system",
        create_version: bool = True
    ) -> KnowledgeEntry:
        """Update a knowledge entry."""

        entry = self.entries.get(entry_id)
        if not entry:
            raise ValueError(f"Entry {entry_id} not found")

        # Create version history
        if create_version:
            old_version = {
                "version": entry.version,
                "content": entry.content,
                "updated_at": entry.updated_at.isoformat(),
                "updated_by": entry.updated_by,
            }
            entry.review_history.append(old_version)
            entry.version += 1

        # Apply updates
        for key, value in updates.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        entry.updated_at = datetime.utcnow()
        entry.updated_by = updated_by

        # Regenerate embedding if content changed
        if "content" in updates or "title" in updates:
            embedding_text = f"{entry.title}\n{entry.summary}\n{entry.content[:2000]}"
            embeddings = await self.processor.generate_embeddings([embedding_text])
            if embeddings:
                entry.embedding = embeddings[0]
                self.vector_store.add(
                    entry.entry_id,
                    entry.embedding,
                    {
                        "type": entry.knowledge_type.value,
                        "category": entry.category,
                        "access_level": entry.access_level.value,
                        "status": entry.status.value,
                    }
                )

        logger.info(f"Updated knowledge entry: {entry_id} (v{entry.version})")

        return entry

    async def delete_entry(self, entry_id: str, hard_delete: bool = False) -> bool:
        """Delete a knowledge entry."""

        entry = self.entries.get(entry_id)
        if not entry:
            return False

        if hard_delete:
            # Remove from all indexes
            self.by_type[entry.knowledge_type].discard(entry_id)
            if entry.category in self.by_category:
                self.by_category[entry.category].discard(entry_id)
            for tag in entry.tags:
                if tag in self.by_tag:
                    self.by_tag[tag].discard(entry_id)

            self.vector_store.delete(entry_id)
            del self.entries[entry_id]
        else:
            # Soft delete (archive)
            entry.status = KnowledgeStatus.ARCHIVED
            entry.updated_at = datetime.utcnow()

        logger.info(f"Deleted knowledge entry: {entry_id} (hard={hard_delete})")

        return True

    async def get_entry(
        self,
        entry_id: str,
        user_id: str = None,
        agent_id: str = None,
        track_view: bool = True
    ) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry by ID."""

        entry = self.entries.get(entry_id)
        if not entry:
            return None

        # Check access
        if not self._check_access(entry, user_id, agent_id):
            logger.warning(f"Access denied to entry {entry_id}")
            return None

        # Track view
        if track_view:
            entry.view_count += 1

        return entry

    def _check_access(
        self,
        entry: KnowledgeEntry,
        user_id: str = None,
        agent_id: str = None
    ) -> bool:
        """Check if user/agent has access to entry."""

        if entry.access_level == AccessLevel.PUBLIC:
            return True

        if entry.access_level == AccessLevel.AI_ONLY:
            return agent_id is not None

        if entry.access_level == AccessLevel.CONFIDENTIAL:
            return user_id in entry.allowed_users

        # For other levels, allow if user or agent is specified
        if user_id:
            if entry.allowed_users and user_id not in entry.allowed_users:
                return False
        if agent_id:
            if entry.allowed_agents and agent_id not in entry.allowed_agents:
                return False

        return True

    # =========================================================================
    # SEARCH
    # =========================================================================

    async def search(
        self,
        query: str,
        knowledge_types: list[KnowledgeType] = None,
        categories: list[str] = None,
        tags: list[str] = None,
        access_levels: list[AccessLevel] = None,
        top_k: int = 10,
        user_id: str = None,
        agent_id: str = None,
        semantic: bool = True
    ) -> list[tuple[KnowledgeEntry, float]]:
        """Search knowledge base."""

        results = []

        if semantic and query:
            # Semantic search using vectors
            query_embedding = await self.processor.generate_embeddings([query])
            if query_embedding:
                filters = {}
                if knowledge_types:
                    filters["type"] = [t.value for t in knowledge_types]
                if categories:
                    filters["category"] = categories
                if access_levels:
                    filters["access_level"] = [a.value for a in access_levels]

                vector_results = self.vector_store.search(
                    query_embedding[0],
                    top_k=top_k * 2,  # Get more for filtering
                    filters=filters
                )

                for entry_id, score, _ in vector_results:
                    entry = self.entries.get(entry_id)
                    if entry and self._check_access(entry, user_id, agent_id):
                        results.append((entry, score))

        else:
            # Keyword search
            query_lower = query.lower()
            for entry in self.entries.values():
                # Apply filters
                if knowledge_types and entry.knowledge_type not in knowledge_types:
                    continue
                if categories and entry.category not in categories:
                    continue
                if tags and not any(t in entry.tags for t in tags):
                    continue
                if access_levels and entry.access_level not in access_levels:
                    continue
                if not self._check_access(entry, user_id, agent_id):
                    continue

                # Calculate keyword score
                score = 0.0
                if query_lower in entry.title.lower():
                    score += 0.5
                if query_lower in entry.content.lower():
                    score += 0.3
                if any(query_lower in kw.lower() for kw in entry.keywords):
                    score += 0.2

                if score > 0:
                    results.append((entry, score))

        # Sort by score and limit
        results.sort(key=lambda x: x[1], reverse=True)

        # Track query
        knowledge_query = KnowledgeQuery(
            query_text=query,
            filters={
                "types": [t.value for t in knowledge_types] if knowledge_types else [],
                "categories": categories or [],
                "tags": tags or [],
            },
            user_id=user_id or "",
            agent_id=agent_id or "",
            results_count=len(results),
            top_result_score=results[0][1] if results else 0.0,
        )
        self.queries[knowledge_query.query_id] = knowledge_query

        return results[:top_k]

    async def find_related(
        self,
        entry_id: str,
        top_k: int = 5
    ) -> list[tuple[KnowledgeEntry, float]]:
        """Find entries related to a given entry."""

        entry = self.entries.get(entry_id)
        if not entry or not entry.embedding:
            return []

        # Search for similar vectors
        vector_results = self.vector_store.search(
            entry.embedding,
            top_k=top_k + 1  # +1 because it will match itself
        )

        results = []
        for result_id, score, _ in vector_results:
            if result_id != entry_id:
                related_entry = self.entries.get(result_id)
                if related_entry:
                    results.append((related_entry, score))

        return results[:top_k]

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    async def bulk_import(
        self,
        entries_data: list[dict[str, Any]],
        knowledge_type: KnowledgeType,
        category: str = "",
        created_by: str = "import"
    ) -> tuple[int, int, list[str]]:
        """Bulk import knowledge entries."""

        created = 0
        failed = 0
        errors = []

        for data in entries_data:
            try:
                await self.create_entry(
                    title=data.get("title", "Untitled"),
                    content=data.get("content", ""),
                    knowledge_type=knowledge_type,
                    category=category,
                    tags=data.get("tags", []),
                    source_type="import",
                    created_by=created_by,
                    metadata=data.get("metadata", {}),
                )
                created += 1
            except Exception as e:
                failed += 1
                errors.append(f"{data.get('title', 'Unknown')}: {str(e)}")

        logger.info(f"Bulk import: {created} created, {failed} failed")

        return created, failed, errors

    async def extract_from_document(
        self,
        document_path: str,
        document_type: str,
        auto_create: bool = True,
        created_by: str = "extraction"
    ) -> KnowledgeExtraction:
        """Extract knowledge from a document."""

        # Read document
        try:
            with open(document_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read document: {e}")
            raise

        # Process document
        extraction = await self.processor.process_document(
            content=content,
            source_type=document_type,
            source_name=Path(document_path).name,
        )

        # Auto-create entries
        if auto_create and extraction.extracted_entries:
            for fact in extraction.extracted_entries:
                if isinstance(fact, str) and len(fact) > 50:
                    await self.create_entry(
                        title=f"Extracted: {fact[:50]}...",
                        content=fact,
                        knowledge_type=KnowledgeType.BEST_PRACTICE,
                        source_type="extracted",
                        source_file=document_path,
                        created_by=created_by,
                        auto_process=False,
                    )

        return extraction

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    async def get_statistics(self) -> dict[str, Any]:
        """Get knowledge base statistics."""

        total_entries = len(self.entries)
        by_status = {}
        by_type = {}
        by_access = {}

        total_words = 0
        total_views = 0

        for entry in self.entries.values():
            by_status[entry.status.value] = by_status.get(entry.status.value, 0) + 1
            by_type[entry.knowledge_type.value] = by_type.get(entry.knowledge_type.value, 0) + 1
            by_access[entry.access_level.value] = by_access.get(entry.access_level.value, 0) + 1
            total_words += entry.word_count
            total_views += entry.view_count

        # Most viewed
        most_viewed = sorted(
            self.entries.values(),
            key=lambda e: e.view_count,
            reverse=True
        )[:10]

        # Most cited by AI
        most_cited = sorted(
            self.entries.values(),
            key=lambda e: e.citation_count,
            reverse=True
        )[:10]

        # Recent queries
        recent_queries = sorted(
            self.queries.values(),
            key=lambda q: q.timestamp,
            reverse=True
        )[:20]

        return {
            "totals": {
                "entries": total_entries,
                "words": total_words,
                "views": total_views,
                "categories": len(self.categories),
                "queries_tracked": len(self.queries),
            },
            "by_status": by_status,
            "by_type": by_type,
            "by_access_level": by_access,
            "most_viewed": [
                {"id": e.entry_id, "title": e.title, "views": e.view_count}
                for e in most_viewed
            ],
            "most_cited_by_ai": [
                {"id": e.entry_id, "title": e.title, "citations": e.citation_count}
                for e in most_cited
            ],
            "recent_queries": [
                {"query": q.query_text, "results": q.results_count, "timestamp": q.timestamp.isoformat()}
                for q in recent_queries
            ],
        }

    async def get_gaps_analysis(self) -> dict[str, Any]:
        """Analyze gaps in knowledge coverage."""

        # Types with no entries
        empty_types = [
            t.value for t in KnowledgeType
            if not self.by_type.get(t, set())
        ]

        # Categories with low coverage
        low_coverage_categories = []
        for cat_id, entry_ids in self.by_category.items():
            if len(entry_ids) < 3:
                low_coverage_categories.append({
                    "category": cat_id,
                    "entry_count": len(entry_ids),
                })

        # Stale content (not updated in 90 days)
        stale_threshold = datetime.utcnow() - timedelta(days=90)
        stale_entries = [
            {
                "id": e.entry_id,
                "title": e.title,
                "last_updated": e.updated_at.isoformat(),
            }
            for e in self.entries.values()
            if e.updated_at < stale_threshold and e.status == KnowledgeStatus.PUBLISHED
        ]

        # Unpopular content (low views, might need improvement)
        unpopular = [
            {
                "id": e.entry_id,
                "title": e.title,
                "views": e.view_count,
                "helpful_rate": (
                    e.helpful_count / (e.helpful_count + e.not_helpful_count)
                    if (e.helpful_count + e.not_helpful_count) > 0 else None
                ),
            }
            for e in self.entries.values()
            if e.view_count < 5 and e.status == KnowledgeStatus.PUBLISHED
        ]

        return {
            "empty_knowledge_types": empty_types,
            "low_coverage_categories": low_coverage_categories,
            "stale_content": stale_entries[:20],
            "unpopular_content": unpopular[:20],
            "recommendations": [
                f"Add content for knowledge type: {t}" for t in empty_types[:5]
            ] + [
                f"Expand category '{c['category']}' (only {c['entry_count']} entries)"
                for c in low_coverage_categories[:5]
            ],
        }

    # =========================================================================
    # AI AGENT INTERFACE
    # =========================================================================

    async def query_for_agent(
        self,
        agent_id: str,
        query: str,
        context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Query knowledge base for an AI agent.
        Returns formatted response suitable for agent consumption.
        """

        # Search with agent access
        results = await self.search(
            query=query,
            agent_id=agent_id,
            semantic=True,
            top_k=5,
        )

        if not results:
            return {
                "success": False,
                "message": "No relevant knowledge found",
                "entries": [],
            }

        # Track citations
        entries_data = []
        for entry, score in results:
            entry.citation_count += 1
            entries_data.append({
                "entry_id": entry.entry_id,
                "title": entry.title,
                "summary": entry.summary or entry.content[:300],
                "content": entry.content,
                "relevance_score": score,
                "knowledge_type": entry.knowledge_type.value,
                "keywords": entry.keywords,
                "source": entry.source_type,
            })

        # Build context for agent
        combined_knowledge = "\n\n---\n\n".join([
            f"**{e['title']}** (relevance: {e['relevance_score']:.2f})\n{e['summary']}"
            for e in entries_data
        ])

        return {
            "success": True,
            "query": query,
            "entries_count": len(entries_data),
            "entries": entries_data,
            "combined_knowledge": combined_knowledge,
            "suggested_follow_ups": await self._generate_follow_up_questions(query, entries_data),
        }

    async def _generate_follow_up_questions(
        self,
        query: str,
        entries: list[dict[str, Any]]
    ) -> list[str]:
        """Generate follow-up questions based on search results."""
        if not entries:
            return []

        # Simple rule-based follow-ups
        follow_ups = []
        for entry in entries[:3]:
            keywords = entry.get("keywords", [])
            if keywords:
                follow_ups.append(f"What is the relationship between {query} and {keywords[0]}?")

        return follow_ups[:3]

    async def add_agent_knowledge(
        self,
        agent_id: str,
        title: str,
        content: str,
        knowledge_type: KnowledgeType,
        confidence: float = 0.8
    ) -> KnowledgeEntry:
        """Allow an AI agent to add knowledge to the base."""

        entry = await self.create_entry(
            title=title,
            content=content,
            knowledge_type=knowledge_type,
            access_level=AccessLevel.AI_ONLY if confidence < 0.9 else AccessLevel.INTERNAL,
            source_type="ai_generated",
            created_by=f"agent:{agent_id}",
            metadata={
                "agent_id": agent_id,
                "confidence_score": confidence,
                "requires_review": confidence < 0.9,
            }
        )

        entry.confidence_score = confidence
        entry.status = KnowledgeStatus.REVIEW if confidence < 0.9 else KnowledgeStatus.PUBLISHED

        logger.info(f"Agent {agent_id} added knowledge: {entry.entry_id}")

        return entry


# =============================================================================
# FASTAPI ROUTER
# =============================================================================

def create_knowledge_router():
    """Create FastAPI router for knowledge base endpoints."""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])
    kb = MasterKnowledgeBase()

    class CreateEntryRequest(BaseModel):
        title: str
        content: str
        knowledge_type: str
        category: str = ""
        tags: list = []
        access_level: str = "internal"

    class SearchRequest(BaseModel):
        query: str
        knowledge_types: list = None
        categories: list = None
        tags: list = None
        top_k: int = 10

    class AgentQueryRequest(BaseModel):
        agent_id: str
        query: str
        context: dict = None

    @router.post("/entries")
    async def create_entry(data: CreateEntryRequest):
        """Create a new knowledge entry."""
        try:
            entry = await kb.create_entry(
                title=data.title,
                content=data.content,
                knowledge_type=KnowledgeType(data.knowledge_type),
                category=data.category,
                tags=data.tags,
                access_level=AccessLevel(data.access_level),
            )
            return {
                "success": True,
                "entry_id": entry.entry_id,
                "title": entry.title,
                "summary": entry.summary,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/entries/{entry_id}")
    async def get_entry(entry_id: str, user_id: str = None, agent_id: str = None):
        """Get a knowledge entry by ID."""
        entry = await kb.get_entry(entry_id, user_id=user_id, agent_id=agent_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        return {
            "entry_id": entry.entry_id,
            "title": entry.title,
            "content": entry.content,
            "summary": entry.summary,
            "knowledge_type": entry.knowledge_type.value,
            "category": entry.category,
            "tags": entry.tags,
            "keywords": entry.keywords,
            "view_count": entry.view_count,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
        }

    @router.post("/search")
    async def search(data: SearchRequest):
        """Search the knowledge base."""
        knowledge_types = None
        if data.knowledge_types:
            knowledge_types = [KnowledgeType(t) for t in data.knowledge_types]

        results = await kb.search(
            query=data.query,
            knowledge_types=knowledge_types,
            categories=data.categories,
            tags=data.tags,
            top_k=data.top_k,
        )

        return {
            "query": data.query,
            "results_count": len(results),
            "results": [
                {
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "summary": entry.summary,
                    "score": score,
                    "knowledge_type": entry.knowledge_type.value,
                }
                for entry, score in results
            ],
        }

    @router.post("/agent/query")
    async def agent_query(data: AgentQueryRequest):
        """Query knowledge base for an AI agent."""
        return await kb.query_for_agent(
            agent_id=data.agent_id,
            query=data.query,
            context=data.context,
        )

    @router.get("/statistics")
    async def get_statistics():
        """Get knowledge base statistics."""
        return await kb.get_statistics()

    @router.get("/gaps")
    async def get_gaps():
        """Get knowledge gaps analysis."""
        return await kb.get_gaps_analysis()

    @router.get("/categories")
    async def get_categories():
        """Get all knowledge categories."""
        return {
            "categories": [
                {
                    "id": c.category_id,
                    "name": c.name,
                    "slug": c.slug,
                    "description": c.description,
                    "entry_count": len(kb.by_category.get(c.slug, set())),
                }
                for c in kb.categories.values()
            ]
        }

    return router


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_knowledge_base_instance: Optional[MasterKnowledgeBase] = None


def get_knowledge_base() -> MasterKnowledgeBase:
    """Get the singleton MasterKnowledgeBase instance."""
    global _knowledge_base_instance
    if _knowledge_base_instance is None:
        _knowledge_base_instance = MasterKnowledgeBase()
    return _knowledge_base_instance


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Demo the knowledge base."""
    print("=" * 70)
    print("MASTER KNOWLEDGE BASE DEMO")
    print("=" * 70)

    kb = get_knowledge_base()

    # Create sample entries
    print("\n1. Creating knowledge entries...")

    entry1 = await kb.create_entry(
        title="How to Process Customer Refunds",
        content="""Standard Operating Procedure for Customer Refunds

1. ELIGIBILITY CHECK
   - Verify purchase date (within 30 days)
   - Confirm product/service has issues
   - Check customer account standing

2. APPROVAL PROCESS
   - Refunds under $100: Automatic approval
   - Refunds $100-$500: Manager approval required
   - Refunds over $500: Director approval required

3. PROCESSING
   - Process through original payment method
   - Update customer record
   - Send confirmation email
   - Log in CRM

4. FOLLOW-UP
   - Customer satisfaction survey (3 days after)
   - Root cause analysis if product defect
   - Update FAQ if common issue""",
        knowledge_type=KnowledgeType.SOP,
        category="support",
        tags=["refunds", "customer-service", "finance"],
    )
    print(f"   Created: {entry1.title}")

    entry2 = await kb.create_entry(
        title="API Authentication Guide",
        content="""API Authentication Documentation

Our API uses JWT tokens for authentication.

## Getting Started

1. Obtain API credentials from the dashboard
2. Generate a token using the /auth/token endpoint
3. Include the token in the Authorization header

## Token Format
Authorization: Bearer <your-token>

## Token Expiration
- Access tokens expire in 1 hour
- Refresh tokens expire in 7 days
- Use /auth/refresh to get new access token

## Rate Limits
- Standard: 100 requests/minute
- Premium: 1000 requests/minute
- Enterprise: Unlimited

## Error Codes
- 401: Invalid or expired token
- 403: Insufficient permissions
- 429: Rate limit exceeded""",
        knowledge_type=KnowledgeType.API_DOCUMENTATION,
        category="engineering",
        tags=["api", "authentication", "security"],
    )
    print(f"   Created: {entry2.title}")

    entry3 = await kb.create_entry(
        title="AI Agent Prompt Template: Customer Support",
        content="""CUSTOMER SUPPORT AI AGENT PROMPT

Role: You are a helpful customer support agent for WeatherCraft ERP.

Guidelines:
1. Be friendly, professional, and empathetic
2. Always greet the customer by name if known
3. Acknowledge their issue before providing solutions
4. If you don't know something, say so and escalate
5. Never share internal system information

Response Structure:
- Acknowledge: "I understand you're experiencing [issue]..."
- Empathize: "That must be frustrating..."
- Solve: "Here's what we can do..."
- Confirm: "Does this resolve your issue?"

Escalation Triggers:
- Customer requests manager
- Legal or compliance issues
- Security concerns
- Repeated same issue (3+ times)

Example Interaction:
Customer: "My invoice is wrong"
Agent: "Hi [Name], I understand you've noticed an issue with your invoice. Let me pull that up and take a look. Can you tell me which invoice number this is regarding?"

Knowledge Base Access:
- Query: "refund policy"
- Query: "billing FAQ"
- Query: "escalation procedures"
""",
        knowledge_type=KnowledgeType.PROMPT_TEMPLATE,
        category="ai-agents",
        tags=["ai", "support", "prompts"],
        access_level=AccessLevel.AI_ONLY,
    )
    print(f"   Created: {entry3.title}")

    # Search
    print("\n2. Searching knowledge base...")

    results = await kb.search("how to handle refunds", top_k=3)
    print("   Query: 'how to handle refunds'")
    print(f"   Results: {len(results)}")
    for entry, score in results:
        print(f"   - {entry.title} (score: {score:.3f})")

    # Find related
    print("\n3. Finding related entries...")

    related = await kb.find_related(entry1.entry_id, top_k=3)
    print(f"   Related to: {entry1.title}")
    for entry, score in related:
        print(f"   - {entry.title} (similarity: {score:.3f})")

    # Agent query
    print("\n4. AI Agent querying knowledge base...")

    agent_result = await kb.query_for_agent(
        agent_id="support_agent_001",
        query="What is the refund policy?",
    )
    print("   Agent query: 'What is the refund policy?'")
    print(f"   Success: {agent_result['success']}")
    print(f"   Entries found: {agent_result['entries_count']}")

    # Statistics
    print("\n5. Knowledge Base Statistics:")
    stats = await kb.get_statistics()
    print(f"   Total entries: {stats['totals']['entries']}")
    print(f"   Total words: {stats['totals']['words']}")
    print(f"   Categories: {stats['totals']['categories']}")

    # Gaps analysis
    print("\n6. Knowledge Gaps Analysis:")
    gaps = await kb.get_gaps_analysis()
    print(f"   Empty knowledge types: {len(gaps['empty_knowledge_types'])}")
    print(f"   Recommendations: {len(gaps['recommendations'])}")
    for rec in gaps['recommendations'][:3]:
        print(f"   - {rec}")

    print("\n" + "=" * 70)
    print("KNOWLEDGE BASE DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

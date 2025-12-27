"""
Multi-AI Product Generation Pipeline
Industry-defining automated digital product creation system

This is the CORE engine that orchestrates multiple AI models to generate
high-quality digital products at scale:
- eBooks and guides
- Templates (business, design, code)
- Tools and micro-applications
- Courses and training materials
- Marketing assets
- Documentation and SOPs

Architecture:
- Multi-model orchestration (Claude, GPT, Gemini)
- Quality assurance pipeline
- Design automation
- Revenue integration
"""

import os
import json
import asyncio
import logging
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import aiohttp

# Database
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class ProductType(Enum):
    """Types of digital products we can generate"""
    EBOOK = "ebook"
    GUIDE = "guide"
    TEMPLATE_BUSINESS = "template_business"
    TEMPLATE_DESIGN = "template_design"
    TEMPLATE_CODE = "template_code"
    CHECKLIST = "checklist"
    WORKSHEET = "worksheet"
    CALCULATOR = "calculator"
    MICRO_TOOL = "micro_tool"
    COURSE = "course"
    VIDEO_SCRIPT = "video_script"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA_PACK = "social_media_pack"
    EMAIL_SEQUENCE = "email_sequence"
    LANDING_PAGE = "landing_page"
    SOP = "sop"
    PLAYBOOK = "playbook"
    SWIPE_FILE = "swipe_file"
    PROMPT_PACK = "prompt_pack"
    API_WRAPPER = "api_wrapper"


class ProductStatus(Enum):
    """Product generation status"""
    QUEUED = "queued"
    RESEARCHING = "researching"
    OUTLINING = "outlining"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    DESIGNING = "designing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class AIModel(Enum):
    """Available AI models for generation"""
    CLAUDE_OPUS = "claude-opus-4"
    CLAUDE_SONNET = "claude-sonnet-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4O = "gpt-4o"
    GEMINI_PRO = "gemini-2.0-flash"
    GEMINI_ULTRA = "gemini-ultra"
    PERPLEXITY = "perplexity"


class QualityTier(Enum):
    """Product quality tiers"""
    STANDARD = "standard"      # Single model, basic review
    PREMIUM = "premium"        # Multi-model, enhanced review
    ULTIMATE = "ultimate"      # Full pipeline, human-like quality


@dataclass
class ProductSpec:
    """Specification for a product to generate"""
    product_id: str
    product_type: ProductType
    title: str
    description: str
    target_audience: str
    quality_tier: QualityTier = QualityTier.PREMIUM
    word_count_target: int = 5000
    style: str = "professional"
    tone: str = "authoritative"
    industry: str = "general"
    keywords: List[str] = field(default_factory=list)
    include_visuals: bool = True
    include_templates: bool = True
    include_examples: bool = True
    custom_instructions: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of product generation"""
    product_id: str
    status: ProductStatus
    content: Dict[str, Any] = field(default_factory=dict)
    assets: List[Dict] = field(default_factory=list)
    quality_score: float = 0.0
    generation_time_seconds: float = 0.0
    models_used: List[str] = field(default_factory=list)
    tokens_used: int = 0
    cost_estimate: float = 0.0
    errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class AIProvider(ABC):
    """Abstract base class for AI providers"""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None,
                       max_tokens: int = 4000) -> str:
        pass

    @abstractmethod
    async def analyze(self, content: str, instruction: str) -> Dict:
        pass


class ClaudeProvider(AIProvider):
    """Claude AI provider"""

    def __init__(self):
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        self.base_url = "https://api.anthropic.com/v1"

    async def generate(self, prompt: str, system_prompt: str = None,
                       max_tokens: int = 4000) -> str:
        async with aiohttp.ClientSession() as session:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            payload = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }

            if system_prompt:
                payload["system"] = system_prompt

            try:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["content"][0]["text"]
                    else:
                        error = await response.text()
                        logger.error(f"Claude API error: {error}")
                        return ""
            except Exception as e:
                logger.error(f"Claude generation failed: {e}")
                return ""

    async def analyze(self, content: str, instruction: str) -> Dict:
        prompt = f"{instruction}\n\nContent to analyze:\n{content}"
        result = await self.generate(prompt, max_tokens=2000)
        try:
            return json.loads(result)
        except:
            return {"analysis": result}


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider"""

    def __init__(self):
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"

    async def generate(self, prompt: str, system_prompt: str = None,
                       max_tokens: int = 4000) -> str:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": "gpt-4-turbo-preview",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }

            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error = await response.text()
                        logger.error(f"OpenAI API error: {error}")
                        return ""
            except Exception as e:
                logger.error(f"OpenAI generation failed: {e}")
                return ""

    async def analyze(self, content: str, instruction: str) -> Dict:
        prompt = f"{instruction}\n\nContent:\n{content}"
        result = await self.generate(prompt, max_tokens=2000)
        try:
            return json.loads(result)
        except:
            return {"analysis": result}


class GeminiProvider(AIProvider):
    """Google Gemini provider"""

    def __init__(self):
        self.api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def generate(self, prompt: str, system_prompt: str = None,
                       max_tokens: int = 4000) -> str:
        async with aiohttp.ClientSession() as session:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.7
                }
            }

            try:
                async with session.post(
                    f"{self.base_url}/models/gemini-2.0-flash:generateContent?key={self.api_key}",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        error = await response.text()
                        logger.error(f"Gemini API error: {error}")
                        return ""
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                return ""

    async def analyze(self, content: str, instruction: str) -> Dict:
        prompt = f"{instruction}\n\nContent:\n{content}"
        result = await self.generate(prompt, max_tokens=2000)
        try:
            return json.loads(result)
        except:
            return {"analysis": result}


class MultiAIOrchestrator:
    """
    Orchestrates multiple AI models for optimal output quality

    Strategies:
    - Sequential refinement: Model A generates, Model B reviews/improves
    - Parallel generation: Multiple models generate, best is selected
    - Consensus: Multiple models generate, outputs are merged
    - Specialized: Different models for different tasks
    """

    def __init__(self):
        self.claude = ClaudeProvider()
        self.openai = OpenAIProvider()
        self.gemini = GeminiProvider()

        # Model strengths for task assignment
        self.model_strengths = {
            "research": [self.gemini, self.claude],
            "creative_writing": [self.claude, self.openai],
            "technical_writing": [self.claude, self.openai],
            "code_generation": [self.claude, self.openai],
            "summarization": [self.gemini, self.claude],
            "analysis": [self.claude, self.gemini],
            "formatting": [self.openai, self.claude],
            "review": [self.claude, self.openai]
        }

    async def generate_with_refinement(self, prompt: str, system_prompt: str = None,
                                        quality_tier: QualityTier = QualityTier.PREMIUM) -> str:
        """Generate content with multi-model refinement"""

        if quality_tier == QualityTier.STANDARD:
            # Single model generation
            return await self.claude.generate(prompt, system_prompt)

        elif quality_tier == QualityTier.PREMIUM:
            # Generate with Claude, refine with GPT
            initial = await self.claude.generate(prompt, system_prompt)

            refine_prompt = f"""Review and improve this content. Fix any issues, enhance clarity,
            and ensure professional quality. Maintain the original intent and structure.

            Content to improve:
            {initial}

            Return only the improved content, no commentary."""

            refined = await self.openai.generate(refine_prompt)
            return refined if refined else initial

        else:  # ULTIMATE
            # Full multi-model pipeline
            # Step 1: Research/outline with Gemini
            research_prompt = f"Create a detailed research outline for: {prompt}"
            outline = await self.gemini.generate(research_prompt, system_prompt)

            # Step 2: Generate with Claude
            generation_prompt = f"""Using this research outline:
            {outline}

            Generate comprehensive content for: {prompt}"""
            generated = await self.claude.generate(generation_prompt, system_prompt, max_tokens=8000)

            # Step 3: Review and enhance with GPT
            review_prompt = f"""You are a professional editor. Review and enhance this content:

            {generated}

            1. Fix any factual errors or inconsistencies
            2. Improve flow and readability
            3. Enhance examples and explanations
            4. Ensure professional quality throughout

            Return the fully enhanced content."""

            final = await self.openai.generate(review_prompt, max_tokens=8000)
            return final if final else generated

    async def generate_parallel_best(self, prompt: str, system_prompt: str = None) -> str:
        """Generate with multiple models in parallel, select best"""

        results = await asyncio.gather(
            self.claude.generate(prompt, system_prompt),
            self.openai.generate(prompt, system_prompt),
            self.gemini.generate(prompt, system_prompt),
            return_exceptions=True
        )

        valid_results = [r for r in results if isinstance(r, str) and len(r) > 100]

        if not valid_results:
            return ""

        if len(valid_results) == 1:
            return valid_results[0]

        # Score and select best
        scores = []
        for result in valid_results:
            score = await self._score_content(result)
            scores.append(score)

        best_idx = scores.index(max(scores))
        return valid_results[best_idx]

    async def _score_content(self, content: str) -> float:
        """Score content quality (0-100)"""
        # Quick heuristic scoring
        score = 50.0

        # Length bonus (up to 20 points)
        word_count = len(content.split())
        score += min(20, word_count / 100)

        # Structure bonus (up to 15 points)
        if "##" in content or "**" in content:
            score += 10
        if "\n\n" in content:
            score += 5

        # Quality indicators (up to 15 points)
        quality_phrases = ["for example", "specifically", "importantly", "in conclusion",
                          "first", "second", "third", "finally", "however", "therefore"]
        for phrase in quality_phrases:
            if phrase.lower() in content.lower():
                score += 1.5

        return min(100, score)


class ProductGenerator:
    """
    Main product generation engine

    Generates complete digital products with:
    - Multi-AI content generation
    - Automatic structuring and formatting
    - Quality assurance
    - Asset generation (covers, graphics, etc.)
    """

    def __init__(self):
        self.db_url = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL')
        self.orchestrator = MultiAIOrchestrator()
        self._initialized = False

        # Product-specific generators
        self.generators = {
            ProductType.EBOOK: self._generate_ebook,
            ProductType.GUIDE: self._generate_guide,
            ProductType.TEMPLATE_BUSINESS: self._generate_business_template,
            ProductType.TEMPLATE_CODE: self._generate_code_template,
            ProductType.CHECKLIST: self._generate_checklist,
            ProductType.COURSE: self._generate_course,
            ProductType.SOP: self._generate_sop,
            ProductType.PLAYBOOK: self._generate_playbook,
            ProductType.EMAIL_SEQUENCE: self._generate_email_sequence,
            ProductType.PROMPT_PACK: self._generate_prompt_pack,
            ProductType.MICRO_TOOL: self._generate_micro_tool,
        }

    def _get_connection(self):
        if not self.db_url:
            raise ValueError("DATABASE_URL not configured")
        return psycopg2.connect(self.db_url)

    async def initialize_tables(self):
        """Create product generation tables"""
        if self._initialized:
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Products table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS generated_products (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        product_type VARCHAR(50) NOT NULL,
                        title VARCHAR(500) NOT NULL,
                        description TEXT,
                        spec JSONB DEFAULT '{}',
                        content JSONB DEFAULT '{}',
                        assets JSONB DEFAULT '[]',
                        status VARCHAR(50) DEFAULT 'queued',
                        quality_score FLOAT,
                        word_count INT,
                        models_used JSONB DEFAULT '[]',
                        tokens_used INT DEFAULT 0,
                        cost_estimate FLOAT DEFAULT 0,
                        generation_time_seconds FLOAT,
                        errors JSONB DEFAULT '[]',
                        published BOOLEAN DEFAULT false,
                        publish_url TEXT,
                        revenue_generated FLOAT DEFAULT 0,
                        downloads INT DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        completed_at TIMESTAMPTZ,
                        published_at TIMESTAMPTZ
                    )
                """)

                # Generation queue
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS product_generation_queue (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        product_id UUID REFERENCES generated_products(id),
                        priority INT DEFAULT 5,
                        scheduled_at TIMESTAMPTZ DEFAULT NOW(),
                        started_at TIMESTAMPTZ,
                        completed_at TIMESTAMPTZ,
                        status VARCHAR(50) DEFAULT 'pending',
                        worker_id VARCHAR(100),
                        retry_count INT DEFAULT 0,
                        max_retries INT DEFAULT 3,
                        error_message TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Product templates
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS product_templates (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        product_type VARCHAR(50) NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        template_content JSONB DEFAULT '{}',
                        variables JSONB DEFAULT '[]',
                        industry VARCHAR(100),
                        quality_tier VARCHAR(50) DEFAULT 'premium',
                        usage_count INT DEFAULT 0,
                        avg_quality_score FLOAT,
                        active BOOLEAN DEFAULT true,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Revenue tracking
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS product_revenue (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        product_id UUID REFERENCES generated_products(id),
                        revenue_type VARCHAR(50),
                        amount FLOAT NOT NULL,
                        currency VARCHAR(10) DEFAULT 'USD',
                        source VARCHAR(100),
                        customer_id UUID,
                        transaction_date TIMESTAMPTZ DEFAULT NOW(),
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Create indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_products_type ON generated_products(product_type)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_products_status ON generated_products(status)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON product_generation_queue(status)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_revenue_product ON product_revenue(product_id)")

                conn.commit()
                self._initialized = True
                logger.info("Product generation tables initialized")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize tables: {e}")
            raise
        finally:
            conn.close()

    async def generate_product(self, spec: ProductSpec) -> GenerationResult:
        """
        Generate a complete digital product
        """
        await self.initialize_tables()

        start_time = datetime.utcnow()
        result = GenerationResult(
            product_id=spec.product_id,
            status=ProductStatus.GENERATING
        )

        try:
            # Get the appropriate generator
            generator = self.generators.get(spec.product_type)
            if not generator:
                generator = self._generate_generic

            # Update status
            await self._update_status(spec.product_id, ProductStatus.GENERATING)

            # Generate the product
            content = await generator(spec)

            # Quality review
            await self._update_status(spec.product_id, ProductStatus.REVIEWING)
            quality_score = await self._quality_review(content, spec)

            # Finalize
            await self._update_status(spec.product_id, ProductStatus.FINALIZING)

            # Calculate metrics
            generation_time = (datetime.utcnow() - start_time).total_seconds()

            result.status = ProductStatus.COMPLETED
            result.content = content
            result.quality_score = quality_score
            result.generation_time_seconds = generation_time
            result.models_used = ["claude-sonnet-4", "gpt-4-turbo", "gemini-2.0-flash"]

            # Store result
            await self._store_result(spec, result)

            return result

        except Exception as e:
            logger.error(f"Product generation failed: {e}")
            result.status = ProductStatus.FAILED
            result.errors.append(str(e))
            return result

    async def _generate_ebook(self, spec: ProductSpec) -> Dict:
        """Generate a complete eBook"""

        # Step 1: Generate outline
        outline_prompt = f"""Create a comprehensive outline for an eBook titled: "{spec.title}"

        Target audience: {spec.target_audience}
        Industry: {spec.industry}
        Tone: {spec.tone}
        Target length: {spec.word_count_target} words

        Include:
        - Introduction chapter
        - 5-8 main chapters with 3-5 sections each
        - Conclusion chapter
        - Appendix suggestions

        For each chapter, include:
        - Chapter title
        - Key topics to cover
        - Practical examples to include
        - Actionable takeaways

        Format as JSON with structure:
        {{
            "title": "...",
            "subtitle": "...",
            "chapters": [
                {{
                    "number": 1,
                    "title": "...",
                    "sections": ["...", "..."],
                    "examples": ["..."],
                    "takeaways": ["..."]
                }}
            ]
        }}"""

        outline_raw = await self.orchestrator.claude.generate(outline_prompt)

        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', outline_raw)
            outline = json.loads(json_match.group()) if json_match else {"chapters": []}
        except:
            outline = {"title": spec.title, "chapters": []}

        # Step 2: Generate each chapter
        chapters = []
        for chapter in outline.get("chapters", [])[:8]:
            chapter_prompt = f"""Write a comprehensive chapter for an eBook.

            Book Title: {spec.title}
            Chapter {chapter.get('number', 1)}: {chapter.get('title', 'Chapter')}

            Sections to cover:
            {json.dumps(chapter.get('sections', []), indent=2)}

            Include these examples:
            {json.dumps(chapter.get('examples', []), indent=2)}

            Target audience: {spec.target_audience}
            Tone: {spec.tone}
            Target length: {spec.word_count_target // 8} words

            Write in a professional, engaging style with:
            - Clear section headers (use ## for sections)
            - Practical examples and case studies
            - Actionable advice
            - Key takeaways at the end

            Write the complete chapter content now:"""

            chapter_content = await self.orchestrator.generate_with_refinement(
                chapter_prompt,
                quality_tier=spec.quality_tier
            )

            chapters.append({
                "number": chapter.get("number", len(chapters) + 1),
                "title": chapter.get("title", f"Chapter {len(chapters) + 1}"),
                "content": chapter_content,
                "word_count": len(chapter_content.split())
            })

        # Step 3: Generate introduction and conclusion
        intro_prompt = f"""Write a compelling introduction for the eBook "{spec.title}".

        The book covers: {spec.description}
        Target audience: {spec.target_audience}

        Include:
        - Hook to grab attention
        - Why this topic matters
        - What readers will learn
        - How to use this book
        - Brief chapter overview

        Length: 500-800 words"""

        introduction = await self.orchestrator.generate_with_refinement(intro_prompt, quality_tier=spec.quality_tier)

        conclusion_prompt = f"""Write a powerful conclusion for the eBook "{spec.title}".

        Include:
        - Summary of key insights
        - Call to action
        - Next steps for readers
        - Final inspirational message

        Length: 400-600 words"""

        conclusion = await self.orchestrator.generate_with_refinement(conclusion_prompt, quality_tier=spec.quality_tier)

        # Compile final eBook
        return {
            "type": "ebook",
            "title": spec.title,
            "subtitle": outline.get("subtitle", ""),
            "description": spec.description,
            "target_audience": spec.target_audience,
            "introduction": introduction,
            "chapters": chapters,
            "conclusion": conclusion,
            "total_word_count": sum(c["word_count"] for c in chapters) + len(introduction.split()) + len(conclusion.split()),
            "metadata": {
                "industry": spec.industry,
                "keywords": spec.keywords,
                "quality_tier": spec.quality_tier.value,
                "generated_at": datetime.utcnow().isoformat()
            }
        }

    async def _generate_guide(self, spec: ProductSpec) -> Dict:
        """Generate a comprehensive guide"""

        prompt = f"""Create a comprehensive, actionable guide on: "{spec.title}"

        Description: {spec.description}
        Target audience: {spec.target_audience}
        Industry: {spec.industry}

        Structure the guide with:
        1. Executive Summary (200 words)
        2. Why This Matters (context and importance)
        3. Step-by-Step Process (numbered, detailed steps)
        4. Best Practices and Tips
        5. Common Mistakes to Avoid
        6. Tools and Resources
        7. Case Study or Example
        8. Quick Reference Checklist
        9. Next Steps and Action Items

        Make it practical, specific, and immediately actionable.
        Include real examples and specific recommendations.
        Target length: {spec.word_count_target} words

        Format with clear headers using ## for sections."""

        content = await self.orchestrator.generate_with_refinement(prompt, quality_tier=spec.quality_tier)

        return {
            "type": "guide",
            "title": spec.title,
            "content": content,
            "word_count": len(content.split()),
            "metadata": spec.metadata
        }

    async def _generate_business_template(self, spec: ProductSpec) -> Dict:
        """Generate business templates (proposals, contracts, etc.)"""

        prompt = f"""Create a professional business template for: "{spec.title}"

        Description: {spec.description}
        Industry: {spec.industry}
        Target users: {spec.target_audience}

        Include:
        1. Template with placeholder variables [VARIABLE_NAME]
        2. Instructions for each section
        3. Example filled-in version
        4. Customization guide
        5. Best practices for using this template

        Make it comprehensive, professional, and immediately usable.
        Use industry-standard formatting and language."""

        content = await self.orchestrator.generate_with_refinement(prompt, quality_tier=spec.quality_tier)

        return {
            "type": "template_business",
            "title": spec.title,
            "template": content,
            "variables": self._extract_variables(content),
            "metadata": spec.metadata
        }

    async def _generate_code_template(self, spec: ProductSpec) -> Dict:
        """Generate code templates and boilerplates"""

        prompt = f"""Create a professional code template/boilerplate for: "{spec.title}"

        Description: {spec.description}
        Target developers: {spec.target_audience}

        Include:
        1. Main code files with comprehensive comments
        2. Configuration files
        3. README.md with setup instructions
        4. Example usage code
        5. Tests (if applicable)
        6. Best practices documentation

        Follow industry best practices:
        - Clean, readable code
        - Proper error handling
        - Type hints/annotations where applicable
        - Security considerations
        - Performance optimizations

        {spec.custom_instructions}"""

        content = await self.orchestrator.generate_with_refinement(prompt, quality_tier=spec.quality_tier)

        return {
            "type": "template_code",
            "title": spec.title,
            "files": self._parse_code_files(content),
            "metadata": spec.metadata
        }

    async def _generate_checklist(self, spec: ProductSpec) -> Dict:
        """Generate comprehensive checklists"""

        prompt = f"""Create a comprehensive, actionable checklist for: "{spec.title}"

        Description: {spec.description}
        Target users: {spec.target_audience}
        Industry: {spec.industry}

        Structure:
        1. Group items by category/phase
        2. Each item should be specific and actionable
        3. Include sub-items where needed
        4. Add notes/tips for complex items
        5. Include estimated time for each section

        Format:
        ## Category Name (Estimated time: X hours)
        - [ ] Specific action item
          - Note: Additional context
        - [ ] Another action item

        Make it thorough - leave nothing out!"""

        content = await self.orchestrator.generate_with_refinement(prompt, quality_tier=spec.quality_tier)

        return {
            "type": "checklist",
            "title": spec.title,
            "content": content,
            "items": self._parse_checklist_items(content),
            "metadata": spec.metadata
        }

    async def _generate_course(self, spec: ProductSpec) -> Dict:
        """Generate a complete course curriculum"""

        # Generate course structure
        structure_prompt = f"""Design a comprehensive online course curriculum for: "{spec.title}"

        Description: {spec.description}
        Target students: {spec.target_audience}

        Create:
        1. Course overview and learning objectives
        2. 6-10 modules with:
           - Module title and description
           - 3-5 lessons per module
           - Learning objectives per lesson
           - Practical exercises
           - Quiz questions
        3. Final project/assessment
        4. Additional resources

        Format as JSON:
        {{
            "title": "...",
            "description": "...",
            "objectives": ["..."],
            "modules": [
                {{
                    "number": 1,
                    "title": "...",
                    "description": "...",
                    "lessons": [
                        {{
                            "title": "...",
                            "objectives": ["..."],
                            "duration_minutes": 30
                        }}
                    ],
                    "exercises": ["..."],
                    "quiz": ["..."]
                }}
            ]
        }}"""

        structure_raw = await self.orchestrator.claude.generate(structure_prompt)

        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', structure_raw)
            structure = json.loads(json_match.group()) if json_match else {}
        except:
            structure = {"title": spec.title, "modules": []}

        # Generate content for each lesson
        modules_with_content = []
        for module in structure.get("modules", [])[:8]:
            lessons_with_content = []
            for lesson in module.get("lessons", [])[:5]:
                lesson_prompt = f"""Write comprehensive lesson content for:

                Course: {spec.title}
                Module: {module.get('title', 'Module')}
                Lesson: {lesson.get('title', 'Lesson')}
                Objectives: {lesson.get('objectives', [])}

                Include:
                - Introduction and context
                - Main content with examples
                - Key concepts explained clearly
                - Practical tips
                - Summary and key takeaways

                Target length: 1000-1500 words
                Tone: Educational but engaging"""

                lesson_content = await self.orchestrator.generate_with_refinement(
                    lesson_prompt,
                    quality_tier=QualityTier.PREMIUM
                )

                lessons_with_content.append({
                    **lesson,
                    "content": lesson_content
                })

            modules_with_content.append({
                **module,
                "lessons": lessons_with_content
            })

        return {
            "type": "course",
            "title": spec.title,
            "description": structure.get("description", spec.description),
            "objectives": structure.get("objectives", []),
            "modules": modules_with_content,
            "metadata": spec.metadata
        }

    async def _generate_sop(self, spec: ProductSpec) -> Dict:
        """Generate Standard Operating Procedure"""

        prompt = f"""Create a comprehensive Standard Operating Procedure (SOP) for: "{spec.title}"

        Description: {spec.description}
        Organization/Industry: {spec.industry}
        Target users: {spec.target_audience}

        Include all standard SOP sections:

        1. **Document Control**
           - SOP Number, Version, Effective Date
           - Author, Reviewer, Approver

        2. **Purpose**
           - Why this SOP exists
           - What it accomplishes

        3. **Scope**
           - What's covered
           - What's NOT covered
           - Who this applies to

        4. **Definitions**
           - Key terms and acronyms

        5. **Responsibilities**
           - Roles involved
           - Who does what

        6. **Procedure** (DETAILED)
           - Step-by-step instructions
           - Decision points with flowchart logic
           - Screenshots/diagram placeholders
           - Troubleshooting for each step

        7. **Quality Control**
           - Verification steps
           - Acceptance criteria

        8. **Documentation**
           - Records to maintain
           - Forms to complete

        9. **References**
           - Related documents
           - Regulatory requirements

        10. **Revision History**

        Make it thorough enough that someone new could follow it perfectly."""

        content = await self.orchestrator.generate_with_refinement(prompt, quality_tier=spec.quality_tier)

        return {
            "type": "sop",
            "title": spec.title,
            "content": content,
            "version": "1.0",
            "metadata": spec.metadata
        }

    async def _generate_playbook(self, spec: ProductSpec) -> Dict:
        """Generate strategic playbook"""

        prompt = f"""Create a comprehensive strategic playbook for: "{spec.title}"

        Description: {spec.description}
        Industry: {spec.industry}
        Target audience: {spec.target_audience}

        Structure:

        1. **Executive Overview**
           - Purpose and goals
           - Success metrics
           - Timeline overview

        2. **Strategic Framework**
           - Core principles
           - Key strategies
           - Competitive positioning

        3. **Tactical Plays** (Create 8-12 specific plays)
           For each play:
           - Play name and objective
           - When to use it
           - Step-by-step execution
           - Resources needed
           - Expected outcomes
           - Success indicators
           - Common pitfalls

        4. **Implementation Guide**
           - Rollout phases
           - Team structure
           - Resource allocation

        5. **Measurement & Optimization**
           - KPIs to track
           - Review cadence
           - Iteration process

        6. **Templates & Tools**
           - Ready-to-use templates
           - Recommended tools

        7. **Case Studies**
           - Example scenarios
           - Lessons learned

        Make it actionable and immediately implementable."""

        content = await self.orchestrator.generate_with_refinement(prompt, quality_tier=spec.quality_tier)

        return {
            "type": "playbook",
            "title": spec.title,
            "content": content,
            "metadata": spec.metadata
        }

    async def _generate_email_sequence(self, spec: ProductSpec) -> Dict:
        """Generate email marketing sequence"""

        prompt = f"""Create a high-converting email sequence for: "{spec.title}"

        Description: {spec.description}
        Target audience: {spec.target_audience}
        Industry: {spec.industry}

        Create a 7-email sequence with:

        For EACH email provide:
        1. **Subject Line** (3 A/B test variants)
        2. **Preview Text**
        3. **Email Body** (full copy)
        4. **Call to Action**
        5. **Send Timing** (day and time recommendation)
        6. **Goal of this email**
        7. **Psychological triggers used**

        Sequence structure:
        - Email 1: Welcome/Hook (Day 0)
        - Email 2: Value/Education (Day 1)
        - Email 3: Story/Social Proof (Day 3)
        - Email 4: Problem Agitation (Day 5)
        - Email 5: Solution Introduction (Day 7)
        - Email 6: Objection Handling (Day 9)
        - Email 7: Urgency/Close (Day 11)

        Make each email:
        - Personally written style
        - Mobile-optimized (short paragraphs)
        - Single clear CTA
        - Value-first approach"""

        content = await self.orchestrator.generate_with_refinement(prompt, quality_tier=spec.quality_tier)

        return {
            "type": "email_sequence",
            "title": spec.title,
            "emails": self._parse_email_sequence(content),
            "raw_content": content,
            "metadata": spec.metadata
        }

    async def _generate_prompt_pack(self, spec: ProductSpec) -> Dict:
        """Generate AI prompt collection"""

        prompt = f"""Create a comprehensive AI prompt pack for: "{spec.title}"

        Description: {spec.description}
        Target users: {spec.target_audience}
        Industry: {spec.industry}

        Create 25+ carefully crafted prompts organized by category.

        For EACH prompt include:
        1. **Prompt Name** (descriptive)
        2. **Category** (e.g., Writing, Analysis, Strategy, Creative)
        3. **Use Case** (when to use this)
        4. **The Prompt** (complete, copy-paste ready)
        5. **Variables to customize** [MARKED_LIKE_THIS]
        6. **Expected Output** (what you'll get)
        7. **Pro Tips** (how to get best results)

        Categories to include:
        - Content Creation (5+ prompts)
        - Analysis & Research (5+ prompts)
        - Strategy & Planning (5+ prompts)
        - Problem Solving (5+ prompts)
        - Communication (5+ prompts)

        Make each prompt:
        - Detailed and specific
        - Immediately usable
        - Optimized for best AI output
        - Include system prompt if needed"""

        content = await self.orchestrator.generate_with_refinement(prompt, quality_tier=spec.quality_tier)

        return {
            "type": "prompt_pack",
            "title": spec.title,
            "prompts": self._parse_prompts(content),
            "raw_content": content,
            "metadata": spec.metadata
        }

    async def _generate_micro_tool(self, spec: ProductSpec) -> Dict:
        """Generate a micro-tool or calculator"""

        prompt = f"""Create a complete micro-tool/calculator for: "{spec.title}"

        Description: {spec.description}
        Target users: {spec.target_audience}

        Provide:

        1. **Tool Specification**
           - Inputs (with validation rules)
           - Calculations/Logic
           - Outputs

        2. **HTML/CSS/JS Code**
           - Complete, standalone implementation
           - Beautiful, modern UI
           - Mobile responsive
           - No external dependencies

        3. **Usage Instructions**

        4. **Embed Code** (for sharing)

        Make it:
        - Visually appealing (use gradients, shadows, modern styling)
        - User-friendly with clear labels
        - Include input validation
        - Show results clearly

        {spec.custom_instructions}"""

        content = await self.orchestrator.generate_with_refinement(prompt, quality_tier=spec.quality_tier)

        return {
            "type": "micro_tool",
            "title": spec.title,
            "code": self._extract_code_blocks(content),
            "instructions": content,
            "metadata": spec.metadata
        }

    async def _generate_generic(self, spec: ProductSpec) -> Dict:
        """Fallback generic generator"""

        prompt = f"""Create comprehensive content for: "{spec.title}"

        Type: {spec.product_type.value}
        Description: {spec.description}
        Target audience: {spec.target_audience}
        Industry: {spec.industry}

        Requirements:
        - Professional quality
        - Well-structured with clear sections
        - Actionable and valuable
        - Target length: {spec.word_count_target} words

        {spec.custom_instructions}"""

        content = await self.orchestrator.generate_with_refinement(prompt, quality_tier=spec.quality_tier)

        return {
            "type": spec.product_type.value,
            "title": spec.title,
            "content": content,
            "metadata": spec.metadata
        }

    async def _quality_review(self, content: Dict, spec: ProductSpec) -> float:
        """Review and score content quality"""

        review_prompt = f"""Evaluate this content for quality (score 0-100):

        Title: {spec.title}
        Type: {spec.product_type.value}
        Target Audience: {spec.target_audience}

        Content preview: {str(content)[:3000]}

        Score on:
        - Completeness (0-20): Does it cover the topic thoroughly?
        - Accuracy (0-20): Is the information correct and reliable?
        - Clarity (0-20): Is it easy to understand?
        - Actionability (0-20): Can readers apply this immediately?
        - Professionalism (0-20): Is it polished and well-formatted?

        Return JSON: {{"total_score": X, "completeness": X, "accuracy": X, "clarity": X, "actionability": X, "professionalism": X, "feedback": "..."}}"""

        review_raw = await self.orchestrator.claude.generate(review_prompt, max_tokens=500)

        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', review_raw)
            review = json.loads(json_match.group()) if json_match else {}
            return review.get("total_score", 75)
        except:
            return 75.0  # Default score

    async def _update_status(self, product_id: str, status: ProductStatus):
        """Update product status in database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE generated_products SET status = %s WHERE id = %s
                """, (status.value, product_id))
                conn.commit()
        except:
            pass
        finally:
            conn.close()

    async def _store_result(self, spec: ProductSpec, result: GenerationResult):
        """Store generation result"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE generated_products SET
                        content = %s,
                        status = %s,
                        quality_score = %s,
                        word_count = %s,
                        models_used = %s,
                        generation_time_seconds = %s,
                        completed_at = NOW()
                    WHERE id = %s
                """, (
                    json.dumps(result.content),
                    result.status.value,
                    result.quality_score,
                    result.content.get("word_count", result.content.get("total_word_count", 0)),
                    json.dumps(result.models_used),
                    result.generation_time_seconds,
                    spec.product_id
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store result: {e}")
        finally:
            conn.close()

    def _extract_variables(self, content: str) -> List[str]:
        """Extract template variables from content"""
        import re
        return list(set(re.findall(r'\[([A-Z_]+)\]', content)))

    def _parse_code_files(self, content: str) -> List[Dict]:
        """Parse code blocks into files"""
        import re
        files = []
        pattern = r'```(\w+)?\n([\s\S]*?)```'
        matches = re.findall(pattern, content)
        for i, (lang, code) in enumerate(matches):
            files.append({
                "filename": f"file_{i+1}.{lang or 'txt'}",
                "language": lang or "text",
                "content": code.strip()
            })
        return files

    def _parse_checklist_items(self, content: str) -> List[Dict]:
        """Parse checklist items"""
        import re
        items = []
        pattern = r'- \[[ x]\] (.+)'
        matches = re.findall(pattern, content)
        for match in matches:
            items.append({"item": match, "completed": False})
        return items

    def _parse_email_sequence(self, content: str) -> List[Dict]:
        """Parse email sequence"""
        # Simple parsing - in production would be more sophisticated
        emails = []
        parts = content.split("Email ")
        for part in parts[1:]:
            emails.append({"content": part.strip()})
        return emails

    def _parse_prompts(self, content: str) -> List[Dict]:
        """Parse prompts from content"""
        prompts = []
        parts = content.split("**Prompt")
        for part in parts[1:]:
            prompts.append({"content": part.strip()})
        return prompts

    def _extract_code_blocks(self, content: str) -> Dict:
        """Extract code blocks"""
        import re
        code = {}
        pattern = r'```(\w+)?\n([\s\S]*?)```'
        matches = re.findall(pattern, content)
        for lang, block in matches:
            if lang not in code:
                code[lang or 'text'] = []
            code[lang or 'text'].append(block.strip())
        return code


# =====================================
# PRODUCT PIPELINE SCHEDULER
# =====================================

class ProductPipelineScheduler:
    """
    Automated product generation pipeline scheduler

    Continuously generates products based on:
    - Market demand signals
    - Content calendar
    - Revenue optimization
    - Gap analysis
    """

    def __init__(self):
        self.generator = ProductGenerator()
        self.db_url = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL')

    def _get_connection(self):
        return psycopg2.connect(self.db_url)

    async def schedule_product(self, spec: ProductSpec, priority: int = 5,
                                scheduled_at: datetime = None) -> str:
        """Schedule a product for generation"""
        await self.generator.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Create product record
                cur.execute("""
                    INSERT INTO generated_products (
                        id, product_type, title, description, spec, status
                    ) VALUES (%s, %s, %s, %s, %s, 'queued')
                    RETURNING id
                """, (
                    spec.product_id,
                    spec.product_type.value,
                    spec.title,
                    spec.description,
                    json.dumps({
                        "target_audience": spec.target_audience,
                        "quality_tier": spec.quality_tier.value,
                        "word_count_target": spec.word_count_target,
                        "style": spec.style,
                        "tone": spec.tone,
                        "industry": spec.industry,
                        "keywords": spec.keywords,
                        "custom_instructions": spec.custom_instructions
                    })
                ))

                # Add to queue
                cur.execute("""
                    INSERT INTO product_generation_queue (
                        product_id, priority, scheduled_at
                    ) VALUES (%s, %s, %s)
                """, (
                    spec.product_id,
                    priority,
                    scheduled_at or datetime.utcnow()
                ))

                conn.commit()
                return spec.product_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to schedule product: {e}")
            raise
        finally:
            conn.close()

    async def process_queue(self, max_items: int = 5):
        """Process pending items in the queue"""
        await self.generator.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get pending items
                cur.execute("""
                    SELECT q.*, p.spec, p.product_type, p.title, p.description
                    FROM product_generation_queue q
                    JOIN generated_products p ON q.product_id = p.id
                    WHERE q.status = 'pending'
                    AND q.scheduled_at <= NOW()
                    ORDER BY q.priority DESC, q.scheduled_at ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                """, (max_items,))

                items = cur.fetchall()

                for item in items:
                    # Mark as processing
                    cur.execute("""
                        UPDATE product_generation_queue
                        SET status = 'processing', started_at = NOW()
                        WHERE id = %s
                    """, (item['id'],))
                    conn.commit()

                    try:
                        # Build spec
                        spec_data = item['spec'] if isinstance(item['spec'], dict) else json.loads(item['spec'])
                        spec = ProductSpec(
                            product_id=str(item['product_id']),
                            product_type=ProductType(item['product_type']),
                            title=item['title'],
                            description=item['description'],
                            target_audience=spec_data.get('target_audience', 'general'),
                            quality_tier=QualityTier(spec_data.get('quality_tier', 'premium')),
                            word_count_target=spec_data.get('word_count_target', 5000),
                            style=spec_data.get('style', 'professional'),
                            tone=spec_data.get('tone', 'authoritative'),
                            industry=spec_data.get('industry', 'general'),
                            keywords=spec_data.get('keywords', []),
                            custom_instructions=spec_data.get('custom_instructions', '')
                        )

                        # Generate
                        result = await self.generator.generate_product(spec)

                        # Update queue
                        cur.execute("""
                            UPDATE product_generation_queue
                            SET status = %s, completed_at = NOW()
                            WHERE id = %s
                        """, ('completed' if result.status == ProductStatus.COMPLETED else 'failed', item['id']))
                        conn.commit()

                    except Exception as e:
                        logger.error(f"Failed to process queue item: {e}")
                        cur.execute("""
                            UPDATE product_generation_queue
                            SET status = 'failed', error_message = %s,
                                retry_count = retry_count + 1
                            WHERE id = %s
                        """, (str(e), item['id']))
                        conn.commit()

        finally:
            conn.close()

    async def get_queue_status(self) -> Dict:
        """Get queue status"""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        status,
                        COUNT(*) as count
                    FROM product_generation_queue
                    GROUP BY status
                """)
                by_status = {row['status']: row['count'] for row in cur.fetchall()}

                cur.execute("""
                    SELECT
                        p.product_type,
                        COUNT(*) as count
                    FROM product_generation_queue q
                    JOIN generated_products p ON q.product_id = p.id
                    WHERE q.status = 'pending'
                    GROUP BY p.product_type
                """)
                by_type = {row['product_type']: row['count'] for row in cur.fetchall()}

                return {
                    "by_status": by_status,
                    "pending_by_type": by_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
        finally:
            conn.close()


# =====================================
# SINGLETON INSTANCES
# =====================================

_product_generator = None
_pipeline_scheduler = None

def get_product_generator() -> ProductGenerator:
    global _product_generator
    if _product_generator is None:
        _product_generator = ProductGenerator()
    return _product_generator

def get_pipeline_scheduler() -> ProductPipelineScheduler:
    global _pipeline_scheduler
    if _pipeline_scheduler is None:
        _pipeline_scheduler = ProductPipelineScheduler()
    return _pipeline_scheduler

#!/usr/bin/env python3
"""
Multi-AI Content Orchestrator - Complete Content Production System
==================================================================
Enterprise-grade content factory using orchestrated AI models for:
- Blog posts (existing)
- Newsletters (new)
- Ebooks (new)
- Training documentation (new)
- Social media content (new)
- Email sequences (new)

Pipeline uses 5+ AI models optimally:
- Perplexity: Real-time research
- Claude: Long-form writing, training docs
- Gemini: Visual design, structure analysis
- GPT-4: QA, editing, fact-checking
- DALL-E/Imagen: Image generation
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field

import psycopg2
from psycopg2.extras import RealDictCursor
import httpx

from ai_core import ai_core
from content_generation_agent import ContentGeneratorAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Supported content types"""
    BLOG_POST = "blog_post"
    NEWSLETTER = "newsletter"
    EBOOK_CHAPTER = "ebook_chapter"
    EBOOK_FULL = "ebook_full"
    TRAINING_DOC = "training_doc"
    SOCIAL_POST = "social_post"
    EMAIL_SEQUENCE = "email_sequence"
    LANDING_PAGE = "landing_page"
    CASE_STUDY = "case_study"
    WHITE_PAPER = "white_paper"


@dataclass
class ContentPiece:
    """Base content output"""
    id: str = ""
    content_type: str = ""
    title: str = ""
    content: str = ""
    excerpt: str = ""
    metadata: dict = field(default_factory=dict)
    models_used: list = field(default_factory=list)
    quality_score: int = 0
    created_at: str = ""
    status: str = "draft"


@dataclass
class Newsletter:
    """Newsletter structure"""
    id: str = ""
    subject: str = ""
    preview_text: str = ""
    header_content: str = ""
    main_story: str = ""
    secondary_stories: list = field(default_factory=list)
    tips_section: str = ""
    call_to_action: str = ""
    footer_content: str = ""
    html_template: str = ""
    plain_text: str = ""
    target_audience: str = ""
    send_date: str = ""


@dataclass
class Ebook:
    """Ebook structure"""
    id: str = ""
    title: str = ""
    subtitle: str = ""
    author: str = ""
    chapters: list = field(default_factory=list)
    table_of_contents: str = ""
    introduction: str = ""
    conclusion: str = ""
    cover_prompt: str = ""
    target_audience: str = ""
    word_count: int = 0
    format: str = "markdown"


@dataclass
class TrainingDoc:
    """Training documentation structure"""
    id: str = ""
    title: str = ""
    module_number: int = 0
    objectives: list = field(default_factory=list)
    prerequisites: list = field(default_factory=list)
    content: str = ""
    exercises: list = field(default_factory=list)
    quiz_questions: list = field(default_factory=list)
    key_takeaways: list = field(default_factory=list)
    resources: list = field(default_factory=list)
    estimated_time: str = ""


def _get_db_config():
    """Get database configuration."""
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        match = re.match(
            r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)',
            database_url
        )
        if match:
            return {
                'host': match.group(3),
                'database': match.group(5),
                'user': match.group(1),
                'password': match.group(2),
                'port': int(match.group(4))
            }
    return {
        "host": os.getenv("DB_HOST", ""),
        "database": os.getenv("DB_NAME", ""),
        "user": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
        "port": int(os.getenv("DB_PORT", "5432"))
    }


def _get_db_connection(**kwargs):
    """Get database connection."""
    db_config = _get_db_config()
    db_config.update(kwargs)
    return psycopg2.connect(**db_config)


class MultiAIContentOrchestrator:
    """
    Multi-AI Content Factory

    Orchestrates multiple AI models to produce high-quality content
    across multiple formats: blogs, newsletters, ebooks, training docs.
    """

    _tables_ensured = False

    def __init__(self):
        self.orchestrator_id = str(uuid.uuid4())
        self.blog_agent = ContentGeneratorAgent()
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.models_used = []
        logger.info("Initialized MultiAIContentOrchestrator")

    def _ensure_tables(self):
        """Verify required tables exist (DDL removed — agent_worker has no DDL permissions)."""
        required_tables = [
                "ai_content_library",
                "ai_newsletters",
                "ai_ebooks",
                "ai_training_docs",
        ]
        try:
            from database.verify_tables import verify_tables_sync
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()
            ok = verify_tables_sync(required_tables, cursor, module_name="multi_ai_content_orchestrator")
            cursor.close()
            conn.close()
            if not ok:
                return
            self._tables_initialized = True
        except Exception as exc:
            logger.error("Table verification failed: %s", exc)
    async def generate_newsletter(self, task: dict) -> dict:
        """
        Generate a complete newsletter using multi-AI pipeline.

        Pipeline:
        1. Perplexity: Research current trends/news
        2. Claude: Write main story and secondary pieces
        3. GPT-4: Write engaging subject line and preview
        4. Gemini: Design layout recommendations
        5. GPT-4: Final QA and polish
        """
        self._ensure_tables()
        self.models_used = []

        topic = task.get("topic", "AI and automation trends")
        brand = task.get("brand", "BrainOps")
        target_audience = task.get("target_audience", "tech professionals and business owners")

        logger.info(f"📧 Generating Newsletter: {topic}")

        try:
            # Stage 1: Research trending topics (Perplexity)
            research = await self._research_for_newsletter(topic)

            # Stage 2: Write main story (Claude)
            main_story = await self._write_newsletter_main_story(topic, research, brand)

            # Stage 3: Write secondary content (Claude)
            secondary_stories = await self._write_secondary_stories(research, brand)

            # Stage 4: Generate tips section (GPT-4)
            tips = await self._generate_newsletter_tips(topic, target_audience)

            # Stage 5: Create subject line and preview (GPT-4)
            subject_data = await self._generate_subject_line(main_story, brand)

            # Stage 6: Design recommendations (Gemini)
            design = await self._get_newsletter_design(topic, brand)

            # Stage 7: Generate CTA (Claude)
            cta = await self._generate_newsletter_cta(topic, brand)

            # Compile newsletter
            newsletter = Newsletter(
                id=str(uuid.uuid4()),
                subject=subject_data.get("subject", f"{brand} Weekly: {topic}"),
                preview_text=subject_data.get("preview", ""),
                header_content=f"# {brand} Weekly Insights\n\n*Your weekly dose of AI and automation intelligence*",
                main_story=main_story,
                secondary_stories=secondary_stories,
                tips_section=tips,
                call_to_action=cta,
                footer_content=self._get_newsletter_footer(brand),
                target_audience=target_audience
            )

            # Generate HTML template
            newsletter.html_template = self._generate_newsletter_html(newsletter, design)
            newsletter.plain_text = self._generate_plain_text(newsletter)

            # Save to database
            newsletter_id = await self._save_newsletter(newsletter)

            logger.info(f"✅ Newsletter generated: {newsletter.subject}")

            return {
                "status": "completed",
                "newsletter_id": newsletter_id,
                "subject": newsletter.subject,
                "preview_text": newsletter.preview_text,
                "word_count": len(newsletter.plain_text.split()),
                "models_used": self.models_used,
                "content_type": "newsletter"
            }

        except Exception as e:
            logger.error(f"Newsletter generation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _research_for_newsletter(self, topic: str) -> dict:
        """Research current trends for newsletter content."""
        if not self.perplexity_key:
            return {"trends": [], "news": [], "stats": []}

        try:
            prompt = f"""Research current trends and news about "{topic}" for a business newsletter.

            Return JSON:
            {{
                "trends": ["trend 1", "trend 2", "trend 3"],
                "news": ["news item 1", "news item 2"],
                "stats": ["statistic 1", "statistic 2"],
                "insights": ["insight 1", "insight 2"]
            }}"""

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sonar-pro",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2000
                    },
                    timeout=60.0
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    self.models_used.append("perplexity:sonar-pro")
                    return ai_core._safe_json(content) or {}

        except Exception as e:
            logger.error(f"Newsletter research failed: {e}")

        return {"trends": [], "news": [], "stats": []}

    async def _write_newsletter_main_story(self, topic: str, research: dict, brand: str) -> str:
        """Write the main newsletter story with Claude."""
        research_context = ""
        if research.get("trends"):
            research_context = f"\nCurrent trends: {', '.join(research['trends'][:3])}"
        if research.get("stats"):
            research_context += f"\nKey stats: {', '.join(research['stats'][:2])}"

        prompt = f"""Write the main story for a {brand} newsletter about: {topic}
        {research_context}

        Guidelines:
        - Length: 300-400 words
        - Tone: Professional but engaging, like a smart friend sharing insights
        - Format: Markdown with one H2 heading
        - Include: One actionable insight, one stat/data point, forward-looking perspective
        - CTA hint: Subtle lead to learn more (not salesy)

        Write ONLY the content, no meta commentary."""

        try:
            content = await ai_core.generate(
                prompt,
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                max_tokens=1500,
                system_prompt="You are an expert newsletter writer. Write engaging, valuable content.",
                prefer_anthropic=True
            )
            self.models_used.append("anthropic:claude-3-sonnet")
            return content
        except Exception as e:
            logger.error(f"Main story generation failed: {e}")
            return f"## {topic}\n\nStay tuned for insights on this topic."

    async def _write_secondary_stories(self, research: dict, brand: str) -> list:
        """Write 2-3 shorter secondary stories."""
        stories = []
        trends = research.get("trends", [])[:3]

        for i, trend in enumerate(trends):
            prompt = f"""Write a brief newsletter snippet (100-150 words) about: {trend}

            Format: Short paragraph with one key takeaway.
            Tone: Informative, quick-read style."""

            try:
                content = await ai_core.generate(
                    prompt,
                    model="gpt-4-turbo-preview",
                    temperature=0.7,
                    max_tokens=300
                )
                stories.append({"title": trend, "content": content})
                if i == 0:
                    self.models_used.append("openai:gpt-4-turbo-preview")
            except Exception as e:
                logger.error(f"Secondary story {i} failed: {e}")

        return stories

    async def _generate_newsletter_tips(self, topic: str, audience: str) -> str:
        """Generate actionable tips section."""
        prompt = f"""Create a "Quick Tips" section for a newsletter about {topic}.
        Target audience: {audience}

        Format:
        ## 💡 Quick Tips

        1. **Tip title**: Brief explanation (1-2 sentences)
        2. **Tip title**: Brief explanation
        3. **Tip title**: Brief explanation

        Make tips actionable and immediately useful."""

        try:
            tips = await ai_core.generate(
                prompt,
                model="gpt-4-turbo-preview",
                temperature=0.6,
                max_tokens=500
            )
            return tips
        except Exception as e:
            return "## 💡 Quick Tips\n\n1. Stay curious\n2. Take action\n3. Share what you learn"

    async def _generate_subject_line(self, main_story: str, brand: str) -> dict:
        """Generate compelling subject line and preview text."""
        prompt = f"""Based on this newsletter content, create an email subject line and preview text:

        Content preview: {main_story[:500]}
        Brand: {brand}

        Return JSON:
        {{
            "subject": "compelling subject line (max 50 chars, no emoji spam)",
            "preview": "preview text that appears after subject (max 100 chars)"
        }}

        Make it curiosity-driving but not clickbait."""

        try:
            response = await ai_core.generate(
                prompt,
                model="gpt-4-turbo-preview",
                temperature=0.7,
                max_tokens=200,
                system_prompt="Output only valid JSON."
            )
            return ai_core._safe_json(response) or {"subject": f"{brand} Weekly", "preview": ""}
        except Exception as e:
            return {"subject": f"{brand} Weekly", "preview": "Your weekly insights"}

    async def _get_newsletter_design(self, topic: str, brand: str) -> dict:
        """Get design recommendations from Gemini."""
        prompt = f"""Suggest email newsletter design elements for topic: {topic}
        Brand: {brand}

        Return JSON:
        {{
            "primary_color": "#hex",
            "accent_color": "#hex",
            "header_style": "description",
            "layout_type": "single-column or two-column",
            "image_suggestion": "hero image description"
        }}"""

        try:
            response = await ai_core.generate(
                prompt,
                model="gemini-2.0-flash",
                temperature=0.5,
                max_tokens=500,
                system_prompt="Output only valid JSON."
            )
            self.models_used.append("google:gemini-2.0-flash")
            return ai_core._safe_json(response) or {}
        except Exception as e:
            return {"primary_color": "#0ea5e9", "accent_color": "#8b5cf6"}

    async def _generate_newsletter_cta(self, topic: str, brand: str) -> str:
        """Generate newsletter call-to-action."""
        prompt = f"""Write a compelling but non-pushy call-to-action for a {brand} newsletter about {topic}.

        Format as a short paragraph (2-3 sentences) with a clear next step.
        Include a button text suggestion in [brackets]."""

        try:
            cta = await ai_core.generate(
                prompt,
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                max_tokens=200,
                prefer_anthropic=True
            )
            return cta
        except Exception as e:
            return f"Ready to dive deeper? [Explore {brand}]"

    def _get_newsletter_footer(self, brand: str) -> str:
        """Standard newsletter footer."""
        return f"""---

*You're receiving this because you signed up for {brand} updates.*

[Unsubscribe](#) | [Update Preferences](#) | [View in Browser](#)

© {datetime.now().year} {brand}. All rights reserved.
"""

    def _generate_newsletter_html(self, newsletter: Newsletter, design: dict) -> str:
        """Generate HTML email template."""
        primary = design.get("primary_color", "#0ea5e9")
        accent = design.get("accent_color", "#8b5cf6")

        # Convert markdown to HTML-safe content
        main_content = newsletter.main_story.replace("\n", "<br>")
        tips_content = newsletter.tips_section.replace("\n", "<br>")

        secondary_html = ""
        for story in newsletter.secondary_stories:
            secondary_html += f"""
            <div style="margin: 20px 0; padding: 15px; background: #f8fafc; border-radius: 8px;">
                <h3 style="color: {primary}; margin: 0 0 10px 0;">{story.get('title', '')}</h3>
                <p style="color: #475569; margin: 0;">{story.get('content', '').replace(chr(10), '<br>')}</p>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{newsletter.subject}</title>
</head>
<body style="margin: 0; padding: 0; background: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; background: white; padding: 40px;">
        <!-- Header -->
        <div style="text-align: center; margin-bottom: 30px; border-bottom: 3px solid {primary}; padding-bottom: 20px;">
            <h1 style="color: {primary}; margin: 0; font-size: 28px;">BrainOps Weekly</h1>
            <p style="color: #64748b; margin: 10px 0 0 0;">Your weekly AI & automation insights</p>
        </div>

        <!-- Main Story -->
        <div style="margin-bottom: 30px;">
            {main_content}
        </div>

        <!-- Secondary Stories -->
        <div style="margin-bottom: 30px;">
            <h2 style="color: {accent}; font-size: 20px;">📌 Also This Week</h2>
            {secondary_html}
        </div>

        <!-- Tips -->
        <div style="margin-bottom: 30px; background: linear-gradient(135deg, {primary}10, {accent}10); padding: 20px; border-radius: 12px;">
            {tips_content}
        </div>

        <!-- CTA -->
        <div style="text-align: center; margin: 30px 0; padding: 20px; background: {primary}; border-radius: 12px;">
            <p style="color: white; margin: 0 0 15px 0;">{newsletter.call_to_action.replace(chr(10), '<br>')}</p>
            <a href="https://brainstackstudio.com" style="display: inline-block; background: white; color: {primary}; padding: 12px 30px; border-radius: 6px; text-decoration: none; font-weight: 600;">Explore BrainStack</a>
        </div>

        <!-- Footer -->
        <div style="text-align: center; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 12px;">
            {newsletter.footer_content.replace(chr(10), '<br>')}
        </div>
    </div>
</body>
</html>"""

        return html

    def _generate_plain_text(self, newsletter: Newsletter) -> str:
        """Generate plain text version."""
        text = f"""{newsletter.subject}
{'=' * len(newsletter.subject)}

{newsletter.header_content}

{newsletter.main_story}

---
ALSO THIS WEEK
---

"""
        for story in newsletter.secondary_stories:
            text += f"• {story.get('title', '')}\n{story.get('content', '')}\n\n"

        text += f"""
---
{newsletter.tips_section}

---
{newsletter.call_to_action}

{newsletter.footer_content}
"""
        return text

    async def _save_newsletter(self, newsletter: Newsletter) -> str:
        """Save newsletter to database."""
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_newsletters
                (subject, preview_text, content, html_template, plain_text, target_audience, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                newsletter.subject,
                newsletter.preview_text,
                json.dumps({
                    "header": newsletter.header_content,
                    "main_story": newsletter.main_story,
                    "secondary_stories": newsletter.secondary_stories,
                    "tips": newsletter.tips_section,
                    "cta": newsletter.call_to_action,
                    "footer": newsletter.footer_content
                }),
                newsletter.html_template,
                newsletter.plain_text,
                newsletter.target_audience,
                "draft"
            ))

            newsletter_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

            return str(newsletter_id)

        except Exception as e:
            logger.error(f"Save newsletter failed: {e}")
            raise e

    # ==================== EBOOK GENERATION ====================

    async def generate_ebook(self, task: dict) -> dict:
        """
        Generate a complete ebook using multi-AI pipeline.

        Pipeline:
        1. Claude: Outline and structure
        2. Perplexity: Research for each chapter
        3. Claude: Write chapters
        4. GPT-4: Edit and QA
        5. Gemini: Cover design prompt
        """
        self._ensure_tables()
        self.models_used = []

        topic = task.get("topic", "AI Automation for Business")
        chapter_count = task.get("chapters", 5)
        target_audience = task.get("target_audience", "business professionals")
        author = task.get("author", "BrainOps AI")

        logger.info(f"📚 Generating Ebook: {topic} ({chapter_count} chapters)")

        try:
            # Stage 1: Generate outline (Claude)
            outline = await self._generate_ebook_outline(topic, chapter_count, target_audience)

            # Stage 2: Write introduction (Claude)
            introduction = await self._write_ebook_section(
                "Introduction",
                f"Write an engaging introduction for an ebook about {topic}. 500-700 words.",
                target_audience
            )

            # Stage 3: Write each chapter
            chapters = []
            for i, chapter_info in enumerate(outline.get("chapters", [])[:chapter_count]):
                logger.info(f"  Writing chapter {i+1}: {chapter_info.get('title', f'Chapter {i+1}')}")

                # Research for chapter (Perplexity)
                research = await self._research_for_chapter(chapter_info.get("title", ""))

                # Write chapter (Claude)
                chapter_content = await self._write_ebook_chapter(
                    chapter_info,
                    research,
                    target_audience,
                    i + 1
                )

                chapters.append({
                    "number": i + 1,
                    "title": chapter_info.get("title", f"Chapter {i+1}"),
                    "content": chapter_content,
                    "word_count": len(chapter_content.split())
                })

            # Stage 4: Write conclusion (Claude)
            conclusion = await self._write_ebook_section(
                "Conclusion",
                f"Write a powerful conclusion for an ebook about {topic}. Summarize key points and inspire action. 400-500 words.",
                target_audience
            )

            # Stage 5: Generate cover design prompt (Gemini)
            cover_prompt = await self._generate_cover_prompt(topic, target_audience)

            # Calculate stats
            total_words = len(introduction.split()) + sum(c["word_count"] for c in chapters) + len(conclusion.split())

            # Generate TOC
            toc = self._generate_toc(chapters)

            # Compile ebook
            ebook = Ebook(
                id=str(uuid.uuid4()),
                title=outline.get("title", topic),
                subtitle=outline.get("subtitle", ""),
                author=author,
                chapters=chapters,
                table_of_contents=toc,
                introduction=introduction,
                conclusion=conclusion,
                cover_prompt=cover_prompt,
                target_audience=target_audience,
                word_count=total_words
            )

            # Save to database
            ebook_id = await self._save_ebook(ebook)

            logger.info(f"✅ Ebook generated: {ebook.title} ({total_words} words)")

            return {
                "status": "completed",
                "ebook_id": ebook_id,
                "title": ebook.title,
                "subtitle": ebook.subtitle,
                "chapters": len(chapters),
                "word_count": total_words,
                "estimated_pages": total_words // 250,
                "models_used": list(set(self.models_used)),
                "content_type": "ebook"
            }

        except Exception as e:
            logger.error(f"Ebook generation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _generate_ebook_outline(self, topic: str, chapters: int, audience: str) -> dict:
        """Generate ebook outline with Claude."""
        prompt = f"""Create an outline for an ebook about: {topic}
        Number of chapters: {chapters}
        Target audience: {audience}

        Return JSON:
        {{
            "title": "compelling book title",
            "subtitle": "explanatory subtitle",
            "chapters": [
                {{
                    "title": "Chapter Title",
                    "summary": "2-3 sentence summary",
                    "key_points": ["point 1", "point 2", "point 3"]
                }}
            ]
        }}"""

        try:
            response = await ai_core.generate(
                prompt,
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                max_tokens=2000,
                system_prompt="You are an expert book author. Output only valid JSON.",
                prefer_anthropic=True
            )
            self.models_used.append("anthropic:claude-3-sonnet")
            return ai_core._safe_json(response) or {"chapters": []}
        except Exception as e:
            logger.error(f"Outline generation failed: {e}")
            return {"title": topic, "chapters": [{"title": f"Chapter {i+1}"} for i in range(chapters)]}

    async def _research_for_chapter(self, chapter_title: str) -> dict:
        """Research content for a chapter."""
        if not self.perplexity_key:
            return {}

        try:
            prompt = f"""Research key information about: {chapter_title}

            Return JSON with: facts, examples, best_practices, common_mistakes"""

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={"Authorization": f"Bearer {self.perplexity_key}", "Content-Type": "application/json"},
                    json={"model": "sonar-pro", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500},
                    timeout=60.0
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if "perplexity:sonar-pro" not in self.models_used:
                        self.models_used.append("perplexity:sonar-pro")
                    return ai_core._safe_json(content) or {}
        except Exception as e:
            logger.error(f"Chapter research failed: {e}")
        return {}

    async def _write_ebook_chapter(self, chapter_info: dict, research: dict, audience: str, chapter_num: int) -> str:
        """Write a full ebook chapter."""
        research_context = ""
        if research.get("facts"):
            research_context = f"\nKey facts: {', '.join(research['facts'][:3])}"
        if research.get("examples"):
            research_context += f"\nExamples to include: {', '.join(research['examples'][:2])}"

        prompt = f"""Write Chapter {chapter_num}: {chapter_info.get('title', '')}

        Summary: {chapter_info.get('summary', '')}
        Key points to cover: {', '.join(chapter_info.get('key_points', []))}
        {research_context}
        Target audience: {audience}

        Guidelines:
        - Length: 1500-2000 words
        - Format: Markdown with H2 and H3 headings
        - Include: Real examples, actionable advice, clear explanations
        - Tone: Authoritative but accessible
        - End with: Key takeaways bullet list

        Write the complete chapter content."""

        try:
            content = await ai_core.generate(
                prompt,
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                max_tokens=4000,
                system_prompt="You are an expert author writing educational content.",
                prefer_anthropic=True
            )
            return content
        except Exception as e:
            logger.error(f"Chapter writing failed: {e}")
            return f"# Chapter {chapter_num}: {chapter_info.get('title', '')}\n\nContent generation pending."

    async def _write_ebook_section(self, section_name: str, prompt: str, audience: str) -> str:
        """Write introduction or conclusion."""
        full_prompt = f"{prompt}\n\nTarget audience: {audience}"

        try:
            content = await ai_core.generate(
                full_prompt,
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                max_tokens=1500,
                prefer_anthropic=True
            )
            return content
        except Exception as e:
            return f"# {section_name}\n\nContent pending."

    async def _generate_cover_prompt(self, topic: str, audience: str) -> str:
        """Generate cover design prompt with Gemini."""
        prompt = f"""Create a DALL-E prompt for an ebook cover:
        Topic: {topic}
        Audience: {audience}

        The prompt should describe a professional, modern book cover design.
        Include: style, colors, imagery, typography suggestions."""

        try:
            response = await ai_core.generate(
                prompt,
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=300
            )
            self.models_used.append("google:gemini-2.0-flash")
            return response
        except Exception as e:
            return f"Professional ebook cover for '{topic}', modern minimalist design, tech-inspired, blue and purple gradient"

    def _generate_toc(self, chapters: list) -> str:
        """Generate table of contents."""
        toc = "# Table of Contents\n\n"
        toc += "- Introduction\n"
        for chapter in chapters:
            toc += f"- Chapter {chapter['number']}: {chapter['title']}\n"
        toc += "- Conclusion\n"
        return toc

    async def _save_ebook(self, ebook: Ebook) -> str:
        """Save ebook to database."""
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_ebooks
                (title, subtitle, author, chapters, metadata, word_count, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                ebook.title,
                ebook.subtitle,
                ebook.author,
                json.dumps({
                    "toc": ebook.table_of_contents,
                    "introduction": ebook.introduction,
                    "chapters": ebook.chapters,
                    "conclusion": ebook.conclusion
                }),
                json.dumps({
                    "cover_prompt": ebook.cover_prompt,
                    "target_audience": ebook.target_audience
                }),
                ebook.word_count,
                "draft"
            ))

            ebook_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

            return str(ebook_id)

        except Exception as e:
            logger.error(f"Save ebook failed: {e}")
            raise e

    # ==================== TRAINING DOCUMENTATION ====================

    async def generate_training_doc(self, task: dict) -> dict:
        """
        Generate training documentation using multi-AI pipeline.

        Pipeline:
        1. Claude: Structure and learning objectives
        2. GPT-4: Create exercises and quizzes
        3. Claude: Write content sections
        4. Gemini: Diagram/visual suggestions
        """
        self._ensure_tables()
        self.models_used = []

        topic = task.get("topic", "AI Automation Basics")
        module_number = task.get("module_number", 1)
        skill_level = task.get("skill_level", "beginner")

        logger.info(f"📖 Generating Training Doc: {topic} (Module {module_number})")

        try:
            # Stage 1: Define learning objectives (Claude)
            structure = await self._define_training_structure(topic, skill_level)

            # Stage 2: Write main content (Claude)
            content = await self._write_training_content(topic, structure, skill_level)

            # Stage 3: Create exercises (GPT-4)
            exercises = await self._create_training_exercises(topic, structure)

            # Stage 4: Create quiz questions (GPT-4)
            quiz = await self._create_quiz_questions(topic, structure)

            # Stage 5: Generate key takeaways (Claude)
            takeaways = await self._generate_key_takeaways(topic, structure)

            # Compile training doc
            training_doc = TrainingDoc(
                id=str(uuid.uuid4()),
                title=f"Module {module_number}: {topic}",
                module_number=module_number,
                objectives=structure.get("objectives", []),
                prerequisites=structure.get("prerequisites", []),
                content=content,
                exercises=exercises,
                quiz_questions=quiz,
                key_takeaways=takeaways,
                resources=structure.get("resources", []),
                estimated_time=structure.get("estimated_time", "30 minutes")
            )

            # Save to database
            doc_id = await self._save_training_doc(training_doc)

            logger.info(f"✅ Training doc generated: {training_doc.title}")

            return {
                "status": "completed",
                "doc_id": doc_id,
                "title": training_doc.title,
                "objectives": len(training_doc.objectives),
                "exercises": len(exercises),
                "quiz_questions": len(quiz),
                "estimated_time": training_doc.estimated_time,
                "models_used": list(set(self.models_used)),
                "content_type": "training_doc"
            }

        except Exception as e:
            logger.error(f"Training doc generation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _define_training_structure(self, topic: str, skill_level: str) -> dict:
        """Define training structure and objectives."""
        prompt = f"""Create a training module structure for: {topic}
        Skill level: {skill_level}

        Return JSON:
        {{
            "objectives": ["objective 1", "objective 2", "objective 3"],
            "prerequisites": ["prereq 1"] or [],
            "sections": ["section 1", "section 2", "section 3"],
            "estimated_time": "X minutes",
            "resources": ["resource 1", "resource 2"]
        }}"""

        try:
            response = await ai_core.generate(
                prompt,
                model="claude-3-sonnet-20240229",
                temperature=0.6,
                max_tokens=1000,
                system_prompt="You are an instructional design expert. Output only valid JSON.",
                prefer_anthropic=True
            )
            self.models_used.append("anthropic:claude-3-sonnet")
            return ai_core._safe_json(response) or {}
        except Exception as e:
            return {"objectives": [], "sections": [topic]}

    async def _write_training_content(self, topic: str, structure: dict, skill_level: str) -> str:
        """Write training content."""
        sections = structure.get("sections", [topic])
        objectives = structure.get("objectives", [])

        prompt = f"""Write training content for: {topic}
        Skill level: {skill_level}

        Learning objectives: {', '.join(objectives)}
        Sections to cover: {', '.join(sections)}

        Guidelines:
        - Use clear, instructional language
        - Include step-by-step instructions where appropriate
        - Add code examples if relevant (use markdown code blocks)
        - Use bullet points and numbered lists
        - Include practical tips and common pitfalls

        Format: Markdown with H2 for each section."""

        try:
            content = await ai_core.generate(
                prompt,
                model="claude-3-sonnet-20240229",
                temperature=0.6,
                max_tokens=3000,
                prefer_anthropic=True
            )
            return content
        except Exception as e:
            return f"# {topic}\n\nTraining content pending."

    async def _create_training_exercises(self, topic: str, structure: dict) -> list:
        """Create hands-on exercises."""
        prompt = f"""Create 3 practical exercises for training on: {topic}

        Return JSON array:
        [
            {{
                "title": "Exercise title",
                "description": "What to do",
                "steps": ["step 1", "step 2"],
                "expected_outcome": "What success looks like",
                "difficulty": "easy/medium/hard"
            }}
        ]"""

        try:
            response = await ai_core.generate(
                prompt,
                model="gpt-4-turbo-preview",
                temperature=0.6,
                max_tokens=1500,
                system_prompt="Output only valid JSON array."
            )
            self.models_used.append("openai:gpt-4-turbo-preview")
            result = ai_core._safe_json(response)
            if isinstance(result, list):
                return result
            return []
        except Exception as e:
            return []

    async def _create_quiz_questions(self, topic: str, structure: dict) -> list:
        """Create quiz questions."""
        objectives = structure.get("objectives", [])

        prompt = f"""Create 5 quiz questions for: {topic}
        Testing these objectives: {', '.join(objectives)}

        Return JSON array:
        [
            {{
                "question": "question text",
                "type": "multiple_choice" or "true_false",
                "options": ["A", "B", "C", "D"] or null,
                "correct_answer": "A" or "true/false",
                "explanation": "why this is correct"
            }}
        ]"""

        try:
            response = await ai_core.generate(
                prompt,
                model="gpt-4-turbo-preview",
                temperature=0.5,
                max_tokens=1500,
                system_prompt="Output only valid JSON array."
            )
            result = ai_core._safe_json(response)
            if isinstance(result, list):
                return result
            return []
        except Exception as e:
            return []

    async def _generate_key_takeaways(self, topic: str, structure: dict) -> list:
        """Generate key takeaways."""
        objectives = structure.get("objectives", [])

        prompt = f"""Generate 5 key takeaways for training on: {topic}
        Based on objectives: {', '.join(objectives)}

        Return as JSON array of strings."""

        try:
            response = await ai_core.generate(
                prompt,
                model="claude-3-sonnet-20240229",
                temperature=0.6,
                max_tokens=500,
                system_prompt="Output only valid JSON array of strings.",
                prefer_anthropic=True
            )
            result = ai_core._safe_json(response)
            if isinstance(result, list):
                return result
            return []
        except Exception as e:
            return ["Master the fundamentals", "Practice consistently", "Apply what you learn"]

    async def _save_training_doc(self, doc: TrainingDoc) -> str:
        """Save training doc to database."""
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_training_docs
                (title, module_number, objectives, prerequisites, content, exercises, quiz_questions, key_takeaways, resources, estimated_time, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                doc.title,
                doc.module_number,
                doc.objectives,
                doc.prerequisites,
                doc.content,
                json.dumps(doc.exercises),
                json.dumps(doc.quiz_questions),
                doc.key_takeaways,
                json.dumps(doc.resources),
                doc.estimated_time,
                "draft"
            ))

            doc_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

            return str(doc_id)

        except Exception as e:
            logger.error(f"Save training doc failed: {e}")
            raise e

    # ==================== MAIN EXECUTION ====================

    async def execute(self, task: dict) -> dict:
        """Execute content generation based on type."""
        content_type = task.get("content_type", "blog_post")

        if content_type == ContentType.BLOG_POST.value or content_type == "blog":
            return await self.blog_agent.execute(task)
        elif content_type == ContentType.NEWSLETTER.value or content_type == "newsletter":
            return await self.generate_newsletter(task)
        elif content_type == ContentType.EBOOK_FULL.value or content_type == "ebook":
            return await self.generate_ebook(task)
        elif content_type == ContentType.TRAINING_DOC.value or content_type == "training":
            return await self.generate_training_doc(task)
        else:
            return {"status": "error", "error": f"Unknown content type: {content_type}"}

    async def generate_content_batch(self, tasks: list) -> list:
        """Generate multiple pieces of content in parallel."""
        results = await asyncio.gather(*[self.execute(task) for task in tasks])
        return list(results)


# Convenience functions for API usage
async def generate_newsletter(topic: str, brand: str = "BrainOps") -> dict:
    """Quick newsletter generation."""
    orchestrator = MultiAIContentOrchestrator()
    return await orchestrator.generate_newsletter({"topic": topic, "brand": brand})


async def generate_ebook(topic: str, chapters: int = 5) -> dict:
    """Quick ebook generation."""
    orchestrator = MultiAIContentOrchestrator()
    return await orchestrator.generate_ebook({"topic": topic, "chapters": chapters})


async def generate_training(topic: str, module: int = 1) -> dict:
    """Quick training doc generation."""
    orchestrator = MultiAIContentOrchestrator()
    return await orchestrator.generate_training_doc({"topic": topic, "module_number": module})


if __name__ == "__main__":
    # Test
    async def test():
        orchestrator = MultiAIContentOrchestrator()

        # Test newsletter
        result = await orchestrator.execute({
            "content_type": "newsletter",
            "topic": "AI Automation Trends 2026",
            "brand": "BrainOps"
        })
        print(f"Newsletter: {result}")

    asyncio.run(test())

#!/usr/bin/env python3
"""
Content Generation AI Agent
Autonomous agent that creates, optimizes, and publishes SEO content.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from ai_core import ai_core

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration (reused from customer_acquisition_agents.py pattern)
def _get_db_config():
    """Get database configuration from environment variables."""
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_port = os.getenv("DB_PORT", "5432")

    missing = []
    if not db_host:
        missing.append("DB_HOST")
    if not db_name:
        missing.append("DB_NAME")
    if not db_user:
        missing.append("DB_USER")
    if not db_password:
        missing.append("DB_PASSWORD")

    if missing:
        raise RuntimeError(
            f"Required environment variables not set: {', '.join(missing)}. "
            "Set these variables before using content agent."
        )

    return {
        "host": db_host,
        "database": db_name,
        "user": db_user,
        "password": db_password,
        "port": int(db_port)
    }

def _get_db_connection(**kwargs):
    """Get database connection with validated config."""
    db_config = _get_db_config()
    db_config.update(kwargs)
    return psycopg2.connect(**db_config)

class ContentGeneratorAgent:
    """Agent that generates and publishes SEO content"""

    _tables_ensured = False

    def __init__(self):
        self.agent_id = str(uuid.uuid4())
        self.agent_name = "ContentGeneratorAgent"
        logger.info(f"Initialized {self.agent_name}")

    def _ensure_tables(self):
        """Ensure content tables exist - lazy initialization"""
        if ContentGeneratorAgent._tables_ensured:
            return
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_posts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title TEXT NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    content TEXT,
                    excerpt TEXT,
                    status VARCHAR(50) DEFAULT 'draft',
                    author_id UUID,
                    seo_metadata JSONB DEFAULT '{}'::jsonb,
                    metrics JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    published_at TIMESTAMPTZ
                );

                CREATE INDEX IF NOT EXISTS idx_content_posts_slug ON content_posts(slug);
                CREATE INDEX IF NOT EXISTS idx_content_posts_status ON content_posts(status);
            """)

            conn.commit()
            cursor.close()
            conn.close()
            ContentGeneratorAgent._tables_ensured = True
            logger.info("Content tables ensured")
        except Exception as e:
            logger.warning(f"Could not ensure content tables: {e}")

    async def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute content generation task"""
        self._ensure_tables()
        action = task.get('action', 'generate_blog')

        if action == 'generate_blog':
            return await self.generate_blog_post(task)
        elif action == 'generate_topic_ideas':
            return await self.generate_topic_ideas(task)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def generate_topic_ideas(self, task: dict) -> dict:
        """Generate blog topic ideas based on industry trends"""
        try:
            industry = task.get("industry", "roofing")
            count = task.get("count", 5)

            prompt = f"""Generate {count} engaging blog post topics for a {industry} company.
            Focus on topics that drive leads, answer common customer questions, and build trust.
            Include a suggested SEO keyword for each.
            
            Return ONLY a JSON array of objects with keys: topic, keyword, target_audience."""

            response = await ai_core.generate(
                prompt,
                model="gpt-4-turbo-preview",
                temperature=0.7,
                system_prompt="You are an SEO content strategist. Output only valid JSON."
            )

            # Use ai_core's safe json parsing if response is string
            if isinstance(response, str):
                topics = ai_core._safe_json(response)
                # If _safe_json returns dict but we expect list, try to find list inside
                if isinstance(topics, dict):
                    # Check if it wrapped in a key like "topics"
                    for key in topics:
                        if isinstance(topics[key], list):
                            topics = topics[key]
                            break
            else:
                topics = response

            return {"status": "completed", "topics": topics}

        except Exception as e:
            logger.error(f"Topic generation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def generate_blog_post(self, task: dict) -> dict:
        """Generate and publish a full blog post"""
        try:
            topic = task.get("topic")
            if not topic:
                # Auto-generate a topic if none provided
                topics_result = await self.generate_topic_ideas({"count": 1})
                if topics_result.get("topics") and isinstance(topics_result["topics"], list) and len(topics_result["topics"]) > 0:
                    topic = topics_result["topics"][0].get("topic")
                else:
                    topic = "The Importance of Regular Roof Inspections"

            logger.info(f"Generating blog post for: {topic}")

            # 1. Generate Outline
            outline_prompt = f"""Create a detailed blog post outline for: "{topic}"
            Target Audience: Homeowners
            Goal: Convert readers to book an inspection
            
            Include:
            - Catchy Title
            - Introduction (Hook)
            - 3-4 Main H2 Headings with key points
            - Conclusion with CTA
            
            Return as JSON."""

            outline_response = await ai_core.generate(
                outline_prompt,
                model="gpt-4-turbo-preview",
                temperature=0.7
            )
            
            # If outline is JSON, extract string representation or use as is
            if isinstance(outline_response, str):
                outline = outline_response
            else:
                outline = json.dumps(outline_response)

            # 2. Write Content
            write_prompt = f"""Write a comprehensive, SEO-optimized blog post based on this outline:
            {outline}
            
            Guidelines:
            - Tone: Professional yet accessible, trustworthy
            - Length: 800-1200 words
            - Format: Markdown
            - Include: Bullet points, short paragraphs
            - CTA: Encourage booking a free inspection at MyRoofGenius.com
            
            Return ONLY the Markdown content."""

            content = await ai_core.generate(
                write_prompt,
                model="gpt-4-turbo-preview", # ai_core handles fallback to Anthropic/Gemini
                temperature=0.7
            )

            # 3. Generate SEO Metadata
            seo_prompt = f"""Generate SEO metadata for this blog post:
            Topic: {topic}
            
            Return JSON with:
            - meta_title (max 60 chars)
            - meta_description (max 160 chars)
            - slug (url-friendly)
            - keywords (array)"""

            seo_response = await ai_core.generate(
                seo_prompt,
                model="gpt-3.5-turbo",
                system_prompt="Output only valid JSON."
            )
            
            if isinstance(seo_response, str):
                seo_data = ai_core._safe_json(seo_response)
            else:
                seo_data = seo_response

            # 4. Publish to Database
            post_id = await self._publish_post(
                title=topic,
                content=content,
                seo_data=seo_data,
                status=task.get("status", "published")
            )

            return {
                "status": "completed",
                "post_id": post_id,
                "title": topic,
                "slug": seo_data.get("slug"),
                "url": f"https://myroofgenius.com/blog/{seo_data.get('slug')}"
            }

        except Exception as e:
            logger.error(f"Blog generation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _publish_post(self, title: str, content: str, seo_data: dict, status: str) -> str:
        """Save post to database"""
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            # Ensure slug is unique
            slug = seo_data.get("slug")
            if not slug:
                 slug = title.lower().replace(" ", "-").replace(r"[^a-z0-9-]", "")

            cursor.execute("SELECT id FROM content_posts WHERE slug = %s", (slug,))
            if cursor.fetchone():
                slug = f"{slug}-{uuid.uuid4().hex[:6]}"

            cursor.execute("""
                INSERT INTO content_posts
                (title, slug, content, excerpt, status, seo_metadata, published_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                title,
                slug,
                content,
                seo_data.get("meta_description"),
                status,
                json.dumps(seo_data),
                datetime.now(timezone.utc) if status == 'published' else None
            ))

            post_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Published post: {title} ({post_id})")
            return str(post_id)

        except Exception as e:
            logger.error(f"Database publish failed: {e}")
            raise e

if __name__ == "__main__":
    # Test run
    agent = ContentGeneratorAgent()
    asyncio.run(agent.execute({"action": "generate_blog", "topic": "5 Signs Your Roof Needs Replacement"}))
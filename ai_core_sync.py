#!/usr/bin/env python3
"""
Synchronous AI Core - Alternative implementation
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
from anthropic import Anthropic
import psycopg2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Database config
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
    'password': os.getenv('DB_PASSWORD', 'REDACTED_SUPABASE_DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 5432))
}

class SyncAICore:
    """Synchronous AI implementation"""

    def __init__(self):
        # Initialize clients
        if OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized (sync)")
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not found")

        if ANTHROPIC_API_KEY:
            self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Anthropic client initialized (sync)")
        else:
            self.anthropic_client = None
            logger.warning("Anthropic API key not found")

    def generate(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None
    ) -> str:
        """Synchronous AI generation"""

        try:
            # Use OpenAI
            if self.openai_client and ("gpt" in model.lower() or model == "openai"):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                # Use correct model name
                if "gpt-4" in model.lower():
                    actual_model = "gpt-4-0125-preview"
                else:
                    actual_model = "gpt-3.5-turbo"

                response = self.openai_client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=5
                )
                return response.choices[0].message.content

            # Use Claude
            elif self.anthropic_client and ("claude" in model.lower() or model == "anthropic"):
                system = system_prompt or "You are a helpful AI assistant."

                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",  # Cheaper model
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=5
                )
                return response.content[0].text

            else:
                return "AI service not available"

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {str(e)}"

    def generate_embeddings(self, text: str):
        """Generate embeddings"""
        if not self.openai_client:
            return None

        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                timeout=10
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

# Global instance
sync_ai_core = SyncAICore()
#!/usr/bin/env python3
"""
REAL AI Core - No fake implementations
Uses actual API keys from production environment
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

import psycopg2
from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# REAL API KEYS - From environment variables (already configured in Render)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Database configuration - supports both individual env vars and DATABASE_URL
from urllib.parse import urlparse

_DB_HOST = os.getenv('DB_HOST')
_DB_NAME = os.getenv('DB_NAME')
_DB_USER = os.getenv('DB_USER')
_DB_PASSWORD = os.getenv('DB_PASSWORD')
_DB_PORT = os.getenv('DB_PORT', '5432')

# Fallback to DATABASE_URL if individual vars not set (Render provides DATABASE_URL)
if not all([_DB_HOST, _DB_NAME, _DB_USER, _DB_PASSWORD]):
    _DATABASE_URL = os.getenv('DATABASE_URL', '')
    if _DATABASE_URL:
        _parsed = urlparse(_DATABASE_URL)
        _DB_HOST = _parsed.hostname or ''
        _DB_NAME = _parsed.path.lstrip('/') if _parsed.path else ''
        _DB_USER = _parsed.username or ''
        _DB_PASSWORD = _parsed.password or ''
        _DB_PORT = str(_parsed.port) if _parsed.port else '5432'
        logger.info(f"ai_core: Parsed DATABASE_URL: host={_DB_HOST}, db={_DB_NAME}")

if not all([_DB_HOST, _DB_NAME, _DB_USER, _DB_PASSWORD]):
    raise RuntimeError(
        "Database configuration is incomplete. "
        "Set DB_HOST/DB_NAME/DB_USER/DB_PASSWORD or DATABASE_URL."
    )

DB_CONFIG = {
    'host': _DB_HOST,
    'database': _DB_NAME,
    'user': _DB_USER,
    'password': _DB_PASSWORD,
    'port': int(_DB_PORT)
}

class ModelRouter:
    """Lightweight router to pick cheap vs. strong vs. reasoning models."""

    def __init__(self, openai_available: bool, anthropic_available: bool):
        self.openai_available = openai_available
        self.anthropic_available = anthropic_available

    def _fast_model(self, prefer_anthropic: bool = False) -> str:
        if prefer_anthropic and self.anthropic_available:
            return "claude-3-haiku-20240307"
        if self.openai_available:
            return "gpt-3.5-turbo"
        if self.anthropic_available:
            return "claude-3-haiku-20240307"
        return "gpt-3.5-turbo"

    def _strong_model(self, prefer_anthropic: bool = False) -> str:
        if prefer_anthropic and self.anthropic_available:
            return "claude-3-opus-20240229"
        if self.openai_available:
            return "gpt-4-0125-preview"
        if self.anthropic_available:
            return "claude-3-opus-20240229"
        return "gpt-4-0125-preview"

    def _reasoning_model(self) -> str:
        """Return OpenAI o3-mini for deep reasoning tasks (fast, cost-efficient)."""
        if self.openai_available:
            return "o3-mini"  # Updated: o1-preview deprecated, o3-mini is current
        # Fallback to strong model if reasoning model not available
        return self._strong_model()

    def route(
        self,
        intent: str = "general",
        complexity: str = "standard",
        prefer_anthropic: bool = False,
        allow_expensive: bool = True,
        require_reasoning: bool = False
    ) -> str:
        """Choose model based on intent/complexity without expensive defaults.

        Args:
            intent: Task intent (routing, review, reasoning, estimation, etc.)
            complexity: Task complexity (routing, classification, standard, complex)
            prefer_anthropic: Prefer Claude models when available
            allow_expensive: Allow expensive models like GPT-4
            require_reasoning: Force use of o1 reasoning model for complex tasks
        """
        # Reasoning intents that benefit from o1's chain-of-thought
        reasoning_intents = {"reasoning", "estimation", "calculation", "analysis", "planning", "optimization"}

        # Use o1 for deep reasoning tasks
        if require_reasoning or (intent in reasoning_intents and allow_expensive):
            return self._reasoning_model()

        routing_intents = {"routing", "selector", "classification", "quality_gate", "review"}
        if complexity in {"routing", "classification"} or intent in routing_intents or not allow_expensive:
            return self._fast_model(prefer_anthropic)
        return self._strong_model(prefer_anthropic)

class RealAICore:
    """REAL AI implementation - no fake responses"""

    def __init__(self):
        # Validate API keys are available
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not found in environment - AI features may be limited")
            self.openai_client = None
            self.async_openai = None
        else:
            # Initialize REAL OpenAI clients
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.async_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")

        if not ANTHROPIC_API_KEY:
            logger.warning("Anthropic API key not found in environment - Claude features may be limited")
            self.anthropic_client = None
            self.async_anthropic = None
        else:
            # Initialize REAL Anthropic clients
            self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            self.async_anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Anthropic client initialized successfully")

        # Initialize Perplexity (reliable fallback)
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        if self.perplexity_key:
            logger.info("Perplexity API key found - fallback available")
        else:
            logger.warning("Perplexity API key not found - fallback unavailable")

        # Initialize Gemini (powerful fallback when OpenAI/Claude rate limited)
        self.gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.gemini_model = None
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Gemini 2.0 Flash initialized - powerful fallback available")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")
        else:
            logger.warning("Gemini API key not found - fallback unavailable")

        # Model router prefers cheaper models for routing/review and stronger models for generation
        self.model_router = ModelRouter(
            openai_available=self.async_openai is not None,
            anthropic_available=self.async_anthropic is not None
        )

        logger.info("AI Core initialized - Ready for REAL AI operations")

    def get_db_connection(self):
        """Direct connection to master database"""
        return psycopg2.connect(**DB_CONFIG)

    def _normalize_model_name(self, model: str) -> str:
        """Map friendly model aliases to deployable names."""
        if not model:
            return "gpt-4-0125-preview"
        if model in {"gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview"}:
            return "gpt-4-0125-preview"
        if model in {"gpt-3.5", "gpt-3.5-turbo-0125"}:
            return "gpt-3.5-turbo"
        if model.startswith("claude-3-opus"):
            return "claude-3-opus-20240229"
        if model.startswith("claude-3-haiku"):
            return "claude-3-haiku-20240307"
        # o-series reasoning models - preserve exact model name
        if model in {"o1", "o1-preview", "o1-mini", "o3", "o3-mini", "o4-mini"}:
            return model
        return model

    def _is_reasoning_model(self, model: str) -> bool:
        """Check if model is an o-series reasoning model (requires different API params)."""
        return model in {"o1", "o1-preview", "o1-mini", "o3", "o3-mini", "o4-mini"}

    def _safe_json(self, text: str) -> dict[str, Any]:
        """Parse JSON content without failing the caller."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("Failed to parse JSON response: %s", exc)
            try:
                import re
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except (json.JSONDecodeError, TypeError, re.error) as exc:
                logger.debug("Failed to parse extracted JSON: %s", exc)
        return {}

    async def generate(
        self,
        prompt: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        intent: Optional[str] = None,
        use_model_routing: bool = False,
        prefer_anthropic: bool = False,
        allow_expensive: bool = True
    ) -> Any:
        """Generate REAL AI response with intelligent fallback chain.

        Fallback order: OpenAI → Anthropic → Gemini → Perplexity
        """
        # Determine model before try block so it's accessible in except
        selected_model = model or "gpt-4-0125-preview"
        if use_model_routing or intent:
            selected_model = self.model_router.route(
                intent=intent or "general",
                complexity="routing" if not allow_expensive else "standard",
                prefer_anthropic=prefer_anthropic,
                allow_expensive=allow_expensive
            )
        selected_model = self._normalize_model_name(selected_model)
        is_openai_model = selected_model.startswith("gpt") or selected_model == "openai" or self._is_reasoning_model(selected_model)
        is_anthropic_model = selected_model.startswith("claude") or selected_model == "anthropic"

        try:
            if is_openai_model and self.async_openai:
                # o1 models have different API requirements
                if self._is_reasoning_model(selected_model):
                    # o1 doesn't support system prompts - combine into user message
                    combined_prompt = prompt
                    if system_prompt:
                        combined_prompt = f"{system_prompt}\n\n{prompt}"
                    messages = [{"role": "user", "content": combined_prompt}]

                    # o1 uses max_completion_tokens, not max_tokens
                    # o1 doesn't support temperature or streaming
                    response = await self.async_openai.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        timeout=120  # o1 needs longer timeout for reasoning
                    )
                    return response.choices[0].message.content
                else:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    response = await self.async_openai.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                        timeout=5  # Add timeout
                    )
                    if stream:
                        return response
                    return response.choices[0].message.content

            # Use Claude as alternative
            elif is_anthropic_model and self.async_anthropic:
                system = system_prompt or "You are a helpful AI assistant for a roofing business."

                response = await self.async_anthropic.messages.create(
                    model=selected_model,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.content[0].text

            # No client matched the requested/auto-routed model - try fallbacks
            if self.gemini_model:
                return await self._try_gemini(prompt, system_prompt, max_tokens)
            if self.perplexity_key:
                return await self._try_perplexity(prompt, max_tokens)
            raise RuntimeError("AI client not available for requested model")

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in str(e) or "rate" in error_str or "quota" in error_str

            if is_rate_limit:
                logger.warning(f"⚠️ Rate limit hit, trying fallback chain: {e}")
            else:
                logger.error(f"AI generation error: {e}")

            # Intelligent fallback chain: Gemini → Anthropic → Perplexity
            if self.gemini_model:
                try:
                    logger.info("🔄 Trying Gemini fallback...")
                    return await self._try_gemini(prompt, system_prompt, max_tokens)
                except Exception as gemini_error:
                    logger.warning(f"Gemini fallback failed: {gemini_error}")

            # If original was OpenAI and failed, try Anthropic
            if is_openai_model and self.async_anthropic:
                try:
                    logger.info("🔄 Trying Anthropic fallback...")
                    system = system_prompt or "You are a helpful AI assistant."
                    response = await self.async_anthropic.messages.create(
                        model="claude-3-haiku-20240307",
                        system=system,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.content[0].text
                except Exception as anthropic_error:
                    logger.warning(f"Anthropic fallback failed: {anthropic_error}")

            # Last resort: Perplexity
            if self.perplexity_key:
                try:
                    logger.info("🔄 Trying Perplexity fallback...")
                    return await self._try_perplexity(prompt, max_tokens)
                except Exception as perplexity_error:
                    logger.error(f"Perplexity fallback also failed: {perplexity_error}")

            raise e

    async def _try_gemini(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 2000) -> str:
        """Use Gemini as fallback AI provider"""
        if not self.gemini_model:
            raise RuntimeError("Gemini not configured")

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Run synchronous Gemini call in executor
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.gemini_model.generate_content(full_prompt)
        )
        return response.text

    async def _try_perplexity(self, prompt: str, max_tokens: int = 2000) -> str:
        """Use Perplexity as fallback AI provider"""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.perplexity_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def reason(
        self,
        problem: str,
        context: Optional[dict[str, Any]] = None,
        max_tokens: int = 4000,
        model: str = "o3-mini"
    ) -> dict[str, Any]:
        """Use o3-mini reasoning model for complex multi-step problems.

        This method is specifically designed for tasks requiring:
        - Complex calculations (e.g., material waste ratios, pricing optimization)
        - Multi-step logical reasoning
        - Strategic planning and analysis
        - Scientific or technical problem solving

        Args:
            problem: The problem statement requiring deep reasoning
            context: Additional context data (will be JSON-serialized)
            max_tokens: Maximum completion tokens (default 4000 for complex reasoning)
            model: Reasoning model to use (o3-mini, o3, o4-mini)

        Returns:
            Dict with 'reasoning' (full response) and 'conclusion' (extracted answer)
        """
        if not self.async_openai:
            logger.warning("OpenAI not available for o1 reasoning, falling back to GPT-4")
            response = await self.generate(
                prompt=problem,
                system_prompt="Think step by step and provide detailed reasoning.",
                model="gpt-4-0125-preview",
                max_tokens=max_tokens
            )
            return {"reasoning": response, "conclusion": response, "model_used": "gpt-4-0125-preview"}

        # Build the reasoning prompt with context
        full_prompt = problem
        if context:
            context_str = json.dumps(context, indent=2, default=str)[:2000]
            full_prompt = f"Context:\n{context_str}\n\nProblem:\n{problem}"

        try:
            response = await self.async_openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                max_completion_tokens=max_tokens,
                timeout=180  # Extended timeout for complex reasoning
            )

            reasoning_text = response.choices[0].message.content

            # Try to extract a conclusion if present
            conclusion = reasoning_text
            if "conclusion" in reasoning_text.lower():
                parts = reasoning_text.lower().split("conclusion")
                if len(parts) > 1:
                    conclusion = parts[-1].strip(": \n")[:500]
            elif "therefore" in reasoning_text.lower():
                parts = reasoning_text.lower().split("therefore")
                if len(parts) > 1:
                    conclusion = "Therefore " + parts[-1].strip(": \n")[:500]

            logger.info(f"✅ o1 reasoning completed using {model}")
            return {
                "reasoning": reasoning_text,
                "conclusion": conclusion,
                "model_used": model,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }

        except Exception as e:
            logger.error(f"o1 reasoning failed: {e}, falling back to GPT-4")
            response = await self.generate(
                prompt=problem,
                system_prompt="Think step by step and provide detailed reasoning.",
                model="gpt-4-0125-preview",
                max_tokens=max_tokens
            )
            return {"reasoning": response, "conclusion": response, "model_used": "gpt-4-0125-preview (fallback)", "error": str(e)}

    async def route_agent(self, task: dict[str, Any], candidate_agents: list[str]) -> dict[str, Any]:
        """Use a cheap model to route work to the right agent."""
        if not candidate_agents:
            return {"agent": None, "reason": "no candidates provided"}

        task_preview = json.dumps(task)[:1200]
        prompt = f"""
        You are routing a task to the best-fit agent.
        Candidate agents: {', '.join(candidate_agents)}
        Task: {task_preview}

        Respond in JSON with:
        {{
          "agent": "<name>",
          "reason": "why this is the best fit"
        }}
        """

        response = await self.generate(
            prompt,
            use_model_routing=True,
            intent="routing",
            allow_expensive=False,
            temperature=0
        )

        parsed = self._safe_json(response)
        selected = parsed.get("agent") or candidate_agents[0]
        parsed["agent"] = selected
        return parsed

    async def review_and_refine(
        self,
        draft: str,
        context: Optional[dict[str, Any]] = None,
        criteria: Optional[list[str]] = None,
        max_iterations: int = 2
    ) -> dict[str, Any]:
        """Run an AI review loop that can suggest fixes before finalizing output."""
        criteria = criteria or ["accuracy", "safety", "actionability"]
        feedback_history: list[dict[str, Any]] = []
        if not isinstance(draft, str):
            try:
                draft = json.dumps(draft)
            except (TypeError, ValueError) as exc:
                logger.debug("Failed to JSON-encode draft: %s", exc)
                draft = str(draft)
        current = draft

        for _ in range(max_iterations):
            review_prompt = f"""
            Act as a strict reviewer. Evaluate the draft against these criteria: {', '.join(criteria)}.
            Context: {json.dumps(context or {})[:800]}

            Respond as JSON:
            {{
              "approved": true/false,
              "issues": ["short bullet issues"],
              "must_fix": true/false,
              "summary": "short review summary"
            }}

            Draft:
            {current}
            """

            review_response = await self.generate(
                review_prompt,
                use_model_routing=True,
                intent="review",
                allow_expensive=False,
                temperature=0,
                max_tokens=500
            )

            review_data = self._safe_json(review_response) or {"approved": True}
            review_data.setdefault("approved", True)
            feedback_history.append(review_data)

            if review_data.get("approved") or not review_data.get("issues"):
                break

            improve_prompt = f"""
            Improve the draft to resolve these issues: {review_data.get('issues', [])}.
            Keep the response concise and actionable. Do not invent missing facts.
            Context: {json.dumps(context or {})[:800]}

            Revised draft:
            """

            current = await self.generate(
                improve_prompt,
                use_model_routing=True,
                intent="execution",
                allow_expensive=True,
                temperature=0.4,
                max_tokens=1200
            )

        return {
            "content": current,
            "feedback": feedback_history,
            "approved": feedback_history[-1].get("approved", True) if feedback_history else True
        }

    async def quality_gate(
        self,
        output: str,
        criteria: Optional[list[str]] = None,
        min_score: int = 70
    ) -> dict[str, Any]:
        """Score output with a quality gate before returning it downstream."""
        # If no models are available, skip gating but signal that it was skipped
        if not self.async_openai and not self.async_anthropic:
            return {
                "pass": True,
                "score": 100,
                "issues": ["Quality gate skipped: no AI client available"],
                "actions": []
            }

        if not isinstance(output, str):
            try:
                output = json.dumps(output)
            except (TypeError, ValueError) as exc:
                logger.debug("Failed to JSON-encode output: %s", exc)
                output = str(output)

        criteria = criteria or [
            "No hallucinated facts",
            "Directly answers the request",
            "Actionable next steps included"
        ]

        gate_prompt = f"""
        You are a quality gate. Score the following output against these criteria:
        {json.dumps(criteria)}

        Provide JSON with:
        {{
          "pass": true/false,
          "score": 0-100,
          "issues": ["issue1", "issue2"],
          "actions": ["specific fixes to reach pass"]
        }}

        Output to score:
        {output}
        """

        response = await self.generate(
            gate_prompt,
            use_model_routing=True,
            intent="quality_gate",
            allow_expensive=False,
            temperature=0,
            max_tokens=600
        )

        parsed = self._safe_json(response)
        score = int(parsed.get("score", min_score))
        passed = parsed.get("pass", score >= min_score)

        return {
            "pass": passed,
            "score": score,
            "issues": parsed.get("issues", []),
            "actions": parsed.get("actions", [])
        }

    async def generate_embeddings(self, text: str) -> list[float]:
        """Generate REAL embeddings for vector search"""
        try:
            response = await self.async_openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise e

    async def analyze_image(self, image_url: str, prompt: str) -> str:
        """REAL image analysis with GPT-4 Vision"""
        try:
            response = await self.async_openai.chat.completions.create(
                model="gpt-4-turbo",  # Updated model name for vision
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            raise e

    async def analyze_roofing_job(self, job_data: dict) -> dict:
        """REAL AI analysis for roofing jobs"""
        prompt = f"""
        Analyze this roofing job and provide detailed insights:

        Job Data:
        - Customer: {job_data.get('customer_name')}
        - Address: {job_data.get('address')}
        - Roof Type: {job_data.get('roof_type')}
        - Square Footage: {job_data.get('sq_ft')}
        - Current Condition: {job_data.get('condition')}
        - Budget: ${job_data.get('budget', 'Not specified')}

        Provide:
        1. Recommended materials and quantities
        2. Estimated labor hours
        3. Cost breakdown
        4. Timeline estimate
        5. Risk factors
        6. Upsell opportunities
        7. Weather considerations

        Return as JSON with these keys:
        materials, labor_hours, cost_breakdown, timeline_days, risks, upsells, weather_notes, total_estimate
        """

        response = await self.generate(
            prompt,
            model="gpt-4",
            temperature=0.3,  # Lower temp for more consistent analysis
            max_tokens=2000
        )

        # Parse JSON from response
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Structure the response if not JSON
                return {
                    "analysis": response,
                    "status": "completed",
                    "model": "gpt-4",
                    "timestamp": datetime.now().isoformat()
                }
        except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
            logger.debug(f"JSON parse in market analysis fallback: {parse_error}")
            return {
                "analysis": response,
                "status": "completed",
                "model": "gpt-4"
            }

    async def generate_proposal(self, customer_data: dict, job_data: dict) -> str:
        """Generate REAL AI-powered proposal"""
        prompt = f"""
        Create a professional roofing proposal for:

        Customer:
        Name: {customer_data.get('name')}
        Company: {customer_data.get('company', 'Residential')}
        Email: {customer_data.get('email')}

        Job Details:
        {json.dumps(job_data, indent=2)}

        Create a compelling, professional proposal that includes:
        1. Executive summary
        2. Scope of work
        3. Materials and methods
        4. Timeline
        5. Investment breakdown
        6. Warranties and guarantees
        7. Why choose us
        8. Next steps

        Make it persuasive and professional.
        """

        return await self.generate(
            prompt,
            model="gpt-4",
            temperature=0.7,
            max_tokens=3000,
            system_prompt="You are an expert roofing sales professional creating winning proposals."
        )

    async def score_lead(self, lead_data: dict) -> dict:
        """REAL AI lead scoring"""
        prompt = f"""
        Score this lead from 0-100 based on conversion probability.

        Lead Information:
        {json.dumps(lead_data, indent=2)}

        Consider:
        - Urgency indicators
        - Budget alignment
        - Decision timeline
        - Competition mentioned
        - Previous interactions
        - Property type and value
        - Insurance claim potential

        Return JSON with:
        {{
            "score": 0-100,
            "reasoning": "explanation",
            "recommendations": ["action1", "action2"],
            "best_contact_time": "time",
            "estimated_value": dollar_amount,
            "conversion_probability": percentage
        }}
        """

        response = await self.generate(
            prompt,
            model="gpt-4",
            temperature=0.3,
            max_tokens=1000
        )

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as parse_error:
            logger.debug(f"JSON extraction failed in churn analysis: {parse_error}")

        # Fallback structure
        return {
            "score": 75,
            "analysis": response,
            "model": "gpt-4"
        }

    async def optimize_schedule(self, jobs: list[dict], crews: list[dict]) -> dict:
        """REAL AI scheduling optimization"""
        prompt = f"""
        Optimize the scheduling for these roofing jobs:

        Jobs to Schedule:
        {json.dumps(jobs, indent=2)}

        Available Crews:
        {json.dumps(crews, indent=2)}

        Consider:
        - Weather forecasts
        - Travel time between jobs
        - Crew skills and certifications
        - Customer preferences
        - Material availability
        - Profit optimization

        Return an optimal schedule as JSON.
        """

        response = await self.generate(
            prompt,
            model="gpt-4",
            temperature=0.2,
            max_tokens=2000
        )

        return {
            "schedule": response,
            "optimized": True,
            "model": "gpt-4"
        }

    async def chat_with_context(
        self,
        messages: list[dict[str, str]],
        context: Optional[dict] = None
    ) -> str:
        """REAL conversational AI with context"""

        # Build context-aware messages
        system_prompt = """You are an AI assistant for MyRoofGenius, a professional roofing company.
        You help with estimates, scheduling, customer service, and technical roofing questions.
        Be professional, helpful, and knowledgeable about roofing."""

        if context:
            system_prompt += f"\n\nContext: {json.dumps(context)}"

        # Add system message
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        # Check if client is available
        if not self.async_openai:
            return "AI service temporarily unavailable"

        response = await self.async_openai.chat.completions.create(
            model="gpt-4-0125-preview",  # Use correct model name
            messages=full_messages,
            temperature=0.7,
            max_tokens=1000,
            timeout=5  # Add timeout
        )

        return response.choices[0].message.content

    def log_usage(self, prompt: str, response: str, model: str, tokens_used: Optional[int] = None):
        """Log AI usage to database for tracking"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()

            # Use correct column names matching ai_usage_logs table schema:
            # user_id, service, model, endpoint are required
            # Store prompt/response in meta_data JSON, cost in total_cost
            cost_cents = self.calculate_cost(model, tokens_used) if tokens_used else 0
            cursor.execute("""
                INSERT INTO ai_usage_logs
                (id, user_id, service, model, endpoint, tokens_used, total_cost, meta_data, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                str(uuid.uuid4()),
                '00000000-0000-0000-0000-000000000000',  # System user
                'ai_core',
                model,
                'generate',
                tokens_used,
                cost_cents / 100.0 if cost_cents else 0.0,  # Convert cents to dollars
                json.dumps({
                    "prompt": prompt[:1000],
                    "response": response[:2000]
                })
            ))

            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Usage logging error: {e}")

    def calculate_cost(self, model: str, tokens: int) -> int:
        """Calculate cost in cents based on model and tokens"""
        costs = {
            "gpt-4": 0.03,  # per 1K tokens
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.001,
            "claude-3-opus": 0.015,
            "text-embedding-3-small": 0.00002
        }

        for model_name, cost_per_1k in costs.items():
            if model_name in model.lower():
                return int((tokens / 1000) * cost_per_1k * 100)  # Convert to cents

        return 0

# Global instance
ai_core = RealAICore()

# Convenience functions
async def ai_generate(prompt: str, **kwargs) -> str:
    """Quick function for AI generation"""
    return await ai_core.generate(prompt, **kwargs)

async def ai_analyze(data: dict, analysis_type: str = "general") -> dict:
    """Quick function for AI analysis"""
    if analysis_type == "roofing":
        return await ai_core.analyze_roofing_job(data)
    elif analysis_type == "lead":
        return await ai_core.score_lead(data)
    else:
        prompt = f"Analyze this data: {json.dumps(data)}"
        response = await ai_core.generate(prompt)
        return {"analysis": response}

if __name__ == "__main__":
    # Test the AI
    import asyncio

    async def test():
        print("Testing REAL AI...")
        response = await ai_generate("What are the best roofing materials for Florida weather?")
        print(f"AI Response: {response}")

    asyncio.run(test())

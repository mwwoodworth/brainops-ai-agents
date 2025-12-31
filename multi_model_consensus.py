#!/usr/bin/env python3
"""
Multi-Model Consensus System - Task 16
System for getting consensus from multiple AI models for improved accuracy and reliability
"""

import os
import logging
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import psycopg2
from psycopg2.extras import RealDictCursor, Json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", "5432")
}


class ConsensusStrategy(Enum):
    """Strategies for reaching consensus"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    UNANIMOUS = "unanimous"
    HIGHEST_CONFIDENCE = "highest_confidence"
    ENSEMBLE = "ensemble"
    DEBATE = "debate"
    HIERARCHICAL = "hierarchical"


class ModelType(Enum):
    """Types of AI models"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT4O = "openai_gpt4o"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    GOOGLE_GEMINI = "google_gemini"
    PERPLEXITY = "perplexity"
    LOCAL_LLAMA = "local_llama"
    HUGGINGFACE = "huggingface"


class ConsensusStatus(Enum):
    """Status of consensus process"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CONSENSUS_REACHED = "consensus_reached"
    DISAGREEMENT = "disagreement"
    PARTIAL_CONSENSUS = "partial_consensus"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ModelResponse:
    """Response from a single model"""
    model: ModelType
    response: str
    confidence: float
    reasoning: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Result of consensus process"""
    consensus_id: str
    final_response: str
    confidence: float
    strategy_used: ConsensusStrategy
    status: ConsensusStatus
    participating_models: List[ModelType]
    model_responses: List[ModelResponse]
    agreement_score: float
    dissenting_opinions: List[Dict]
    metadata: Dict = field(default_factory=dict)


class ModelProvider:
    """Provider for AI model interactions"""

    def __init__(self):
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_client = None
        self._model_weights = {
            ModelType.OPENAI_GPT4: 1.0,
            ModelType.OPENAI_GPT4O: 1.0,
            ModelType.ANTHROPIC_CLAUDE: 1.0,
            ModelType.GOOGLE_GEMINI: 0.9,
            ModelType.PERPLEXITY: 0.8,
            ModelType.HUGGINGFACE: 0.7,
            ModelType.LOCAL_LLAMA: 0.6
        }

    def _get_openai_client(self):
        """Lazy load OpenAI client"""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                logger.warning(f"OpenAI client not available: {e}")
        return self._openai_client

    def _get_anthropic_client(self):
        """Lazy load Anthropic client"""
        if self._anthropic_client is None:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
            except Exception as e:
                logger.warning(f"Anthropic client not available: {e}")
        return self._anthropic_client

    async def query_model(
        self,
        model: ModelType,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> ModelResponse:
        """Query a specific AI model"""
        start_time = datetime.now(timezone.utc)

        try:
            if model in [ModelType.OPENAI_GPT4, ModelType.OPENAI_GPT4O]:
                response = await self._query_openai(
                    model, prompt, system_prompt, max_tokens, temperature
                )
            elif model == ModelType.ANTHROPIC_CLAUDE:
                response = await self._query_anthropic(
                    prompt, system_prompt, max_tokens, temperature
                )
            elif model == ModelType.GOOGLE_GEMINI:
                response = await self._query_gemini(
                    prompt, system_prompt, max_tokens, temperature
                )
            else:
                # Fallback response for unavailable models
                response = ModelResponse(
                    model=model,
                    response=f"Model {model.value} is not currently available",
                    confidence=0.0,
                    metadata={"available": False}
                )

            latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            response.latency_ms = latency
            return response

        except Exception as e:
            logger.error(f"Error querying {model.value}: {e}")
            return ModelResponse(
                model=model,
                response="",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    async def _query_openai(
        self,
        model: ModelType,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> ModelResponse:
        """Query OpenAI models"""
        client = self._get_openai_client()
        if not client:
            return ModelResponse(
                model=model,
                response="OpenAI client not available",
                confidence=0.0
            )

        model_name = "gpt-4" if model == ModelType.OPENAI_GPT4 else "gpt-4o"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return ModelResponse(
            model=model,
            response=response.choices[0].message.content,
            confidence=0.85,  # OpenAI doesn't provide confidence
            tokens_used=response.usage.total_tokens if response.usage else 0
        )

    async def _query_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> ModelResponse:
        """Query Anthropic Claude"""
        client = self._get_anthropic_client()
        if not client:
            return ModelResponse(
                model=ModelType.ANTHROPIC_CLAUDE,
                response="Anthropic client not available",
                confidence=0.0
            )

        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )

        return ModelResponse(
            model=ModelType.ANTHROPIC_CLAUDE,
            response=response.content[0].text,
            confidence=0.88,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens
        )

    async def _query_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> ModelResponse:
        """Query Google Gemini"""
        # Placeholder - would integrate with Google AI Studio
        return ModelResponse(
            model=ModelType.GOOGLE_GEMINI,
            response="Gemini integration pending",
            confidence=0.0,
            metadata={"available": False}
        )

    def get_model_weight(self, model: ModelType) -> float:
        """Get weight for a model"""
        return self._model_weights.get(model, 0.5)


class ConsensusEngine:
    """Engine for reaching consensus between models"""

    def __init__(self, model_provider: ModelProvider):
        self.model_provider = model_provider
        self.strategies: Dict[ConsensusStrategy, Callable] = {
            ConsensusStrategy.MAJORITY_VOTE: self._majority_vote,
            ConsensusStrategy.WEIGHTED_AVERAGE: self._weighted_average,
            ConsensusStrategy.HIGHEST_CONFIDENCE: self._highest_confidence,
            ConsensusStrategy.ENSEMBLE: self._ensemble,
            ConsensusStrategy.DEBATE: self._debate,
            ConsensusStrategy.HIERARCHICAL: self._hierarchical
        }

    async def reach_consensus(
        self,
        prompt: str,
        models: List[ModelType],
        strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_AVERAGE,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        timeout: float = 30.0
    ) -> ConsensusResult:
        """Reach consensus between multiple models"""
        consensus_id = str(uuid.uuid4())

        try:
            # Query all models concurrently
            tasks = [
                self.model_provider.query_model(
                    model, prompt, system_prompt, max_tokens
                )
                for model in models
            ]

            # Wait for responses with timeout
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )

            # Filter valid responses
            valid_responses = [
                r for r in responses
                if isinstance(r, ModelResponse) and r.confidence > 0
            ]

            if not valid_responses:
                return ConsensusResult(
                    consensus_id=consensus_id,
                    final_response="No valid responses received",
                    confidence=0.0,
                    strategy_used=strategy,
                    status=ConsensusStatus.FAILED,
                    participating_models=models,
                    model_responses=[],
                    agreement_score=0.0,
                    dissenting_opinions=[]
                )

            # Apply consensus strategy
            strategy_func = self.strategies.get(
                strategy, self._weighted_average
            )
            result = await strategy_func(valid_responses, prompt)
            result.consensus_id = consensus_id

            return result

        except asyncio.TimeoutError:
            return ConsensusResult(
                consensus_id=consensus_id,
                final_response="Consensus process timed out",
                confidence=0.0,
                strategy_used=strategy,
                status=ConsensusStatus.TIMEOUT,
                participating_models=models,
                model_responses=[],
                agreement_score=0.0,
                dissenting_opinions=[]
            )
        except Exception as e:
            logger.error(f"Error in consensus process: {e}")
            return ConsensusResult(
                consensus_id=consensus_id,
                final_response=str(e),
                confidence=0.0,
                strategy_used=strategy,
                status=ConsensusStatus.FAILED,
                participating_models=models,
                model_responses=[],
                agreement_score=0.0,
                dissenting_opinions=[]
            )

    async def _majority_vote(
        self,
        responses: List[ModelResponse],
        prompt: str
    ) -> ConsensusResult:
        """Simple majority vote consensus"""
        if not responses:
            return self._empty_result(ConsensusStrategy.MAJORITY_VOTE)

        # Use the response with highest occurrence (simplified)
        best_response = max(responses, key=lambda r: r.confidence)

        return ConsensusResult(
            consensus_id="",
            final_response=best_response.response,
            confidence=best_response.confidence,
            strategy_used=ConsensusStrategy.MAJORITY_VOTE,
            status=ConsensusStatus.CONSENSUS_REACHED,
            participating_models=[r.model for r in responses],
            model_responses=responses,
            agreement_score=self._calculate_agreement(responses),
            dissenting_opinions=self._find_dissenters(responses, best_response)
        )

    async def _weighted_average(
        self,
        responses: List[ModelResponse],
        prompt: str
    ) -> ConsensusResult:
        """Weighted average based on model reliability and confidence"""
        if not responses:
            return self._empty_result(ConsensusStrategy.WEIGHTED_AVERAGE)

        # Calculate weighted scores
        weighted_responses = []
        for response in responses:
            weight = self.model_provider.get_model_weight(response.model)
            score = response.confidence * weight
            weighted_responses.append((response, score))

        # Select best weighted response
        best = max(weighted_responses, key=lambda x: x[1])
        weighted_confidence = sum(wr[1] for wr in weighted_responses) / len(weighted_responses)

        return ConsensusResult(
            consensus_id="",
            final_response=best[0].response,
            confidence=weighted_confidence,
            strategy_used=ConsensusStrategy.WEIGHTED_AVERAGE,
            status=ConsensusStatus.CONSENSUS_REACHED,
            participating_models=[r.model for r in responses],
            model_responses=responses,
            agreement_score=self._calculate_agreement(responses),
            dissenting_opinions=self._find_dissenters(responses, best[0])
        )

    async def _highest_confidence(
        self,
        responses: List[ModelResponse],
        prompt: str
    ) -> ConsensusResult:
        """Select response with highest confidence"""
        if not responses:
            return self._empty_result(ConsensusStrategy.HIGHEST_CONFIDENCE)

        best = max(responses, key=lambda r: r.confidence)

        return ConsensusResult(
            consensus_id="",
            final_response=best.response,
            confidence=best.confidence,
            strategy_used=ConsensusStrategy.HIGHEST_CONFIDENCE,
            status=ConsensusStatus.CONSENSUS_REACHED,
            participating_models=[r.model for r in responses],
            model_responses=responses,
            agreement_score=self._calculate_agreement(responses),
            dissenting_opinions=self._find_dissenters(responses, best)
        )

    async def _ensemble(
        self,
        responses: List[ModelResponse],
        prompt: str
    ) -> ConsensusResult:
        """Combine responses into ensemble answer"""
        if not responses:
            return self._empty_result(ConsensusStrategy.ENSEMBLE)

        # Combine all responses into a synthesized answer
        combined_parts = []
        for response in responses:
            if response.response:
                combined_parts.append(f"[{response.model.value}]: {response.response}")

        combined = "\n\n".join(combined_parts)
        avg_confidence = sum(r.confidence for r in responses) / len(responses)

        return ConsensusResult(
            consensus_id="",
            final_response=combined,
            confidence=avg_confidence,
            strategy_used=ConsensusStrategy.ENSEMBLE,
            status=ConsensusStatus.CONSENSUS_REACHED,
            participating_models=[r.model for r in responses],
            model_responses=responses,
            agreement_score=self._calculate_agreement(responses),
            dissenting_opinions=[]
        )

    async def _debate(
        self,
        responses: List[ModelResponse],
        prompt: str
    ) -> ConsensusResult:
        """Have models debate and refine answers"""
        if len(responses) < 2:
            return await self._highest_confidence(responses, prompt)

        # Get top two responses
        sorted_responses = sorted(responses, key=lambda r: r.confidence, reverse=True)
        top_two = sorted_responses[:2]

        # Create debate prompt for synthesis
        debate_prompt = f"""
        Two AI models provided different responses to this question:

        Original Question: {prompt}

        Response 1 ({top_two[0].model.value}): {top_two[0].response}

        Response 2 ({top_two[1].model.value}): {top_two[1].response}

        Synthesize these responses into a single, well-reasoned answer that
        incorporates the best elements of both.
        """

        # Query the highest-confidence model for synthesis
        synthesis = await self.model_provider.query_model(
            top_two[0].model,
            debate_prompt,
            "You are an expert at synthesizing multiple perspectives into a coherent answer."
        )

        return ConsensusResult(
            consensus_id="",
            final_response=synthesis.response,
            confidence=max(r.confidence for r in top_two),
            strategy_used=ConsensusStrategy.DEBATE,
            status=ConsensusStatus.CONSENSUS_REACHED,
            participating_models=[r.model for r in responses],
            model_responses=responses + [synthesis],
            agreement_score=self._calculate_agreement(responses),
            dissenting_opinions=[]
        )

    async def _hierarchical(
        self,
        responses: List[ModelResponse],
        prompt: str
    ) -> ConsensusResult:
        """Hierarchical consensus with verification"""
        if not responses:
            return self._empty_result(ConsensusStrategy.HIERARCHICAL)

        # Sort by weight
        weighted = [
            (r, self.model_provider.get_model_weight(r.model))
            for r in responses
        ]
        weighted.sort(key=lambda x: x[1], reverse=True)

        # Primary answer from highest-weight model
        primary = weighted[0][0]

        # Check if other models agree
        agreements = sum(
            1 for r, _ in weighted[1:]
            if self._responses_similar(primary.response, r.response)
        )

        agreement_ratio = agreements / max(len(weighted) - 1, 1)
        status = (
            ConsensusStatus.CONSENSUS_REACHED if agreement_ratio > 0.5
            else ConsensusStatus.PARTIAL_CONSENSUS if agreement_ratio > 0.25
            else ConsensusStatus.DISAGREEMENT
        )

        return ConsensusResult(
            consensus_id="",
            final_response=primary.response,
            confidence=primary.confidence * (0.5 + 0.5 * agreement_ratio),
            strategy_used=ConsensusStrategy.HIERARCHICAL,
            status=status,
            participating_models=[r.model for r in responses],
            model_responses=responses,
            agreement_score=agreement_ratio,
            dissenting_opinions=self._find_dissenters(responses, primary)
        )

    def _calculate_agreement(self, responses: List[ModelResponse]) -> float:
        """Calculate agreement score between responses"""
        if len(responses) < 2:
            return 1.0

        # Simple similarity based on response length and content overlap
        texts = [r.response.lower() for r in responses]
        words_sets = [set(t.split()) for t in texts]

        total_similarity = 0
        comparisons = 0

        for i in range(len(words_sets)):
            for j in range(i + 1, len(words_sets)):
                intersection = len(words_sets[i] & words_sets[j])
                union = len(words_sets[i] | words_sets[j])
                if union > 0:
                    total_similarity += intersection / union
                comparisons += 1

        return total_similarity / max(comparisons, 1)

    def _responses_similar(self, resp1: str, resp2: str, threshold: float = 0.3) -> bool:
        """Check if two responses are similar"""
        words1 = set(resp1.lower().split())
        words2 = set(resp2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return (intersection / max(union, 1)) > threshold

    def _find_dissenters(
        self,
        responses: List[ModelResponse],
        consensus: ModelResponse
    ) -> List[Dict]:
        """Find responses that disagree with consensus"""
        dissenters = []
        for response in responses:
            if response.model != consensus.model:
                if not self._responses_similar(response.response, consensus.response):
                    dissenters.append({
                        "model": response.model.value,
                        "response": response.response[:200],
                        "confidence": response.confidence
                    })
        return dissenters

    def _empty_result(self, strategy: ConsensusStrategy) -> ConsensusResult:
        """Create empty result for failed consensus"""
        return ConsensusResult(
            consensus_id="",
            final_response="",
            confidence=0.0,
            strategy_used=strategy,
            status=ConsensusStatus.FAILED,
            participating_models=[],
            model_responses=[],
            agreement_score=0.0,
            dissenting_opinions=[]
        )


class MultiModelConsensusSystem:
    """Main multi-model consensus system"""

    def __init__(self):
        self.model_provider = ModelProvider()
        self.consensus_engine = ConsensusEngine(self.model_provider)
        self.conn = None
        self._init_database()

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**DB_CONFIG)
        return self.conn

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_consensus_requests (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    prompt TEXT NOT NULL,
                    strategy VARCHAR(50),
                    models_requested JSONB DEFAULT '[]',
                    final_response TEXT,
                    confidence FLOAT,
                    agreement_score FLOAT,
                    status VARCHAR(50),
                    model_responses JSONB DEFAULT '[]',
                    dissenting_opinions JSONB DEFAULT '[]',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_model_performance (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_type VARCHAR(50),
                    total_queries INT DEFAULT 0,
                    successful_queries INT DEFAULT 0,
                    avg_confidence FLOAT DEFAULT 0.0,
                    avg_latency_ms FLOAT DEFAULT 0.0,
                    consensus_contributions INT DEFAULT 0,
                    dissent_count INT DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_consensus_status
                ON ai_consensus_requests(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_consensus_created
                ON ai_consensus_requests(created_at)
            """)

            conn.commit()
            cursor.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    async def get_consensus(
        self,
        prompt: str,
        models: Optional[List[ModelType]] = None,
        strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_AVERAGE,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        timeout: float = 30.0
    ) -> ConsensusResult:
        """Get consensus from multiple AI models"""
        if models is None:
            models = [
                ModelType.OPENAI_GPT4,
                ModelType.ANTHROPIC_CLAUDE,
                ModelType.OPENAI_GPT4O
            ]

        # Log request
        request_id = str(uuid.uuid4())
        await self._log_request(request_id, prompt, strategy, models)

        # Get consensus
        result = await self.consensus_engine.reach_consensus(
            prompt=prompt,
            models=models,
            strategy=strategy,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            timeout=timeout
        )

        # Log result
        await self._log_result(request_id, result)

        # Update model performance
        await self._update_model_performance(result)

        return result

    async def _log_request(
        self,
        request_id: str,
        prompt: str,
        strategy: ConsensusStrategy,
        models: List[ModelType]
    ):
        """Log consensus request"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_consensus_requests
                (id, prompt, strategy, models_requested, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                request_id,
                prompt,
                strategy.value,
                Json([m.value for m in models]),
                ConsensusStatus.IN_PROGRESS.value
            ))

            conn.commit()
            cursor.close()

        except Exception as e:
            logger.error(f"Failed to log request: {e}")

    async def _log_result(self, request_id: str, result: ConsensusResult):
        """Log consensus result"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            model_responses_data = [
                {
                    "model": r.model.value,
                    "response": r.response[:500],  # Truncate for storage
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms,
                    "tokens_used": r.tokens_used
                }
                for r in result.model_responses
            ]

            cursor.execute("""
                UPDATE ai_consensus_requests
                SET final_response = %s,
                    confidence = %s,
                    agreement_score = %s,
                    status = %s,
                    model_responses = %s,
                    dissenting_opinions = %s,
                    completed_at = NOW()
                WHERE id = %s
            """, (
                result.final_response,
                result.confidence,
                result.agreement_score,
                result.status.value,
                Json(model_responses_data),
                Json(result.dissenting_opinions),
                request_id
            ))

            conn.commit()
            cursor.close()

        except Exception as e:
            logger.error(f"Failed to log result: {e}")

    async def _update_model_performance(self, result: ConsensusResult):
        """Update model performance metrics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            for response in result.model_responses:
                cursor.execute("""
                    INSERT INTO ai_model_performance (model_type, total_queries,
                        successful_queries, avg_confidence, avg_latency_ms,
                        consensus_contributions, dissent_count)
                    VALUES (%s, 1, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_type) DO UPDATE SET
                        total_queries = ai_model_performance.total_queries + 1,
                        successful_queries = ai_model_performance.successful_queries +
                            CASE WHEN %s THEN 1 ELSE 0 END,
                        avg_confidence = (ai_model_performance.avg_confidence *
                            ai_model_performance.total_queries + %s) /
                            (ai_model_performance.total_queries + 1),
                        avg_latency_ms = (ai_model_performance.avg_latency_ms *
                            ai_model_performance.total_queries + %s) /
                            (ai_model_performance.total_queries + 1),
                        consensus_contributions = ai_model_performance.consensus_contributions +
                            CASE WHEN %s THEN 1 ELSE 0 END,
                        dissent_count = ai_model_performance.dissent_count +
                            CASE WHEN %s THEN 1 ELSE 0 END,
                        updated_at = NOW()
                """, (
                    response.model.value,
                    1 if response.confidence > 0 else 0,
                    response.confidence,
                    response.latency_ms,
                    1 if response.model.value == result.model_responses[0].model.value else 0,
                    1 if any(d['model'] == response.model.value for d in result.dissenting_opinions) else 0,
                    response.confidence > 0,
                    response.confidence,
                    response.latency_ms,
                    response.model.value == result.model_responses[0].model.value,
                    any(d['model'] == response.model.value for d in result.dissenting_opinions)
                ))

            conn.commit()
            cursor.close()

        except Exception as e:
            logger.error(f"Failed to update model performance: {e}")

    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for all models"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM ai_model_performance
                ORDER BY total_queries DESC
            """)

            stats = cursor.fetchall()
            cursor.close()

            return {
                "models": [dict(s) for s in stats],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get model statistics: {e}")
            return {"models": [], "error": str(e)}

    async def get_consensus_history(
        self,
        limit: int = 50,
        status: Optional[ConsensusStatus] = None
    ) -> List[Dict]:
        """Get history of consensus requests"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = """
                SELECT id, prompt, strategy, status, confidence,
                       agreement_score, created_at, completed_at
                FROM ai_consensus_requests
            """
            params = []

            if status:
                query += " WHERE status = %s"
                params.append(status.value)

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            history = cursor.fetchall()
            cursor.close()

            return [dict(h) for h in history]

        except Exception as e:
            logger.error(f"Failed to get consensus history: {e}")
            return []


# Singleton instance
_consensus_system: Optional[MultiModelConsensusSystem] = None


def get_multi_model_consensus():
    """Get or create the multi-model consensus system instance"""
    global _consensus_system
    if _consensus_system is None:
        _consensus_system = MultiModelConsensusSystem()
    return _consensus_system


# Export main components
__all__ = [
    'MultiModelConsensusSystem',
    'get_multi_model_consensus',
    'ConsensusStrategy',
    'ConsensusStatus',
    'ModelType',
    'ConsensusResult',
    'ModelResponse'
]

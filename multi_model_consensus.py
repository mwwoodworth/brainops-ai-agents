#!/usr/bin/env python3
"""
Multi-Model Consensus System - Task 16
System for getting consensus from multiple AI models for improved accuracy and reliability
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - validate required environment variables
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "5432"))
    }


class ConsensusStrategy(Enum):
    """Strategies for reaching consensus"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    UNANIMOUS = "unanimous"
    HIGHEST_CONFIDENCE = "highest_confidence"
    ENSEMBLE = "ensemble"
    DEBATE = "debate"  # Simple single-round synthesis
    MULTI_ROUND_DEBATE = "multi_round_debate"  # Full structured debate with rebuttals
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
    metadata: dict = field(default_factory=dict)


@dataclass
class DebatePosition:
    """A model's position in a debate round"""
    model: ModelType
    position: str
    key_arguments: list[str]
    confidence: float
    round_number: int
    is_rebuttal: bool = False
    responding_to: Optional[str] = None  # Model being rebutted
    metadata: dict = field(default_factory=dict)


@dataclass
class DebateRound:
    """A single round of debate"""
    round_number: int
    round_type: str  # "initial", "rebuttal", "synthesis"
    positions: list[DebatePosition]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    round_summary: Optional[str] = None


@dataclass
class DebateResult:
    """Complete result of a multi-round debate"""
    debate_id: str
    topic: str
    rounds: list[DebateRound]
    final_consensus: str
    confidence: float
    participating_models: list[ModelType]
    agreement_evolution: list[float]  # Agreement score per round
    key_points_of_agreement: list[str]
    remaining_disagreements: list[str]
    total_rounds: int
    debate_duration_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Result of consensus process"""
    consensus_id: str
    final_response: str
    confidence: float
    strategy_used: ConsensusStrategy
    status: ConsensusStatus
    participating_models: list[ModelType]
    model_responses: list[ModelResponse]
    agreement_score: float
    dissenting_opinions: list[dict]
    metadata: dict = field(default_factory=dict)


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
        try:
            from google import genai
            from google.genai import types as _genai_types

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return ModelResponse(
                    model=ModelType.GOOGLE_GEMINI,
                    response="Google API key not configured",
                    confidence=0.0,
                    metadata={"available": False}
                )

            client = genai.Client(api_key=api_key)

            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Run in thread pool for async compatibility
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=full_prompt,
                    config=_genai_types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
            )

            return ModelResponse(
                model=ModelType.GOOGLE_GEMINI,
                response=response.text,
                confidence=0.82,  # Gemini doesn't provide confidence scores
                metadata={"available": True}
            )

        except ImportError:
            return ModelResponse(
                model=ModelType.GOOGLE_GEMINI,
                response="Google GenAI SDK not installed",
                confidence=0.0,
                metadata={"available": False, "error": "ImportError"}
            )
        except Exception as e:
            logger.error(f"Gemini query failed: {e}")
            return ModelResponse(
                model=ModelType.GOOGLE_GEMINI,
                response=f"Gemini query failed: {str(e)}",
                confidence=0.0,
                metadata={"available": False, "error": str(e)}
            )

    def get_model_weight(self, model: ModelType) -> float:
        """Get weight for a model"""
        return self._model_weights.get(model, 0.5)


class ConsensusEngine:
    """Engine for reaching consensus between models"""

    def __init__(self, model_provider: ModelProvider):
        self.model_provider = model_provider
        self.strategies: dict[ConsensusStrategy, Callable] = {
            ConsensusStrategy.MAJORITY_VOTE: self._majority_vote,
            ConsensusStrategy.WEIGHTED_AVERAGE: self._weighted_average,
            ConsensusStrategy.HIGHEST_CONFIDENCE: self._highest_confidence,
            ConsensusStrategy.ENSEMBLE: self._ensemble,
            ConsensusStrategy.DEBATE: self._debate,
            ConsensusStrategy.MULTI_ROUND_DEBATE: self._multi_round_debate_strategy,
            ConsensusStrategy.HIERARCHICAL: self._hierarchical
        }

    async def reach_consensus(
        self,
        prompt: str,
        models: list[ModelType],
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
        responses: list[ModelResponse],
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
        responses: list[ModelResponse],
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
        responses: list[ModelResponse],
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
        responses: list[ModelResponse],
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
        responses: list[ModelResponse],
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

    async def _multi_round_debate_strategy(
        self,
        responses: list[ModelResponse],
        prompt: str
    ) -> ConsensusResult:
        """
        Wrapper to call multi_round_debate from the strategy pattern.
        Converts DebateResult to ConsensusResult.
        """
        if len(responses) < 2:
            return await self._highest_confidence(responses, prompt)

        # Extract models from responses
        models = [r.model for r in responses]

        # Run the full multi-round debate
        debate_result = await self.multi_round_debate(
            topic=prompt,
            models=models,
            rounds=3,
            initial_responses=responses
        )

        # Convert DebateResult to ConsensusResult
        all_responses = responses.copy()

        # Add synthesis responses from debate rounds
        for round_data in debate_result.rounds:
            for position in round_data.positions:
                all_responses.append(ModelResponse(
                    model=position.model,
                    response=position.position,
                    confidence=position.confidence,
                    reasoning="; ".join(position.key_arguments),
                    metadata={
                        "round": position.round_number,
                        "is_rebuttal": position.is_rebuttal,
                        "round_type": round_data.round_type
                    }
                ))

        # Determine status based on remaining disagreements
        if not debate_result.remaining_disagreements:
            status = ConsensusStatus.CONSENSUS_REACHED
        elif len(debate_result.remaining_disagreements) <= 2:
            status = ConsensusStatus.PARTIAL_CONSENSUS
        else:
            status = ConsensusStatus.DISAGREEMENT

        return ConsensusResult(
            consensus_id=debate_result.debate_id,
            final_response=debate_result.final_consensus,
            confidence=debate_result.confidence,
            strategy_used=ConsensusStrategy.MULTI_ROUND_DEBATE,
            status=status,
            participating_models=debate_result.participating_models,
            model_responses=all_responses,
            agreement_score=debate_result.agreement_evolution[-1] if debate_result.agreement_evolution else 0.0,
            dissenting_opinions=[
                {"disagreement": d} for d in debate_result.remaining_disagreements
            ],
            metadata={
                "total_rounds": debate_result.total_rounds,
                "debate_duration_ms": debate_result.debate_duration_ms,
                "key_points_of_agreement": debate_result.key_points_of_agreement,
                "agreement_evolution": debate_result.agreement_evolution
            }
        )

    async def multi_round_debate(
        self,
        topic: str,
        models: Optional[list[ModelType]] = None,
        rounds: int = 3,
        initial_responses: Optional[list[ModelResponse]] = None,
        system_prompt: Optional[str] = None
    ) -> DebateResult:
        """
        Conduct a structured multi-round debate between AI models.

        Round 1: Initial positions from each model
        Round 2: Rebuttals and counter-arguments
        Round 3: Final synthesis and consensus building

        Args:
            topic: The topic/question to debate
            models: List of models to participate (defaults to available models)
            rounds: Number of debate rounds (default 3)
            initial_responses: Pre-existing responses to use for Round 1
            system_prompt: Optional system context

        Returns:
            DebateResult with complete debate history and final consensus
        """
        start_time = datetime.now(timezone.utc)
        debate_id = str(uuid.uuid4())

        if models is None:
            models = [
                ModelType.OPENAI_GPT4,
                ModelType.ANTHROPIC_CLAUDE,
                ModelType.OPENAI_GPT4O
            ]

        debate_rounds: list[DebateRound] = []
        all_positions: dict[ModelType, list[DebatePosition]] = {m: [] for m in models}
        agreement_evolution: list[float] = []

        logger.info(f"Starting multi-round debate on topic: {topic[:100]}...")

        # ============================================
        # ROUND 1: INITIAL POSITIONS
        # ============================================
        round1_positions = []

        if initial_responses:
            # Use provided initial responses
            for response in initial_responses:
                if response.model in models:
                    position = DebatePosition(
                        model=response.model,
                        position=response.response,
                        key_arguments=self._extract_key_arguments(response.response),
                        confidence=response.confidence,
                        round_number=1,
                        is_rebuttal=False
                    )
                    round1_positions.append(position)
                    all_positions[response.model].append(position)
        else:
            # Query each model for initial position
            initial_prompt = f"""
You are participating in a structured debate. Present your initial position on the following topic.

TOPIC: {topic}

Provide:
1. Your clear position/answer
2. 3-5 key arguments supporting your position
3. Any important caveats or conditions

Be specific and well-reasoned. Other AI models will review and potentially challenge your position.
"""
            tasks = [
                self.model_provider.query_model(
                    model, initial_prompt, system_prompt
                )
                for model in models
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for model, response in zip(models, responses):
                if isinstance(response, ModelResponse) and response.confidence > 0:
                    position = DebatePosition(
                        model=model,
                        position=response.response,
                        key_arguments=self._extract_key_arguments(response.response),
                        confidence=response.confidence,
                        round_number=1,
                        is_rebuttal=False
                    )
                    round1_positions.append(position)
                    all_positions[model].append(position)

        round1 = DebateRound(
            round_number=1,
            round_type="initial",
            positions=round1_positions,
            round_summary=f"Initial positions from {len(round1_positions)} models"
        )
        debate_rounds.append(round1)

        # Calculate initial agreement
        initial_agreement = self._calculate_position_agreement(round1_positions)
        agreement_evolution.append(initial_agreement)

        logger.info(f"Round 1 complete. Initial agreement: {initial_agreement:.2%}")

        if len(round1_positions) < 2:
            # Not enough participants for debate
            return self._create_debate_result(
                debate_id, topic, debate_rounds, round1_positions,
                agreement_evolution, models, start_time
            )

        # ============================================
        # ROUND 2: REBUTTALS AND COUNTER-ARGUMENTS
        # ============================================
        if rounds >= 2:
            round2_positions = []

            for current_model in models:
                if current_model not in [p.model for p in round1_positions]:
                    continue

                # Get other models' positions
                other_positions = [
                    p for p in round1_positions
                    if p.model != current_model
                ]

                if not other_positions:
                    continue

                # Format other positions for rebuttal
                positions_text = "\n\n".join([
                    f"**{p.model.value}**: {p.position}\n"
                    f"Key arguments: {', '.join(p.key_arguments[:3])}"
                    for p in other_positions
                ])

                # Get current model's own position
                own_position = next(
                    (p for p in round1_positions if p.model == current_model),
                    None
                )

                rebuttal_prompt = f"""
You are in Round 2 of a structured debate. Review the other participants' positions and provide rebuttals.

ORIGINAL TOPIC: {topic}

YOUR INITIAL POSITION:
{own_position.position if own_position else "Not available"}

OTHER PARTICIPANTS' POSITIONS:
{positions_text}

Provide a rebuttal that:
1. Identifies weaknesses or gaps in other arguments
2. Addresses any valid criticisms of your position
3. Strengthens your argument with additional evidence or reasoning
4. Acknowledges any points of agreement
5. Clarifies any misunderstandings

Be constructive but rigorous. The goal is to reach the best possible answer through debate.
"""

                response = await self.model_provider.query_model(
                    current_model, rebuttal_prompt, system_prompt
                )

                if response.confidence > 0:
                    position = DebatePosition(
                        model=current_model,
                        position=response.response,
                        key_arguments=self._extract_key_arguments(response.response),
                        confidence=response.confidence,
                        round_number=2,
                        is_rebuttal=True,
                        responding_to=", ".join([p.model.value for p in other_positions])
                    )
                    round2_positions.append(position)
                    all_positions[current_model].append(position)

            round2 = DebateRound(
                round_number=2,
                round_type="rebuttal",
                positions=round2_positions,
                round_summary=f"Rebuttals from {len(round2_positions)} models"
            )
            debate_rounds.append(round2)

            # Calculate agreement after rebuttals
            rebuttal_agreement = self._calculate_position_agreement(round2_positions)
            agreement_evolution.append(rebuttal_agreement)

            logger.info(f"Round 2 complete. Agreement after rebuttals: {rebuttal_agreement:.2%}")

        # ============================================
        # ROUND 3: SYNTHESIS AND CONSENSUS BUILDING
        # ============================================
        if rounds >= 3:
            round3_positions = []

            # Collect all arguments from previous rounds
            all_arguments = []
            for model, positions in all_positions.items():
                for pos in positions:
                    all_arguments.extend([
                        f"[{model.value}, Round {pos.round_number}] {arg}"
                        for arg in pos.key_arguments
                    ])

            # Format debate history
            debate_history = self._format_debate_history(debate_rounds)

            for current_model in models:
                if current_model not in [p.model for p in round1_positions]:
                    continue

                synthesis_prompt = f"""
You are in the FINAL ROUND of a structured debate. Your goal is to synthesize all perspectives into a coherent consensus.

ORIGINAL TOPIC: {topic}

COMPLETE DEBATE HISTORY:
{debate_history}

ALL KEY ARGUMENTS RAISED:
{chr(10).join(all_arguments[:20])}

Provide a FINAL SYNTHESIS that:
1. Identifies the core points of agreement across all participants
2. Acknowledges remaining disagreements (if any)
3. Presents the strongest, most well-supported answer
4. Integrates the best arguments from each participant
5. Notes any conditions or caveats that emerged

Your response should represent what a reasonable observer would conclude after watching this entire debate.
Rate your confidence in this synthesis (0-100%).
"""

                response = await self.model_provider.query_model(
                    current_model, synthesis_prompt, system_prompt
                )

                if response.confidence > 0:
                    position = DebatePosition(
                        model=current_model,
                        position=response.response,
                        key_arguments=self._extract_key_arguments(response.response),
                        confidence=response.confidence,
                        round_number=3,
                        is_rebuttal=False
                    )
                    round3_positions.append(position)
                    all_positions[current_model].append(position)

            round3 = DebateRound(
                round_number=3,
                round_type="synthesis",
                positions=round3_positions,
                round_summary=f"Final synthesis from {len(round3_positions)} models"
            )
            debate_rounds.append(round3)

            # Calculate final agreement
            final_agreement = self._calculate_position_agreement(round3_positions)
            agreement_evolution.append(final_agreement)

            logger.info(f"Round 3 complete. Final agreement: {final_agreement:.2%}")

        return self._create_debate_result(
            debate_id, topic, debate_rounds,
            debate_rounds[-1].positions if debate_rounds else [],
            agreement_evolution, models, start_time
        )

    def _extract_key_arguments(self, text: str, max_args: int = 5) -> list[str]:
        """Extract key arguments from a response text"""
        arguments = []

        # Look for numbered points
        import re
        numbered_pattern = r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-•]\s*)(.+?)(?=\n|$)'
        matches = re.findall(numbered_pattern, text, re.MULTILINE)

        for match in matches[:max_args]:
            cleaned = match.strip()
            if len(cleaned) > 20 and len(cleaned) < 500:  # Reasonable argument length
                arguments.append(cleaned)

        # If not enough numbered points, extract sentences
        if len(arguments) < 2:
            sentences = text.split('.')
            for sentence in sentences:
                cleaned = sentence.strip()
                if len(cleaned) > 30 and len(cleaned) < 300:
                    if any(word in cleaned.lower() for word in
                           ['because', 'therefore', 'however', 'important', 'key', 'main']):
                        arguments.append(cleaned)
                        if len(arguments) >= max_args:
                            break

        return arguments[:max_args] if arguments else ["Position stated without explicit arguments"]

    def _calculate_position_agreement(self, positions: list[DebatePosition]) -> float:
        """Calculate agreement score between debate positions"""
        if len(positions) < 2:
            return 1.0

        texts = [p.position.lower() for p in positions]
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

    def _format_debate_history(self, rounds: list[DebateRound]) -> str:
        """Format debate history for synthesis prompt"""
        history_parts = []

        for round_data in rounds:
            history_parts.append(f"\n=== ROUND {round_data.round_number}: {round_data.round_type.upper()} ===")
            for position in round_data.positions:
                history_parts.append(f"\n[{position.model.value}]:")
                history_parts.append(position.position[:800])  # Truncate long positions
                if position.is_rebuttal and position.responding_to:
                    history_parts.append(f"  (Responding to: {position.responding_to})")

        return "\n".join(history_parts)

    def _create_debate_result(
        self,
        debate_id: str,
        topic: str,
        rounds: list[DebateRound],
        final_positions: list[DebatePosition],
        agreement_evolution: list[float],
        models: list[ModelType],
        start_time: datetime
    ) -> DebateResult:
        """Create the final debate result"""

        # Build final consensus from synthesis positions
        if final_positions:
            # Use weighted combination of final positions
            weighted_positions = [
                (p, self.model_provider.get_model_weight(p.model) * p.confidence)
                for p in final_positions
            ]
            weighted_positions.sort(key=lambda x: x[1], reverse=True)

            # Use highest-weighted synthesis as primary consensus
            best_position = weighted_positions[0][0]
            final_consensus = best_position.position
            final_confidence = best_position.confidence

            # Extract points of agreement and disagreement
            all_arguments = []
            for pos in final_positions:
                all_arguments.extend(pos.key_arguments)

            # Find common themes (simplified)
            argument_words = [set(arg.lower().split()) for arg in all_arguments]
            common_words = set.intersection(*argument_words) if argument_words else set()

            key_agreements = list(set([
                arg for arg in all_arguments
                if len(common_words & set(arg.lower().split())) > 3
            ]))[:5]

            # Find disagreements by checking low agreement
            remaining_disagreements = []
            if agreement_evolution and agreement_evolution[-1] < 0.7:
                # There's still disagreement
                for pos in final_positions[1:]:
                    if not self._responses_similar(pos.position, best_position.position):
                        remaining_disagreements.append(
                            f"{pos.model.value} disagrees: {pos.key_arguments[0] if pos.key_arguments else 'Different conclusion'}"
                        )
        else:
            final_consensus = "Debate could not reach a conclusion due to insufficient participation."
            final_confidence = 0.0
            key_agreements = []
            remaining_disagreements = ["Insufficient participation"]

        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return DebateResult(
            debate_id=debate_id,
            topic=topic,
            rounds=rounds,
            final_consensus=final_consensus,
            confidence=final_confidence,
            participating_models=[p.model for p in final_positions] if final_positions else models,
            agreement_evolution=agreement_evolution,
            key_points_of_agreement=key_agreements,
            remaining_disagreements=remaining_disagreements,
            total_rounds=len(rounds),
            debate_duration_ms=duration_ms
        )

    async def _hierarchical(
        self,
        responses: list[ModelResponse],
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

    def _calculate_agreement(self, responses: list[ModelResponse]) -> float:
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
        responses: list[ModelResponse],
        consensus: ModelResponse
    ) -> list[dict]:
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
            self.conn = psycopg2.connect(**_get_db_config())
        return self.conn

    def _init_database(self):
        """Verify required tables exist (DDL removed — agent_worker has no DDL permissions)."""
        required_tables = [
                "ai_consensus_requests",
                "ai_model_performance",
        ]
        try:
            from database.verify_tables import verify_tables_sync
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()
            ok = verify_tables_sync(required_tables, cursor, module_name="multi_model_consensus")
            cursor.close()
            conn.close()
            if not ok:
                return
            self._tables_initialized = True
        except Exception as exc:
            logger.error("Table verification failed: %s", exc)
    async def get_consensus(
        self,
        prompt: str,
        models: Optional[list[ModelType]] = None,
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

    async def run_multi_round_debate(
        self,
        topic: str,
        models: Optional[list[ModelType]] = None,
        rounds: int = 3,
        system_prompt: Optional[str] = None,
        timeout: float = 120.0
    ) -> DebateResult:
        """
        Run a full multi-round debate on a topic.

        This is the main entry point for conducting structured debates between
        multiple AI models with proper argumentation, rebuttals, and synthesis.

        Args:
            topic: The topic/question to debate
            models: List of models to participate (defaults to GPT-4, Claude, GPT-4o)
            rounds: Number of debate rounds (default 3: initial, rebuttal, synthesis)
            system_prompt: Optional system context for all models
            timeout: Maximum time for debate in seconds (default 120s)

        Returns:
            DebateResult with complete debate history and final consensus

        Example:
            >>> system = get_multi_model_consensus()
            >>> result = await system.run_multi_round_debate(
            ...     topic="What is the best approach for implementing microservices?",
            ...     models=[ModelType.OPENAI_GPT4, ModelType.ANTHROPIC_CLAUDE],
            ...     rounds=3
            ... )
            >>> print(result.final_consensus)
            >>> print(f"Agreement evolution: {result.agreement_evolution}")
        """
        if models is None:
            models = [
                ModelType.OPENAI_GPT4,
                ModelType.ANTHROPIC_CLAUDE,
                ModelType.OPENAI_GPT4O
            ]

        logger.info(f"Starting multi-round debate with {len(models)} models on: {topic[:100]}...")

        try:
            # Run the debate with timeout
            result = await asyncio.wait_for(
                self.consensus_engine.multi_round_debate(
                    topic=topic,
                    models=models,
                    rounds=rounds,
                    system_prompt=system_prompt
                ),
                timeout=timeout
            )

            # Log the debate result
            await self._log_debate_result(result)

            return result

        except asyncio.TimeoutError:
            logger.error(f"Multi-round debate timed out after {timeout}s")
            return DebateResult(
                debate_id=str(uuid.uuid4()),
                topic=topic,
                rounds=[],
                final_consensus="Debate timed out before reaching consensus",
                confidence=0.0,
                participating_models=models,
                agreement_evolution=[],
                key_points_of_agreement=[],
                remaining_disagreements=["Debate timed out"],
                total_rounds=0,
                debate_duration_ms=timeout * 1000,
                metadata={"error": "timeout"}
            )

    async def _log_debate_result(self, result: DebateResult):
        """Log debate result to database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Store debate in consensus_requests table with special metadata
            rounds_data = [
                {
                    "round_number": r.round_number,
                    "round_type": r.round_type,
                    "positions": [
                        {
                            "model": p.model.value,
                            "position": p.position[:500],
                            "key_arguments": p.key_arguments[:3],
                            "confidence": p.confidence,
                            "is_rebuttal": p.is_rebuttal
                        }
                        for p in r.positions
                    ],
                    "round_summary": r.round_summary
                }
                for r in result.rounds
            ]

            cursor.execute("""
                INSERT INTO ai_consensus_requests
                (id, prompt, strategy, models_requested, final_response,
                 confidence, agreement_score, status, model_responses,
                 dissenting_opinions, metadata, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                result.debate_id,
                result.topic,
                ConsensusStrategy.MULTI_ROUND_DEBATE.value,
                Json([m.value for m in result.participating_models]),
                result.final_consensus,
                result.confidence,
                result.agreement_evolution[-1] if result.agreement_evolution else 0.0,
                ConsensusStatus.CONSENSUS_REACHED.value if result.confidence > 0.5 else ConsensusStatus.PARTIAL_CONSENSUS.value,
                Json(rounds_data),
                Json([{"disagreement": d} for d in result.remaining_disagreements]),
                Json({
                    "debate_type": "multi_round",
                    "total_rounds": result.total_rounds,
                    "debate_duration_ms": result.debate_duration_ms,
                    "agreement_evolution": result.agreement_evolution,
                    "key_points_of_agreement": result.key_points_of_agreement
                })
            ))

            conn.commit()
            cursor.close()

            logger.info(f"Debate {result.debate_id} logged to database")

        except Exception as e:
            logger.error(f"Failed to log debate result: {e}")

    async def _log_request(
        self,
        request_id: str,
        prompt: str,
        strategy: ConsensusStrategy,
        models: list[ModelType]
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

    async def get_model_statistics(self) -> dict[str, Any]:
        """Get performance statistics for all models"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT model_id, model_name, total_queries, successful_queries,
                       failed_queries, avg_latency_ms, accuracy_score, cost_efficiency,
                       last_used_at, created_at, updated_at
                FROM ai_model_performance
                ORDER BY total_queries DESC
                LIMIT 100
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
    ) -> list[dict]:
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
    'ModelResponse',
    'DebatePosition',
    'DebateRound',
    'DebateResult'
]

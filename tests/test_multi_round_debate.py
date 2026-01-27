#!/usr/bin/env python3
"""
Test script for the multi-round debate engine.
Tests the implementation without making actual API calls.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Import the modules to test
from multi_model_consensus import (
    ConsensusEngine,
    ConsensusStrategy,
    ConsensusStatus,
    DebatePosition,
    DebateRound,
    DebateResult,
    ModelProvider,
    ModelResponse,
    ModelType,
)


def create_mock_response(model: ModelType, text: str, confidence: float = 0.85) -> ModelResponse:
    """Create a mock model response"""
    return ModelResponse(
        model=model,
        response=text,
        confidence=confidence,
        reasoning=f"Mock reasoning for {model.value}",
        latency_ms=100.0,
        tokens_used=150
    )


async def test_debate_data_structures():
    """Test the new data structures"""
    print("\n=== Testing Data Structures ===")

    # Test DebatePosition
    position = DebatePosition(
        model=ModelType.OPENAI_GPT4,
        position="I believe the answer is X because...",
        key_arguments=["Argument 1", "Argument 2", "Argument 3"],
        confidence=0.85,
        round_number=1,
        is_rebuttal=False
    )
    assert position.model == ModelType.OPENAI_GPT4
    assert len(position.key_arguments) == 3
    print("  DebatePosition: PASSED")

    # Test DebateRound
    round_data = DebateRound(
        round_number=1,
        round_type="initial",
        positions=[position],
        round_summary="Initial positions from 1 model"
    )
    assert round_data.round_number == 1
    assert round_data.round_type == "initial"
    print("  DebateRound: PASSED")

    # Test DebateResult
    result = DebateResult(
        debate_id="test-123",
        topic="Test topic",
        rounds=[round_data],
        final_consensus="Final answer",
        confidence=0.9,
        participating_models=[ModelType.OPENAI_GPT4],
        agreement_evolution=[0.5, 0.7, 0.9],
        key_points_of_agreement=["Point 1", "Point 2"],
        remaining_disagreements=[],
        total_rounds=3,
        debate_duration_ms=5000.0
    )
    assert result.debate_id == "test-123"
    assert len(result.agreement_evolution) == 3
    print("  DebateResult: PASSED")


async def test_multi_round_debate_strategy():
    """Test the multi-round debate strategy enum"""
    print("\n=== Testing ConsensusStrategy Enum ===")

    # Verify the new strategy exists
    assert ConsensusStrategy.MULTI_ROUND_DEBATE.value == "multi_round_debate"
    print("  MULTI_ROUND_DEBATE strategy exists: PASSED")

    # Verify all strategies
    expected_strategies = [
        "majority_vote", "weighted_average", "unanimous",
        "highest_confidence", "ensemble", "debate",
        "multi_round_debate", "hierarchical"
    ]
    actual_strategies = [s.value for s in ConsensusStrategy]
    for strategy in expected_strategies:
        assert strategy in actual_strategies, f"Missing strategy: {strategy}"
    print("  All strategies present: PASSED")


async def test_extract_key_arguments():
    """Test argument extraction from text"""
    print("\n=== Testing Argument Extraction ===")

    provider = ModelProvider()
    engine = ConsensusEngine(provider)

    # Test numbered list extraction
    text_with_numbers = """
    My position is clear:
    1. First argument about the topic
    2. Second important point to consider
    3. Third supporting evidence here
    """
    args = engine._extract_key_arguments(text_with_numbers)
    assert len(args) >= 1, "Should extract at least one argument"
    print(f"  Extracted {len(args)} arguments from numbered text: PASSED")

    # Test bullet extraction
    text_with_bullets = """
    Key points:
    - Important because it affects performance
    - Therefore we should consider alternatives
    - However there are trade-offs
    """
    args2 = engine._extract_key_arguments(text_with_bullets)
    assert len(args2) >= 1, "Should extract from bullets"
    print(f"  Extracted {len(args2)} arguments from bullets: PASSED")


async def test_calculate_position_agreement():
    """Test agreement calculation between positions"""
    print("\n=== Testing Position Agreement Calculation ===")

    provider = ModelProvider()
    engine = ConsensusEngine(provider)

    # Create similar positions
    pos1 = DebatePosition(
        model=ModelType.OPENAI_GPT4,
        position="The best approach is to use microservices with proper isolation",
        key_arguments=["Isolation", "Scalability"],
        confidence=0.85,
        round_number=1
    )
    pos2 = DebatePosition(
        model=ModelType.ANTHROPIC_CLAUDE,
        position="Using microservices with isolation is the optimal strategy",
        key_arguments=["Isolation", "Flexibility"],
        confidence=0.88,
        round_number=1
    )

    agreement = engine._calculate_position_agreement([pos1, pos2])
    assert 0 <= agreement <= 1, "Agreement should be between 0 and 1"
    print(f"  Agreement score: {agreement:.2%}")
    print("  Position agreement calculation: PASSED")

    # Test single position
    single_agreement = engine._calculate_position_agreement([pos1])
    assert single_agreement == 1.0, "Single position should have 100% agreement"
    print("  Single position agreement: PASSED")


async def test_format_debate_history():
    """Test debate history formatting"""
    print("\n=== Testing Debate History Formatting ===")

    provider = ModelProvider()
    engine = ConsensusEngine(provider)

    positions = [
        DebatePosition(
            model=ModelType.OPENAI_GPT4,
            position="My initial position on the topic...",
            key_arguments=["Arg 1"],
            confidence=0.85,
            round_number=1
        ),
        DebatePosition(
            model=ModelType.ANTHROPIC_CLAUDE,
            position="I present a different view...",
            key_arguments=["Arg 2"],
            confidence=0.88,
            round_number=1
        )
    ]

    rounds = [
        DebateRound(
            round_number=1,
            round_type="initial",
            positions=positions,
            round_summary="Initial positions"
        )
    ]

    history = engine._format_debate_history(rounds)
    assert "ROUND 1" in history
    assert "INITIAL" in history
    assert "openai_gpt4" in history
    assert "anthropic_claude" in history
    print("  Debate history formatting: PASSED")


async def test_multi_round_debate_mock():
    """Test multi-round debate with mocked model calls"""
    print("\n=== Testing Multi-Round Debate (Mocked) ===")

    provider = ModelProvider()
    engine = ConsensusEngine(provider)

    # Mock the query_model method
    call_count = 0

    async def mock_query(model, prompt, system_prompt=None, max_tokens=1000, temperature=0.7):
        nonlocal call_count
        call_count += 1

        if "initial position" in prompt.lower() or "Round 1" not in prompt:
            # Round 1: Initial positions
            if model == ModelType.OPENAI_GPT4:
                return create_mock_response(
                    model,
                    "1. Microservices are best for scalability\n2. Container orchestration is key\n3. API gateways provide security",
                    0.85
                )
            elif model == ModelType.ANTHROPIC_CLAUDE:
                return create_mock_response(
                    model,
                    "1. Monoliths can be more maintainable\n2. Start simple, scale later\n3. Consider team size",
                    0.88
                )
            else:
                return create_mock_response(
                    model,
                    "1. Hybrid approach works best\n2. Domain-driven design matters\n3. Infrastructure as code",
                    0.82
                )
        elif "rebuttal" in prompt.lower():
            # Round 2: Rebuttals
            return create_mock_response(
                model,
                f"I acknowledge valid points but maintain that my approach is better because of specific trade-offs. "
                f"The key consideration is context-specific requirements.",
                0.80
            )
        else:
            # Round 3: Synthesis
            return create_mock_response(
                model,
                "SYNTHESIS: After considering all perspectives, the consensus is that the choice between "
                "microservices and monoliths depends on team size, complexity, and scaling needs. "
                "Key agreements: 1) Start simple 2) Design for change 3) Consider operational costs. "
                "Confidence: 85%",
                0.85
            )

    # Patch the query_model method
    engine.model_provider.query_model = mock_query

    # Run the debate
    models = [ModelType.OPENAI_GPT4, ModelType.ANTHROPIC_CLAUDE, ModelType.OPENAI_GPT4O]
    result = await engine.multi_round_debate(
        topic="What is the best architecture approach: microservices or monolith?",
        models=models,
        rounds=3
    )

    assert result.debate_id is not None
    assert result.total_rounds >= 1
    assert len(result.rounds) >= 1
    assert result.final_consensus != ""
    assert len(result.agreement_evolution) >= 1

    print(f"  Debate ID: {result.debate_id[:8]}...")
    print(f"  Total rounds: {result.total_rounds}")
    print(f"  Model calls made: {call_count}")
    print(f"  Agreement evolution: {[f'{a:.2%}' for a in result.agreement_evolution]}")
    print(f"  Final confidence: {result.confidence:.2%}")
    print("  Multi-round debate execution: PASSED")


async def test_debate_to_consensus_conversion():
    """Test conversion from DebateResult to ConsensusResult"""
    print("\n=== Testing Debate to Consensus Conversion ===")

    provider = ModelProvider()
    engine = ConsensusEngine(provider)

    # Create mock initial responses
    initial_responses = [
        create_mock_response(ModelType.OPENAI_GPT4, "Position A with key points", 0.85),
        create_mock_response(ModelType.ANTHROPIC_CLAUDE, "Position B with arguments", 0.88)
    ]

    # Mock query_model for subsequent rounds
    async def mock_query(model, prompt, system_prompt=None, max_tokens=1000, temperature=0.7):
        return create_mock_response(model, f"Response from {model.value}", 0.80)

    engine.model_provider.query_model = mock_query

    # Run the strategy wrapper
    result = await engine._multi_round_debate_strategy(initial_responses, "Test topic")

    assert result.strategy_used == ConsensusStrategy.MULTI_ROUND_DEBATE
    assert result.final_response != ""
    assert result.consensus_id != ""
    assert len(result.model_responses) > len(initial_responses)  # Should include debate rounds
    assert "total_rounds" in result.metadata
    print(f"  Strategy used: {result.strategy_used.value}")
    print(f"  Status: {result.status.value}")
    print("  Debate to consensus conversion: PASSED")


async def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("MULTI-ROUND DEBATE ENGINE TEST SUITE")
    print("=" * 60)

    try:
        await test_debate_data_structures()
        await test_multi_round_debate_strategy()
        await test_extract_key_arguments()
        await test_calculate_position_agreement()
        await test_format_debate_history()
        await test_multi_round_debate_mock()
        await test_debate_to_consensus_conversion()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)

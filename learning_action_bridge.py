#!/usr/bin/env python3
"""
Learning-Action Bridge - The Critical Link for True Intelligence
=================================================================
This module bridges the gap between learning and action, ensuring that
what the AI learns actually changes its behavior.

THE PROBLEM IT SOLVES:
- Learning loop runs in isolation, discovers patterns
- Agents make decisions without consulting learned knowledge
- Self-improvement is simulated, not real

THE SOLUTION:
1. Converts learning outcomes to actionable behavior rules
2. Provides agents with learning-informed guidance
3. Tracks which learned behaviors actually improve outcomes
4. Creates a genuine feedback loop from experience to action

Author: BrainOps AI OS
Version: 1.0.0 - True Learning
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from enum import Enum

from safe_task import create_safe_task

# Import unified memory
try:
    from unified_memory_manager import get_memory_manager, Memory, MemoryType
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

logger = logging.getLogger(__name__)

# System-level tenant ID for cross-tenant learning operations (must be valid UUID)
SYSTEM_TENANT_ID = "00000000-0000-0000-0000-000000000001"


class RuleType(Enum):
    """Types of behavior rules"""
    AVOIDANCE = "avoidance"      # Don't do X because it failed
    PREFERENCE = "preference"    # Prefer X because it succeeded
    OPTIMIZATION = "optimization"  # Improve X using pattern Y
    THRESHOLD = "threshold"      # Trigger action when condition met
    SEQUENCE = "sequence"        # Do X then Y then Z


@dataclass
class BehaviorRule:
    """A learned behavior rule"""
    id: str
    rule_type: RuleType
    trigger: str                  # What triggers this rule
    action: str                   # What action to take
    confidence: float             # How confident we are in this rule
    success_count: int = 0        # How many times it worked
    failure_count: int = 0        # How many times it failed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_applied: Optional[datetime] = None
    source_insight: Optional[str] = None  # ID of the insight that created this rule


@dataclass
class LearningOutcome:
    """An outcome from the learning loop"""
    id: str
    outcome_type: str             # pattern, correction, optimization
    description: str
    data: dict[str, Any]
    confidence: float
    timestamp: datetime


class LearningActionBridge:
    """
    The critical bridge between learning and action.

    This is what makes the AI ACTUALLY learn from experience.
    """

    def __init__(self):
        self.behavior_rules: dict[str, BehaviorRule] = {}
        self.pending_outcomes: list[LearningOutcome] = []
        self.rule_performance: dict[str, list[bool]] = defaultdict(list)

        # Configuration
        self.min_confidence_to_apply = 0.6
        self.max_rules_per_trigger = 5
        self.rule_decay_days = 30

        # State
        self.last_sync = None
        self.rules_applied_count = 0
        self.rules_created_count = 0

    def _get_memory_manager(self):
        """Get memory manager with proper error handling"""
        if not MEMORY_AVAILABLE:
            return None
        try:
            return get_memory_manager()
        except Exception as e:
            logger.warning(f"Failed to get memory manager: {e}")
            return None

    async def sync_from_learning(self) -> int:
        """
        Sync learning outcomes into behavior rules.
        This is the KEY function that makes learning actionable.
        """
        memory = self._get_memory_manager()
        if not memory:
            logger.warning("Cannot sync - memory unavailable")
            return 0

        rules_created = 0

        try:
            # Get recent learning insights (use system tenant for cross-tenant learning)
            learnings = memory.recall(
                "learning insight pattern outcome success failure",
                tenant_id=SYSTEM_TENANT_ID,
                limit=100
            )

            for learning_mem in learnings:
                content = learning_mem.get('content', {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        continue

                # Skip if already processed
                insight_id = learning_mem.get('id', '')
                if any(r.source_insight == str(insight_id) for r in self.behavior_rules.values()):
                    continue

                # Convert learning to rule based on type
                outcome_type = content.get('type', content.get('outcome_type', ''))

                rule = None
                if outcome_type == 'error_pattern' or content.get('outcome') == 'failure':
                    rule = self._create_avoidance_rule(content, str(insight_id))
                elif outcome_type == 'success_pattern' or content.get('outcome') == 'success':
                    rule = self._create_preference_rule(content, str(insight_id))
                elif outcome_type == 'optimization':
                    rule = self._create_optimization_rule(content, str(insight_id))

                if rule:
                    self.behavior_rules[rule.id] = rule
                    rules_created += 1
                    logger.info(f"Created behavior rule: {rule.rule_type.value} - {rule.action[:50]}")

            self.last_sync = datetime.now(timezone.utc)
            self.rules_created_count += rules_created

            # Persist rules to memory for cross-session continuity
            await self._persist_rules()

            return rules_created

        except Exception as e:
            logger.error(f"Failed to sync from learning: {e}")
            return 0

    def _create_avoidance_rule(self, content: dict, insight_id: str) -> Optional[BehaviorRule]:
        """Create an avoidance rule from a failure pattern"""
        import uuid

        trigger = content.get('trigger', content.get('context', 'unknown'))
        error_type = content.get('error_type', content.get('pattern', ''))

        if not error_type:
            return None

        return BehaviorRule(
            id=str(uuid.uuid4()),
            rule_type=RuleType.AVOIDANCE,
            trigger=trigger,
            action=f"AVOID: {error_type}. Previous failures indicate this approach doesn't work.",
            confidence=content.get('confidence', 0.7),
            source_insight=insight_id
        )

    def _create_preference_rule(self, content: dict, insight_id: str) -> Optional[BehaviorRule]:
        """Create a preference rule from a success pattern"""
        import uuid

        trigger = content.get('trigger', content.get('context', 'unknown'))
        successful_approach = content.get('response', content.get('approach', ''))

        if not successful_approach:
            return None

        return BehaviorRule(
            id=str(uuid.uuid4()),
            rule_type=RuleType.PREFERENCE,
            trigger=trigger,
            action=f"PREFER: {successful_approach}. Previous successes validate this approach.",
            confidence=content.get('confidence', 0.7),
            source_insight=insight_id
        )

    def _create_optimization_rule(self, content: dict, insight_id: str) -> Optional[BehaviorRule]:
        """Create an optimization rule from an optimization insight"""
        import uuid

        trigger = content.get('trigger', content.get('domain', 'general'))
        optimization = content.get('optimization', content.get('recommendation', ''))

        if not optimization:
            return None

        return BehaviorRule(
            id=str(uuid.uuid4()),
            rule_type=RuleType.OPTIMIZATION,
            trigger=trigger,
            action=f"OPTIMIZE: {optimization}",
            confidence=content.get('confidence', 0.6),
            source_insight=insight_id
        )

    async def _persist_rules(self):
        """Persist behavior rules to memory for cross-session continuity"""
        memory = self._get_memory_manager()
        if not memory:
            return

        try:
            # Serialize rules
            rules_data = {
                rule_id: {
                    "id": rule.id,
                    "rule_type": rule.rule_type.value,
                    "trigger": rule.trigger,
                    "action": rule.action,
                    "confidence": rule.confidence,
                    "success_count": rule.success_count,
                    "failure_count": rule.failure_count,
                    "created_at": rule.created_at.isoformat(),
                    "source_insight": rule.source_insight
                }
                for rule_id, rule in self.behavior_rules.items()
            }

            memory.store(Memory(
                memory_type=MemoryType.PROCEDURAL,
                content={
                    "type": "behavior_rules_snapshot",
                    "rules": rules_data,
                    "count": len(rules_data),
                    "synced_at": datetime.now(timezone.utc).isoformat()
                },
                source_system="learning_action_bridge",
                source_agent="rule_persister",
                created_by="learning_bridge",
                importance_score=0.95,  # Very important
                tags=["behavior_rules", "learning", "snapshot"],
                tenant_id=SYSTEM_TENANT_ID  # System-level rules
            ))

            logger.info(f"Persisted {len(rules_data)} behavior rules to memory")

        except Exception as e:
            logger.error(f"Failed to persist rules: {e}")

    async def load_persisted_rules(self):
        """Load behavior rules from memory on startup"""
        memory = self._get_memory_manager()
        if not memory:
            return

        try:
            # Find most recent rules snapshot (use system tenant for cross-tenant rules)
            snapshots = memory.recall(
                "behavior_rules_snapshot",
                tenant_id=SYSTEM_TENANT_ID,
                limit=1
            )

            if not snapshots:
                logger.info("No persisted behavior rules found - starting fresh")
                return

            content = snapshots[0].get('content', {})
            if isinstance(content, str):
                content = json.loads(content)

            rules_data = content.get('rules', {})

            for rule_id, rule_data in rules_data.items():
                try:
                    rule = BehaviorRule(
                        id=rule_data['id'],
                        rule_type=RuleType(rule_data['rule_type']),
                        trigger=rule_data['trigger'],
                        action=rule_data['action'],
                        confidence=rule_data['confidence'],
                        success_count=rule_data.get('success_count', 0),
                        failure_count=rule_data.get('failure_count', 0),
                        created_at=datetime.fromisoformat(rule_data['created_at']),
                        source_insight=rule_data.get('source_insight')
                    )
                    self.behavior_rules[rule.id] = rule
                except Exception as e:
                    logger.warning(f"Failed to load rule {rule_id}: {e}")

            logger.info(f"Loaded {len(self.behavior_rules)} behavior rules from memory")

        except Exception as e:
            logger.error(f"Failed to load persisted rules: {e}")

    def get_guidance(self, context: str, action_type: Optional[str] = None) -> list[dict]:
        """
        Get learning-informed guidance for a given context.
        THIS IS WHAT AGENTS CALL BEFORE MAKING DECISIONS.
        """
        guidance = []

        context_lower = context.lower()

        for rule in self.behavior_rules.values():
            # Check if rule applies to this context
            trigger_lower = rule.trigger.lower()

            # Match by trigger keywords
            if any(word in context_lower for word in trigger_lower.split()[:3]):
                # Check confidence threshold
                if rule.confidence >= self.min_confidence_to_apply:
                    guidance.append({
                        "rule_id": rule.id,
                        "type": rule.rule_type.value,
                        "action": rule.action,
                        "confidence": rule.confidence,
                        "track_record": f"{rule.success_count} successes, {rule.failure_count} failures"
                    })

        # Sort by confidence
        guidance.sort(key=lambda x: x['confidence'], reverse=True)

        # Limit to top rules
        return guidance[:self.max_rules_per_trigger]

    def report_outcome(self, rule_id: str, success: bool):
        """
        Report the outcome of applying a rule.
        This creates the feedback loop that improves rule quality.
        """
        if rule_id not in self.behavior_rules:
            logger.warning(f"Unknown rule ID: {rule_id}")
            return

        rule = self.behavior_rules[rule_id]
        rule.last_applied = datetime.now(timezone.utc)

        if success:
            rule.success_count += 1
            # Boost confidence on success
            rule.confidence = min(1.0, rule.confidence + 0.05)
        else:
            rule.failure_count += 1
            # Reduce confidence on failure
            rule.confidence = max(0.1, rule.confidence - 0.1)

        # Track performance
        self.rule_performance[rule_id].append(success)

        # If rule has too many failures, consider removing it
        if rule.failure_count > 5 and rule.failure_count > rule.success_count * 2:
            logger.info(f"Removing underperforming rule: {rule_id}")
            del self.behavior_rules[rule_id]

        self.rules_applied_count += 1
        logger.debug(f"Reported outcome for rule {rule_id}: {'success' if success else 'failure'}")

    def prune_old_rules(self) -> int:
        """Remove old rules that haven't been used recently"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.rule_decay_days)
        pruned = 0

        to_remove = []
        for rule_id, rule in self.behavior_rules.items():
            # Prune if never applied and old
            if rule.last_applied is None and rule.created_at < cutoff:
                to_remove.append(rule_id)
            # Prune if last applied long ago and low confidence
            elif rule.last_applied and rule.last_applied < cutoff and rule.confidence < 0.5:
                to_remove.append(rule_id)

        for rule_id in to_remove:
            del self.behavior_rules[rule_id]
            pruned += 1

        if pruned:
            logger.info(f"Pruned {pruned} stale behavior rules")

        return pruned

    def get_status(self) -> dict[str, Any]:
        """Get the current status of the learning-action bridge"""
        rule_types = defaultdict(int)
        for rule in self.behavior_rules.values():
            rule_types[rule.rule_type.value] += 1

        avg_confidence = 0.0
        if self.behavior_rules:
            avg_confidence = sum(r.confidence for r in self.behavior_rules.values()) / len(self.behavior_rules)

        return {
            "total_rules": len(self.behavior_rules),
            "rules_by_type": dict(rule_types),
            "average_confidence": round(avg_confidence, 3),
            "rules_applied": self.rules_applied_count,
            "rules_created": self.rules_created_count,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "memory_available": MEMORY_AVAILABLE
        }


# Singleton instance
_bridge: Optional[LearningActionBridge] = None


async def get_learning_bridge() -> LearningActionBridge:
    """Get or create the learning-action bridge"""
    global _bridge
    if _bridge is None:
        _bridge = LearningActionBridge()
        await _bridge.load_persisted_rules()
        logger.info("Learning-Action Bridge initialized")
    return _bridge


async def sync_learning_to_actions():
    """Convenience function to sync learning outcomes to behavior rules"""
    bridge = await get_learning_bridge()
    return await bridge.sync_from_learning()


def get_behavior_guidance(context: str) -> list[dict]:
    """Convenience function for agents to get guidance"""
    global _bridge
    if _bridge is None:
        return []  # Bridge not initialized yet
    return _bridge.get_guidance(context)


# Background task to periodically sync learning
async def run_bridge_sync_loop(interval_seconds: int = 300):
    """Run the bridge sync loop in the background"""
    bridge = await get_learning_bridge()

    while True:
        try:
            rules_created = await bridge.sync_from_learning()
            pruned = bridge.prune_old_rules()

            if rules_created or pruned:
                logger.info(f"Bridge sync: +{rules_created} rules, -{pruned} pruned")

        except Exception as e:
            logger.error(f"Bridge sync error: {e}")

        await asyncio.sleep(interval_seconds)


if __name__ == "__main__":
    # Test the bridge
    async def test():
        print("=" * 60)
        print("LEARNING-ACTION BRIDGE TEST")
        print("=" * 60)

        bridge = await get_learning_bridge()

        # Test sync
        print("\n1. Syncing from learning...")
        rules_created = await bridge.sync_from_learning()
        print(f"   Created {rules_created} new rules")

        # Test guidance
        print("\n2. Getting guidance for 'error handling'...")
        guidance = bridge.get_guidance("error handling high cpu")
        print(f"   Found {len(guidance)} relevant rules")
        for g in guidance[:3]:
            print(f"   - [{g['type']}] {g['action'][:50]}... (conf: {g['confidence']:.2f})")

        # Print status
        print("\n3. Bridge status:")
        status = bridge.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

    asyncio.run(test())

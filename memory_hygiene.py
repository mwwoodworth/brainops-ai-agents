"""
BrainOps Memory Hygiene System
===============================
Automated maintenance for the memory system.

Responsibilities:
- Deduplicate near-identical entries
- Detect conflicts between active decisions
- Degrade confidence scores over time
- Mark stale SOPs as DEGRADED
- Open re-verification tasks automatically
- Ensure superseded items are not retrieved as active truth

Part of BrainOps OS Total Completion Protocol.
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

from memory_enforcement import (
    EvidenceLevel,
    MemoryObjectType,
    VerificationState,
    get_enforcement_engine,
)

logger = logging.getLogger(__name__)

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457")


class MemoryHygieneSystem:
    """
    Automated memory hygiene and lifecycle management.

    Runs scheduled jobs to maintain memory health:
    - Deduplication
    - Conflict detection
    - Staleness degradation
    - Re-verification scheduling
    """

    def __init__(self, pool=None):
        self.pool = pool
        self._initialized = False

    async def initialize(self):
        """Lazy initialization"""
        if not self._initialized:
            if self.pool is None:
                engine = get_enforcement_engine()
                await engine.initialize()
                self.pool = engine.pool
            self._initialized = True

    async def run_full_hygiene(self) -> dict[str, Any]:
        """
        Run all hygiene operations.
        Returns a report of actions taken.
        """
        await self.initialize()

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "operations": {}
        }

        # 1. Degrade stale verifications
        degrade_count = await self.degrade_stale_verifications()
        report["operations"]["degraded_verifications"] = degrade_count

        # 2. Detect and record conflicts
        conflict_count = await self.detect_conflicts()
        report["operations"]["new_conflicts_detected"] = conflict_count

        # 3. Deduplicate memories
        dedup_count = await self.deduplicate_memories()
        report["operations"]["duplicates_merged"] = dedup_count

        # 4. Update confidence decay
        decay_count = await self.apply_confidence_decay()
        report["operations"]["confidence_decayed"] = decay_count

        # 5. Mark superseded items
        superseded_count = await self.mark_superseded_items()
        report["operations"]["items_superseded"] = superseded_count

        # 6. Create re-verification tasks
        reverify_count = await self.create_reverification_tasks()
        report["operations"]["reverification_tasks_created"] = reverify_count

        # 7. Clean expired memories
        expired_count = await self.clean_expired_memories()
        report["operations"]["expired_memories_cleaned"] = expired_count

        # 8. Update truth backlog stats
        backlog_stats = await self.get_backlog_stats()
        report["truth_backlog"] = backlog_stats

        logger.info(f"Memory hygiene complete: {json.dumps(report)}")
        return report

    async def degrade_stale_verifications(self) -> int:
        """
        Mark verified memories as DEGRADED if past expiration.
        """
        await self.initialize()

        try:
            result = await self.pool.execute("""
                UPDATE unified_ai_memory
                SET verification_state = 'DEGRADED',
                    confidence_score = GREATEST(confidence_score - 0.2, 0.0),
                    updated_at = NOW()
                WHERE verification_state = 'VERIFIED'
                  AND verification_expires_at IS NOT NULL
                  AND verification_expires_at < NOW()
            """)

            count = int(result.split()[-1]) if result else 0
            logger.info(f"Degraded {count} stale verifications")
            return count

        except Exception as e:
            logger.error(f"Failed to degrade stale verifications: {e}")
            return 0

    async def detect_conflicts(self) -> int:
        """
        Detect conflicting memories and record them.
        """
        await self.initialize()

        try:
            # Run the conflict detection function
            result = await self.pool.fetchval("SELECT detect_memory_conflicts()")
            logger.info(f"Detected {result} new conflicts")
            return result or 0

        except Exception as e:
            logger.error(f"Failed to detect conflicts: {e}")
            return 0

    async def deduplicate_memories(self) -> int:
        """
        Find and merge near-duplicate memories.
        """
        await self.initialize()

        try:
            # Find duplicates by content_hash
            duplicates = await self.pool.fetch("""
                SELECT content_hash, array_agg(id ORDER BY importance_score DESC, created_at ASC) as ids
                FROM unified_ai_memory
                WHERE content_hash IS NOT NULL
                  AND verification_state != 'BROKEN'
                  AND (expires_at IS NULL OR expires_at > NOW())
                GROUP BY content_hash
                HAVING COUNT(*) > 1
                LIMIT 100
            """)

            merged_count = 0
            for dup in duplicates:
                ids = dup["ids"]
                if len(ids) < 2:
                    continue

                # Keep the first (highest importance), mark others as superseded
                primary_id = ids[0]
                superseded_ids = ids[1:]

                # Update superseded memories
                await self.pool.execute("""
                    UPDATE unified_ai_memory
                    SET verification_state = 'BROKEN',
                        metadata = metadata || jsonb_build_object(
                            'superseded_by', $1::text,
                            'superseded_at', NOW()
                        ),
                        updated_at = NOW()
                    WHERE id = ANY($2::uuid[])
                """, str(primary_id), superseded_ids)

                # Update primary to note it supersedes others
                await self.pool.execute("""
                    UPDATE unified_ai_memory
                    SET supersedes = supersedes || $1::uuid[],
                        updated_at = NOW()
                    WHERE id = $2::uuid
                """, superseded_ids, primary_id)

                merged_count += len(superseded_ids)

            logger.info(f"Merged {merged_count} duplicate memories")
            return merged_count

        except Exception as e:
            logger.error(f"Failed to deduplicate memories: {e}")
            return 0

    async def apply_confidence_decay(self) -> int:
        """
        Apply time-based confidence decay to unverified memories.
        """
        await self.initialize()

        try:
            # Decay confidence for old unverified memories
            result = await self.pool.execute("""
                UPDATE unified_ai_memory
                SET confidence_score = GREATEST(confidence_score - 0.05, 0.0),
                    updated_at = NOW()
                WHERE verification_state = 'UNVERIFIED'
                  AND created_at < NOW() - INTERVAL '7 days'
                  AND confidence_score > 0.0
                  AND last_verified_at IS NULL
            """)

            count = int(result.split()[-1]) if result else 0
            logger.info(f"Applied confidence decay to {count} memories")
            return count

        except Exception as e:
            logger.error(f"Failed to apply confidence decay: {e}")
            return 0

    async def mark_superseded_items(self) -> int:
        """
        Mark items as superseded based on supersedes relationships.
        """
        await self.initialize()

        try:
            # Find memories that should be marked superseded
            result = await self.pool.execute("""
                UPDATE unified_ai_memory m
                SET verification_state = 'BROKEN',
                    metadata = metadata || jsonb_build_object(
                        'supersession_reason', 'explicitly_superseded'
                    ),
                    updated_at = NOW()
                FROM unified_ai_memory newer
                WHERE m.id = ANY(newer.supersedes)
                  AND m.verification_state NOT IN ('BROKEN')
                  AND newer.verification_state IN ('VERIFIED', 'UNVERIFIED')
            """)

            count = int(result.split()[-1]) if result else 0
            logger.info(f"Marked {count} items as superseded")
            return count

        except Exception as e:
            logger.error(f"Failed to mark superseded items: {e}")
            return 0

    async def create_reverification_tasks(self) -> int:
        """
        Create tasks for memories needing re-verification.
        """
        await self.initialize()

        try:
            # Find memories needing re-verification
            memories = await self.pool.fetch("""
                SELECT id, memory_type, object_type, owner, project
                FROM unified_ai_memory
                WHERE (
                    (verification_state = 'DEGRADED')
                    OR (verification_state = 'VERIFIED' AND verification_expires_at < NOW() + INTERVAL '7 days')
                    OR (verification_state = 'UNVERIFIED' AND importance_score > 0.7 AND created_at > NOW() - INTERVAL '30 days')
                )
                AND NOT EXISTS (
                    SELECT 1 FROM unified_ai_memory task
                    WHERE task.object_type = 'task'
                      AND task.content->>'target_memory_id' = unified_ai_memory.id::text
                      AND task.verification_state != 'BROKEN'
                )
                LIMIT 50
            """)

            created = 0
            for mem in memories:
                # Create a re-verification task
                await self.pool.execute("""
                    INSERT INTO unified_ai_memory (
                        memory_type, object_type, content, importance_score,
                        tags, source_system, source_agent, created_by, owner, project
                    ) VALUES (
                        'procedural', 'task',
                        $1::jsonb,
                        0.8,
                        ARRAY['reverification', 'hygiene'],
                        'memory_hygiene',
                        'hygiene_system',
                        'memory_hygiene_system',
                        $2,
                        $3
                    )
                """,
                    json.dumps({
                        "task_type": "reverification",
                        "target_memory_id": str(mem["id"]),
                        "target_object_type": mem["object_type"],
                        "reason": "scheduled_reverification",
                        "created_at": datetime.utcnow().isoformat()
                    }),
                    mem["owner"],
                    mem["project"]
                )
                created += 1

            logger.info(f"Created {created} re-verification tasks")
            return created

        except Exception as e:
            logger.error(f"Failed to create re-verification tasks: {e}")
            return 0

    async def clean_expired_memories(self) -> int:
        """
        Remove (soft-delete) expired memories.
        """
        await self.initialize()

        try:
            result = await self.pool.execute("""
                UPDATE unified_ai_memory
                SET verification_state = 'BROKEN',
                    metadata = metadata || jsonb_build_object(
                        'expired_at', NOW(),
                        'original_expires_at', expires_at
                    ),
                    updated_at = NOW()
                WHERE expires_at IS NOT NULL
                  AND expires_at < NOW()
                  AND verification_state != 'BROKEN'
            """)

            count = int(result.split()[-1]) if result else 0
            logger.info(f"Cleaned {count} expired memories")
            return count

        except Exception as e:
            logger.error(f"Failed to clean expired memories: {e}")
            return 0

    async def get_backlog_stats(self) -> dict[str, Any]:
        """
        Get statistics about the truth backlog.
        """
        await self.initialize()

        try:
            stats = await self.pool.fetchrow("""
                SELECT
                    COUNT(*) FILTER (WHERE verification_state = 'UNVERIFIED') as unverified,
                    COUNT(*) FILTER (WHERE verification_state = 'DEGRADED') as degraded,
                    COUNT(*) FILTER (WHERE verification_state = 'BROKEN') as broken,
                    COUNT(*) FILTER (WHERE verification_state = 'VERIFIED') as verified,
                    COUNT(*) FILTER (WHERE owner IS NULL AND importance_score > 0.5) as ownerless,
                    COUNT(*) FILTER (WHERE verification_expires_at < NOW()) as expired_verification,
                    AVG(confidence_score) as avg_confidence
                FROM unified_ai_memory
                WHERE (expires_at IS NULL OR expires_at > NOW())
            """)

            # Get conflict count
            conflict_count = await self.pool.fetchval("""
                SELECT COUNT(*) FROM memory_conflicts WHERE resolution_status = 'open'
            """)

            return {
                "unverified": stats["unverified"] or 0,
                "degraded": stats["degraded"] or 0,
                "broken": stats["broken"] or 0,
                "verified": stats["verified"] or 0,
                "ownerless": stats["ownerless"] or 0,
                "expired_verification": stats["expired_verification"] or 0,
                "open_conflicts": conflict_count or 0,
                "avg_confidence": float(stats["avg_confidence"] or 0.0)
            }

        except Exception as e:
            logger.error(f"Failed to get backlog stats: {e}")
            return {}


# =============================================================================
# SCHEDULER INTEGRATION
# =============================================================================

async def run_scheduled_hygiene():
    """
    Entry point for scheduled hygiene job.
    Called by APScheduler.
    """
    hygiene = MemoryHygieneSystem()
    report = await hygiene.run_full_hygiene()
    return report


# =============================================================================
# SINGLETON
# =============================================================================

_hygiene_system: Optional[MemoryHygieneSystem] = None


def get_hygiene_system() -> MemoryHygieneSystem:
    """Get or create the singleton hygiene system"""
    global _hygiene_system
    if _hygiene_system is None:
        _hygiene_system = MemoryHygieneSystem()
    return _hygiene_system

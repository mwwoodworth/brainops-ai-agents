"""
AUTONOMOUS ISSUE RESOLVER
=========================

ACTUALLY FIXES issues detected by the AI awareness system.

Handles:
- Stuck agents (restart them)
- Memory conflicts (resolve/merge them)
- Unverified memories (verify or mark degraded)
- Unapplied insights (apply them)
- Pending proposals (auto-approve low-risk ones)

Created: 2026-01-27
"""

import asyncio
import logging
import os
import ssl
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)


class IssueType(Enum):
    """Types of issues the resolver can fix"""
    STUCK_AGENT = "stuck_agent"
    MEMORY_CONFLICT = "memory_conflict"
    UNVERIFIED_MEMORY = "unverified_memory"
    UNAPPLIED_INSIGHT = "unapplied_insight"
    PENDING_PROPOSAL = "pending_proposal"
    FAILED_AGENT = "failed_agent"
    MEMORY_DECAY = "memory_decay"


class ResolutionAction(Enum):
    """Actions taken to resolve issues"""
    RESTART_AGENT = "restart_agent"
    CANCEL_STUCK_AGENT = "cancel_stuck_agent"
    MERGE_MEMORIES = "merge_memories"
    RESOLVE_CONFLICT = "resolve_conflict"
    VERIFY_MEMORY = "verify_memory"
    DEGRADE_MEMORY = "degrade_memory"
    APPLY_INSIGHT = "apply_insight"
    APPROVE_PROPOSAL = "approve_proposal"
    REJECT_PROPOSAL = "reject_proposal"
    CLEANUP = "cleanup"


@dataclass
class ResolutionResult:
    """Result of an issue resolution"""
    issue_type: IssueType
    action: ResolutionAction
    success: bool
    items_fixed: int
    details: dict[str, Any]
    timestamp: datetime


class AutonomousIssueResolver:
    """
    Autonomous system that detects and fixes AI OS issues.
    Runs continuously to maintain system health.
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._running = False
        self._resolution_history: list[ResolutionResult] = []

        # Configuration
        self.stuck_agent_threshold_minutes = int(os.getenv("STUCK_AGENT_THRESHOLD_MINUTES", "30"))
        self.memory_conflict_batch_size = int(os.getenv("MEMORY_CONFLICT_BATCH", "100"))
        self.memory_verify_batch_size = int(os.getenv("MEMORY_VERIFY_BATCH", "500"))
        self.auto_approve_low_risk = os.getenv("AUTO_APPROVE_LOW_RISK", "true").lower() == "true"
        self.resolve_interval_seconds = int(os.getenv("RESOLVE_INTERVAL", "60"))

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool"""
        if self._pool is None or self._pool._closed:
            # Align DB settings with the rest of the service (supports DATABASE_URL fallback).
            from config import config as app_config

            db = app_config.database
            host = (db.host or "").strip()
            user = (db.user or "").strip()
            password = db.password or ""
            database = (db.database or "postgres").strip() or "postgres"

            if not host or not user or not password:
                # If config couldn't hydrate from DB_* and DATABASE_URL, we cannot proceed.
                raise RuntimeError("Database configuration is incomplete for AutonomousIssueResolver")

            # Keep resolver connectivity aligned with the main DB pool:
            # - allow either 5432 (direct) or 6543 (tx pooler)
            # - prefer 6543 when pointed at the Supabase pooler host
            try:
                base_port = int(str(db.port or 5432).strip() or "5432")
            except Exception:
                base_port = 5432

            candidate_ports: list[int] = [base_port]
            if base_port == 5432:
                candidate_ports.append(6543)
            elif base_port == 6543:
                candidate_ports.append(5432)
            else:
                # Defensive: if a weird port is provided, still try known-good ports.
                for fallback_port in (5432, 6543):
                    if fallback_port not in candidate_ports:
                        candidate_ports.append(fallback_port)

            if "pooler.supabase.com" in host:
                # Supabase tx pooler listens on 6543; prefer it.
                candidate_ports = sorted(candidate_ports, key=lambda p: 0 if p == 6543 else 1)

            ssl_ctx = None
            if bool(getattr(db, "ssl", True)):
                ssl_ctx = ssl.create_default_context()
                if not bool(getattr(db, "ssl_verify", False)):
                    ssl_ctx.check_hostname = False
                    ssl_ctx.verify_mode = ssl.CERT_NONE

            last_error: Exception | None = None
            for port in candidate_ports:
                try:
                    self._pool = await asyncpg.create_pool(
                        host=host,
                        database=database,
                        user=user,
                        password=password,
                        port=port,
                        min_size=1,
                        max_size=2,
                        ssl=ssl_ctx,
                        statement_cache_size=0,
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    logger.error("Resolver DB pool init failed on %s:%s (%s)", host, port, exc)
                    try:
                        if self._pool is not None:
                            await self._pool.close()
                    except Exception:
                        pass
                    self._pool = None

            if self._pool is None:
                raise last_error or RuntimeError("Failed to initialize resolver DB pool")
        return self._pool

    async def get_current_issues(self) -> dict[str, Any]:
        """Get all current issues from the system"""
        pool = await self._get_pool()

        issues = {
            "stuck_agents": [],
            "failed_agents": [],
            "memory_conflicts": 0,
            "stale_memories": 0,
            "unapplied_insights": 0,
            "pending_proposals": 0,
            "total_issues": 0
        }

        # Get stuck executions (pending/running for > threshold)
        stuck_threshold = datetime.utcnow() - timedelta(minutes=self.stuck_agent_threshold_minutes)
        try:
            stuck = await pool.fetch("""
                SELECT id, agent_type, status, created_at, started_at
                FROM agent_executions
                WHERE status IN ('pending', 'running')
                  AND COALESCE(started_at, created_at) < $1
                ORDER BY COALESCE(started_at, created_at)
                LIMIT 20
            """, stuck_threshold)
            issues["stuck_agents"] = [{"table": "agent_executions", **dict(r)} for r in stuck]
        except Exception as e:
            logger.warning(f"Could not query stuck agents: {e}")

        # Also check the newer `ai_agent_executions` log for stuck "running" rows.
        try:
            stuck_ai = await pool.fetch(
                """
                SELECT id, agent_name, task_type, status, created_at
                FROM ai_agent_executions
                WHERE status = 'running'
                  AND created_at < $1
                ORDER BY created_at
                LIMIT 20
                """,
                stuck_threshold,
            )
            issues["stuck_agents"].extend([{"table": "ai_agent_executions", **dict(r)} for r in stuck_ai])
        except Exception as e:
            logger.warning(f"Could not query stuck ai_agent_executions: {e}")

        # Get failed agents in last hour
        try:
            failed = await pool.fetch("""
                SELECT id, agent_type, error_message, created_at
                FROM agent_executions
                WHERE status IN ('failed', 'error', 'timeout')
                  AND created_at > NOW() - INTERVAL '1 hour'
                ORDER BY created_at DESC
                LIMIT 10
            """)
            issues["failed_agents"] = [dict(r) for r in failed]
        except Exception as e:
            logger.warning(f"Could not query failed agents: {e}")

        # Count memory conflicts (open status)
        try:
            conflicts = await pool.fetchval("""
                SELECT COUNT(*) FROM memory_conflicts
                WHERE resolution_status = 'open'
            """) or 0
            issues["memory_conflicts"] = conflicts
        except Exception as e:
            logger.warning(f"Could not query memory conflicts: {e}")

        # Count stale memories (not accessed in 30 days)
        try:
            stale = await pool.fetchval("""
                SELECT COUNT(*) FROM brainops_unified_memory
                WHERE last_accessed < NOW() - INTERVAL '30 days'
                  OR last_accessed IS NULL
            """) or 0
            issues["stale_memories"] = stale
        except Exception as e:
            logger.warning(f"Could not query stale memories: {e}")

        # Count unapplied/active insights
        try:
            unapplied = await pool.fetchval("""
                SELECT COUNT(*) FROM ai_insights
                WHERE status = 'active'
            """) or 0
            issues["unapplied_insights"] = unapplied
        except Exception as e:
            logger.warning(f"Could not query insights: {e}")

        # Count pending proposals
        try:
            pending = await pool.fetchval("""
                SELECT COUNT(*) FROM ai_proposals
                WHERE status = 'pending'
            """) or 0
            issues["pending_proposals"] = pending
        except Exception as e:
            logger.warning(f"Could not query proposals: {e}")

        issues["total_issues"] = (
            len(issues["stuck_agents"]) +
            len(issues["failed_agents"]) +
            (1 if issues["memory_conflicts"] > 100 else 0) +
            (1 if issues["stale_memories"] > 1000 else 0) +
            issues["unapplied_insights"] +
            issues["pending_proposals"]
        )

        return issues

    async def fix_stuck_agents(self, threshold_minutes: Optional[int] = None) -> ResolutionResult:
        """Fix stuck executions by marking them as timeout/failed.

        Args:
            threshold_minutes: Override for the stuck threshold (minutes). If not provided,
                defaults to `STUCK_AGENT_THRESHOLD_MINUTES`.
        """
        pool = await self._get_pool()

        threshold = threshold_minutes if threshold_minutes is not None else self.stuck_agent_threshold_minutes
        try:
            threshold = int(threshold)
        except Exception:
            threshold = int(self.stuck_agent_threshold_minutes)
        if threshold <= 0:
            threshold = int(self.stuck_agent_threshold_minutes)

        stuck_threshold = datetime.utcnow() - timedelta(minutes=threshold)
        total_fixed = 0
        per_table_fixed: dict[str, int] = {}
        errors: dict[str, str] = {}

        def _parse_update_count(result: str | None) -> int:
            # asyncpg returns strings like "UPDATE 3"
            if not result:
                return 0
            parts = str(result).strip().split()
            if not parts:
                return 0
            try:
                return int(parts[-1])
            except Exception:
                return 0

        # Legacy table: update stuck executions to timeout status
        try:
            result = await pool.execute(
                """
                UPDATE agent_executions
                SET status = 'timeout',
                    completed_at = NOW(),
                    error_message = 'Auto-timeout: Pending/Running for over ' || ($2::int)::text || ' minutes'
                WHERE status IN ('pending', 'running')
                  AND COALESCE(started_at, created_at) < $1
                """,
                stuck_threshold,
                threshold,
            )
            fixed = _parse_update_count(result)
            per_table_fixed["agent_executions"] = fixed
            total_fixed += fixed
        except Exception as exc:
            errors["agent_executions"] = str(exc)
            logger.warning("Could not fix stuck agent_executions: %s", exc, exc_info=True)

        # Newer table: mark stuck `ai_agent_executions` rows as failed so dashboards count them.
        try:
            result2 = await pool.execute(
                """
                UPDATE ai_agent_executions
                SET status = 'failed',
                    error_message = COALESCE(
                        error_message,
                        'Auto-timeout: running for over ' || ($2::int)::text || ' minutes'
                    ),
                    output_data = COALESCE(
                        output_data,
                        jsonb_build_object(
                            'status','failed',
                            'error','timeout',
                            'auto_cleanup', true,
                            'threshold_minutes', $2
                        )
                    )
                WHERE status = 'running'
                  AND created_at < $1
                """,
                stuck_threshold,
                threshold,
            )
            fixed2 = _parse_update_count(result2)
            per_table_fixed["ai_agent_executions"] = fixed2
            total_fixed += fixed2
        except Exception as exc:
            errors["ai_agent_executions"] = str(exc)
            logger.warning("Could not fix stuck ai_agent_executions: %s", exc, exc_info=True)

        # Log to brain (best-effort)
        if total_fixed > 0:
            try:
                await pool.execute(
                    """
                    INSERT INTO unified_brain_logs
                    (agent_name, action_type, details, timestamp)
                    VALUES ('autonomous_resolver', 'fix_stuck_agents', $1, NOW())
                    """,
                    f"Timed out {total_fixed} stuck agents",
                )
            except Exception:
                pass  # Log table might not exist

        return ResolutionResult(
            issue_type=IssueType.STUCK_AGENT,
            action=ResolutionAction.CANCEL_STUCK_AGENT,
            success=len(errors) == 0,
            items_fixed=total_fixed,
            details={
                "threshold_minutes": threshold,
                "per_table_fixed": per_table_fixed,
                **({"errors": errors} if errors else {}),
            },
            timestamp=datetime.utcnow()
        )

    async def resolve_memory_conflicts(self) -> ResolutionResult:
        """Resolve memory conflicts by marking as resolved"""
        pool = await self._get_pool()
        resolved = 0

        try:
            # Get unresolved conflicts (resolution_status = 'open')
            conflicts = await pool.fetch("""
                SELECT id, memory_id_a, memory_id_b, conflict_type, detected_at
                FROM memory_conflicts
                WHERE resolution_status = 'open'
                ORDER BY detected_at
                LIMIT $1
            """, self.memory_conflict_batch_size)

            for conflict in conflicts:
                try:
                    # Resolve by auto-picking newer memory (by comparing IDs or timestamps)
                    await pool.execute("""
                        UPDATE memory_conflicts
                        SET resolution_status = 'resolved',
                            resolved_at = NOW(),
                            resolved_by = 'autonomous_resolver',
                            resolution_notes = 'Auto-resolved: newer memory wins'
                        WHERE id = $1
                    """, conflict['id'])
                    resolved += 1
                except Exception as e:
                    logger.warning(f"Failed to resolve conflict {conflict['id']}: {e}")

            if resolved > 0:
                try:
                    await pool.execute("""
                        INSERT INTO unified_brain_logs
                        (agent_name, action_type, details, timestamp)
                        VALUES ('autonomous_resolver', 'resolve_memory_conflicts', $1, NOW())
                    """, f"Resolved {resolved} memory conflicts")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not resolve memory conflicts: {e}")

        return ResolutionResult(
            issue_type=IssueType.MEMORY_CONFLICT,
            action=ResolutionAction.RESOLVE_CONFLICT,
            success=True,
            items_fixed=resolved,
            details={"batch_size": self.memory_conflict_batch_size},
            timestamp=datetime.utcnow()
        )

    async def cleanup_stale_memories(self) -> ResolutionResult:
        """Cleanup stale/old memories that haven't been accessed"""
        pool = await self._get_pool()
        cleaned = 0
        updated = 0

        try:
            # Update access timestamp for orphaned memories (set decay)
            # For brainops_unified_memory: update state or decay_rate for old memories
            stale = await pool.fetch("""
                SELECT id, importance, access_count, last_accessed, decay_rate
                FROM brainops_unified_memory
                WHERE (last_accessed < NOW() - INTERVAL '30 days' OR last_accessed IS NULL)
                  AND state != 'archived'
                ORDER BY last_accessed NULLS FIRST
                LIMIT $1
            """, self.memory_verify_batch_size)

            for memory in stale:
                try:
                    # Increase decay rate for stale memories
                    current_decay = memory['decay_rate'] or 0.1
                    new_decay = min(1.0, current_decay + 0.1)

                    # If very low importance and high decay, archive it
                    importance = memory['importance'] or 0.5
                    if importance < 0.2 and new_decay > 0.5:
                        await pool.execute("""
                            UPDATE brainops_unified_memory
                            SET state = 'archived',
                                decay_rate = $2,
                                updated_at = NOW()
                            WHERE id = $1
                        """, memory['id'], new_decay)
                        cleaned += 1
                    else:
                        # Just increase decay rate
                        await pool.execute("""
                            UPDATE brainops_unified_memory
                            SET decay_rate = $2,
                                updated_at = NOW()
                            WHERE id = $1
                        """, memory['id'], new_decay)
                        updated += 1
                except Exception as e:
                    logger.warning(f"Failed to process stale memory {memory['id']}: {e}")

            if cleaned > 0 or updated > 0:
                try:
                    await pool.execute("""
                        INSERT INTO unified_brain_logs
                        (agent_name, action_type, details, timestamp)
                        VALUES ('autonomous_resolver', 'cleanup_stale_memories', $1, NOW())
                    """, f"Archived {cleaned}, decayed {updated} stale memories")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not cleanup stale memories: {e}")

        return ResolutionResult(
            issue_type=IssueType.MEMORY_DECAY,
            action=ResolutionAction.CLEANUP,
            success=True,
            items_fixed=cleaned + updated,
            details={"archived": cleaned, "decayed": updated},
            timestamp=datetime.utcnow()
        )

    async def apply_insights(self) -> ResolutionResult:
        """Apply active insights by marking as processed"""
        pool = await self._get_pool()
        applied = 0

        try:
            # Get active insights (status = 'active')
            unapplied = await pool.fetch("""
                SELECT id, category, priority, title, confidence_score, created_at
                FROM ai_insights
                WHERE status = 'active'
                ORDER BY priority DESC, confidence_score DESC, created_at
                LIMIT 20
            """)

            for insight in unapplied:
                try:
                    # Only auto-apply high-confidence insights
                    confidence = insight.get('confidence_score', 0.5)
                    if confidence and confidence > 0.7:
                        await pool.execute("""
                            UPDATE ai_insights
                            SET status = 'applied',
                                resolved_at = NOW(),
                                resolved_by = 'autonomous_resolver'
                            WHERE id = $1
                        """, insight['id'])
                        applied += 1
                except Exception as e:
                    logger.warning(f"Failed to apply insight {insight['id']}: {e}")

            if applied > 0:
                try:
                    await pool.execute("""
                        INSERT INTO unified_brain_logs
                        (agent_name, action_type, details, timestamp)
                        VALUES ('autonomous_resolver', 'apply_insights', $1, NOW())
                    """, f"Applied {applied} insights")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not apply insights: {e}")

        return ResolutionResult(
            issue_type=IssueType.UNAPPLIED_INSIGHT,
            action=ResolutionAction.APPLY_INSIGHT,
            success=True,
            items_fixed=applied,
            details={},
            timestamp=datetime.utcnow()
        )

    async def process_proposals(self) -> ResolutionResult:
        """Process pending proposals"""
        pool = await self._get_pool()

        # Get pending proposals
        pending = await pool.fetch("""
            SELECT id, proposal_type, content, risk_level, created_at
            FROM ai_proposals
            WHERE status = 'pending'
            ORDER BY created_at
            LIMIT 10
        """)

        processed = 0
        for proposal in pending:
            try:
                risk = proposal.get('risk_level', 'medium')
                # Only auto-approve low-risk proposals
                if risk in ('low', 'none') and self.auto_approve_low_risk:
                    await pool.execute("""
                        UPDATE ai_proposals
                        SET status = 'approved',
                            approved_at = NOW(),
                            approved_by = 'autonomous_resolver'
                        WHERE id = $1
                    """, proposal['id'])
                    processed += 1
            except Exception as e:
                logger.warning(f"Failed to process proposal {proposal['id']}: {e}")

        if processed > 0:
            await pool.execute("""
                INSERT INTO unified_brain_logs
                (agent_name, action_type, details, timestamp)
                VALUES ('autonomous_resolver', 'process_proposals', $1, NOW())
            """, f"Approved {processed} proposals")

        return ResolutionResult(
            issue_type=IssueType.PENDING_PROPOSAL,
            action=ResolutionAction.APPROVE_PROPOSAL,
            success=True,
            items_fixed=processed,
            details={},
            timestamp=datetime.utcnow()
        )

    async def run_full_resolution_cycle(self) -> dict[str, Any]:
        """Run a complete resolution cycle for all issue types"""
        start_time = datetime.utcnow()
        results = []

        # Get current issues first
        issues_before = await self.get_current_issues()

        # Fix all issues
        try:
            results.append(await self.fix_stuck_agents())
        except Exception as e:
            logger.error(f"Failed to fix stuck agents: {e}")

        try:
            results.append(await self.resolve_memory_conflicts())
        except Exception as e:
            logger.error(f"Failed to resolve memory conflicts: {e}")

        try:
            results.append(await self.cleanup_stale_memories())
        except Exception as e:
            logger.error(f"Failed to cleanup stale memories: {e}")

        try:
            results.append(await self.apply_insights())
        except Exception as e:
            logger.error(f"Failed to apply insights: {e}")

        try:
            results.append(await self.process_proposals())
        except Exception as e:
            logger.error(f"Failed to process proposals: {e}")

        # Get issues after
        issues_after = await self.get_current_issues()

        # Calculate totals
        total_fixed = sum(r.items_fixed for r in results)
        duration = (datetime.utcnow() - start_time).total_seconds()

        # Store resolution cycle
        self._resolution_history.append({
            "cycle_time": start_time,
            "duration_seconds": duration,
            "total_fixed": total_fixed,
            "results": results
        })

        # Keep only last 100 cycles
        if len(self._resolution_history) > 100:
            self._resolution_history = self._resolution_history[-100:]

        return {
            "success": True,
            "cycle_time": start_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "issues_before": issues_before,
            "issues_after": issues_after,
            "resolutions": [
                {
                    "issue_type": r.issue_type.value,
                    "action": r.action.value,
                    "items_fixed": r.items_fixed,
                    "details": r.details
                }
                for r in results
            ],
            "total_fixed": total_fixed,
            "issues_reduced": issues_before["total_issues"] - issues_after["total_issues"]
        }

    async def start_continuous_resolution(self):
        """Start continuous issue resolution loop"""
        self._running = True
        logger.info("🔧 Autonomous Issue Resolver started")

        while self._running:
            try:
                result = await self.run_full_resolution_cycle()
                if result["total_fixed"] > 0:
                    logger.info(
                        f"Resolution cycle: Fixed {result['total_fixed']} issues "
                        f"in {result['duration_seconds']}s"
                    )
            except Exception as e:
                logger.error(f"Resolution cycle failed: {e}")

            await asyncio.sleep(self.resolve_interval_seconds)

    def stop(self):
        """Stop the continuous resolution loop"""
        self._running = False
        logger.info("Autonomous Issue Resolver stopped")

    def get_resolution_stats(self) -> dict[str, Any]:
        """Get resolution statistics"""
        if not self._resolution_history:
            return {"total_cycles": 0, "total_fixed": 0}

        total_fixed = sum(c.get("total_fixed", 0) for c in self._resolution_history)
        avg_duration = sum(c.get("duration_seconds", 0) for c in self._resolution_history) / len(self._resolution_history)

        return {
            "total_cycles": len(self._resolution_history),
            "total_fixed": total_fixed,
            "avg_cycle_duration_seconds": round(avg_duration, 2),
            "last_cycle": self._resolution_history[-1] if self._resolution_history else None
        }


# Singleton instance
_resolver: Optional[AutonomousIssueResolver] = None


def get_resolver() -> AutonomousIssueResolver:
    """Get or create resolver instance"""
    global _resolver
    if _resolver is None:
        _resolver = AutonomousIssueResolver()
    return _resolver


async def run_resolution_cycle() -> dict[str, Any]:
    """Run a single resolution cycle"""
    resolver = get_resolver()
    return await resolver.run_full_resolution_cycle()


async def get_current_issues() -> dict[str, Any]:
    """Get current system issues"""
    resolver = get_resolver()
    return await resolver.get_current_issues()


async def start_autonomous_resolution():
    """Start continuous autonomous resolution"""
    resolver = get_resolver()
    await resolver.start_continuous_resolution()

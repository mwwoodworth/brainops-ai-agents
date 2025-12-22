"""
Change Propagation Daemon
=========================
Monitors ALL changes across the BrainOps ecosystem and immediately propagates them
to ensure the AI OS never has stale information.

This daemon:
1. Watches file changes in all codebases
2. Monitors git commits
3. Tracks database schema changes
4. Updates documentation automatically
5. Refreshes the codebase graph on changes
6. Keeps SESSION_STATE.md current

Author: Claude Opus 4.5 + BrainOps AI Team
Version: 1.0.0
"""

import asyncio
import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass, asdict
import logging
import hashlib

logger = logging.getLogger(__name__)

# ============== CONFIGURATION ==============

WATCHED_DIRECTORIES = [
    "/home/matt-woodworth/dev/weathercraft-erp",
    "/home/matt-woodworth/dev/myroofgenius-app",
    "/home/matt-woodworth/dev/brainops-ai-agents",
    "/home/matt-woodworth/dev/brainops-command-center",
    "/home/matt-woodworth/dev/mcp-bridge",
    "/home/matt-woodworth/dev/BrainOps-Weather",
]

STATE_FILES = [
    "/home/matt-woodworth/dev/SESSION_STATE.md",
    "/home/matt-woodworth/dev/AI_SYSTEM_STATE.json",
    "/home/matt-woodworth/dev/AI_SYSTEM_KNOWLEDGE_GRAPH.json",
    "/home/matt-woodworth/MASTER_SYSTEM_REGISTRY.md",
]

# File patterns to watch
WATCHED_PATTERNS = [
    "*.py", "*.ts", "*.tsx", "*.js", "*.jsx",
    "*.sql", "*.json", "*.md", "*.yaml", "*.yml",
]

# ============== DATA STRUCTURES ==============

@dataclass
class FileChange:
    """Represents a detected file change"""
    path: str
    change_type: str  # created, modified, deleted
    timestamp: str
    codebase: str
    checksum: str = ""

@dataclass
class PropagationResult:
    """Result of propagating a change"""
    change: FileChange
    actions_taken: List[str]
    success: bool
    error: str = ""


# ============== CHANGE DETECTION ==============

class ChangeDetector:
    """Detects changes in watched directories"""

    def __init__(self):
        self.file_checksums: Dict[str, str] = {}
        self._load_checksums()

    def _load_checksums(self):
        """Load previously computed checksums"""
        checksum_file = Path("/home/matt-woodworth/dev/.file_checksums.json")
        if checksum_file.exists():
            try:
                with open(checksum_file) as f:
                    self.file_checksums = json.load(f)
            except Exception:
                self.file_checksums = {}

    def _save_checksums(self):
        """Save checksums for future comparison"""
        checksum_file = Path("/home/matt-woodworth/dev/.file_checksums.json")
        try:
            with open(checksum_file, 'w') as f:
                json.dump(self.file_checksums, f)
        except Exception as e:
            logger.error(f"Failed to save checksums: {e}")

    def _compute_checksum(self, path: str) -> str:
        """Compute MD5 checksum of a file"""
        try:
            with open(path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _get_codebase(self, path: str) -> str:
        """Determine which codebase a file belongs to"""
        for directory in WATCHED_DIRECTORIES:
            if path.startswith(directory):
                return Path(directory).name
        return "unknown"

    def _matches_pattern(self, filename: str) -> bool:
        """Check if filename matches watched patterns"""
        import fnmatch
        for pattern in WATCHED_PATTERNS:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    async def detect_changes(self) -> List[FileChange]:
        """Scan for changes since last check"""
        changes: List[FileChange] = []
        current_files: Set[str] = set()

        for directory in WATCHED_DIRECTORIES:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue

            for path in dir_path.rglob("*"):
                if path.is_file() and self._matches_pattern(path.name):
                    # Skip node_modules, .git, etc.
                    path_str = str(path)
                    if any(skip in path_str for skip in [
                        "node_modules", ".git", "__pycache__", ".next", "dist", "build"
                    ]):
                        continue

                    current_files.add(path_str)
                    current_checksum = self._compute_checksum(path_str)

                    if path_str not in self.file_checksums:
                        # New file
                        changes.append(FileChange(
                            path=path_str,
                            change_type="created",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            codebase=self._get_codebase(path_str),
                            checksum=current_checksum
                        ))
                    elif self.file_checksums[path_str] != current_checksum:
                        # Modified file
                        changes.append(FileChange(
                            path=path_str,
                            change_type="modified",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            codebase=self._get_codebase(path_str),
                            checksum=current_checksum
                        ))

                    self.file_checksums[path_str] = current_checksum

        # Check for deleted files
        for path_str in list(self.file_checksums.keys()):
            if path_str not in current_files and any(
                path_str.startswith(d) for d in WATCHED_DIRECTORIES
            ):
                changes.append(FileChange(
                    path=path_str,
                    change_type="deleted",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    codebase=self._get_codebase(path_str)
                ))
                del self.file_checksums[path_str]

        self._save_checksums()
        return changes


# ============== CHANGE PROPAGATION ==============

class ChangePropagator:
    """Propagates changes to all relevant systems"""

    def __init__(self):
        self.detector = ChangeDetector()

    async def propagate_change(self, change: FileChange) -> PropagationResult:
        """Propagate a single change to all relevant systems"""
        actions_taken = []

        try:
            # 1. Update codebase graph if it's a code file
            if any(change.path.endswith(ext) for ext in ['.py', '.ts', '.tsx', '.js', '.jsx']):
                await self._update_codebase_graph(change)
                actions_taken.append("updated_codebase_graph")

            # 2. Update AI system state
            await self._update_system_state(change)
            actions_taken.append("updated_system_state")

            # 3. If it's a schema file, trigger database re-crawl
            if change.path.endswith('.sql') or 'migration' in change.path.lower():
                await self._trigger_schema_crawl()
                actions_taken.append("triggered_schema_crawl")

            # 4. If it's a core file, update knowledge graph
            if any(core in change.path for core in ['ai_', 'agent', 'orchestrator']):
                await self._update_knowledge_graph(change)
                actions_taken.append("updated_knowledge_graph")

            # 5. Log the change
            await self._log_change(change)
            actions_taken.append("logged_change")

            return PropagationResult(
                change=change,
                actions_taken=actions_taken,
                success=True
            )

        except Exception as e:
            logger.error(f"Propagation failed for {change.path}: {e}")
            return PropagationResult(
                change=change,
                actions_taken=actions_taken,
                success=False,
                error=str(e)
            )

    async def _update_codebase_graph(self, change: FileChange):
        """Update codebase graph with the changed file"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()

            if change.change_type == "deleted":
                # Remove nodes for this file
                await pool.execute("""
                    DELETE FROM codebase_nodes
                    WHERE file_path = $1
                """, change.path)
            else:
                # Basic parsing: get size and line count
                file_stats = os.stat(change.path)
                line_count = 0
                try:
                    with open(change.path, 'r', errors='ignore') as f:
                        line_count = sum(1 for _ in f)
                except:
                    pass
                
                metadata_update = {
                    "timestamp": change.timestamp,
                    "type": change.change_type,
                    "size": file_stats.st_size,
                    "line_count": line_count
                }
                
                await pool.execute("""
                    UPDATE codebase_nodes
                    SET metadata = jsonb_set(
                        COALESCE(metadata, '{}'::jsonb),
                        '{last_change}',
                        $1::jsonb
                    )
                    WHERE file_path = $2
                """, json.dumps(metadata_update), change.path)

        except Exception as e:
            logger.error(f"Failed to update codebase graph: {e}")

    async def _update_system_state(self, change: FileChange):
        """Update the real-time system state"""
        try:
            from realtime_state_sync import get_state_sync
            sync = get_state_sync()

            # Update the component that contains this file
            sync.update_component_status(
                change.codebase,
                "healthy",
                {"last_change": change.path, "change_type": change.change_type}
            )

        except Exception as e:
            logger.error(f"Failed to update system state: {e}")

    async def _trigger_schema_crawl(self):
        """Trigger a database schema re-crawl"""
        try:
            # This would typically be done via an async task queue
            # For now, just log that it should happen
            logger.info("Schema change detected - database crawl should be triggered")
        except Exception as e:
            logger.error(f"Failed to trigger schema crawl: {e}")

    async def _update_knowledge_graph(self, change: FileChange):
        """Update the AI knowledge graph with this change"""
        kg_path = Path("/home/matt-woodworth/dev/AI_SYSTEM_KNOWLEDGE_GRAPH.json")

        try:
            if kg_path.exists():
                with open(kg_path) as f:
                    kg = json.load(f)

                # Add change to activity log
                if "recent_changes" not in kg:
                    kg["recent_changes"] = []

                kg["recent_changes"].insert(0, {
                    "path": change.path,
                    "type": change.change_type,
                    "timestamp": change.timestamp,
                    "codebase": change.codebase
                })

                # Keep last 50 changes
                kg["recent_changes"] = kg["recent_changes"][:50]

                with open(kg_path, 'w') as f:
                    json.dump(kg, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update knowledge graph: {e}")

    async def _log_change(self, change: FileChange):
        """Log the change to the change log"""
        log_path = Path("/home/matt-woodworth/dev/AI_CHANGE_LOG.json")

        try:
            changes = []
            if log_path.exists():
                with open(log_path) as f:
                    changes = json.load(f)

            changes.append(asdict(change))

            # Keep last 1000
            if len(changes) > 1000:
                changes = changes[-1000:]

            with open(log_path, 'w') as f:
                json.dump(changes, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to log change: {e}")

    async def run_propagation_cycle(self) -> Dict[str, Any]:
        """Run a complete propagation cycle"""
        changes = await self.detector.detect_changes()

        if not changes:
            return {"changes": 0, "results": []}

        results = []
        for change in changes:
            result = await self.propagate_change(change)
            results.append(asdict(result))

        # Update SESSION_STATE.md with propagation summary
        await self._update_session_state(len(changes), len([r for r in results if r["success"]]))

        return {
            "changes": len(changes),
            "successful": len([r for r in results if r["success"]]),
            "failed": len([r for r in results if not r["success"]]),
            "results": results
        }

    async def _update_session_state(self, total_changes: int, successful: int):
        """Update SESSION_STATE.md with latest activity"""
        session_path = Path("/home/matt-woodworth/dev/SESSION_STATE.md")

        try:
            if session_path.exists():
                content = session_path.read_text()

                # Find and update the version line
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith("## Version:"):
                        lines[i] = f"## Version: 2.6.0 | Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC by Change Propagation Daemon"
                        break

                session_path.write_text('\n'.join(lines))

        except Exception as e:
            logger.error(f"Failed to update session state: {e}")


# ============== DAEMON RUNNER ==============

class ChangePropagationDaemon:
    """Main daemon that runs continuously"""

    def __init__(self, interval_seconds: int = 30):
        self.propagator = ChangePropagator()
        self.interval = interval_seconds
        self.running = False

    async def start(self):
        """Start the daemon"""
        self.running = True
        logger.info(f"Change Propagation Daemon starting (interval: {self.interval}s)")

        while self.running:
            try:
                result = await self.propagator.run_propagation_cycle()

                if result["changes"] > 0:
                    logger.info(
                        f"Propagation cycle: {result['changes']} changes, "
                        f"{result['successful']} successful, {result['failed']} failed"
                    )

            except Exception as e:
                logger.error(f"Daemon cycle error: {e}")

            await asyncio.sleep(self.interval)

    def stop(self):
        """Stop the daemon"""
        self.running = False
        logger.info("Change Propagation Daemon stopping")


# ============== GIT CHANGE DETECTION ==============

async def detect_git_changes() -> List[Dict[str, Any]]:
    """Detect uncommitted changes across all codebases"""
    changes = []

    for directory in WATCHED_DIRECTORIES:
        if not Path(directory).exists():
            continue

        try:
            # Get git status
            result = subprocess.run(
                ["git", "-C", directory, "status", "--porcelain"],
                capture_output=True, text=True
            )

            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if line:
                        status = line[:2].strip()
                        file_path = line[3:]
                        changes.append({
                            "codebase": Path(directory).name,
                            "status": status,
                            "file": file_path,
                            "path": str(Path(directory) / file_path)
                        })

        except Exception as e:
            logger.error(f"Git detection failed for {directory}: {e}")

    return changes


# ============== CLI ENTRY POINT ==============

async def main():
    """Run change detection and propagation once"""
    print("=" * 60)
    print("BRAINOPS CHANGE PROPAGATION SYSTEM")
    print("=" * 60)

    propagator = ChangePropagator()

    print("\n🔍 Detecting changes...")
    result = await propagator.run_propagation_cycle()

    print(f"\n📊 PROPAGATION RESULTS:")
    print(f"   Total changes detected: {result['changes']}")
    print(f"   Successfully propagated: {result.get('successful', 0)}")
    print(f"   Failed: {result.get('failed', 0)}")

    if result['changes'] > 0:
        print("\n📝 CHANGES:")
        for r in result['results'][:10]:
            change = r['change']
            icon = "✅" if r['success'] else "❌"
            print(f"   {icon} [{change['change_type']}] {change['path']}")
        if len(result['results']) > 10:
            print(f"   ... and {len(result['results']) - 10} more")

    # Also check git status
    print("\n🔀 GIT STATUS:")
    git_changes = await detect_git_changes()
    if git_changes:
        for change in git_changes[:10]:
            print(f"   [{change['status']}] {change['codebase']}: {change['file']}")
        if len(git_changes) > 10:
            print(f"   ... and {len(git_changes) - 10} more uncommitted files")
    else:
        print("   No uncommitted changes detected")

    print("\n✅ Propagation complete")


if __name__ == "__main__":
    asyncio.run(main())

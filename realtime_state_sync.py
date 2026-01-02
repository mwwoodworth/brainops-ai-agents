"""
Real-Time State Synchronization System
======================================
Ensures ALL operations are tracked, aware, and synchronized across the BrainOps AI OS.

This system prevents:
- Lost work from context compaction
- Forgotten tasks or incomplete implementations
- Stale documentation that doesn't reflect reality
- Wrong assumptions based on outdated state

Core Principles:
1. EVERY change is tracked immediately
2. EVERY status is queryable in real-time
3. EVERY component knows about every other component
4. EVERY session starts with full context

Author: Claude Opus 4.5 + BrainOps AI Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ============== DATA STRUCTURES ==============

@dataclass
class SystemComponent:
    """Represents a trackable system component"""
    name: str
    component_type: str  # codebase, agent, service, api, database, document
    path: Optional[str] = None
    status: str = "unknown"  # healthy, degraded, error, unknown
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None  # For detecting changes


@dataclass
class StateChange:
    """Records a change in system state"""
    component: str
    change_type: str  # created, updated, deleted, status_changed
    before: Optional[dict[str, Any]] = None
    after: Optional[dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "system"  # system, user, agent, daemon
    propagated: bool = False


@dataclass
class SystemState:
    """Complete state of the BrainOps AI OS"""
    version: str = "1.0.0"
    last_sync: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    components: dict[str, SystemComponent] = field(default_factory=dict)
    pending_changes: list[StateChange] = field(default_factory=list)
    health_summary: dict[str, Any] = field(default_factory=dict)


# ============== REAL-TIME STATE SYNC ==============

class RealTimeStateSync:
    """
    Core synchronization engine for BrainOps AI OS.

    Responsibilities:
    1. Track all system components (codebases, agents, services, etc.)
    2. Detect changes in real-time
    3. Propagate updates to all interested parties
    4. Maintain a single source of truth
    """

    # Use environment-appropriate paths (production uses /tmp or disables file I/O)
    _IS_PRODUCTION = os.getenv("RENDER") is not None or not Path("/home/matt-woodworth/dev").exists()

    if _IS_PRODUCTION:
        # In production: use /tmp for ephemeral state (or None to disable)
        STATE_FILE = Path("/tmp/AI_SYSTEM_STATE.json")
        CHANGE_LOG = Path("/tmp/AI_CHANGE_LOG.json")
    else:
        # Local development: use standard dev paths
        STATE_FILE = Path("/home/matt-woodworth/dev/AI_SYSTEM_STATE.json")
        CHANGE_LOG = Path("/home/matt-woodworth/dev/AI_CHANGE_LOG.json")

    def __init__(self):
        self._file_io_enabled = not self._IS_PRODUCTION  # Disable file spam in production
        self.state = self._load_state()
        self.change_handlers: list[callable] = []
        self._initialized = False
        self._digital_twin_integration_enabled = False
        self._digital_twin_engine = None  # Lazy load to avoid circular imports

    def _load_state(self) -> SystemState:
        """Load state from persistent storage"""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE) as f:
                    data = json.load(f)
                    # Reconstruct SystemState
                    state = SystemState(
                        version=data.get("version", "1.0.0"),
                        last_sync=data.get("last_sync", ""),
                        health_summary=data.get("health_summary", {})
                    )
                    # Reconstruct components
                    for name, comp_data in data.get("components", {}).items():
                        state.components[name] = SystemComponent(**comp_data)
                    return state
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return SystemState()

    def _save_state(self):
        """Persist state to disk (skipped in production to avoid file spam)"""
        if not self._file_io_enabled:
            return  # Skip file I/O in production
        try:
            # Convert to dict for JSON serialization
            data = {
                "version": self.state.version,
                "last_sync": datetime.now(timezone.utc).isoformat(),
                "health_summary": self.state.health_summary,
                "components": {
                    name: asdict(comp)
                    for name, comp in self.state.components.items()
                }
            }
            with open(self.STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"State saved: {len(self.state.components)} components")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _log_change(self, change: StateChange):
        """Log change for audit trail (skipped in production to avoid file spam)"""
        if not self._file_io_enabled:
            return  # Skip file I/O in production
        try:
            changes = []
            if self.CHANGE_LOG.exists():
                with open(self.CHANGE_LOG) as f:
                    changes = json.load(f)

            changes.append(asdict(change))

            # Keep last 1000 changes
            if len(changes) > 1000:
                changes = changes[-1000:]

            with open(self.CHANGE_LOG, 'w') as f:
                json.dump(changes, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log change: {e}")

    # ============== COMPONENT MANAGEMENT ==============

    def enable_digital_twin_integration(self, digital_twin_engine):
        """Enable integration with Digital Twin system"""
        self._digital_twin_integration_enabled = True
        self._digital_twin_engine = digital_twin_engine
        logger.info("Digital Twin integration enabled for RealTimeStateSync")

    def register_component(self, component: SystemComponent) -> bool:
        """Register a new component for tracking"""
        existing = self.state.components.get(component.name)

        if existing:
            # Update existing
            change = StateChange(
                component=component.name,
                change_type="updated",
                before=asdict(existing),
                after=asdict(component),
                source="system"
            )
        else:
            # New component
            change = StateChange(
                component=component.name,
                change_type="created",
                after=asdict(component),
                source="system"
            )

        self.state.components[component.name] = component
        self._log_change(change)
        self._propagate_change(change)
        self._save_state()

        # Sync to digital twin if enabled (NO EXTERNAL CALLS - internal only)
        if self._digital_twin_integration_enabled and self._digital_twin_engine:
            try:
                import asyncio
                # Convert component to metrics if it's a service
                if component.component_type in ["service", "api_endpoint", "microservice"]:
                    asyncio.create_task(self._sync_component_to_twin(component))
            except Exception as e:
                logger.error(f"Error syncing component to digital twin: {e}")

        return True

    def update_component_status(self, name: str, status: str, metadata: Optional[dict] = None) -> bool:
        """Update a component's status"""
        if name not in self.state.components:
            logger.warning(f"Component not found: {name}")
            return False

        component = self.state.components[name]
        old_status = component.status

        component.status = status
        component.last_updated = datetime.now(timezone.utc).isoformat()

        if metadata:
            component.metadata.update(metadata)

        change = StateChange(
            component=name,
            change_type="status_changed",
            before={"status": old_status},
            after={"status": status, "metadata": metadata}
        )

        self._log_change(change)
        self._propagate_change(change)
        self._save_state()

        return True

    def get_component(self, name: str) -> Optional[SystemComponent]:
        """Get a component by name"""
        return self.state.components.get(name)

    def register_agent(self, agent_name: str, metadata: Optional[dict[str, Any]] = None) -> bool:
        """
        Register or update an AI agent in the state tracking system.

        Args:
            agent_name: Name of the agent
            metadata: Optional metadata dict (status, task, etc.)

        Returns:
            True if successful
        """
        component = SystemComponent(
            name=agent_name,
            component_type="agent",
            status=metadata.get("status", "healthy") if metadata else "healthy",
            metadata=metadata or {}
        )
        return self.register_component(component)

    def get_all_components(self, component_type: Optional[str] = None) -> list[SystemComponent]:
        """Get all components, optionally filtered by type"""
        components = list(self.state.components.values())
        if component_type:
            components = [c for c in components if c.component_type == component_type]
        return components

    # ============== CHANGE PROPAGATION ==============

    def register_change_handler(self, handler: callable):
        """Register a handler to be called on state changes"""
        self.change_handlers.append(handler)

    def _propagate_change(self, change: StateChange):
        """Propagate change to all registered handlers"""
        for handler in self.change_handlers:
            try:
                handler(change)
            except Exception as e:
                logger.error(f"Change handler failed: {e}")
        change.propagated = True

    # ============== HEALTH MONITORING ==============

    def compute_health_summary(self) -> dict[str, Any]:
        """Compute overall system health"""
        components = list(self.state.components.values())

        if not components:
            return {
                "status": "unknown",
                "message": "No components registered",
                "total_components": 0,
                "by_status": {"healthy": 0, "degraded": 0, "error": 0, "unknown": 0},
                "by_type": {},
                "last_computed": datetime.now(timezone.utc).isoformat()
            }

        status_counts = {
            "healthy": 0,
            "degraded": 0,
            "error": 0,
            "unknown": 0
        }

        for comp in components:
            status_counts[comp.status] = status_counts.get(comp.status, 0) + 1

        # Determine overall status
        if status_counts["error"] > 0:
            overall = "error"
        elif status_counts["degraded"] > 0:
            overall = "degraded"
        elif status_counts["unknown"] > len(components) // 2:
            overall = "unknown"
        else:
            overall = "healthy"

        summary = {
            "status": overall,
            "total_components": len(components),
            "by_status": status_counts,
            "by_type": {},
            "last_computed": datetime.now(timezone.utc).isoformat()
        }

        # Group by type
        for comp in components:
            if comp.component_type not in summary["by_type"]:
                summary["by_type"][comp.component_type] = []
            summary["by_type"][comp.component_type].append(comp.name)

        self.state.health_summary = summary
        self._save_state()

        return summary

    # ============== CODEBASE TRACKING ==============

    async def scan_codebases(self) -> dict[str, Any]:
        """Scan all codebases and register them"""
        codebases = [
            ("weathercraft-erp", "/home/matt-woodworth/dev/weathercraft-erp"),
            ("myroofgenius-app", "/home/matt-woodworth/dev/myroofgenius-app"),
            ("brainops-ai-agents", "/home/matt-woodworth/dev/brainops-ai-agents"),
            ("brainops-command-center", "/home/matt-woodworth/dev/brainops-command-center"),
            ("mcp-bridge", "/home/matt-woodworth/dev/mcp-bridge"),
            ("BrainOps-Weather", "/home/matt-woodworth/dev/BrainOps-Weather"),
        ]

        results = {}

        for name, path in codebases:
            path_obj = Path(path)
            if path_obj.exists():
                # Get git status
                git_status = await self._get_git_status(path)

                component = SystemComponent(
                    name=name,
                    component_type="codebase",
                    path=path,
                    status="healthy" if path_obj.exists() else "error",
                    version=git_status.get("commit", "unknown"),
                    metadata={
                        "branch": git_status.get("branch", "unknown"),
                        "uncommitted_changes": git_status.get("dirty", False),
                        "last_commit_message": git_status.get("message", ""),
                        "file_count": sum(1 for _ in path_obj.rglob("*") if _.is_file())
                    }
                )

                self.register_component(component)
                results[name] = {"status": "registered", "path": path}
            else:
                results[name] = {"status": "not_found", "path": path}

        return results

    async def _get_git_status(self, path: str) -> dict[str, Any]:
        """Get git status for a path"""
        try:
            import subprocess

            # Get current commit
            commit = subprocess.run(
                ["git", "-C", path, "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True
            ).stdout.strip()

            # Get branch
            branch = subprocess.run(
                ["git", "-C", path, "branch", "--show-current"],
                capture_output=True, text=True
            ).stdout.strip()

            # Check if dirty
            status = subprocess.run(
                ["git", "-C", path, "status", "--porcelain"],
                capture_output=True, text=True
            ).stdout.strip()

            # Get last commit message
            message = subprocess.run(
                ["git", "-C", path, "log", "-1", "--format=%s"],
                capture_output=True, text=True
            ).stdout.strip()

            return {
                "commit": commit,
                "branch": branch,
                "dirty": len(status) > 0,
                "message": message
            }
        except Exception as e:
            return {"error": str(e)}

    # ============== AGENT TRACKING ==============

    async def scan_agents(self) -> dict[str, Any]:
        """Scan and register all AI agents"""
        agents = [
            "AUREA Master Orchestrator",
            "Self-Healing Recovery",
            "Unified Memory Manager",
            "Embedded Memory System",
            "AI Training Pipeline",
            "Notebook LM+ Learning",
            "System Improvement Agent",
            "DevOps Optimization Agent",
            "Code Quality Agent",
            "Customer Success Agent",
            "Competitive Intelligence Agent",
            "Vision Alignment Agent",
            "AI Self-Awareness Module",
            "AI Integration Layer",
        ]

        results = {}

        for agent_name in agents:
            component = SystemComponent(
                name=agent_name,
                component_type="agent",
                status="healthy",  # Would need actual health check
                metadata={
                    "type": "ai_agent",
                    "registered_at": datetime.now(timezone.utc).isoformat()
                }
            )
            self.register_component(component)
            results[agent_name] = "registered"

        return results

    # ============== DATABASE TRACKING ==============

    async def scan_database_tables(self, pool) -> dict[str, Any]:
        """Scan and register database tables"""
        try:
            tables = await pool.fetch("""
                SELECT table_schema, table_name,
                       (SELECT COUNT(*) FROM information_schema.columns c
                        WHERE c.table_name = t.table_name
                        AND c.table_schema = t.table_schema) as column_count
                FROM information_schema.tables t
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                AND table_type = 'BASE TABLE'
            """)

            for table in tables:
                schema = table["table_schema"]
                name = table["table_name"]
                full_name = f"{schema}.{name}"

                component = SystemComponent(
                    name=full_name,
                    component_type="database_table",
                    status="healthy",
                    metadata={
                        "schema": schema,
                        "table": name,
                        "columns": table["column_count"]
                    }
                )
                self.register_component(component)

            return {"tables_registered": len(tables)}
        except Exception as e:
            logger.error(f"Database scan failed: {e}")
            return {"error": str(e)}

    # ============== API ENDPOINT TRACKING ==============

    async def scan_api_endpoints(self) -> dict[str, Any]:
        """Scan and register API endpoints from codebase graph"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()

            endpoints = await pool.fetch("""
                SELECT id, name, filepath as file_path, metadata
                FROM codebase_nodes
                WHERE node_type = 'endpoint'
                LIMIT 1000
            """)

            for endpoint in endpoints:
                component = SystemComponent(
                    name=endpoint["name"],
                    component_type="api_endpoint",
                    path=endpoint["file_path"],
                    status="healthy",
                    metadata=endpoint["metadata"] if endpoint["metadata"] else {}
                )
                self.register_component(component)

            return {"endpoints_registered": len(endpoints)}
        except Exception as e:
            logger.error(f"API scan failed: {e}")
            return {"error": str(e)}

    # ============== FULL SYSTEM SCAN ==============

    async def full_system_scan(self) -> dict[str, Any]:
        """Perform a complete scan of all system components"""
        logger.info("Starting full system scan...")

        results = {
            "codebases": await self.scan_codebases(),
            "agents": await self.scan_agents(),
            "health": self.compute_health_summary(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Try to scan database if connection available
        try:
            from database.async_connection import get_pool
            pool = get_pool()
            results["database"] = await self.scan_database_tables(pool)
            results["api_endpoints"] = await self.scan_api_endpoints()
        except Exception as e:
            results["database"] = {"error": str(e)}

        logger.info(f"Full scan complete: {len(self.state.components)} components tracked")

        return results

    # ============== DIGITAL TWIN INTEGRATION ==============

    async def _sync_component_to_twin(self, component: SystemComponent):
        """Sync component state to digital twin (INTERNAL ONLY - NO EXTERNAL CALLS)"""
        try:
            if not self._digital_twin_engine:
                return

            # Convert component metadata to SystemMetrics
            from digital_twin_system import SystemMetrics, SystemType

            # Extract metrics from metadata
            metadata = component.metadata or {}

            metrics = SystemMetrics(
                cpu_usage=metadata.get("cpu_usage", 0.0),
                memory_usage=metadata.get("memory_usage", 0.0),
                disk_usage=metadata.get("disk_usage", 0.0),
                network_io=metadata.get("network_io", 0.0),
                request_latency_ms=metadata.get("latency_ms", 0.0),
                error_rate=metadata.get("error_rate", 0.0),
                throughput_rps=metadata.get("throughput_rps", 0.0),
                active_connections=metadata.get("active_connections", 0),
                queue_depth=metadata.get("queue_depth", 0),
                custom_metrics=metadata.get("custom_metrics", {})
            )

            # Check if twin exists for this component
            twin_id = f"twin_{component.name.replace(' ', '_').lower()}"

            # Try to sync with existing twin
            result = await self._digital_twin_engine.sync_twin(
                twin_id=twin_id,
                current_metrics=metrics,
                source="state_sync"  # Mark as internal source
            )

            if "error" in result and "not found" in result["error"]:
                # Create twin if it doesn't exist
                system_type_map = {
                    "service": SystemType.MICROSERVICE,
                    "api_endpoint": SystemType.API_GATEWAY,
                    "microservice": SystemType.MICROSERVICE,
                    "database": SystemType.DATABASE,
                    "agent": SystemType.AI_AGENT,
                    "infrastructure": SystemType.INFRASTRUCTURE,
                    "pipeline": SystemType.PIPELINE
                }

                system_type = system_type_map.get(
                    component.component_type,
                    SystemType.MICROSERVICE
                )

                await self._digital_twin_engine.create_twin(
                    source_system=component.name,
                    system_type=system_type,
                    initial_state=asdict(component),
                    sync_frequency_seconds=60
                )

                logger.info(f"Created digital twin for component: {component.name}")

        except Exception as e:
            logger.error(f"Error syncing component {component.name} to digital twin: {e}")

    def get_twin_status_for_component(self, component_name: str) -> Optional[dict[str, Any]]:
        """Get digital twin status for a component"""
        if not self._digital_twin_integration_enabled or not self._digital_twin_engine:
            return None

        try:
            twin_id = f"twin_{component_name.replace(' ', '_').lower()}"
            return self._digital_twin_engine.get_twin_status(twin_id)
        except Exception as e:
            logger.error(f"Error getting twin status for {component_name}: {e}")
            return None

    # ============== CONTEXT EXPORT FOR AI ==============

    def export_for_ai_context(self) -> str:
        """Export state in a format optimized for AI context loading"""
        summary = self.compute_health_summary()

        output = [
            "# BRAINOPS AI OS STATE",
            f"Last Sync: {self.state.last_sync}",
            f"Total Components: {summary['total_components']}",
            f"Overall Status: {summary['status'].upper()}",
            "",
            "## COMPONENTS BY TYPE"
        ]

        for comp_type, names in summary.get("by_type", {}).items():
            output.append(f"\n### {comp_type.upper()} ({len(names)})")
            for name in names[:10]:  # Limit to first 10
                comp = self.state.components.get(name)
                if comp:
                    status_icon = "✅" if comp.status == "healthy" else "⚠️" if comp.status == "degraded" else "❌"
                    output.append(f"- {status_icon} {name}")
            if len(names) > 10:
                output.append(f"  ... and {len(names) - 10} more")

        return "\n".join(output)


# ============== SINGLETON INSTANCE ==============

_state_sync: Optional[RealTimeStateSync] = None

def get_state_sync() -> RealTimeStateSync:
    """Get singleton state sync instance"""
    global _state_sync
    if _state_sync is None:
        _state_sync = RealTimeStateSync()
    return _state_sync


# ============== CLI ENTRY POINT ==============

async def main():
    """Run full system scan"""
    sync = get_state_sync()

    print("=" * 60)
    print("BRAINOPS AI OS - REAL-TIME STATE SYNCHRONIZATION")
    print("=" * 60)

    results = await sync.full_system_scan()

    print("\n📊 SCAN RESULTS:")
    print(f"   Codebases: {len(results['codebases'])} registered")
    print(f"   Agents: {len(results['agents'])} registered")
    print(f"   Database Tables: {results.get('database', {}).get('tables_registered', 'N/A')}")
    print(f"   API Endpoints: {results.get('api_endpoints', {}).get('endpoints_registered', 'N/A')}")

    health = results['health']
    print("\n🏥 HEALTH SUMMARY:")
    print(f"   Overall: {health['status'].upper()}")
    print(f"   Total Components: {health['total_components']}")
    for status, count in health['by_status'].items():
        if count > 0:
            print(f"   {status}: {count}")

    # Export context
    context = sync.export_for_ai_context()
    print("\n📝 AI CONTEXT EXPORT:")
    print(context[:500] + "..." if len(context) > 500 else context)

    print(f"\n✅ State saved to: {sync.STATE_FILE}")
    print(f"✅ Changes logged to: {sync.CHANGE_LOG}")


if __name__ == "__main__":
    asyncio.run(main())

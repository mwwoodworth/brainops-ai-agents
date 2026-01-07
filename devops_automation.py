#!/usr/bin/env python3
"""
DEVOPS AUTOMATION MODULE
Permanent knowledge and automated operations for BrainOps AI OS.

Features:
- Automated deployment triggers
- Self-healing with automatic rollback
- Knowledge persistence across sessions
- CI/CD pipeline automation
- Service health correlation
- Intelligent escalation

Author: BrainOps AI OS
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class DeploymentStatus(str, Enum):
    """Deployment status states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ServiceType(str, Enum):
    """Service deployment types"""
    RENDER = "render"
    VERCEL = "vercel"
    DOCKER = "docker"
    GITHUB_ACTIONS = "github_actions"


@dataclass
class DeploymentRecord:
    """Record of a deployment"""
    deployment_id: str
    service: str
    service_type: ServiceType
    version: str
    status: DeploymentStatus
    triggered_by: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    rollback_to: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEntry:
    """Knowledge to persist permanently"""
    key: str
    value: Any
    category: str  # devops, deployment, incident, learning
    source: str
    created_at: str
    expires_at: Optional[str] = None


class DevOpsAutomation:
    """
    Automated DevOps operations with permanent knowledge retention.
    """

    def __init__(self):
        self.render_api_key = os.getenv("RENDER_API_KEY", "")
        self.vercel_token = os.getenv("VERCEL_TOKEN", "")
        self.docker_hub_user = os.getenv("DOCKER_HUB_USER", "mwwoodworth")
        self.brainops_api_key = os.getenv("BRAINOPS_API_KEY")

        # Service registry with Render service IDs
        self.services = {
            "brainops_ai_agents": {
                "type": ServiceType.RENDER,
                "render_id": "srv-d413iu75r7bs738btc10",
                "docker_image": "mwwoodworth/brainops-ai-agents",
                "health_url": "https://brainops-ai-agents.onrender.com/health",
                "critical": True
            },
            "brainops_backend": {
                "type": ServiceType.RENDER,
                "render_id": "srv-d1tfs4idbo4c73di6k00",
                "health_url": "https://brainops-backend-prod.onrender.com/health",
                "critical": True
            },
            "mcp_bridge": {
                "type": ServiceType.RENDER,
                "render_id": "srv-d4rhvg63jp1c73918770",
                "health_url": "https://brainops-mcp-bridge.onrender.com/health",
                "critical": True
            },
            "weathercraft_erp": {
                "type": ServiceType.VERCEL,
                "project": "weathercraft-erp",
                "health_url": "https://weathercraft-erp.vercel.app",
                "critical": True
            },
            "myroofgenius": {
                "type": ServiceType.VERCEL,
                "project": "myroofgenius-app",
                "health_url": "https://myroofgenius.com",
                "critical": True
            }
        }

        # Knowledge cache
        self._knowledge_cache: dict[str, KnowledgeEntry] = {}

        # Deployment history
        self._deployment_history: list[DeploymentRecord] = []

        logger.info(f"DevOpsAutomation initialized with {len(self.services)} services")

    async def trigger_deployment(
        self,
        service: str,
        version: Optional[str] = None,
        triggered_by: str = "automated"
    ) -> DeploymentRecord:
        """
        Trigger a deployment for a service.
        """
        if service not in self.services:
            raise ValueError(f"Unknown service: {service}")

        config = self.services[service]
        deployment_id = f"dep-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{service}"

        record = DeploymentRecord(
            deployment_id=deployment_id,
            service=service,
            service_type=config["type"],
            version=version or "latest",
            status=DeploymentStatus.IN_PROGRESS,
            triggered_by=triggered_by,
            started_at=datetime.now(timezone.utc).isoformat()
        )

        try:
            if config["type"] == ServiceType.RENDER:
                result = await self._deploy_render(config["render_id"])
                record.metadata["render_deploy_id"] = result.get("id")

            elif config["type"] == ServiceType.VERCEL:
                # Vercel deploys automatically on git push
                record.metadata["note"] = "Vercel deploys on git push"

            record.status = DeploymentStatus.SUCCESS
            record.completed_at = datetime.now(timezone.utc).isoformat()

            # Persist knowledge
            await self._persist_deployment_knowledge(record)

        except Exception as e:
            record.status = DeploymentStatus.FAILED
            record.error = str(e)
            record.completed_at = datetime.now(timezone.utc).isoformat()
            logger.error(f"Deployment failed for {service}: {e}")

            # Try automatic rollback for critical services
            if config.get("critical") and record.status == DeploymentStatus.FAILED:
                await self._attempt_rollback(service, record)

        self._deployment_history.append(record)
        return record

    async def _deploy_render(self, service_id: str) -> dict:
        """Trigger a Render deployment"""
        if not self.render_api_key:
            raise ValueError("RENDER_API_KEY not configured")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.render.com/v1/services/{service_id}/deploys",
                headers={
                    "Authorization": f"Bearer {self.render_api_key}",
                    "Content-Type": "application/json"
                },
                json={"clearCache": "clear"}
            ) as response:
                if response.status not in (200, 201):
                    text = await response.text()
                    raise Exception(f"Render deploy failed: {response.status} - {text}")

                return await response.json()

    async def _attempt_rollback(
        self,
        service: str,
        failed_record: DeploymentRecord
    ) -> Optional[DeploymentRecord]:
        """Attempt to rollback a failed deployment"""
        logger.warning(f"Attempting rollback for {service}")

        # Find last successful deployment
        last_success = None
        for record in reversed(self._deployment_history):
            if record.service == service and record.status == DeploymentStatus.SUCCESS:
                last_success = record
                break

        if not last_success:
            logger.error(f"No successful deployment found for {service} rollback")
            return None

        # Trigger rollback deployment
        try:
            rollback_record = await self.trigger_deployment(
                service=service,
                version=last_success.version,
                triggered_by="automatic_rollback"
            )
            rollback_record.rollback_to = last_success.version

            # Persist rollback knowledge
            await self._persist_knowledge(
                key=f"rollback_{service}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                value={
                    "service": service,
                    "failed_deployment": failed_record.deployment_id,
                    "rolled_back_to": last_success.version,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                category="incident"
            )

            return rollback_record

        except Exception as e:
            logger.error(f"Rollback failed for {service}: {e}")
            return None

    async def check_all_health(self) -> dict[str, dict]:
        """Check health of all services"""
        results = {}

        async with aiohttp.ClientSession() as session:
            for name, config in self.services.items():
                try:
                    async with session.get(
                        config["health_url"],
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        results[name] = {
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "http_code": response.status,
                            "type": config["type"].value
                        }
                except Exception as e:
                    results[name] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "type": config["type"].value
                    }

        return results

    async def auto_heal_service(self, service: str) -> dict:
        """Attempt to auto-heal an unhealthy service"""
        if service not in self.services:
            return {"success": False, "error": f"Unknown service: {service}"}

        config = self.services[service]

        # Step 1: Check current health
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    config["health_url"],
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return {"success": True, "action": "none_needed", "status": "already_healthy"}
            except Exception:
                pass

        # Step 2: Trigger redeploy
        try:
            deployment = await self.trigger_deployment(
                service=service,
                triggered_by="auto_heal"
            )

            # Step 3: Wait and verify
            await asyncio.sleep(60)  # Wait for deploy

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    config["health_url"],
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "action": "redeploy",
                            "deployment_id": deployment.deployment_id,
                            "status": "healed"
                        }

            return {
                "success": False,
                "action": "redeploy",
                "deployment_id": deployment.deployment_id,
                "status": "still_unhealthy"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _persist_knowledge(
        self,
        key: str,
        value: Any,
        category: str = "devops",
        expires_at: Optional[str] = None
    ) -> bool:
        """Persist knowledge to the brain"""
        try:
            from database.async_connection import get_pool, using_fallback

            if using_fallback():
                # Use in-memory cache
                self._knowledge_cache[key] = KnowledgeEntry(
                    key=key,
                    value=value,
                    category=category,
                    source="devops_automation",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    expires_at=expires_at
                )
                return True

            pool = get_pool()

            # Ensure table exists
            await pool.execute("""
                CREATE TABLE IF NOT EXISTS ai_devops_knowledge (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    key TEXT UNIQUE NOT NULL,
                    value JSONB NOT NULL,
                    category TEXT NOT NULL,
                    source TEXT DEFAULT 'devops_automation',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            await pool.execute("""
                INSERT INTO ai_devops_knowledge (key, value, category, source, expires_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    category = EXCLUDED.category,
                    updated_at = NOW()
            """, key, json.dumps(value), category, "devops_automation", expires_at)

            return True

        except Exception as e:
            logger.error(f"Failed to persist knowledge: {e}")
            return False

    async def _persist_deployment_knowledge(self, record: DeploymentRecord) -> None:
        """Persist deployment record to knowledge base"""
        await self._persist_knowledge(
            key=f"deployment_{record.deployment_id}",
            value={
                "deployment_id": record.deployment_id,
                "service": record.service,
                "service_type": record.service_type.value,
                "version": record.version,
                "status": record.status.value,
                "triggered_by": record.triggered_by,
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "error": record.error,
                "metadata": record.metadata
            },
            category="deployment"
        )

    async def get_knowledge(self, key: str) -> Optional[Any]:
        """Retrieve knowledge from the brain"""
        try:
            from database.async_connection import get_pool, using_fallback

            if using_fallback():
                entry = self._knowledge_cache.get(key)
                return entry.value if entry else None

            pool = get_pool()
            row = await pool.fetchrow(
                "SELECT value FROM ai_devops_knowledge WHERE key = $1",
                key
            )

            if row:
                return json.loads(row["value"]) if isinstance(row["value"], str) else row["value"]
            return None

        except Exception as e:
            logger.error(f"Failed to get knowledge: {e}")
            return None

    async def get_deployment_history(
        self,
        service: Optional[str] = None,
        limit: int = 20
    ) -> list[dict]:
        """Get deployment history"""
        try:
            from database.async_connection import get_pool, using_fallback

            if using_fallback():
                history = self._deployment_history
                if service:
                    history = [r for r in history if r.service == service]
                return [
                    {
                        "deployment_id": r.deployment_id,
                        "service": r.service,
                        "version": r.version,
                        "status": r.status.value,
                        "triggered_by": r.triggered_by,
                        "started_at": r.started_at,
                        "completed_at": r.completed_at
                    }
                    for r in history[-limit:]
                ]

            pool = get_pool()

            query = """
                SELECT key, value FROM ai_devops_knowledge
                WHERE category = 'deployment'
            """
            if service:
                query += f" AND value->>'service' = '{service}'"
            query += f" ORDER BY created_at DESC LIMIT {limit}"

            rows = await pool.fetch(query)
            return [
                json.loads(row["value"]) if isinstance(row["value"], str) else row["value"]
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get deployment history: {e}")
            return []

    async def learn_from_incident(
        self,
        service: str,
        incident_type: str,
        resolution: str,
        root_cause: Optional[str] = None
    ) -> bool:
        """Learn from an incident for future prevention"""
        learning = {
            "service": service,
            "incident_type": incident_type,
            "resolution": resolution,
            "root_cause": root_cause,
            "learned_at": datetime.now(timezone.utc).isoformat()
        }

        return await self._persist_knowledge(
            key=f"learning_{service}_{incident_type}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            value=learning,
            category="learning"
        )

    def get_stats(self) -> dict:
        """Get automation statistics"""
        total_deploys = len(self._deployment_history)
        successful = sum(1 for r in self._deployment_history if r.status == DeploymentStatus.SUCCESS)
        failed = sum(1 for r in self._deployment_history if r.status == DeploymentStatus.FAILED)

        return {
            "total_deployments": total_deploys,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total_deploys * 100) if total_deploys > 0 else 0,
            "services_managed": len(self.services),
            "knowledge_entries_cached": len(self._knowledge_cache)
        }


# Global instance
_devops_automation: Optional[DevOpsAutomation] = None


def get_devops_automation() -> DevOpsAutomation:
    """Get or create the global DevOps automation instance"""
    global _devops_automation
    if _devops_automation is None:
        _devops_automation = DevOpsAutomation()
    return _devops_automation


# Convenience functions
async def trigger_service_deploy(service: str, triggered_by: str = "automated") -> dict:
    """Trigger a deployment for a service"""
    automation = get_devops_automation()
    record = await automation.trigger_deployment(service, triggered_by=triggered_by)
    return {
        "deployment_id": record.deployment_id,
        "status": record.status.value,
        "service": record.service
    }


async def auto_heal(service: str) -> dict:
    """Auto-heal an unhealthy service"""
    automation = get_devops_automation()
    return await automation.auto_heal_service(service)


async def get_all_service_health() -> dict:
    """Get health of all services"""
    automation = get_devops_automation()
    return await automation.check_all_health()

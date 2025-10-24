#!/usr/bin/env python3

"""
BrainOps Deployment Monitor Agent
Real-time monitoring of Render and Vercel deployments with log access
"""

import os
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

class DeploymentStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    LIVE = "live"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class DeploymentMonitorAgent:
    """Agent that monitors deployments across Render and Vercel"""

    def __init__(self):
        self.render_api_key = os.getenv("RENDER_API_KEY")
        self.vercel_token = os.getenv("VERCEL_TOKEN", "vCDh2d4AgYXPAs0089MvQcHs")
        self.services = {
            "brainops-ai-agents": {
                "platform": "render",
                "url": "https://brainops-ai-agents.onrender.com",
                "service_id": None,  # Will be fetched
                "github_repo": "mwwoodworth/brainops-ai-agents"
            },
            "brainops-backend": {
                "platform": "render",
                "url": "https://brainops-backend-prod.onrender.com",
                "service_id": None,
                "docker_image": "mwwoodworth/brainops-backend"
            },
            "brainops-command-center": {
                "platform": "vercel",
                "url": "https://brainops-command-center.vercel.app",
                "project_id": None,  # Will be fetched
                "github_repo": "mwwoodworth/brainops-command-center"
            },
            "weathercraft-erp": {
                "platform": "vercel",
                "url": "https://weathercraft-erp.vercel.app",
                "project_id": None,
                "github_repo": "mwwoodworth/weathercraft-erp"
            },
            "myroofgenius-app": {
                "platform": "vercel",
                "url": "https://myroofgenius.com",
                "project_id": None,
                "github_repo": "mwwoodworth/myroofgenius-app"
            }
        }
        self.deployment_history = {}
        self.current_deployments = {}
        self.build_logs = {}

    async def fetch_render_services(self) -> List[Dict]:
        """Fetch all Render services"""
        if not self.render_api_key:
            return []

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.render_api_key}"}
            async with session.get(
                "https://api.render.com/v1/services",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("services", [])
                return []

    async def fetch_render_deploys(self, service_id: str) -> List[Dict]:
        """Fetch deployment history for a Render service"""
        if not self.render_api_key or not service_id:
            return []

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.render_api_key}"}
            async with session.get(
                f"https://api.render.com/v1/services/{service_id}/deploys",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("deploys", [])
                return []

    async def fetch_render_logs(self, service_id: str, deploy_id: str) -> str:
        """Fetch build logs for a Render deployment"""
        if not self.render_api_key or not service_id or not deploy_id:
            return ""

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.render_api_key}"}
            # Get build logs
            async with session.get(
                f"https://api.render.com/v1/services/{service_id}/deploys/{deploy_id}/logs",
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.text()
                return ""

    async def fetch_vercel_projects(self) -> List[Dict]:
        """Fetch all Vercel projects"""
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.vercel_token}"}
            async with session.get(
                "https://api.vercel.com/v9/projects",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("projects", [])
                return []

    async def fetch_vercel_deployments(self, project_id: str = None) -> List[Dict]:
        """Fetch Vercel deployments"""
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.vercel_token}"}
            url = "https://api.vercel.com/v6/deployments"
            if project_id:
                url += f"?projectId={project_id}"

            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("deployments", [])
                return []

    async def fetch_vercel_build_logs(self, deployment_id: str) -> Dict:
        """Fetch build logs for a Vercel deployment"""
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.vercel_token}"}

            # Get build info
            async with session.get(
                f"https://api.vercel.com/v13/deployments/{deployment_id}",
                headers=headers
            ) as response:
                if response.status == 200:
                    deployment = await response.json()

                    # Get actual build logs
                    async with session.get(
                        f"https://api.vercel.com/v2/deployments/{deployment_id}/events",
                        headers=headers
                    ) as log_response:
                        if log_response.status == 200:
                            logs = await log_response.json()
                            return {
                                "deployment": deployment,
                                "logs": logs
                            }
        return {}

    async def monitor_deployment(self, service_name: str) -> Dict[str, Any]:
        """Monitor a specific service deployment"""
        service = self.services.get(service_name)
        if not service:
            return {"error": f"Unknown service: {service_name}"}

        result = {
            "service": service_name,
            "platform": service["platform"],
            "url": service["url"],
            "status": DeploymentStatus.PENDING.value,
            "current_deployment": None,
            "recent_logs": None,
            "health_check": None
        }

        if service["platform"] == "render":
            # Monitor Render deployment
            if service.get("service_id"):
                deploys = await self.fetch_render_deploys(service["service_id"])
                if deploys:
                    latest = deploys[0]
                    result["current_deployment"] = {
                        "id": latest.get("id"),
                        "status": latest.get("status"),
                        "created_at": latest.get("createdAt"),
                        "updated_at": latest.get("updatedAt"),
                        "commit": latest.get("commit", {}).get("id")
                    }
                    result["status"] = self._map_render_status(latest.get("status"))

                    # Get logs if building or failed
                    if result["status"] in [DeploymentStatus.BUILDING.value, DeploymentStatus.FAILED.value]:
                        logs = await self.fetch_render_logs(service["service_id"], latest.get("id"))
                        result["recent_logs"] = logs[-2000:] if logs else None  # Last 2000 chars

        elif service["platform"] == "vercel":
            # Monitor Vercel deployment
            deployments = await self.fetch_vercel_deployments(service.get("project_id"))

            # Find deployments for this project
            project_deployments = [
                d for d in deployments
                if d.get("name") == service_name or
                   d.get("project") == service_name or
                   service["url"] in d.get("url", "")
            ]

            if project_deployments:
                latest = project_deployments[0]
                result["current_deployment"] = {
                    "id": latest.get("uid"),
                    "state": latest.get("state"),
                    "created_at": latest.get("created"),
                    "url": latest.get("url"),
                    "readyState": latest.get("readyState")
                }
                result["status"] = self._map_vercel_status(latest.get("readyState"))

                # Get build logs if error or building
                if result["status"] in [DeploymentStatus.BUILDING.value, DeploymentStatus.FAILED.value]:
                    log_data = await self.fetch_vercel_build_logs(latest.get("uid"))
                    if log_data.get("logs"):
                        result["recent_logs"] = log_data["logs"]

        # Perform health check
        result["health_check"] = await self.check_service_health(service["url"])

        return result

    def _map_render_status(self, render_status: str) -> str:
        """Map Render status to our DeploymentStatus"""
        status_map = {
            "created": DeploymentStatus.PENDING.value,
            "build_in_progress": DeploymentStatus.BUILDING.value,
            "update_in_progress": DeploymentStatus.DEPLOYING.value,
            "live": DeploymentStatus.LIVE.value,
            "deactivated": DeploymentStatus.ROLLED_BACK.value,
            "build_failed": DeploymentStatus.FAILED.value,
            "update_failed": DeploymentStatus.FAILED.value,
            "canceled": DeploymentStatus.FAILED.value
        }
        return status_map.get(render_status, DeploymentStatus.PENDING.value)

    def _map_vercel_status(self, vercel_state: str) -> str:
        """Map Vercel readyState to our DeploymentStatus"""
        state_map = {
            "QUEUED": DeploymentStatus.PENDING.value,
            "BUILDING": DeploymentStatus.BUILDING.value,
            "DEPLOYING": DeploymentStatus.DEPLOYING.value,
            "READY": DeploymentStatus.LIVE.value,
            "ERROR": DeploymentStatus.FAILED.value,
            "CANCELED": DeploymentStatus.FAILED.value
        }
        return state_map.get(vercel_state, DeploymentStatus.PENDING.value)

    async def check_service_health(self, url: str) -> Dict[str, Any]:
        """Check if a service is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "response_time_ms": response.headers.get("X-Response-Time", "N/A"),
                            "version": data.get("version", "unknown"),
                            "details": data
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "http_status": response.status
                        }
        except asyncio.TimeoutError:
            return {"status": "timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def monitor_all_deployments(self) -> Dict[str, Any]:
        """Monitor all configured services"""
        results = {}

        # First, try to populate service IDs if we have API keys
        if self.render_api_key:
            render_services = await self.fetch_render_services()
            for service in render_services:
                for name, config in self.services.items():
                    if config["platform"] == "render" and service.get("name") == name:
                        config["service_id"] = service.get("id")

        if self.vercel_token:
            vercel_projects = await self.fetch_vercel_projects()
            for project in vercel_projects:
                for name, config in self.services.items():
                    if config["platform"] == "vercel" and project.get("name") == name:
                        config["project_id"] = project.get("id")

        # Monitor each service
        for service_name in self.services:
            results[service_name] = await self.monitor_deployment(service_name)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "services": results,
            "summary": self._generate_summary(results)
        }

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate a summary of all deployments"""
        total = len(results)
        live = sum(1 for r in results.values() if r.get("status") == DeploymentStatus.LIVE.value)
        building = sum(1 for r in results.values() if r.get("status") == DeploymentStatus.BUILDING.value)
        failed = sum(1 for r in results.values() if r.get("status") == DeploymentStatus.FAILED.value)
        healthy = sum(1 for r in results.values() if r.get("health_check", {}).get("status") == "healthy")

        return {
            "total_services": total,
            "live": live,
            "building": building,
            "failed": failed,
            "healthy": healthy,
            "all_operational": live == total and healthy == total
        }

    async def get_error_logs(self, service_name: str, last_n_lines: int = 100) -> str:
        """Get recent error logs for a service"""
        service = self.services.get(service_name)
        if not service:
            return f"Unknown service: {service_name}"

        result = await self.monitor_deployment(service_name)

        if result.get("recent_logs"):
            logs = result["recent_logs"]
            # Filter for errors/warnings
            error_lines = []
            for line in logs.split('\n')[-last_n_lines:]:
                if any(keyword in line.lower() for keyword in ['error', 'fail', 'exception', 'warning', 'critical']):
                    error_lines.append(line)

            return '\n'.join(error_lines) if error_lines else "No errors found in recent logs"

        return "No logs available"

    def format_report(self, monitoring_data: Dict) -> str:
        """Format monitoring data as a readable report"""
        report = []
        report.append("=" * 60)
        report.append(f"DEPLOYMENT STATUS REPORT - {monitoring_data['timestamp']}")
        report.append("=" * 60)

        summary = monitoring_data['summary']
        report.append("\nSUMMARY:")
        report.append(f"  Total Services: {summary['total_services']}")
        report.append(f"  Live: {summary['live']} | Building: {summary['building']} | Failed: {summary['failed']}")
        report.append(f"  Healthy: {summary['healthy']}/{summary['total_services']}")

        if summary['all_operational']:
            report.append("\n✅ ALL SYSTEMS OPERATIONAL")
        else:
            report.append("\n⚠️ SOME SYSTEMS NEED ATTENTION")

        report.append("\nDETAILED STATUS:")
        report.append("-" * 60)

        for service_name, data in monitoring_data['services'].items():
            status_emoji = {
                "live": "✅",
                "building": "🔨",
                "deploying": "🚀",
                "failed": "❌",
                "pending": "⏳"
            }.get(data['status'], "❓")

            report.append(f"\n{service_name} ({data['platform'].upper()})")
            report.append(f"  Status: {status_emoji} {data['status'].upper()}")
            report.append(f"  URL: {data['url']}")

            if data.get('current_deployment'):
                dep = data['current_deployment']
                report.append(f"  Deployment: {dep.get('id', 'N/A')[:8]}...")
                report.append(f"  Created: {dep.get('created_at', 'N/A')}")

            if data.get('health_check'):
                health = data['health_check']
                health_emoji = "✅" if health['status'] == 'healthy' else "❌"
                report.append(f"  Health: {health_emoji} {health['status']}")
                if health.get('version'):
                    report.append(f"  Version: {health['version']}")

            if data['status'] == 'failed' and data.get('recent_logs'):
                report.append("  ⚠️ Check logs for errors")

        report.append("\n" + "=" * 60)
        return '\n'.join(report)


async def main():
    """Test the deployment monitor"""
    monitor = DeploymentMonitorAgent()

    print("Monitoring all deployments...")
    data = await monitor.monitor_all_deployments()

    print(monitor.format_report(data))

    # Check for specific service errors
    if not data['summary']['all_operational']:
        print("\n" + "=" * 60)
        print("CHECKING ERROR LOGS:")
        print("=" * 60)

        for service_name, service_data in data['services'].items():
            if service_data['status'] in ['failed', 'building']:
                print(f"\n{service_name} Recent Errors:")
                errors = await monitor.get_error_logs(service_name, 50)
                print(errors[:1000])  # First 1000 chars of errors


if __name__ == "__main__":
    asyncio.run(main())
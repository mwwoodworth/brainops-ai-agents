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

import logging

logger = logging.getLogger(__name__)


class DeploymentMonitorAgent:
    """Agent that monitors deployments across Render and Vercel with AUREA integration"""

    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.agent_type = "deployment_monitor"

        # AUREA Integration for decision recording and learning
        try:
            from aurea_integration import AUREAIntegration
            self.aurea = AUREAIntegration(tenant_id, self.agent_type)
        except ImportError:
            logger.warning("AUREA integration not available")
            self.aurea = None

        self.render_api_key = os.getenv("RENDER_API_KEY")
        self.vercel_token = os.getenv("VERCEL_TOKEN")
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
        if not self.vercel_token:
            return []

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

    async def calculate_deployment_risk(self, service_name: str, deployment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate deployment risk score based on multiple factors"""
        try:
            risk_factors = {
                "change_size": 0,  # Number of files/lines changed
                "test_coverage": 0,  # Test pass rate
                "recent_failures": 0,  # Recent deployment failures
                "time_of_day": 0,  # Deploying during business hours is riskier
                "dependencies": 0,  # Number of dependent services
                "rollback_time": 0,  # Time to rollback if needed
                "monitoring_coverage": 0,  # Quality of monitoring
            }

            # Analyze change size
            files_changed = deployment_details.get("files_changed", [])
            risk_factors["change_size"] = min(10, len(files_changed) / 2)  # 0-10 scale

            # Check test coverage
            test_results = deployment_details.get("test_results", {})
            if test_results:
                passed = test_results.get("passed", 0)
                total = test_results.get("total", 1)
                test_pass_rate = passed / total if total > 0 else 0
                risk_factors["test_coverage"] = (1 - test_pass_rate) * 10

            # Check recent deployment history
            service = self.services.get(service_name)
            if service:
                # Time of day risk (higher during business hours)
                current_hour = datetime.utcnow().hour
                if 9 <= current_hour <= 17:  # Business hours UTC
                    risk_factors["time_of_day"] = 5
                elif 6 <= current_hour <= 21:
                    risk_factors["time_of_day"] = 3

                # Dependency risk
                risk_factors["dependencies"] = min(10, len(service.get("dependencies", [])) * 2)

            # Calculate total risk score (0-100)
            total_risk = sum(risk_factors.values())
            risk_score = min(100, int(total_risk * 1.5))

            # Determine risk level
            if risk_score >= 70:
                risk_level = "critical"
                recommendation = "Deployment not recommended - too many risk factors"
            elif risk_score >= 50:
                risk_level = "high"
                recommendation = "Deploy with caution - consider deploying during low-traffic hours"
            elif risk_score >= 30:
                risk_level = "medium"
                recommendation = "Acceptable risk - ensure monitoring is active"
            else:
                risk_level = "low"
                recommendation = "Low risk deployment - proceed with standard procedures"

            return {
                "service": service_name,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendation": recommendation,
                "calculated_at": datetime.utcnow().isoformat(),
                "should_proceed": risk_score < 70,
                "mitigation_steps": self._generate_mitigation_steps(risk_factors)
            }

        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return {
                "error": str(e),
                "risk_score": 100,
                "risk_level": "unknown",
                "should_proceed": False
            }

    def _generate_mitigation_steps(self, risk_factors: Dict[str, float]) -> List[str]:
        """Generate mitigation steps based on risk factors"""
        steps = []

        if risk_factors["change_size"] > 5:
            steps.append("Consider breaking deployment into smaller chunks")

        if risk_factors["test_coverage"] > 5:
            steps.append("Improve test coverage before deploying")

        if risk_factors["time_of_day"] > 3:
            steps.append("Consider deploying during off-peak hours")

        if risk_factors["dependencies"] > 5:
            steps.append("Verify all dependent services are healthy")

        if risk_factors["recent_failures"] > 3:
            steps.append("Review recent failure patterns before proceeding")

        steps.append("Ensure rollback procedure is documented and tested")
        steps.append("Have monitoring dashboards ready")

        return steps

    async def auto_rollback_decision(self, service_name: str, deployment_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if automatic rollback should be triggered"""
        try:
            rollback_triggers = {
                "error_rate_spike": False,
                "response_time_degradation": False,
                "health_check_failures": False,
                "traffic_drop": False,
                "cpu_spike": False,
                "memory_spike": False
            }

            # Check error rate
            error_rate = metrics.get("error_rate", 0)
            baseline_error_rate = metrics.get("baseline_error_rate", 0)
            if error_rate > baseline_error_rate * 3:  # 3x increase
                rollback_triggers["error_rate_spike"] = True

            # Check response time
            response_time = metrics.get("avg_response_time_ms", 0)
            baseline_response_time = metrics.get("baseline_response_time_ms", 0)
            if response_time > baseline_response_time * 2:  # 2x slower
                rollback_triggers["response_time_degradation"] = True

            # Check health status
            health_failures = metrics.get("health_check_failures", 0)
            if health_failures >= 3:  # 3 consecutive failures
                rollback_triggers["health_check_failures"] = True

            # Check traffic
            current_traffic = metrics.get("requests_per_minute", 0)
            baseline_traffic = metrics.get("baseline_requests_per_minute", 1)
            if current_traffic < baseline_traffic * 0.5:  # 50% drop
                rollback_triggers["traffic_drop"] = True

            # Check CPU
            cpu_usage = metrics.get("cpu_usage_percent", 0)
            if cpu_usage > 90:
                rollback_triggers["cpu_spike"] = True

            # Check memory
            memory_usage = metrics.get("memory_usage_percent", 0)
            if memory_usage > 90:
                rollback_triggers["memory_spike"] = True

            # Determine if rollback should occur
            triggered_count = sum(1 for v in rollback_triggers.values() if v)
            should_rollback = triggered_count >= 2  # At least 2 triggers

            severity = "critical" if triggered_count >= 3 else "high" if triggered_count >= 2 else "medium" if triggered_count >= 1 else "low"

            return {
                "service": service_name,
                "deployment_id": deployment_id,
                "should_rollback": should_rollback,
                "severity": severity,
                "triggers": rollback_triggers,
                "triggered_count": triggered_count,
                "reason": self._generate_rollback_reason(rollback_triggers),
                "evaluated_at": datetime.utcnow().isoformat(),
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Rollback decision failed: {e}")
            return {
                "error": str(e),
                "should_rollback": True,  # Fail-safe: rollback on errors
                "severity": "critical"
            }

    def _generate_rollback_reason(self, triggers: Dict[str, bool]) -> str:
        """Generate human-readable rollback reason"""
        active_triggers = [k.replace('_', ' ').title() for k, v in triggers.items() if v]

        if not active_triggers:
            return "No rollback triggers detected"

        if len(active_triggers) == 1:
            return f"Rollback triggered by: {active_triggers[0]}"

        return f"Multiple issues detected: {', '.join(active_triggers)}"

    async def monitor_deployment_metrics(self, service_name: str, duration_minutes: int = 10) -> Dict[str, Any]:
        """Monitor deployment metrics for a period after deployment"""
        try:
            service = self.services.get(service_name)
            if not service:
                return {"error": f"Unknown service: {service_name}"}

            metrics_history = []
            start_time = datetime.utcnow()

            # In a real implementation, this would poll metrics over time
            # For now, we'll simulate with a single check
            health_check = await self.check_service_health(service["url"])

            metrics = {
                "service": service_name,
                "monitoring_started": start_time.isoformat(),
                "duration_minutes": duration_minutes,
                "health_status": health_check.get("status"),
                "error_rate": 0,  # Would come from APM/logging
                "avg_response_time_ms": 0,  # Would come from metrics
                "requests_per_minute": 0,  # Would come from metrics
                "cpu_usage_percent": 0,  # Would come from platform API
                "memory_usage_percent": 0,  # Would come from platform API
                "baseline_error_rate": 0.5,
                "baseline_response_time_ms": 200,
                "baseline_requests_per_minute": 100
            }

            # Check if rollback is needed
            rollback_decision = await self.auto_rollback_decision(service_name, "current", metrics)

            return {
                "metrics": metrics,
                "rollback_decision": rollback_decision,
                "status": "healthy" if not rollback_decision["should_rollback"] else "unhealthy"
            }

        except Exception as e:
            logger.error(f"Metrics monitoring failed: {e}")
            return {"error": str(e)}

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

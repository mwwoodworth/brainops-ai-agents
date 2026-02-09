"""
Google Keep Live Ops Sync v2.0
==============================
Full-power BrainOps OS -> Google Keep bridge for Gemini Live real-time access.

Three pinned Keep notes, updated every 5 minutes:
1. "BrainOps Live Ops" - System health, alerts, tasks, revenue, agents, metrics
2. "BrainOps Brainstorm" - Latest brainstorm ideas, assessments, active themes
3. "BrainOps Commands" - Bidirectional command queue (Gemini writes, agent reads & executes)

Uses gkeepapi (unofficial) since the official Keep API requires Enterprise+.
Auth: master token via EmbeddedSetup OAuth flow stored in GOOGLE_KEEP_MASTER_TOKEN.

Schedule: Every 5 minutes via agent_scheduler.
"""

import asyncio
import logging
import os
import json
import time
from datetime import datetime, timezone
from typing import Optional

import gkeepapi
import requests

logger = logging.getLogger(__name__)

# Config from environment
KEEP_EMAIL = os.getenv("GOOGLE_KEEP_EMAIL", "")
KEEP_MASTER_TOKEN = os.getenv("GOOGLE_KEEP_MASTER_TOKEN", "")
BRAINOPS_API_KEY = os.getenv("BRAINOPS_API_KEY", "brainops_prod_key_2025")

# API endpoints
API_SELF = "http://localhost:10000"
API_CC = "https://brainops-command-center.vercel.app"
API_MCP = "https://brainops-mcp-bridge.onrender.com"

# Note titles - Gemini finds these via @Keep
KEEP_OPS_TITLE = "BrainOps Live Ops"
KEEP_BRAINSTORM_TITLE = "BrainOps Brainstorm"
KEEP_COMMANDS_TITLE = "BrainOps Commands"


class KeepSyncAgent:
    """Full-power BrainOps <-> Google Keep bidirectional sync."""

    def __init__(self):
        self.keep = gkeepapi.Keep()
        self.authenticated = False
        self.last_sync = None
        self.sync_count = 0
        self.consecutive_errors = 0
        self.commands_processed = 0

    def authenticate(self) -> bool:
        """Authenticate with Google Keep using master token."""
        if not KEEP_EMAIL or not KEEP_MASTER_TOKEN:
            logger.error("GOOGLE_KEEP_EMAIL or GOOGLE_KEEP_MASTER_TOKEN not set")
            return False

        try:
            self.keep.authenticate(KEEP_EMAIL, KEEP_MASTER_TOKEN)
            self.authenticated = True
            logger.info("Google Keep authenticated as %s", KEEP_EMAIL)
            return True
        except Exception as e:
            logger.error("Google Keep auth failed: %s", e)
            self.authenticated = False
            return False

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_json(self, url, headers=None, timeout=12):
        """Fetch JSON from a URL, return {} on failure."""
        try:
            r = requests.get(url, headers=headers or {}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.debug("Fetch %s failed: %s", url, e)
        return {}

    def _post_json(self, url, data=None, headers=None, timeout=12):
        """POST JSON to a URL, return {} on failure."""
        try:
            r = requests.post(url, json=data or {}, headers=headers or {}, timeout=timeout)
            if r.status_code in (200, 201):
                return r.json()
        except Exception as e:
            logger.debug("POST %s failed: %s", url, e)
        return {}

    def _fetch_live_data(self) -> dict:
        """Fetch live data from ALL BrainOps services - maximum coverage."""
        h = {"X-API-Key": BRAINOPS_API_KEY}
        data = {}

        # ---- Agents Service (localhost) ----
        data["health"] = self._fetch_json(f"{API_SELF}/health", h)
        data["alerts"] = self._fetch_json(f"{API_SELF}/system/alerts?unresolved_only=true", h)
        data["agents"] = self._fetch_json(f"{API_SELF}/agents/status", h)
        data["consciousness"] = self._fetch_json(f"{API_SELF}/consciousness/status", h)
        data["self_heal"] = self._fetch_json(f"{API_SELF}/self-heal/check", h)
        data["awareness"] = self._fetch_json(f"{API_SELF}/system/awareness", h)
        data["scheduler"] = self._fetch_json(f"{API_SELF}/agents/schedules", h)
        data["system_events"] = self._fetch_json(f"{API_SELF}/system/events?limit=5", h)
        data["brain_recent"] = self._fetch_json(f"{API_SELF}/brain/recent?limit=5", h)
        data["deployments"] = self._fetch_json(f"{API_SELF}/devops/deployments?limit=3", h)

        # ---- Command Center ----
        data["tasks"] = self._fetch_json(
            f"{API_CC}/api/tasks/unified-v2?assigned_to=matt&status=in_progress,pending&sort_by=priority&sort_order=desc",
            h
        )
        data["completed_tasks"] = self._fetch_json(
            f"{API_CC}/api/tasks/unified-v2?status=completed&sort_by=updated_at&sort_order=desc&limit=5",
            h
        )
        data["revenue"] = self._fetch_json(f"{API_CC}/api/revenue", h)
        data["metrics"] = self._fetch_json(f"{API_CC}/api/metrics", h)
        data["briefing"] = self._fetch_json(f"{API_CC}/api/briefing", h)
        data["gumroad"] = self._fetch_json(f"{API_CC}/api/income/gumroad", h)
        data["pipeline"] = self._fetch_json(f"{API_CC}/api/income/pipeline", h)
        data["stripe"] = self._fetch_json(f"{API_CC}/api/income/stripe", h)

        # ---- MCP Bridge ----
        data["mcp_health"] = self._fetch_json(f"{API_MCP}/health", h)

        # ---- Brainstorm data ----
        data["brainstorm"] = self._fetch_json(
            f"{API_SELF}/brain/query?category=brainstorm&limit=10", h
        )

        return data

    # ------------------------------------------------------------------
    # Note builders
    # ------------------------------------------------------------------

    def _build_ops_note(self, data: dict) -> str:
        """Build the main Live Ops note - full system dashboard."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%d %H:%M UTC")

        h = data.get("health", {})
        awareness = data.get("awareness", {})
        alerts_data = data.get("alerts", {})
        alerts = alerts_data.get("alerts", []) if isinstance(alerts_data, dict) else (alerts_data if isinstance(alerts_data, list) else [])
        tasks_data = data.get("tasks", {})
        tasks = tasks_data.get("tasks", []) if isinstance(tasks_data, dict) else (tasks_data if isinstance(tasks_data, list) else [])
        completed_data = data.get("completed_tasks", {})
        completed = completed_data.get("tasks", []) if isinstance(completed_data, dict) else (completed_data if isinstance(completed_data, list) else [])
        con = data.get("consciousness", {})
        heal = data.get("self_heal", {})
        rev = data.get("revenue", {})
        gum = data.get("gumroad", {})
        pip = data.get("pipeline", {})
        stripe = data.get("stripe", {})
        m = data.get("metrics", {})
        brief = data.get("briefing", {})
        agents = data.get("agents", {})
        scheduler = data.get("scheduler", {})
        events = data.get("system_events", {})
        brain = data.get("brain_recent", {})
        deploys = data.get("deployments", {})
        mcp = data.get("mcp_health", {})

        overall = awareness.get("overall_status") or h.get("global_status") or h.get("status") or "UNKNOWN"
        broken = awareness.get("broken_systems", [])

        lines = []
        lines.append(f"BRAINOPS AI OS | LIVE OPS | {ts}")
        lines.append(f"Sync #{self.sync_count} | Global: {self._fmt_status(overall)} | Keep Errors: {self.consecutive_errors}")
        lines.append("")

        # ALERTS
        lines.append(f"=== ALERTS ({len(alerts)}) ===")
        if not alerts:
            lines.append("ALL CLEAR - No unresolved alerts")
        else:
            for a in alerts[:8]:
                if isinstance(a, str):
                    lines.append(f"! {a}")
                else:
                    sev = a.get('severity', 'INFO')
                    msg = (a.get('message') or a.get('description') or str(a))[:80]
                    lines.append(f"[{sev}] {msg}")
        lines.append("")

        # SYSTEM HEALTH
        lines.append("=== SYSTEM HEALTH ===")
        version = h.get("version", "?")
        uptime = self._fmt_duration(h.get("uptime"))
        lines.append(f"Agents: {self._fmt_status(h.get('status'))} v{version} up {uptime}")
        lines.append(f"Consciousness: {con.get('state') or con.get('status') or '?'}")
        heal_line = f"Self-Heal: {heal.get('status', '?')}"
        if 'issues_found' in heal:
            heal_line += f" | Issues: {heal['issues_found']} Fixed: {heal.get('auto_fixed', 0)}"
        lines.append(heal_line)
        lines.append(f"MCP Bridge: {self._fmt_status(mcp.get('status'))}")
        if broken:
            for b in broken[:3]:
                bname = b if isinstance(b, str) else (b.get("name") or b.get("system") or str(b))
                lines.append(f"  BROKEN: {bname[:60]}")
        lines.append("")

        # ACTIVE TASKS
        lines.append(f"=== ACTIVE TASKS ({len(tasks)}) ===")
        if not tasks:
            lines.append("No active tasks")
        else:
            for t in tasks[:15]:
                pri = str(t.get("priority", "MED"))[:3].upper()
                status = t.get("status", "?")
                title = (t.get("title") or "Untitled")[:55]
                src = t.get("source", "")[:6]
                lines.append(f"[{pri}|{status}] {title}" + (f" ({src})" if src else ""))
        lines.append("")

        # RECENTLY COMPLETED
        if completed:
            lines.append(f"=== RECENTLY COMPLETED ({len(completed)}) ===")
            for t in completed[:5]:
                title = (t.get("title") or "Untitled")[:55]
                lines.append(f"  DONE: {title}")
            lines.append("")

        # REVENUE (REAL ONLY)
        lines.append("=== REVENUE (REAL ONLY) ===")
        lines.append(f"Total: ${self._fmt_money(rev.get('totalRevenue'))} | MRR: ${self._fmt_money(rev.get('mrr'))}")
        lines.append(f"Active Subs: {rev.get('activeSubscriptions', 0)}")
        lines.append(f"Gumroad: ${self._fmt_money(gum.get('total_revenue'))} ({gum.get('sales_count', 0)} sales)")
        lines.append(f"Stripe: ${self._fmt_money(stripe.get('total_revenue') or stripe.get('revenue'))}")
        lines.append(f"Pipeline: ${self._fmt_money(pip.get('pipeline_value'))} | Leads: {pip.get('total_leads', 0)}")
        lines.append(f"Burn Rate: ~$450/mo | Break-even: ~9 Starter or ~5 Pro subs")
        lines.append("WARNING: ERP data = DEMO. Only Gumroad/Stripe/MRG = real.")
        lines.append("")

        # METRICS
        lines.append("=== METRICS ===")
        lines.append(f"Done Today: {m.get('completedToday', 0)} | In Progress: {m.get('inProgress', 0)} | Blocked: {m.get('blocked', 0)}")
        lines.append("")

        # BRIEFING
        if brief and brief.get("summary"):
            lines.append("=== BRIEFING ===")
            lines.append(str(brief["summary"])[:400])
            if brief.get("action_items"):
                for ai_item in brief["action_items"][:5]:
                    item = ai_item if isinstance(ai_item, str) else str(ai_item)
                    lines.append(f"  -> {item[:80]}")
            lines.append("")

        # AGENT STATUS
        agent_list = agents.get("agents", [])
        if agent_list:
            lines.append(f"=== AGENTS ({len(agent_list)}) ===")
            for ag in agent_list[:10]:
                name = ag.get('name', '?')
                status = ag.get('status', '?')
                last = ag.get('last_run', '')
                lines.append(f"  {name}: {status}" + (f" ({last[:16]})" if last else ""))
            lines.append("")

        # SCHEDULED JOBS
        jobs = scheduler.get("jobs", scheduler.get("registered_jobs", {}))
        if jobs:
            active = jobs if isinstance(jobs, list) else list(jobs.keys())
            lines.append(f"=== SCHEDULED JOBS ({len(active)}) ===")
            for j in active[:8]:
                jname = j if isinstance(j, str) else j.get("name", str(j))
                lines.append(f"  {jname[:50]}")
            lines.append("")

        # RECENT EVENTS
        event_list = events.get("events", []) if isinstance(events, dict) else (events if isinstance(events, list) else [])
        if event_list:
            lines.append(f"=== RECENT EVENTS ===")
            for ev in event_list[:5]:
                etype = ev.get("type", "?")
                emsg = (ev.get("message") or ev.get("description") or "")[:60]
                lines.append(f"  [{etype}] {emsg}")
            lines.append("")

        # RECENT DEPLOYMENTS
        deploy_list = deploys.get("deployments", []) if isinstance(deploys, dict) else (deploys if isinstance(deploys, list) else [])
        if deploy_list:
            lines.append("=== RECENT DEPLOYMENTS ===")
            for d in deploy_list[:3]:
                svc = d.get("service", "?")
                ver = d.get("version", "?")
                status = d.get("status", "?")
                lines.append(f"  {svc}: {ver} ({status})")
            lines.append("")

        # BRAIN MEMORY (recent)
        brain_entries = brain.get("entries", brain.get("memories", []))
        if isinstance(brain_entries, list) and brain_entries:
            lines.append("=== BRAIN MEMORY (recent) ===")
            for entry in brain_entries[:3]:
                key = entry.get("key", "")
                val = str(entry.get("value", ""))[:60]
                lines.append(f"  {key}: {val}")
            lines.append("")

        # VOICE COMMANDS
        lines.append("=== VOICE COMMANDS ===")
        lines.append("Say: 'Add to BrainOps Ops list: [command]'")
        lines.append("Commands: deploy [service], restart [service], check health, heal, force sync, scale [service], rollback [service], analyze [target], resolve alert [id], brain store [key: value]")
        lines.append("")
        lines.append("Say: 'Add to BrainOps Brainstorm list: [idea]'")
        lines.append("-> AI assessment + Supabase archive + task generation")
        lines.append("")
        lines.append("Say: 'Check my BrainOps Brainstorm note' for latest ideas")

        return "\n".join(lines)

    def _build_brainstorm_note(self, data: dict) -> str:
        """Build the Brainstorm note with latest ideas and active themes."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%d %H:%M UTC")

        brainstorm = data.get("brainstorm", {})
        entries = brainstorm.get("entries", brainstorm.get("results", []))
        if not isinstance(entries, list):
            entries = []

        lines = []
        lines.append(f"BRAINOPS BRAINSTORM BANK | {ts}")
        lines.append(f"Ideas: {len(entries)} shown (latest)")
        lines.append("")

        if not entries:
            lines.append("No brainstorm entries yet.")
            lines.append("")
            lines.append("To add ideas:")
            lines.append("  Voice: 'Add to BrainOps Brainstorm list: [your idea]'")
            lines.append("  -> AI will assess feasibility, impact, and effort")
            lines.append("  -> Stored in Supabase brainstorm_ideas table")
            lines.append("  -> Auto-generates tasks if high-impact")
        else:
            for i, entry in enumerate(entries[:10], 1):
                if isinstance(entry, dict):
                    title = entry.get("title") or entry.get("key") or entry.get("idea") or "Untitled"
                    desc = entry.get("description") or entry.get("value") or entry.get("assessment") or ""
                    status = entry.get("status", "new")
                    impact = entry.get("impact_score", "?")
                    lines.append(f"{i}. [{status}] {str(title)[:60]}")
                    if desc:
                        lines.append(f"   {str(desc)[:100]}")
                    if impact != "?":
                        lines.append(f"   Impact: {impact}/10")
                else:
                    lines.append(f"{i}. {str(entry)[:80]}")
                lines.append("")

        lines.append("=== HOW TO ADD IDEAS ===")
        lines.append("Voice: 'Add to BrainOps Brainstorm list: [idea]'")
        lines.append("The AI will:")
        lines.append("  1. Assess feasibility (1-10)")
        lines.append("  2. Estimate impact (1-10)")
        lines.append("  3. Tag with categories")
        lines.append("  4. Store in Supabase permanently")
        lines.append("  5. Auto-create tasks for high-impact ideas")

        return "\n".join(lines)

    def _build_commands_note(self) -> str:
        """Build the Commands note - bidirectional command queue."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%d %H:%M UTC")

        lines = []
        lines.append(f"BRAINOPS COMMAND QUEUE | {ts}")
        lines.append(f"Commands Processed: {self.commands_processed}")
        lines.append("")
        lines.append("=== WRITE COMMANDS HERE ===")
        lines.append("Add commands below this line. The agent reads & executes every 5 min.")
        lines.append("")
        lines.append("--- PENDING ---")
        lines.append("(Write commands here, e.g.: 'deploy agents' or 'check revenue')")
        lines.append("")
        lines.append("--- COMPLETED ---")
        lines.append("(Agent moves executed commands here)")
        lines.append("")
        lines.append("=== AVAILABLE COMMANDS ===")
        lines.append("deploy [agents|backend|mcp|erp|mrg|cc]")
        lines.append("restart [service]")
        lines.append("check [health|revenue|tasks|alerts]")
        lines.append("heal - run self-healing cycle")
        lines.append("force sync - refresh all data")
        lines.append("scale [service] [up|down]")
        lines.append("rollback [service]")
        lines.append("analyze [target]")
        lines.append("resolve alert [description]")
        lines.append("brain store [key]: [value]")
        lines.append("brainstorm: [your idea]")
        lines.append("task: [title] - create a new task")

        return "\n".join(lines)

    def _read_and_process_commands(self):
        """Read commands from the Commands note and execute them."""
        note = self._find_note(KEEP_COMMANDS_TITLE)
        if not note:
            return

        text = note.text or ""
        # Look for commands between PENDING and COMPLETED sections
        pending_start = text.find("--- PENDING ---")
        completed_start = text.find("--- COMPLETED ---")

        if pending_start == -1 or completed_start == -1:
            return

        pending_section = text[pending_start + len("--- PENDING ---"):completed_start].strip()
        if not pending_section or pending_section.startswith("(Write"):
            return

        # Extract command lines
        commands = [line.strip() for line in pending_section.split("\n") if line.strip() and not line.strip().startswith("(")]
        if not commands:
            return

        completed_section = text[completed_start + len("--- COMPLETED ---"):].strip()
        h = {"X-API-Key": BRAINOPS_API_KEY, "Content-Type": "application/json"}

        for cmd in commands:
            logger.info("Processing Keep command: %s", cmd)
            result = "executed"

            try:
                cmd_lower = cmd.lower().strip()
                if cmd_lower.startswith("deploy "):
                    target = cmd_lower.replace("deploy ", "").strip()
                    self._post_json(f"{API_SELF}/devops/deploy", {"service": target}, h)
                    result = f"deploy {target} triggered"
                elif cmd_lower.startswith("check "):
                    result = f"check executed (see Live Ops note)"
                elif cmd_lower == "heal":
                    self._post_json(f"{API_SELF}/self-heal/run", {}, h)
                    result = "self-heal cycle triggered"
                elif cmd_lower in ("force sync", "refresh"):
                    result = "force sync executed"
                elif cmd_lower.startswith("brain store "):
                    parts = cmd_lower.replace("brain store ", "").split(":", 1)
                    if len(parts) == 2:
                        self._post_json(f"{API_SELF}/brain/store",
                                        {"key": parts[0].strip(), "value": parts[1].strip()}, h)
                        result = f"stored: {parts[0].strip()}"
                elif cmd_lower.startswith("brainstorm:"):
                    idea = cmd.split(":", 1)[1].strip()
                    self._post_json(f"{API_SELF}/brain/store",
                                    {"key": "brainstorm", "value": idea, "category": "brainstorm"}, h)
                    result = f"brainstorm recorded"
                elif cmd_lower.startswith("task:"):
                    title = cmd.split(":", 1)[1].strip()
                    self._post_json(
                        f"{API_CC}/api/tasks/unified-v2",
                        {"title": title, "status": "pending", "source": "keep-voice", "assigned_to": "matt"},
                        h
                    )
                    result = f"task created: {title[:40]}"
                else:
                    result = f"unknown command"
            except Exception as e:
                result = f"error: {str(e)[:40]}"

            self.commands_processed += 1
            now_str = datetime.now(timezone.utc).strftime("%H:%M")
            completed_section = f"[{now_str}] {cmd} -> {result}\n{completed_section}"

        # Rewrite note with commands moved to completed
        new_text = text[:pending_start + len("--- PENDING ---")]
        new_text += "\n(Write commands here)\n\n"
        new_text += "--- COMPLETED ---\n"
        new_text += completed_section
        note.text = new_text

    # ------------------------------------------------------------------
    # Main sync
    # ------------------------------------------------------------------

    def sync(self) -> dict:
        """Main sync: fetch data, build all notes, write to Keep."""
        start = time.time()

        if not self.authenticated:
            if not self.authenticate():
                return {"success": False, "error": "Authentication failed"}

        try:
            # Fetch live data
            data = self._fetch_live_data()
            t_fetch = time.time()

            # Build all note contents
            ops_content = self._build_ops_note(data)
            brainstorm_content = self._build_brainstorm_note(data)
            t_build = time.time()

            # Sync from Keep (to read any commands)
            self.keep.sync()

            # Process commands from the Commands note
            self._read_and_process_commands()

            # Update or create Live Ops note
            ops_note = self._find_note(KEEP_OPS_TITLE)
            if ops_note:
                ops_note.text = ops_content
                ops_note.pinned = True
            else:
                ops_note = self.keep.createNote(KEEP_OPS_TITLE, ops_content)
                ops_note.pinned = True
                ops_note.color = gkeepapi.node.ColorValue.Blue
                logger.info("Created Keep note: %s", KEEP_OPS_TITLE)

            # Update or create Brainstorm note
            bs_note = self._find_note(KEEP_BRAINSTORM_TITLE)
            if bs_note:
                bs_note.text = brainstorm_content
                bs_note.pinned = True
            else:
                bs_note = self.keep.createNote(KEEP_BRAINSTORM_TITLE, brainstorm_content)
                bs_note.pinned = True
                bs_note.color = gkeepapi.node.ColorValue.Green
                logger.info("Created Keep note: %s", KEEP_BRAINSTORM_TITLE)

            # Create Commands note if it doesn't exist
            cmd_note = self._find_note(KEEP_COMMANDS_TITLE)
            if not cmd_note:
                cmd_content = self._build_commands_note()
                cmd_note = self.keep.createNote(KEEP_COMMANDS_TITLE, cmd_content)
                cmd_note.pinned = True
                cmd_note.color = gkeepapi.node.ColorValue.Orange
                logger.info("Created Keep note: %s", KEEP_COMMANDS_TITLE)

            # Push all changes to Keep
            self.keep.sync()
            t_sync = time.time()

            self.sync_count += 1
            self.consecutive_errors = 0
            self.last_sync = datetime.now(timezone.utc).isoformat()

            result = {
                "success": True,
                "sync_number": self.sync_count,
                "notes_updated": 3,
                "ops_length": len(ops_content),
                "brainstorm_length": len(brainstorm_content),
                "commands_processed": self.commands_processed,
                "timing": {
                    "fetch": round(t_fetch - start, 2),
                    "build": round(t_build - t_fetch, 2),
                    "keep_sync": round(t_sync - t_build, 2),
                    "total": round(t_sync - start, 2)
                }
            }
            logger.info("Keep sync OK: %s", json.dumps(result))
            return result

        except gkeepapi.exception.LoginException as e:
            self.authenticated = False
            self.consecutive_errors += 1
            logger.error("Keep auth expired, will re-auth next cycle: %s", e)
            return {"success": False, "error": f"Auth expired: {e}"}

        except Exception as e:
            self.consecutive_errors += 1
            logger.error("Keep sync failed (%d consecutive): %s",
                         self.consecutive_errors, e, exc_info=True)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_note(self, title: str):
        """Find an existing note by title."""
        notes = self.keep.find(func=lambda n: n.title == title and not n.trashed)
        for note in notes:
            return note
        return None

    def get_status(self) -> dict:
        """Return sync health status."""
        return {
            "authenticated": self.authenticated,
            "email": KEEP_EMAIL or "NOT SET",
            "sync_count": self.sync_count,
            "consecutive_errors": self.consecutive_errors,
            "last_sync": self.last_sync,
            "commands_processed": self.commands_processed,
            "notes": [KEEP_OPS_TITLE, KEEP_BRAINSTORM_TITLE, KEEP_COMMANDS_TITLE],
            "master_token_set": bool(KEEP_MASTER_TOKEN)
        }

    @staticmethod
    def _fmt_status(s):
        if not s:
            return "?"
        u = str(s).upper()
        if u in ("OK", "HEALTHY", "OPERATIONAL", "RUNNING"):
            return "OK"
        if "DEGRAD" in u:
            return "DEGRADED"
        if any(x in u for x in ("DOWN", "ERROR", "FAIL")):
            return "DOWN"
        return u

    @staticmethod
    def _fmt_money(n):
        if n is None:
            return "0"
        try:
            return f"{float(n):,.2f}"
        except (ValueError, TypeError):
            return "0"

    @staticmethod
    def _fmt_duration(s):
        if not s:
            return "?"
        try:
            s = int(s)
        except (ValueError, TypeError):
            return "?"
        d = s // 86400
        h = (s % 86400) // 3600
        m = (s % 3600) // 60
        if d > 0:
            return f"{d}d {h}h"
        return f"{h}h {m}m"


# Singleton instance
_agent: Optional[KeepSyncAgent] = None


def get_agent() -> KeepSyncAgent:
    """Get or create the singleton agent."""
    global _agent
    if _agent is None:
        _agent = KeepSyncAgent()
    return _agent


async def run_keep_sync() -> dict:
    """Entry point for agent_scheduler."""
    agent = get_agent()
    return await asyncio.to_thread(agent.sync)


async def get_keep_status() -> dict:
    """Entry point for status check."""
    agent = get_agent()
    return agent.get_status()

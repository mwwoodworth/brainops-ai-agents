import asyncio
import difflib
import inspect
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

from safe_task import create_safe_task
import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from google.generativeai import GenerativeModel, configure

# Persona/profile resolution for scoped capability gating
try:
    from aurea_personas import build_execution_profile
except Exception:
    build_execution_profile = None

# Import Power Layer for full operational capability
try:
    from aurea_power_layer import get_power_layer
    POWER_LAYER_AVAILABLE = True
except ImportError:
    POWER_LAYER_AVAILABLE = False
    get_power_layer = None

logger = logging.getLogger(__name__)

# Perplexity fallback for when OpenAI quota is exceeded
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _env_flag(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


ENABLE_AUREA_COMPLEX_COMMAND_NLU = _env_flag("ENABLE_AUREA_COMPLEX_COMMAND_NLU", "false")
ENABLE_AUREA_COMMAND_SUGGESTIONS = _env_flag("ENABLE_AUREA_COMMAND_SUGGESTIONS", "false")
ENABLE_AUREA_CONTEXT_AWARE_EXECUTION = _env_flag("ENABLE_AUREA_CONTEXT_AWARE_EXECUTION", "false")
ENABLE_AUREA_COMMAND_CHAINING = _env_flag("ENABLE_AUREA_COMMAND_CHAINING", "false")
ENABLE_AUREA_COMMAND_MACROS = _env_flag("ENABLE_AUREA_COMMAND_MACROS", "false")
ENABLE_AUREA_VOICE_COMMAND_FOUNDATION = _env_flag("ENABLE_AUREA_VOICE_COMMAND_FOUNDATION", "false")

async def _perplexity_fallback(prompt: str, system_prompt: str = "") -> str:
    """Use Perplexity as fallback when OpenAI fails."""
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("PERPLEXITY_API_KEY not configured for fallback")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.7
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

async def _gemini_fallback(prompt: str, system_prompt: str = "") -> str:
    """Use Gemini as fallback when others fail."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured for fallback")

    configure(api_key=GEMINI_API_KEY)
    model = GenerativeModel("gemini-2.0-flash")
    
    # Combine system and user prompt for Gemini (it supports system instructions but direct concat is safer for simple fallback)
    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    # Run in executor to avoid blocking async loop
    response = await asyncio.to_thread(model.generate_content, full_prompt)
    return response.text

class AUREANLUProcessor:
    def __init__(self, llm_model: Any, integration_layer: Any, aurea_instance: Any, ai_board_instance: Any,
                 db_pool: Any = None, mcp_client: Any = None, execution_profile: Optional[dict[str, Any]] = None):
        self.llm = llm_model
        self.integration_layer = integration_layer
        self.aurea = aurea_instance
        self.ai_board = ai_board_instance
        self.execution_profile = execution_profile or {"persona_id": "core", "allowed_scopes": ["admin"]}
        if build_execution_profile and not execution_profile:
            try:
                self.execution_profile = build_execution_profile({})
            except Exception:
                pass
        self.allowed_scopes = self._normalize_allowed_scopes(
            self.execution_profile.get("allowed_scopes")
        )

        # Initialize Power Layer for full operational capability
        self.power_layer = None
        if POWER_LAYER_AVAILABLE:
            try:
                self.power_layer = get_power_layer(db_pool=db_pool, mcp_client=mcp_client)
                logger.info("🔋 AUREA NLU initialized with Power Layer - full operational capability enabled")
            except Exception as e:
                logger.warning(f"Power Layer initialization failed: {e}")

        if self.integration_layer is None:
            logger.error("AUREA NLU initialized without integration layer; task and workflow skills will be unavailable.")
        if self.aurea is None:
            logger.error("AUREA NLU initialized without AUREA instance; orchestrator skills will be unavailable.")
        if self.ai_board is None:
            logger.warning("AUREA NLU initialized without AI Board instance; board-related skills will be unavailable.")

        # Dynamically built registry of executable actions
        self.full_skill_registry = self._build_skill_registry()
        self.skill_registry = self._filter_skills_by_scope(self.full_skill_registry)
        self.blocked_skill_names = sorted(set(self.full_skill_registry) - set(self.skill_registry))
        logger.info(
            "AUREA NLU profile=%s scopes=%s skills=%s blocked=%s",
            self.execution_profile.get("persona_id", "core"),
            sorted(self.allowed_scopes),
            len(self.skill_registry),
            len(self.blocked_skill_names),
        )
        self.command_history: list[dict[str, Any]] = []
        self.command_macros = self._load_command_macros()

    @staticmethod
    def _normalize_allowed_scopes(raw_scopes: Any) -> set[str]:
        scopes: set[str] = set()
        if isinstance(raw_scopes, str):
            scopes = {part.strip().lower() for part in raw_scopes.split(",") if part.strip()}
        elif isinstance(raw_scopes, (list, tuple, set)):
            scopes = {str(part).strip().lower() for part in raw_scopes if str(part).strip()}

        scopes = {scope for scope in scopes if scope in {"read_only", "operator", "admin"}}
        if "admin" in scopes:
            scopes.update({"operator", "read_only"})
        elif "operator" in scopes:
            scopes.add("read_only")
        if not scopes:
            scopes = {"read_only"}
        return scopes

    def _scope_allows(self, required_scope: Any) -> bool:
        scope = str(required_scope or "read_only").strip().lower()
        if scope not in {"read_only", "operator", "admin"}:
            scope = "read_only"
        return scope in self.allowed_scopes

    def _filter_skills_by_scope(self, registry: dict[str, Any]) -> dict[str, Any]:
        filtered: dict[str, Any] = {}
        for skill_name, skill_data in (registry or {}).items():
            required_scope = skill_data.get("scope", "read_only")
            if self._scope_allows(required_scope):
                filtered[skill_name] = skill_data
        return filtered

    def _build_skill_registry(self) -> dict[str, Any]:
        """
        Build the skill registry based on which subsystems are actually available.
        This prevents NLU from advertising skills that would crash at runtime.
        """
        registry: dict[str, Any] = {}

        # Integration layer-dependent skills
        if self.integration_layer is not None:
            registry.update(
                {
                    "create_task": {
                        "description": "Create a new AI task for autonomous execution.",
                        "parameters": {
                            "task_type": "string",
                            "description": "string",
                            "priority": "enum (low, medium, high, critical)",
                            "auto_execute": "boolean",
                            "due_date": "string (ISO format)",
                        },
                        "action": self.integration_layer.create_task,
                        "scope": "operator",
                    },
                    "get_task_status": {
                        "description": "Get the current status of an AI task.",
                        "parameters": {"task_id": "string"},
                        "action": self.integration_layer.get_task_status,
                        "scope": "read_only",
                    },
                    "list_tasks": {
                        "description": "List all AI tasks with optional filters.",
                        "parameters": {
                            "status": "enum (pending, in_progress, completed, failed, cancelled)",
                            "limit": "integer",
                        },
                        "action": self.integration_layer.list_tasks,
                        "scope": "read_only",
                    },
                    "execute_task": {
                        "description": "Manually trigger execution of a specific AI task.",
                        "parameters": {"task_id": "string"},
                        "action": self.integration_layer.execute_ai_task,
                        "scope": "operator",
                    },
                    "get_task_stats": {
                        "description": "Get statistics about the AI task system.",
                        "parameters": {},
                        "action": self.integration_layer.get_task_stats,
                        "scope": "read_only",
                    },
                    "orchestrate_workflow": {
                        "description": "Execute a complex multi-stage workflow using LangGraph orchestration.",
                        "parameters": {"task_description": "string", "context": "object"},
                        "action": self.integration_layer.orchestrate_complex_workflow,
                        "scope": "operator",
                    },
                }
            )

        # AUREA orchestrator-dependent skills
        if self.aurea is not None:
            registry.update(
                {
                    "set_autonomy_level": {
                        "description": "Set AUREA's autonomous operation level (0-100).",
                        "parameters": {"level": "integer (0, 25, 50, 75, 100)"},
                        "action": self.aurea.set_autonomy_level,
                        "scope": "admin",
                    },
                    "get_aurea_status": {
                        "description": "Get AUREA's current operational status and metrics.",
                        "parameters": {},
                        "action": self.aurea.get_status,
                        "scope": "read_only",
                    },
                    "start_aurea": {
                        "description": "Start AUREA's autonomous orchestration loop.",
                        "parameters": {},
                        "action": self.aurea.orchestrate,
                        "scope": "admin",
                    },
                    "stop_aurea": {
                        "description": "Stop AUREA's autonomous orchestration loop.",
                        "parameters": {},
                        "action": self.aurea.stop,
                        "scope": "admin",
                    },
                    "get_system_health": {
                        "description": "Get a comprehensive health report of the entire AI OS.",
                        "parameters": {},
                        "action": self.aurea.get_status,
                        "scope": "read_only",
                    },
                }
            )

        # AI Board-dependent skills
        if self.ai_board is not None:
            registry.update(
                {
                    "get_board_status": {
                        "description": "Get the current status of the AI Board of Directors.",
                        "parameters": {},
                        "action": self.ai_board.get_board_status,
                        "scope": "read_only",
                    },
                    "submit_proposal": {
                        "description": "Submit a strategic proposal to the AI Board for review.",
                        "parameters": {
                            "title": "string",
                            "description": "string",
                            "type": "enum (STRATEGIC, OPERATIONAL)",
                            "urgency": "integer (1-10)",
                        },
                        "action": self.ai_board.submit_proposal,
                        "scope": "admin",
                    },
                    "convene_board_meeting": {
                        "description": "Convene an immediate meeting of the AI Board of Directors.",
                        "parameters": {},
                        "action": self.ai_board.convene_meeting,
                        "scope": "admin",
                    },
                }
            )

        # Power Layer skills - full operational capability
        if self.power_layer is not None:
            power_skills = self.power_layer.get_skill_registry()
            registry.update(power_skills)
            logger.info(f"🔋 Added {len(power_skills)} Power Layer skills to AUREA NLU")

        return registry

    def _get_serializable_registry(self) -> dict[str, dict[str, Any]]:
        """Get a JSON-serializable version of the skill registry (without action functions)."""
        return {
            skill_name: {
                "description": skill_data.get("description", ""),
                "parameters": skill_data.get("parameters", {}),
                "scope": skill_data.get("scope", "read_only"),
            }
            for skill_name, skill_data in self.skill_registry.items()
        }

    def _load_command_macros(self) -> dict[str, list[str]]:
        """Load command macros from defaults + optional environment JSON."""
        default_macros: dict[str, list[str]] = {
            "ops_health_brief": [
                "get system health",
                "get aurea status",
                "get task stats",
            ],
            "ops_recovery_brief": [
                "get system health",
                "list tasks",
                "get board status",
            ],
        }
        raw = os.getenv("AUREA_COMMAND_MACROS_JSON", "").strip()
        if not raw:
            return default_macros
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return default_macros
            for macro_name, steps in parsed.items():
                if isinstance(steps, list):
                    valid_steps = [str(step).strip() for step in steps if str(step).strip()]
                    if valid_steps:
                        default_macros[str(macro_name).strip().lower()] = valid_steps
        except Exception as exc:
            logger.warning("Failed loading AUREA_COMMAND_MACROS_JSON: %s", exc)
        return default_macros

    def _expand_macro_command(self, command_text: str) -> Optional[list[str]]:
        if not ENABLE_AUREA_COMMAND_MACROS:
            return None
        normalized = (command_text or "").strip().lower()
        macro_name = None

        if normalized.startswith("macro:"):
            macro_name = normalized.split(":", 1)[1].strip()
        else:
            macro_match = re.match(r"^(?:run|execute)\s+macro\s+([a-zA-Z0-9_\-]+)$", normalized)
            if macro_match:
                macro_name = macro_match.group(1).strip()

        if not macro_name:
            return None
        return self.command_macros.get(macro_name)

    def _split_chained_commands(self, command_text: str) -> list[str]:
        """Split a command into sequential steps when chaining is enabled."""
        if not ENABLE_AUREA_COMMAND_CHAINING:
            return [command_text]

        macro_steps = self._expand_macro_command(command_text)
        if macro_steps:
            return macro_steps

        normalized = (command_text or "").strip()
        if not normalized:
            return []

        separators = r"\s*(?:\n+|;|->|\|>|\s+then\s+|\s+and\s+then\s+|\s*&&\s*)\s*"
        parts = [part.strip() for part in re.split(separators, normalized) if part.strip()]
        return parts if len(parts) > 1 else [normalized]

    def _record_command_history(
        self,
        command_text: str,
        result: dict[str, Any],
        source: str = "text",
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "command": command_text,
            "source": source,
            "status": result.get("status", "unknown"),
        }
        self.command_history.append(entry)
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]

    async def _call_maybe_async(self, func: Any, **kwargs) -> Any:
        """Invoke sync/async callables with a single helper."""
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        outcome = func(**kwargs)
        if inspect.isawaitable(outcome):
            return await outcome
        return outcome

    async def _build_execution_context(self, command_text: str) -> dict[str, Any]:
        """Context snapshot to support context-aware command execution."""
        if not ENABLE_AUREA_CONTEXT_AWARE_EXECUTION:
            return {}

        context: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "command_text": command_text,
            "recent_commands": self.command_history[-5:],
        }

        if self.aurea is not None and hasattr(self.aurea, "get_status"):
            try:
                get_status = getattr(self.aurea, "get_status")
                context["aurea_status"] = await self._call_maybe_async(get_status)
            except Exception as exc:
                context["aurea_status_error"] = str(exc)

        if self.integration_layer is not None and hasattr(self.integration_layer, "get_task_stats"):
            try:
                get_task_stats = getattr(self.integration_layer, "get_task_stats")
                context["task_stats"] = await self._call_maybe_async(get_task_stats)
            except Exception as exc:
                context["task_stats_error"] = str(exc)

        return context

    def _apply_context_aware_defaults(
        self,
        intent: str,
        parameters: dict[str, Any],
        execution_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject low-risk defaults using current system context."""
        if not ENABLE_AUREA_CONTEXT_AWARE_EXECUTION:
            return parameters

        adjusted = dict(parameters or {})
        aurea_status = execution_context.get("aurea_status") or {}
        alerts = aurea_status.get("alerts") if isinstance(aurea_status, dict) else []
        high_alert_count = len(alerts) >= 3 if isinstance(alerts, list) else False

        if intent == "create_task" and "priority" not in adjusted:
            adjusted["priority"] = "high" if high_alert_count else "medium"

        if intent == "orchestrate_workflow":
            workflow_context = adjusted.get("context") or {}
            if isinstance(workflow_context, dict):
                workflow_context.setdefault("aurea_context", execution_context)
                adjusted["context"] = workflow_context

        return adjusted

    def suggest_commands(self, partial_command: str, limit: int = 5) -> dict[str, Any]:
        """Provide command suggestion/autocomplete candidates."""
        if not ENABLE_AUREA_COMMAND_SUGGESTIONS:
            return {"status": "skipped", "reason": "ENABLE_AUREA_COMMAND_SUGGESTIONS=false", "suggestions": []}

        query = (partial_command or "").strip().lower()
        if not query:
            return {"status": "ok", "suggestions": []}

        candidates: list[dict[str, Any]] = []
        for intent_name, skill in self.skill_registry.items():
            candidates.append(
                {
                    "command": intent_name,
                    "description": skill.get("description", ""),
                    "source": "skill",
                }
            )
        for macro_name, steps in self.command_macros.items():
            candidates.append(
                {
                    "command": f"macro:{macro_name}",
                    "description": f"{len(steps)}-step macro",
                    "source": "macro",
                }
            )

        starts_with = [c for c in candidates if c["command"].lower().startswith(query)]
        contains = [
            c for c in candidates
            if query in c["command"].lower() or query in c["description"].lower()
        ]
        close = {
            name: True
            for name in difflib.get_close_matches(
                query, [c["command"].lower() for c in candidates], n=limit * 2, cutoff=0.4
            )
        }
        fuzzy = [c for c in candidates if c["command"].lower() in close]

        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for bucket in (starts_with, contains, fuzzy):
            for candidate in bucket:
                key = candidate["command"].lower()
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(candidate)
                if len(ordered) >= limit:
                    break
            if len(ordered) >= limit:
                break

        return {"status": "ok", "suggestions": ordered}

    def autocomplete_command(self, partial_command: str, limit: int = 5) -> dict[str, Any]:
        """Alias for UI autocomplete integrations."""
        return self.suggest_commands(partial_command, limit=limit)

    async def execute_command_chain(
        self,
        command_steps: list[str],
        original_command: str,
    ) -> dict[str, Any]:
        """Execute chained commands in sequence with short-circuit on failure."""
        if not ENABLE_AUREA_COMMAND_CHAINING:
            return {"status": "skipped", "reason": "ENABLE_AUREA_COMMAND_CHAINING=false"}

        results: list[dict[str, Any]] = []
        for index, step in enumerate(command_steps, start=1):
            step_result = await self.execute_natural_language_command(step, _internal_chain=True)
            results.append({"step": index, "command": step, "result": step_result})
            if step_result.get("status") in {"failed", "pending_confirmation", "clarification_needed"}:
                return {
                    "status": "partial",
                    "executed_steps": index,
                    "total_steps": len(command_steps),
                    "results": results,
                    "original_command": original_command,
                }

        return {
            "status": "success",
            "executed_steps": len(command_steps),
            "total_steps": len(command_steps),
            "results": results,
            "original_command": original_command,
        }

    def register_command_macro(
        self,
        macro_name: str,
        steps: list[str],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Register a command macro at runtime."""
        if not ENABLE_AUREA_COMMAND_MACROS:
            return {"status": "skipped", "reason": "ENABLE_AUREA_COMMAND_MACROS=false"}
        normalized = (macro_name or "").strip().lower()
        cleaned_steps = [str(step).strip() for step in (steps or []) if str(step).strip()]
        if not normalized:
            return {"status": "error", "message": "macro_name is required"}
        if not cleaned_steps:
            return {"status": "error", "message": "steps must contain at least one command"}
        if normalized in self.command_macros and not overwrite:
            return {"status": "error", "message": f"macro '{normalized}' already exists"}
        self.command_macros[normalized] = cleaned_steps
        return {"status": "registered", "macro": normalized, "steps": cleaned_steps}

    async def execute_voice_command(self, voice_payload: dict[str, Any]) -> dict[str, Any]:
        """Voice command support foundation (transcript routing + metadata)."""
        if not ENABLE_AUREA_VOICE_COMMAND_FOUNDATION:
            return {"status": "skipped", "reason": "ENABLE_AUREA_VOICE_COMMAND_FOUNDATION=false"}

        transcript = (voice_payload or {}).get("transcript")
        confidence = float((voice_payload or {}).get("confidence", 0.0) or 0.0)
        language = (voice_payload or {}).get("language", "en-US")
        if not transcript:
            return {"status": "error", "message": "voice transcript missing"}
        if confidence and confidence < 0.55:
            return {
                "status": "clarification_needed",
                "message": "Low-confidence voice transcript. Please confirm phrasing.",
                "transcript": transcript,
                "confidence": confidence,
            }

        command_result = await self.execute_natural_language_command(transcript)
        return {
            "status": "success",
            "transcript": transcript,
            "confidence": confidence,
            "language": language,
            "result": command_result,
        }

    async def analyze_command_intent(self, command_text: str) -> dict[str, Any]:
        """Use LLM to analyze natural language command intent and extract parameters."""
        if ENABLE_AUREA_COMPLEX_COMMAND_NLU:
            split_steps = self._split_chained_commands(command_text)
            if len(split_steps) > 1:
                return {
                    "intent": "COMMAND_CHAIN",
                    "parameters": {"steps": split_steps},
                    "confidence": 0.95,
                    "requires_confirmation": False,
                    "clarification_needed": None,
                }

        execution_context = await self._build_execution_context(command_text)
        serializable_registry = self._get_serializable_registry()
        profile_summary = {
            "persona_id": self.execution_profile.get("persona_id"),
            "persona_name": self.execution_profile.get("persona_name"),
            "source": self.execution_profile.get("source"),
            "user_role": self.execution_profile.get("user_role"),
            "allowed_scopes": sorted(self.allowed_scopes),
            "blocked_skills_count": len(self.blocked_skill_names),
        }
        context_prompt = ""
        if ENABLE_AUREA_CONTEXT_AWARE_EXECUTION and execution_context:
            safe_context = {
                "timestamp": execution_context.get("timestamp"),
                "recent_commands": execution_context.get("recent_commands", []),
                "aurea_status": execution_context.get("aurea_status", {}),
                "task_stats": execution_context.get("task_stats", {}),
            }
            context_prompt = f"\nCurrent runtime context (for parameter defaults/risk): {json.dumps(safe_context, default=str)}"

        system_prompt = f"""You are AUREA, the Founder's Executive AI Assistant.
            Your role is to interpret natural language commands for the entire BrainOps AI OS.
            Based on the user's command, identify the primary intent and extract all relevant parameters.
            You have access to the following tools/skills:
{json.dumps(serializable_registry, indent=2)}
{context_prompt}
            Execution profile and scope limits:
{json.dumps(profile_summary, indent=2)}

            If the command is ambiguous or requires more information, ask clarifying questions.
            If the command implies a high-impact action, set 'requires_confirmation' to true.
            Never select a skill outside the provided allowed scopes.
            Respond ONLY in JSON format with 'intent', 'parameters', 'confidence', 'requires_confirmation', and 'clarification_needed'.
            Ensure 'parameters' matches the schema of the identified intent's action. If no clear intent is found, use 'UNKNOWN'."""

        user_prompt = f"User command: \"{command_text}\""
        response_content = None

        # Try OpenAI first, fallback to Perplexity then Gemini
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            response = await self.llm.ainvoke(prompt_template.format_messages())
            response_content = response.content
        except Exception as openai_error:
            logger.warning(f"OpenAI failed for AUREA NLU, trying Perplexity fallback: {openai_error}")
            try:
                response_content = await _perplexity_fallback(user_prompt, system_prompt)
                logger.info("Successfully used Perplexity fallback for AUREA NLU")
            except Exception as perplexity_error:
                logger.warning(f"Perplexity fallback failed, trying Gemini: {perplexity_error}")
                try:
                    response_content = await _gemini_fallback(user_prompt, system_prompt)
                    logger.info("Successfully used Gemini fallback for AUREA NLU")
                except Exception as gemini_error:
                    logger.error(f"All AI providers failed: {gemini_error}")
                    return {"intent": "UNKNOWN", "parameters": {}, "confidence": 0.0, "requires_confirmation": False, "clarification_needed": f"AI providers unavailable: {openai_error}"}

        try:
            # Clean response content - extract JSON from potential markdown code blocks
            clean_content = response_content.strip()
            if clean_content.startswith("```"):
                # Extract JSON from code block
                lines = clean_content.split('\n')
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```") and not in_block:
                        in_block = True
                        continue
                    elif line.startswith("```") and in_block:
                        break
                    elif in_block:
                        json_lines.append(line)
                clean_content = '\n'.join(json_lines)

            parsed_response = json.loads(clean_content)
            # Basic validation of parsed_response structure
            if not all(k in parsed_response for k in ["intent", "parameters", "confidence"]):
                raise ValueError("Missing required keys in LLM response")
            parsed_response.setdefault("requires_confirmation", False)
            parsed_response.setdefault("clarification_needed", None)
            if ENABLE_AUREA_COMMAND_SUGGESTIONS and parsed_response.get("intent") == "UNKNOWN":
                parsed_response["suggestions"] = self.suggest_commands(command_text).get("suggestions", [])
            parsed_response["execution_context"] = execution_context
            return parsed_response
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {response_content}. Error: {e}")
            response = {
                "intent": "UNKNOWN",
                "parameters": {},
                "confidence": 0.0,
                "requires_confirmation": False,
                "clarification_needed": "Could not parse LLM response as valid JSON.",
            }
            if ENABLE_AUREA_COMMAND_SUGGESTIONS:
                response["suggestions"] = self.suggest_commands(command_text).get("suggestions", [])
            response["execution_context"] = execution_context
            return response
        except ValueError as e:
            logger.error(f"Invalid structure in LLM response: {response_content}. Error: {e}")
            response = {
                "intent": "UNKNOWN",
                "parameters": {},
                "confidence": 0.0,
                "requires_confirmation": False,
                "clarification_needed": f"Invalid structure in LLM response: {e}",
            }
            if ENABLE_AUREA_COMMAND_SUGGESTIONS:
                response["suggestions"] = self.suggest_commands(command_text).get("suggestions", [])
            response["execution_context"] = execution_context
            return response

    async def _execute_skill_intent(
        self,
        intent: str,
        parameters: dict[str, Any],
        command_text: str,
        intent_data: dict[str, Any],
    ) -> dict[str, Any]:
        skill = self.skill_registry.get(intent)
        if not (skill and skill.get("action")):
            if intent in self.full_skill_registry and intent not in self.skill_registry:
                return {
                    "status": "permission_denied",
                    "message": (
                        f"Intent '{intent}' is outside the allowed capability scopes for "
                        f"persona '{self.execution_profile.get('persona_id', 'core')}'."
                    ),
                    "required_scope": self.full_skill_registry[intent].get("scope", "read_only"),
                    "allowed_scopes": sorted(self.allowed_scopes),
                    "intent_data": intent_data,
                }
            response: dict[str, Any] = {
                "status": "unknown_intent",
                "message": f"AUREA does not know how to execute '{intent}'. Please try rephrasing or ask for available commands.",
                "intent_data": intent_data,
            }
            if ENABLE_AUREA_COMMAND_SUGGESTIONS:
                response["suggestions"] = self.suggest_commands(command_text).get("suggestions", [])
            return response

        try:
            required_scope = skill.get("scope", "read_only")
            if not self._scope_allows(required_scope):
                return {
                    "status": "permission_denied",
                    "message": (
                        f"Intent '{intent}' requires scope '{required_scope}', "
                        f"but allowed scopes are {sorted(self.allowed_scopes)}."
                    ),
                    "required_scope": required_scope,
                    "allowed_scopes": sorted(self.allowed_scopes),
                    "intent_data": intent_data,
                }

            execution_context = intent_data.get("execution_context", {})
            adjusted_parameters = self._apply_context_aware_defaults(intent, parameters, execution_context)

            action_func = skill["action"]
            sig = inspect.signature(action_func)
            filtered_params = {k: v for k, v in adjusted_parameters.items() if k in sig.parameters}

            if intent == "start_aurea":
                create_safe_task(action_func(**filtered_params))
                return {
                    "status": "success",
                    "result": "AUREA orchestration started in background.",
                    "message": f"Command '{command_text}' executed successfully by AUREA.",
                    "intent_data": intent_data,
                }

            if intent == "convene_board_meeting":
                create_safe_task(action_func(**filtered_params))
                return {
                    "status": "success",
                    "result": "AI Board meeting convened in background.",
                    "message": f"Command '{command_text}' executed successfully by AUREA.",
                    "intent_data": intent_data,
                }

            if intent == "orchestrate_workflow":
                context = adjusted_parameters.get("context") or {}
                result = await self.integration_layer.orchestrate_complex_workflow(
                    task_description=adjusted_parameters.get("task_description"),
                    context=context,
                )
                return {
                    "status": "success",
                    "result": result,
                    "message": f"Command '{command_text}' executed successfully by AUREA.",
                    "intent_data": intent_data,
                }

            result = await self._call_maybe_async(action_func, **filtered_params)
            return {
                "status": "success",
                "result": result,
                "message": f"Command '{command_text}' executed successfully by AUREA.",
                "intent_data": intent_data,
            }
        except Exception as exc:
            logger.error("Error executing skill '%s' with parameters %s: %s", intent, parameters, exc)
            return {
                "status": "failed",
                "error": str(exc),
                "message": f"Failed to execute command '{command_text}' by AUREA.",
                "intent_data": intent_data,
            }

    async def execute_natural_language_command(
        self, command_text: str, _internal_chain: bool = False
    ) -> dict[str, Any]:
        """Analyze, confirm (if needed), and execute a natural language command."""
        if ENABLE_AUREA_COMMAND_CHAINING and not _internal_chain:
            chain_steps = self._split_chained_commands(command_text)
            if len(chain_steps) > 1:
                chain_result = await self.execute_command_chain(chain_steps, original_command=command_text)
                self._record_command_history(command_text, chain_result)
                return chain_result

        intent_data = await self.analyze_command_intent(command_text)
        if intent_data.get("intent") == "COMMAND_CHAIN":
            steps = intent_data.get("parameters", {}).get("steps", [])
            chain_result = await self.execute_command_chain(steps, original_command=command_text)
            self._record_command_history(command_text, chain_result)
            return chain_result

        if intent_data.get("clarification_needed"):
            result = {
                "status": "clarification_needed",
                "message": intent_data["clarification_needed"],
                "original_command": command_text,
            }
            self._record_command_history(command_text, result)
            return result

        if intent_data.get("requires_confirmation"):
            logger.info(
                "Human confirmation required for intent: %s with params: %s",
                intent_data.get("intent"),
                intent_data.get("parameters"),
            )
            if os.getenv("AUREA_AUTO_CONFIRM", "false").lower() != "true":
                result = {
                    "status": "pending_confirmation",
                    "message": f"Action '{intent_data.get('intent')}' requires confirmation.",
                    "intent_data": intent_data,
                    "confirmation_token": f"confirm_{int(datetime.now().timestamp())}",
                }
                self._record_command_history(command_text, result)
                return result
            logger.warning("⚠️ Auto-confirming action due to AUREA_AUTO_CONFIRM=true")

        intent = intent_data.get("intent")
        parameters = intent_data.get("parameters", {})
        result = await self._execute_skill_intent(intent, parameters, command_text, intent_data)
        self._record_command_history(command_text, result)
        return result

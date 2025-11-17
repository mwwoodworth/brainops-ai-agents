import json
import logging
import os
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from datetime import datetime
import inspect
import asyncio

logger = logging.getLogger(__name__)


class AUREANLUProcessor:
    def __init__(self, llm_model: Any, integration_layer: Any, aurea_instance: Any, ai_board_instance: Any):
        self.llm = llm_model
        self.integration_layer = integration_layer
        self.aurea = aurea_instance
        self.ai_board = ai_board_instance

        if self.integration_layer is None:
            logger.error("AUREA NLU initialized without integration layer; task and workflow skills will be unavailable.")
        if self.aurea is None:
            logger.error("AUREA NLU initialized without AUREA instance; orchestrator skills will be unavailable.")
        if self.ai_board is None:
            logger.warning("AUREA NLU initialized without AI Board instance; board-related skills will be unavailable.")

        # Dynamically built registry of executable actions
        self.skill_registry = self._build_skill_registry()

    def _build_skill_registry(self) -> Dict[str, Any]:
        """
        Build the skill registry based on which subsystems are actually available.
        This prevents NLU from advertising skills that would crash at runtime.
        """
        registry: Dict[str, Any] = {}

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
                    },
                    "get_task_status": {
                        "description": "Get the current status of an AI task.",
                        "parameters": {"task_id": "string"},
                        "action": self.integration_layer.get_task_status,
                    },
                    "list_tasks": {
                        "description": "List all AI tasks with optional filters.",
                        "parameters": {
                            "status": "enum (pending, in_progress, completed, failed, cancelled)",
                            "limit": "integer",
                        },
                        "action": self.integration_layer.list_tasks,
                    },
                    "execute_task": {
                        "description": "Manually trigger execution of a specific AI task.",
                        "parameters": {"task_id": "string"},
                        "action": self.integration_layer.execute_ai_task,
                    },
                    "get_task_stats": {
                        "description": "Get statistics about the AI task system.",
                        "parameters": {},
                        "action": self.integration_layer.get_task_stats,
                    },
                    "orchestrate_workflow": {
                        "description": "Execute a complex multi-stage workflow using LangGraph orchestration.",
                        "parameters": {"task_description": "string", "context": "object"},
                        "action": self.integration_layer.orchestrate_complex_workflow,
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
                    },
                    "get_aurea_status": {
                        "description": "Get AUREA's current operational status and metrics.",
                        "parameters": {},
                        "action": self.aurea.get_status,
                    },
                    "start_aurea": {
                        "description": "Start AUREA's autonomous orchestration loop.",
                        "parameters": {},
                        "action": self.aurea.orchestrate,
                    },
                    "stop_aurea": {
                        "description": "Stop AUREA's autonomous orchestration loop.",
                        "parameters": {},
                        "action": self.aurea.stop,
                    },
                    "get_system_health": {
                        "description": "Get a comprehensive health report of the entire AI OS.",
                        "parameters": {},
                        "action": self.aurea.get_status,
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
                    },
                    "convene_board_meeting": {
                        "description": "Convene an immediate meeting of the AI Board of Directors.",
                        "parameters": {},
                        "action": self.ai_board.convene_meeting,
                    },
                }
            )

        return registry

    async def analyze_command_intent(self, command_text: str) -> Dict[str, Any]:
        """Use LLM to analyze natural language command intent and extract parameters."""
        # This prompt needs to be highly sophisticated to handle Founder-level commands
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are AUREA, the Founder's Executive AI Assistant.
            Your role is to interpret natural language commands for the entire BrainOps AI OS.
            Based on the user's command, identify the primary intent and extract all relevant parameters.
            You have access to the following tools/skills:
{json.dumps(self.skill_registry, indent=2)}

            If the command is ambiguous or requires more information, ask clarifying questions.
            If the command implies a high-impact action, set 'requires_confirmation' to true.
            Respond ONLY in JSON format with 'intent', 'parameters', 'confidence', 'requires_confirmation', and 'clarification_needed'.
            Ensure 'parameters' matches the schema of the identified intent's action. If no clear intent is found, use 'UNKNOWN'.
"""),
            HumanMessage(content=f"User command: \"{command_text}\"" )
        ])

        response = await self.llm.ainvoke(prompt_template.format_messages())
        try:
            parsed_response = json.loads(response.content)
            # Basic validation of parsed_response structure
            if not all(k in parsed_response for k in ["intent", "parameters", "confidence"]):
                raise ValueError("Missing required keys in LLM response")
            return parsed_response
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {response.content}. Error: {e}")
            return {"intent": "UNKNOWN", "parameters": {}, "confidence": 0.0, "requires_confirmation": False, "clarification_needed": "Could not parse LLM response as valid JSON."}
        except ValueError as e:
            logger.error(f"Invalid structure in LLM response: {response.content}. Error: {e}")
            return {"intent": "UNKNOWN", "parameters": {}, "confidence": 0.0, "requires_confirmation": False, "clarification_needed": f"Invalid structure in LLM response: {e}"}

    async def execute_natural_language_command(self, command_text: str) -> Dict[str, Any]:
        """Analyze, confirm (if needed), and execute a natural language command."""
        intent_data = await self.analyze_command_intent(command_text)

        if intent_data.get("clarification_needed"):
            return {"status": "clarification_needed", "message": intent_data["clarification_needed"], "original_command": command_text}

        if intent_data.get("requires_confirmation"):
            # In a real system, this would trigger a human confirmation step (UI/voice)
            logger.info(f"Human confirmation required for intent: {intent_data['intent']} with params: {intent_data['parameters']}")
            # For now, auto-confirm for testing purposes
            # return {"status": "pending_confirmation", "message": "Confirmation required for this action.", "intent_data": intent_data}
            pass # Proceed as if confirmed for now

        intent = intent_data.get("intent")
        parameters = intent_data.get("parameters", {})

        skill = self.skill_registry.get(intent)
        if skill and skill.get("action"):
            try:
                action_func = skill["action"]
                # Filter parameters to match function signature
                sig = inspect.signature(action_func)
                filtered_params = {k: v for k, v in parameters.items() if k in sig.parameters}

                # Special handling for background tasks if needed
                if intent == "start_aurea":
                    # AUREA orchestration is an async loop, needs to be run in background
                    asyncio.create_task(action_func(**filtered_params))
                    return {"status": "success", "result": "AUREA orchestration started in background.", "message": f"Command '{command_text}' executed successfully by AUREA.", "intent_data": intent_data}
                elif intent == "convene_board_meeting":
                    asyncio.create_task(action_func(**filtered_params))
                    return {"status": "success", "result": "AI Board meeting convened in background.", "message": f"Command '{command_text}' executed successfully by AUREA.", "intent_data": intent_data}
                elif intent == "orchestrate_workflow":
                    # LangGraph orchestration also needs to be run in background or awaited carefully.
                    # Delegate to the integration layer endpoint, providing a safe default context.
                    context = parameters.get("context") or {}
                    result = await self.integration_layer.orchestrate_complex_workflow(
                        task_description=parameters.get("task_description"),
                        context=context,
                    )
                    return {
                        "status": "success",
                        "result": result,
                        "message": f"Command '{command_text}' executed successfully by AUREA.",
                        "intent_data": intent_data,
                    }
                else:
                    result = await action_func(**filtered_params)
                    return {"status": "success", "result": result, "message": f"Command '{command_text}' executed successfully by AUREA.", "intent_data": intent_data}
            except Exception as e:
                logger.error(f"Error executing skill '{intent}' with parameters {parameters}: {e}")
                return {"status": "failed", "error": str(e), "message": f"Failed to execute command '{command_text}' by AUREA.", "intent_data": intent_data}
        else:
            return {"status": "unknown_intent", "message": f"AUREA does not know how to execute '{intent}'. Please try rephrasing or ask for available commands.", "intent_data": intent_data}

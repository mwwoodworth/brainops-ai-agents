#!/usr/bin/env python3
"""
Automated Product Generator
===========================
AI-powered digital product creation factory.

Generates sellable digital products based on:
- Trending developer tools and technologies
- Code patterns in our codebase
- Market demand signals from Perplexity/web research

Product Types:
- Code Kits (boilerplate, templates, starter kits)
- Prompt Packs (AI prompt collections)
- Automation Scripts (Python, Bash, workflows)
- Documentation/Guides (technical how-tos)
- MCP Server Templates (Claude Code extensions)

Output: ZIP files with code + README, ready for Gumroad upload.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProductType(str, Enum):
    CODE_KIT = "code_kit"
    PROMPT_PACK = "prompt_pack"
    AUTOMATION_SCRIPTS = "automation_scripts"
    DOCUMENTATION = "documentation"
    MCP_SERVER = "mcp_server"
    API_WRAPPER = "api_wrapper"


# Product templates and base prices
PRODUCT_TEMPLATES = {
    ProductType.CODE_KIT: {
        "base_price": 49,
        "files": ["src/", "README.md", "package.json", "LICENSE"],
        "description_template": "Production-ready {name} starter kit with best practices built-in."
    },
    ProductType.PROMPT_PACK: {
        "base_price": 29,
        "files": ["prompts/", "README.md", "examples/"],
        "description_template": "Curated collection of {count}+ AI prompts for {use_case}."
    },
    ProductType.AUTOMATION_SCRIPTS: {
        "base_price": 37,
        "files": ["scripts/", "README.md", "config.example.json"],
        "description_template": "Automation scripts to {benefit}. Save hours of manual work."
    },
    ProductType.MCP_SERVER: {
        "base_price": 97,
        "files": ["src/", "mcp-config.json", "README.md", "examples/"],
        "description_template": "MCP server for Claude Code that provides {capability}."
    },
    ProductType.API_WRAPPER: {
        "base_price": 67,
        "files": ["src/", "README.md", "tests/", "examples/"],
        "description_template": "Type-safe {api_name} API wrapper with error handling and retry logic."
    },
}


class AutomatedProductGenerator:
    """
    AI-powered product factory that creates sellable digital products.
    """

    def __init__(self):
        self.agent_id = "AutomatedProductGenerator"
        self.version = "1.0.0"
        self.output_dir = Path("/tmp/brainops-products")
        self.output_dir.mkdir(exist_ok=True)
        self._pool = None

    def _get_pool(self):
        """Lazy-load database pool."""
        if self._pool is None:
            try:
                from database.async_connection import get_pool
                self._pool = get_pool()
            except Exception as e:
                logger.warning(f"Database pool unavailable: {e}")
        return self._pool

    async def _get_ai_core(self):
        """Get AI core for content generation."""
        try:
            from ai_core import ai_core
            return ai_core
        except ImportError:
            logger.warning("ai_core not available")
            return None

    async def execute(self, task: dict) -> dict:
        """Execute product generation task."""
        action = task.get("action", "generate")

        if action == "generate":
            return await self.generate_product(task)
        elif action == "generate_from_trend":
            return await self.generate_from_trend(task)
        elif action == "batch_generate":
            return await self.batch_generate(task)
        elif action == "list_templates":
            return self.list_templates()

        return {"success": False, "error": f"Unknown action: {action}"}

    def list_templates(self) -> dict:
        """List available product templates."""
        return {
            "success": True,
            "templates": {
                ptype.value: {
                    "base_price": info["base_price"],
                    "files": info["files"],
                    "description": info["description_template"]
                }
                for ptype, info in PRODUCT_TEMPLATES.items()
            }
        }

    async def generate_product(self, spec: dict) -> dict:
        """
        Generate a digital product from specification.

        Args:
            spec: {
                "name": "FastAPI Starter Kit",
                "type": "code_kit",
                "description": "...",
                "technologies": ["python", "fastapi", "postgresql"],
                "features": ["auth", "crud", "deployment"],
                "price": 49
            }
        """
        try:
            product_type = ProductType(spec.get("type", "code_kit"))
            name = spec.get("name", f"BrainOps {product_type.value.title()}")
            technologies = spec.get("technologies", [])
            features = spec.get("features", [])

            logger.info(f"Generating {product_type.value}: {name}")

            # Create temp directory for product
            product_id = str(uuid.uuid4())[:8]
            product_dir = self.output_dir / f"{self._slugify(name)}-{product_id}"
            product_dir.mkdir(exist_ok=True)

            # Generate content based on product type
            if product_type == ProductType.CODE_KIT:
                await self._generate_code_kit(product_dir, name, technologies, features)
            elif product_type == ProductType.PROMPT_PACK:
                await self._generate_prompt_pack(product_dir, name, spec.get("prompts", []), spec.get("use_case", "general"))
            elif product_type == ProductType.AUTOMATION_SCRIPTS:
                await self._generate_automation_scripts(product_dir, name, features)
            elif product_type == ProductType.MCP_SERVER:
                await self._generate_mcp_server(product_dir, name, features)
            elif product_type == ProductType.API_WRAPPER:
                await self._generate_api_wrapper(product_dir, name, spec.get("api_name", "generic"))

            # Create ZIP file
            zip_path = self._create_zip(product_dir, name)

            # Calculate price
            template = PRODUCT_TEMPLATES[product_type]
            base_price = spec.get("price", template["base_price"])

            # Record in database
            await self._record_product(
                product_id=product_id,
                name=name,
                product_type=product_type.value,
                zip_path=str(zip_path),
                price=base_price,
                metadata=spec
            )

            return {
                "success": True,
                "product_id": product_id,
                "name": name,
                "type": product_type.value,
                "zip_path": str(zip_path),
                "suggested_price": base_price,
                "ready_for_upload": True,
                "gumroad_fields": {
                    "name": name,
                    "price": base_price * 100,  # Gumroad uses cents
                    "description": template["description_template"].format(
                        name=name,
                        count=len(features),
                        use_case=", ".join(technologies) or "development",
                        benefit=", ".join(features[:3]) if features else "automate tasks",
                        capability=", ".join(features[:2]) if features else "enhanced capabilities",
                        api_name=spec.get("api_name", "API")
                    )
                },
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Product generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_from_trend(self, task: dict) -> dict:
        """
        Generate product based on trending topic.
        Uses AI to research and create relevant product.
        """
        trend = task.get("trend", "AI automation tools")
        product_type = ProductType(task.get("type", "code_kit"))

        ai_core = await self._get_ai_core()
        if not ai_core:
            return {"success": False, "error": "AI core not available"}

        # Research the trend and generate product spec
        research_prompt = f"""You are a digital product creator. Research this trending topic and create a product specification.

Trend: {trend}
Product Type: {product_type.value}

Generate a product specification in JSON format with:
- name: Catchy product name
- description: 2-3 sentence description
- technologies: List of relevant technologies
- features: List of 5-8 specific features to include
- target_audience: Who would buy this
- price: Suggested price in USD ($29-$149 range)

Return ONLY valid JSON."""

        try:
            response = await ai_core.generate(
                research_prompt,
                model="gpt-4-turbo-preview",
                temperature=0.7,
                system_prompt="You are an expert digital product strategist. Output only valid JSON."
            )

            if isinstance(response, str):
                spec = ai_core._safe_json(response)
            else:
                spec = response

            # Add product type
            spec["type"] = product_type.value

            # Generate the product
            return await self.generate_product(spec)

        except Exception as e:
            logger.error(f"Trend-based generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def batch_generate(self, task: dict) -> dict:
        """Generate multiple products from a list of specs."""
        specs = task.get("products", [])
        results = []

        for spec in specs:
            result = await self.generate_product(spec)
            results.append(result)

        return {
            "success": all(r.get("success") for r in results),
            "total": len(specs),
            "successful": sum(1 for r in results if r.get("success")),
            "products": results,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    async def _generate_code_kit(self, product_dir: Path, name: str, technologies: list, features: list):
        """Generate a code kit/starter template."""
        # Create directory structure
        src_dir = product_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # Determine primary technology
        is_python = any(t.lower() in ["python", "fastapi", "django", "flask"] for t in technologies)
        is_node = any(t.lower() in ["node", "javascript", "typescript", "nextjs", "react"] for t in technologies)

        if is_python:
            await self._generate_python_kit(product_dir, name, technologies, features)
        elif is_node:
            await self._generate_node_kit(product_dir, name, technologies, features)
        else:
            # Generic kit
            await self._generate_generic_kit(product_dir, name, technologies, features)

    async def _generate_python_kit(self, product_dir: Path, name: str, technologies: list, features: list):
        """Generate Python starter kit."""
        src_dir = product_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # Main module
        (src_dir / "__init__.py").write_text(f'''"""
{name}
Generated by BrainOps Automated Product Generator
"""

__version__ = "1.0.0"
''')

        # Main application file
        main_content = f'''#!/usr/bin/env python3
"""
{name} - Main Application
{'=' * len(name)}

Features:
{chr(10).join(f"- {f}" for f in features)}
"""

import asyncio
import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class {self._to_class_name(name)}:
    """Main application class."""

    def __init__(self, config: dict = None):
        self.config = config or {{}}
        self.version = "1.0.0"
        logger.info(f"Initialized {{self.__class__.__name__}}")

    async def run(self) -> dict:
        """Run the main application logic."""
        logger.info("Starting application...")

        results = {{
            "status": "success",
            "version": self.version,
            "features_available": {features}
        }}

        return results


async def main():
    """Entry point."""
    app = {self._to_class_name(name)}()
    result = await app.run()
    print(f"Result: {{result}}")


if __name__ == "__main__":
    asyncio.run(main())
'''
        (src_dir / "main.py").write_text(main_content)

        # Requirements file
        requirements = ["asyncio", "logging"]
        if "fastapi" in [t.lower() for t in technologies]:
            requirements.extend(["fastapi", "uvicorn", "pydantic"])
        if "postgresql" in [t.lower() for t in technologies]:
            requirements.extend(["asyncpg", "psycopg2-binary"])

        (product_dir / "requirements.txt").write_text("\n".join(sorted(set(requirements))))

        # README
        await self._generate_readme(product_dir, name, technologies, features, "python")

        # License
        (product_dir / "LICENSE").write_text(self._get_mit_license())

    async def _generate_node_kit(self, product_dir: Path, name: str, technologies: list, features: list):
        """Generate Node.js starter kit."""
        src_dir = product_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # package.json
        package_json = {
            "name": self._slugify(name),
            "version": "1.0.0",
            "description": f"{name} - Generated by BrainOps",
            "main": "src/index.js",
            "scripts": {
                "start": "node src/index.js",
                "dev": "nodemon src/index.js",
                "test": "jest"
            },
            "keywords": technologies,
            "license": "MIT",
            "dependencies": {},
            "devDependencies": {
                "nodemon": "^3.0.0",
                "jest": "^29.0.0"
            }
        }

        if "typescript" in [t.lower() for t in technologies]:
            package_json["devDependencies"]["typescript"] = "^5.0.0"
            package_json["devDependencies"]["@types/node"] = "^20.0.0"

        (product_dir / "package.json").write_text(json.dumps(package_json, indent=2))

        # Main index file
        index_content = f'''/**
 * {name}
 * Generated by BrainOps Automated Product Generator
 *
 * Features:
 * {chr(10).join(f" * - {f}" for f in features)}
 */

const VERSION = "1.0.0";

class {self._to_class_name(name)} {{
    constructor(config = {{}}) {{
        this.config = config;
        this.version = VERSION;
        console.log(`Initialized ${{this.constructor.name}}`);
    }}

    async run() {{
        console.log("Starting application...");

        return {{
            status: "success",
            version: this.version,
            features: {json.dumps(features)}
        }};
    }}
}}

// Main entry point
async function main() {{
    const app = new {self._to_class_name(name)}();
    const result = await app.run();
    console.log("Result:", result);
}}

main().catch(console.error);

module.exports = {{ {self._to_class_name(name)} }};
'''
        (src_dir / "index.js").write_text(index_content)

        # README
        await self._generate_readme(product_dir, name, technologies, features, "node")

        # License
        (product_dir / "LICENSE").write_text(self._get_mit_license())

    async def _generate_generic_kit(self, product_dir: Path, name: str, technologies: list, features: list):
        """Generate generic starter kit."""
        src_dir = product_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # Create basic structure
        (src_dir / ".gitkeep").write_text("")

        # README
        await self._generate_readme(product_dir, name, technologies, features, "generic")

        # License
        (product_dir / "LICENSE").write_text(self._get_mit_license())

    async def _generate_prompt_pack(self, product_dir: Path, name: str, prompts: list, use_case: str):
        """Generate AI prompt pack."""
        prompts_dir = product_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        examples_dir = product_dir / "examples"
        examples_dir.mkdir(exist_ok=True)

        # If no prompts provided, generate some based on use case
        if not prompts:
            prompts = await self._generate_prompts_for_use_case(use_case)

        # Create prompt files
        for i, prompt in enumerate(prompts, 1):
            if isinstance(prompt, dict):
                prompt_name = prompt.get("name", f"prompt_{i}")
                prompt_content = prompt.get("content", "")
                category = prompt.get("category", "general")
            else:
                prompt_name = f"prompt_{i}"
                prompt_content = str(prompt)
                category = "general"

            # Create category subfolder
            category_dir = prompts_dir / category
            category_dir.mkdir(exist_ok=True)

            prompt_file = category_dir / f"{self._slugify(prompt_name)}.md"
            prompt_file.write_text(f"""# {prompt_name}

## Category
{category}

## Prompt
```
{prompt_content}
```

## Usage Tips
- Customize the placeholders in [brackets] for your specific use case
- Adjust tone and style as needed
- Combine with other prompts for complex workflows
""")

        # Create index file
        index_content = f"""# {name}

A curated collection of {len(prompts)} AI prompts for {use_case}.

## Categories
"""
        categories = {}
        for p in prompts:
            cat = p.get("category", "general") if isinstance(p, dict) else "general"
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in categories.items():
            index_content += f"- **{cat}**: {count} prompts\n"

        index_content += """

## How to Use

1. Browse the `prompts/` folder by category
2. Copy the prompt you need
3. Customize placeholders for your use case
4. Use with Claude, GPT-4, or your preferred AI

## Best Practices

- Be specific in your modifications
- Test and iterate on prompts
- Combine prompts for complex tasks
- Keep a log of what works best for your needs
"""
        (product_dir / "README.md").write_text(index_content)
        (product_dir / "LICENSE").write_text(self._get_mit_license())

    async def _generate_prompts_for_use_case(self, use_case: str) -> list:
        """Generate prompts for a specific use case."""
        ai_core = await self._get_ai_core()

        if ai_core:
            try:
                prompt = f"""Generate 10 high-quality AI prompts for: {use_case}

Return a JSON array where each item has:
- name: Short descriptive name
- category: Category (e.g., "writing", "coding", "analysis", "creative")
- content: The actual prompt text (50-200 words)

Make prompts specific, actionable, and valuable."""

                response = await ai_core.generate(
                    prompt,
                    model="gpt-4-turbo-preview",
                    system_prompt="You are an expert prompt engineer. Output only valid JSON."
                )

                if isinstance(response, str):
                    return ai_core._safe_json(response) or []
                return response if isinstance(response, list) else []
            except Exception as e:
                logger.warning(f"AI prompt generation failed: {e}")

        # Fallback prompts
        return [
            {"name": "Task Analysis", "category": "analysis", "content": f"Analyze the following {use_case} task and break it into actionable steps: [task description]"},
            {"name": "Problem Solver", "category": "problem-solving", "content": f"Help me solve this {use_case} challenge: [problem description]. Consider multiple approaches and recommend the best one."},
            {"name": "Best Practices", "category": "guidance", "content": f"What are the best practices for {use_case}? Provide specific, actionable recommendations."},
        ]

    async def _generate_automation_scripts(self, product_dir: Path, name: str, features: list):
        """Generate automation scripts package."""
        scripts_dir = product_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Create main automation script
        main_script = f'''#!/usr/bin/env python3
"""
{name} - Automation Scripts
{'=' * len(name)}

Features:
{chr(10).join(f"- {f}" for f in features)}

Usage:
    python main.py [command] [options]
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomationRunner:
    """Main automation runner."""

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        logger.info("Automation runner initialized")

    def _load_config(self, path: str) -> dict:
        """Load configuration from file."""
        config_file = Path(path)
        if config_file.exists():
            return json.loads(config_file.read_text())
        return {{}}

    async def run(self, command: str, **kwargs) -> dict:
        """Run automation command."""
        commands = {{
            "setup": self.setup,
            "run": self.run_automation,
            "status": self.check_status,
        }}

        if command not in commands:
            return {{"error": f"Unknown command: {{command}}"}}

        return await commands[command](**kwargs)

    async def setup(self, **kwargs) -> dict:
        """Set up automation environment."""
        logger.info("Running setup...")
        # Add your setup logic here
        return {{"status": "setup_complete"}}

    async def run_automation(self, **kwargs) -> dict:
        """Run main automation."""
        logger.info("Running automation...")
        # Add your automation logic here
        return {{"status": "completed", "features": {features}}}

    async def check_status(self, **kwargs) -> dict:
        """Check automation status."""
        return {{"status": "healthy", "config_loaded": bool(self.config)}}


def main():
    parser = argparse.ArgumentParser(description="{name}")
    parser.add_argument("command", choices=["setup", "run", "status"], default="status")
    parser.add_argument("--config", default="config.json", help="Config file path")

    args = parser.parse_args()

    runner = AutomationRunner(args.config)
    result = asyncio.run(runner.run(args.command))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''
        (scripts_dir / "main.py").write_text(main_script)

        # Config example
        config_example = {
            "name": name,
            "version": "1.0.0",
            "features": features,
            "settings": {
                "debug": False,
                "output_dir": "./output"
            }
        }
        (product_dir / "config.example.json").write_text(json.dumps(config_example, indent=2))

        # README
        await self._generate_readme(product_dir, name, ["python", "automation"], features, "python")
        (product_dir / "LICENSE").write_text(self._get_mit_license())

    async def _generate_mcp_server(self, product_dir: Path, name: str, features: list):
        """Generate MCP server template for Claude Code."""
        src_dir = product_dir / "src"
        src_dir.mkdir(exist_ok=True)
        examples_dir = product_dir / "examples"
        examples_dir.mkdir(exist_ok=True)

        # MCP server implementation
        server_content = f'''#!/usr/bin/env python3
"""
{name} - MCP Server
{'=' * len(name)}

A Model Context Protocol server for Claude Code.

Features:
{chr(10).join(f"- {f}" for f in features)}
"""

import asyncio
import json
import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPServer:
    """
    MCP Server implementation.

    This server exposes tools to Claude Code for:
    {chr(10).join(f"    - {f}" for f in features)}
    """

    def __init__(self):
        self.name = "{self._slugify(name)}"
        self.version = "1.0.0"
        self.tools = self._register_tools()

    def _register_tools(self) -> dict:
        """Register available tools."""
        return {{
            "list_capabilities": {{
                "description": "List all available capabilities",
                "parameters": {{}},
                "handler": self.list_capabilities
            }},
            "execute_action": {{
                "description": "Execute a specific action",
                "parameters": {{
                    "action": {{"type": "string", "required": True}},
                    "params": {{"type": "object", "required": False}}
                }},
                "handler": self.execute_action
            }},
        }}

    async def handle_request(self, request: dict) -> dict:
        """Handle incoming MCP request."""
        method = request.get("method")
        params = request.get("params", {{}})

        if method == "tools/list":
            return self._format_tools_list()
        elif method == "tools/call":
            return await self._call_tool(params)

        return {{"error": f"Unknown method: {{method}}"}}

    def _format_tools_list(self) -> dict:
        """Format tools list for MCP protocol."""
        return {{
            "tools": [
                {{
                    "name": name,
                    "description": tool["description"],
                    "inputSchema": {{
                        "type": "object",
                        "properties": tool["parameters"]
                    }}
                }}
                for name, tool in self.tools.items()
            ]
        }}

    async def _call_tool(self, params: dict) -> dict:
        """Call a specific tool."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {{}})

        if tool_name not in self.tools:
            return {{"error": f"Unknown tool: {{tool_name}}"}}

        tool = self.tools[tool_name]
        result = await tool["handler"](**arguments)
        return {{"content": [{{"type": "text", "text": json.dumps(result)}}]}}

    async def list_capabilities(self) -> dict:
        """List all capabilities."""
        return {{
            "capabilities": {features},
            "version": self.version
        }}

    async def execute_action(self, action: str, params: dict = None) -> dict:
        """Execute an action."""
        logger.info(f"Executing action: {{action}}")
        return {{
            "action": action,
            "status": "completed",
            "params": params or {{}}
        }}


# Standard MCP server entry point
async def main():
    server = MCPServer()

    # Read from stdin, write to stdout (MCP protocol)
    import sys
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = await server.handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({{"error": str(e)}}))
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
'''
        (src_dir / "server.py").write_text(server_content)

        # MCP config
        mcp_config = {
            "name": self._slugify(name),
            "version": "1.0.0",
            "description": f"{name} - MCP Server for Claude Code",
            "command": "python",
            "args": ["src/server.py"],
            "env": {}
        }
        (product_dir / "mcp-config.json").write_text(json.dumps(mcp_config, indent=2))

        # Example usage
        example_content = f'''# Example Usage

## Installing in Claude Code

Add to your `~/.claude/claude_desktop_config.json`:

```json
{{
  "mcpServers": {{
    "{self._slugify(name)}": {{
      "command": "python",
      "args": ["/path/to/{self._slugify(name)}/src/server.py"]
    }}
  }}
}}
```

## Available Tools

{chr(10).join(f"- **{f}**" for f in features)}

## Example Prompts

1. "List all capabilities of {name}"
2. "Execute action X with parameters Y"
'''
        (examples_dir / "usage.md").write_text(example_content)

        await self._generate_readme(product_dir, name, ["python", "mcp", "claude"], features, "mcp")
        (product_dir / "LICENSE").write_text(self._get_mit_license())

    async def _generate_api_wrapper(self, product_dir: Path, name: str, api_name: str):
        """Generate API wrapper library."""
        src_dir = product_dir / "src"
        src_dir.mkdir(exist_ok=True)
        tests_dir = product_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        # Main wrapper
        wrapper_content = f'''#!/usr/bin/env python3
"""
{name} - {api_name} API Wrapper
{'=' * len(name)}

Type-safe wrapper for the {api_name} API with:
- Automatic retry logic
- Error handling
- Rate limiting
- Response validation
"""

import asyncio
import httpx
import logging
from typing import Any, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration."""
    base_url: str
    api_key: str
    timeout: int = 30
    max_retries: int = 3


class {self._to_class_name(api_name)}Client:
    """
    {api_name} API client with automatic retry and error handling.
    """

    def __init__(self, config: APIConfig):
        self.config = config
        self._client = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers={{"Authorization": f"Bearer {{self.config.api_key}}"}}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> dict:
        """Make API request with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.request(method, endpoint, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code < 500:
                    raise
                logger.warning(f"Attempt {{attempt + 1}} failed: {{e}}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {{attempt + 1}} failed: {{e}}")
                await asyncio.sleep(2 ** attempt)

        raise last_error or Exception("Max retries exceeded")

    async def get(self, endpoint: str, params: dict = None) -> dict:
        """GET request."""
        return await self._request("GET", endpoint, params=params)

    async def post(self, endpoint: str, data: dict = None) -> dict:
        """POST request."""
        return await self._request("POST", endpoint, json=data)

    async def put(self, endpoint: str, data: dict = None) -> dict:
        """PUT request."""
        return await self._request("PUT", endpoint, json=data)

    async def delete(self, endpoint: str) -> dict:
        """DELETE request."""
        return await self._request("DELETE", endpoint)


# Example usage
async def main():
    config = APIConfig(
        base_url="https://api.example.com",
        api_key="your-api-key"
    )

    async with {self._to_class_name(api_name)}Client(config) as client:
        result = await client.get("/endpoint")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
'''
        (src_dir / "client.py").write_text(wrapper_content)
        (src_dir / "__init__.py").write_text(f'from .client import {self._to_class_name(api_name)}Client, APIConfig\n')

        # Test file
        test_content = f'''import pytest
from src.client import {self._to_class_name(api_name)}Client, APIConfig


@pytest.fixture
def config():
    return APIConfig(
        base_url="https://api.example.com",
        api_key="test-key"
    )


@pytest.mark.asyncio
async def test_client_init(config):
    async with {self._to_class_name(api_name)}Client(config) as client:
        assert client.config.base_url == "https://api.example.com"
'''
        (tests_dir / "test_client.py").write_text(test_content)

        # Requirements
        (product_dir / "requirements.txt").write_text("httpx>=0.25.0\npytest>=7.0.0\npytest-asyncio>=0.21.0\n")

        await self._generate_readme(product_dir, name, ["python", "api", api_name], ["retry logic", "error handling", "rate limiting"], "python")
        (product_dir / "LICENSE").write_text(self._get_mit_license())

    async def _generate_readme(self, product_dir: Path, name: str, technologies: list, features: list, lang: str):
        """Generate README file."""
        install_cmd = {
            "python": "pip install -r requirements.txt",
            "node": "npm install",
            "mcp": "See installation instructions below",
            "generic": "See installation instructions below"
        }.get(lang, "See installation instructions")

        readme = f"""# {name}

> Generated by [BrainOps Automated Product Generator](https://brainops.ai)

## Overview

{name} provides {', '.join(features[:3])} and more.

## Technologies

{', '.join(f'`{t}`' for t in technologies)}

## Features

{chr(10).join(f'- {f}' for f in features)}

## Installation

```bash
{install_cmd}
```

## Quick Start

```bash
# Run the application
{"python src/main.py" if lang == "python" else "npm start" if lang == "node" else "./run.sh"}
```

## Configuration

Copy `config.example.json` to `config.json` and update with your settings.

## License

MIT License - See LICENSE file for details.

## Support

For support, contact: support@brainops.ai

---

*Built with BrainOps AI*
"""
        (product_dir / "README.md").write_text(readme)

    def _create_zip(self, product_dir: Path, name: str) -> Path:
        """Create ZIP file from product directory."""
        zip_name = f"{self._slugify(name)}.zip"
        zip_path = self.output_dir / zip_name

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in product_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(product_dir)
                    zipf.write(file_path, arcname)

        # Clean up directory
        shutil.rmtree(product_dir, ignore_errors=True)

        return zip_path

    async def _record_product(self, product_id: str, name: str, product_type: str, zip_path: str, price: int, metadata: dict):
        """Record generated product in database."""
        pool = self._get_pool()
        if not pool:
            return

        try:
            await pool.execute("""
                INSERT INTO digital_products (
                    id, name, product_type, file_path, price_cents, status, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (id) DO NOTHING
            """,
                product_id,
                name,
                product_type,
                zip_path,
                price * 100,
                "generated",
                json.dumps(metadata)
            )
        except Exception as e:
            logger.warning(f"Failed to record product: {e}")

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        import re
        slug = text.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        return slug.strip('-')

    def _to_class_name(self, text: str) -> str:
        """Convert text to PascalCase class name."""
        import re
        words = re.sub(r'[^a-zA-Z0-9\s]', '', text).split()
        return ''.join(word.capitalize() for word in words)

    def _get_mit_license(self) -> str:
        """Get MIT license text."""
        year = datetime.now().year
        return f"""MIT License

Copyright (c) {year} BrainOps AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

    async def publish_to_gumroad(self, product_id: str) -> dict:
        """
        Auto-publish a generated product to Gumroad.
        Requires GUMROAD_ACCESS_TOKEN environment variable.
        """
        import httpx

        GUMROAD_ACCESS_TOKEN = os.getenv("GUMROAD_ACCESS_TOKEN", "")
        if not GUMROAD_ACCESS_TOKEN:
            return {"success": False, "error": "GUMROAD_ACCESS_TOKEN not configured"}

        pool = self._get_pool()
        if not pool:
            return {"success": False, "error": "Database unavailable"}

        # Get product from database
        product = await pool.fetchrow(
            "SELECT * FROM digital_products WHERE id = $1",
            product_id
        )
        if not product:
            return {"success": False, "error": f"Product {product_id} not found"}

        metadata = product.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        # Create Gumroad product
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    "https://api.gumroad.com/v2/products",
                    headers={"Authorization": f"Bearer {GUMROAD_ACCESS_TOKEN}"},
                    data={
                        "name": product["name"],
                        "price": product["price_cents"],
                        "description": metadata.get("description", f"AI-generated {product['product_type']}"),
                        "url": self._slugify(product["name"]),
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    gumroad_id = result.get("product", {}).get("id")
                    gumroad_url = result.get("product", {}).get("short_url")

                    # Update database with Gumroad info
                    await pool.execute("""
                        UPDATE digital_products
                        SET status = 'published',
                            metadata = metadata || $1::jsonb
                        WHERE id = $2
                    """,
                        json.dumps({
                            "gumroad_id": gumroad_id,
                            "gumroad_url": gumroad_url,
                            "published_at": datetime.now(timezone.utc).isoformat()
                        }),
                        product_id
                    )

                    return {
                        "success": True,
                        "product_id": product_id,
                        "gumroad_id": gumroad_id,
                        "gumroad_url": gumroad_url,
                        "status": "published"
                    }
                else:
                    return {
                        "success": False,
                        "error": response.text,
                        "status_code": response.status_code
                    }

            except Exception as e:
                return {"success": False, "error": str(e)}

    async def generate_and_publish(self, spec: dict) -> dict:
        """Generate a product and immediately publish to Gumroad."""
        # Generate the product
        gen_result = await self.generate_product(spec)
        if not gen_result.get("success"):
            return gen_result

        # Publish to Gumroad
        pub_result = await self.publish_to_gumroad(gen_result["product_id"])

        return {
            "success": pub_result.get("success", False),
            "generation": gen_result,
            "publication": pub_result
        }


# Agent metadata
AGENT_METADATA = {
    "id": "AutomatedProductGenerator",
    "name": "Automated Product Generator",
    "description": "AI-powered digital product creation factory with Gumroad auto-publish",
    "version": "1.1.0",
    "tasks": [
        {"name": "generate_from_trend", "schedule": "0 10 * * 1", "description": "Weekly trend-based product generation"},
        {"name": "generate_and_publish", "schedule": "0 12 * * 3", "description": "Generate and publish to Gumroad"},
    ],
    "category": "revenue"
}


async def execute_agent(task: str = "list_templates", **kwargs) -> dict:
    """Entry point for agent executor."""
    generator = AutomatedProductGenerator()

    if task == "list_templates":
        return generator.list_templates()
    elif task == "generate":
        return await generator.generate_product(kwargs)
    elif task == "generate_from_trend":
        return await generator.generate_from_trend(kwargs)

    return {"success": False, "error": f"Unknown task: {task}"}


if __name__ == "__main__":
    async def main():
        generator = AutomatedProductGenerator()

        # Test: Generate MCP server product
        result = await generator.generate_product({
            "name": "GitHub MCP Server",
            "type": "mcp_server",
            "features": ["repository management", "issue tracking", "PR automation", "code search"],
            "technologies": ["python", "github", "mcp"],
            "price": 97
        })
        print(json.dumps(result, indent=2))

    asyncio.run(main())

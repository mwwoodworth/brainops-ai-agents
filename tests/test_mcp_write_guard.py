import os

import pytest

from mcp_integration import MCPClient, MCPServer


@pytest.mark.asyncio
async def test_mcp_blocks_write_tools_by_default(monkeypatch):
    # Ensure all known toggles are disabled for this test.
    for k in (
        "BRAINOPS_OPS_AUTOFIX_ENABLED",
        "BRAINOPS_AUTOFIX_ENABLED",
        "BRAINOPS_MCP_WRITE_ENABLED",
    ):
        monkeypatch.delenv(k, raising=False)

    client = MCPClient(base_url="https://example.invalid", api_key="dummy")
    res = await client.execute_tool(MCPServer.RENDER, "render_restart_service", {"serviceId": "srv_x"})
    assert res.success is False
    assert res.error and "blocked" in res.error.lower()


def test_mcp_tool_intent_classification():
    assert MCPClient._mcp_tool_is_write_intent("render", "render_get_service", {"serviceId": "srv"}) is False
    assert MCPClient._mcp_tool_is_write_intent("render", "render_list_services", {}) is False
    assert MCPClient._mcp_tool_is_write_intent("render", "render_restart_service", {"serviceId": "srv"}) is True

    assert MCPClient._mcp_tool_is_write_intent("vercel", "getProject", {"projectId": "p"}) is False
    assert MCPClient._mcp_tool_is_write_intent("vercel", "createDeployment", {}) is True

    assert MCPClient._mcp_tool_is_write_intent("github", "listRepos", {}) is False
    assert MCPClient._mcp_tool_is_write_intent("github", "createIssue", {}) is True

    assert MCPClient._mcp_tool_is_write_intent("stripe", "getBalance", {}) is False
    assert MCPClient._mcp_tool_is_write_intent("stripe", "createCustomer", {}) is True

    assert MCPClient._mcp_tool_is_write_intent("docker", "listContainers", {}) is False
    assert MCPClient._mcp_tool_is_write_intent("docker", "startContainer", {}) is True


def test_supabase_sql_read_only_heuristics():
    assert MCPClient._supabase_sql_is_read_only("SELECT 1") is True
    assert MCPClient._supabase_sql_is_read_only("/* hi */ SELECT 1") is True
    assert MCPClient._supabase_sql_is_read_only("-- hi\nSELECT 1") is True
    assert MCPClient._supabase_sql_is_read_only("WITH x AS (SELECT 1) SELECT * FROM x") is True

    # Any write-ish verb should flip to non-read-only.
    assert MCPClient._supabase_sql_is_read_only("INSERT INTO t VALUES (1)") is False
    assert MCPClient._supabase_sql_is_read_only("WITH x AS (INSERT INTO t VALUES (1) RETURNING 1) SELECT * FROM x") is False


@pytest.mark.asyncio
async def test_supabase_select_validation_returns_well_formed_result(monkeypatch):
    # This should not raise TypeError due to missing MCPToolResult fields.
    client = MCPClient(base_url="https://example.invalid", api_key="dummy")
    res = await client.supabase_select("not a table", columns="*")
    assert res.success is False
    assert res.server == MCPServer.SUPABASE.value
    assert res.tool == "supabase_select"


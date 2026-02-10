import e2e_system_verification as mod


def test_apply_scope_skip_erp_filters_erp_tests_and_adjusts_chatgpt_agent_ui():
    tests = [
        mod.EndpointTest(name="ERP - Homepage", url=mod.ERP_URL),
        mod.EndpointTest(name="MyRoofGenius - Homepage", url=mod.MRG_URL),
        mod.EndpointTest(
            name="ChatGPT Agent UI - Quick",
            url=f"{mod.BRAINOPS_API_URL}/api/v1/always-know/chatgpt-agent-test",
            expected_fields=["mrg_healthy", "erp_healthy"],
            validation_func="validate_chatgpt_agent_quick",
        ),
    ]

    scoped = mod.E2ESystemVerification._apply_scope(tests, skip_erp=True)
    names = [t.name for t in scoped]
    assert "ERP - Homepage" not in names
    assert "MyRoofGenius - Homepage" in names

    chat = next(t for t in scoped if t.name == "ChatGPT Agent UI - Quick")
    assert "skip_erp=true" in chat.url
    assert chat.validation_func == "validate_chatgpt_agent_quick_non_erp"
    assert chat.expected_fields == ["mrg_healthy", "erp_skipped"]


def test_apply_api_key_override_does_not_override_mcp_key(monkeypatch):
    monkeypatch.setattr(mod, "API_KEY", "brainkey")

    tests = [
        mod.EndpointTest(name="Brain", url="https://example.com/a", headers={"X-API-Key": "brainkey"}),
        mod.EndpointTest(name="MCP", url="https://example.com/b", headers={"X-API-Key": "mcpkey"}),
        mod.EndpointTest(name="NoAuth", url="https://example.com/c", headers={}),
    ]

    out = mod.E2ESystemVerification._apply_api_key_override(tests, api_key_override="override")
    assert out[0].headers["X-API-Key"] == "override"
    assert out[1].headers["X-API-Key"] == "mcpkey"
    assert out[2].headers == {}


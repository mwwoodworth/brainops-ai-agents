from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mcp_server import is_safe_sql_query


def test_select_with_params_ok():
    query = "SELECT id FROM analytics_events WHERE tenant_id = %s LIMIT %s"
    ok, reason = is_safe_sql_query(query, ["tenant-1", 10])
    assert ok, reason


def test_rejects_string_literal():
    ok, _ = is_safe_sql_query("SELECT 'x'", [])
    assert not ok


def test_rejects_numeric_literal():
    ok, _ = is_safe_sql_query("SELECT id FROM analytics_events LIMIT 10", [])
    assert not ok


def test_rejects_multiple_statements():
    ok, _ = is_safe_sql_query("SELECT id FROM analytics_events; SELECT 1", [])
    assert not ok


def test_rejects_subquery():
    ok, _ = is_safe_sql_query("SELECT * FROM (SELECT id FROM analytics_events) t", [])
    assert not ok


def test_rejects_placeholder_mismatch():
    ok, _ = is_safe_sql_query("SELECT id FROM analytics_events WHERE tenant_id = %s", [])
    assert not ok

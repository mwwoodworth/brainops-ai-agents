#!/usr/bin/env python3
"""Static DB safety audit for tenant scoping and SQL parameterization."""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / f"deep_audit_{datetime.now(UTC).date().isoformat()}"


@dataclass
class DbFileAudit:
    file: str
    uses_tenant_pool: bool
    uses_psycopg_connect: bool
    has_set_local_tenant: bool
    execute_with_dynamic_sql: int
    execute_calls: int
    status: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_python_files() -> list[Path]:
    files = sorted(ROOT.rglob("*.py"))
    return [
        p
        for p in files
        if ".venv" not in p.parts and p != Path(__file__).resolve() and "reports" not in p.parts
    ]


def _is_dynamic_sql_arg(arg: ast.AST) -> bool:
    if isinstance(arg, (ast.JoinedStr, ast.BinOp)):
        return True
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
        # sql.SQL(...).format(...)
        if arg.func.attr == "format":
            return True
    return False


def _audit_file(path: Path) -> DbFileAudit:
    rel = str(path.relative_to(ROOT))
    source = _read(path)
    uses_tenant_pool = "get_tenant_pool(" in source or "TenantScopedPool" in source
    uses_psycopg_connect = "psycopg2.connect(" in source
    has_set_local_tenant = (
        "set_config('app.current_tenant_id'" in source or "SET LOCAL app.current_tenant_id" in source
    )

    execute_with_dynamic_sql = 0
    execute_calls = 0
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return DbFileAudit(
            file=rel,
            uses_tenant_pool=uses_tenant_pool,
            uses_psycopg_connect=uses_psycopg_connect,
            has_set_local_tenant=has_set_local_tenant,
            execute_with_dynamic_sql=0,
            execute_calls=0,
            status="PARSE_ERROR",
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "execute":
                execute_calls += 1
                if node.args and _is_dynamic_sql_arg(node.args[0]):
                    execute_with_dynamic_sql += 1

    if uses_tenant_pool and execute_with_dynamic_sql == 0 and not uses_psycopg_connect:
        status = "TENANT_SAFE"
    elif uses_psycopg_connect or execute_with_dynamic_sql > 0:
        status = "REVIEW_REQUIRED"
    else:
        status = "PARTIAL"

    return DbFileAudit(
        file=rel,
        uses_tenant_pool=uses_tenant_pool,
        uses_psycopg_connect=uses_psycopg_connect,
        has_set_local_tenant=has_set_local_tenant,
        execute_with_dynamic_sql=execute_with_dynamic_sql,
        execute_calls=execute_calls,
        status=status,
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audits = [_audit_file(path) for path in _iter_python_files()]
    rows = [a.__dict__ for a in audits if a.execute_calls > 0 or a.uses_psycopg_connect or a.uses_tenant_pool]
    rows.sort(key=lambda r: (r["status"], r["file"]))

    csv_path = REPORT_DIR / "db_safety_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "uses_tenant_pool",
                "uses_psycopg_connect",
                "has_set_local_tenant",
                "execute_with_dynamic_sql",
                "execute_calls",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["status"] for row in rows)
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "audited_files": len(rows),
        "status_counts": dict(counts),
        "review_required_files": [
            row["file"] for row in rows if row["status"] == "REVIEW_REQUIRED"
        ],
        "output_csv": str(csv_path.relative_to(ROOT)),
    }
    (REPORT_DIR / "db_safety_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

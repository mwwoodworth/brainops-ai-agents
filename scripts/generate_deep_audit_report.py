#!/usr/bin/env python3
"""Generate deep audit artifacts for BrainOps AI Agents."""

from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "app.py"
REPORT_DIR = ROOT / "reports" / f"deep_audit_{datetime.now(UTC).date().isoformat()}"

ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}
STUB_MARKERS = (
    "stub",
    "placeholder",
    "not implemented",
    "todo",
    "returns 404",
    "deprecated",
    "skeleton",
)
MEMORY_MARKERS = (
    "app.state.memory",
    "get_memory_manager",
    "memory.store",
    "memory.recall",
    "brain.store",
    "brain.recall",
    "UnifiedMemoryManager",
    "live_brain",
    "store_async",
    "recall(",
)

SYSTEM_FLAG_TO_FILE = {
    "AUREA_AVAILABLE": "aurea_orchestrator.py",
    "SELF_HEALING_AVAILABLE": "self_healing_recovery.py",
    "MEMORY_AVAILABLE": "unified_memory_manager.py",
    "EMBEDDED_MEMORY_AVAILABLE": "embedded_memory_system.py",
    "TRAINING_AVAILABLE": "ai_training_pipeline.py",
    "LEARNING_AVAILABLE": "notebook_lm_plus.py",
    "SCHEDULER_AVAILABLE": "agent_scheduler.py",
    "AI_AVAILABLE": "ai_core.py",
    "SYSTEM_IMPROVEMENT_AVAILABLE": "system_improvement_agent.py",
    "DEVOPS_AGENT_AVAILABLE": "devops_optimization_agent.py",
    "CODE_QUALITY_AVAILABLE": "code_quality_agent.py",
    "CUSTOMER_SUCCESS_AVAILABLE": "customer_success_agent.py",
    "COMPETITIVE_INTEL_AVAILABLE": "competitive_intelligence_agent.py",
    "VISION_ALIGNMENT_AVAILABLE": "vision_alignment_agent.py",
    "RECONCILER_AVAILABLE": "self_healing_reconciler.py",
    "BLEEDING_EDGE_AVAILABLE": "api/bleeding_edge.py",
    "AUTONOMOUS_RESOLVER_AVAILABLE": "api/autonomous_resolver.py",
    "MEMORY_ENFORCEMENT_AVAILABLE": "api/memory_enforcement_api.py",
    "MEMORY_HYGIENE_AVAILABLE": "api/memory_hygiene_api.py",
    "WORKFLOWS_AVAILABLE": "api/workflows.py",
}


@dataclass
class EndpointRecord:
    method: str
    path: str
    function: str
    line: int
    uses_persistent_memory: bool
    status: str
    notes: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _is_stub_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in STUB_MARKERS)


def _route_path_from_decorator(dec: ast.Call) -> str:
    if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
        return dec.args[0].value
    return "<dynamic>"


def _analyze_endpoint_node(source_lines: list[str], node: ast.AST) -> list[EndpointRecord]:
    records: list[EndpointRecord] = []
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return records
    function_source = "\n".join(source_lines[node.lineno - 1 : node.end_lineno or node.lineno])
    uses_memory = any(marker in function_source for marker in MEMORY_MARKERS)
    lowered = function_source.lower()
    is_stub = _is_stub_text(function_source) or "raise runtimeerror(\"agent" in lowered
    if "not available" in lowered and not uses_memory:
        status = "LIKELY_DEGRADED"
        notes = "Dependency-gated"
    elif is_stub and not uses_memory:
        status = "LIKELY_STUB"
        notes = "Stub markers detected"
    else:
        status = "LIKELY_REAL"
        notes = "Has implementation logic"
    for dec in node.decorator_list:
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
            if (
                isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "app"
                and dec.func.attr in ROUTE_METHODS
            ):
                records.append(
                    EndpointRecord(
                        method=dec.func.attr.upper(),
                        path=_route_path_from_decorator(dec),
                        function=node.name,
                        line=node.lineno,
                        uses_persistent_memory=uses_memory,
                        status=status,
                        notes=notes,
                    )
                )
    return records


def audit_endpoints() -> tuple[list[EndpointRecord], list[dict[str, Any]]]:
    source = _read(APP_FILE)
    tree = ast.parse(source, filename=str(APP_FILE))
    lines = source.splitlines()
    endpoints: list[EndpointRecord] = []
    for node in tree.body:
        endpoints.extend(_analyze_endpoint_node(lines, node))

    by_signature: dict[tuple[str, str], list[EndpointRecord]] = defaultdict(list)
    for rec in endpoints:
        by_signature[(rec.method, rec.path)].append(rec)

    duplicates: list[dict[str, Any]] = []
    for (method, path), rows in sorted(by_signature.items()):
        if len(rows) > 1:
            duplicates.append(
                {
                    "method": method,
                    "path": path,
                    "count": len(rows),
                    "functions": [r.function for r in rows],
                    "lines": [r.line for r in rows],
                }
            )
    return endpoints, duplicates


def _extract_active_system_specs() -> list[dict[str, str]]:
    source = _read(APP_FILE)
    tree = ast.parse(source, filename=str(APP_FILE))
    specs: list[dict[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_collect_active_systems":
            for stmt in node.body:
                if not isinstance(stmt, ast.If):
                    continue
                if not stmt.body:
                    continue
                call = stmt.body[0]
                if (
                    isinstance(call, ast.Expr)
                    and isinstance(call.value, ast.Call)
                    and isinstance(call.value.func, ast.Attribute)
                    and isinstance(call.value.func.value, ast.Name)
                    and call.value.func.value.id == "active"
                    and call.value.func.attr == "append"
                    and call.value.args
                    and isinstance(call.value.args[0], ast.Constant)
                    and isinstance(call.value.args[0].value, str)
                ):
                    cond = ast.get_source_segment(source, stmt.test) or ""
                    flag_match = re.search(r"([A-Z_]+_AVAILABLE)", cond)
                    specs.append(
                        {
                            "system_name": call.value.args[0].value,
                            "flag": flag_match.group(1) if flag_match else "<unknown>",
                            "condition": cond,
                        }
                    )
            break
    return specs


def _file_stub_score(path: Path) -> int:
    text = _read(path)
    score = 0
    lowered = text.lower()
    if _is_stub_text(text):
        score += 2
    if " pass\n" in text and len(text.splitlines()) < 200:
        score += 1
    if "guardrail-first stub" in lowered:
        score += 3
    return score


def audit_active_systems() -> list[dict[str, Any]]:
    systems = _extract_active_system_specs()
    rows: list[dict[str, Any]] = []
    for system in systems:
        flag = system["flag"]
        file_rel = SYSTEM_FLAG_TO_FILE.get(flag)
        path = ROOT / file_rel if file_rel else None
        exists = bool(path and path.exists())
        stub_score = _file_stub_score(path) if exists else 0
        if not exists:
            status = "DEAD"
        elif stub_score >= 3:
            status = "ACTIVE_STUB"
        else:
            status = "ACTIVE_REAL"
        rows.append(
            {
                "system_name": system["system_name"],
                "flag": flag,
                "implementation_file": file_rel or "<unknown>",
                "exists": exists,
                "stub_score": stub_score,
                "status": status,
                "condition": system["condition"],
            }
        )
    return rows


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT)
    return ".".join(rel.with_suffix("").parts)


def _import_targets(node: ast.AST) -> list[str]:
    targets: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            targets.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            targets.append(node.module)
    return targets


def audit_python_files() -> list[dict[str, Any]]:
    py_files = sorted(ROOT.rglob("*.py"))
    py_files = [
        p
        for p in py_files
        if ".venv" not in p.parts and p != Path(__file__).resolve()
    ]
    module_to_path = {_module_name(path): path for path in py_files}
    inbound_refs: Counter[str] = Counter()
    app_imports: set[str] = set()

    for path in py_files:
        try:
            tree = ast.parse(_read(path), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for target in _import_targets(node):
                    for module_name, module_path in module_to_path.items():
                        if target == module_name or target.startswith(f"{module_name}."):
                            rel = str(module_path.relative_to(ROOT))
                            inbound_refs[rel] += 1
                            if path == APP_FILE:
                                app_imports.add(rel)

    rows: list[dict[str, Any]] = []
    for path in py_files:
        rel = str(path.relative_to(ROOT))
        ref_count = int(inbound_refs.get(rel, 0))
        stub_score = _file_stub_score(path)
        is_archive = "_archive" in path.parts
        is_deprecated = path.name.startswith("_deprecated")
        if is_archive or is_deprecated:
            status = "DEAD"
        elif stub_score >= 3:
            status = "ACTIVE_STUB"
        elif ref_count == 0 and rel not in {"app.py", "run.py"} and "tests/" not in rel:
            status = "DEAD"
        else:
            status = "ACTIVE_REAL"
        rows.append(
            {
                "file": rel,
                "status": status,
                "inbound_references": ref_count,
                "referenced_by_app": rel in app_imports,
                "stub_score": stub_score,
                "loc": len(_read(path).splitlines()),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    endpoints, duplicates = audit_endpoints()
    endpoint_rows = [
        {
            "method": e.method,
            "path": e.path,
            "function": e.function,
            "line": e.line,
            "uses_persistent_memory": e.uses_persistent_memory,
            "status": e.status,
            "notes": e.notes,
        }
        for e in endpoints
    ]
    _write_csv(
        REPORT_DIR / "endpoint_audit.csv",
        endpoint_rows,
        [
            "method",
            "path",
            "function",
            "line",
            "uses_persistent_memory",
            "status",
            "notes",
        ],
    )
    (REPORT_DIR / "endpoint_audit.json").write_text(
        json.dumps({"endpoints": endpoint_rows, "duplicates": duplicates}, indent=2),
        encoding="utf-8",
    )

    systems = audit_active_systems()
    _write_csv(
        REPORT_DIR / "active_systems_audit.csv",
        systems,
        [
            "system_name",
            "flag",
            "implementation_file",
            "exists",
            "stub_score",
            "status",
            "condition",
        ],
    )
    (REPORT_DIR / "active_systems_audit.json").write_text(
        json.dumps(systems, indent=2), encoding="utf-8"
    )

    file_rows = audit_python_files()
    _write_csv(
        REPORT_DIR / "python_file_status.csv",
        file_rows,
        ["file", "status", "inbound_references", "referenced_by_app", "stub_score", "loc"],
    )

    file_status_counts = Counter(row["status"] for row in file_rows)
    endpoint_status_counts = Counter(row["status"] for row in endpoint_rows)
    system_status_counts = Counter(row["status"] for row in systems)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "app_file": str(APP_FILE.relative_to(ROOT)),
        "endpoint_count": len(endpoint_rows),
        "endpoint_status_counts": dict(endpoint_status_counts),
        "duplicate_route_count": len(duplicates),
        "active_system_count": len(systems),
        "active_system_status_counts": dict(system_status_counts),
        "python_file_count": len(file_rows),
        "python_file_status_counts": dict(file_status_counts),
        "report_dir": str(REPORT_DIR.relative_to(ROOT)),
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Deep Audit Summary",
        "",
        f"- Generated (UTC): `{summary['generated_at_utc']}`",
        f"- Endpoints audited: `{summary['endpoint_count']}`",
        f"- Duplicate app routes: `{summary['duplicate_route_count']}`",
        f"- Active systems audited: `{summary['active_system_count']}`",
        f"- Python files audited: `{summary['python_file_count']}`",
        "",
        "## Status Counts",
        "",
        f"- Endpoints: `{dict(endpoint_status_counts)}`",
        f"- Active systems: `{dict(system_status_counts)}`",
        f"- Python files: `{dict(file_status_counts)}`",
        "",
        "## Artifacts",
        "",
        f"- `reports/{REPORT_DIR.name}/endpoint_audit.csv`",
        f"- `reports/{REPORT_DIR.name}/active_systems_audit.csv`",
        f"- `reports/{REPORT_DIR.name}/python_file_status.csv`",
    ]
    (REPORT_DIR / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

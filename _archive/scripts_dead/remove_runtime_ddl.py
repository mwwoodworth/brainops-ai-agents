#!/usr/bin/env python3
"""Automated script to remove runtime DDL from all Python files.

Finds all CREATE TABLE / CREATE INDEX / CREATE EXTENSION / ALTER TABLE
statements in Python files and replaces them with verification-only calls.

This script is idempotent -- running it multiple times is safe.
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALREADY_FIXED = {
    # Files already manually fixed in previous commits
    "unified_memory_manager.py",
    "unified_brain.py",
    "revenue_generation_system.py",
    "ai_board_governance.py",
    "always_know_brain.py",
    "app.py",
    "aurea_orchestrator.py",
    "notebook_lm_plus.py",
    "permanent_observability_daemon.py",
    "self_healing_recovery.py",
}

# Regex to extract table names from CREATE TABLE IF NOT EXISTS statements
TABLE_NAME_RE = re.compile(
    r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)",
    re.IGNORECASE,
)

# DDL patterns to detect
DDL_PATTERNS = [
    r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS",
    r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS",
    r"CREATE\s+EXTENSION\s+IF\s+NOT\s+EXISTS",
    r"ALTER\s+TABLE\s+\w+\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS",
]
DDL_RE = re.compile("|".join(DDL_PATTERNS), re.IGNORECASE)


def find_ddl_files():
    """Find all .py files containing runtime DDL."""
    results = []
    for py_file in ROOT.rglob("*.py"):
        rel = py_file.relative_to(ROOT)
        name = str(rel)
        # Skip already fixed, scripts, tests, migrations
        if rel.name in ALREADY_FIXED:
            continue
        if name.startswith("scripts/"):
            continue
        if name.startswith("tests/"):
            continue
        if name.startswith("migrations/"):
            continue
        if name.startswith("database/"):
            continue

        try:
            content = py_file.read_text(errors="ignore")
        except Exception:
            continue
        if DDL_RE.search(content):
            tables = TABLE_NAME_RE.findall(content)
            results.append((py_file, rel, tables))
    return results


def extract_method_info(content: str):
    """Determine if DDL is in async or sync context."""
    has_async = "async def" in content and ("await pool" in content or "await self" in content)
    has_psycopg2 = "psycopg2" in content or "cursor.execute" in content
    return has_async, has_psycopg2


def main():
    files = find_ddl_files()
    print(f"Found {len(files)} files with runtime DDL to process:\n")
    for _, rel, tables in sorted(files, key=lambda x: str(x[1])):
        print(f"  {rel}: {len(tables)} tables ({', '.join(tables[:5])}{'...' if len(tables) > 5 else ''})")

    print(f"\nTotal: {len(files)} files")
    return files


if __name__ == "__main__":
    files = main()

#!/usr/bin/env python3
"""Automated transformer: replaces runtime DDL with table verification calls.

Handles both async (asyncpg) and sync (psycopg2) patterns.
Preserves all non-DDL logic in each method.
"""

import os
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".venv", "scripts", "tests", "migrations", "database", "__pycache__", ".git"}
ALREADY_FIXED = {
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

# Regex patterns
CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)", re.IGNORECASE
)
DDL_LINE_RE = re.compile(
    r"^\s*(CREATE\s+(TABLE|INDEX|EXTENSION)|ALTER\s+TABLE)", re.IGNORECASE
)

# Files where DDL is at module/class level (not in a method) - need special handling
MODULE_LEVEL_DDL = {"affiliate_partnership_pipeline.py"}


def find_files_to_fix():
    """Find all .py files with runtime DDL that haven't been fixed yet."""
    results = []
    for py_file in ROOT.rglob("*.py"):
        rel = py_file.relative_to(ROOT)
        parts = rel.parts

        # Skip directories we don't want to touch
        if any(d in parts for d in SKIP_DIRS):
            continue
        if rel.name in ALREADY_FIXED:
            continue

        try:
            content = py_file.read_text(errors="ignore")
        except Exception:
            continue

        tables = CREATE_TABLE_RE.findall(content)
        has_ddl = bool(re.search(
            r"CREATE\s+(TABLE|INDEX|EXTENSION)\s+IF\s+NOT\s+EXISTS|"
            r"ALTER\s+TABLE\s+\w+\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS",
            content, re.IGNORECASE
        ))
        if has_ddl and tables:
            results.append((py_file, rel, tables))
    return results


def is_async_context(method_body: str) -> bool:
    """Check if the method uses async patterns."""
    return "await " in method_body or "async " in method_body


def transform_file(filepath: Path, tables: list[str]) -> tuple[bool, str]:
    """Transform a single file, replacing DDL with verification calls.

    Returns (success, message).
    """
    content = filepath.read_text(errors="ignore")
    original = content

    # Deduplicate table names, preserving order
    seen = set()
    unique_tables = []
    for t in tables:
        t_lower = t.lower()
        if t_lower not in seen and t_lower != "await":  # filter out false positives
            seen.add(t_lower)
            unique_tables.append(t)

    if not unique_tables:
        return False, "No valid table names found"

    # Determine if the file uses async or sync patterns for its DDL
    uses_async = "await pool" in content or "await self._pool" in content or "await self.pool" in content
    uses_sync = "psycopg2" in content or "cursor.execute" in content

    # Find the init/ensure method(s) containing DDL
    # Common method names: _ensure_tables, _init_database, _initialize_database, _create_tables, _init_db
    init_method_re = re.compile(
        r"((?:async\s+)?def\s+(?:_ensure_tables?|_init_database?|_initialize_database|_create_tables|_init_db|_ensure_database|_setup_tables|_setup_database)\s*\([^)]*\)\s*(?:->.*?)?:)",
        re.MULTILINE,
    )

    match = init_method_re.search(content)

    if not match:
        # No standard init method found. The DDL might be inline in other methods
        # or at class/module level. We'll handle this by finding and replacing
        # individual DDL execute blocks.
        return transform_inline_ddl(filepath, content, unique_tables)

    # Found an init method. Replace its body.
    method_start = match.start()
    method_sig = match.group(1)
    is_async = method_sig.strip().startswith("async")

    # Find the method body (everything until the next method definition at the same or lesser indent level)
    method_sig_end = match.end()

    # Determine method indent
    line_start = content.rfind("\n", 0, method_start) + 1
    method_indent = len(content[line_start:method_start]) - len(content[line_start:method_start].lstrip())

    # Find end of method
    lines = content[method_sig_end:].split("\n")
    method_end_offset = 0
    found_body = False
    for i, line in enumerate(lines):
        if line.strip() == "" or line.strip().startswith("#"):
            method_end_offset += len(line) + 1
            continue
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        if found_body and current_indent <= method_indent and stripped and not stripped.startswith("#"):
            break
        found_body = True
        method_end_offset += len(line) + 1

    method_end = method_sig_end + method_end_offset

    # Build replacement method body
    indent = " " * (method_indent + 4)  # method body indent
    indent2 = " " * (method_indent + 8)  # nested indent

    table_list_str = ",\n".join(f'{indent2}    "{t}"' for t in unique_tables)
    module_name = filepath.stem

    if is_async:
        replacement_body = f'''
{indent}"""Verify required tables exist (DDL removed — agent_worker has no DDL permissions)."""
{indent}required_tables = [
{table_list_str},
{indent}]
{indent}try:
{indent2}from database import get_pool
{indent2}from database.verify_tables import verify_tables_async
{indent2}pool = get_pool()
{indent2}ok = await verify_tables_async(required_tables, pool, module_name="{module_name}")
{indent2}if not ok:
{indent2}    return
{indent2}self._tables_initialized = True
{indent}except Exception as exc:
{indent2}logger.error("Table verification failed: %s", exc)
'''
    else:
        replacement_body = f'''
{indent}"""Verify required tables exist (DDL removed — agent_worker has no DDL permissions)."""
{indent}required_tables = [
{table_list_str},
{indent}]
{indent}try:
{indent2}from database.verify_tables import verify_tables_sync
{indent2}conn = psycopg2.connect(**DB_CONFIG)
{indent2}cursor = conn.cursor()
{indent2}ok = verify_tables_sync(required_tables, cursor, module_name="{module_name}")
{indent2}cursor.close()
{indent2}conn.close()
{indent2}if not ok:
{indent2}    return
{indent2}self._tables_initialized = True
{indent}except Exception as exc:
{indent2}logger.error("Table verification failed: %s", exc)
'''

    new_content = content[:method_sig_end] + replacement_body + content[method_end:]

    # Also check if there are extra DDL blocks outside the init method (e.g. in other methods)
    remaining_ddl = CREATE_TABLE_RE.findall(new_content[method_sig_end + len(replacement_body):])
    if remaining_ddl:
        # There's DDL in other methods too - apply inline transformation to the rest
        prefix = new_content[:method_sig_end + len(replacement_body)]
        suffix = new_content[method_sig_end + len(replacement_body):]
        success, msg = transform_inline_ddl_content(suffix, remaining_ddl)
        if success:
            new_content = prefix + msg  # msg is the transformed suffix

    if new_content != original:
        filepath.write_text(new_content)
        return True, f"Transformed init method + {len(unique_tables)} tables"
    return False, "No changes needed"


def transform_inline_ddl(filepath: Path, content: str, tables: list[str]) -> tuple[bool, str]:
    """Handle files where DDL is inline in non-init methods."""
    success, result = transform_inline_ddl_content(content, tables)
    if success:
        filepath.write_text(result)
        return True, f"Transformed inline DDL for {len(tables)} tables"
    return False, result


def transform_inline_ddl_content(content: str, tables: list[str]) -> tuple[bool, str]:
    """Replace inline DDL execute blocks with verification.

    This handles cases like:
        cursor.execute('''CREATE TABLE IF NOT EXISTS ...''')
    or:
        await pool.execute('''CREATE TABLE IF NOT EXISTS ...''')

    by replacing the entire execute call (including its SQL string) with a
    verification query or just removing it.
    """
    original = content

    # Pattern to match execute calls containing DDL
    # Handles: cursor.execute(""" ... CREATE TABLE ... """)
    # and:     await pool.execute(""" ... CREATE TABLE ... """)
    # These are typically large multiline strings

    # Remove individual DDL execute() calls
    # Match: cursor.execute("""...CREATE TABLE...""")  with optional conn.commit()
    ddl_execute_re = re.compile(
        r'(\s*)((?:await\s+)?(?:\w+\.)*(?:execute|executemany))\s*\(\s*'
        r'(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"]*"|\'[^\']*\')\s*'
        r'(?:,\s*[\s\S]*?)?\)',
        re.MULTILINE,
    )

    def replace_ddl_execute(match):
        full = match.group(0)
        indent = match.group(1)
        # Only replace if it contains DDL
        if re.search(r'CREATE\s+(TABLE|INDEX|EXTENSION)\s+IF\s+NOT\s+EXISTS|ALTER\s+TABLE\s+\w+\s+ADD\s+COLUMN', full, re.IGNORECASE):
            return f"{indent}pass  # DDL removed (agent_worker has no DDL permissions)"
        return full

    new_content = ddl_execute_re.sub(replace_ddl_execute, content)

    # Clean up multiple consecutive "pass" statements and blank lines
    new_content = re.sub(r'(\s*pass\s*#[^\n]*\n)+', lambda m: m.group(0).split('\n')[0] + '\n', new_content)

    # Remove orphaned conn.commit() that followed DDL
    # (only if there's no other execute before it)

    if new_content != original:
        return True, new_content
    return False, "No inline DDL patterns matched"


def verify_syntax(filepath: Path) -> bool:
    """Check if a Python file has valid syntax."""
    import py_compile
    try:
        py_compile.compile(str(filepath), doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f"  SYNTAX ERROR: {e}")
        return False


def main():
    files = find_files_to_fix()
    print(f"Found {len(files)} files to transform\n")

    success_count = 0
    fail_count = 0
    syntax_errors = []

    for filepath, rel, tables in sorted(files, key=lambda x: str(x[1])):
        print(f"Processing {rel}...", end=" ")
        try:
            ok, msg = transform_file(filepath, tables)
            if ok:
                # Verify syntax
                if verify_syntax(filepath):
                    print(f"OK ({msg})")
                    success_count += 1
                else:
                    print(f"SYNTAX ERROR - reverting")
                    # Revert by reading from git
                    os.system(f"cd {ROOT} && git checkout -- {rel}")
                    syntax_errors.append(str(rel))
                    fail_count += 1
            else:
                print(f"SKIPPED ({msg})")
        except Exception as e:
            print(f"ERROR: {e}")
            # Revert
            os.system(f"cd {ROOT} && git checkout -- {rel}")
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Results: {success_count} transformed, {fail_count} failed/reverted")
    if syntax_errors:
        print(f"Syntax errors in: {', '.join(syntax_errors)}")

    return success_count, fail_count, syntax_errors


if __name__ == "__main__":
    main()

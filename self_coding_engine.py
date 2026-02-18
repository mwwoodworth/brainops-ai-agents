#!/usr/bin/env python3
"""
Self-Coding Engine (Phase 1: Close the "Action Gap")
====================================================
Turns approved improvement proposals into real code changes safely:
- Creates a git branch
- Generates a patch via RealAICore (LLM)
- Runs repo verification commands
- Pushes the branch and opens a PR (optional)

Safety rails (defaults are SAFE / OFF):
- ENABLE_SELF_CODING_ENGINE=true is required to do anything
- No auto-merge: PRs must be reviewed/approved
- Repo allowlist via SELF_CODING_ALLOWED_REPO_PREFIXES
- Risk gating: high/critical blocked unless SELF_CODING_ALLOW_HIGH_RISK=true
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_command(raw: str | None) -> list[str]:
    if not raw:
        return []
    if "," in raw:
        return _split_csv(raw)
    try:
        return [part for part in shlex.split(raw) if part.strip()]
    except ValueError:
        # Fallback: naive split
        return [part for part in raw.split() if part.strip()]


def _slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip()).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:max_len] or "change"


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_seconds: float = 60.0,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=check,
    )


def _git(repo_path: Path, args: list[str], *, timeout_seconds: float = 60.0) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=repo_path, timeout_seconds=timeout_seconds, check=False)


def _ensure_repo_allowed(repo_path: Path) -> tuple[bool, str]:
    allowed_prefixes = _split_csv(os.getenv("SELF_CODING_ALLOWED_REPO_PREFIXES"))
    if not allowed_prefixes:
        # Safe default: only allow this repo.
        allowed_prefixes = [str(Path(__file__).resolve().parent)]

    repo_str = str(repo_path.resolve())
    if not any(repo_str.startswith(prefix.rstrip("/") + "/") or repo_str == prefix.rstrip("/") for prefix in allowed_prefixes):
        return False, f"repo_path_not_allowed: {repo_str}"
    return True, "ok"


def _ensure_clean_worktree(repo_path: Path) -> tuple[bool, str]:
    status = _git(repo_path, ["status", "--porcelain"])
    if status.returncode != 0:
        return False, f"git_status_failed: {status.stderr.strip() or status.stdout.strip()}"
    if (status.stdout or "").strip():
        return False, "worktree_not_clean"
    return True, "ok"


def _ensure_git_identity(repo_path: Path) -> None:
    # Git commits can fail in automation if user identity isn't configured.
    name = _git(repo_path, ["config", "--get", "user.name"])
    email = _git(repo_path, ["config", "--get", "user.email"])
    if name.returncode != 0 or not (name.stdout or "").strip():
        _git(repo_path, ["config", "user.name", os.getenv("SELF_CODING_GIT_NAME", "BrainOps SelfBuilder")])
    if email.returncode != 0 or not (email.stdout or "").strip():
        _git(repo_path, ["config", "user.email", os.getenv("SELF_CODING_GIT_EMAIL", "selfbuilder@brainops.local")])


def _extract_unified_diff(text: str) -> str | None:
    if not text:
        return None
    # Prefer a real unified diff.
    if "diff --git " in text:
        return text[text.index("diff --git ") :].strip()
    # Some models wrap in fences; best-effort extraction.
    match = re.search(r"```diff\\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        inner = match.group(1)
        if "diff --git " in inner:
            return inner[inner.index("diff --git ") :].strip()
        return inner.strip()
    return None


def _diff_file_paths(unified_diff: str) -> list[str]:
    paths: list[str] = []
    if not unified_diff:
        return paths
    for line in unified_diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        m = re.match(r"^diff --git a/(.+?) b/(.+?)\\s*$", line)
        if not m:
            continue
        a_path = m.group(1).strip()
        b_path = m.group(2).strip()
        # Prefer the post-image path.
        path = b_path or a_path
        if path:
            paths.append(path)
    # De-dup preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


def _diff_allowed(unified_diff: str) -> tuple[bool, str]:
    allowed_suffixes = _split_csv(os.getenv("SELF_CODING_ALLOWED_FILE_SUFFIXES"))
    if not allowed_suffixes:
        allowed_suffixes = [".py", ".md", ".txt", ".json", ".yml", ".yaml"]

    blocked_tokens = {"..", "\x00"}
    blocked_paths = {".env", "BrainOps.env", "_secure/"}

    paths = _diff_file_paths(unified_diff)
    if not paths:
        return False, "diff_has_no_files"

    for path in paths:
        if any(tok in path for tok in blocked_tokens):
            return False, f"diff_path_blocked:{path}"
        lowered = path.lower()
        if any(bp.lower() in lowered for bp in blocked_paths):
            return False, f"diff_path_blocked:{path}"
        if not any(lowered.endswith(sfx.lower()) for sfx in allowed_suffixes):
            return False, f"diff_suffix_not_allowed:{path}"

    return True, "ok"


@dataclass(frozen=True)
class PatchContext:
    repo_path: Path
    proposal_id: str
    title: str
    description: str
    improvement_type: str
    risk_level: str
    implementation_steps: list[str]
    success_criteria: list[str]
    codebase_prompt_context: str = ""


def _build_patch_prompt(ctx: PatchContext, file_snippets: list[dict[str, Any]]) -> str:
    snippets_text = []
    for snippet in file_snippets:
        path = snippet.get("path", "unknown")
        start = snippet.get("start_line", 1)
        end = snippet.get("end_line", 1)
        content = snippet.get("content", "")
        snippets_text.append(
            f"### {path} (lines {start}-{end})\n```python\n{content}\n```"
        )

    return "\n".join(
        [
            "# Improvement Proposal",
            f"- id: {ctx.proposal_id}",
            f"- title: {ctx.title}",
            f"- improvement_type: {ctx.improvement_type}",
            f"- risk_level: {ctx.risk_level}",
            "",
            ctx.description or "",
            "",
            "## Implementation Steps",
            "\n".join([f"- {step}" for step in (ctx.implementation_steps or [])]),
            "",
            "## Success Criteria",
            "\n".join([f"- {crit}" for crit in (ctx.success_criteria or [])]),
            "",
            "## Codebase Context (from graph)",
            (ctx.codebase_prompt_context or "").strip(),
            "",
            "## Relevant Code Snippets",
            "\n\n".join(snippets_text) if snippets_text else "(none provided)",
            "",
            "## Task",
            "Generate a minimal, correct patch that moves this proposal forward.",
            "Constraints:",
            "- Output ONLY a unified diff starting with 'diff --git'.",
            "- Touch as few files as possible.",
            "- Do not add new dependencies unless absolutely necessary.",
            "- Do not include secrets, API keys, or credentials in code or comments.",
            "",
        ]
    ).strip()


def _read_snippet(repo_path: Path, rel_path: str, *, around_line: int | None, radius: int = 120) -> dict[str, Any] | None:
    rel = rel_path.strip().lstrip("/")
    path = repo_path / rel
    if not path.exists() or not path.is_file():
        return None

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    if not lines:
        return {"path": rel, "start_line": 1, "end_line": 1, "content": ""}

    if around_line is None or around_line <= 0:
        start = 1
        end = min(len(lines), 200)
    else:
        start = max(1, around_line - radius)
        end = min(len(lines), around_line + radius)

    content = "\n".join(lines[start - 1 : end])
    return {"path": rel, "start_line": start, "end_line": end, "content": content}


def _extract_candidate_files_from_prompt_context(prompt_context: str, *, repo_basename: str) -> list[tuple[str, Optional[int]]]:
    """
    Parse GraphContextProvider.to_prompt_context() output:
      - brainops-ai-agents/agent_executor.py (line 6350)
    """
    candidates: list[tuple[str, Optional[int]]] = []
    if not prompt_context:
        return candidates

    # Best-effort parsing; keep it forgiving.
    for line in prompt_context.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        m = re.match(r"^-\\s+([^\\s]+)\\s+\\(line\\s+(\\d+)\\)\\s*$", line)
        if not m:
            continue
        full_path = m.group(1)
        try:
            line_no = int(m.group(2))
        except Exception:
            line_no = None

        # full_path is like repo/file. Prefer files within target repo.
        prefix = repo_basename.rstrip("/") + "/"
        if full_path.startswith(prefix):
            candidates.append((full_path[len(prefix) :], line_no))

    # De-dup while preserving order.
    seen: set[str] = set()
    deduped: list[tuple[str, Optional[int]]] = []
    for path, line_no in candidates:
        if path in seen:
            continue
        seen.add(path)
        deduped.append((path, line_no))
    return deduped


def _safe_branch_name(proposal_id: str, title: str) -> str:
    short = (proposal_id or "").split("-")[0][:8] or "proposal"
    return f"selfbuild/{short}-{_slugify(title)}"


def _risk_allowed(risk_level: str) -> bool:
    risk = (risk_level or "").strip().lower()
    if risk in {"high", "critical"} and not _env_bool("SELF_CODING_ALLOW_HIGH_RISK", default=False):
        return False
    return True


async def implement_proposal_with_pr(
    *,
    ctx: PatchContext,
    ai_core: Any,
    mcp_client: Any | None,
    github_repo: str | None,
    base_branch: str = "main",
    test_command: list[str] | None = None,
    max_patch_tokens: int = 3000,
) -> dict[str, Any]:
    """
    Implement a proposal by generating and applying a patch and optionally creating a PR.
    """
    if not _env_bool("ENABLE_SELF_CODING_ENGINE", default=False):
        return {"status": "skipped", "reason": "self_coding_engine_disabled"}

    allowed, reason = _ensure_repo_allowed(ctx.repo_path)
    if not allowed:
        return {"status": "skipped", "reason": reason}

    if not _risk_allowed(ctx.risk_level):
        return {"status": "skipped", "reason": f"risk_blocked:{ctx.risk_level}"}

    if not ai_core:
        return {"status": "error", "error": "ai_core_unavailable"}

    ok, clean_reason = _ensure_clean_worktree(ctx.repo_path)
    if not ok:
        return {"status": "error", "error": clean_reason}

    branch_name = _safe_branch_name(ctx.proposal_id, ctx.title)

    original_branch = (_git(ctx.repo_path, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout or "").strip()
    original_head = (_git(ctx.repo_path, ["rev-parse", "HEAD"]).stdout or "").strip()

    # Build small, relevant snippets for the LLM.
    repo_basename = ctx.repo_path.name
    candidates = _extract_candidate_files_from_prompt_context(ctx.codebase_prompt_context, repo_basename=repo_basename)
    file_snippets: list[dict[str, Any]] = []
    for rel_path, line_no in candidates[:3]:
        snippet = _read_snippet(ctx.repo_path, rel_path, around_line=line_no, radius=120)
        if snippet:
            file_snippets.append(snippet)

    prompt = _build_patch_prompt(ctx, file_snippets)
    system_prompt = (
        "You are a senior software engineer. "
        "Return ONLY a unified diff (no explanations). "
        "The patch must apply cleanly."
    )

    model = os.getenv("SELF_CODING_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4-0125-preview"
    patch_text_raw = await ai_core.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=0.2,
        max_tokens=max_patch_tokens,
        intent="review",
        use_model_routing=False,
    )

    unified_diff = _extract_unified_diff(str(patch_text_raw))
    if not unified_diff:
        return {"status": "error", "error": "no_unified_diff_returned"}

    allowed, diff_reason = _diff_allowed(unified_diff)
    if not allowed:
        return {"status": "error", "error": diff_reason}

    _ensure_git_identity(ctx.repo_path)

    # Create branch off base branch.
    checkout_base = _git(ctx.repo_path, ["checkout", base_branch])
    if checkout_base.returncode != 0:
        # Some repos may not have local base branch; try origin/<base_branch>.
        checkout_base = _git(ctx.repo_path, ["checkout", "-B", base_branch, f"origin/{base_branch}"])
    if checkout_base.returncode != 0:
        return {"status": "error", "error": f"git_checkout_base_failed:{checkout_base.stderr.strip()}"}

    create_branch = _git(ctx.repo_path, ["checkout", "-b", branch_name])
    if create_branch.returncode != 0:
        return {"status": "error", "error": f"git_create_branch_failed:{create_branch.stderr.strip()}"}

    try:
        apply_patch = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=str(ctx.repo_path),
            input=unified_diff,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        _git(ctx.repo_path, ["reset", "--hard"])
        if original_branch:
            _git(ctx.repo_path, ["checkout", original_branch])
        return {"status": "error", "error": f"git_apply_failed:{exc}"}

    if apply_patch.returncode != 0:
        _git(ctx.repo_path, ["reset", "--hard"])
        if original_branch:
            _git(ctx.repo_path, ["checkout", original_branch])
        return {"status": "error", "error": f"git_apply_failed:{apply_patch.stderr.strip()}"}

    # Stage + test + commit.
    add = _git(ctx.repo_path, ["add", "-A"])
    if add.returncode != 0:
        _git(ctx.repo_path, ["reset", "--hard"])
        if original_branch:
            _git(ctx.repo_path, ["checkout", original_branch])
        return {"status": "error", "error": f"git_add_failed:{add.stderr.strip()}"}

    if not _env_bool("SELF_CODING_SKIP_TESTS", default=False):
        cmd = test_command or _parse_command(os.getenv("SELF_CODING_TEST_COMMAND")) or ["python3", "-m", "pytest", "-q", "tests"]
        test = _run(cmd, cwd=ctx.repo_path, timeout_seconds=float(os.getenv("SELF_CODING_TEST_TIMEOUT_SECONDS", "600")), check=False)
        if test.returncode != 0:
            _git(ctx.repo_path, ["reset", "--hard"])
            if original_branch:
                _git(ctx.repo_path, ["checkout", original_branch])
            return {
                "status": "error",
                "error": "tests_failed",
                "test_stdout": (test.stdout or "")[-2000:],
                "test_stderr": (test.stderr or "")[-2000:],
            }

    commit_msg = f"selfbuild: {ctx.title.strip()} ({ctx.proposal_id[:8]})"
    commit = _git(ctx.repo_path, ["commit", "-m", commit_msg], timeout_seconds=60.0)
    if commit.returncode != 0:
        _git(ctx.repo_path, ["reset", "--hard"])
        if original_branch:
            _git(ctx.repo_path, ["checkout", original_branch])
        return {"status": "error", "error": f"git_commit_failed:{commit.stderr.strip() or commit.stdout.strip()}"}

    commit_sha = (_git(ctx.repo_path, ["rev-parse", "HEAD"]).stdout or "").strip()

    pushed = False
    push_out = ""
    if _env_bool("SELF_CODING_ENABLE_PUSH", default=False):
        push = _git(ctx.repo_path, ["push", "-u", "origin", branch_name], timeout_seconds=float(os.getenv("SELF_CODING_PUSH_TIMEOUT_SECONDS", "180")))
        pushed = push.returncode == 0
        push_out = (push.stdout or "").strip() or (push.stderr or "").strip()

    pr = None
    if pushed and _env_bool("SELF_CODING_ENABLE_PR", default=False) and mcp_client and github_repo:
        body = "\n".join(
            [
                "Automated SelfBuilder PR generated from an approved improvement proposal.",
                "",
                f"Proposal ID: {ctx.proposal_id}",
                f"Type: {ctx.improvement_type}",
                f"Risk: {ctx.risk_level}",
                "",
                "Implementation steps:",
                *[f"- {s}" for s in (ctx.implementation_steps or [])],
                "",
                "Success criteria:",
                *[f"- {s}" for s in (ctx.success_criteria or [])],
                "",
            ]
        ).strip()
        try:
            pr_result = await mcp_client.github_create_pr(
                repo=github_repo,
                title=f"SelfBuilder: {ctx.title}",
                body=body,
                head=branch_name,
                base=base_branch,
            )
            if getattr(pr_result, "success", False):
                pr = pr_result.result
            else:
                pr = {"error": getattr(pr_result, "error", "pr_create_failed")}
        except Exception as exc:
            pr = {"error": str(exc)}

    # Return to original branch to reduce surprise in dev environments.
    if original_branch and original_branch != branch_name:
        _git(ctx.repo_path, ["checkout", original_branch])

    return {
        "status": "completed",
        "branch": branch_name,
        "commit": commit_sha,
        "pushed": pushed,
        "push_output": push_out,
        "pr": pr,
        "base_branch": base_branch,
        "repo_path": str(ctx.repo_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_branch": original_branch,
        "original_head": original_head,
    }

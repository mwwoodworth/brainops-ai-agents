# Self-Coding Engine (Phase 1)

This repo includes a feature-flagged self-modification pipeline that converts approved learning proposals into pull requests for human/governance review.

## How It Works

1. `LearningFeedbackLoop` detects patterns and writes rows to `public.ai_improvement_proposals`.
2. A human (or governance workflow) approves a proposal (`status='approved'`).
3. On the next feedback loop application pass, complex improvements are queued to `public.ai_task_queue` with `task_type='self_build'` (only when enabled).
4. `AITaskQueueConsumer` dispatches `self_build` tasks to `SelfBuildingAgent`.
5. `SelfBuildingAgent` calls the self-coding engine to:
   - create a git branch
   - generate a unified diff via `RealAICore`
   - apply it (with file suffix allowlist)
   - run tests
   - commit, optionally push, optionally open a PR
6. Proposal rows are updated with status and a short audit note (branch/commit/PR).

## Key Code

- Self-coding engine: `self_coding_engine.py`
- SelfBuilder action: `agent_executor.py` (`SelfBuildingAgent.implement_improvement_proposal`)
- Proposal queueing: `learning_feedback_loop.py` (`_queue_self_build_task`)
- Task dispatch: `ai_task_queue_consumer.py` (`_handle_self_build`)

## Safety Rails (Defaults Are Safe/Off)

`ENABLE_SELF_CODING_ENGINE=true`
- Required for queueing (LearningFeedbackLoop) and execution (task consumer + engine).

`SELF_CODING_ALLOWED_REPO_PREFIXES=/abs/path1,/abs/path2`
- Repo allowlist (defaults to only this repo if unset).

`SELF_CODING_ALLOW_HIGH_RISK=true`
- Required to act on proposals with `risk_level` `high` or `critical`.

`SELF_CODING_ALLOWED_FILE_SUFFIXES=.py,.md,.txt,...`
- Patch allowlist (default: `.py,.md,.txt,.json,.yml,.yaml`).
- Blocks `.env`, `BrainOps.env`, and `_secure/` paths.

`SELF_CODING_MODEL=gpt-4-0125-preview`
- Model used by `RealAICore` for diff generation (override as needed).

`SELF_CODING_TEST_COMMAND="python3 -m pytest -q tests"`
- Optional test command (string). If unset, defaults to `python3 -m pytest -q tests`.

`SELF_CODING_SKIP_TESTS=true`
- Disables the test gate (not recommended).

`SELF_CODING_ENABLE_PUSH=true`
- Enables `git push -u origin <branch>`.

`SELF_CODING_ENABLE_PR=true`
- Requires push enabled and MCP/GitHub integration configured; opens a PR against `SELF_CODING_BASE_BRANCH` (default `main`).

`AI_TASK_QUEUE_SELF_BUILD_TIMEOUT_SECONDS=900`
- Timeout for `self_build` tasks in the queue consumer.

## Expected Statuses

The system may set proposal statuses to:
- `queued_for_self_build`
- `self_build_completed` (branch+commit created; may not have PR)
- `pr_opened` (PR created; no auto-merge)

## Common Failure Modes

- `worktree_not_clean`: repo has uncommitted changes; automation refuses to run.
- `diff_suffix_not_allowed:*`: patch attempted to touch blocked file types.
- `tests_failed`: patch applied but tests failed; changes are reset (safe cleanup).
- `ai_core_unavailable`: `RealAICore` not configured (missing API keys / SDK).


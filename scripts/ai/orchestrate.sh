#!/bin/bash

# BrainOps Orchestration Wrapper
# Usage: brain [claude|codex|gemini] [args...]

TOOL=$1
shift
ARGS="$@"

# 1. Fetch Context
echo "🧠 [BrainOps] Syncing Neural Context..."
LATEST_CONTEXT=$(python3 /home/matt-woodworth/dev/scripts/ai/brain-bridge.py recall "devops status" 2>/dev/null | head -n 5)

# 2. Inject Context (Environment Variable Strategy)
export BRAINOPS_CONTEXT="$LATEST_CONTEXT"

# 3. Execute Tool
case "$TOOL" in
    "claude")
        echo "🚀 Launching Claude Code (Orchestrator Mode)..."
        # We assume 'claude' is in path. We can pass context via a temp file or env var if supported.
        # Claude Code reads context from files well.
        echo "Context Loaded: $LATEST_CONTEXT" > /tmp/brainops_context.md
        claude "$ARGS"
        ;;
    "codex")
        echo "⚡ Launching Codex (Code Factory)..."
        codex "$ARGS"
        ;;
    "gemini")
        echo "🔬 Launching Gemini (Deep Analysis)..."
        gemini "$ARGS"
        ;;
    *)
        echo "Usage: brain [claude|codex|gemini] [args]"
        ;;
esac

# 4. Post-Execution Sync (Optional - ask user to save summary)
# echo "💾 Syncing session to Brain..."
# python3 scripts/ai/brain-bridge.py store "Session finished: $TOOL $ARGS"

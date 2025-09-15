#!/bin/bash
# Production startup script for BrainOps AI Agents
# Ensures permanent operation with auto-recovery

echo "🚀 Starting BrainOps AI Operating System v2.0"
echo "================================================"
echo "Timestamp: $(date)"
echo "Environment: Production"
echo "Python: $(python3 --version)"

# Export environment variables
export PYTHONUNBUFFERED=1
export DB_HOST=${DB_HOST:-"aws-0-us-east-2.pooler.supabase.com"}
export DB_NAME=${DB_NAME:-"postgres"}
export DB_USER=${DB_USER:-"postgres.yomagoqdmxszqtdwuhab"}
export DB_PASSWORD=${DB_PASSWORD:-"REDACTED_SUPABASE_DB_PASSWORD"}
export DB_PORT=${DB_PORT:-5432}

echo "✅ Environment configured"

# Function to handle shutdown
cleanup() {
    echo "Shutting down gracefully..."
    kill $MAIN_PID 2>/dev/null
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start main application
while true; do
    echo "Starting AI agent system..."
    python3 main.py &
    MAIN_PID=$!

    # Wait for process
    wait $MAIN_PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Process completed successfully"
        break
    else
        echo "⚠️ Process failed with code $EXIT_CODE"
        echo "Restarting in 10 seconds..."
        sleep 10
    fi
done
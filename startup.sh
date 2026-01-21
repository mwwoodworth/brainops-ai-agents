#!/bin/bash
# Production startup script for BrainOps AI Agents
# Ensures permanent operation with auto-recovery

echo "🚀 Starting BrainOps AI Operating System v2.0"
echo "================================================"
echo "Timestamp: $(date)"
echo "Environment: Production"
echo "Python: $(python3 --version)"
echo "Starting app.py (NOT web_service.py)"

# Export environment variables - support both DATABASE_URL and individual vars
export PYTHONUNBUFFERED=1

# Parse DATABASE_URL if individual vars not set (Render provides DATABASE_URL)
if [ -n "$DATABASE_URL" ] && [ -z "$DB_PASSWORD" ]; then
    echo "📌 Parsing DATABASE_URL for database credentials..."
    # Extract components from DATABASE_URL: postgresql://user:password@host[:port]/database
    export DB_USER=$(echo "$DATABASE_URL" | sed -n 's|.*://\([^:]*\):.*|\1|p')
    export DB_PASSWORD=$(echo "$DATABASE_URL" | sed -n 's|.*://[^:]*:\([^@]*\)@.*|\1|p')
    # Handle both host:port and host/database formats
    export DB_HOST=$(echo "$DATABASE_URL" | sed -n 's|.*@\([^:/]*\).*|\1|p')
    export DB_PORT=$(echo "$DATABASE_URL" | sed -n 's|.*@[^:]*:\([0-9]*\)/.*|\1|p')
    export DB_NAME=$(echo "$DATABASE_URL" | sed -n 's|.*/\([^?]*\).*|\1|p')
    echo "✅ Extracted: host=$DB_HOST, db=$DB_NAME, user=$DB_USER, port=${DB_PORT:-5432}"
fi

# Fallback for non-sensitive values only (port and database name)
export DB_NAME=${DB_NAME:-"postgres"}
export DB_PORT=${DB_PORT:-5432}

# SECURITY: Host, User, and Password are REQUIRED - fail if not set
if [ -z "$DB_HOST" ]; then
    echo "❌ ERROR: DB_HOST not set!"
    echo "   Set DATABASE_URL or DB_HOST environment variable"
    exit 1
fi

if [ -z "$DB_USER" ]; then
    echo "❌ ERROR: DB_USER not set!"
    echo "   Set DATABASE_URL or DB_USER environment variable"
    exit 1
fi

if [ -z "$DB_PASSWORD" ]; then
    echo "❌ ERROR: DB_PASSWORD not set!"
    echo "   Set DATABASE_URL or DB_PASSWORD environment variable"
    exit 1
fi

echo "✅ Environment configured"

# Function to handle shutdown
cleanup() {
    echo "Shutting down gracefully..."
    kill $MAIN_PID 2>/dev/null
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start web service
while true; do
    echo "Starting AI agent web service via app.py (v9.0.0 - stable production)..."
    python3 app.py &
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
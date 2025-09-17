#!/bin/bash

# Deploy Supabase Edge Functions
echo "🚀 Deploying Supabase Edge Functions..."

# Check if supabase CLI is installed
if ! command -v supabase &> /dev/null; then
    echo "Installing Supabase CLI..."
    npm install -g supabase
fi

# Login to Supabase (requires access token)
echo "Logging in to Supabase..."
supabase login

# Link to project
echo "Linking to project..."
supabase link --project-ref yomagoqdmxszqtdwuhab

# Deploy each function
echo "Deploying ai-chat function..."
supabase functions deploy ai-chat --no-verify-jwt

echo "Deploying ai-analyze function..."
supabase functions deploy ai-analyze --no-verify-jwt

echo "Deploying ai-execute function..."
supabase functions deploy ai-execute --no-verify-jwt

echo "✅ Edge Functions deployed successfully!"

# Display function URLs
echo ""
echo "📍 Function Endpoints:"
echo "  - Chat: https://yomagoqdmxszqtdwuhab.supabase.co/functions/v1/ai-chat"
echo "  - Analyze: https://yomagoqdmxszqtdwuhab.supabase.co/functions/v1/ai-analyze"
echo "  - Execute: https://yomagoqdmxszqtdwuhab.supabase.co/functions/v1/ai-execute"
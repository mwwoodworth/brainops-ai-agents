#!/bin/bash
# Content Automation Script - Uses Gemini CLI (subscription-based, no quota issues)
# Run via cron for automated content generation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Source environment
source /home/matt-woodworth/dev/_secure/BrainOps.env 2>/dev/null || true

# Log file
LOG_FILE="/tmp/content_automation.log"
echo "$(date): Starting content automation" >> "$LOG_FILE"

# Topics for Gumroad products
TOPICS=(
    "How to Build Scalable AI Agents with Python|Business Automation Toolkit|biz-automation"
    "Command Center Dashboard Design Patterns|Command Center UI Kit|command-center"
    "Multi-LLM Orchestration Best Practices|AI Orchestration Framework|ai-orchestration"
    "Production-Ready AI Prompts Guide|AI Prompt Engineering Pack|prompt-pack"
    "SaaS Automation Scripts Every Startup Needs|SaaS Automation Scripts|saas-scripts"
    "Getting Started with MCP Servers|MCP Server Starter Kit|mcp-starter"
)

# Select a random topic
RANDOM_INDEX=$((RANDOM % ${#TOPICS[@]}))
IFS='|' read -r TOPIC PRODUCT PERMALINK <<< "${TOPICS[$RANDOM_INDEX]}"

echo "$(date): Generating post about: $TOPIC" >> "$LOG_FILE"

# Generate content using Gemini CLI
CONTENT=$(echo "Write a comprehensive, SEO-optimized blog post (700-900 words) about '$TOPIC'.
Target audience: developers and business owners interested in AI automation.
Include:
1. Engaging title with numbers or power words
2. Compelling introduction
3. 4 main sections with ## H2 headings
4. Practical examples
5. Call-to-action for $PRODUCT (https://brainstack.gumroad.com/l/$PERMALINK)
6. Professional but accessible tone
Format as markdown." | gemini 2>/dev/null)

if [ -z "$CONTENT" ]; then
    echo "$(date): ERROR - No content generated" >> "$LOG_FILE"
    exit 1
fi

# Extract title from content (first # line)
TITLE=$(echo "$CONTENT" | grep "^# " | head -1 | sed 's/^# //')
SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//' | sed 's/-$//')
EXCERPT=$(echo "$CONTENT" | grep -v "^#" | head -3 | tr '\n' ' ' | cut -c1-200)

echo "$(date): Generated: $TITLE" >> "$LOG_FILE"

# Insert into database
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d postgres -c "
INSERT INTO content_posts (id, title, slug, content, excerpt, status, created_at)
VALUES (gen_random_uuid(),
'$(echo "$TITLE" | sed "s/'/''/g")',
'$SLUG',
'$(echo "$CONTENT" | sed "s/'/''/g")',
'$(echo "$EXCERPT" | sed "s/'/''/g")',
'published',
NOW());" 2>/dev/null

echo "$(date): Post saved to database" >> "$LOG_FILE"
echo "$(date): Content automation complete" >> "$LOG_FILE"

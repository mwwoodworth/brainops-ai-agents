#!/bin/bash
# Comprehensive System Test Script
# Tests all 7 activated systems

echo "🔍 BRAINOPS SYSTEM TEST - v6.0.4"
echo "================================="
echo ""

BASE_URL="https://brainops-ai-agents.onrender.com"

echo "1️⃣  Testing Root Endpoint..."
curl -s "$BASE_URL/" | jq '{version, status, ai_enabled, scheduler_enabled}' || echo "❌ Root failed"
echo ""

echo "2️⃣  Testing Health Endpoint..."
HEALTH=$(curl -s "$BASE_URL/health")
echo "$HEALTH" | jq '{version, build: .build[0:19], database, system_count, active_systems}' || echo "❌ Health failed"
echo ""

echo "3️⃣  Checking System Count..."
SYSTEM_COUNT=$(echo "$HEALTH" | jq -r '.system_count')
echo "Active Systems: $SYSTEM_COUNT/7"
if [ "$SYSTEM_COUNT" = "7" ]; then
    echo "✅ ALL SYSTEMS ACTIVE!"
elif [ "$SYSTEM_COUNT" = "6" ]; then
    echo "⚠️  6/7 active (scheduler pending)"
else
    echo "❌ Only $SYSTEM_COUNT/7 active"
fi
echo ""

echo "4️⃣  Listing Active Systems..."
echo "$HEALTH" | jq -r '.active_systems[]' | while read system; do
    echo "  ✅ $system"
done
echo ""

echo "5️⃣  Checking Capabilities..."
echo "$HEALTH" | jq '.capabilities' || echo "❌ Capabilities check failed"
echo ""

FALSE_COUNT=$(echo "$HEALTH" | jq '[.capabilities[] | select(. == false)] | length')
if [ "$FALSE_COUNT" = "0" ]; then
    echo "✅ All capabilities TRUE"
else
    echo "⚠️  $FALSE_COUNT capabilities are FALSE"
    echo "$HEALTH" | jq '.capabilities | to_entries | map(select(.value == false))'
fi
echo ""

echo "6️⃣  Testing Database Connection..."
DB_STATUS=$(echo "$HEALTH" | jq -r '.database')
if [ "$DB_STATUS" = "connected" ]; then
    echo "✅ Database connected"
else
    echo "❌ Database: $DB_STATUS"
fi
echo ""

echo "7️⃣  Final Score..."
if [ "$SYSTEM_COUNT" = "7" ] && [ "$FALSE_COUNT" = "0" ] && [ "$DB_STATUS" = "connected" ]; then
    echo "🎉 PERFECT SCORE: 100% (7/7 systems active)"
    exit 0
elif [ "$SYSTEM_COUNT" = "6" ]; then
    echo "⚠️  GOOD SCORE: 85.7% (6/7 systems active)"
    exit 0
else
    echo "❌ NEEDS WORK: $(echo "scale=1; $SYSTEM_COUNT * 100 / 7" | bc)% ($SYSTEM_COUNT/7 systems active)"
    exit 1
fi

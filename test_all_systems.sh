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
EXPECTED_SYSTEM_COUNT="${EXPECTED_SYSTEM_COUNT:-16}"
echo "Active Systems: $SYSTEM_COUNT/$EXPECTED_SYSTEM_COUNT"
if [ "$SYSTEM_COUNT" -ge "$EXPECTED_SYSTEM_COUNT" ]; then
    echo "✅ ALL SYSTEMS ACTIVE OR MORE"
else
    echo "⚠️  $SYSTEM_COUNT/$EXPECTED_SYSTEM_COUNT systems active"
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
if [ "$SYSTEM_COUNT" -ge "$EXPECTED_SYSTEM_COUNT" ] && [ "$FALSE_COUNT" = "0" ] && [ "$DB_STATUS" = "connected" ]; then
    echo "🎉 PERFECT SCORE: 100% ($SYSTEM_COUNT systems active)"
    exit 0
else
    if [ "$DB_STATUS" != "connected" ]; then
        echo "❌ Database not connected: $DB_STATUS"
    fi
    if [ "$FALSE_COUNT" -ne "0" ]; then
        echo "⚠️  $FALSE_COUNT capabilities reported FALSE"
    fi
    if [ "$SYSTEM_COUNT" -lt "$EXPECTED_SYSTEM_COUNT" ]; then
        echo "⚠️  $SYSTEM_COUNT/$EXPECTED_SYSTEM_COUNT systems active"
    fi
    SYSTEM_SCORE=$(echo "scale=1; 100 * $SYSTEM_COUNT / $EXPECTED_SYSTEM_COUNT" | bc)
    echo "⚠️  SCORE: $SYSTEM_SCORE%"
    exit 1
fi

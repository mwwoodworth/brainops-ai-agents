#!/bin/bash
# Test script for embedded memory sync fix in production

API_BASE="https://brainops-ai-agents.onrender.com"
API_KEY="brainops_prod_key_2025"

echo "🧪 Testing Embedded Memory Sync Fix in Production"
echo "=================================================="
echo ""

# 1. Check service health
echo "1. Checking service health..."
HEALTH=$(curl -s "$API_BASE/health" | jq -r '.status')
if [ "$HEALTH" == "healthy" ]; then
    echo "   ✅ Service is healthy"
else
    echo "   ❌ Service is not healthy: $HEALTH"
    exit 1
fi
echo ""

# 2. Check memory stats
echo "2. Checking memory stats (before force sync)..."
STATS=$(curl -s "$API_BASE/memory/stats")

if echo "$STATS" | jq -e '.enabled' > /dev/null 2>&1; then
    echo "   ✅ Memory stats endpoint working"
    TOTAL_MEMORIES=$(echo "$STATS" | jq -r '.total_memories')
    POOL_CONNECTED=$(echo "$STATS" | jq -r '.pool_connected')
    LAST_SYNC=$(echo "$STATS" | jq -r '.last_sync')

    echo "   📊 Total memories: $TOTAL_MEMORIES"
    echo "   🔌 Pool connected: $POOL_CONNECTED"
    echo "   🕒 Last sync: $LAST_SYNC"
else
    echo "   ❌ Memory stats endpoint not available yet (deployment in progress?)"
    echo "   Response: $STATS"
fi
echo ""

# 3. Wait a bit for background sync
echo "3. Waiting 60 seconds for background sync to complete..."
sleep 60
echo ""

# 4. Check stats again
echo "4. Checking memory stats (after background sync)..."
STATS2=$(curl -s "$API_BASE/memory/stats")

if echo "$STATS2" | jq -e '.enabled' > /dev/null 2>&1; then
    TOTAL_MEMORIES2=$(echo "$STATS2" | jq -r '.total_memories')
    POOL_CONNECTED2=$(echo "$STATS2" | jq -r '.pool_connected')
    LAST_SYNC2=$(echo "$STATS2" | jq -r '.last_sync')

    echo "   📊 Total memories: $TOTAL_MEMORIES2"
    echo "   🔌 Pool connected: $POOL_CONNECTED2"
    echo "   🕒 Last sync: $LAST_SYNC2"

    if [ "$TOTAL_MEMORIES2" -gt 0 ]; then
        echo "   ✅ Background sync populated local cache!"
    else
        echo "   ⚠️ Local cache still empty, trying force sync..."
    fi
else
    echo "   ❌ Memory stats endpoint error"
fi
echo ""

# 5. Try force sync if still empty
if [ "${TOTAL_MEMORIES2:-0}" -eq 0 ]; then
    echo "5. Triggering force sync..."
    FORCE_RESULT=$(curl -s -X POST "$API_BASE/memory/force-sync" \
        -H "X-API-Key: $API_KEY")

    if echo "$FORCE_RESULT" | jq -e '.success' > /dev/null 2>&1; then
        echo "   ✅ Force sync successful"
        BEFORE=$(echo "$FORCE_RESULT" | jq -r '.before_count')
        AFTER=$(echo "$FORCE_RESULT" | jq -r '.after_count')
        SYNCED=$(echo "$FORCE_RESULT" | jq -r '.synced_count')

        echo "   📊 Before: $BEFORE"
        echo "   📊 After: $AFTER"
        echo "   📊 Synced: $SYNCED"
    else
        echo "   ❌ Force sync failed"
        echo "   Response: $FORCE_RESULT"
    fi
else
    echo "5. Skipping force sync (background sync already populated cache)"
fi
echo ""

# 6. Final stats check
echo "6. Final memory stats check..."
FINAL_STATS=$(curl -s "$API_BASE/memory/stats")

if echo "$FINAL_STATS" | jq -e '.enabled' > /dev/null 2>&1; then
    FINAL_COUNT=$(echo "$FINAL_STATS" | jq -r '.total_memories')
    echo "   📊 Final total memories: $FINAL_COUNT"

    if [ "$FINAL_COUNT" -gt 0 ]; then
        echo "   ✅ SUCCESS! Embedded memory sync is working!"
    else
        echo "   ❌ FAILED! Local cache still empty"
    fi
else
    echo "   ❌ Could not get final stats"
fi
echo ""

echo "=================================================="
echo "🎉 Test complete!"

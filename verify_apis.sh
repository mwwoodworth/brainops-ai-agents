#!/bin/bash

echo "=== API Endpoint Verification ==="

echo "1. Testing BrainOps Backend Health..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://brainops-backend-prod.onrender.com/api/v1/health)
if [ "$HEALTH_STATUS" == "200" ]; then
    echo "✅ Backend Health Check Passed (200)"
else
    echo "❌ Backend Health Check Failed (Status: $HEALTH_STATUS)"
fi

echo "2. Testing Weathercraft ERP API (Customers Endpoint)..."
ERP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://weathercraft-erp.vercel.app/api/customers)
echo "   Response Code: $ERP_STATUS"
if [ "$ERP_STATUS" == "401" ] || [ "$ERP_STATUS" == "403" ]; then
    echo "✅ Security Check Passed (Endpoint Protected)"
elif [ "$ERP_STATUS" == "200" ]; then
    echo "⚠️  WARNING: Customers endpoint is PUBLIC (200)"
else
    echo "❌ Unexpected Status: $ERP_STATUS"
fi

echo "3. Testing MRG Stripe Checkout Endpoint..."
# Test 3a: Empty POST (Expect 500 based on previous run, or 400)
STRIPE_STATUS=$(curl -X POST -s -o /dev/null -w "%{http_code}" https://myroofgenius.com/api/stripe/create-checkout-session)
echo "   Empty POST Code: $STRIPE_STATUS"

# Test 3b: JSON POST (Expect 400 'priceId required' or 200 if valid-ish)
# Sending minimal valid structure to pass JSON parsing
STRIPE_JSON_STATUS=$(curl -X POST -s -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" -d '{"mode": "payment", "price": 100, "productName": "Verification"}' https://myroofgenius.com/api/stripe/create-checkout-session)
echo "   JSON POST Code:  $STRIPE_JSON_STATUS"

if [ "$STRIPE_JSON_STATUS" == "200" ]; then
    echo "✅ Stripe Endpoint Functional (200)"
elif [ "$STRIPE_JSON_STATUS" == "400" ]; then
    echo "✅ Stripe Endpoint Reachable & Validating (400 - Expected for test data)"
elif [ "$STRIPE_JSON_STATUS" == "500" ]; then
    echo "❌ Stripe Endpoint Error (500) - Unhandled Exception"
else
    echo "❌ Unexpected Status: $STRIPE_JSON_STATUS"
fi

echo "=== API Verification Complete ==="

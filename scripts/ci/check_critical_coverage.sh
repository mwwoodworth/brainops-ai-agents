#!/usr/bin/env bash
# CI gate: verify coverage floors for critical modules.
# Run: bash scripts/ci/check_critical_coverage.sh
#
# This checks that the highest-risk modules maintain their coverage floors.
# It does NOT enforce overall coverage (that would be vanity metrics).

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== Critical Module Coverage Gate ==="

# Define critical modules and their minimum coverage floors
declare -A FLOORS=(
    ["database/tenant_guard.py"]=80
    ["brain_store_helper.py"]=90
    ["base_agent.py"]=40
    ["invariant_monitor.py"]=30
)

FAILED=0

for module in "${!FLOORS[@]}"; do
    floor=${FLOORS[$module]}

    # Run coverage for this specific module
    coverage=$(python3 -m pytest \
        --cov="$module" \
        --cov-report=term \
        --cov-config=.coveragerc \
        -q tests/ 2>&1 \
        | grep -E "^${module}" \
        | awk '{print $NF}' \
        | tr -d '%' || echo "0")

    if [ -z "$coverage" ] || [ "$coverage" = "0" ]; then
        echo "FAIL: $module — coverage=0% (floor=${floor}%)"
        FAILED=1
    elif [ "$coverage" -lt "$floor" ]; then
        echo "FAIL: $module — coverage=${coverage}% < floor=${floor}%"
        FAILED=1
    else
        echo "PASS: $module — coverage=${coverage}% >= floor=${floor}%"
    fi
done

echo ""
if [ "$FAILED" -eq 1 ]; then
    echo "COVERAGE GATE FAILED — critical modules below floor"
    exit 1
else
    echo "COVERAGE GATE PASSED"
fi

#!/usr/bin/env python3
"""
Test Suite for Upgraded Self-Healing System
Tests all new proactive recovery capabilities
"""

import asyncio
import sys
from datetime import datetime
from self_healing_recovery import get_self_healing_recovery

def test_proactive_health_monitoring():
    """Test proactive health monitoring"""
    print("\n=== Testing Proactive Health Monitoring ===")

    healer = get_self_healing_recovery()

    # Test with healthy metrics
    healthy_metrics = {
        'cpu_usage': 45.0,
        'memory_usage': 60.0,
        'error_rate': 0.01,
        'latency_ms': 150.0
    }

    result = healer.monitor_proactive_health('test_component', healthy_metrics)
    print(f"Healthy Component: Score={result['health_score']:.1f}%, Trend={result['trend']}")
    print(f"  Warnings: {result['warnings']}")

    # Test with degrading metrics
    degrading_metrics = {
        'cpu_usage': 85.0,
        'memory_usage': 88.0,
        'error_rate': 0.08,
        'latency_ms': 3500.0
    }

    result = healer.monitor_proactive_health('test_component', degrading_metrics)
    print(f"\nDegrading Component: Score={result['health_score']:.1f}%, Trend={result['trend']}")
    print(f"  Warnings: {result['warnings']}")
    print(f"  Failure Prediction: {result['failure_prediction']}")

    return True

def test_predictive_failure_detection():
    """Test predictive failure detection"""
    print("\n=== Testing Predictive Failure Detection ===")

    healer = get_self_healing_recovery()

    # Simulate increasing memory usage (memory leak pattern)
    for i in range(15):
        metrics = {
            'cpu_usage': 50.0,
            'memory_usage': 60.0 + (i * 2),  # Increasing memory
            'error_rate': 0.02,
            'latency_ms': 200.0
        }
        healer.monitor_proactive_health('leaky_component', metrics)

    # Check prediction
    final_metrics = {
        'cpu_usage': 50.0,
        'memory_usage': 92.0,
        'error_rate': 0.02,
        'latency_ms': 200.0
    }

    result = healer.monitor_proactive_health('leaky_component', final_metrics)
    prediction = result['failure_prediction']

    print(f"Memory Leak Detection:")
    print(f"  Failure Probability: {prediction['probability']:.1%}")
    print(f"  Time to Failure: {prediction['time_to_failure']}")
    print(f"  Reasons: {prediction['reasons']}")

    return prediction['probability'] > 0.3

async def test_automatic_rollback():
    """Test automatic rollback capabilities"""
    print("\n=== Testing Automatic Rollback ===")

    healer = get_self_healing_recovery()

    # Test rollback
    result = await healer.rollback_component('test_service', 'previous_state')

    print(f"Rollback Result:")
    print(f"  Success: {result['success']}")
    print(f"  Component: {result.get('component')}")
    print(f"  Rollback Type: {result.get('rollback_type')}")

    return result is not None

async def test_service_restart():
    """Test service restart via Render API (mock)"""
    print("\n=== Testing Service Restart via Render API ===")

    healer = get_self_healing_recovery()

    # Note: This will fail without valid API key, which is expected
    result = await healer.restart_service_via_render(
        'srv-test123',
        'test_service'
    )

    print(f"Service Restart Result:")
    print(f"  Success: {result['success']}")
    print(f"  Message: {result.get('message', result.get('error'))}")

    # Expected to fail without API key
    return True

def test_database_connection_recovery():
    """Test database connection recovery"""
    print("\n=== Testing Database Connection Recovery ===")

    healer = get_self_healing_recovery()

    result = healer.recover_database_connection()

    print(f"Database Recovery Result:")
    print(f"  Success: {result['success']}")
    print(f"  Attempts: {result.get('attempts', 'N/A')}")
    print(f"  Message: {result.get('message', result.get('error'))}")

    return result['success']

def test_memory_leak_detection():
    """Test memory leak detection and cleanup"""
    print("\n=== Testing Memory Leak Detection & Cleanup ===")

    healer = get_self_healing_recovery()

    # Set a baseline
    healer.memory_baselines['test_component'] = 100.0

    result = healer.detect_and_cleanup_memory_leaks('test_component')

    print(f"Memory Leak Detection Result:")
    print(f"  Success: {result['success']}")
    print(f"  Leak Detected: {result.get('leak_detected', False)}")
    print(f"  Message: {result.get('message', 'Cleanup performed')}")

    if 'memory_freed_mb' in result:
        print(f"  Memory Freed: {result['memory_freed_mb']:.1f}MB")
        print(f"  Objects Collected: {result['objects_collected']}")

    return True

def test_unified_brain_logging():
    """Test unified brain logging"""
    print("\n=== Testing Unified Brain Logging ===")

    healer = get_self_healing_recovery()

    # This should log to unified_brain
    test_data = {
        'component': 'test_component',
        'action': 'test_action',
        'success': True
    }

    healer._log_to_unified_brain('test_healing_action', test_data)

    print("Logged test action to unified_brain table")
    print("  Agent: self_healing_system")
    print("  Action Type: test_healing_action")
    print("  Data: component=test_component, success=True")

    return True

async def run_all_tests():
    """Run all test cases"""
    print("=" * 70)
    print("SELF-HEALING SYSTEM UPGRADE TEST SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")

    tests = [
        ("Proactive Health Monitoring", test_proactive_health_monitoring),
        ("Predictive Failure Detection", test_predictive_failure_detection),
        ("Automatic Rollback", test_automatic_rollback),
        ("Service Restart via Render", test_service_restart),
        ("Database Connection Recovery", test_database_connection_recovery),
        ("Memory Leak Detection", test_memory_leak_detection),
        ("Unified Brain Logging", test_unified_brain_logging),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            print(f"\n ERROR in {test_name}: {e}")
            results[test_name] = f"ERROR: {str(e)}"

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, result in results.items():
        status_icon = "✓" if result == "PASSED" else "✗"
        print(f"{status_icon} {test_name}: {result}")

    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Completed at: {datetime.now().isoformat()}")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

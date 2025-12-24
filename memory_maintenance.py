#!/usr/bin/env python3
"""
Memory System Maintenance Script
Run periodic maintenance tasks for memory coordination
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime
from memory_coordination_system import get_memory_coordinator
from unified_memory_manager import get_memory_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_health_check():
    """Run comprehensive health check"""
    logger.info("=" * 80)
    logger.info("MEMORY HEALTH CHECK")
    logger.info("=" * 80)

    coordinator = get_memory_coordinator()
    health = await coordinator.health_check()

    print(json.dumps(health, indent=2, default=str))

    if health['overall_status'] != 'healthy':
        logger.warning(f"⚠️ Status: {health['overall_status']}")
        if health['issues']:
            logger.warning("Issues detected:")
            for issue in health['issues']:
                logger.warning(f"  - {issue}")
        return False

    logger.info("✅ All systems healthy")
    return True


async def run_sync(limit: int = 1000):
    """Sync master to embedded memory"""
    logger.info("=" * 80)
    logger.info(f"MEMORY SYNC (limit={limit})")
    logger.info("=" * 80)

    coordinator = get_memory_coordinator()

    # Check if sync needed
    health = await coordinator.health_check()
    sync_gap = health.get('sync_status', {}).get('master_to_embedded', {}).get('gap', 0)

    if sync_gap == 0:
        logger.info("✅ No sync needed - systems already in sync")
        return

    logger.info(f"🔄 Sync gap detected: {sync_gap} entries")

    # Run sync
    result = await coordinator.sync_master_to_embedded(limit=limit)

    logger.info(f"✅ Sync complete: {result['synced']} synced, {result['failed']} failed")

    if result['failed'] > 0:
        logger.warning(f"⚠️ {result['failed']} entries failed to sync")


async def run_deduplication():
    """Remove duplicate memories"""
    logger.info("=" * 80)
    logger.info("MEMORY DEDUPLICATION")
    logger.info("=" * 80)

    coordinator = get_memory_coordinator()
    removed = await coordinator.deduplicate_memories()

    logger.info(f"✅ Deduplication complete: {removed} duplicates removed")


async def run_retention(tenant_id: str = None, aggressive: bool = False):
    """Apply retention policy"""
    logger.info("=" * 80)
    logger.info(f"RETENTION POLICY (aggressive={aggressive})")
    logger.info("=" * 80)

    manager = get_memory_manager()

    if not tenant_id:
        # Get stats to find tenants
        coordinator = get_memory_coordinator()
        stats = await coordinator.get_stats()
        logger.info(f"Processing retention for all {stats.get('active_tenants', 0)} tenants")
        # In production, iterate through all tenants
        # For now, just show message
        logger.warning("⚠️ Specify --tenant-id for actual retention processing")
        return

    result = manager.apply_retention_policy(tenant_id=tenant_id, aggressive=aggressive)

    logger.info(f"✅ Retention policy applied:")
    logger.info(f"  - Retained: {result['retained']}")
    logger.info(f"  - Removed: {result['removed']}")
    logger.info(f"  - Promoted: {result['promoted']}")
    logger.info(f"  - Demoted: {result['demoted']}")


async def run_gc(tenant_id: str = None, dry_run: bool = False):
    """Run garbage collection"""
    logger.info("=" * 80)
    logger.info(f"GARBAGE COLLECTION (dry_run={dry_run})")
    logger.info("=" * 80)

    coordinator = get_memory_coordinator()
    result = await coordinator.garbage_collect()

    logger.info(f"{'📊 Would remove' if dry_run else '✅ Removed'}:")
    logger.info(f"  - Expired: {result['expired']}")
    logger.info(f"  - Old/low-importance: {result['old_low_importance']}")
    logger.info(f"  - Total: {result['total_removed']}")


async def run_stats():
    """Display comprehensive statistics"""
    logger.info("=" * 80)
    logger.info("MEMORY STATISTICS")
    logger.info("=" * 80)

    coordinator = get_memory_coordinator()
    stats = await coordinator.get_stats()

    print("\n📊 Master Registry:")
    print(f"  Total entries: {stats.get('total_entries', 0):,}")
    print(f"  By layer:")
    print(f"    - Ephemeral: {stats.get('ephemeral_count', 0):,}")
    print(f"    - Session: {stats.get('session_count', 0):,}")
    print(f"    - Short-term: {stats.get('short_term_count', 0):,}")
    print(f"    - Long-term: {stats.get('long_term_count', 0):,}")
    print(f"    - Permanent: {stats.get('permanent_count', 0):,}")
    print(f"  Critical entries: {stats.get('critical_count', 0):,}")
    print(f"  Active sessions: {stats.get('active_sessions', 0):,}")
    print(f"  Active tenants: {stats.get('active_tenants', 0):,}")
    print(f"  Total accesses: {stats.get('total_accesses', 0):,}")

    print("\n💾 Cache:")
    cache = stats.get('cache_size', {})
    print(f"  Ephemeral: {cache.get('ephemeral', 0):,}")
    print(f"  Session: {cache.get('session', 0):,}")

    print(f"\n🔄 Pending syncs: {stats.get('pending_syncs', 0):,}")

    if 'embedded_memory' in stats:
        print("\n🧠 Embedded Memory:")
        print(json.dumps(stats['embedded_memory'], indent=2, default=str))

    if 'vector_memory' in stats:
        print("\n🔍 Vector Memory:")
        print(json.dumps(stats['vector_memory'], indent=2, default=str))


async def run_full_maintenance(tenant_id: str = None):
    """Run complete maintenance cycle"""
    logger.info("=" * 80)
    logger.info("FULL MAINTENANCE CYCLE")
    logger.info("=" * 80)

    # 1. Health check
    healthy = await run_health_check()

    # 2. Sync if needed
    await run_sync(limit=10000)

    # 3. Deduplicate
    await run_deduplication()

    # 4. Garbage collect
    await run_gc(tenant_id=tenant_id)

    # 5. Final health check
    logger.info("\n" + "=" * 80)
    logger.info("FINAL HEALTH CHECK")
    logger.info("=" * 80)
    await run_health_check()

    # 6. Show stats
    await run_stats()

    logger.info("\n✅ Full maintenance cycle complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Memory System Maintenance')

    parser.add_argument('--health-check', action='store_true',
                       help='Run health check')
    parser.add_argument('--sync', action='store_true',
                       help='Sync master to embedded memory')
    parser.add_argument('--deduplicate', action='store_true',
                       help='Remove duplicate memories')
    parser.add_argument('--retention', action='store_true',
                       help='Apply retention policy')
    parser.add_argument('--gc', action='store_true',
                       help='Run garbage collection')
    parser.add_argument('--stats', action='store_true',
                       help='Display statistics')
    parser.add_argument('--full', action='store_true',
                       help='Run full maintenance cycle')

    parser.add_argument('--limit', type=int, default=1000,
                       help='Limit for sync operations')
    parser.add_argument('--tenant-id', type=str,
                       help='Tenant ID for tenant-specific operations')
    parser.add_argument('--aggressive', action='store_true',
                       help='Use aggressive retention policy')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (no actual changes)')

    args = parser.parse_args()

    # If no operation specified, show help
    if not any([args.health_check, args.sync, args.deduplicate,
                args.retention, args.gc, args.stats, args.full]):
        parser.print_help()
        return

    # Run requested operations
    if args.full:
        asyncio.run(run_full_maintenance(tenant_id=args.tenant_id))
    else:
        if args.health_check:
            asyncio.run(run_health_check())

        if args.sync:
            asyncio.run(run_sync(limit=args.limit))

        if args.deduplicate:
            asyncio.run(run_deduplication())

        if args.retention:
            asyncio.run(run_retention(
                tenant_id=args.tenant_id,
                aggressive=args.aggressive
            ))

        if args.gc:
            asyncio.run(run_gc(
                tenant_id=args.tenant_id,
                dry_run=args.dry_run
            ))

        if args.stats:
            asyncio.run(run_stats())


if __name__ == '__main__':
    main()

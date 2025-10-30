#!/usr/bin/env python3
"""Measure startup timing to identify blocking operations."""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from src.database.sqlite_manager import get_db_manager
from src.services.startup_service import get_startup_service

async def measure_startup():
    print("=== Measuring Startup Performance ===\n")

    total_start = time.time()

    # Step 1: Get DB Manager
    step_start = time.time()
    db_mgr = get_db_manager()
    step_duration = time.time() - step_start
    print(f"Step 1: get_db_manager() - {step_duration:.3f}s")

    # Step 2: Initialize DB
    step_start = time.time()
    await db_mgr.initialize()
    step_duration = time.time() - step_start
    print(f"Step 2: db_mgr.initialize() - {step_duration:.3f}s")

    # Step 3: Get Startup Service
    step_start = time.time()
    startup_service = get_startup_service(db_mgr)
    step_duration = time.time() - step_start
    print(f"Step 3: get_startup_service() - {step_duration:.3f}s")

    # Step 4: Load Startup Data (with detailed timing)
    print("\nStep 4: load_startup_data() breakdown:")
    step_start = time.time()

    # Manually call the substeps to measure
    await startup_service._ensure_db()

    # Articles
    substep_start = time.time()
    articles = await db_mgr.get_all_articles(limit=100)
    substep_duration = time.time() - substep_start
    print(f"  - Load articles: {substep_duration:.3f}s ({len(articles)} articles)")

    # Cache age
    substep_start = time.time()
    cache_age = await db_mgr.get_cache_age_hours()
    substep_duration = time.time() - substep_start
    print(f"  - Get cache age: {substep_duration:.3f}s")

    # Timestamps
    substep_start = time.time()
    timestamps = await db_mgr.get_setting('last_refresh_timestamps', {})
    substep_duration = time.time() - substep_start
    print(f"  - Load timestamps: {substep_duration:.3f}s")

    # Weather
    substep_start = time.time()
    weather_cache = await db_mgr.get_cache('weather_data')
    substep_duration = time.time() - substep_start
    print(f"  - Load weather: {substep_duration:.3f}s")

    # Emails
    substep_start = time.time()
    email_cache = await db_mgr.get_cache('emails_data')
    substep_duration = time.time() - substep_start
    print(f"  - Load emails: {substep_duration:.3f}s")

    # Calendar
    substep_start = time.time()
    calendar_cache = await db_mgr.get_cache('calendar_data')
    substep_duration = time.time() - substep_start
    print(f"  - Load calendar: {substep_duration:.3f}s")

    step_duration = time.time() - step_start
    print(f"  TOTAL load_startup_data(): {step_duration:.3f}s")

    total_duration = time.time() - total_start
    print(f"\n=== TOTAL STARTUP TIME: {total_duration:.3f}s ===")

    # Summary
    print(f"\nData loaded:")
    print(f"  - Articles: {len(articles)}")
    print(f"  - Weather: {'Yes' if weather_cache else 'No'}")
    print(f"  - Emails: {'Yes' if email_cache else 'No'}")
    print(f"  - Calendar: {'Yes' if calendar_cache else 'No'}")
    print(f"  - Cache age: {cache_age}h" if cache_age else "  - Cache age: None")

    return total_duration

if __name__ == "__main__":
    duration = asyncio.run(measure_startup())

    if duration < 1.0:
        print(f"\n✅ FAST - Startup completed in {duration:.3f}s")
        exit(0)
    elif duration < 2.0:
        print(f"\n⚠️  ACCEPTABLE - Startup completed in {duration:.3f}s")
        exit(0)
    else:
        print(f"\n❌ SLOW - Startup took {duration:.3f}s (should be < 1s)")
        exit(1)

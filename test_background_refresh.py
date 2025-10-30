#!/usr/bin/env python3
"""Test if background refresh works and saves to database."""

import asyncio
from src.services.background_refresh_service import get_background_refresh_service
from src.database.sqlite_manager import get_db_manager

async def test_background_refresh():
    print("\n=== Testing Background Refresh ===\n")

    # Check initial database state
    db = get_db_manager()
    await db.initialize()

    initial_articles = await db.get_all_articles(limit=100)
    print(f"📊 Initial articles in DB: {len(initial_articles)}")

    # Run background refresh
    print("\n🔄 Running background refresh...")
    refresh_service = get_background_refresh_service()

    def progress_callback(source: str, progress: float):
        print(f"  Progress - {source}: {progress * 100:.0f}%")

    results = await refresh_service.refresh_all_sources(progress_callback=progress_callback)

    print(f"\n📊 Refresh Results:")
    for source, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {source}: {success}")

    # Check final database state
    final_articles = await db.get_all_articles(limit=100)
    print(f"\n📊 Final articles in DB: {len(final_articles)}")
    print(f"📊 New articles added: {len(final_articles) - len(initial_articles)}")

    if final_articles:
        print(f"\n📰 First article:")
        print(f"  Title: {final_articles[0].title}")
        print(f"  Source: {final_articles[0].source_name}")
        print(f"  URL: {final_articles[0].url}")

    return len(final_articles) > 0

if __name__ == "__main__":
    success = asyncio.run(test_background_refresh())
    exit(0 if success else 1)

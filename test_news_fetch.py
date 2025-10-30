#!/usr/bin/env python3
"""Test if news fetching works at all."""

import asyncio
from src.services.news_service import get_news_service

async def test_fetch():
    print("Testing news fetch...")
    service = get_news_service()

    try:
        articles = await service.fetch_all_news(max_articles=5)
        print(f"\n✅ SUCCESS: Fetched {len(articles)} articles")

        if articles:
            print("\nFirst article:")
            print(f"  Title: {articles[0].title}")
            print(f"  Source: {articles[0].source_name}")
            print(f"  URL: {articles[0].url}")

        return True
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fetch())
    exit(0 if success else 1)

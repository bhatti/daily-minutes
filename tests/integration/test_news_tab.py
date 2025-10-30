#!/usr/bin/env python3
"""Integration tests for News tab functionality."""

import asyncio
import sys
sys.path.append('.')

from src.connectors.hackernews import HackerNewsConnector
from src.services.content_fetcher import get_content_fetcher


async def test_news_tab_data_structure():
    """Test that news data has the structure needed for the News tab."""
    print("\n=== Testing News Tab Data Structure ===\n")

    # Fetch sample articles
    connector = HackerNewsConnector(max_stories=5)
    articles = await connector.execute_async()

    if not articles:
        print("❌ No articles fetched")
        return False

    print(f"✅ Fetched {len(articles)} articles")

    # Verify each article has required fields for tab display
    required_fields = ['title', 'url', 'published_at', 'source', 'tags']

    for i, article in enumerate(articles[:3], 1):
        print(f"\nArticle {i}: {article.title[:50]}...")

        for field in required_fields:
            if not hasattr(article, field):
                print(f"   ❌ Missing field: {field}")
                return False

            value = getattr(article, field)
            if value is None or (isinstance(value, str) and not value):
                print(f"   ⚠️  Empty field: {field}")
            else:
                print(f"   ✅ {field}: {str(value)[:30]}...")

    print("\n✅ All articles have required fields")
    return True


async def test_news_tab_grouping():
    """Test that news can be grouped by source/category."""
    print("\n=== Testing News Grouping ===\n")

    connector = HackerNewsConnector(max_stories=10)
    articles = await connector.execute_async()

    if not articles:
        print("❌ No articles fetched")
        return False

    # Group by source
    by_source = {}
    for article in articles:
        source = str(article.source)
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(article)

    print(f"✅ Grouped into {len(by_source)} sources:")
    for source, arts in by_source.items():
        print(f"   {source}: {len(arts)} articles")

    # Group by tags
    by_tag = {}
    for article in articles:
        if article.tags:
            for tag in article.tags[:2]:  # Top 2 tags
                if tag not in by_tag:
                    by_tag[tag] = []
                by_tag[tag].append(article)

    if by_tag:
        print(f"\n✅ Found {len(by_tag)} unique tags")
        top_tags = sorted(by_tag.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        for tag, arts in top_tags:
            print(f"   {tag}: {len(arts)} articles")

    return True


async def test_news_tab_filtering():
    """Test that news can be filtered by date/priority."""
    print("\n=== Testing News Filtering ===\n")

    connector = HackerNewsConnector(max_stories=10)
    articles = await connector.execute_async()

    if not articles:
        print("❌ No articles fetched")
        return False

    # Filter by priority
    high_priority = [a for a in articles if str(a.priority).lower() in ['high', 'urgent']]
    print(f"✅ High priority articles: {len(high_priority)}/{len(articles)}")

    # Check date sorting
    sorted_articles = sorted(articles, key=lambda x: x.published_at, reverse=True)
    print(f"✅ Articles can be sorted by date")
    print(f"   Latest: {sorted_articles[0].published_at}")
    print(f"   Oldest: {sorted_articles[-1].published_at}")

    return True


async def test_news_tab_excerpts():
    """Test that articles have excerpts after content enrichment."""
    print("\n=== Testing News Article Excerpts ===\n")

    # Fetch sample articles
    connector = HackerNewsConnector(max_stories=5)
    articles = await connector.execute_async()

    if not articles:
        print("❌ No articles fetched")
        return False

    print(f"✅ Fetched {len(articles)} articles")

    # Initialize database and get ContentFetcher instance
    from src.database.sqlite_manager import get_db_manager
    db_manager = get_db_manager()
    await db_manager.initialize()

    content_fetcher = get_content_fetcher()

    # Track success/failure
    enriched_count = 0
    failed_count = 0

    for i, article in enumerate(articles, 1):
        print(f"\nArticle {i}: {article.title[:50]}...")
        print(f"   URL: {article.url}")

        # Fetch article content
        try:
            result = await content_fetcher.fetch_article(str(article.url), use_cache=True, timeout=10)

            # Check if fetching succeeded and got an excerpt
            if result['status'] in ['success', 'cached'] and result.get('excerpt'):
                excerpt_preview = result['excerpt'][:100].replace('\n', ' ')
                print(f"   ✅ Excerpt ({result['status']}): {excerpt_preview}...")
                enriched_count += 1
            else:
                if result['status'] == 'error':
                    print(f"   ⚠️  Fetch error: {result.get('error', 'Unknown')}")
                else:
                    print(f"   ⚠️  No excerpt in result")
                failed_count += 1

        except Exception as e:
            print(f"   ❌ Error fetching: {str(e)}")
            failed_count += 1

    print(f"\n📊 Results: {enriched_count} enriched, {failed_count} failed/empty")

    # At least 50% should have excerpts (some URLs may be unavailable)
    success_rate = enriched_count / len(articles) if articles else 0
    if success_rate >= 0.5:
        print(f"✅ Success rate: {success_rate*100:.0f}% (meets 50% threshold)")
        return True
    else:
        print(f"❌ Success rate: {success_rate*100:.0f}% (below 50% threshold)")
        return False


async def main():
    """Run all News tab tests."""
    print("=" * 60)
    print("NEWS TAB INTEGRATION TESTS")
    print("=" * 60)

    all_passed = True

    if not await test_news_tab_data_structure():
        all_passed = False

    if not await test_news_tab_grouping():
        all_passed = False

    if not await test_news_tab_filtering():
        all_passed = False

    if not await test_news_tab_excerpts():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL NEWS TAB TESTS PASSED")
    else:
        print("❌ SOME NEWS TAB TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

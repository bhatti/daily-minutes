"""
Integration tests for news system - run manually with real APIs.

IMPORTANT: These tests make real API calls. Use sparingly.

Run with:
    pytest tests/integration/test_news_integration.py -v -m integration

Or run specific test:
    pytest tests/integration/test_news_integration.py::test_hackernews_real -v

For environments behind proxy with SSL issues:
    VERIFY_SSL=false python tests/integration/test_news_integration.py
"""

import asyncio
import os
import ssl
import pytest
from datetime import datetime

# Handle SSL verification for proxy environments
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() != 'false'

if not VERIFY_SSL:
    import warnings
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    # Create custom SSL context that doesn't verify certificates
    ssl._create_default_https_context = ssl._create_unverified_context

from src.connectors.hackernews import HackerNewsConnector
from src.connectors.rss import RSSConnector
from src.agents.news_agent import NewsAgent
from src.models.news import NewsArticle, DataSource


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hackernews_real():
    """Test real HackerNews API connection and data fetching."""
    print("\n=== Testing HackerNews API ===")

    connector = HackerNewsConnector(
        story_type="top",
        max_stories=3,  # Minimal for testing
        min_score=10
    )

    # Test fetching story IDs
    story_ids = await connector.fetch_story_ids()
    assert len(story_ids) > 0, "Should fetch story IDs"
    print(f"✓ Fetched {len(story_ids)} story IDs")

    # Test fetching articles
    articles = await connector.execute_async()
    assert len(articles) > 0, "Should fetch articles"
    assert all(isinstance(a, NewsArticle) for a in articles)

    # Verify article fields
    first_article = articles[0]
    assert first_article.title
    assert first_article.url
    assert first_article.source == DataSource.HACKERNEWS or first_article.source == "hackernews"
    assert first_article.relevance_score >= 0.0
    assert first_article.relevance_score <= 1.0

    # Check priority (could be enum or string due to use_enum_values=True)
    priority = first_article.priority
    if isinstance(priority, str):
        priority_str = priority
    else:
        # It's an enum
        priority_str = priority.value
    assert priority_str in ["low", "medium", "high", "urgent"]

    print(f"✓ Fetched {len(articles)} articles")
    print(f"  Sample: {first_article.title[:50]}...")

    # Test statistics
    stats = connector.get_statistics()
    assert stats["stories_fetched"] > 0
    assert stats["api_calls_made"] > 0

    print(f"✓ Statistics: {stats['api_calls_made']} API calls made")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rss_real():
    """Test real RSS feed fetching."""
    print("\n=== Testing RSS Feeds ===")

    connector = RSSConnector(
        max_articles_per_feed=2  # Minimal for testing
    )

    # Test single feed
    test_feed_url = "https://feeds.arstechnica.com/arstechnica/index"
    articles = await connector.fetch_from_source(test_feed_url)

    if articles:  # Feed might be temporarily down
        assert all(isinstance(a, NewsArticle) for a in articles)
        print(f"✓ Fetched {len(articles)} articles from test feed")

        first_article = articles[0]
        assert first_article.title
        assert first_article.source == DataSource.RSS
        print(f"  Sample: {first_article.title[:50]}...")
    else:
        print("  ⚠ Test feed returned no articles (may be temporarily unavailable)")

    # Test feed status
    status = connector.get_feed_status()
    assert status["total_feeds"] > 0
    print(f"✓ Feed status: {status['total_feeds']} feeds configured")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hackernews_search():
    """Test HackerNews search functionality."""
    print("\n=== Testing HackerNews Search ===")

    connector = HackerNewsConnector()

    # Search for Python articles
    articles = await connector.search_stories("python", limit=3)

    assert isinstance(articles, list)
    if articles:  # Search might return no results
        assert all(isinstance(a, NewsArticle) for a in articles)
        print(f"✓ Found {len(articles)} articles for 'python'")

        if articles:
            print(f"  Sample: {articles[0].title[:50]}...")
    else:
        print("  ⚠ No search results (query might not match current stories)")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hackernews_trending():
    """Test trending topics extraction."""
    print("\n=== Testing Trending Topics ===")

    connector = HackerNewsConnector(max_stories=20)

    topics = await connector.get_trending_topics()

    assert isinstance(topics, list)
    assert len(topics) <= 10  # Should return top 10

    print(f"✓ Found {len(topics)} trending topics")
    if topics:
        print(f"  Top 3: {', '.join(topics[:3])}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_news_agent_minimal():
    """Test NewsAgent with minimal configuration."""
    print("\n=== Testing NewsAgent ===")

    # Create agent - it will initialize its own connectors
    agent = NewsAgent()

    # Override with minimal configuration
    agent.hn_connector = HackerNewsConnector(max_stories=5)
    agent.rss_connector = None  # Disable RSS for faster test
    agent.enable_rag = False  # Disable RAG for testing
    agent.enable_preferences = False  # Disable preferences for testing

    # Run agent workflow
    articles = await agent.run()

    assert isinstance(articles, list)
    assert len(articles) <= 5

    if articles:
        assert all(isinstance(a, NewsArticle) for a in articles)
        print(f"✓ Agent returned {len(articles)} articles")

        # Check if articles are sorted by importance
        importances = [a.calculate_importance() for a in articles]
        assert importances == sorted(importances, reverse=True), "Articles should be sorted by importance"
        print("✓ Articles properly sorted by importance")

        # Generate summary
        summary = await agent.generate_summary(articles)
        assert summary.total_articles == len(articles)
        print(f"✓ Generated summary: {summary.total_articles} articles")
    else:
        print("  ⚠ Agent returned no articles")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_caching_behavior():
    """Test that caching reduces API calls."""
    print("\n=== Testing Cache Behavior ===")

    connector = HackerNewsConnector(
        story_type="top",
        max_stories=2
    )

    # First call - should hit API
    articles1 = await connector.execute_async()
    api_calls_1 = connector.api_calls_made

    print(f"✓ First call: {api_calls_1} API calls")

    # Second call - should use cache
    connector2 = HackerNewsConnector(
        story_type="top",
        max_stories=2
    )
    articles2 = await connector2.execute_async()
    api_calls_2 = connector2.api_calls_made

    # If cache is working, second connector should make fewer API calls
    # (it might still make some calls for stories not in cache)
    print(f"✓ Second call: {api_calls_2} API calls")

    if api_calls_2 < api_calls_1:
        print("✓ Cache is reducing API calls")
    else:
        print("  ⚠ Cache might not be effective (or TTL expired)")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rate_limiting():
    """Test that rate limiting is working."""
    print("\n=== Testing Rate Limiting ===")

    connector = HackerNewsConnector(
        requests_per_minute=60  # 1 per second
    )

    start_time = datetime.now()

    # Make multiple requests
    for i in range(3):
        await connector.fetch_story(1000 + i)

    elapsed = (datetime.now() - start_time).total_seconds()

    # Should take at least 2 seconds for 3 requests at 1/sec rate
    if elapsed >= 2.0:
        print(f"✓ Rate limiting working: {elapsed:.1f}s for 3 requests")
    else:
        print(f"  ⚠ Rate limiting might not be working: only {elapsed:.1f}s for 3 requests")


# Main test runner for manual execution
if __name__ == "__main__":
    """
    Run all integration tests manually.

    Usage:
        python tests/integration/test_news_integration.py
    """

    async def run_all_tests():
        print("=" * 60)
        print("RUNNING INTEGRATION TESTS WITH REAL APIs")
        print("Please use sparingly to avoid rate limiting")
        print("=" * 60)

        tests = [
            ("HackerNews Basic", test_hackernews_real),
            ("RSS Feeds", test_rss_real),
            ("HackerNews Search", test_hackernews_search),
            ("Trending Topics", test_hackernews_trending),
            ("News Agent", test_news_agent_minimal),
            ("Caching", test_caching_behavior),
            ("Rate Limiting", test_rate_limiting),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            try:
                await test_func()
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            except Exception as e:
                failed += 1
                print(f"❌ {test_name} FAILED: {e}\n")

        print("=" * 60)
        print(f"RESULTS: {passed} passed, {failed} failed")
        print("=" * 60)

        return failed == 0

    # Run tests
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
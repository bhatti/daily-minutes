"""Unit tests for RSS connector with mocks."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import feedparser

from src.connectors.rss import RSSConnector
from src.models.news import NewsArticle, RSSFeed, DataSource, Priority


@pytest.fixture
def mock_rss_connector():
    """Create RSS connector with mocked cache."""
    with patch('src.connectors.rss.CacheManager'):
        connector = RSSConnector(
            max_articles_per_feed=5
        )
        # Mock cache methods
        connector.cache_manager.get = MagicMock(return_value=None)
        connector.cache_manager.put = MagicMock()
        return connector


@pytest.fixture
def sample_feed_content():
    """Sample RSS feed content."""
    return """<?xml version="1.0"?>
    <rss version="2.0">
        <channel>
            <title>Test Feed</title>
            <link>https://example.com</link>
            <description>Test Description</description>
            <item>
                <title>Article 1: Python AI Tutorial</title>
                <link>https://example.com/article1</link>
                <description>Learn Python for AI development</description>
                <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                <author>author1@example.com</author>
                <category>AI</category>
                <category>Python</category>
                <guid>article-1</guid>
            </item>
            <item>
                <title>Article 2: Machine Learning News</title>
                <link>https://example.com/article2</link>
                <description>Latest ML breakthroughs</description>
                <pubDate>Mon, 01 Jan 2024 11:00:00 GMT</pubDate>
                <author>author2@example.com</author>
                <category>Machine Learning</category>
                <guid>article-2</guid>
            </item>
        </channel>
    </rss>"""


@pytest.fixture
def parsed_feed(sample_feed_content):
    """Parse sample feed content."""
    return feedparser.parse(sample_feed_content)


@pytest.mark.asyncio
async def test_initialize_default_feeds(mock_rss_connector):
    """Test initialization with default feeds."""
    # Should have default feeds initialized
    assert len(mock_rss_connector.feeds) > 0

    # Check some expected default feeds
    feed_urls = [f.url for f in mock_rss_connector.feeds]
    assert "https://feeds.arstechnica.com/arstechnica/index" in feed_urls
    assert "https://techcrunch.com/feed/" in feed_urls

    # All feeds should be tech category
    assert all(f.category == "technology" for f in mock_rss_connector.feeds)


def test_add_feed(mock_rss_connector):
    """Test adding a new feed."""
    initial_count = len(mock_rss_connector.feeds)

    mock_rss_connector.add_feed(
        url="https://newsite.com/feed.xml",
        name="New Site",
        category="science"
    )

    assert len(mock_rss_connector.feeds) == initial_count + 1

    # Check the new feed
    new_feed = mock_rss_connector.feeds[-1]
    assert new_feed.url == "https://newsite.com/feed.xml"
    assert new_feed.name == "New Site"
    assert new_feed.category == "science"
    assert new_feed.max_articles == 5  # From connector's max_articles_per_feed


@pytest.mark.asyncio
async def test_fetch_from_source_with_cache_hit(mock_rss_connector):
    """Test fetching from source with cache hit."""
    # Setup cache hit
    cached_articles = [
        NewsArticle(
            title="Cached Article",
            url="https://example.com/cached",
            source=DataSource.RSS,
            source_name="Test Feed"
        )
    ]
    mock_rss_connector.cache_manager.get.return_value = cached_articles

    # Add a test feed
    test_feed = RSSFeed(
        name="Test Feed",
        url="https://test.com/feed.xml",
        category="test"
    )
    mock_rss_connector.feeds = [test_feed]

    # Fetch from source
    articles = await mock_rss_connector.fetch_from_source("https://test.com/feed.xml")

    assert articles == cached_articles
    assert len(articles) == 1
    mock_rss_connector.cache_manager.get.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_from_source_from_api(mock_rss_connector, sample_feed_content):
    """Test fetching from source via API (mocked)."""
    # Setup feed
    test_feed = RSSFeed(
        name="Test Feed",
        url="https://test.com/feed.xml",
        category="test"
    )
    mock_rss_connector.feeds = [test_feed]

    # Mock httpx response
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.text = sample_feed_content
        mock_response.raise_for_status = MagicMock()

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        # Fetch from source
        articles = await mock_rss_connector.fetch_from_source("https://test.com/feed.xml")

        assert len(articles) == 2
        assert articles[0].title == "Article 1: Python AI Tutorial"
        assert articles[1].title == "Article 2: Machine Learning News"

        # Check feed was updated
        assert test_feed.last_fetched is not None
        assert mock_rss_connector.feeds_processed == 1
        assert mock_rss_connector.articles_fetched == 2

        # Cache should be updated
        mock_rss_connector.cache_manager.put.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_with_feed_filters(mock_rss_connector, sample_feed_content):
    """Test fetching with feed filter keywords."""
    # Setup feed with filters
    test_feed = RSSFeed(
        name="Filtered Feed",
        url="https://test.com/feed.xml",
        category="test",
        filter_keywords=["python", "ai"],  # Only articles with these keywords
        exclude_keywords=["tutorial"]  # Exclude articles with this keyword
    )
    mock_rss_connector.feeds = [test_feed]

    # Mock httpx response
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.text = sample_feed_content
        mock_response.raise_for_status = MagicMock()

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        articles = await mock_rss_connector.fetch_from_source("https://test.com/feed.xml")

        # Article 1 has "python" and "ai" but also "tutorial" - should be excluded
        # Article 2 doesn't have required keywords - should be excluded
        assert len(articles) == 0


@pytest.mark.asyncio
async def test_entry_to_article_conversion(mock_rss_connector, parsed_feed):
    """Test converting RSS entry to NewsArticle."""
    test_feed = RSSFeed(
        name="Test Feed",
        url="https://test.com/feed.xml",
        category="technology"
    )

    entry = parsed_feed.entries[0]
    article = mock_rss_connector._entry_to_article(entry, test_feed)

    assert isinstance(article, NewsArticle)
    assert article.title == "Article 1: Python AI Tutorial"
    assert article.url == "https://example.com/article1"
    assert article.source == DataSource.RSS
    assert article.source_name == "Test Feed"
    assert article.author == "author1@example.com"
    assert "Learn Python for AI development" in article.description
    assert "ai" in article.tags
    assert "python" in article.tags
    assert "technology" in article.tags  # From feed category
    assert article.priority == Priority.MEDIUM
    assert article.metadata["feed_name"] == "Test Feed"
    assert article.metadata["guid"] == "article-1"


@pytest.mark.asyncio
async def test_calculate_relevance(mock_rss_connector):
    """Test relevance score calculation."""
    # High relevance - AI/ML keywords
    score1 = mock_rss_connector._calculate_relevance(
        "GPT-4 Breakthrough in Machine Learning",
        "New LLM model announced with funding",
        ["ai", "machine-learning"]
    )
    assert score1 > 0.8

    # Medium relevance - some keywords
    score2 = mock_rss_connector._calculate_relevance(
        "Python Tutorial",
        "Learn Python programming",
        ["python"]
    )
    assert 0.4 < score2 < 0.7

    # Low relevance - no special keywords
    score3 = mock_rss_connector._calculate_relevance(
        "Generic News",
        "Something happened today",
        []
    )
    assert score3 <= 0.5


@pytest.mark.asyncio
async def test_fetch_all_feeds(mock_rss_connector):
    """Test fetching from all feeds in parallel."""
    # Setup multiple feeds
    feed1 = RSSFeed(name="Feed1", url="https://feed1.com/rss")
    feed2 = RSSFeed(name="Feed2", url="https://feed2.com/rss")
    mock_rss_connector.feeds = [feed1, feed2]
    mock_rss_connector.sources = [feed1.url, feed2.url]

    # Mock fetch_from_source to return different articles
    async def mock_fetch(source):
        if source == feed1.url:
            return [NewsArticle(
                title="Article from Feed1",
                url="https://feed1.com/1",
                source=DataSource.RSS,
                source_name="Feed1",
                published_at=datetime(2024, 1, 1, 12, 0)
            )]
        else:
            return [NewsArticle(
                title="Article from Feed2",
                url="https://feed2.com/1",
                source=DataSource.RSS,
                source_name="Feed2",
                published_at=datetime(2024, 1, 1, 13, 0)
            )]

    mock_rss_connector.fetch_from_source = mock_fetch

    articles = await mock_rss_connector.fetch_all_feeds()

    assert len(articles) == 2
    # Should be sorted by published date (newest first)
    assert articles[0].title == "Article from Feed2"
    assert articles[1].title == "Article from Feed1"


@pytest.mark.asyncio
async def test_error_handling(mock_rss_connector):
    """Test error handling in fetch operations."""
    test_feed = RSSFeed(
        name="Error Feed",
        url="https://error.com/feed.xml",
        category="test"
    )
    mock_rss_connector.feeds = [test_feed]

    # Simulate connection error
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        articles = await mock_rss_connector.fetch_from_source("https://error.com/feed.xml")

        assert articles == []
        assert test_feed.error_count == 1
        assert test_feed.last_error == "Connection failed"


@pytest.mark.asyncio
async def test_parse_error_handling(mock_rss_connector):
    """Test handling of RSS parse errors."""
    test_feed = RSSFeed(
        name="Bad Feed",
        url="https://bad.com/feed.xml",
        category="test"
    )
    mock_rss_connector.feeds = [test_feed]

    # Invalid RSS content
    bad_rss = "This is not valid RSS/XML"

    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.text = bad_rss
        mock_response.raise_for_status = MagicMock()

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        articles = await mock_rss_connector.fetch_from_source("https://bad.com/feed.xml")

        assert articles == []
        assert test_feed.error_count == 1
        assert "Parse error" in test_feed.last_error


@pytest.mark.asyncio
async def test_feed_should_fetch_logic(mock_rss_connector):
    """Test feed fetch timing logic."""
    test_feed = RSSFeed(
        name="Timed Feed",
        url="https://timed.com/feed.xml",
        category="test",
        update_frequency=3600  # 1 hour
    )
    mock_rss_connector.feeds = [test_feed]

    # First time should fetch
    assert test_feed.should_fetch() is True

    # Record fetch
    test_feed.record_fetch()

    # Immediately after should not fetch
    with patch('httpx.AsyncClient'):
        articles = await mock_rss_connector.fetch_from_source("https://timed.com/feed.xml")
        assert articles == []  # Skipped due to timing


def test_get_feed_status(mock_rss_connector):
    """Test getting feed status information."""
    # Setup feeds with different states
    feed1 = RSSFeed(name="Active", url="https://active.com/rss", is_active=True)
    feed2 = RSSFeed(name="Inactive", url="https://inactive.com/rss", is_active=False)
    feed3 = RSSFeed(name="Error", url="https://error.com/rss", is_active=True)
    feed3.record_error("Test error")

    mock_rss_connector.feeds = [feed1, feed2, feed3]
    mock_rss_connector.feeds_processed = 10
    mock_rss_connector.articles_fetched = 50

    status = mock_rss_connector.get_feed_status()

    assert status["total_feeds"] == 3
    assert status["active_feeds"] == 2
    assert status["feeds_with_errors"] == 1
    assert status["feeds_processed"] == 10
    assert status["articles_fetched"] == 50
    assert len(status["feed_details"]) == 3

    # Check feed details
    error_feed = next(f for f in status["feed_details"] if f["name"] == "Error")
    assert error_feed["error_count"] == 1
    assert error_feed["last_error"] == "Test error"


@pytest.mark.asyncio
async def test_html_cleaning_in_description(mock_rss_connector):
    """Test HTML tag removal from descriptions."""
    feed_with_html = """<?xml version="1.0"?>
    <rss version="2.0">
        <channel>
            <title>Test</title>
            <item>
                <title>Article</title>
                <link>https://example.com/1</link>
                <description><p>This is <strong>HTML</strong> content with <a href="#">links</a></p></description>
                <guid>1</guid>
            </item>
        </channel>
    </rss>"""

    test_feed = RSSFeed(name="HTML Feed", url="https://html.com/feed.xml")
    mock_rss_connector.feeds = [test_feed]

    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.text = feed_with_html
        mock_response.raise_for_status = MagicMock()

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        articles = await mock_rss_connector.fetch_from_source("https://html.com/feed.xml")

        assert len(articles) == 1
        # HTML tags should be removed
        assert "<p>" not in articles[0].description
        assert "<strong>" not in articles[0].description
        assert "This is HTML content with links" in articles[0].description
"""Unit tests for HackerNews connector with mocks."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.connectors.hackernews import HackerNewsConnector
from src.models.news import NewsArticle, DataSource, Priority


@pytest.fixture
def mock_hn_connector():
    """Create HackerNews connector with mocked cache."""
    from src.models.cache import CacheManager

    connector = HackerNewsConnector(
        story_type="top",
        max_stories=10,
        min_score=50
    )
    # Mock cache methods on the real cache manager
    connector.cache_manager.get = MagicMock(return_value=None)
    connector.cache_manager.put = MagicMock()
    return connector


@pytest.fixture
def sample_story():
    """Sample HackerNews story data."""
    return {
        "id": 12345,
        "type": "story",
        "title": "Test Story About Python and AI",
        "url": "https://example.com/story",
        "by": "testuser",
        "time": 1700000000,
        "score": 150,
        "descendants": 75,
        "text": None
    }


@pytest.mark.asyncio
async def test_fetch_story_ids_from_api(mock_hn_connector):
    """Test fetching story IDs from API (mocked)."""
    # Mock httpx response
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.json.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        mock_response.raise_for_status = MagicMock()

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        story_ids = await mock_hn_connector.fetch_story_ids()

        assert len(story_ids) == 10  # Limited by max_stories
        assert story_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Check cache was updated
        mock_hn_connector.cache_manager.put.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_story_ids_from_cache(mock_hn_connector):
    """Test fetching story IDs from cache."""
    # Mock cache hit
    cached_ids = [100, 101, 102]
    mock_hn_connector.cache_manager.get.return_value = cached_ids

    story_ids = await mock_hn_connector.fetch_story_ids()

    assert story_ids == cached_ids
    # Should not make API call
    mock_hn_connector.cache_manager.put.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_story(mock_hn_connector, sample_story):
    """Test fetching individual story."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.json.return_value = sample_story
        mock_response.raise_for_status = MagicMock()

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        story = await mock_hn_connector.fetch_story(12345)

        assert story == sample_story
        assert story["title"] == "Test Story About Python and AI"
        assert story["score"] == 150


@pytest.mark.asyncio
async def test_story_to_article_conversion(mock_hn_connector, sample_story):
    """Test converting HackerNews story to NewsArticle."""
    article = mock_hn_connector.story_to_article(sample_story)

    assert isinstance(article, NewsArticle)
    assert article.title == "Test Story About Python and AI"
    assert article.url == "https://example.com/story"
    assert article.source == DataSource.HACKERNEWS
    assert article.source_name == "HackerNews"
    assert article.author == "testuser"
    assert article.priority == Priority.MEDIUM  # score=150, descendants=75
    assert article.relevance_score > 0.0 and article.relevance_score <= 1.0
    assert article.metadata["hn_id"] == 12345
    assert article.metadata["score"] == 150
    assert article.metadata["comments"] == 75


@pytest.mark.asyncio
async def test_story_to_article_high_priority(mock_hn_connector):
    """Test high priority story conversion."""
    high_score_story = {
        "id": 99999,
        "type": "story",
        "title": "Major Breakthrough",
        "url": "https://example.com/breakthrough",
        "by": "famous",
        "time": 1700000000,
        "score": 600,  # High score
        "descendants": 200,  # Many comments
    }

    article = mock_hn_connector.story_to_article(high_score_story)
    assert article.priority == Priority.HIGH


@pytest.mark.asyncio
async def test_extract_tags(mock_hn_connector):
    """Test tag extraction from story."""
    # Story with tech keywords
    story_with_keywords = {
        "id": 1,
        "type": "story",
        "title": "Python Machine Learning with Docker on AWS",
        "url": "https://example.com",
        "by": "user",
        "time": 1700000000,
        "score": 100,
        "descendants": 10
    }

    article = mock_hn_connector.story_to_article(story_with_keywords)

    assert "python" in article.tags
    assert "machine learning" in article.tags
    assert "docker" in article.tags
    assert "aws" in article.tags


@pytest.mark.asyncio
async def test_extract_special_tags(mock_hn_connector):
    """Test extraction of special HN tags."""
    # Ask HN story
    ask_story = {
        "id": 1,
        "type": "story",
        "title": "Ask HN: How do you manage technical debt?",
        "url": None,
        "by": "user",
        "time": 1700000000,
        "score": 50,
        "descendants": 20,
        "text": "Question details here"
    }

    article = mock_hn_connector.story_to_article(ask_story)
    assert "ask-hn" in article.tags

    # Show HN story
    show_story = {
        "id": 2,
        "type": "story",
        "title": "Show HN: My new project",
        "url": "https://project.com",
        "by": "creator",
        "time": 1700000000,
        "score": 75,
        "descendants": 15
    }

    article = mock_hn_connector.story_to_article(show_story)
    assert "show-hn" in article.tags


@pytest.mark.asyncio
async def test_process_async_workflow(mock_hn_connector, sample_story):
    """Test the full async processing workflow."""
    # Mock fetch_story_ids
    mock_hn_connector.fetch_story_ids = AsyncMock(return_value=[1, 2, 3])

    # Mock fetch_story
    mock_hn_connector.fetch_story = AsyncMock(return_value=sample_story)

    # Execute workflow
    articles = await mock_hn_connector.execute_async()

    assert len(articles) == 3
    assert all(isinstance(a, NewsArticle) for a in articles)
    assert mock_hn_connector.stories_fetched == 3


@pytest.mark.asyncio
async def test_filter_by_min_score(mock_hn_connector):
    """Test filtering stories by minimum score."""
    low_score_story = {
        "id": 1,
        "type": "story",
        "title": "Low Score Story",
        "url": "https://example.com",
        "by": "user",
        "time": 1700000000,
        "score": 10,  # Below min_score of 50
        "descendants": 5
    }

    # Mock fetch_story to return low score story
    mock_hn_connector.fetch_story = AsyncMock(return_value=low_score_story)

    article = await mock_hn_connector.process_item_async(1)
    assert article is None  # Should be filtered out


@pytest.mark.asyncio
async def test_rate_limiting(mock_hn_connector):
    """Test rate limiting behavior."""
    # Set last request time to now
    mock_hn_connector.last_request_time = datetime.now()
    mock_hn_connector.requests_per_minute = 60  # 1 per second

    # Mock sleep to track if it was called
    with patch('asyncio.sleep') as mock_sleep:
        mock_sleep.return_value = None

        # This should trigger rate limiting
        await mock_hn_connector._rate_limit()

        # Sleep should have been called since not enough time passed
        mock_sleep.assert_called()


@pytest.mark.asyncio
async def test_get_statistics(mock_hn_connector):
    """Test statistics gathering."""
    mock_hn_connector.stories_fetched = 100
    mock_hn_connector.api_calls_made = 110

    stats = mock_hn_connector.get_statistics()

    assert stats["stories_fetched"] == 100
    assert stats["api_calls_made"] == 110
    assert stats["story_type"] == "top"
    assert "cache_stats" in stats
    assert "performance" in stats


@pytest.mark.asyncio
async def test_search_stories(mock_hn_connector):
    """Test search functionality using Algolia API."""
    search_response = {
        "hits": [
            {
                "objectID": "123",
                "title": "Search Result 1",
                "url": "https://result1.com",
                "author": "user1",
                "created_at_i": 1700000000,
                "points": 100,
                "num_comments": 50
            },
            {
                "objectID": "124",
                "title": "Search Result 2",
                "url": "https://result2.com",
                "author": "user2",
                "created_at_i": 1700000100,
                "points": 200,
                "num_comments": 75
            }
        ]
    }

    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.json.return_value = search_response
        mock_response.raise_for_status = MagicMock()

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        articles = await mock_hn_connector.search_stories("python", limit=2)

        assert len(articles) == 2
        assert articles[0].title == "Search Result 1"
        assert articles[1].title == "Search Result 2"
        assert all(isinstance(a, NewsArticle) for a in articles)


@pytest.mark.asyncio
async def test_error_handling(mock_hn_connector):
    """Test error handling in fetch operations."""
    with patch('httpx.AsyncClient') as mock_client:
        # Simulate connection error
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        story_ids = await mock_hn_connector.fetch_story_ids()
        assert story_ids == []  # Should return empty list on error

        story = await mock_hn_connector.fetch_story(12345)
        assert story is None  # Should return None on error
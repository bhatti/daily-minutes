#!/usr/bin/env python3
"""Unit tests for SQLite Manager using in-memory database."""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from ulid import ULID

from src.database.sqlite_manager import SQLiteManager
from src.models.news import NewsArticle, DataSource, Priority


class TestSQLiteManager:
    """Test SQLiteManager with in-memory database."""

    @pytest_asyncio.fixture
    async def db_manager(self):
        """Create in-memory database manager for testing."""
        manager = SQLiteManager(db_path=":memory:")
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_database_initialization(self, db_manager):
        """Test database initializes with correct schema."""
        # Should not raise any errors
        assert db_manager._initialized is True

        # Test double initialization (should be safe)
        await db_manager.initialize()
        assert db_manager._initialized is True

    @pytest.mark.asyncio
    async def test_generate_identifier_with_url(self):
        """Test identifier generation with URL."""
        url = "https://example.com/article"
        identifier = SQLiteManager.generate_identifier("article", url)

        assert identifier == url

    @pytest.mark.asyncio
    async def test_generate_identifier_without_url(self):
        """Test identifier generation without URL (uses ULID)."""
        identifier = SQLiteManager.generate_identifier("note")

        # Should have format "note:ULID"
        assert identifier.startswith("note:")

        # Extract ULID part and validate
        ulid_part = identifier.split(":", 1)[1]
        assert len(ulid_part) == 26  # ULID length

        # Should be able to parse as ULID
        parsed_ulid = ULID.from_str(ulid_part)
        assert parsed_ulid is not None

    @pytest.mark.asyncio
    async def test_ulid_ordering(self):
        """Test that ULIDs maintain time ordering."""
        import time

        id1 = SQLiteManager.generate_identifier("note")
        time.sleep(0.001)  # Small delay
        id2 = SQLiteManager.generate_identifier("note")

        # ULIDs should be sortable by time
        assert id1 < id2  # Lexicographically sorted = time sorted

    @pytest.mark.asyncio
    async def test_save_and_get_content(self, db_manager):
        """Test saving and retrieving content."""
        content_id = await db_manager.save_content(
            title="Test Article",
            source="hackernews",
            url="https://example.com/test",
            content_type="article",
            processed_content="This is test content",
            summary="Test summary",
            key_points=["Point 1", "Point 2"],
            metadata={"author": "John Doe", "score": 100}
        )

        assert content_id is not None

        # Get by identifier (URL in this case)
        content = await db_manager.get_content("https://example.com/test")

        assert content is not None
        assert content["title"] == "Test Article"
        assert content["source"] == "hackernews"
        assert content["processed_content"] == "This is test content"
        assert content["summary"] == "Test summary"
        assert content["access_count"] == 1  # Incremented on get

        # Verify JSON fields
        import json
        metadata = json.loads(content["metadata"])
        assert metadata["author"] == "John Doe"
        assert metadata["score"] == 100

        key_points = json.loads(content["key_points"])
        assert len(key_points) == 2
        assert "Point 1" in key_points

    @pytest.mark.asyncio
    async def test_save_content_without_url(self, db_manager):
        """Test saving content without URL (generates ULID identifier)."""
        content_id = await db_manager.save_content(
            title="My Note",
            source="manual",
            content_type="note",
            processed_content="Note content here"
        )

        assert content_id is not None

        # Content should exist in database
        stats = await db_manager.get_content_stats()
        assert stats["total_content"] == 1

    @pytest.mark.asyncio
    async def test_update_existing_content(self, db_manager):
        """Test updating existing content."""
        url = "https://example.com/update-test"

        # Save initial content
        await db_manager.save_content(
            title="Original Title",
            source="test",
            url=url,
            summary="Original summary"
        )

        # Update with same URL
        await db_manager.save_content(
            title="Updated Title",
            source="test",
            url=url,
            summary="Updated summary"
        )

        # Should still be one record
        content = await db_manager.get_content(url)
        assert content["title"] == "Updated Title"
        assert content["summary"] == "Updated summary"

        # Verify only one record exists
        stats = await db_manager.get_content_stats()
        assert stats["total_content"] == 1

    @pytest.mark.asyncio
    async def test_get_content_by_url(self, db_manager):
        """Test getting content by URL."""
        url = "https://example.com/url-test"

        await db_manager.save_content(
            title="URL Test",
            source="test",
            url=url
        )

        # Get by URL specifically
        content = await db_manager.get_content_by_url(url)
        assert content is not None
        assert content["url"] == url

        # Nonexistent URL
        content = await db_manager.get_content_by_url("https://nonexistent.com")
        assert content is None

    @pytest.mark.asyncio
    async def test_save_article(self, db_manager):
        """Test saving NewsArticle object."""
        article = NewsArticle(
            title="Test HN Article",
            url="https://news.ycombinator.com/item?id=123",
            source=DataSource.HACKERNEWS,
            source_name="HackerNews",
            author="testuser",
            published_at=datetime.now(),
            description="Article description",
            tags=["tech", "ai"],
            priority=Priority.HIGH,
            relevance_score=0.9
        )

        content_id = await db_manager.save_article(article)
        assert content_id is not None

        # Retrieve and verify (convert URL to string)
        content = await db_manager.get_content(str(article.url))
        assert content["title"] == "Test HN Article"
        assert content["content_type"] == "article"

        # Check metadata
        import json
        metadata = json.loads(content["metadata"])
        assert metadata["author"] == "testuser"
        assert "tech" in metadata["tags"]
        assert metadata["relevance_score"] == 0.9

    @pytest.mark.asyncio
    async def test_access_tracking(self, db_manager):
        """Test that access count and last_accessed are updated."""
        url = "https://example.com/access-test"

        await db_manager.save_content(
            title="Access Test",
            source="test",
            url=url
        )

        # Access multiple times
        for i in range(3):
            content = await db_manager.get_content(url)
            assert content["access_count"] == i + 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_content(self, db_manager):
        """Test cleanup of expired content."""
        # Save content that expires immediately
        await db_manager.save_content(
            title="Expired Content",
            source="test",
            url="https://example.com/expired",
            expires_in_days=-1  # Already expired
        )

        # Save content that doesn't expire yet
        await db_manager.save_content(
            title="Valid Content",
            source="test",
            url="https://example.com/valid",
            expires_in_days=30
        )

        stats = await db_manager.get_content_stats()
        assert stats["total_content"] == 2
        assert stats["expired_content"] == 1

        # Cleanup expired
        deleted = await db_manager.cleanup_expired_content()
        assert deleted == 1

        # Verify only valid content remains
        stats = await db_manager.get_content_stats()
        assert stats["total_content"] == 1

        content = await db_manager.get_content("https://example.com/valid")
        assert content is not None

    @pytest.mark.asyncio
    async def test_kv_store_settings(self, db_manager):
        """Test key-value store for settings."""
        # Set settings
        await db_manager.set_setting("theme", "dark")
        await db_manager.set_setting("max_articles", 50)
        await db_manager.set_setting("preferences", {"lang": "en", "tz": "UTC"})

        # Get settings
        theme = await db_manager.get_setting("theme")
        assert theme == "dark"

        max_articles = await db_manager.get_setting("max_articles")
        assert max_articles == 50

        prefs = await db_manager.get_setting("preferences")
        assert prefs["lang"] == "en"

        # Default value
        unknown = await db_manager.get_setting("unknown_key", "default")
        assert unknown == "default"

    @pytest.mark.asyncio
    async def test_kv_store_cache(self, db_manager):
        """Test key-value store for caching with expiration."""
        # Set cache with expiration
        await db_manager.set_cache("api_response", {"data": "test"}, expires_in_seconds=1)

        # Should be available immediately
        cached = await db_manager.get_cache("api_response")
        assert cached == {"data": "test"}

        # Wait for expiration
        import time
        time.sleep(1.5)

        # Should be None after expiration
        cached = await db_manager.get_cache("api_response")
        assert cached is None

    @pytest.mark.asyncio
    async def test_kv_store_permanent_cache(self, db_manager):
        """Test cache without expiration."""
        await db_manager.set_cache("permanent_data", {"value": 123})

        # Should always be available
        cached = await db_manager.get_cache("permanent_data")
        assert cached == {"value": 123}

    @pytest.mark.asyncio
    async def test_content_stats(self, db_manager):
        """Test content statistics."""
        # Add various content
        await db_manager.save_content(
            title="HN Article 1",
            source="hackernews",
            url="https://hn.com/1",
            content_type="article"
        )
        await db_manager.save_content(
            title="HN Article 2",
            source="hackernews",
            url="https://hn.com/2",
            content_type="article"
        )
        await db_manager.save_content(
            title="RSS Article",
            source="rss",
            url="https://rss.com/1",
            content_type="article"
        )
        await db_manager.save_content(
            title="Note",
            source="manual",
            content_type="note"
        )

        stats = await db_manager.get_content_stats()

        assert stats["total_content"] == 4
        assert len(stats["by_source"]) > 0

        # Find hackernews stats
        hn_stats = [s for s in stats["by_source"] if s["source"] == "hackernews"]
        assert len(hn_stats) == 1
        assert hn_stats[0]["count"] == 2

    @pytest.mark.asyncio
    async def test_content_hash_generation(self, db_manager):
        """Test content hash for deduplication."""
        url1 = "https://example.com/hash1"
        url2 = "https://example.com/hash2"

        # Save same content with different URLs
        await db_manager.save_content(
            title="Same Content",
            source="test",
            url=url1,
            processed_content="Identical content"
        )
        await db_manager.save_content(
            title="Same Content",
            source="test",
            url=url2,
            processed_content="Identical content"
        )

        content1 = await db_manager.get_content(url1)
        content2 = await db_manager.get_content(url2)

        # Should have same content hash
        assert content1["content_hash"] == content2["content_hash"]

    @pytest.mark.asyncio
    async def test_timestamps(self, db_manager):
        """Test that timestamps are set correctly."""
        url = "https://example.com/timestamps"

        await db_manager.save_content(
            title="Timestamp Test",
            source="test",
            url=url,
            published_at=datetime(2024, 1, 1)
        )

        content = await db_manager.get_content(url)

        # Check timestamps exist
        assert content["created_at"] is not None
        assert content["updated_at"] is not None
        assert content["fetched_at"] is not None
        assert content["published_at"] is not None

        # Update content - sleep longer because SQLite CURRENT_TIMESTAMP has 1-second resolution
        import asyncio
        await asyncio.sleep(1.1)
        await db_manager.save_content(
            title="Updated Timestamp Test",
            source="test",
            url=url
        )

        updated_content = await db_manager.get_content(url)

        # updated_at should change
        assert updated_content["updated_at"] != content["updated_at"]

    @pytest.mark.asyncio
    async def test_multiple_content_types(self, db_manager):
        """Test storing different content types."""
        # Article
        await db_manager.save_content(
            title="Article",
            source="test",
            url="https://example.com/article",
            content_type="article"
        )

        # Note
        await db_manager.save_content(
            title="Note",
            source="manual",
            content_type="note"
        )

        # Summary
        await db_manager.save_content(
            title="Summary",
            source="ai",
            content_type="summary"
        )

        stats = await db_manager.get_content_stats()
        assert stats["total_content"] == 3

    @pytest.mark.asyncio
    async def test_get_cache_age_hours_no_cache(self, db_manager):
        """Test cache age returns None when no cache exists."""
        age = await db_manager.get_cache_age_hours()
        assert age is None

    @pytest.mark.asyncio
    async def test_get_cache_age_hours_with_cache(self, db_manager):
        """Test cache age calculation."""
        # Add a cache entry
        await db_manager.set_cache("test_key", {"data": "value"}, expires_in_seconds=3600)

        # Should return age close to 0 (just created)
        age = await db_manager.get_cache_age_hours()
        assert age is not None
        assert age < 0.1  # Less than 6 minutes old

    @pytest.mark.asyncio
    async def test_get_news_cache_age_hours_no_cache(self, db_manager):
        """Test news cache age returns None when no news cache exists."""
        age = await db_manager.get_news_cache_age_hours()
        assert age is None

    @pytest.mark.asyncio
    async def test_get_news_cache_age_hours_with_hackernews(self, db_manager):
        """Test news cache age with HackerNews cache."""
        # Add HackerNews cache entry
        await db_manager.set_cache("hackernews_top", {"stories": []}, expires_in_seconds=3600)

        age = await db_manager.get_news_cache_age_hours()
        assert age is not None
        assert age < 0.1

    @pytest.mark.asyncio
    async def test_get_news_cache_age_hours_with_rss(self, db_manager):
        """Test news cache age with RSS cache."""
        # Add RSS cache entry
        await db_manager.set_cache("rss_feed_123", {"articles": []}, expires_in_seconds=3600)

        age = await db_manager.get_news_cache_age_hours()
        assert age is not None
        assert age < 0.1

    @pytest.mark.asyncio
    async def test_get_weather_cache_age_hours_no_cache(self, db_manager):
        """Test weather cache age returns None when no weather cache exists."""
        age = await db_manager.get_weather_cache_age_hours()
        assert age is None

    @pytest.mark.asyncio
    async def test_get_weather_cache_age_hours_with_cache(self, db_manager):
        """Test weather cache age with weather cache."""
        # Add weather cache entry
        await db_manager.set_cache("weather_Seattle", {"temp": 15}, expires_in_seconds=3600)

        age = await db_manager.get_weather_cache_age_hours()
        assert age is not None
        assert age < 0.1

    @pytest.mark.asyncio
    async def test_cache_age_ignores_expired(self, db_manager):
        """Test cache age methods ignore expired entries."""
        # Add an expired cache entry (already expired)
        await db_manager.set_cache("old_key", {"data": "old"}, expires_in_seconds=-1)

        # Add a valid entry
        await db_manager.set_cache("new_key", {"data": "new"}, expires_in_seconds=3600)

        # Should return age of valid entry (close to 0), not the expired one
        age = await db_manager.get_cache_age_hours()
        assert age is not None
        assert age < 0.1  # Should be the new entry, not the expired one


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

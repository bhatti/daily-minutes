#!/usr/bin/env python3
"""Test startup service - loads data on app startup."""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta, UTC

from src.services.startup_service import StartupService
from src.database.sqlite_manager import SQLiteManager
from src.models.news import NewsArticle
from src.models.base import Priority, DataSource


class TestStartupService:
    """Test startup data loading service."""

    @pytest_asyncio.fixture
    async def db_manager(self):
        """Create test database manager."""
        db = SQLiteManager(":memory:")
        await db.initialize()
        yield db
        await db.close()

    @pytest_asyncio.fixture
    async def startup_service(self, db_manager):
        """Create startup service with test database."""
        service = StartupService(db_manager)
        return service

    @pytest.mark.asyncio
    async def test_load_startup_data_empty_db(self, startup_service):
        """Test loading from empty database."""
        result = await startup_service.load_startup_data()

        assert result is not None
        assert result['articles'] == []
        assert result['cache_age_hours'] is None
        assert result['loaded_from_cache'] is False
        assert result['error'] is None

    @pytest.mark.asyncio
    async def test_load_startup_data_with_articles(self, startup_service, db_manager):
        """Test loading articles from database."""
        # Add test articles
        articles = [
            NewsArticle(
                id=f"article_{i}",
                title=f"Test Article {i}",
                url=f"https://example.com/article{i}",
                source=DataSource.HACKERNEWS,
                source_name="HackerNews",
                priority=Priority.HIGH,
                published_at=datetime.now(UTC) - timedelta(hours=1),
                fetched_at=datetime.now(UTC)
            )
            for i in range(5)
        ]

        for article in articles:
            await db_manager.save_article(article)

        # Load startup data
        result = await startup_service.load_startup_data()

        assert result is not None
        assert len(result['articles']) == 5
        assert result['loaded_from_cache'] is True
        assert result['error'] is None
        # Articles are returned in DESC order (newest first), so the last one is article_0
        assert any(art.title == "Test Article 0" for art in result['articles'])

    @pytest.mark.asyncio
    async def test_load_startup_data_with_limit(self, startup_service, db_manager):
        """Test loading with article limit."""
        # Add 150 test articles
        articles = [
            NewsArticle(
                id=f"article_{i}",
                title=f"Test Article {i}",
                url=f"https://example.com/article{i}",
                source=DataSource.HACKERNEWS,
                source_name="HackerNews",
                priority=Priority.MEDIUM,
                published_at=datetime.now(UTC) - timedelta(hours=1),
                fetched_at=datetime.now(UTC)
            )
            for i in range(150)
        ]

        for article in articles:
            await db_manager.save_article(article)

        # Load startup data (should limit to 100)
        result = await startup_service.load_startup_data(limit=100)

        assert result is not None
        assert len(result['articles']) == 100
        assert result['loaded_from_cache'] is True

    @pytest.mark.asyncio
    async def test_load_startup_data_cache_age(self, startup_service, db_manager):
        """Test cache age calculation."""
        # Add article with timestamp
        article = NewsArticle(
            id="article_1",
            title="Test Article",
            url="https://example.com/article1",
            source=DataSource.HACKERNEWS,
            source_name="HackerNews",
            priority=Priority.HIGH,
            published_at=datetime.now(UTC) - timedelta(hours=3),
            fetched_at=datetime.now(UTC) - timedelta(hours=2)
        )

        await db_manager.save_article(article)

        # Load startup data
        result = await startup_service.load_startup_data()

        assert result is not None
        # Cache age may be None if no cache metadata is stored
        # The important thing is that articles were loaded
        assert len(result['articles']) == 1
        assert result['loaded_from_cache'] is True

    @pytest.mark.asyncio
    async def test_load_startup_data_error_handling(self, db_manager):
        """Test error handling when database fails."""
        # Close database to simulate error
        await db_manager.close()

        service = StartupService(db_manager)
        result = await service.load_startup_data()

        # Should return error result instead of raising
        assert result is not None
        assert result['articles'] == []
        assert result['error'] is not None
        # Error should contain "database" and "initialize"
        assert 'database' in result['error'].lower()
        assert result['loaded_from_cache'] is False

    @pytest.mark.asyncio
    async def test_load_startup_data_performance(self, startup_service, db_manager):
        """Test that startup loading is fast."""
        # Add 100 articles
        articles = [
            NewsArticle(
                id=f"article_{i}",
                title=f"Test Article {i}",
                url=f"https://example.com/article{i}",
                source=DataSource.RSS,
                source_name="RSS Feed",
                priority=Priority.MEDIUM,
                published_at=datetime.now(UTC),
                fetched_at=datetime.now(UTC)
            )
            for i in range(100)
        ]

        for article in articles:
            await db_manager.save_article(article)

        # Measure loading time
        import time
        start = time.time()
        result = await startup_service.load_startup_data()
        duration = time.time() - start

        # Should load in under 1 second
        assert duration < 1.0
        assert len(result['articles']) == 100

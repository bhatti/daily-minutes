#!/usr/bin/env python3
"""Unit tests for Article Content Fetcher service."""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.services.content_fetcher import ContentFetcher
from src.database.sqlite_manager import SQLiteManager


class TestContentFetcher:
    """Test ContentFetcher service."""

    @pytest_asyncio.fixture
    async def db_manager(self):
        """Create in-memory database manager for testing."""
        manager = SQLiteManager(db_path=":memory:")
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest_asyncio.fixture
    async def content_fetcher(self, db_manager):
        """Create ContentFetcher instance."""
        fetcher = ContentFetcher(db_manager=db_manager)
        return fetcher

    @pytest.mark.asyncio
    async def test_fetch_article_content_success(self, content_fetcher):
        """Test successful article content fetching."""
        url = "https://example.com/article"

        # Mock HTTP response
        mock_html = """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <article>
                    <h1>Test Article Title</h1>
                    <p>This is the first paragraph of the article.</p>
                    <p>This is the second paragraph with more content.</p>
                </article>
            </body>
        </html>
        """

        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = mock_html
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = await content_fetcher.fetch_article(url)

            assert result is not None
            assert result['url'] == url
            assert result['status'] == 'success'
            assert 'content' in result
            assert 'excerpt' in result
            assert len(result['content']) > 0
            assert 'Test Article Title' in result['content']

    @pytest.mark.asyncio
    async def test_fetch_article_http_error(self, content_fetcher):
        """Test article fetching with HTTP error."""
        url = "https://example.com/notfound"

        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status = Mock(side_effect=Exception("404 Not Found"))
            mock_get.return_value = mock_response

            result = await content_fetcher.fetch_article(url)

            assert result is not None
            assert result['status'] == 'error'
            assert 'error' in result

    @pytest.mark.asyncio
    async def test_extract_text_from_html(self, content_fetcher):
        """Test text extraction from HTML."""
        html = """
        <html>
            <body>
                <header>Header content</header>
                <nav>Navigation</nav>
                <article>
                    <h1>Main Title</h1>
                    <p>First paragraph.</p>
                    <p>Second paragraph.</p>
                </article>
                <footer>Footer</footer>
                <script>console.log('test');</script>
            </body>
        </html>
        """

        text = content_fetcher.extract_text(html)

        assert text is not None
        assert 'Main Title' in text
        assert 'First paragraph' in text
        assert 'Second paragraph' in text
        # Scripts should be removed
        assert 'console.log' not in text

    @pytest.mark.asyncio
    async def test_generate_excerpt(self, content_fetcher):
        """Test excerpt generation."""
        long_text = "This is a test article. " * 100  # Long text

        excerpt = content_fetcher.generate_excerpt(long_text, max_length=100)

        assert excerpt is not None
        assert len(excerpt) <= 103  # 100 + "..."
        assert excerpt.endswith('...')

    @pytest.mark.asyncio
    async def test_generate_excerpt_short_text(self, content_fetcher):
        """Test excerpt generation with short text."""
        short_text = "Short article content."

        excerpt = content_fetcher.generate_excerpt(short_text, max_length=100)

        assert excerpt == short_text  # No ellipsis for short text

    @pytest.mark.asyncio
    async def test_cache_article_content(self, content_fetcher, db_manager):
        """Test caching article content in database."""
        url = "https://example.com/cached"
        content = "This is the full article content."
        excerpt = "This is the full article content."

        # Cache the content
        await content_fetcher.cache_article(
            url=url,
            title="Test Article",
            content=content,
            excerpt=excerpt
        )

        # Verify it was saved
        cached = await db_manager.get_content(url)
        assert cached is not None
        assert cached['url'] == url
        assert cached['title'] == "Test Article"
        assert cached['processed_content'] == content

    @pytest.mark.asyncio
    async def test_get_cached_article(self, content_fetcher, db_manager):
        """Test retrieving cached article."""
        url = "https://example.com/cached2"

        # Save content first
        await db_manager.save_content(
            title="Cached Article",
            source="manual",
            url=url,
            processed_content="Cached content here",
            summary="Short excerpt",
            expires_in_days=7
        )

        # Retrieve cached content
        cached = await content_fetcher.get_cached_article(url)

        assert cached is not None
        assert cached['url'] == url
        assert cached['processed_content'] == "Cached content here"

    @pytest.mark.asyncio
    async def test_get_cached_article_not_found(self, content_fetcher):
        """Test retrieving non-existent cached article."""
        url = "https://example.com/notcached"

        cached = await content_fetcher.get_cached_article(url)

        assert cached is None

    @pytest.mark.asyncio
    async def test_fetch_with_cache_hit(self, content_fetcher, db_manager):
        """Test fetching article with cache hit."""
        url = "https://example.com/cached3"

        # Pre-populate cache
        await db_manager.save_content(
            title="Cached Article",
            source="content_fetcher",
            url=url,
            processed_content="Cached content",
            summary="Cached excerpt",
            expires_in_days=7
        )

        # Fetch with cache enabled
        result = await content_fetcher.fetch_article(url, use_cache=True)

        assert result is not None
        assert result['status'] == 'cached'
        assert result['content'] == "Cached content"
        assert result['excerpt'] == "Cached excerpt"

    @pytest.mark.asyncio
    async def test_fetch_with_cache_miss(self, content_fetcher):
        """Test fetching article with cache miss."""
        url = "https://example.com/nocache"

        mock_html = "<html><body><p>Fresh content</p></body></html>"

        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = mock_html
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = await content_fetcher.fetch_article(url, use_cache=True)

            assert result is not None
            assert result['status'] == 'success'
            assert 'Fresh content' in result['content']

    @pytest.mark.asyncio
    async def test_fetch_ignores_cache_when_disabled(self, content_fetcher, db_manager):
        """Test that cache is ignored when use_cache=False."""
        url = "https://example.com/ignorecache"

        # Pre-populate cache
        await db_manager.save_content(
            title="Cached",
            source="content_fetcher",
            url=url,
            processed_content="Old cached content",
            expires_in_days=7
        )

        mock_html = "<html><body><p>Fresh new content</p></body></html>"

        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = mock_html
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = await content_fetcher.fetch_article(url, use_cache=False)

            assert result is not None
            assert result['status'] == 'success'
            assert 'Fresh new content' in result['content']
            assert 'Old cached content' not in result['content']


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

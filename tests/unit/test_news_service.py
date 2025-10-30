"""Unit tests for NewsService - TDD approach."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.models.news import NewsArticle
from src.models.base import DataSource


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager."""
    config_mgr = Mock()
    config_mgr.get_num_sources.return_value = 2
    config_mgr.get_per_source_limit.return_value = 10
    config_mgr.get_max_articles.return_value = 30
    config_mgr.get_content_threads.return_value = 5
    config_mgr.is_source_enabled.side_effect = lambda x: True
    config_mgr.get_rss_feeds.return_value = [
        {"name": "Test Feed", "url": "https://example.com/feed"}
    ]
    return config_mgr


@pytest.fixture
def sample_articles():
    """Sample news articles."""
    return [
        NewsArticle(
            title="Test Article 1",
            url="https://example.com/1",
            source=DataSource.HACKERNEWS,
            source_name="HackerNews",
            description=None
        ),
        NewsArticle(
            title="Test Article 2",
            url="https://example.com/2",
            source=DataSource.RSS,
            source_name="RSS Feed",
            description=None
        ),
    ]


@pytest.mark.asyncio
class TestNewsService:
    """Test NewsService functionality using TDD."""

    async def test_fetch_news_from_all_sources(self, mock_config_manager, sample_articles):
        """Test fetching news from all enabled sources."""
        from src.services.news_service import NewsService

        with patch('src.services.news_service.get_config_manager', return_value=mock_config_manager), \
             patch('src.services.news_service.HackerNewsConnector') as mock_hn, \
             patch('src.services.news_service.RSSConnector') as mock_rss:

            # Setup mocks
            mock_hn_instance = AsyncMock()
            mock_hn_instance.execute_async.return_value = [sample_articles[0]]
            mock_hn.return_value = mock_hn_instance

            mock_rss_instance = AsyncMock()
            mock_rss_instance.fetch_all_feeds.return_value = [sample_articles[1]]
            mock_rss.return_value = mock_rss_instance

            # Test
            service = NewsService()
            articles = await service.fetch_news_from_sources()

            # Verify
            assert len(articles) == 2
            assert articles[0].title == "Test Article 1"
            assert articles[1].title == "Test Article 2"
            mock_hn.assert_called_once_with(max_stories=10)
            mock_rss.assert_called_once_with(max_articles_per_feed=10)

    async def test_fetch_news_with_disabled_source(self, mock_config_manager):
        """Test that disabled sources are not fetched."""
        from src.services.news_service import NewsService

        mock_config_manager.is_source_enabled.side_effect = lambda x: x == "hackernews"

        with patch('src.services.news_service.get_config_manager', return_value=mock_config_manager), \
             patch('src.services.news_service.HackerNewsConnector') as mock_hn, \
             patch('src.services.news_service.RSSConnector') as mock_rss:

            mock_hn_instance = AsyncMock()
            mock_hn_instance.execute_async.return_value = []
            mock_hn.return_value = mock_hn_instance

            service = NewsService()
            await service.fetch_news_from_sources()

            # HackerNews should be called, RSS should not
            mock_hn.assert_called_once()
            mock_rss.assert_not_called()

    async def test_enrich_articles_with_analysis(self, sample_articles):
        """Test enriching articles with AI analysis."""
        from src.services.news_service import NewsService

        with patch('src.services.news_service.get_content_fetcher') as mock_fetcher, \
             patch('src.services.news_service.get_article_analyzer') as mock_analyzer_getter:

            # Mock content fetcher
            mock_fetcher_instance = AsyncMock()
            mock_fetcher_instance.fetch_article.return_value = {
                'status': 'success',
                'content': 'Test article content'
            }
            mock_fetcher.return_value = mock_fetcher_instance

            # Mock article analyzer
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_article.return_value = {
                'analysis': 'Test analysis of the article',
                'key_learnings': ['Learning 1', 'Learning 2', 'Learning 3'],
                'category': 'Technology',
                'impact': 'high'
            }
            mock_analyzer_getter.return_value = mock_analyzer

            service = NewsService()
            enriched = await service.enrich_with_analysis(sample_articles)

            # Verify all articles got analysis
            assert len(enriched) == 2
            for article in enriched:
                assert 'Test analysis' in article.description
                assert 'Key Learnings:' in article.description
                assert '• Learning 1' in article.description

    async def test_enrich_handles_analysis_errors(self, sample_articles):
        """Test that analysis enrichment handles errors gracefully."""
        from src.services.news_service import NewsService

        with patch('src.services.news_service.get_content_fetcher') as mock_fetcher, \
             patch('src.services.news_service.get_article_analyzer') as mock_analyzer_getter:

            # Mock content fetcher
            mock_fetcher_instance = AsyncMock()
            mock_fetcher_instance.fetch_article.return_value = {
                'status': 'success',
                'content': 'Test content'
            }
            mock_fetcher.return_value = mock_fetcher_instance

            # Mock article analyzer with error
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_article.return_value = {
                'error': 'AI service unavailable',
                'analysis': 'Analysis failed'
            }
            mock_analyzer_getter.return_value = mock_analyzer

            service = NewsService()
            enriched = await service.enrich_with_analysis(sample_articles)

            # Verify error messages are stored
            assert len(enriched) == 2
            assert all(article.description.startswith("⚠️ Analysis unavailable:") for article in enriched)

    async def test_respects_max_articles_limit(self, mock_config_manager):
        """Test that max_articles limit is respected."""
        from src.services.news_service import NewsService

        # Create 20 articles
        articles = [
            NewsArticle(
                title=f"Article {i}",
                url=f"https://example.com/{i}",
                source=DataSource.HACKERNEWS,
                source_name="HackerNews"
            )
            for i in range(20)
        ]

        with patch('src.services.news_service.get_config_manager', return_value=mock_config_manager), \
             patch('src.services.news_service.HackerNewsConnector') as mock_hn, \
             patch('src.services.news_service.RSSConnector') as mock_rss:

            mock_hn_instance = AsyncMock()
            mock_hn_instance.execute_async.return_value = articles[:15]
            mock_hn.return_value = mock_hn_instance

            mock_rss_instance = AsyncMock()
            mock_rss_instance.fetch_all_feeds.return_value = articles[15:]
            mock_rss.return_value = mock_rss_instance

            service = NewsService()
            result = await service.fetch_all_news(max_articles=10)

            # Should limit to 10 articles
            assert len(result) <= 10

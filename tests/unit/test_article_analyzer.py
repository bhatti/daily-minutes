"""Unit tests for ArticleAnalyzer - TDD approach."""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.models.news import NewsArticle
from src.models.base import DataSource


@pytest.fixture
def sample_article():
    """Sample news article for testing."""
    return NewsArticle(
        title="OpenAI Launches GPT-5 with Revolutionary Features",
        url="https://example.com/gpt5-launch",
        source=DataSource.HACKERNEWS,
        source_name="HackerNews",
        description="OpenAI has announced GPT-5, featuring advanced reasoning capabilities, multimodal understanding, and significantly improved performance across various tasks."
    )


@pytest.fixture
def sample_article_content():
    """Sample article content."""
    return """
    OpenAI has announced the launch of GPT-5, their most advanced language model yet.
    The new model features revolutionary improvements in reasoning, multimodal understanding,
    and task performance. Key features include:
    - 10x faster inference speed
    - Improved accuracy on complex reasoning tasks
    - Native support for images, audio, and video
    - Better alignment with human values
    GPT-5 represents a major leap forward in AI capabilities.
    """


@pytest_asyncio.fixture
async def test_db():
    """Provide a fresh in-memory SQLite database for each test."""
    from src.database.sqlite_manager import SQLiteManager

    db = SQLiteManager(db_path=":memory:")
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def mock_ollama_service():
    """Mock Ollama service."""
    service = AsyncMock()

    # Mock OllamaResponse object
    mock_response = AsyncMock()
    mock_response.content = """Analysis: This article discusses OpenAI's launch of GPT-5, highlighting its revolutionary features including advanced reasoning, multimodal capabilities, and improved performance metrics.

Key Learnings:
- GPT-5 offers 10x faster inference compared to previous versions
- The model now supports multimodal inputs including images, audio, and video
- Significant improvements in complex reasoning tasks
- Better alignment with human values and safety

Category: AI/Technology

Impact: High"""

    service.chat.return_value = mock_response
    return service


@pytest.mark.asyncio
class TestArticleAnalyzer:
    """Test ArticleAnalyzer functionality using TDD."""

    async def test_analyze_article_with_content(self, test_db, sample_article, sample_article_content, mock_ollama_service):
        """Test analyzing article with full content."""
        from src.services.article_analyzer import ArticleAnalyzer

        with patch('src.services.ollama_service.get_ollama_service', return_value=mock_ollama_service), \
             patch('src.services.article_analyzer.get_db_manager', return_value=test_db):
            analyzer = ArticleAnalyzer()
            result = await analyzer.analyze_article(
                title=sample_article.title,
                content=sample_article_content,
                url=str(sample_article.url)
            )

            # Verify result structure
            assert result is not None
            assert 'analysis' in result
            assert 'key_learnings' in result
            assert isinstance(result['key_learnings'], list)
            assert len(result['key_learnings']) > 0

            # Verify Ollama was called
            mock_ollama_service.chat.assert_called_once()

    async def test_analyze_article_with_title_only(self, test_db, sample_article, mock_ollama_service):
        """Test analyzing article with only title (no content)."""
        from src.services.article_analyzer import ArticleAnalyzer

        with patch('src.services.ollama_service.get_ollama_service', return_value=mock_ollama_service), \
             patch('src.services.article_analyzer.get_db_manager', return_value=test_db):
            analyzer = ArticleAnalyzer()
            result = await analyzer.analyze_article(
                title=sample_article.title,
                content=None,
                url=str(sample_article.url)
            )

            # Should still work with just title
            assert result is not None
            assert 'analysis' in result

    async def test_analyze_article_handles_ollama_error(self, test_db, sample_article, sample_article_content):
        """Test that analyzer handles Ollama errors gracefully."""
        from src.services.article_analyzer import ArticleAnalyzer

        mock_service = AsyncMock()
        mock_service.chat.side_effect = Exception("Ollama connection failed")

        with patch('src.services.ollama_service.get_ollama_service', return_value=mock_service), \
             patch('src.services.article_analyzer.get_db_manager', return_value=test_db):
            analyzer = ArticleAnalyzer()
            result = await analyzer.analyze_article(
                title=sample_article.title,
                content=sample_article_content,
                url=str(sample_article.url)
            )

            # Should return error result instead of raising
            assert result is not None
            assert 'error' in result or 'analysis' in result
            # If error, analysis should indicate the problem
            if 'error' in result:
                assert 'failed' in result['error'].lower() or 'error' in result['error'].lower()

    async def test_analyze_article_caches_results(self, test_db, sample_article, sample_article_content, mock_ollama_service):
        """Test that analyzer caches results for same URL."""
        from src.services.article_analyzer import ArticleAnalyzer

        with patch('src.services.ollama_service.get_ollama_service', return_value=mock_ollama_service), \
             patch('src.services.article_analyzer.get_db_manager', return_value=test_db):
            analyzer = ArticleAnalyzer()

            # First call
            result1 = await analyzer.analyze_article(
                title=sample_article.title,
                content=sample_article_content,
                url=str(sample_article.url),
                use_cache=True
            )

            # Second call with same URL
            result2 = await analyzer.analyze_article(
                title=sample_article.title,
                content=sample_article_content,
                url=str(sample_article.url),
                use_cache=True
            )

            # Should have cached result
            assert result1 == result2
            # Ollama should only be called once (first time)
            assert mock_ollama_service.chat.call_count == 1

    async def test_analyze_article_respects_cache_flag(self, test_db, sample_article, sample_article_content, mock_ollama_service):
        """Test that cache can be bypassed with use_cache=False."""
        from src.services.article_analyzer import ArticleAnalyzer

        with patch('src.services.ollama_service.get_ollama_service', return_value=mock_ollama_service), \
             patch('src.services.article_analyzer.get_db_manager', return_value=test_db):
            analyzer = ArticleAnalyzer()

            # First call
            await analyzer.analyze_article(
                title=sample_article.title,
                content=sample_article_content,
                url=str(sample_article.url),
                use_cache=False
            )

            # Second call with use_cache=False
            await analyzer.analyze_article(
                title=sample_article.title,
                content=sample_article_content,
                url=str(sample_article.url),
                use_cache=False
            )

            # Should call Ollama twice (no caching)
            assert mock_ollama_service.chat.call_count == 2

    async def test_analyze_batch_articles(self, test_db, mock_ollama_service):
        """Test analyzing multiple articles in batch."""
        from src.services.article_analyzer import ArticleAnalyzer

        articles_data = [
            {"title": "Article 1", "content": "Content 1", "url": "https://example.com/1"},
            {"title": "Article 2", "content": "Content 2", "url": "https://example.com/2"},
            {"title": "Article 3", "content": "Content 3", "url": "https://example.com/3"},
        ]

        with patch('src.services.ollama_service.get_ollama_service', return_value=mock_ollama_service), \
             patch('src.services.article_analyzer.get_db_manager', return_value=test_db):
            analyzer = ArticleAnalyzer()
            results = await analyzer.analyze_batch(articles_data, max_concurrent=2)

            # Should return results for all articles
            assert len(results) == 3
            assert all('analysis' in r or 'error' in r for r in results)

    async def test_generate_prompt_formats_correctly(self, sample_article, sample_article_content):
        """Test that analysis prompt is formatted correctly."""
        from src.services.article_analyzer import ArticleAnalyzer

        analyzer = ArticleAnalyzer()
        prompt = analyzer._generate_analysis_prompt(
            title=sample_article.title,
            content=sample_article_content
        )

        # Prompt should include title and content
        assert sample_article.title in prompt
        assert "analysis" in prompt.lower()
        assert "key learning" in prompt.lower() or "takeaway" in prompt.lower()

    async def test_parse_analysis_response(self):
        """Test parsing Ollama response into structured format."""
        from src.services.article_analyzer import ArticleAnalyzer

        analyzer = ArticleAnalyzer()

        # Sample Ollama response
        ollama_response = """
        Analysis: This article discusses the launch of GPT-5 with advanced features.

        Key Learnings:
        - 10x faster inference speed
        - Multimodal support for images, audio, video
        - Improved reasoning capabilities

        Category: AI/Technology
        Impact: High
        """

        result = analyzer._parse_analysis_response(ollama_response)

        # Should extract all components
        assert 'analysis' in result
        assert 'key_learnings' in result
        assert isinstance(result['key_learnings'], list)
        assert len(result['key_learnings']) >= 2

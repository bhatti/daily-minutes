"""Unit tests for news models."""

from datetime import datetime

import pytest

from src.models.news import NewsArticle, NewsSummary, RSSFeed, DataSource, Priority


class TestNewsArticle:
    """Tests for NewsArticle model."""

    def test_create_news_article(self):
        """Test creating a news article."""
        article = NewsArticle(
            title="Test Article",
            url="https://example.com/article",
            source=DataSource.HACKERNEWS,
            source_name="HackerNews",
            author="test_author",
            published_at=datetime.now(),
            description="Test description",
            tags=["test", "python"],
            priority=Priority.HIGH,
            relevance_score=0.8
        )

        assert article.title == "Test Article"
        assert article.source == DataSource.HACKERNEWS
        assert article.priority == Priority.HIGH
        assert article.relevance_score == 0.8
        assert len(article.tags) == 2

    def test_calculate_importance(self):
        """Test importance calculation."""
        # High priority, high relevance
        article = NewsArticle(
            title="Important News",
            url="https://example.com",
            source=DataSource.NEWS_API,
            source_name="Test",
            priority=Priority.HIGH,
            relevance_score=0.9
        )

        importance = article.calculate_importance()
        assert importance > 0.7  # Should be high

        # Low priority, low relevance
        article.priority = Priority.LOW
        article.relevance_score = 0.2
        importance = article.calculate_importance()
        assert importance < 0.5  # Should be low

    def test_mark_as_read(self):
        """Test marking article as read."""
        article = NewsArticle(
            title="Test",
            url="https://example.com",
            source=DataSource.RSS,
            source_name="Test"
        )

        assert article.is_read is False
        article.mark_as_read()
        assert article.is_read is True

    def test_toggle_star(self):
        """Test toggling starred status."""
        article = NewsArticle(
            title="Test",
            url="https://example.com",
            source=DataSource.RSS,
            source_name="Test"
        )

        assert article.is_starred is False
        article.toggle_star()
        assert article.is_starred is True
        article.toggle_star()
        assert article.is_starred is False

    def test_clean_tags(self):
        """Test tag cleaning and deduplication."""
        article = NewsArticle(
            title="Test",
            url="https://example.com",
            source=DataSource.RSS,
            source_name="Test",
            tags=["Python", "python", " TEST ", "test", "ai"]
        )

        # Should be deduplicated and lowercased
        assert len(article.tags) == 3  # python, test, ai
        assert "python" in article.tags
        assert "test" in article.tags
        assert "ai" in article.tags


class TestRSSFeed:
    """Tests for RSSFeed model."""

    def test_create_rss_feed(self):
        """Test creating RSS feed."""
        feed = RSSFeed(
            name="Test Feed",
            url="https://example.com/feed.xml",
            category="technology",
            max_articles=20,
            filter_keywords=["python", "ai"],
            exclude_keywords=["spam"]
        )

        assert feed.name == "Test Feed"
        assert feed.category == "technology"
        assert feed.max_articles == 20
        assert len(feed.filter_keywords) == 2
        assert len(feed.exclude_keywords) == 1

    def test_should_fetch(self):
        """Test fetch timing logic."""
        feed = RSSFeed(
            name="Test",
            url="https://example.com/feed.xml",
            update_frequency=3600  # 1 hour
        )

        # Should fetch on first run
        assert feed.should_fetch() is True

        # After recording fetch
        feed.record_fetch()
        assert feed.should_fetch() is False  # Too soon

        # Simulate time passing
        from datetime import timedelta
        feed.last_fetched = datetime.now() - timedelta(hours=2)
        assert feed.should_fetch() is True  # Enough time passed

    def test_record_error(self):
        """Test error recording."""
        feed = RSSFeed(
            name="Test",
            url="https://example.com/feed.xml"
        )

        assert feed.error_count == 0
        assert feed.last_error is None

        feed.record_error("Connection failed")
        assert feed.error_count == 1
        assert feed.last_error == "Connection failed"

        feed.record_error("Parse error")
        assert feed.error_count == 2
        assert feed.last_error == "Parse error"

    def test_matches_filters(self):
        """Test article filtering."""
        feed = RSSFeed(
            name="Test",
            url="https://example.com/feed.xml",
            filter_keywords=["python", "ai"],
            exclude_keywords=["spam", "advertisement"]
        )

        # Article with filter keywords - should match
        article1 = NewsArticle(
            title="Python AI Tutorial",
            url="https://example.com/1",
            source=DataSource.RSS,
            source_name="Test",
            description="Learn Python for AI"
        )
        assert feed.matches_filters(article1) is True

        # Article with exclude keywords - should not match
        article2 = NewsArticle(
            title="Python Spam",
            url="https://example.com/2",
            source=DataSource.RSS,
            source_name="Test",
            description="This is spam"
        )
        assert feed.matches_filters(article2) is False

        # Article without any keywords - should not match if filters exist
        article3 = NewsArticle(
            title="Unrelated Article",
            url="https://example.com/3",
            source=DataSource.RSS,
            source_name="Test",
            description="Something else"
        )
        assert feed.matches_filters(article3) is False


class TestNewsSummary:
    """Tests for NewsSummary model."""

    def test_create_news_summary(self):
        """Test creating news summary."""
        summary = NewsSummary(
            date=datetime.now(),
            source=DataSource.CUSTOM,
            total_articles=10,
            unread_articles=5,
            starred_articles=2
        )

        assert summary.total_articles == 10
        assert summary.unread_articles == 5
        assert summary.starred_articles == 2

    def test_add_article(self):
        """Test adding article to summary."""
        summary = NewsSummary(
            date=datetime.now(),
            source=DataSource.CUSTOM
        )

        article = NewsArticle(
            title="Test Article",
            url="https://example.com",
            source=DataSource.RSS,
            source_name="Test",
            tags=["python", "ai"],
            sentiment_score=0.5,
            is_starred=True
        )

        summary.add_article(article)

        assert summary.total_articles == 1
        assert summary.unread_articles == 1  # Not read by default
        assert summary.starred_articles == 1
        assert "python" in summary.categories
        assert "ai" in summary.categories
        assert summary.sentiment_distribution["positive"] == 1

    def test_get_top_articles(self):
        """Test getting top articles by importance."""
        summary = NewsSummary(
            date=datetime.now(),
            source=DataSource.CUSTOM
        )

        # Add articles with different importance
        for i in range(10):
            article = NewsArticle(
                title=f"Article {i}",
                url=f"https://example.com/{i}",
                source=DataSource.RSS,
                source_name="Test",
                relevance_score=i / 10.0  # 0.0 to 0.9
            )
            summary.top_articles.append(article)

        top_5 = summary.get_top_articles(5)
        assert len(top_5) == 5

        # Should be sorted by importance (highest first)
        scores = [a.calculate_importance() for a in top_5]
        assert scores == sorted(scores, reverse=True)

    def test_generate_brief(self):
        """Test generating summary brief."""
        summary = NewsSummary(
            date=datetime.now(),
            source=DataSource.CUSTOM,
            total_articles=20,
            unread_articles=10,
            starred_articles=3,
            key_topics=["ai", "python", "machine-learning"]
        )

        summary.sentiment_distribution = {
            "positive": 10,
            "neutral": 7,
            "negative": 3
        }

        brief = summary.generate_brief()

        assert "20" in brief  # Total articles
        assert "10 unread" in brief
        assert "3 starred" in brief
        assert "ai" in brief.lower()
        assert "Sentiment:" in brief


class TestValidation:
    """Tests for model validation."""

    def test_relevance_score_validation(self):
        """Test relevance score bounds."""
        article = NewsArticle(
            title="Test",
            url="https://example.com",
            source=DataSource.RSS,
            source_name="Test",
            relevance_score=0.5
        )
        assert article.relevance_score == 0.5

        # Test bounds
        article.relevance_score = 0.0
        assert article.relevance_score == 0.0

        article.relevance_score = 1.0
        assert article.relevance_score == 1.0

    def test_sentiment_score_validation(self):
        """Test sentiment score bounds."""
        article = NewsArticle(
            title="Test",
            url="https://example.com",
            source=DataSource.RSS,
            source_name="Test",
            sentiment_score=0.0
        )
        assert article.sentiment_score == 0.0

        article.sentiment_score = -1.0
        assert article.sentiment_score == -1.0

        article.sentiment_score = 1.0
        assert article.sentiment_score == 1.0
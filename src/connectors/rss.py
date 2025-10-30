"""RSS feed connector with async support."""

import asyncio
import os
from datetime import datetime
from typing import Any, ClassVar, List, Optional

import feedparser
import httpx
from pydantic import Field, HttpUrl

from src.core.logging import get_logger
from src.models.async_base import AsyncAggregator
from src.models.cache import CacheManager, CacheStrategy
from src.models.news import NewsArticle, RSSFeed, DataSource, Priority

logger = get_logger(__name__)

# SSL verification configuration for proxy environments
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() != 'false'


class RSSConnector(AsyncAggregator):
    """Connector for RSS feeds with parallel fetching."""

    # Predefined tech RSS feeds (from FREE_APIS_GUIDE.md)
    DEFAULT_TECH_FEEDS: ClassVar[List[str]] = [
        # Major Tech News
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.theverge.com/rss/index.xml",
        "https://www.wired.com/feed/rss",
        "https://techcrunch.com/feed/",
        "https://thenextweb.com/feed/",

        # Developer & Programming
        "https://stackoverflow.blog/feed/",
        "https://github.blog/feed/",

        # AI/ML Focused
        "https://openai.com/blog/rss/",
        "https://huggingface.co/blog/feed.xml",
        "https://ai.googleblog.com/feeds/posts/default",
    ]

    feeds: List[RSSFeed] = Field(default_factory=list, description="RSS feeds to monitor")
    max_articles_per_feed: int = Field(10, gt=0, description="Max articles per feed")

    # Cache
    cache_manager: Optional[CacheManager] = Field(None, description="Cache manager")
    cache_ttl_seconds: int = Field(900, gt=0, description="Cache TTL (15 minutes)")

    # Metrics
    feeds_processed: int = Field(0, ge=0, description="Total feeds processed")
    articles_fetched: int = Field(0, ge=0, description="Total articles fetched")

    def __init__(self, **data):
        """Initialize RSS connector with default feeds."""
        # Set up sources for AsyncAggregator
        if "sources" not in data:
            data["sources"] = []

        super().__init__(**data)

        # Initialize default feeds if none provided
        if not self.feeds:
            self._initialize_default_feeds()

        # Update sources from feeds (convert HttpUrl to string)
        self.sources = [str(feed.url) for feed in self.feeds]

        # Initialize cache
        if not self.cache_manager:
            self.cache_manager = CacheManager()
            self.cache_manager.create_store(
                "rss",
                strategy=CacheStrategy.TTL,
                max_size_mb=100,
                default_ttl_seconds=self.cache_ttl_seconds
            )

    def _initialize_default_feeds(self) -> None:
        """Initialize with default tech feeds."""
        for url in self.DEFAULT_TECH_FEEDS:
            name = url.split("/")[2]  # Extract domain as name
            feed = RSSFeed(
                name=name,
                url=url,
                category="technology",
                max_articles=self.max_articles_per_feed
            )
            self.feeds.append(feed)

    def add_feed(self, url: str, name: Optional[str] = None, category: str = "general") -> None:
        """Add a new RSS feed."""
        feed = RSSFeed(
            name=name or url.split("/")[2],
            url=url,
            category=category,
            max_articles=self.max_articles_per_feed
        )
        self.feeds.append(feed)
        self.sources.append(str(url))
        logger.info("added_feed", url=url, name=name)

    async def fetch_from_source(self, source: str) -> List[NewsArticle]:
        """Fetch articles from a single RSS feed."""
        # Find corresponding feed
        feed = next((f for f in self.feeds if f.url == source), None)
        if not feed:
            return []

        # Check if should fetch
        if not feed.should_fetch():
            logger.debug("skip_fetch", feed=feed.name, reason="too_soon")
            return []

        # Check cache
        cache_key = f"rss_feed_{feed.name}"
        cached = self.cache_manager.get(cache_key, store="rss")
        if cached:
            logger.info("cache_hit", feed=feed.name)
            return cached

        try:
            # Fetch feed (convert HttpUrl to string)
            async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
                response = await client.get(str(feed.url), timeout=15.0)
                response.raise_for_status()
                content = response.text

            # Parse feed
            parsed = feedparser.parse(content)

            if parsed.bozo:
                feed.record_error(f"Parse error: {parsed.bozo_exception}")
                logger.error("feed_parse_error", feed=feed.name, error=str(parsed.bozo_exception))
                return []

            # Convert entries to articles
            articles = []
            for entry in parsed.entries[:feed.max_articles]:
                article = self._entry_to_article(entry, feed)

                # Apply feed filters
                if feed.matches_filters(article):
                    articles.append(article)

            # Update feed status
            feed.record_fetch()
            self.feeds_processed += 1
            self.articles_fetched += len(articles)

            # Cache results
            self.cache_manager.put(
                cache_key,
                articles,
                ttl_seconds=self.cache_ttl_seconds,
                store="rss"
            )

            logger.info("fetched_feed", feed=feed.name, articles=len(articles))
            return articles

        except Exception as e:
            feed.record_error(str(e))
            logger.error("fetch_feed_failed", feed=feed.name, error=str(e))
            return []

    def _entry_to_article(self, entry: Any, feed: RSSFeed) -> NewsArticle:
        """Convert RSS entry to NewsArticle."""
        # Parse published date
        published_at = None
        if hasattr(entry, 'published_parsed'):
            try:
                published_at = datetime(*entry.published_parsed[:6])
            except:
                published_at = datetime.now()
        else:
            published_at = datetime.now()

        # Extract description
        description = ""
        if hasattr(entry, 'summary'):
            description = entry.summary
        elif hasattr(entry, 'description'):
            description = entry.description

        # Clean HTML from description
        import re
        description = re.sub('<[^<]+?>', '', description)[:500]

        # Extract tags from categories
        tags = []
        if hasattr(entry, 'tags'):
            tags = [tag.term.lower() for tag in entry.tags]

        # Add feed category
        tags.append(feed.category)

        # Calculate relevance based on keywords
        relevance_score = self._calculate_relevance(entry.title, description, tags)

        article = NewsArticle(
            title=entry.title,
            url=entry.link,
            source=DataSource.RSS,
            source_name=feed.name,
            author=getattr(entry, 'author', None),
            published_at=published_at,
            description=description,
            tags=tags,
            priority=Priority.MEDIUM,
            relevance_score=relevance_score,
            metadata={
                "feed_name": feed.name,
                "feed_category": feed.category,
                "guid": getattr(entry, 'id', entry.link)
            }
        )

        return article

    def _calculate_relevance(self, title: str, description: str, tags: List[str]) -> float:
        """Calculate article relevance score."""
        score = 0.5  # Base score

        # High-value keywords
        high_value_keywords = [
            "ai", "llm", "gpt", "machine learning", "breakthrough",
            "launch", "release", "announces", "funding", "acquisition"
        ]

        text = f"{title} {description}".lower()

        for keyword in high_value_keywords:
            if keyword in text:
                score += 0.1

        # Boost for AI/ML tags
        ai_tags = ["ai", "ml", "artificial-intelligence", "machine-learning"]
        if any(tag in tags for tag in ai_tags):
            score += 0.2

        return min(1.0, score)

    async def fetch_all_feeds(self) -> List[NewsArticle]:
        """Fetch articles from all feeds in parallel."""
        # Use AsyncAggregator's parallel fetching
        aggregated = await self.fetch_async()

        # Flatten results
        all_articles = []
        for source_articles in aggregated.values():
            if isinstance(source_articles, list):
                all_articles.extend(source_articles)

        # Sort by published date
        all_articles.sort(key=lambda a: a.published_at, reverse=True)

        logger.info("fetched_all_feeds",
                   feeds=len(self.feeds),
                   articles=len(all_articles))

        return all_articles

    async def process_async(self, data: dict) -> List[NewsArticle]:
        """Process aggregated feed data."""
        # Data is already articles from fetch_from_source
        return await self.fetch_all_feeds()

    def get_feed_status(self) -> dict:
        """Get status of all feeds."""
        return {
            "total_feeds": len(self.feeds),
            "active_feeds": sum(1 for f in self.feeds if f.is_active),
            "feeds_with_errors": sum(1 for f in self.feeds if f.error_count > 0),
            "feeds_processed": self.feeds_processed,
            "articles_fetched": self.articles_fetched,
            "feed_details": [
                {
                    "name": f.name,
                    "category": f.category,
                    "is_active": f.is_active,
                    "last_fetched": f.last_fetched.isoformat() if f.last_fetched else None,
                    "error_count": f.error_count,
                    "last_error": f.last_error
                }
                for f in self.feeds
            ]
        }
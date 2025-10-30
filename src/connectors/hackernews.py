"""HackerNews API connector with async support and caching."""

import asyncio
import os
import ssl
from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional

import httpx
from pydantic import Field

from src.core.logging import get_logger
from src.models.async_base import AsyncBatchProcessor
from src.models.cache import CacheManager, CacheStrategy
from src.models.news import NewsArticle, DataSource, Priority

logger = get_logger(__name__)

# SSL verification configuration for proxy environments
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() != 'false'


class HackerNewsConnector(AsyncBatchProcessor):
    """Connector for HackerNews API with intelligent fetching."""

    # API endpoints
    BASE_URL: ClassVar[str] = "https://hacker-news.firebaseio.com/v0"

    ENDPOINTS: ClassVar[Dict[str, str]] = {
        "top": "https://hacker-news.firebaseio.com/v0/topstories.json",
        "best": "https://hacker-news.firebaseio.com/v0/beststories.json",
        "new": "https://hacker-news.firebaseio.com/v0/newstories.json",
        "ask": "https://hacker-news.firebaseio.com/v0/askstories.json",
        "show": "https://hacker-news.firebaseio.com/v0/showstories.json",
        "job": "https://hacker-news.firebaseio.com/v0/jobstories.json",
        "item": "https://hacker-news.firebaseio.com/v0/item/{item_id}.json",
        "user": "https://hacker-news.firebaseio.com/v0/user/{user_id}.json",
    }

    # Configuration
    story_type: str = Field("top", description="Type of stories to fetch")
    max_stories: int = Field(30, gt=0, le=500, description="Maximum stories to fetch")
    include_comments: bool = Field(False, description="Include comment analysis")
    min_score: int = Field(10, ge=0, description="Minimum story score")

    # Rate limiting (be respectful)
    requests_per_minute: int = Field(30, gt=0, description="Max requests per minute")
    last_request_time: Optional[datetime] = Field(None, description="Last request timestamp")

    # Cache
    cache_manager: Optional[CacheManager] = Field(None, description="Cache manager")
    cache_ttl_seconds: int = Field(600, gt=0, description="Cache TTL (10 minutes default)")

    # Metrics
    stories_fetched: int = Field(0, ge=0, description="Total stories fetched")
    api_calls_made: int = Field(0, ge=0, description="Total API calls made")

    def __init__(self, **data):
        """Initialize HackerNews connector with cache."""
        super().__init__(**data)

        # Initialize cache
        if not self.cache_manager:
            self.cache_manager = CacheManager()
            self.cache_manager.create_store(
                "hackernews",
                strategy=CacheStrategy.LRU,
                max_size_mb=50,
                default_ttl_seconds=self.cache_ttl_seconds
            )

    async def _rate_limit(self) -> None:
        """Implement rate limiting."""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            min_interval = 60.0 / self.requests_per_minute

            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

        self.last_request_time = datetime.now()

    async def fetch_story_ids(self) -> List[int]:
        """Fetch story IDs based on story type."""
        cache_key = f"hn_stories_{self.story_type}"

        # Check cache
        cached = self.cache_manager.get(cache_key, store="hackernews")
        if cached:
            logger.info("cache_hit", key=cache_key, count=len(cached))
            return cached

        # Fetch from API
        await self._rate_limit()

        async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
            try:
                response = await client.get(
                    self.ENDPOINTS[self.story_type],
                    timeout=10.0
                )
                response.raise_for_status()
                story_ids = response.json()[:self.max_stories]

                # Cache result
                self.cache_manager.put(
                    cache_key,
                    story_ids,
                    ttl_seconds=self.cache_ttl_seconds,
                    store="hackernews"
                )

                self.api_calls_made += 1
                logger.info("fetched_story_ids", count=len(story_ids))
                return story_ids

            except Exception as e:
                logger.error("fetch_story_ids_failed", error=str(e))
                return []

    async def fetch_story(self, story_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single story by ID."""
        cache_key = f"hn_story_{story_id}"

        # Check cache
        cached = self.cache_manager.get(cache_key, store="hackernews")
        if cached:
            return cached

        # Fetch from API
        await self._rate_limit()

        async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
            try:
                response = await client.get(
                    self.ENDPOINTS["item"].format(item_id=story_id),
                    timeout=10.0
                )
                response.raise_for_status()
                story = response.json()

                # Cache result
                self.cache_manager.put(
                    cache_key,
                    story,
                    ttl_seconds=self.cache_ttl_seconds * 2,  # Longer TTL for individual stories
                    store="hackernews"
                )

                self.api_calls_made += 1
                return story

            except Exception as e:
                logger.error("fetch_story_failed", story_id=story_id, error=str(e))
                return None

    async def fetch_async(self) -> List[int]:
        """Fetch story IDs - implements AsyncBatchProcessor."""
        return await self.fetch_story_ids()

    async def process_item_async(self, item: int) -> Optional[NewsArticle]:
        """Process a single story ID into NewsArticle."""
        story = await self.fetch_story(item)

        if not story or story.get("type") != "story":
            return None

        # Filter by score
        score = story.get("score", 0)
        if score < self.min_score:
            return None

        # Convert to NewsArticle
        return self.story_to_article(story)

    async def process_async(self, story_ids: List[int]) -> List[NewsArticle]:
        """Process story IDs into NewsArticles - implements AsyncBatchProcessor."""
        # Process stories in parallel batches
        articles = await self.process_batch(story_ids)

        # Filter out None values
        valid_articles = [a for a in articles if a is not None]

        self.stories_fetched += len(valid_articles)
        logger.info("processed_stories", count=len(valid_articles))

        return valid_articles

    def story_to_article(self, story: Dict[str, Any]) -> NewsArticle:
        """Convert HackerNews story to NewsArticle model."""
        # Calculate priority based on score and comments
        score = story.get("score", 0)
        descendants = story.get("descendants", 0)

        if score > 500 or descendants > 100:
            priority = Priority.HIGH
        elif score > 100 or descendants > 50:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW

        # Calculate relevance score (0-1)
        relevance_score = min(1.0, (score / 500) * 0.7 + (descendants / 200) * 0.3)

        # Create article
        article = NewsArticle(
            title=story.get("title", ""),
            url=story.get("url", f"https://news.ycombinator.com/item?id={story['id']}"),
            source=DataSource.HACKERNEWS,
            source_name="HackerNews",
            author=story.get("by", "unknown"),
            published_at=datetime.fromtimestamp(story.get("time", 0)),
            description=story.get("text", ""),  # For Ask HN, Show HN
            tags=self._extract_tags(story),
            priority=priority,
            relevance_score=relevance_score,
            metadata={
                "hn_id": story.get("id"),
                "score": score,
                "comments": descendants,
                "type": story.get("type"),
            }
        )

        return article

    def _extract_tags(self, story: Dict[str, Any]) -> List[str]:
        """Extract tags from story title and content."""
        tags = []
        title = story.get("title", "").lower()

        # Common tech keywords
        tech_keywords = [
            "ai", "ml", "machine learning", "llm", "gpt", "python", "javascript",
            "rust", "go", "kubernetes", "docker", "aws", "cloud", "startup",
            "security", "privacy", "open source", "database", "api", "framework"
        ]

        for keyword in tech_keywords:
            if keyword in title:
                tags.append(keyword)

        # Special HN tags
        if title.startswith("ask hn:"):
            tags.append("ask-hn")
        elif title.startswith("show hn:"):
            tags.append("show-hn")
        elif title.startswith("launch hn:"):
            tags.append("launch-hn")

        # Job posts
        if story.get("type") == "job":
            tags.append("job")

        return tags

    async def get_trending_topics(self) -> List[str]:
        """Analyze trending topics from recent stories."""
        # Get top stories
        story_ids = await self.fetch_story_ids()
        articles = await self.process_async(story_ids[:50])

        # Count tag frequencies
        tag_counts = {}
        for article in articles:
            for tag in article.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Sort by frequency
        trending = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

        return [tag for tag, _ in trending[:10]]

    async def get_user_submissions(self, username: str, limit: int = 10) -> List[NewsArticle]:
        """Get recent submissions from a specific user."""
        cache_key = f"hn_user_{username}"

        # Check cache
        cached = self.cache_manager.get(cache_key, store="hackernews")
        if cached:
            return cached

        # Fetch user data
        await self._rate_limit()

        async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
            try:
                response = await client.get(
                    self.ENDPOINTS["user"].format(user_id=username),
                    timeout=10.0
                )
                response.raise_for_status()
                user_data = response.json()

                # Get user's submissions
                submissions = user_data.get("submitted", [])[:limit]

                # Fetch and process stories
                articles = []
                for sub_id in submissions:
                    article = await self.process_item_async(sub_id)
                    if article:
                        articles.append(article)

                # Cache result
                self.cache_manager.put(
                    cache_key,
                    articles,
                    ttl_seconds=self.cache_ttl_seconds,
                    store="hackernews"
                )

                return articles

            except Exception as e:
                logger.error("fetch_user_failed", username=username, error=str(e))
                return []

    async def search_stories(self, query: str, limit: int = 20) -> List[NewsArticle]:
        """Search stories using Algolia HN Search API."""
        search_url = "https://hn.algolia.com/api/v1/search"

        async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
            try:
                response = await client.get(
                    search_url,
                    params={
                        "query": query,
                        "tags": "story",
                        "hitsPerPage": limit
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()

                articles = []
                for hit in data.get("hits", []):
                    # Convert Algolia hit to HN story format
                    story = {
                        "id": hit.get("objectID"),
                        "title": hit.get("title"),
                        "url": hit.get("url"),
                        "by": hit.get("author"),
                        "time": hit.get("created_at_i"),
                        "score": hit.get("points", 0),
                        "descendants": hit.get("num_comments", 0),
                        "type": "story"
                    }

                    article = self.story_to_article(story)
                    articles.append(article)

                return articles

            except Exception as e:
                logger.error("search_failed", query=query, error=str(e))
                return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        cache_stats = self.cache_manager.get_global_statistics() if self.cache_manager else {}

        return {
            "stories_fetched": self.stories_fetched,
            "api_calls_made": self.api_calls_made,
            "story_type": self.story_type,
            "cache_stats": cache_stats,
            "performance": self.get_performance_stats()
        }
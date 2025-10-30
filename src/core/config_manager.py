"""Configuration manager with database persistence."""

import asyncio
from typing import Any, Optional, Dict
from src.core.logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """
    Manage application configuration with database persistence.

    All config values are stored in the database (kv_store table with category='config').
    Provides sync and async access methods with in-memory caching.
    """

    def __init__(self):
        """Initialize config manager."""
        self._cache: Dict[str, Any] = {}
        self._initialized = False
        self._db_manager = None

    async def _ensure_initialized(self):
        """Ensure database is initialized."""
        if not self._initialized:
            from src.database.sqlite_manager import get_db_manager
            self._db_manager = get_db_manager()
            await self._db_manager.initialize()
            self._initialized = True

    async def get_async(self, key: str, default: Any = None) -> Any:
        """
        Get config value asynchronously.

        Args:
            key: Config key (e.g., 'http.verify_ssl')
            default: Default value if not found

        Returns:
            Config value or default
        """
        await self._ensure_initialized()

        # Check cache first
        if key in self._cache:
            return self._cache[key]

        # Fetch from database
        value = await self._db_manager.get_setting(key, default)

        # Cache the value
        self._cache[key] = value

        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value synchronously.

        Args:
            key: Config key (e.g., 'http.verify_ssl')
            default: Default value if not found

        Returns:
            Config value or default
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        # If not cached, we need to fetch from DB
        # Try to use existing event loop or create new one
        try:
            loop = asyncio.get_running_loop()
            # If loop is running, we can't use run_until_complete
            # Return default and log warning
            logger.warning("config_sync_get_in_async_context",
                         key=key,
                         message="Returning default value - use get_async() in async context")
            return default
        except RuntimeError:
            # No running loop, safe to create one
            return asyncio.run(self.get_async(key, default))

    async def set_async(self, key: str, value: Any):
        """
        Set config value asynchronously.

        Args:
            key: Config key
            value: Config value
        """
        await self._ensure_initialized()

        # Update database
        await self._db_manager.set_setting(key, value)

        # Update cache
        self._cache[key] = value

        logger.info("config_set", key=key, value=value)

    def set(self, key: str, value: Any):
        """
        Set config value synchronously.

        Args:
            key: Config key
            value: Config value
        """
        try:
            loop = asyncio.get_running_loop()
            # If loop is running, we can't use run_until_complete
            logger.error("config_sync_set_in_async_context",
                        key=key,
                        message="Cannot set config in async context - use set_async()")
            raise RuntimeError("Cannot use sync set() in async context - use set_async()")
        except RuntimeError:
            # No running loop, safe to create one
            asyncio.run(self.set_async(key, value))

    async def get_all_async(self) -> Dict[str, Any]:
        """
        Get all config values asynchronously.

        Returns:
            Dictionary of all config key-value pairs
        """
        await self._ensure_initialized()

        # For now, return cached values
        # TODO: Could fetch all from DB if needed
        return self._cache.copy()

    def invalidate_cache(self, key: Optional[str] = None):
        """
        Invalidate cache for a specific key or all keys.

        Args:
            key: Optional key to invalidate (None = invalidate all)
        """
        if key:
            self._cache.pop(key, None)
            logger.debug("config_cache_invalidated", key=key)
        else:
            self._cache.clear()
            logger.debug("config_cache_cleared")

    def get_per_source_limit(self, num_sources: int) -> int:
        """
        Calculate per-source article limit.

        Formula: min(max_articles / num_sources, max_per_source_base)

        Args:
            num_sources: Number of news sources being fetched from

        Returns:
            Per-source article limit
        """
        max_articles = self.get("news.max_articles", 30)
        max_per_source_base = self.get("news.max_per_source_base", 10)

        per_source = max_articles // num_sources if num_sources > 0 else max_per_source_base
        limit = min(per_source, max_per_source_base)

        logger.debug("calculated_per_source_limit",
                    num_sources=num_sources,
                    max_articles=max_articles,
                    per_source_base=max_per_source_base,
                    calculated_limit=limit)

        return limit

    def get_content_threads(self) -> int:
        """
        Get number of parallel threads for content fetching.

        Returns:
            Thread pool size for content fetching
        """
        return self.get("news.content_threads", 5)

    def get_max_articles(self) -> int:
        """
        Get maximum total articles to fetch.

        Returns:
            Maximum article count
        """
        return self.get("news.max_articles", 30)

    def get_news_sources(self) -> list:
        """
        Get list of enabled news sources with their configurations.

        Returns:
            List of source configurations:
            [
                {"type": "hackernews", "enabled": true},
                {"type": "rss", "enabled": true, "feeds": [
                    {"name": "TechCrunch", "url": "https://..."},
                    ...
                ]}
            ]
        """
        default_sources = [
            {"type": "hackernews", "enabled": True},
            {
                "type": "rss",
                "enabled": True,
                "feeds": [
                    # Tech news (free)
                    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "tech"},
                    {"name": "Ars Technica", "url": "https://arstechnica.com/feed/", "category": "tech"},
                    {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml", "category": "tech"},

                    # Market/Financial news (FREE sources - for TLDR bullet #3)
                    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rss", "category": "market"},
                    # {"name": "CNBC", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "category": "market"},  # Disabled - can enable if needed
                ]
            }
        ]
        return self.get("news.sources", default_sources)

    def set_news_sources(self, sources: list):
        """
        Set list of enabled news sources.

        Args:
            sources: List of source configurations
        """
        self.set("news.sources", sources)
        logger.info("news_sources_updated", sources=sources)

    def get_num_sources(self) -> int:
        """
        Get number of enabled news sources.

        Returns:
            Count of enabled sources
        """
        sources = self.get_news_sources()
        return len([s for s in sources if s.get("enabled", True)])

    def get_rss_feeds(self) -> list:
        """
        Get list of RSS feed URLs from enabled RSS sources.

        Returns:
            List of RSS feed configs [{"name": "...", "url": "..."}, ...]
        """
        sources = self.get_news_sources()
        for source in sources:
            if source.get("type") == "rss" and source.get("enabled", True):
                return source.get("feeds", [])
        return []

    def is_source_enabled(self, source_type: str) -> bool:
        """
        Check if a specific source type is enabled.

        Args:
            source_type: Source type (e.g., "hackernews", "rss")

        Returns:
            True if source is enabled
        """
        sources = self.get_news_sources()
        for source in sources:
            if source.get("type") == source_type:
                return source.get("enabled", True)
        return False

    # =========================================================================
    # Cache Configuration
    # =========================================================================

    def get_auto_refresh_hours(self) -> int:
        """
        Get auto-refresh interval in hours.

        Returns:
            Auto-refresh interval (default: 4 hours, range: 1-12)
        """
        return self.get("cache.auto_refresh_hours", 4)

    def get_news_list_cache_hours(self) -> int:
        """
        Get cache duration for news lists (top stories, RSS feeds).
        Should match auto_refresh_hours since news lists change.

        Returns:
            Cache duration in hours
        """
        # News list cache should match auto-refresh to avoid stale data
        return self.get_auto_refresh_hours()

    def get_article_content_cache_days(self) -> int:
        """
        Get cache duration for article content.
        Article content doesn't change, so cache for long time.

        Returns:
            Cache duration in days (default: 100 days)
        """
        return self.get("cache.article_content_days", 100)

    def get_article_analysis_cache_days(self) -> int:
        """
        Get cache duration for article AI analysis.
        Analysis doesn't change once generated, so cache for long time.

        Returns:
            Cache duration in days (default: 100 days)
        """
        return self.get("cache.article_analysis_days", 100)

    def get_weather_cache_hours(self) -> int:
        """
        Get cache duration for weather data.
        Should match auto_refresh_hours since weather changes.

        Returns:
            Cache duration in hours
        """
        # Weather cache should match auto-refresh
        return self.get_auto_refresh_hours()

    def set_auto_refresh_hours(self, hours: int):
        """
        Set auto-refresh interval in hours.

        Args:
            hours: Auto-refresh interval (1-12 hours)
        """
        if not (1 <= hours <= 12):
            raise ValueError("Auto-refresh hours must be between 1 and 12")
        self.set("cache.auto_refresh_hours", hours)
        logger.info("auto_refresh_hours_updated", hours=hours)

    def mark_ai_summary_stale(self):
        """
        Mark AI summary as stale (needs regeneration).

        Called when underlying data changes (new articles, new analysis).
        """
        self.set("ai.summary_stale", True)
        logger.info("ai_summary_marked_stale")

    async def mark_ai_summary_stale_async(self):
        """
        Mark AI summary as stale (needs regeneration) - async version.

        Called when underlying data changes (new articles, new analysis).
        Use this version when calling from async context.
        """
        await self.set_async("ai.summary_stale", True)
        logger.info("ai_summary_marked_stale")

    def is_ai_summary_stale(self) -> bool:
        """
        Check if AI summary needs regeneration.

        Returns:
            True if summary is stale or missing
        """
        return self.get("ai.summary_stale", True)

    def mark_ai_summary_fresh(self):
        """
        Mark AI summary as fresh (just regenerated).

        Called after AI summary is generated.
        """
        self.set("ai.summary_stale", False)
        logger.info("ai_summary_marked_fresh")

    async def load_defaults_async(self):
        """
        Load default configuration values if not already set.

        This is called on first initialization to set sensible defaults.
        """
        await self._ensure_initialized()

        defaults = {
            # HTTP settings
            "http.verify_ssl": True,
            "http.timeout": 30,
            "http.user_agent": "Daily-Minutes/1.0",

            # News settings
            "news.max_articles": 30,
            "news.max_per_source_base": 10,  # Base limit per source, actual = min(max_articles/num_sources, base)
            "news.content_threads": 5,
            "news.sources": [
                {"type": "hackernews", "enabled": True},
                {
                    "type": "rss",
                    "enabled": True,
                    "feeds": [
                        # Tech news (free)
                        {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "tech"},
                        {"name": "Ars Technica", "url": "https://arstechnica.com/feed/", "category": "tech"},
                        {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml", "category": "tech"},

                        # Market/Financial news (FREE sources - for TLDR bullet #3)
                        {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rss", "category": "market"},
                        # {"name": "CNBC", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "category": "market"},  # Disabled - can enable if needed
                    ]
                }
            ],

            # Weather settings
            "weather.default_location": "Seattle",
            "weather.temp_unit": "Fahrenheit",

            # Cache settings (smart caching strategy)
            "cache.auto_refresh_hours": 4,  # Auto-refresh interval: 1-12 hours
            "cache.article_content_days": 100,  # Article content cache (permanent data)
            "cache.article_analysis_days": 100,  # AI analysis cache (permanent data)
            # Note: News list & weather cache automatically match auto_refresh_hours

            # AI settings
            "ai.summary_stale": True,  # Whether AI summary needs regeneration

            # UI settings
            "ui.theme": "light",
            "ui.auto_refresh": False,
            "ui.refresh_interval_minutes": 15,
        }

        # Only set if not already in database
        for key, value in defaults.items():
            existing = await self.get_async(key, None)
            if existing is None:
                await self.set_async(key, value)

        logger.info("config_defaults_loaded", count=len(defaults))

    def is_dev_mode(self) -> bool:
        """
        Check if application is running in development/testing mode.

        Reads from DEV_MODE environment variable.
        When enabled, shows testing tools and mock data features in UI.

        Returns:
            True if DEV_MODE is enabled, False otherwise
        """
        import os
        return os.getenv("DEV_MODE", "false").lower() == "true"


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get or create config manager instance.

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


async def initialize_config():
    """
    Initialize configuration system and load defaults.

    Should be called once at application startup.
    """
    config_mgr = get_config_manager()
    await config_mgr.load_defaults_async()
    logger.info("config_system_initialized")

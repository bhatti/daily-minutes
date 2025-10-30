"""Startup service - loads cached data on application startup.

This service is separated from the UI to enable testing and reusability.
It loads articles and other cached data from the database when the app starts.
"""

from typing import Dict, Any, Optional, List
from src.core.logging import get_logger
from src.database.sqlite_manager import SQLiteManager
from src.models.news import NewsArticle

logger = get_logger(__name__)


class StartupService:
    """Service for loading data on application startup."""

    def __init__(self, db_manager: Optional[SQLiteManager] = None):
        """Initialize startup service.

        Args:
            db_manager: Optional database manager. If None, uses singleton.
        """
        self.db_manager = db_manager

    async def _ensure_db(self):
        """Ensure database manager is initialized."""
        if self.db_manager is None:
            from src.database.sqlite_manager import get_db_manager
            self.db_manager = get_db_manager()
            await self.db_manager.initialize()

    async def load_startup_data(self, limit: int = 100) -> Dict[str, Any]:
        """Load cached data from database on startup.

        This method loads ALL data types:
        - News articles (up to limit)
        - Weather data
        - Email data
        - Calendar events
        - Last refresh timestamps for each category

        Args:
            limit: Maximum number of articles to load (default: 100)

        Returns:
            Dictionary with:
                - articles: List of NewsArticle objects
                - weather_data: Weather data dict (None if not cached)
                - emails: List of email dicts
                - calendar_events: List of calendar event dicts
                - last_refresh: Dict mapping category -> timestamp string
                - cache_age_hours: Age of news cache in hours (None if no cache)
                - loaded_from_cache: Boolean indicating if news was loaded
                - error: Error message if loading failed (None on success)
        """
        result = {
            'articles': [],
            'weather_data': None,
            'emails': [],
            'calendar_events': [],
            'daily_brief': None,
            'last_refresh': {},
            'cache_age_hours': None,
            'loaded_from_cache': False,
            'error': None
        }

        try:
            await self._ensure_db()

            # Load news articles
            logger.info("startup_loading_articles", limit=limit)
            articles = await self.db_manager.get_all_articles(limit=limit)
            if articles:
                result['articles'] = articles
                result['loaded_from_cache'] = True
                logger.info("startup_articles_loaded", count=len(articles))

            # Get news cache age
            cache_age = await self.db_manager.get_cache_age_hours()
            result['cache_age_hours'] = cache_age

            # Load individual refresh timestamps from settings
            # Background refresh service saves them as separate keys
            timestamps = {}
            last_news = await self.db_manager.get_setting('last_news_refresh')
            if last_news:
                timestamps['news'] = last_news

            last_weather = await self.db_manager.get_setting('last_weather_refresh')
            if last_weather:
                timestamps['weather'] = last_weather

            last_email = await self.db_manager.get_setting('last_email_refresh')
            if last_email:
                timestamps['email'] = last_email

            last_calendar = await self.db_manager.get_setting('last_calendar_refresh')
            if last_calendar:
                timestamps['calendar'] = last_calendar

            result['last_refresh'] = timestamps

            # Load weather data from cache
            weather_cache = await self.db_manager.get_cache('weather_data')
            if weather_cache:
                result['weather_data'] = weather_cache
                logger.info("startup_weather_loaded")

            # Load emails from cache (if any)
            email_cache = await self.db_manager.get_cache('emails_data')
            if email_cache:
                result['emails'] = email_cache
                logger.info("startup_emails_loaded", count=len(email_cache))

            # Load calendar events from cache (if any)
            calendar_cache = await self.db_manager.get_cache('calendar_data')
            if calendar_cache:
                result['calendar_events'] = calendar_cache
                logger.info("startup_calendar_loaded", count=len(calendar_cache))

            # Load daily brief from cache (if any)
            brief_cache = await self.db_manager.get_cache('daily_brief_data')
            if brief_cache:
                result['daily_brief'] = brief_cache
                logger.info("startup_brief_loaded",
                           generated_at=brief_cache.get('generated_at'))

            logger.info("startup_load_complete",
                       articles=len(result['articles']),
                       weather=bool(result['weather_data']),
                       emails=len(result['emails']),
                       calendar=len(result['calendar_events']),
                       brief=bool(result['daily_brief']))

        except Exception as e:
            error_msg = str(e)
            logger.error("startup_load_failed", error=error_msg)
            result['error'] = error_msg

        return result

    async def get_cache_summary(self) -> Dict[str, Any]:
        """Get summary of cached data.

        Returns:
            Dictionary with cache statistics:
                - article_count: Number of cached articles
                - cache_age_hours: Age of cache in hours
                - has_fresh_data: Whether cache is fresh (< 4 hours old)
        """
        await self._ensure_db()

        article_count = len(await self.db_manager.get_all_articles(limit=1000))
        cache_age = await self.db_manager.get_cache_age_hours()

        return {
            'article_count': article_count,
            'cache_age_hours': cache_age,
            'has_fresh_data': cache_age is not None and cache_age < 4
        }


# Singleton instance
_startup_service: Optional[StartupService] = None


def get_startup_service(db_manager: Optional[SQLiteManager] = None) -> StartupService:
    """Get or create StartupService singleton.

    Args:
        db_manager: Optional database manager to use

    Returns:
        StartupService instance
    """
    global _startup_service
    if _startup_service is None:
        _startup_service = StartupService(db_manager)
    return _startup_service

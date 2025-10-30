"""MCP Scheduler - Periodic background refresh of all MCP data sources.

This service runs in the background and automatically refreshes data from:
- News (configurable interval, default 60 min)
- Weather (configurable interval, default 60 min)
- Email (configurable interval, default 60 min)
- Calendar (configurable interval, default 60 min)
- Slack (configurable interval, default 60 min)

All data is stored in the database which serves as both cache and source of truth.
"""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional
from src.core.logging import get_logger
from src.core.scheduler_config import get_scheduler_config
from src.services.background_refresh_service import get_background_refresh_service
from src.database.sqlite_manager import get_db_manager

logger = get_logger(__name__)


class MCPScheduler:
    """Periodic scheduler for MCP data sources with smart refresh intervals."""

    def __init__(self):
        """Initialize MCP scheduler."""
        self.config = get_scheduler_config()
        self.refresh_service = get_background_refresh_service()
        self.db_manager = get_db_manager()

        # Status tracking service for diagnostics
        from src.services.system_status_service import get_system_status_service
        self.status_service = get_system_status_service()

        # Track last refresh time per source
        self._last_refresh: Dict[str, datetime] = {}

        # Scheduler control
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

        logger.info("mcp_scheduler_initialized", config={
            "news_interval": self.config.NEWS_REFRESH_INTERVAL,
            "weather_interval": self.config.WEATHER_REFRESH_INTERVAL,
            "email_interval": self.config.EMAIL_REFRESH_INTERVAL,
            "calendar_interval": self.config.CALENDAR_REFRESH_INTERVAL,
        })

    def start(self):
        """Start the background scheduler in a separate thread."""
        if self._running:
            logger.warning("mcp_scheduler_already_running")
            return

        if not self.config.ENABLE_BACKGROUND_SCHEDULER:
            logger.info("mcp_scheduler_disabled_by_config")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler_loop,
            daemon=True,
            name="MCPScheduler"
        )
        self._scheduler_thread.start()
        logger.info("mcp_scheduler_started")

    def stop(self):
        """Stop the background scheduler."""
        if not self._running:
            return

        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)

        logger.info("mcp_scheduler_stopped")

    def _run_scheduler_loop(self):
        """Main scheduler loop running in background thread."""
        logger.info("mcp_scheduler_loop_started")

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run initial refresh on startup (if cache is empty/stale)
            loop.run_until_complete(self._check_and_refresh_all())

            # Main scheduler loop
            while self._running:
                try:
                    # Check each source and refresh if needed
                    loop.run_until_complete(self._check_and_refresh_all())

                    # Sleep for 60 seconds before next check
                    # (we check every minute but only refresh based on intervals)
                    for _ in range(60):
                        if not self._running:
                            break
                        asyncio.sleep(1)

                except Exception as e:
                    logger.error("mcp_scheduler_loop_error", error=str(e), exc_info=True)
                    asyncio.sleep(60)  # Wait before retrying

        finally:
            loop.close()
            logger.info("mcp_scheduler_loop_stopped")

    async def _check_and_refresh_all(self):
        """Check all sources and refresh if needed based on intervals."""
        sources_config = {
            'news': (self.config.NEWS_REFRESH_INTERVAL, self.config.ENABLE_NEWS_SYNC),
            'weather': (self.config.WEATHER_REFRESH_INTERVAL, self.config.ENABLE_WEATHER_SYNC),
            'email': (self.config.EMAIL_REFRESH_INTERVAL, self.config.ENABLE_EMAIL_SYNC),
            'calendar': (self.config.CALENDAR_REFRESH_INTERVAL, self.config.ENABLE_CALENDAR_SYNC),
        }

        for source, (interval_minutes, enabled) in sources_config.items():
            if not enabled:
                logger.debug(f"mcp_scheduler_source_disabled", source=source)
                # Record disabled status
                self.status_service.record_source_disabled(source)
                continue

            # Check if source needs refresh
            if await self._should_refresh(source, interval_minutes):
                logger.info(f"mcp_scheduler_refreshing_source", source=source)

                # Record refresh start
                self.status_service.record_refresh_start(source, interval_minutes)

                try:
                    success = await self.refresh_service.refresh_single_source(source)

                    if success:
                        self._last_refresh[source] = datetime.now()

                        # Get item count from database for reporting
                        items_fetched = await self._get_items_count(source)

                        # Record success
                        self.status_service.record_refresh_success(
                            source,
                            items_fetched=items_fetched,
                            next_refresh_minutes=interval_minutes
                        )

                        logger.info(f"mcp_scheduler_refresh_success",
                                  source=source,
                                  items=items_fetched,
                                  next_refresh_minutes=interval_minutes)
                    else:
                        # Record failure
                        self.status_service.record_refresh_error(source, "Refresh returned False")
                        logger.warning(f"mcp_scheduler_refresh_failed", source=source)

                except Exception as e:
                    # Record error
                    self.status_service.record_refresh_error(source, str(e))
                    logger.error(f"mcp_scheduler_refresh_error",
                               source=source,
                               error=str(e),
                               exc_info=True)

    async def _get_items_count(self, source: str) -> int:
        """Get count of items for a source from database.

        Args:
            source: Source name (news, weather, email, calendar)

        Returns:
            Number of items in database for that source
        """
        try:
            if source == 'news':
                articles = await self.db_manager.get_all_articles(limit=10000)
                return len(articles)
            elif source == 'weather':
                weather_data = await self.db_manager.get_cache('weather_data')
                return 1 if weather_data else 0
            elif source == 'email':
                # TODO: Implement when email storage is added
                return 0
            elif source == 'calendar':
                # TODO: Implement when calendar storage is added
                return 0
            else:
                return 0
        except Exception as e:
            logger.error(f"mcp_scheduler_count_error", source=source, error=str(e))
            return 0

    async def _should_refresh(self, source: str, interval_minutes: int) -> bool:
        """Check if a source should be refreshed based on its interval.

        Args:
            source: Source name (news, weather, email, calendar)
            interval_minutes: Refresh interval in minutes

        Returns:
            True if source should be refreshed
        """
        # Check cache age from database
        try:
            if source == 'news':
                cache_age_hours = await self.db_manager.get_news_cache_age_hours()
            elif source == 'weather':
                cache_age_hours = await self.db_manager.get_weather_cache_age_hours()
            else:
                # For email/calendar, check last refresh time
                if source not in self._last_refresh:
                    return True  # Never refreshed, do it now

                last_refresh = self._last_refresh[source]
                elapsed = (datetime.now() - last_refresh).total_seconds() / 60
                return elapsed >= interval_minutes

            # For news/weather, check cache age
            if cache_age_hours is None:
                return True  # No cache, refresh now

            cache_age_minutes = cache_age_hours * 60
            should_refresh = cache_age_minutes >= interval_minutes

            if should_refresh:
                logger.debug(f"mcp_scheduler_cache_stale",
                           source=source,
                           cache_age_minutes=cache_age_minutes,
                           interval_minutes=interval_minutes)

            return should_refresh

        except Exception as e:
            logger.error(f"mcp_scheduler_should_refresh_error",
                       source=source,
                       error=str(e))
            return False

    def get_status(self) -> Dict:
        """Get current scheduler status.

        Returns:
            Dict with scheduler status information
        """
        return {
            "running": self._running,
            "enabled": self.config.ENABLE_BACKGROUND_SCHEDULER,
            "last_refresh": {
                source: timestamp.isoformat() if timestamp else None
                for source, timestamp in self._last_refresh.items()
            },
            "intervals": {
                "news": self.config.NEWS_REFRESH_INTERVAL,
                "weather": self.config.WEATHER_REFRESH_INTERVAL,
                "email": self.config.EMAIL_REFRESH_INTERVAL,
                "calendar": self.config.CALENDAR_REFRESH_INTERVAL,
            }
        }


# Global singleton instance
_mcp_scheduler: Optional[MCPScheduler] = None


def get_mcp_scheduler() -> MCPScheduler:
    """Get the global MCP scheduler instance.

    Returns:
        MCPScheduler instance
    """
    global _mcp_scheduler

    if _mcp_scheduler is None:
        _mcp_scheduler = MCPScheduler()

    return _mcp_scheduler


def start_mcp_scheduler():
    """Start the global MCP scheduler."""
    # Check if scheduler is enabled before creating instance
    # This prevents heavy service initialization when scheduler is disabled
    config = get_scheduler_config()
    if not config.ENABLE_BACKGROUND_SCHEDULER:
        logger.info("mcp_scheduler_disabled_by_config")
        return

    scheduler = get_mcp_scheduler()
    scheduler.start()


def stop_mcp_scheduler():
    """Stop the global MCP scheduler."""
    global _mcp_scheduler

    if _mcp_scheduler is not None:
        _mcp_scheduler.stop()

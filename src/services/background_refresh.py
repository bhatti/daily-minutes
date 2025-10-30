"""Background Refresh Service - The heart of Daily Minutes automation.

This service runs scheduled refreshes of news and weather data in the background,
enabling the core "daily minutes" morning briefing functionality.

Features:
- Scheduled refreshes using APScheduler
- Observability via activity_log and structured logging
- Configurable schedule (default: 5am daily)
- Metrics tracking (articles fetched, duration, errors)
- Can run standalone or as part of the application
"""

import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.core.logging import get_logger
from src.database.sqlite_manager import get_db_manager
from src.core.config_manager import get_config_manager

logger = get_logger(__name__)


class BackgroundRefreshService:
    """
    Background service for automated data refreshes.

    This is the core automation component that enables "daily minutes" -
    fetching news and weather on a schedule without user interaction.
    """

    def __init__(self):
        """Initialize background refresh service."""
        self.scheduler = BackgroundScheduler()
        self.db_manager = get_db_manager()
        self.config_manager = get_config_manager()
        self.is_running = False

        logger.info("background_refresh_service_initialized")

    async def _log_refresh_event(
        self,
        refresh_type: str,
        status: str,
        details: Dict[str, Any]
    ):
        """
        Log refresh event to activity_log for observability.

        Args:
            refresh_type: Type of refresh (news, weather, both)
            status: Status (success, failed, partial)
            details: Event details (duration, counts, errors)
        """
        await self.db_manager.initialize()

        action = f"background_refresh_{refresh_type}"

        async with self.db_manager._get_connection() as db:
            await db.execute(
                """INSERT INTO activity_log
                   (action, entity_type, details, source)
                   VALUES (?, ?, ?, ?)""",
                (action, refresh_type, str(details), "background_service")
            )
            await db.commit()

        logger.info(
            "refresh_event_logged",
            refresh_type=refresh_type,
            status=status,
            details=details
        )

    async def refresh_news(self) -> Dict[str, Any]:
        """
        Refresh news data.

        Returns:
            Dict with refresh results (articles_count, duration, status, error)
        """
        start_time = time.time()
        result = {
            "status": "failed",
            "articles_count": 0,
            "duration_seconds": 0,
            "error": None
        }

        try:
            logger.info("background_news_refresh_started")

            # Import here to avoid circular dependencies
            from src.services.news_service import get_news_service

            news_service = get_news_service()
            max_articles = await self.config_manager.get_async("news.max_articles", 30)

            # Fetch news
            articles = await news_service.fetch_all_news(max_articles=max_articles)

            result["articles_count"] = len(articles) if articles else 0
            result["status"] = "success"

            logger.info(
                "background_news_refresh_completed",
                articles_count=result["articles_count"],
                duration=time.time() - start_time
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(
                "background_news_refresh_failed",
                error=str(e),
                duration=time.time() - start_time
            )

        result["duration_seconds"] = round(time.time() - start_time, 2)
        return result

    async def refresh_weather(self) -> Dict[str, Any]:
        """
        Refresh weather data.

        Returns:
            Dict with refresh results (status, duration, error)
        """
        start_time = time.time()
        result = {
            "status": "failed",
            "duration_seconds": 0,
            "error": None,
            "location": None
        }

        try:
            logger.info("background_weather_refresh_started")

            # Import here to avoid circular dependencies
            from src.connectors.weather import get_weather_service

            weather_service = get_weather_service()
            default_location = await self.config_manager.get_async("weather.default_location", "Seattle")

            # Fetch weather
            weather_data = await weather_service.get_current_weather(location=default_location)

            if weather_data:
                result["status"] = "success"
                result["location"] = weather_data.location

            logger.info(
                "background_weather_refresh_completed",
                location=default_location,
                duration=time.time() - start_time
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(
                "background_weather_refresh_failed",
                error=str(e),
                duration=time.time() - start_time
            )

        result["duration_seconds"] = round(time.time() - start_time, 2)
        return result

    async def refresh_all(self):
        """
        Refresh both news and weather.

        This is the main scheduled job that runs daily.
        """
        start_time = time.time()

        logger.info("background_refresh_all_started")

        try:
            # Refresh news and weather in parallel
            news_result, weather_result = await asyncio.gather(
                self.refresh_news(),
                self.refresh_weather(),
                return_exceptions=True
            )

            # Handle exceptions from gather
            if isinstance(news_result, Exception):
                news_result = {
                    "status": "failed",
                    "error": str(news_result),
                    "articles_count": 0,
                    "duration_seconds": 0
                }

            if isinstance(weather_result, Exception):
                weather_result = {
                    "status": "failed",
                    "error": str(weather_result),
                    "duration_seconds": 0
                }

            # Determine overall status
            if news_result["status"] == "success" and weather_result["status"] == "success":
                overall_status = "success"
            elif news_result["status"] == "success" or weather_result["status"] == "success":
                overall_status = "partial"
            else:
                overall_status = "failed"

            total_duration = round(time.time() - start_time, 2)

            # Log event to database
            await self._log_refresh_event(
                refresh_type="all",
                status=overall_status,
                details={
                    "news_articles": news_result.get("articles_count", 0),
                    "news_status": news_result["status"],
                    "news_error": news_result.get("error"),
                    "weather_status": weather_result["status"],
                    "weather_location": weather_result.get("location"),
                    "weather_error": weather_result.get("error"),
                    "total_duration_seconds": total_duration
                }
            )

            logger.info(
                "background_refresh_all_completed",
                status=overall_status,
                news_articles=news_result.get("articles_count", 0),
                duration=total_duration
            )

        except Exception as e:
            logger.error(
                "background_refresh_all_failed",
                error=str(e),
                duration=time.time() - start_time
            )

            # Still log the failed event
            await self._log_refresh_event(
                refresh_type="all",
                status="failed",
                details={
                    "error": str(e),
                    "total_duration_seconds": round(time.time() - start_time, 2)
                }
            )

    def _run_refresh_job(self):
        """
        Run refresh job (wrapper for scheduler).

        This is called by APScheduler and runs the async refresh_all().
        """
        logger.info("scheduled_refresh_triggered")

        # Run async function in event loop
        asyncio.run(self.refresh_all())

    def start(self, schedule: Optional[str] = None):
        """
        Start the background refresh service.

        Args:
            schedule: Cron schedule string (default: "0 5 * * *" = 5am daily)
        """
        if self.is_running:
            logger.warning("background_refresh_already_running")
            return

        # Default schedule: 5am every day
        if schedule is None:
            schedule = "0 5 * * *"  # minute hour day month day_of_week

        # Parse cron schedule
        cron_trigger = CronTrigger.from_crontab(schedule)

        # Add job to scheduler
        self.scheduler.add_job(
            self._run_refresh_job,
            trigger=cron_trigger,
            id="daily_refresh",
            name="Daily News and Weather Refresh",
            replace_existing=True
        )

        # Start scheduler
        self.scheduler.start()
        self.is_running = True

        logger.info(
            "background_refresh_service_started",
            schedule=schedule,
            next_run=self.scheduler.get_job("daily_refresh").next_run_time
        )

    def stop(self):
        """Stop the background refresh service."""
        if not self.is_running:
            return

        self.scheduler.shutdown(wait=True)
        self.is_running = False

        logger.info("background_refresh_service_stopped")

    def run_once(self):
        """Run a single refresh immediately (for testing or manual triggers)."""
        logger.info("manual_refresh_triggered")
        asyncio.run(self.refresh_all())

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the background service.

        Returns:
            Dict with service status
        """
        if not self.is_running:
            return {
                "running": False,
                "next_run": None
            }

        job = self.scheduler.get_job("daily_refresh")

        return {
            "running": True,
            "next_run": job.next_run_time.isoformat() if job and job.next_run_time else None,
            "schedule": str(job.trigger) if job else None
        }


# Singleton instance
_background_service: Optional[BackgroundRefreshService] = None


def get_background_service() -> BackgroundRefreshService:
    """
    Get or create background refresh service instance.

    Returns:
        BackgroundRefreshService instance
    """
    global _background_service
    if _background_service is None:
        _background_service = BackgroundRefreshService()
    return _background_service


# CLI entry point
def main():
    """CLI entry point for running background refresh service."""
    import sys

    service = get_background_service()

    if len(sys.argv) > 1 and sys.argv[1] == "once":
        # Run once and exit
        print("Running single refresh...")
        service.run_once()
        print("Refresh complete!")
    else:
        # Run as daemon
        print("Starting background refresh service...")
        print("Press Ctrl+C to stop")

        service.start()

        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping background refresh service...")
            service.stop()
            print("Service stopped")


if __name__ == "__main__":
    main()

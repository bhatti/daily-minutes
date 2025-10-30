"""Scheduler Configuration for Background Tasks.

All timing intervals are configurable here.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class SchedulerConfig(BaseSettings):
    """Configuration for background schedulers and refresh intervals."""

    # MCP Data Refresh Intervals (in minutes)
    EMAIL_REFRESH_INTERVAL: int = 60  # Fetch emails every 60 minutes
    CALENDAR_REFRESH_INTERVAL: int = 60  # Fetch calendar every 60 minutes
    NEWS_REFRESH_INTERVAL: int = 60  # Fetch news every 60 minutes
    WEATHER_REFRESH_INTERVAL: int = 60  # Fetch weather every 60 minutes

    # AI Brief Generation Interval (in minutes)
    BRIEF_GENERATION_INTERVAL: int = 5  # Generate AI brief every 5 minutes

    # UI Auto-Refresh Intervals (in seconds)
    UI_INITIAL_LOAD_DELAY: int = 60  # First auto-refresh after 60 seconds
    UI_REFRESH_INTERVAL: int = 300  # Then refresh every 5 minutes (300 seconds)

    # Cache Expiry (in hours)
    CACHE_EXPIRY_HOURS: int = 1  # Consider cache stale after 1 hour

    # Brief Generation Settings
    BRIEF_ONLY_ON_NEW_DATA: bool = True  # Only generate brief if MCP data changed
    BRIEF_MIN_ITEMS: int = 1  # Minimum items required to generate brief

    # Background Service Settings
    ENABLE_BACKGROUND_SCHEDULER: bool = True  # Master switch for all background tasks
    ENABLE_EMAIL_SYNC: bool = True
    ENABLE_CALENDAR_SYNC: bool = True
    ENABLE_NEWS_SYNC: bool = True
    ENABLE_WEATHER_SYNC: bool = True
    ENABLE_AUTO_BRIEF: bool = True

    class Config:
        env_prefix = "SCHEDULER_"
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env that don't match this config


# Global singleton instance
_scheduler_config: Optional[SchedulerConfig] = None


def get_scheduler_config() -> SchedulerConfig:
    """Get the global scheduler configuration instance.

    Returns:
        SchedulerConfig instance
    """
    global _scheduler_config

    if _scheduler_config is None:
        _scheduler_config = SchedulerConfig()

    return _scheduler_config


def reload_scheduler_config():
    """Reload the scheduler configuration from environment/file."""
    global _scheduler_config
    _scheduler_config = SchedulerConfig()
    return _scheduler_config

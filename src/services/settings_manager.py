"""SettingsManager - Centralized settings management for UI."""

import re
from typing import Dict, List, Any
from src.core.config_manager import get_config_manager
from src.core.logging import get_logger

logger = get_logger(__name__)


class SettingsManager:
    """
    Manager for application settings.

    Provides high-level interface for:
    - Loading settings organized by category
    - Updating settings with validation
    - Managing news sources and RSS feeds
    - Resetting to defaults
    """

    def __init__(self):
        """Initialize SettingsManager."""
        self.config_mgr = get_config_manager()

    def get_news_settings(self) -> Dict[str, Any]:
        """
        Get news-related settings.

        Returns:
            Dictionary of news settings
        """
        return {
            "max_articles": self.config_mgr.get("news.max_articles", 30),
            "max_per_source_base": self.config_mgr.get("news.max_per_source_base", 10),
            "content_threads": self.config_mgr.get("news.content_threads", 5),
        }

    def get_http_settings(self) -> Dict[str, Any]:
        """
        Get HTTP-related settings.

        Returns:
            Dictionary of HTTP settings
        """
        return {
            "verify_ssl": self.config_mgr.get("http.verify_ssl", True),
            "timeout": self.config_mgr.get("http.timeout", 30),
            "user_agent": self.config_mgr.get("http.user_agent", "Daily-Minutes/1.0"),
        }

    def get_weather_settings(self) -> Dict[str, Any]:
        """
        Get weather-related settings.

        Returns:
            Dictionary of weather settings
        """
        return {
            "default_location": self.config_mgr.get("weather.default_location", "Seattle"),
            "temp_unit": self.config_mgr.get("weather.temp_unit", "Fahrenheit"),
        }

    def get_ui_settings(self) -> Dict[str, Any]:
        """
        Get UI-related settings.

        Returns:
            Dictionary of UI settings
        """
        return {
            "theme": self.config_mgr.get("ui.theme", "light"),
            "auto_refresh": self.config_mgr.get("ui.auto_refresh", False),
            "refresh_interval_minutes": self.config_mgr.get("ui.refresh_interval_minutes", 15),
        }

    def get_refresh_intervals(self) -> Dict[str, int]:
        """
        Get refresh interval settings for all categories.

        Returns:
            Dictionary mapping category name to refresh interval in minutes
        """
        return {
            "news": self.config_mgr.get("refresh.news_interval_minutes", 15),
            "weather": self.config_mgr.get("refresh.weather_interval_minutes", 30),
            "email": self.config_mgr.get("refresh.email_interval_minutes", 5),
            "calendar": self.config_mgr.get("refresh.calendar_interval_minutes", 10),
        }

    def update_refresh_intervals(
        self,
        news_interval: int = None,
        weather_interval: int = None,
        email_interval: int = None,
        calendar_interval: int = None
    ):
        """
        Update refresh interval settings for individual categories.

        Args:
            news_interval: News refresh interval in minutes
            weather_interval: Weather refresh interval in minutes
            email_interval: Email refresh interval in minutes
            calendar_interval: Calendar refresh interval in minutes
        """
        if news_interval is not None:
            self.config_mgr.set("refresh.news_interval_minutes", news_interval)
            logger.info("updated_setting", key="refresh.news_interval_minutes", value=news_interval)

        if weather_interval is not None:
            self.config_mgr.set("refresh.weather_interval_minutes", weather_interval)
            logger.info("updated_setting", key="refresh.weather_interval_minutes", value=weather_interval)

        if email_interval is not None:
            self.config_mgr.set("refresh.email_interval_minutes", email_interval)
            logger.info("updated_setting", key="refresh.email_interval_minutes", value=email_interval)

        if calendar_interval is not None:
            self.config_mgr.set("refresh.calendar_interval_minutes", calendar_interval)
            logger.info("updated_setting", key="refresh.calendar_interval_minutes", value=calendar_interval)

    def validate_refresh_interval(self, interval: int) -> Dict[str, Any]:
        """
        Validate refresh interval value.

        Args:
            interval: Refresh interval in minutes

        Returns:
            Dictionary with validation result and errors
        """
        errors = []

        if interval < 1:
            errors.append("Refresh interval must be at least 1 minute")
        elif interval > 1440:  # 24 hours
            errors.append("Refresh interval cannot exceed 1440 minutes (24 hours)")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def get_news_sources_config(self) -> List[Dict[str, Any]]:
        """
        Get news sources configuration.

        Returns:
            List of source configurations
        """
        return self.config_mgr.get_news_sources()

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings organized by category.

        Returns:
            Dictionary with categorized settings
        """
        return {
            "news": self.get_news_settings(),
            "http": self.get_http_settings(),
            "weather": self.get_weather_settings(),
            "ui": self.get_ui_settings(),
            "sources": self.get_news_sources_config(),
        }

    def update_news_settings(
        self,
        max_articles: int = None,
        max_per_source_base: int = None,
        content_threads: int = None
    ):
        """
        Update news settings.

        Args:
            max_articles: Maximum total articles
            max_per_source_base: Base limit per source
            content_threads: Number of content fetching threads
        """
        if max_articles is not None:
            self.config_mgr.set("news.max_articles", max_articles)
            logger.info("updated_setting", key="news.max_articles", value=max_articles)

        if max_per_source_base is not None:
            self.config_mgr.set("news.max_per_source_base", max_per_source_base)
            logger.info("updated_setting", key="news.max_per_source_base", value=max_per_source_base)

        if content_threads is not None:
            self.config_mgr.set("news.content_threads", content_threads)
            logger.info("updated_setting", key="news.content_threads", value=content_threads)

    def update_http_settings(
        self,
        verify_ssl: bool = None,
        timeout: int = None,
        user_agent: str = None
    ):
        """
        Update HTTP settings.

        Args:
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
            user_agent: User agent string
        """
        if verify_ssl is not None:
            self.config_mgr.set("http.verify_ssl", verify_ssl)
            logger.info("updated_setting", key="http.verify_ssl", value=verify_ssl)

        if timeout is not None:
            self.config_mgr.set("http.timeout", timeout)
            logger.info("updated_setting", key="http.timeout", value=timeout)

        if user_agent is not None:
            self.config_mgr.set("http.user_agent", user_agent)
            logger.info("updated_setting", key="http.user_agent", value=user_agent)

    def update_weather_settings(
        self,
        default_location: str = None,
        temp_unit: str = None
    ):
        """
        Update weather settings.

        Args:
            default_location: Default location for weather
            temp_unit: Temperature unit (Celsius or Fahrenheit)
        """
        if default_location is not None:
            self.config_mgr.set("weather.default_location", default_location)
            logger.info("updated_setting", key="weather.default_location", value=default_location)

        if temp_unit is not None:
            self.config_mgr.set("weather.temp_unit", temp_unit)
            logger.info("updated_setting", key="weather.temp_unit", value=temp_unit)

    def validate_news_settings(
        self,
        max_articles: int,
        max_per_source_base: int,
        content_threads: int
    ) -> Dict[str, Any]:
        """
        Validate news settings.

        Args:
            max_articles: Maximum total articles
            max_per_source_base: Base limit per source
            content_threads: Number of content fetching threads

        Returns:
            Dictionary with validation result and errors
        """
        errors = []

        # Validate max_articles
        if max_articles <= 0:
            errors.append("Max articles must be greater than 0")
        elif max_articles > 100:
            errors.append("Max articles cannot exceed 100")

        # Validate max_per_source_base
        if max_per_source_base <= 0:
            errors.append("Max per source must be greater than 0")
        elif max_per_source_base > 50:
            errors.append("Max per source cannot exceed 50")

        # Validate content_threads
        if content_threads <= 0:
            errors.append("Content threads must be greater than 0")
        elif content_threads > 20:
            errors.append("Content threads cannot exceed 20")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def validate_rss_url(self, url: str) -> bool:
        """
        Validate RSS feed URL.

        Args:
            url: RSS feed URL

        Returns:
            True if valid, False otherwise
        """
        if not url:
            return False

        # Basic URL pattern matching
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return bool(url_pattern.match(url))

    def add_rss_feed(self, name: str, url: str):
        """
        Add a new RSS feed.

        Args:
            name: Feed name
            url: Feed URL
        """
        sources = self.get_news_sources_config()

        # Find RSS source
        rss_source = next((s for s in sources if s["type"] == "rss"), None)

        if rss_source:
            # Add new feed
            rss_source["feeds"].append({"name": name, "url": url})
        else:
            # Create RSS source with new feed
            sources.append({
                "type": "rss",
                "enabled": True,
                "feeds": [{"name": name, "url": url}]
            })

        self.config_mgr.set_news_sources(sources)
        logger.info("added_rss_feed", name=name, url=url)

    def remove_rss_feed(self, name: str):
        """
        Remove an RSS feed by name.

        Args:
            name: Feed name to remove
        """
        sources = self.get_news_sources_config()

        # Find RSS source
        rss_source = next((s for s in sources if s["type"] == "rss"), None)

        if rss_source:
            # Remove feed by name
            rss_source["feeds"] = [f for f in rss_source["feeds"] if f["name"] != name]

        self.config_mgr.set_news_sources(sources)
        logger.info("removed_rss_feed", name=name)

    def toggle_source_enabled(self, source_type: str, enabled: bool):
        """
        Toggle a news source enabled/disabled.

        Args:
            source_type: Source type (e.g., "hackernews", "rss")
            enabled: Whether to enable or disable
        """
        sources = self.get_news_sources_config()

        # Find source and update
        for source in sources:
            if source["type"] == source_type:
                source["enabled"] = enabled
                break

        self.config_mgr.set_news_sources(sources)
        logger.info("toggled_source", source_type=source_type, enabled=enabled)

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        # Reset news settings
        self.update_news_settings(
            max_articles=30,
            max_per_source_base=10,
            content_threads=5
        )

        # Reset HTTP settings
        self.update_http_settings(
            verify_ssl=True,
            timeout=30,
            user_agent="Daily-Minutes/1.0"
        )

        # Reset weather settings
        self.update_weather_settings(
            default_location="Seattle",
            temp_unit="Fahrenheit"
        )

        # Reset sources
        default_sources = [
            {"type": "hackernews", "enabled": True},
            {
                "type": "rss",
                "enabled": True,
                "feeds": [
                    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
                    {"name": "Ars Technica", "url": "https://arstechnica.com/feed/"},
                    {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
                ]
            }
        ]
        self.config_mgr.set_news_sources(default_sources)

        logger.info("settings_reset_to_defaults")


# Singleton instance
_settings_manager = None


def get_settings_manager() -> SettingsManager:
    """
    Get or create SettingsManager instance.

    Returns:
        SettingsManager instance
    """
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager

"""Unit tests for SettingsManager - TDD approach."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.models.base import DataSource


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager."""
    config_mgr = Mock()

    # Default values
    config_mgr.get.side_effect = lambda key, default: {
        "news.max_articles": 30,
        "news.max_per_source_base": 10,
        "news.content_threads": 5,
        "http.verify_ssl": True,
        "http.timeout": 30,
        "http.user_agent": "Daily-Minutes/1.0",
        "weather.default_location": "Seattle",
        "weather.temp_unit": "Fahrenheit",
        "ui.theme": "light",
        "ui.auto_refresh": False,
        "ui.refresh_interval_minutes": 15,
    }.get(key, default)

    config_mgr.get_news_sources.return_value = [
        {"type": "hackernews", "enabled": True},
        {
            "type": "rss",
            "enabled": True,
            "feeds": [
                {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
                {"name": "Ars Technica", "url": "https://arstechnica.com/feed/"},
            ]
        }
    ]

    config_mgr.set = Mock()
    config_mgr.set_news_sources = Mock()

    return config_mgr


@pytest.mark.asyncio
class TestSettingsManager:
    """Test SettingsManager functionality using TDD."""

    def test_load_news_settings(self, mock_config_manager):
        """Test loading news settings."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            settings = manager.get_news_settings()

            assert settings["max_articles"] == 30
            assert settings["max_per_source_base"] == 10
            assert settings["content_threads"] == 5

    def test_load_http_settings(self, mock_config_manager):
        """Test loading HTTP settings."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            settings = manager.get_http_settings()

            assert settings["verify_ssl"] is True
            assert settings["timeout"] == 30
            assert settings["user_agent"] == "Daily-Minutes/1.0"

    def test_load_news_sources(self, mock_config_manager):
        """Test loading news sources configuration."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            sources = manager.get_news_sources_config()

            assert len(sources) == 2
            assert sources[0]["type"] == "hackernews"
            assert sources[0]["enabled"] is True
            assert sources[1]["type"] == "rss"
            assert len(sources[1]["feeds"]) == 2

    def test_update_news_settings(self, mock_config_manager):
        """Test updating news settings."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            manager.update_news_settings(
                max_articles=50,
                max_per_source_base=15,
                content_threads=10
            )

            # Verify settings were updated
            assert mock_config_manager.set.call_count == 3
            mock_config_manager.set.assert_any_call("news.max_articles", 50)
            mock_config_manager.set.assert_any_call("news.max_per_source_base", 15)
            mock_config_manager.set.assert_any_call("news.content_threads", 10)

    def test_validate_news_settings_valid(self, mock_config_manager):
        """Test validation accepts valid news settings."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            result = manager.validate_news_settings(
                max_articles=30,
                max_per_source_base=10,
                content_threads=5
            )

            assert result["valid"] is True
            assert len(result["errors"]) == 0

    def test_validate_news_settings_invalid(self, mock_config_manager):
        """Test validation rejects invalid news settings."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            result = manager.validate_news_settings(
                max_articles=-10,  # Invalid: negative
                max_per_source_base=0,  # Invalid: zero
                content_threads=100  # Invalid: too high
            )

            assert result["valid"] is False
            assert len(result["errors"]) > 0

    def test_add_rss_feed(self, mock_config_manager):
        """Test adding a new RSS feed."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            manager.add_rss_feed("New Feed", "https://example.com/feed")

            # Verify set_news_sources was called
            mock_config_manager.set_news_sources.assert_called_once()

            # Get the updated sources
            updated_sources = mock_config_manager.set_news_sources.call_args[0][0]

            # Find RSS source
            rss_source = next(s for s in updated_sources if s["type"] == "rss")
            assert len(rss_source["feeds"]) == 3  # Original 2 + 1 new
            assert any(f["name"] == "New Feed" for f in rss_source["feeds"])

    def test_remove_rss_feed(self, mock_config_manager):
        """Test removing an RSS feed."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            manager.remove_rss_feed("TechCrunch")

            # Verify set_news_sources was called
            mock_config_manager.set_news_sources.assert_called_once()

            # Get the updated sources
            updated_sources = mock_config_manager.set_news_sources.call_args[0][0]

            # Find RSS source
            rss_source = next(s for s in updated_sources if s["type"] == "rss")
            assert len(rss_source["feeds"]) == 1  # Original 2 - 1 removed
            assert not any(f["name"] == "TechCrunch" for f in rss_source["feeds"])

    def test_toggle_source_enabled(self, mock_config_manager):
        """Test toggling a source enabled/disabled."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            manager.toggle_source_enabled("hackernews", False)

            # Verify set_news_sources was called
            mock_config_manager.set_news_sources.assert_called_once()

            # Get the updated sources
            updated_sources = mock_config_manager.set_news_sources.call_args[0][0]

            # Find HackerNews source
            hn_source = next(s for s in updated_sources if s["type"] == "hackernews")
            assert hn_source["enabled"] is False

    def test_get_all_settings(self, mock_config_manager):
        """Test getting all settings organized by category."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            all_settings = manager.get_all_settings()

            assert "news" in all_settings
            assert "http" in all_settings
            assert "weather" in all_settings
            assert "ui" in all_settings
            assert "sources" in all_settings

    def test_validate_rss_url(self, mock_config_manager):
        """Test RSS URL validation."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()

            # Valid URLs
            assert manager.validate_rss_url("https://example.com/feed") is True
            assert manager.validate_rss_url("http://example.com/rss") is True

            # Invalid URLs
            assert manager.validate_rss_url("not-a-url") is False
            assert manager.validate_rss_url("") is False

    def test_reset_to_defaults(self, mock_config_manager):
        """Test resetting all settings to defaults."""
        from src.services.settings_manager import SettingsManager

        with patch('src.services.settings_manager.get_config_manager', return_value=mock_config_manager):
            manager = SettingsManager()
            manager.reset_to_defaults()

            # Verify multiple set calls were made
            assert mock_config_manager.set.call_count > 0

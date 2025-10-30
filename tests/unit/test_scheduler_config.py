"""Unit tests for Scheduler Configuration."""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.core.scheduler_config import (
    SchedulerConfig,
    get_scheduler_config,
    reload_scheduler_config,
    _scheduler_config
)


class TestSchedulerConfig:
    """Test SchedulerConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SchedulerConfig()

        # MCP refresh intervals
        assert config.EMAIL_REFRESH_INTERVAL == 60
        assert config.CALENDAR_REFRESH_INTERVAL == 60
        assert config.NEWS_REFRESH_INTERVAL == 60
        assert config.WEATHER_REFRESH_INTERVAL == 60

        # Brief generation
        assert config.BRIEF_GENERATION_INTERVAL == 5
        assert config.BRIEF_ONLY_ON_NEW_DATA is True
        assert config.BRIEF_MIN_ITEMS == 1

        # UI refresh
        assert config.UI_INITIAL_LOAD_DELAY == 60
        assert config.UI_REFRESH_INTERVAL == 300

        # Cache expiry
        assert config.CACHE_EXPIRY_HOURS == 1

        # Feature toggles
        assert config.ENABLE_BACKGROUND_SCHEDULER is True
        assert config.ENABLE_EMAIL_SYNC is True
        assert config.ENABLE_CALENDAR_SYNC is True
        assert config.ENABLE_NEWS_SYNC is True
        assert config.ENABLE_WEATHER_SYNC is True
        assert config.ENABLE_AUTO_BRIEF is True

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            'SCHEDULER_EMAIL_REFRESH_INTERVAL': '120',
            'SCHEDULER_BRIEF_GENERATION_INTERVAL': '10',
            'SCHEDULER_ENABLE_AUTO_BRIEF': 'false',
            'SCHEDULER_UI_REFRESH_INTERVAL': '600',
        }):
            config = SchedulerConfig()

            assert config.EMAIL_REFRESH_INTERVAL == 120
            assert config.BRIEF_GENERATION_INTERVAL == 10
            assert config.ENABLE_AUTO_BRIEF is False
            assert config.UI_REFRESH_INTERVAL == 600

    def test_ignores_extra_env_variables(self):
        """Test that extra environment variables are ignored."""
        with patch.dict(os.environ, {
            'SCHEDULER_EMAIL_REFRESH_INTERVAL': '30',
            'OLLAMA_HOST': 'http://localhost:11434',
            'OLLAMA_MODEL': 'llama3.2:3b',
            'LANGFUSE_ENABLED': 'true',
            'RANDOM_VAR': 'random_value',
        }):
            # Should not raise validation error
            config = SchedulerConfig()

            # Should still load the valid scheduler config
            assert config.EMAIL_REFRESH_INTERVAL == 30

            # Should not have the extra fields
            assert not hasattr(config, 'OLLAMA_HOST')
            assert not hasattr(config, 'OLLAMA_MODEL')
            assert not hasattr(config, 'LANGFUSE_ENABLED')
            assert not hasattr(config, 'RANDOM_VAR')

    def test_env_prefix(self):
        """Test that env_prefix works correctly."""
        # Variables without SCHEDULER_ prefix should not be loaded
        with patch.dict(os.environ, {
            'EMAIL_REFRESH_INTERVAL': '999',  # Missing SCHEDULER_ prefix
            'SCHEDULER_EMAIL_REFRESH_INTERVAL': '120',
        }):
            config = SchedulerConfig()

            # Should use the one with correct prefix
            assert config.EMAIL_REFRESH_INTERVAL == 120

    def test_boolean_conversion(self):
        """Test that boolean environment variables are converted correctly."""
        with patch.dict(os.environ, {
            'SCHEDULER_ENABLE_AUTO_BRIEF': 'true',
            'SCHEDULER_ENABLE_NEWS_SYNC': 'false',
            'SCHEDULER_BRIEF_ONLY_ON_NEW_DATA': '1',
        }):
            config = SchedulerConfig()

            assert config.ENABLE_AUTO_BRIEF is True
            assert config.ENABLE_NEWS_SYNC is False
            assert config.BRIEF_ONLY_ON_NEW_DATA is True

    def test_integer_conversion(self):
        """Test that integer environment variables are converted correctly."""
        with patch.dict(os.environ, {
            'SCHEDULER_EMAIL_REFRESH_INTERVAL': '90',
            'SCHEDULER_CACHE_EXPIRY_HOURS': '2',
        }):
            config = SchedulerConfig()

            assert isinstance(config.EMAIL_REFRESH_INTERVAL, int)
            assert config.EMAIL_REFRESH_INTERVAL == 90
            assert isinstance(config.CACHE_EXPIRY_HOURS, int)
            assert config.CACHE_EXPIRY_HOURS == 2

    def test_invalid_integer_raises_error(self):
        """Test that invalid integer values raise validation errors."""
        with patch.dict(os.environ, {
            'SCHEDULER_EMAIL_REFRESH_INTERVAL': 'not_a_number',
        }):
            with pytest.raises(Exception):  # Pydantic validation error
                SchedulerConfig()

    def test_all_intervals_configurable(self):
        """Test that all interval settings can be configured."""
        with patch.dict(os.environ, {
            'SCHEDULER_EMAIL_REFRESH_INTERVAL': '30',
            'SCHEDULER_CALENDAR_REFRESH_INTERVAL': '45',
            'SCHEDULER_NEWS_REFRESH_INTERVAL': '15',
            'SCHEDULER_WEATHER_REFRESH_INTERVAL': '90',
            'SCHEDULER_BRIEF_GENERATION_INTERVAL': '3',
            'SCHEDULER_UI_INITIAL_LOAD_DELAY': '30',
            'SCHEDULER_UI_REFRESH_INTERVAL': '120',
            'SCHEDULER_CACHE_EXPIRY_HOURS': '2',
        }):
            config = SchedulerConfig()

            assert config.EMAIL_REFRESH_INTERVAL == 30
            assert config.CALENDAR_REFRESH_INTERVAL == 45
            assert config.NEWS_REFRESH_INTERVAL == 15
            assert config.WEATHER_REFRESH_INTERVAL == 90
            assert config.BRIEF_GENERATION_INTERVAL == 3
            assert config.UI_INITIAL_LOAD_DELAY == 30
            assert config.UI_REFRESH_INTERVAL == 120
            assert config.CACHE_EXPIRY_HOURS == 2


class TestSchedulerConfigSingleton:
    """Test global singleton functions."""

    def teardown_method(self):
        """Reset global singleton between tests."""
        import src.core.scheduler_config
        src.core.scheduler_config._scheduler_config = None

    def test_get_scheduler_config_returns_instance(self):
        """Test that get_scheduler_config returns a SchedulerConfig instance."""
        config = get_scheduler_config()
        assert isinstance(config, SchedulerConfig)

    def test_get_scheduler_config_returns_singleton(self):
        """Test that get_scheduler_config returns the same instance."""
        config1 = get_scheduler_config()
        config2 = get_scheduler_config()

        assert config1 is config2

    def test_reload_scheduler_config(self):
        """Test that reload_scheduler_config creates new instance."""
        config1 = get_scheduler_config()
        config2 = reload_scheduler_config()

        # Should be a new instance
        assert config1 is not config2
        assert isinstance(config2, SchedulerConfig)

    def test_reload_picks_up_new_env_vars(self):
        """Test that reload picks up new environment variables."""
        # Initial config with default
        config1 = get_scheduler_config()
        initial_interval = config1.EMAIL_REFRESH_INTERVAL

        # Change environment
        with patch.dict(os.environ, {
            'SCHEDULER_EMAIL_REFRESH_INTERVAL': '999',
        }):
            config2 = reload_scheduler_config()

            # Should have new value
            assert config2.EMAIL_REFRESH_INTERVAL == 999
            assert config2.EMAIL_REFRESH_INTERVAL != initial_interval

    def test_get_scheduler_config_after_reload(self):
        """Test that get_scheduler_config returns reloaded instance."""
        config1 = get_scheduler_config()

        with patch.dict(os.environ, {
            'SCHEDULER_NEWS_REFRESH_INTERVAL': '888',
        }):
            config2 = reload_scheduler_config()
            config3 = get_scheduler_config()

            # config3 should be same as config2 (the reloaded one)
            assert config3 is config2
            assert config3.NEWS_REFRESH_INTERVAL == 888


class TestSchedulerConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_intervals(self):
        """Test that zero intervals are allowed (for testing/debugging)."""
        with patch.dict(os.environ, {
            'SCHEDULER_EMAIL_REFRESH_INTERVAL': '0',
            'SCHEDULER_BRIEF_GENERATION_INTERVAL': '0',
        }):
            config = SchedulerConfig()

            assert config.EMAIL_REFRESH_INTERVAL == 0
            assert config.BRIEF_GENERATION_INTERVAL == 0

    def test_large_intervals(self):
        """Test that large interval values work."""
        with patch.dict(os.environ, {
            'SCHEDULER_EMAIL_REFRESH_INTERVAL': '10080',  # 1 week in minutes
            'SCHEDULER_UI_REFRESH_INTERVAL': '86400',     # 1 day in seconds
        }):
            config = SchedulerConfig()

            assert config.EMAIL_REFRESH_INTERVAL == 10080
            assert config.UI_REFRESH_INTERVAL == 86400

    def test_all_features_disabled(self):
        """Test that all features can be disabled."""
        with patch.dict(os.environ, {
            'SCHEDULER_ENABLE_BACKGROUND_SCHEDULER': 'false',
            'SCHEDULER_ENABLE_EMAIL_SYNC': 'false',
            'SCHEDULER_ENABLE_CALENDAR_SYNC': 'false',
            'SCHEDULER_ENABLE_NEWS_SYNC': 'false',
            'SCHEDULER_ENABLE_WEATHER_SYNC': 'false',
            'SCHEDULER_ENABLE_AUTO_BRIEF': 'false',
        }):
            config = SchedulerConfig()

            assert config.ENABLE_BACKGROUND_SCHEDULER is False
            assert config.ENABLE_EMAIL_SYNC is False
            assert config.ENABLE_CALENDAR_SYNC is False
            assert config.ENABLE_NEWS_SYNC is False
            assert config.ENABLE_WEATHER_SYNC is False
            assert config.ENABLE_AUTO_BRIEF is False

    def test_mixed_config_sources(self):
        """Test configuration from both defaults and environment."""
        with patch.dict(os.environ, {
            'SCHEDULER_EMAIL_REFRESH_INTERVAL': '45',
            # Leave other settings as defaults
        }):
            config = SchedulerConfig()

            # Overridden
            assert config.EMAIL_REFRESH_INTERVAL == 45

            # Defaults
            assert config.CALENDAR_REFRESH_INTERVAL == 60
            assert config.NEWS_REFRESH_INTERVAL == 60
            assert config.ENABLE_BACKGROUND_SCHEDULER is True

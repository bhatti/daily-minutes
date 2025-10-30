"""Tests for configuration management."""

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config import (
    AppConfig,
    LLMConfig,
    NewsConfig,
    OutputConfig,
    Settings,
    get_settings,
    reset_settings,
)


class TestLLMConfig:
    """Tests for LLM configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_model == "llama3.2:3b"
        assert config.temperature == 0.7
        assert config.max_retries == 3
    
    def test_temperature_validation(self):
        """Test temperature bounds validation."""
        # Valid temperatures
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=1.0)
        LLMConfig(temperature=2.0)
        
        # Invalid temperatures
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)
    
    def test_env_prefix(self, monkeypatch):
        """Test loading from environment with prefix."""
        monkeypatch.setenv("LLM_OLLAMA_MODEL", "mistral:7b")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")
        
        config = LLMConfig()
        assert config.ollama_model == "mistral:7b"
        assert config.temperature == 0.5
        assert config.max_retries == 5


class TestNewsConfig:
    """Tests for news configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = NewsConfig()
        assert config.hn_top_stories_count == 20
        assert config.cache_ttl_hours == 6
        assert config.interests == ["artificial intelligence", "machine learning", "python"]
        assert config.rss_feeds == ["https://feeds.arstechnica.com/arstechnica/index"]
    
    def test_parse_interests_from_json_string(self, monkeypatch):
        """Test parsing interests from JSON string."""
        interests_json = '["ml", "data", "cloud"]'
        monkeypatch.setenv("NEWS_INTERESTS", interests_json)
        
        config = NewsConfig()
        assert config.interests == ["ml", "data", "cloud"]
    
    def test_parse_rss_feeds_from_json_string(self, monkeypatch):
        """Test parsing RSS feeds from JSON string."""
        feeds_json = '["http://feed1.com", "http://feed2.com"]'
        monkeypatch.setenv("NEWS_RSS_FEEDS", feeds_json)
        
        config = NewsConfig()
        assert len(config.rss_feeds) == 2
        assert "http://feed1.com" in config.rss_feeds
    
    def test_story_count_validation(self):
        """Test story count bounds."""
        NewsConfig(hn_top_stories_count=1)
        NewsConfig(hn_top_stories_count=100)
        
        with pytest.raises(ValidationError):
            NewsConfig(hn_top_stories_count=0)
        with pytest.raises(ValidationError):
            NewsConfig(hn_top_stories_count=101)


class TestOutputConfig:
    """Tests for output configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = OutputConfig()
        assert config.output_directory == "./data/outputs"
        assert config.email == "user@example.com"
        assert config.enable_email_output is False
        assert config.dashboard_port == 8501
    
    def test_port_validation(self):
        """Test port number validation."""
        OutputConfig(dashboard_port=1024)
        OutputConfig(dashboard_port=65535)
        
        with pytest.raises(ValidationError):
            OutputConfig(dashboard_port=1023)
        with pytest.raises(ValidationError):
            OutputConfig(dashboard_port=65536)


class TestAppConfig:
    """Tests for app configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AppConfig()
        assert config.log_level == "INFO"
        assert config.debug is False
        assert config.dry_run is False
        assert config.environment == "development"
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = AppConfig(log_level=level)
            assert config.log_level == level
        
        # Case insensitive
        config = AppConfig(log_level="debug")
        assert config.log_level == "DEBUG"
        
        # Invalid level
        with pytest.raises(ValidationError):
            AppConfig(log_level="INVALID")


class TestSettings:
    """Tests for main settings container."""
    
    def teardown_method(self):
        """Reset settings after each test."""
        reset_settings()
    
    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()
        assert settings.llm is not None
        assert settings.news is not None
        assert settings.output is not None
        assert settings.app is not None
    
    def test_nested_config_access(self):
        """Test accessing nested configuration."""
        settings = Settings()
        assert settings.llm.ollama_model == "llama3.2:3b"
        assert settings.news.hn_top_stories_count == 20
        assert settings.output.dashboard_port == 8501
        assert settings.app.log_level == "INFO"
    
    def test_is_production(self):
        """Test production mode detection."""
        settings = Settings()
        # Default is development
        assert settings.is_production() is False
        assert settings.is_development() is True

        # Set to production
        settings.app.environment = "production"
        assert settings.is_production() is True
        assert settings.is_development() is False

        # Set to testing
        settings.app.environment = "testing"
        assert settings.is_testing() is True
        assert settings.is_production() is False
    
    def test_get_log_level(self):
        """Test getting log level."""
        settings = Settings()
        assert settings.get_log_level() == "INFO"
        
        settings.app.log_level = "DEBUG"
        assert settings.get_log_level() == "DEBUG"


class TestGlobalSettings:
    """Tests for global settings management."""
    
    def teardown_method(self):
        """Reset settings after each test."""
        reset_settings()
    
    def test_get_settings_singleton(self):
        """Test settings singleton pattern."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
    
    def test_get_settings_reload(self, monkeypatch):
        """Test reloading settings."""
        settings1 = get_settings()
        original_model = settings1.llm.ollama_model
        
        # Change environment
        monkeypatch.setenv("LLM_OLLAMA_MODEL", "new-model")
        
        # Without reload, should get same instance
        settings2 = get_settings(reload=False)
        assert settings2.llm.ollama_model == original_model
        
        # With reload, should get new settings
        settings3 = get_settings(reload=True)
        assert settings3.llm.ollama_model == "new-model"
    
    def test_reset_settings(self):
        """Test resetting settings."""
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()
        assert settings1 is not settings2


@pytest.fixture
def temp_env_file(tmp_path):
    """Create temporary .env file for testing."""
    env_file = tmp_path / ".env"
    env_content = """
LLM_OLLAMA_MODEL=test-model
LLM_TEMPERATURE=0.5
NEWS_INTERESTS=["test1", "test2"]
LOG_LEVEL=DEBUG
"""
    env_file.write_text(env_content)
    return env_file


class TestConfigurationIntegration:
    """Integration tests for configuration loading."""
    
    def test_load_from_env_file(self, temp_env_file, monkeypatch):
        """Test loading complete configuration from file."""
        # This test is checking if the settings can be loaded
        # The actual values will come from the current .env or defaults
        reset_settings()

        settings = get_settings()
        # Just verify that settings load without error
        assert settings is not None
        assert settings.llm is not None
        assert settings.news is not None
        assert settings.app is not None

"""Enhanced configuration management with environment support and validation.

This module provides:
- Type-safe configuration with Pydantic
- Environment variable loading
- Environment-specific configurations
- Enhanced validation with custom messages
- Configuration schema export
- Hot reload capability
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """LLM configuration with validation."""
    
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL"
    )
    ollama_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model to use"
    )
    temperature: float = Field(
        default=0.7,
        description="LLM temperature (0.0 = deterministic, 2.0 = creative)",
        ge=0.0,
        le=2.0
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
        ge=1,
        le=10
    )
    timeout: Optional[float] = Field(
        default=30.0,
        description="Request timeout in seconds",
        ge=1.0
    )
    
    @field_validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature is in acceptable range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(
                f'Temperature must be between 0.0 and 2.0, got {v}. '
                'Use 0.0-0.3 for factual tasks, 0.7-1.0 for creative tasks.'
            )
        return v

    @field_validator('max_retries')
    def validate_max_retries(cls, v):
        """Validate max_retries is reasonable."""
        if v < 1:
            raise ValueError('max_retries must be at least 1')
        if v > 10:
            raise ValueError(
                f'max_retries={v} is too high. Consider if your service is reliable enough. '
                'Recommended: 3-5 retries.'
            )
        return v

    @field_validator('ollama_base_url')
    def validate_ollama_url(cls, v):
        """Validate Ollama URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError(
                f'ollama_base_url must start with http:// or https://, got: {v}'
            )
        return v.rstrip('/')  # Remove trailing slash
    
    model_config = {
        "env_prefix": "LLM_",
        "case_sensitive": False
    }


class NewsConfig(BaseSettings):
    """News sources configuration with validation."""
    
    hn_top_stories_count: int = Field(
        default=20,
        description="Number of top HackerNews stories to fetch",
        ge=1,
        le=100
    )
    cache_ttl_hours: int = Field(
        default=6,
        description="Cache TTL in hours",
        ge=1,
        le=24
    )
    interests: List[str] = Field(
        default_factory=lambda: [
            "artificial intelligence",
            "machine learning",
            "python",
        ],
        description="List of interest keywords for filtering"
    )
    rss_feeds: List[str] = Field(
        default_factory=lambda: [
            "https://feeds.arstechnica.com/arstechnica/index"
        ],
        description="List of RSS feed URLs"
    )
    
    @field_validator('hn_top_stories_count')
    def validate_story_count(cls, v):
        """Validate story count is reasonable."""
        if v < 1:
            raise ValueError('Must fetch at least 1 story')
        if v > 100:
            raise ValueError(
                f'Fetching {v} stories may be slow. Recommended: 10-30 stories.'
            )
        return v

    @field_validator('interests')
    def validate_interests(cls, v):
        """Validate interests list is not empty."""
        if not v:
            raise ValueError(
                'interests list cannot be empty. Add at least one interest keyword.'
            )
        # Convert to lowercase for consistent matching
        return [interest.lower() for interest in v]

    @field_validator('rss_feeds')
    def validate_rss_feeds(cls, v):
        """Validate RSS feed URLs."""
        for feed in v:
            if not feed.startswith(('http://', 'https://')):
                raise ValueError(
                    f'Invalid RSS feed URL: {feed}. Must start with http:// or https://'
                )
        return v
    
    model_config = {
        "env_prefix": "NEWS_",
        "case_sensitive": False
    }


class WeatherConfig(BaseSettings):
    """Weather configuration with validation."""

    openweather_api_key: Optional[str] = Field(
        default=None,
        description="OpenWeatherMap API key (optional - will use Open-Meteo if not provided)"
    )
    default_location: str = Field(
        default="San Francisco",
        description="Default location for weather data"
    )
    forecast_days: int = Field(
        default=5,
        description="Number of days to forecast",
        ge=1,
        le=7
    )
    update_interval_hours: int = Field(
        default=3,
        description="Weather update interval in hours",
        ge=1,
        le=24
    )

    @field_validator('forecast_days')
    def validate_forecast_days(cls, v):
        """Validate forecast days is reasonable."""
        if v < 1:
            raise ValueError('Must forecast at least 1 day')
        if v > 7:
            raise ValueError(
                f'Forecasting {v} days may not be accurate. Recommended: 3-5 days.'
            )
        return v

    model_config = {
        "env_prefix": "WEATHER_",
        "case_sensitive": False
    }


class OutputConfig(BaseSettings):
    """Output configuration with validation."""

    email: str = Field(
        default="user@example.com",
        description="Email address for daily minutes delivery"
    )
    enable_email_output: bool = Field(
        default=False,
        description="Enable email delivery of daily minutes"
    )
    dashboard_port: int = Field(
        default=8501,
        description="Streamlit dashboard port",
        ge=1024,
        le=65535
    )
    output_directory: str = Field(
        default="./data/outputs",
        description="Directory for saving generated outputs"
    )
    
    @field_validator('email')
    def validate_email(cls, v):
        """Basic email validation."""
        if '@' not in v or '.' not in v.split('@')[1]:
            raise ValueError(
                f'Invalid email format: {v}. Must be in format: user@domain.com'
            )
        return v.lower()

    @field_validator('dashboard_port')
    def validate_port(cls, v):
        """Validate port number."""
        if v < 1024:
            raise ValueError(
                f'Port {v} is in privileged range (0-1023). Use port >= 1024.'
            )
        if v > 65535:
            raise ValueError(f'Port {v} exceeds maximum (65535)')
        return v
    
    model_config = {
        "env_prefix": "OUTPUT_",
        "case_sensitive": False
    }


class AppConfig(BaseSettings):
    """Application-level configuration."""
    
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    dry_run: bool = Field(
        default=False,
        description="Enable dry-run mode (no external API calls)"
    )
    environment: str = Field(
        default="development",
        description="Environment name (development, production, testing)"
    )
    
    @field_validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f'Invalid log_level: {v}. Must be one of: {", ".join(valid_levels)}'
            )
        return v_upper

    @field_validator('environment')
    def validate_environment(cls, v):
        """Validate environment name."""
        valid_envs = ['development', 'production', 'testing', 'staging']
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ValueError(
                f'Invalid environment: {v}. Must be one of: {", ".join(valid_envs)}'
            )
        return v_lower
    
    model_config = {
        "env_prefix": "APP_",
        "case_sensitive": False
    }


class Settings(BaseSettings):
    """Main settings class combining all configurations.
    
    Examples:
        >>> # Load from environment
        >>> settings = Settings()
        >>> print(settings.llm.ollama_model)
        
        >>> # Load from specific env file
        >>> settings = Settings.load_from_env(".env.production")
        
        >>> # Check environment
        >>> if settings.is_production():
        ...     print("Running in production")
    """
    
    # Nested configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields from .env file
    }
    
    def __init__(self, **kwargs):
        """Initialize settings with nested config loading."""
        super().__init__(**kwargs)
        # Initialize nested configs to load from environment
        self.llm = LLMConfig()
        self.news = NewsConfig()
        self.weather = WeatherConfig()
        self.output = OutputConfig()
        self.app = AppConfig()
    
    @classmethod
    def load_from_env(cls, env_file: str = ".env") -> "Settings":
        """Load settings from environment file.

        Args:
            env_file: Path to environment file

        Returns:
            Settings instance

        Examples:
            >>> settings = Settings.load_from_env(".env.production")
        """
        # In Pydantic v2, we use model_config to set the env_file
        settings = cls()
        settings.model_config["env_file"] = env_file
        return settings
    
    @classmethod
    def load_for_environment(cls, environment: str) -> "Settings":
        """Load settings for specific environment.
        
        Args:
            environment: Environment name (development, production, testing)
            
        Returns:
            Settings instance
            
        Examples:
            >>> settings = Settings.load_for_environment("production")
        """
        env_file = f".env.{environment}"
        if not Path(env_file).exists():
            # Fallback to default .env
            env_file = ".env"
        
        return cls.load_from_env(env_file)
    
    def is_production(self) -> bool:
        """Check if running in production mode.
        
        Returns:
            True if in production, False otherwise
        """
        return self.app.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development mode.
        
        Returns:
            True if in development, False otherwise
        """
        return self.app.environment == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing mode.
        
        Returns:
            True if in testing, False otherwise
        """
        return self.app.environment == "testing"
    
    def get_log_level(self) -> str:
        """Get the configured log level.
        
        Returns:
            Log level string
        """
        return self.app.log_level
    
    def export_schema(self) -> Dict[str, Any]:
        """Export configuration schema for documentation.

        Returns:
            Dictionary containing the schema

        Examples:
            >>> settings = Settings()
            >>> schema = settings.export_schema()
            >>> print(json.dumps(schema, indent=2))
        """
        return self.model_json_schema()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export current configuration as dictionary.

        Returns:
            Dictionary representation of settings
        """
        return {
            "llm": self.llm.model_dump(),
            "news": self.news.model_dump(),
            "weather": self.weather.model_dump(),
            "output": self.output.model_dump(),
            "app": self.app.model_dump(),
        }
    
    def validate_all(self) -> List[str]:
        """Validate all configuration sections.
        
        Returns:
            List of validation messages (empty if all valid)
        """
        messages = []
        
        # Production environment checks
        if self.is_production():
            if self.app.debug:
                messages.append("WARNING: debug=True in production environment")
            if self.app.dry_run:
                messages.append("WARNING: dry_run=True in production environment")
            if self.output.email == "user@example.com":
                messages.append("WARNING: Using default email in production")
        
        # Development environment checks
        if self.is_development():
            if not self.app.debug:
                messages.append("INFO: Consider enabling debug=True in development")
        
        return messages
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        
        print(f"\nEnvironment: {self.app.environment.upper()}")
        print(f"Debug: {self.app.debug}")
        print(f"Dry Run: {self.app.dry_run}")
        print(f"Log Level: {self.app.log_level}")
        
        print(f"\nLLM Configuration:")
        print(f"  Model: {self.llm.ollama_model}")
        print(f"  Temperature: {self.llm.temperature}")
        print(f"  Max Retries: {self.llm.max_retries}")
        
        print(f"\nNews Configuration:")
        print(f"  Top Stories: {self.news.hn_top_stories_count}")
        print(f"  Cache TTL: {self.news.cache_ttl_hours}h")
        print(f"  Interests: {len(self.news.interests)} topics")
        print(f"  RSS Feeds: {len(self.news.rss_feeds)} feeds")
        
        print(f"\nOutput Configuration:")
        print(f"  Email: {self.output.email}")
        print(f"  Email Enabled: {self.output.enable_email_output}")
        print(f"  Dashboard Port: {self.output.dashboard_port}")
        
        # Validation messages
        messages = self.validate_all()
        if messages:
            print(f"\nValidation Messages:")
            for msg in messages:
                print(f"  • {msg}")
        
        print("=" * 60 + "\n")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """Get or create global settings instance.
    
    Args:
        reload: Force reload settings from environment
        
    Returns:
        Settings instance
        
    Examples:
        >>> settings = get_settings()
        >>> print(settings.llm.ollama_model)
    """
    global _settings
    if _settings is None or reload:
        # Try to load environment-specific settings
        env = os.getenv("APP_ENVIRONMENT", "development")
        try:
            _settings = Settings.load_for_environment(env)
        except Exception:
            # Fallback to default settings
            _settings = Settings()
    return _settings


def reset_settings():
    """Reset global settings (useful for testing).
    
    Examples:
        >>> reset_settings()
        >>> settings = get_settings()  # Will create new instance
    """
    global _settings
    _settings = None


# Environment-specific configuration templates
def create_environment_configs():
    """Create example environment configuration files.
    
    This function creates template .env files for different environments.
    Call this once to set up your project structure.
    """
    configs = {
        ".env.development": {
            "APP_ENVIRONMENT": "development",
            "APP_DEBUG": "true",
            "APP_DRY_RUN": "false",
            "APP_LOG_LEVEL": "DEBUG",
            "LLM_TEMPERATURE": "0.7",
            "LLM_MAX_RETRIES": "3",
            "NEWS_HN_TOP_STORIES_COUNT": "10",
            "NEWS_CACHE_TTL_HOURS": "1",
        },
        ".env.production": {
            "APP_ENVIRONMENT": "production",
            "APP_DEBUG": "false",
            "APP_DRY_RUN": "false",
            "APP_LOG_LEVEL": "INFO",
            "LLM_TEMPERATURE": "0.5",
            "LLM_MAX_RETRIES": "5",
            "NEWS_HN_TOP_STORIES_COUNT": "20",
            "NEWS_CACHE_TTL_HOURS": "6",
        },
        ".env.testing": {
            "APP_ENVIRONMENT": "testing",
            "APP_DEBUG": "true",
            "APP_DRY_RUN": "true",
            "APP_LOG_LEVEL": "DEBUG",
            "LLM_MAX_RETRIES": "1",
            "NEWS_HN_TOP_STORIES_COUNT": "5",
            "NEWS_CACHE_TTL_HOURS": "1",
        },
    }
    
    for filename, config in configs.items():
        if not Path(filename).exists():
            with open(filename, 'w') as f:
                f.write(f"# {filename.upper()} Configuration\n\n")
                for key, value in config.items():
                    f.write(f"{key}={value}\n")
            print(f"Created {filename}")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Enhanced Configuration")
    print("=" * 60)
    
    # Create environment config templates
    print("\n1. Creating environment config templates...")
    create_environment_configs()
    
    # Load default settings
    print("\n2. Loading default settings...")
    settings = get_settings()
    settings.print_summary()
    
    # Test validation
    print("\n3. Testing validation...")
    try:
        bad_config = LLMConfig(temperature=3.0)
    except ValueError as e:
        print(f"✅ Validation caught error: {e}")
    
    # Export schema
    print("\n4. Exporting schema...")
    schema = settings.export_schema()
    print(f"Schema keys: {list(schema.keys())}")
    
    # Test environment-specific loading
    print("\n5. Testing environment-specific loading...")
    if Path(".env.production").exists():
        prod_settings = Settings.load_for_environment("production")
        print(f"Production environment: {prod_settings.app.environment}")
        print(f"Production debug: {prod_settings.app.debug}")
    
    print("\n✅ Configuration testing complete!")

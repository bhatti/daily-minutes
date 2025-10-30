#!/usr/bin/env python3
"""Unit tests for Background Refresh Service using TDD approach."""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.background_refresh import BackgroundRefreshService, get_background_service


class TestBackgroundRefreshService:
    """Test BackgroundRefreshService with mocked dependencies."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create background refresh service for testing."""
        service = BackgroundRefreshService()
        return service

    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert service.scheduler is not None
        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_refresh_news_success(self, service):
        """Test successful news refresh."""
        # Mock news service
        mock_news_service = AsyncMock()
        mock_news_service.fetch_all_news = AsyncMock(return_value=[
            MagicMock(title="Article 1"),
            MagicMock(title="Article 2"),
            MagicMock(title="Article 3")
        ])

        with patch('src.services.news_service.get_news_service', return_value=mock_news_service):
            with patch.object(service.config_manager, 'get_async', return_value=30):
                result = await service.refresh_news()

        assert result["status"] == "success"
        assert result["articles_count"] == 3
        assert result["error"] is None
        assert result["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_refresh_news_failure(self, service):
        """Test news refresh handles errors."""
        # Mock news service to raise exception
        mock_news_service = AsyncMock()
        mock_news_service.fetch_all_news = AsyncMock(side_effect=Exception("Network error"))

        with patch('src.services.news_service.get_news_service', return_value=mock_news_service):
            with patch.object(service.config_manager, 'get_async', return_value=30):
                result = await service.refresh_news()

        assert result["status"] == "failed"
        assert result["articles_count"] == 0
        assert result["error"] == "Network error"
        assert result["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_refresh_weather_success(self, service):
        """Test successful weather refresh."""
        # Mock weather service
        mock_weather_data = MagicMock()
        mock_weather_data.location = "Seattle"

        mock_weather_service = AsyncMock()
        mock_weather_service.get_current_weather = AsyncMock(return_value=mock_weather_data)

        with patch('src.connectors.weather.get_weather_service', return_value=mock_weather_service):
            with patch.object(service.config_manager, 'get_async', return_value="Seattle"):
                result = await service.refresh_weather()

        assert result["status"] == "success"
        assert result["location"] == "Seattle"
        assert result["error"] is None
        assert result["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_refresh_weather_failure(self, service):
        """Test weather refresh handles errors."""
        # Mock weather service to raise exception
        mock_weather_service = AsyncMock()
        mock_weather_service.get_current_weather = AsyncMock(side_effect=Exception("API error"))

        with patch('src.connectors.weather.get_weather_service', return_value=mock_weather_service):
            with patch.object(service.config_manager, 'get_async', return_value="Seattle"):
                result = await service.refresh_weather()

        assert result["status"] == "failed"
        assert result["error"] == "API error"
        assert result["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_log_refresh_event(self, service):
        """Test refresh event logging to database."""
        # Mock database
        mock_db = AsyncMock()
        mock_cursor = AsyncMock()

        with patch.object(service.db_manager, 'initialize', new_callable=AsyncMock):
            with patch.object(service.db_manager, '_get_connection') as mock_conn:
                mock_conn.return_value.__aenter__.return_value = mock_db

                await service._log_refresh_event(
                    refresh_type="news",
                    status="success",
                    details={"articles_count": 5}
                )

        # Verify database insert was called
        mock_db.execute.assert_called_once()
        args = mock_db.execute.call_args[0]
        assert "INSERT INTO activity_log" in args[0]
        assert "background_refresh_news" in args[1]

    @pytest.mark.asyncio
    async def test_refresh_all_success(self, service):
        """Test refresh all (news + weather) with both succeeding."""
        # Mock both services
        mock_news_service = AsyncMock()
        mock_news_service.fetch_all_news = AsyncMock(return_value=[MagicMock()])

        mock_weather_data = MagicMock()
        mock_weather_data.location = "Seattle"
        mock_weather_service = AsyncMock()
        mock_weather_service.get_current_weather = AsyncMock(return_value=mock_weather_data)

        with patch('src.services.news_service.get_news_service', return_value=mock_news_service):
            with patch('src.connectors.weather.get_weather_service', return_value=mock_weather_service):
                with patch.object(service.config_manager, 'get_async', return_value=30):
                    with patch.object(service.config_manager, 'get_async', return_value="Seattle"):
                        with patch.object(service, '_log_refresh_event', new_callable=AsyncMock) as mock_log:
                            await service.refresh_all()

        # Verify event was logged with success status
        mock_log.assert_called_once()
        args = mock_log.call_args[1]
        assert args["status"] == "success"
        assert args["details"]["news_articles"] == 1
        assert args["details"]["news_status"] == "success"
        assert args["details"]["weather_status"] == "success"

    @pytest.mark.asyncio
    async def test_refresh_all_partial_success(self, service):
        """Test refresh all with one service failing."""
        # Mock news service to succeed
        mock_news_service = AsyncMock()
        mock_news_service.fetch_all_news = AsyncMock(return_value=[MagicMock(), MagicMock()])

        # Mock weather service to fail
        mock_weather_service = AsyncMock()
        mock_weather_service.get_current_weather = AsyncMock(side_effect=Exception("Weather API down"))

        with patch('src.services.news_service.get_news_service', return_value=mock_news_service):
            with patch('src.connectors.weather.get_weather_service', return_value=mock_weather_service):
                with patch.object(service.config_manager, 'get_async', return_value=30):
                    with patch.object(service.config_manager, 'get_async', return_value="Seattle"):
                        with patch.object(service, '_log_refresh_event', new_callable=AsyncMock) as mock_log:
                            await service.refresh_all()

        # Verify event was logged with partial status
        mock_log.assert_called_once()
        args = mock_log.call_args[1]
        assert args["status"] == "partial"
        assert args["details"]["news_status"] == "success"
        assert args["details"]["weather_status"] == "failed"

    @pytest.mark.asyncio
    async def test_refresh_all_complete_failure(self, service):
        """Test refresh all with both services failing."""
        # Mock both services to fail
        mock_news_service = AsyncMock()
        mock_news_service.fetch_all_news = AsyncMock(side_effect=Exception("News error"))

        mock_weather_service = AsyncMock()
        mock_weather_service.get_current_weather = AsyncMock(side_effect=Exception("Weather error"))

        with patch('src.services.news_service.get_news_service', return_value=mock_news_service):
            with patch('src.connectors.weather.get_weather_service', return_value=mock_weather_service):
                with patch.object(service.config_manager, 'get_async', return_value=30):
                    with patch.object(service.config_manager, 'get_async', return_value="Seattle"):
                        with patch.object(service, '_log_refresh_event', new_callable=AsyncMock) as mock_log:
                            await service.refresh_all()

        # Verify event was logged with failed status
        mock_log.assert_called_once()
        args = mock_log.call_args[1]
        assert args["status"] == "failed"

    def test_scheduler_start_stop(self, service):
        """Test scheduler can start and stop."""
        # Start scheduler
        service.start(schedule="0 5 * * *")

        assert service.is_running is True
        assert service.scheduler.get_job("daily_refresh") is not None

        # Stop scheduler
        service.stop()

        assert service.is_running is False

    def test_scheduler_prevents_duplicate_start(self, service):
        """Test starting already running service logs warning."""
        service.start()

        # Try to start again
        service.start()  # Should log warning but not error

        service.stop()

    def test_get_status_not_running(self, service):
        """Test get_status when service is not running."""
        status = service.get_status()

        assert status["running"] is False
        assert status["next_run"] is None

    def test_get_status_running(self, service):
        """Test get_status when service is running."""
        service.start(schedule="0 5 * * *")

        status = service.get_status()

        assert status["running"] is True
        assert status["next_run"] is not None
        assert "5" in status["schedule"]  # Should contain hour 5

        service.stop()

    def test_singleton_pattern(self):
        """Test get_background_service returns singleton."""
        service1 = get_background_service()
        service2 = get_background_service()

        assert service1 is service2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

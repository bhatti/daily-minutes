"""Unit tests for BackgroundRefreshService."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.services.background_refresh_service import (
    BackgroundRefreshService,
    get_background_refresh_service
)


@pytest.fixture
def refresh_service():
    """Create a fresh BackgroundRefreshService for each test."""
    return BackgroundRefreshService()


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    db = AsyncMock()
    db.save_article = AsyncMock(return_value=1)
    db.set_setting = AsyncMock()
    db.get_setting = AsyncMock(return_value=[])
    return db


class TestBackgroundRefreshService:
    """Test suite for BackgroundRefreshService."""

    @pytest.mark.asyncio
    async def test_refresh_all_sources_calls_all_connectors(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test that refresh_all_sources calls all data source refreshers."""
        refresh_service.db_manager = mock_db_manager

        # Mock all refresh methods
        with patch.object(refresh_service, '_refresh_news', return_value=True) as mock_news, \
             patch.object(refresh_service, '_refresh_weather', return_value=True) as mock_weather, \
             patch.object(refresh_service, '_refresh_email', return_value=True) as mock_email, \
             patch.object(refresh_service, '_refresh_calendar', return_value=True) as mock_calendar:

            results = await refresh_service.refresh_all_sources()

            # Verify all sources were called
            mock_news.assert_called_once()
            mock_weather.assert_called_once()
            mock_email.assert_called_once()
            mock_calendar.assert_called_once()

            # Verify results
            assert results == {
                'news': True,
                'weather': True,
                'email': True,
                'calendar': True
            }

    @pytest.mark.asyncio
    async def test_refresh_all_sources_with_progress_callback(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test that progress callback is called during refresh."""
        refresh_service.db_manager = mock_db_manager

        progress_calls = []

        def progress_callback(source, progress):
            progress_calls.append((source, progress))

        with patch.object(refresh_service, '_refresh_news', return_value=True), \
             patch.object(refresh_service, '_refresh_weather', return_value=True), \
             patch.object(refresh_service, '_refresh_email', return_value=True), \
             patch.object(refresh_service, '_refresh_calendar', return_value=True):

            await refresh_service.refresh_all_sources(progress_callback=progress_callback)

            # Verify progress callback was called for each source
            assert len(progress_calls) == 4
            assert progress_calls[0][0] == 'news'
            assert progress_calls[3][0] == 'calendar'
            assert progress_calls[3][1] == 1.0  # Last one should be 100%

    @pytest.mark.asyncio
    async def test_refresh_single_source_news(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test refreshing news source."""
        refresh_service.db_manager = mock_db_manager

        # Mock news service
        mock_article = Mock()
        mock_article.title = "Test Article"

        mock_news_service = AsyncMock()
        mock_news_service.fetch_all_news = AsyncMock(return_value=[mock_article])

        with patch('src.services.news_service.get_news_service',
                  return_value=mock_news_service):

            result = await refresh_service.refresh_single_source('news')

            assert result is True
            mock_news_service.fetch_all_news.assert_called_once()
            mock_db_manager.save_article.assert_called_once_with(mock_article)

    @pytest.mark.asyncio
    async def test_refresh_single_source_weather(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test refreshing weather source."""
        refresh_service.db_manager = mock_db_manager

        # Mock weather data
        mock_weather = Mock()
        mock_weather.temperature = 20.0

        mock_weather_service = AsyncMock()
        mock_weather_service.get_current_weather = AsyncMock(return_value=mock_weather)

        mock_config = Mock()
        mock_config.get = Mock(return_value="Seattle")

        with patch('src.services.weather_service.get_weather_service',
                  return_value=mock_weather_service), \
             patch('src.core.config_manager.get_config_manager',
                  return_value=mock_config):

            result = await refresh_service.refresh_single_source('weather')

            assert result is True
            mock_weather_service.get_current_weather.assert_called_once_with("Seattle")

    @pytest.mark.asyncio
    async def test_refresh_single_source_email(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test refreshing email source."""
        refresh_service.db_manager = mock_db_manager

        # Mock email service
        mock_email = Mock()
        mock_email.subject = "Test Email"

        mock_email_service = AsyncMock()
        mock_email_service.fetch_emails = AsyncMock(return_value=[mock_email])

        with patch('src.services.email_service.get_email_service',
                  return_value=mock_email_service):

            result = await refresh_service.refresh_single_source('email')

            assert result is True
            mock_email_service.fetch_emails.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_single_source_calendar(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test refreshing calendar source."""
        refresh_service.db_manager = mock_db_manager

        # Mock calendar service
        mock_event = Mock()
        mock_event.summary = "Test Event"

        mock_calendar_service = AsyncMock()
        mock_calendar_service.fetch_events = AsyncMock(return_value=[mock_event])

        with patch('src.services.calendar_service.get_calendar_service',
                  return_value=mock_calendar_service):

            result = await refresh_service.refresh_single_source('calendar')

            assert result is True
            mock_calendar_service.fetch_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_single_source_unknown_source(
        self,
        refresh_service
    ):
        """Test handling of unknown source."""
        result = await refresh_service.refresh_single_source('unknown_source')

        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_tracking_logs_correctly(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test that refresh operations are logged to database."""
        refresh_service.db_manager = mock_db_manager

        # Set up mock to return empty list initially
        mock_db_manager.get_setting.return_value = []

        await refresh_service._log_refresh_operation(
            source="news",
            status="success",
            details={"count": 10}
        )

        # Verify logging
        mock_db_manager.get_setting.assert_called()
        mock_db_manager.set_setting.assert_called()

        # Check the logged data
        call_args = mock_db_manager.set_setting.call_args
        logged_data = call_args[0][1]  # Second argument is the value

        assert isinstance(logged_data, list)
        assert len(logged_data) == 1
        assert logged_data[0]['status'] == 'success'
        assert logged_data[0]['details'] == {'count': 10}

    @pytest.mark.asyncio
    async def test_concurrent_refresh_prevention(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test that concurrent refreshes are prevented."""
        refresh_service.db_manager = mock_db_manager

        # Mock refresh to take some time
        async def slow_refresh(*args, **kwargs):
            await asyncio.sleep(0.1)
            return True

        with patch.object(refresh_service, '_refresh_news', side_effect=slow_refresh), \
             patch.object(refresh_service, '_refresh_weather', side_effect=slow_refresh), \
             patch.object(refresh_service, '_refresh_email', side_effect=slow_refresh), \
             patch.object(refresh_service, '_refresh_calendar', side_effect=slow_refresh):

            # Start first refresh
            task1 = asyncio.create_task(refresh_service.refresh_all_sources())

            # Wait a bit then try to start second refresh
            await asyncio.sleep(0.05)
            task2 = asyncio.create_task(refresh_service.refresh_all_sources())

            results1 = await task1
            results2 = await task2

            # First should succeed, second should be blocked
            assert len(results1) == 4  # All sources refreshed
            assert len(results2) == 0  # Blocked

    @pytest.mark.asyncio
    async def test_is_refresh_in_progress(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test is_refresh_in_progress flag."""
        refresh_service.db_manager = mock_db_manager

        # Initially not in progress
        assert refresh_service.is_refresh_in_progress() is False

        # Mock refresh to take some time
        async def slow_refresh(*args, **kwargs):
            await asyncio.sleep(0.1)
            return True

        with patch.object(refresh_service, '_refresh_news', side_effect=slow_refresh), \
             patch.object(refresh_service, '_refresh_weather', side_effect=slow_refresh), \
             patch.object(refresh_service, '_refresh_email', side_effect=slow_refresh), \
             patch.object(refresh_service, '_refresh_calendar', side_effect=slow_refresh):

            # Start refresh
            task = asyncio.create_task(refresh_service.refresh_all_sources())

            # Should be in progress now
            await asyncio.sleep(0.05)
            assert refresh_service.is_refresh_in_progress() is True

            # Wait for completion
            await task

            # Should be done now
            assert refresh_service.is_refresh_in_progress() is False

    @pytest.mark.asyncio
    async def test_get_refresh_history(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test retrieving refresh history."""
        refresh_service.db_manager = mock_db_manager

        # Mock history data
        mock_history = [
            {
                "timestamp": "2025-01-01T10:00:00",
                "status": "success",
                "details": {"count": 10}
            },
            {
                "timestamp": "2025-01-01T09:00:00",
                "status": "success",
                "details": {"count": 8}
            }
        ]

        mock_db_manager.get_setting.return_value = mock_history

        history = await refresh_service.get_refresh_history(source="news", limit=10)

        assert len(history) == 2
        assert history[0]['status'] == 'success'
        assert history[0]['details']['count'] == 10

    @pytest.mark.asyncio
    async def test_get_refresh_history_limit(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test that get_refresh_history respects limit parameter."""
        refresh_service.db_manager = mock_db_manager

        # Mock history with 10 entries
        mock_history = [{"timestamp": f"2025-01-01T{i:02d}:00:00", "status": "success"}
                       for i in range(10)]

        mock_db_manager.get_setting.return_value = mock_history

        history = await refresh_service.get_refresh_history(source="news", limit=5)

        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_refresh_all_sources_partial_failure(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test handling of partial failures during refresh."""
        refresh_service.db_manager = mock_db_manager

        # Mock some sources to succeed, some to fail
        with patch.object(refresh_service, '_refresh_news', return_value=True), \
             patch.object(refresh_service, '_refresh_weather', return_value=False), \
             patch.object(refresh_service, '_refresh_email', return_value=True), \
             patch.object(refresh_service, '_refresh_calendar', return_value=False):

            results = await refresh_service.refresh_all_sources()

            assert results['news'] is True
            assert results['weather'] is False
            assert results['email'] is True
            assert results['calendar'] is False

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that get_background_refresh_service returns singleton."""
        service1 = get_background_refresh_service()
        service2 = get_background_refresh_service()

        assert service1 is service2


class TestRefreshErrorHandling:
    """Test error handling in background refresh."""

    @pytest.mark.asyncio
    async def test_refresh_news_handles_exception(
        self,
        refresh_service,
        mock_db_manager
    ):
        """Test that _refresh_news handles exceptions gracefully."""
        refresh_service.db_manager = mock_db_manager

        mock_news_service = AsyncMock()
        mock_news_service.fetch_all_news = AsyncMock(side_effect=Exception("Network error"))

        with patch('src.services.news_service.get_news_service',
                  return_value=mock_news_service):

            result = await refresh_service._refresh_news()

            assert result is False

    @pytest.mark.asyncio
    async def test_refresh_single_source_handles_exception(
        self,
        refresh_service
    ):
        """Test that refresh_single_source handles exceptions."""
        with patch.object(refresh_service, '_refresh_news',
                         side_effect=Exception("Test error")):

            result = await refresh_service.refresh_single_source('news')

            assert result is False

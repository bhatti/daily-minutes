"""Unit tests for MCP Scheduler."""

import pytest
import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch, call
from src.services.mcp_scheduler import (
    MCPScheduler,
    get_mcp_scheduler,
    start_mcp_scheduler,
    stop_mcp_scheduler,
)


class TestMCPScheduler:
    """Test MCPScheduler class."""

    @pytest.fixture
    def mock_config(self):
        """Mock scheduler config."""
        config = MagicMock()
        config.NEWS_REFRESH_INTERVAL = 60
        config.WEATHER_REFRESH_INTERVAL = 60
        config.EMAIL_REFRESH_INTERVAL = 60
        config.CALENDAR_REFRESH_INTERVAL = 60
        config.ENABLE_BACKGROUND_SCHEDULER = True
        config.ENABLE_NEWS_SYNC = True
        config.ENABLE_WEATHER_SYNC = True
        config.ENABLE_EMAIL_SYNC = True
        config.ENABLE_CALENDAR_SYNC = True
        return config

    @pytest.fixture
    def mock_refresh_service(self):
        """Mock background refresh service."""
        service = MagicMock()
        service.refresh_single_source = AsyncMock(return_value=True)
        return service

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        db = MagicMock()
        db.get_news_cache_age_hours = AsyncMock(return_value=None)
        db.get_weather_cache_age_hours = AsyncMock(return_value=None)
        return db

    @pytest.fixture
    def scheduler(self, mock_config, mock_refresh_service, mock_db_manager):
        """Create scheduler instance with mocked dependencies."""
        with patch('src.services.mcp_scheduler.get_scheduler_config', return_value=mock_config), \
             patch('src.services.mcp_scheduler.get_background_refresh_service', return_value=mock_refresh_service), \
             patch('src.services.mcp_scheduler.get_db_manager', return_value=mock_db_manager):
            scheduler = MCPScheduler()
            yield scheduler
            # Cleanup
            if scheduler._running:
                scheduler.stop()

    def test_initialization(self, scheduler, mock_config):
        """Test scheduler initializes correctly."""
        assert scheduler.config == mock_config
        assert scheduler._running is False
        assert scheduler._scheduler_thread is None
        assert isinstance(scheduler._last_refresh, dict)
        assert len(scheduler._last_refresh) == 0

    def test_start_creates_thread(self, scheduler):
        """Test that start creates a background thread."""
        scheduler.start()

        assert scheduler._running is True
        assert scheduler._scheduler_thread is not None
        assert isinstance(scheduler._scheduler_thread, threading.Thread)
        assert scheduler._scheduler_thread.daemon is True
        assert scheduler._scheduler_thread.name == "MCPScheduler"

        # Cleanup
        scheduler.stop()

    def test_start_when_already_running(self, scheduler):
        """Test that calling start when already running does nothing."""
        scheduler.start()
        thread1 = scheduler._scheduler_thread

        # Try to start again
        scheduler.start()
        thread2 = scheduler._scheduler_thread

        # Should be the same thread
        assert thread1 is thread2

        # Cleanup
        scheduler.stop()

    def test_start_disabled_by_config(self, scheduler, mock_config):
        """Test that scheduler doesn't start if disabled in config."""
        mock_config.ENABLE_BACKGROUND_SCHEDULER = False

        scheduler.start()

        assert scheduler._running is False
        assert scheduler._scheduler_thread is None

    def test_stop(self, scheduler):
        """Test that stop terminates the background thread."""
        scheduler.start()
        assert scheduler._running is True

        scheduler.stop()

        assert scheduler._running is False
        # Thread should finish
        time.sleep(0.5)
        if scheduler._scheduler_thread:
            assert not scheduler._scheduler_thread.is_alive()

    def test_stop_when_not_running(self, scheduler):
        """Test that calling stop when not running does nothing."""
        # Should not raise an error
        scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_should_refresh_no_cache(self, scheduler, mock_db_manager):
        """Test _should_refresh returns True when no cache exists."""
        mock_db_manager.get_news_cache_age_hours = AsyncMock(return_value=None)

        should_refresh = await scheduler._should_refresh('news', 60)

        assert should_refresh is True

    @pytest.mark.asyncio
    async def test_should_refresh_stale_cache(self, scheduler, mock_db_manager):
        """Test _should_refresh returns True when cache is stale."""
        # Cache is 2 hours old, interval is 60 minutes
        mock_db_manager.get_news_cache_age_hours = AsyncMock(return_value=2.0)

        should_refresh = await scheduler._should_refresh('news', 60)

        assert should_refresh is True

    @pytest.mark.asyncio
    async def test_should_refresh_fresh_cache(self, scheduler, mock_db_manager):
        """Test _should_refresh returns False when cache is fresh."""
        # Cache is 0.5 hours old (30 minutes), interval is 60 minutes
        mock_db_manager.get_news_cache_age_hours = AsyncMock(return_value=0.5)

        should_refresh = await scheduler._should_refresh('news', 60)

        assert should_refresh is False

    @pytest.mark.asyncio
    async def test_should_refresh_email_never_refreshed(self, scheduler):
        """Test _should_refresh returns True for email when never refreshed."""
        should_refresh = await scheduler._should_refresh('email', 60)

        assert should_refresh is True

    @pytest.mark.asyncio
    async def test_should_refresh_email_interval_elapsed(self, scheduler):
        """Test _should_refresh returns True for email when interval elapsed."""
        # Set last refresh to 2 hours ago
        scheduler._last_refresh['email'] = datetime.now() - timedelta(hours=2)

        should_refresh = await scheduler._should_refresh('email', 60)  # 60 minute interval

        assert should_refresh is True

    @pytest.mark.asyncio
    async def test_should_refresh_email_interval_not_elapsed(self, scheduler):
        """Test _should_refresh returns False for email when interval not elapsed."""
        # Set last refresh to 30 minutes ago
        scheduler._last_refresh['email'] = datetime.now() - timedelta(minutes=30)

        should_refresh = await scheduler._should_refresh('email', 60)  # 60 minute interval

        assert should_refresh is False

    @pytest.mark.asyncio
    async def test_check_and_refresh_all_enabled_sources(self, scheduler, mock_refresh_service):
        """Test _check_and_refresh_all refreshes all enabled sources."""
        # Mock all sources as needing refresh
        with patch.object(scheduler, '_should_refresh', return_value=True):
            await scheduler._check_and_refresh_all()

        # Should call refresh for all 4 sources
        assert mock_refresh_service.refresh_single_source.call_count == 4
        calls = [call[0][0] for call in mock_refresh_service.refresh_single_source.call_args_list]
        assert 'news' in calls
        assert 'weather' in calls
        assert 'email' in calls
        assert 'calendar' in calls

    @pytest.mark.asyncio
    async def test_check_and_refresh_all_disabled_source(self, scheduler, mock_refresh_service, mock_config):
        """Test _check_and_refresh_all skips disabled sources."""
        mock_config.ENABLE_NEWS_SYNC = False
        mock_config.ENABLE_EMAIL_SYNC = False

        with patch.object(scheduler, '_should_refresh', return_value=True):
            await scheduler._check_and_refresh_all()

        # Should only call refresh for enabled sources (weather, calendar)
        assert mock_refresh_service.refresh_single_source.call_count == 2
        calls = [call[0][0] for call in mock_refresh_service.refresh_single_source.call_args_list]
        assert 'news' not in calls
        assert 'email' not in calls
        assert 'weather' in calls
        assert 'calendar' in calls

    @pytest.mark.asyncio
    async def test_check_and_refresh_all_updates_last_refresh(self, scheduler, mock_refresh_service):
        """Test _check_and_refresh_all updates last_refresh on success."""
        with patch.object(scheduler, '_should_refresh', return_value=True):
            before_time = datetime.now()
            await scheduler._check_and_refresh_all()
            after_time = datetime.now()

        # Should have updated last_refresh for all sources
        for source in ['news', 'weather', 'email', 'calendar']:
            assert source in scheduler._last_refresh
            refresh_time = scheduler._last_refresh[source]
            assert before_time <= refresh_time <= after_time

    @pytest.mark.asyncio
    async def test_check_and_refresh_all_handles_failure(self, scheduler, mock_refresh_service):
        """Test _check_and_refresh_all handles refresh failures gracefully."""
        # Mock one source failing
        async def mock_refresh(source):
            if source == 'news':
                return False  # Failure
            return True  # Success

        mock_refresh_service.refresh_single_source = AsyncMock(side_effect=mock_refresh)

        with patch.object(scheduler, '_should_refresh', return_value=True):
            await scheduler._check_and_refresh_all()

        # Should have tried all sources
        assert mock_refresh_service.refresh_single_source.call_count == 4

        # Failed source should not have last_refresh updated
        assert 'news' not in scheduler._last_refresh

        # Successful sources should have last_refresh updated
        assert 'weather' in scheduler._last_refresh
        assert 'email' in scheduler._last_refresh
        assert 'calendar' in scheduler._last_refresh

    @pytest.mark.asyncio
    async def test_check_and_refresh_all_handles_exception(self, scheduler, mock_refresh_service):
        """Test _check_and_refresh_all handles exceptions gracefully."""
        # Mock refresh raising exception
        mock_refresh_service.refresh_single_source = AsyncMock(
            side_effect=Exception("Network error")
        )

        with patch.object(scheduler, '_should_refresh', return_value=True):
            # Should not raise exception
            await scheduler._check_and_refresh_all()

        # Should have tried to refresh
        assert mock_refresh_service.refresh_single_source.called

    def test_get_status_not_running(self, scheduler, mock_config):
        """Test get_status returns correct info when not running."""
        status = scheduler.get_status()

        assert status['running'] is False
        assert status['enabled'] == mock_config.ENABLE_BACKGROUND_SCHEDULER
        assert isinstance(status['last_refresh'], dict)
        assert isinstance(status['intervals'], dict)
        assert status['intervals']['news'] == 60
        assert status['intervals']['weather'] == 60

    def test_get_status_running(self, scheduler):
        """Test get_status returns correct info when running."""
        scheduler.start()
        time.sleep(0.1)

        status = scheduler.get_status()

        assert status['running'] is True
        assert status['enabled'] is True

        # Cleanup
        scheduler.stop()

    def test_get_status_with_refresh_history(self, scheduler):
        """Test get_status includes refresh timestamps."""
        scheduler._last_refresh['news'] = datetime(2025, 1, 15, 10, 30, 0)
        scheduler._last_refresh['weather'] = datetime(2025, 1, 15, 11, 0, 0)

        status = scheduler.get_status()

        assert 'news' in status['last_refresh']
        assert 'weather' in status['last_refresh']
        assert status['last_refresh']['news'] == '2025-01-15T10:30:00'
        assert status['last_refresh']['weather'] == '2025-01-15T11:00:00'


class TestMCPSchedulerSingleton:
    """Test global singleton functions."""

    def teardown_method(self):
        """Reset global singleton between tests."""
        import src.services.mcp_scheduler
        if src.services.mcp_scheduler._mcp_scheduler is not None:
            if src.services.mcp_scheduler._mcp_scheduler._running:
                src.services.mcp_scheduler._mcp_scheduler.stop()
        src.services.mcp_scheduler._mcp_scheduler = None

    @patch('src.services.mcp_scheduler.get_scheduler_config')
    @patch('src.services.mcp_scheduler.get_background_refresh_service')
    @patch('src.services.mcp_scheduler.get_db_manager')
    def test_get_mcp_scheduler_returns_instance(self, mock_db, mock_refresh, mock_config):
        """Test that get_mcp_scheduler returns a MCPScheduler instance."""
        scheduler = get_mcp_scheduler()
        assert isinstance(scheduler, MCPScheduler)

    @patch('src.services.mcp_scheduler.get_scheduler_config')
    @patch('src.services.mcp_scheduler.get_background_refresh_service')
    @patch('src.services.mcp_scheduler.get_db_manager')
    def test_get_mcp_scheduler_returns_singleton(self, mock_db, mock_refresh, mock_config):
        """Test that get_mcp_scheduler returns the same instance."""
        scheduler1 = get_mcp_scheduler()
        scheduler2 = get_mcp_scheduler()

        assert scheduler1 is scheduler2

    @patch('src.services.mcp_scheduler.get_scheduler_config')
    @patch('src.services.mcp_scheduler.get_background_refresh_service')
    @patch('src.services.mcp_scheduler.get_db_manager')
    def test_start_mcp_scheduler(self, mock_db, mock_refresh, mock_config):
        """Test start_mcp_scheduler starts the scheduler."""
        mock_config.return_value.ENABLE_BACKGROUND_SCHEDULER = True

        start_mcp_scheduler()

        scheduler = get_mcp_scheduler()
        time.sleep(0.1)
        assert scheduler._running is True

        # Cleanup
        stop_mcp_scheduler()

    @patch('src.services.mcp_scheduler.get_scheduler_config')
    @patch('src.services.mcp_scheduler.get_background_refresh_service')
    @patch('src.services.mcp_scheduler.get_db_manager')
    def test_stop_mcp_scheduler(self, mock_db, mock_refresh, mock_config):
        """Test stop_mcp_scheduler stops the scheduler."""
        mock_config.return_value.ENABLE_BACKGROUND_SCHEDULER = True

        start_mcp_scheduler()
        scheduler = get_mcp_scheduler()
        time.sleep(0.1)
        assert scheduler._running is True

        stop_mcp_scheduler()
        time.sleep(0.5)
        assert scheduler._running is False


class TestMCPSchedulerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_config(self):
        """Mock scheduler config."""
        config = MagicMock()
        config.NEWS_REFRESH_INTERVAL = 60
        config.WEATHER_REFRESH_INTERVAL = 60
        config.EMAIL_REFRESH_INTERVAL = 60
        config.CALENDAR_REFRESH_INTERVAL = 60
        config.ENABLE_BACKGROUND_SCHEDULER = True
        config.ENABLE_NEWS_SYNC = True
        config.ENABLE_WEATHER_SYNC = True
        config.ENABLE_EMAIL_SYNC = True
        config.ENABLE_CALENDAR_SYNC = True
        return config

    @pytest.fixture
    def mock_refresh_service(self):
        """Mock background refresh service."""
        service = MagicMock()
        service.refresh_single_source = AsyncMock(return_value=True)
        return service

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        db = MagicMock()
        db.get_news_cache_age_hours = AsyncMock(return_value=None)
        db.get_weather_cache_age_hours = AsyncMock(return_value=None)
        return db

    @pytest.fixture
    def scheduler(self, mock_config, mock_refresh_service, mock_db_manager):
        """Create scheduler instance."""
        with patch('src.services.mcp_scheduler.get_scheduler_config', return_value=mock_config), \
             patch('src.services.mcp_scheduler.get_background_refresh_service', return_value=mock_refresh_service), \
             patch('src.services.mcp_scheduler.get_db_manager', return_value=mock_db_manager):
            scheduler = MCPScheduler()
            yield scheduler
            if scheduler._running:
                scheduler.stop()

    def test_different_intervals_per_source(self, scheduler):
        """Test that different sources can have different intervals."""
        mock_config = MagicMock()
        mock_config.NEWS_REFRESH_INTERVAL = 30
        mock_config.WEATHER_REFRESH_INTERVAL = 60
        mock_config.EMAIL_REFRESH_INTERVAL = 90
        mock_config.CALENDAR_REFRESH_INTERVAL = 120

        scheduler.config = mock_config
        status = scheduler.get_status()

        assert status['intervals']['news'] == 30
        assert status['intervals']['weather'] == 60
        assert status['intervals']['email'] == 90
        assert status['intervals']['calendar'] == 120

    @pytest.mark.asyncio
    async def test_should_refresh_error_handling(self, scheduler, mock_db_manager):
        """Test _should_refresh handles database errors gracefully."""
        mock_db_manager.get_news_cache_age_hours = AsyncMock(
            side_effect=Exception("Database error")
        )

        # Should return False on error (don't refresh if we can't determine cache age)
        should_refresh = await scheduler._should_refresh('news', 60)

        assert should_refresh is False

    def test_thread_daemon_mode(self, scheduler):
        """Test that scheduler thread is daemon (won't block program exit)."""
        scheduler.start()

        assert scheduler._scheduler_thread.daemon is True

        scheduler.stop()

    @pytest.mark.asyncio
    async def test_partial_source_refresh(self, scheduler, mock_refresh_service):
        """Test that if only some sources need refresh, only those are refreshed."""
        # News needs refresh, weather doesn't
        async def should_refresh_mock(source, interval):
            return source == 'news'

        with patch.object(scheduler, '_should_refresh', side_effect=should_refresh_mock):
            await scheduler._check_and_refresh_all()

        # Should only refresh news
        news_calls = [call for call in mock_refresh_service.refresh_single_source.call_args_list
                     if call[0][0] == 'news']
        assert len(news_calls) > 0

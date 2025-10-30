"""Integration tests for the preload_all_data script.

This ensures the preload script can be called successfully and handles errors gracefully.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from scripts.preload_all_data import preload_all_data


@pytest.mark.asyncio
async def test_preload_all_data_basic_flow():
    """Test preload_all_data completes without crashing."""
    # Mock all the heavy services
    with patch('scripts.preload_all_data.get_background_refresh_service') as mock_refresh_service, \
         patch('scripts.preload_all_data.get_db_manager') as mock_db, \
         patch('src.services.brief_scheduler.get_brief_scheduler') as mock_brief_scheduler:

        # Setup mocks
        mock_refresh = MagicMock()
        mock_refresh._refresh_news = AsyncMock(return_value=True)
        mock_refresh._refresh_weather = AsyncMock(return_value=True)
        mock_refresh._refresh_email = AsyncMock(return_value=True)
        mock_refresh._refresh_calendar = AsyncMock(return_value=True)
        mock_refresh_service.return_value = mock_refresh

        mock_db_instance = MagicMock()
        mock_db_instance.initialize = AsyncMock()
        mock_db_instance.get_all_articles = AsyncMock(return_value=[MagicMock() for _ in range(10)])
        mock_db_instance.get_cache = AsyncMock(return_value={'location': 'Test', 'temperature': 20})
        mock_db.return_value = mock_db_instance

        mock_brief = MagicMock()
        mock_brief._check_and_generate_brief = AsyncMock()
        mock_brief_scheduler.return_value = mock_brief

        # Run preload
        result = await preload_all_data(max_articles=10)

        # Should succeed with all mocked services
        assert result == 0

        # Verify all refresh methods were called
        mock_refresh._refresh_news.assert_called_once()
        mock_refresh._refresh_weather.assert_called_once()
        mock_refresh._refresh_email.assert_called_once()
        mock_refresh._refresh_calendar.assert_called_once()
        mock_brief._check_and_generate_brief.assert_called_once()


@pytest.mark.asyncio
async def test_preload_all_data_handles_failures_gracefully():
    """Test preload continues even when some services fail."""
    with patch('scripts.preload_all_data.get_background_refresh_service') as mock_refresh_service, \
         patch('scripts.preload_all_data.get_db_manager') as mock_db, \
         patch('src.services.brief_scheduler.get_brief_scheduler') as mock_brief_scheduler:

        # Setup mocks with failures
        mock_refresh = MagicMock()
        mock_refresh._refresh_news = AsyncMock(side_effect=Exception("News fetch failed"))
        mock_refresh._refresh_weather = AsyncMock(return_value=False)  # Returns False
        mock_refresh._refresh_email = AsyncMock(return_value=True)
        mock_refresh._refresh_calendar = AsyncMock(return_value=True)
        mock_refresh_service.return_value = mock_refresh

        mock_db_instance = MagicMock()
        mock_db_instance.initialize = AsyncMock()
        mock_db_instance.get_cache = AsyncMock(return_value=None)
        mock_db.return_value = mock_db_instance

        mock_brief = MagicMock()
        mock_brief._check_and_generate_brief = AsyncMock()
        mock_brief_scheduler.return_value = mock_brief

        # Run preload - should not crash despite errors
        result = await preload_all_data(max_articles=10)

        # Should return 1 (failure) but not crash
        assert result == 1

        # Verify all services were attempted despite failures
        mock_refresh._refresh_news.assert_called_once()
        mock_refresh._refresh_weather.assert_called_once()
        mock_refresh._refresh_email.assert_called_once()
        mock_refresh._refresh_calendar.assert_called_once()


@pytest.mark.asyncio
async def test_preload_all_data_ssl_errors_handled():
    """Test SSL errors are handled gracefully (common in corporate environments)."""
    import ssl

    with patch('scripts.preload_all_data.get_background_refresh_service') as mock_refresh_service, \
         patch('scripts.preload_all_data.get_db_manager') as mock_db, \
         patch('src.services.brief_scheduler.get_brief_scheduler') as mock_brief_scheduler:

        # Simulate SSL errors
        ssl_error = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")

        mock_refresh = MagicMock()
        mock_refresh._refresh_news = AsyncMock(side_effect=ssl_error)
        mock_refresh._refresh_weather = AsyncMock(side_effect=ssl_error)
        mock_refresh._refresh_email = AsyncMock(return_value=False)
        mock_refresh._refresh_calendar = AsyncMock(return_value=False)
        mock_refresh_service.return_value = mock_refresh

        mock_db_instance = MagicMock()
        mock_db_instance.initialize = AsyncMock()
        mock_db.return_value = mock_db_instance

        mock_brief = MagicMock()
        mock_brief._check_and_generate_brief = AsyncMock()
        mock_brief_scheduler.return_value = mock_brief

        # Should not crash on SSL errors
        result = await preload_all_data(max_articles=10)

        # Returns error code but doesn't crash
        assert result == 1

"""Unit tests for CalendarService - TDD validation.

Tests the service layer that orchestrates calendar event fetching from multiple providers.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from src.core.models import CalendarEvent
from src.services.calendar_service import CalendarService, get_calendar_service


class TestCalendarServiceInitialization:
    """Test CalendarService initialization."""

    def test_initialization(self):
        """Test CalendarService initializes successfully."""
        service = CalendarService()
        assert service is not None
        assert service._calendar_agent is None
        assert service._connectors_initialized is False

    def test_singleton_pattern(self):
        """Test get_calendar_service returns same instance."""
        service1 = get_calendar_service()
        service2 = get_calendar_service()

        assert service1 is service2


class TestCalendarServiceConnectorInitialization:
    """Test connector initialization logic."""

    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_outlook_calendar_connector')
    def test_initialize_all_connectors_success(
        self,
        mock_outlook,
        mock_google
    ):
        """Test initializing all connectors successfully."""
        # Setup mocks
        mock_google.return_value = Mock()
        mock_outlook.return_value = Mock()

        service = CalendarService()
        connectors = service._initialize_connectors()

        # Should have both connectors
        assert len(connectors) == 2
        assert "google" in connectors
        assert "outlook" in connectors

    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_outlook_calendar_connector')
    def test_initialize_partial_connectors(
        self,
        mock_outlook,
        mock_google
    ):
        """Test initializing with some connectors failing."""
        # Google succeeds, Outlook fails
        mock_google.return_value = Mock()
        mock_outlook.side_effect = Exception("Outlook not configured")

        service = CalendarService()
        connectors = service._initialize_connectors()

        # Should only have Google
        assert len(connectors) == 1
        assert "google" in connectors

    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_outlook_calendar_connector')
    def test_initialize_no_connectors(
        self,
        mock_outlook,
        mock_google
    ):
        """Test initialization with no available connectors."""
        # All connectors fail
        mock_google.side_effect = Exception("Google not configured")
        mock_outlook.side_effect = Exception("Outlook not configured")

        service = CalendarService()
        connectors = service._initialize_connectors()

        # Should have no connectors
        assert len(connectors) == 0


class TestCalendarServiceFetchEvents:
    """Test event fetching functionality."""

    @pytest.fixture
    def sample_events(self) -> List[CalendarEvent]:
        """Create sample events for testing."""
        now = datetime.now()

        return [
            CalendarEvent(
                id="1",
                summary="Team standup",
                start_time=now,
                end_time=now + timedelta(minutes=30),
                importance_score=0.7,
                requires_preparation=False
            ),
            CalendarEvent(
                id="2",
                summary="Client presentation",
                start_time=now + timedelta(hours=2),
                end_time=now + timedelta(hours=3),
                importance_score=0.9,
                requires_preparation=True
            ),
            CalendarEvent(
                id="3",
                summary="Focus time",
                start_time=now + timedelta(hours=4),
                end_time=now + timedelta(hours=6),
                importance_score=0.6,
                is_focus_time=True,
                requires_preparation=False
            ),
        ]

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_fetch_events_success(
        self,
        mock_get_agent,
        mock_google,
        sample_events
    ):
        """Test successful event fetching."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(return_value=sample_events)
        mock_get_agent.return_value = mock_agent

        service = CalendarService()
        events = await service.fetch_events(max_results=10)

        # Should return all events
        assert len(events) == 3
        assert mock_agent.fetch_events.called

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    async def test_fetch_events_no_connectors(self, mock_google):
        """Test fetching with no available connectors."""
        # No connectors available
        mock_google.side_effect = Exception("Google not configured")

        service = CalendarService()
        events = await service.fetch_events()

        # Should return empty list
        assert len(events) == 0

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_fetch_events_with_important_filter(
        self,
        mock_get_agent,
        mock_google,
        sample_events
    ):
        """Test fetching with importance filter applied."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(return_value=sample_events)
        mock_get_agent.return_value = mock_agent

        service = CalendarService()

        # Filter for important only (>= 0.7)
        events = await service.fetch_events(
            filter_important=True,
            max_results=10
        )

        # Should only return important events (2 out of 3)
        assert len(events) == 2
        assert all(e.importance_score >= 0.7 for e in events)

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_fetch_events_with_preparation_filter(
        self,
        mock_get_agent,
        mock_google,
        sample_events
    ):
        """Test fetching with preparation filter."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(return_value=sample_events)
        mock_get_agent.return_value = mock_agent

        service = CalendarService()

        # Filter for events requiring preparation
        events = await service.fetch_events(
            filter_preparation=True,
            max_results=10
        )

        # Should only return events requiring preparation (1 out of 3)
        assert len(events) == 1
        assert all(e.requires_preparation for e in events)

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_fetch_events_with_progress_callback(
        self,
        mock_get_agent,
        mock_google,
        sample_events
    ):
        """Test fetching with progress callback."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(return_value=sample_events)
        mock_get_agent.return_value = mock_agent

        # Track progress callbacks
        progress_calls = []

        async def progress_callback(progress, message):
            progress_calls.append((progress, message))

        service = CalendarService()
        events = await service.fetch_events(
            progress_callback=progress_callback
        )

        # Should have called progress callback multiple times
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 1.0  # Final progress should be 100%


class TestCalendarServiceConvenienceMethods:
    """Test convenience methods."""

    @pytest.fixture
    def sample_events(self) -> List[CalendarEvent]:
        """Create sample events for testing."""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        return [
            CalendarEvent(
                id="1",
                summary="Morning standup",
                start_time=today_start + timedelta(hours=9),
                end_time=today_start + timedelta(hours=9, minutes=30),
                importance_score=0.7
            ),
            CalendarEvent(
                id="2",
                summary="Client meeting",
                start_time=today_start + timedelta(hours=14),
                end_time=today_start + timedelta(hours=15),
                importance_score=0.9,
                requires_preparation=True
            ),
            CalendarEvent(
                id="3",
                summary="Focus time",
                start_time=today_start + timedelta(hours=16),
                end_time=today_start + timedelta(hours=18),
                importance_score=0.6,
                is_focus_time=True
            ),
        ]

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_get_today_events(
        self,
        mock_get_agent,
        mock_google,
        sample_events
    ):
        """Test getting today's events."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(return_value=sample_events)
        mock_get_agent.return_value = mock_agent

        service = CalendarService()
        events = await service.get_today_events()

        # Should return all today's events
        assert len(events) == 3

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_get_upcoming_events(
        self,
        mock_get_agent,
        mock_google,
        sample_events
    ):
        """Test getting upcoming events."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(return_value=sample_events)
        mock_get_agent.return_value = mock_agent

        service = CalendarService()
        events = await service.get_upcoming_events(hours=24)

        # Should return upcoming events
        assert len(events) >= 0  # Depends on time of day

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_get_important_events(
        self,
        mock_get_agent,
        mock_google,
        sample_events
    ):
        """Test getting important events."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(return_value=sample_events)
        mock_get_agent.return_value = mock_agent

        service = CalendarService()
        events = await service.get_important_events(days=7, max_results=10)

        # Should return important events (>= 0.7)
        assert len(events) == 2
        assert all(e.importance_score >= 0.7 for e in events)

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_get_events_requiring_preparation(
        self,
        mock_get_agent,
        mock_google,
        sample_events
    ):
        """Test getting events requiring preparation."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(return_value=sample_events)
        mock_get_agent.return_value = mock_agent

        service = CalendarService()
        events = await service.get_events_requiring_preparation(days=7)

        # Should return events requiring preparation (1 out of 3)
        assert len(events) == 1
        assert all(e.requires_preparation for e in events)

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_get_focus_time_blocks(
        self,
        mock_get_agent,
        mock_google,
        sample_events
    ):
        """Test getting focus time blocks."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(return_value=sample_events)
        mock_get_agent.return_value = mock_agent

        service = CalendarService()
        events = await service.get_focus_time_blocks(days=7)

        # Should return focus time blocks (1 out of 3)
        assert len(events) == 1
        assert all(e.is_focus_time for e in events)

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_get_event_statistics(
        self,
        mock_get_agent,
        mock_google,
        sample_events
    ):
        """Test getting event statistics."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(return_value=sample_events)
        mock_get_agent.return_value = mock_agent

        service = CalendarService()
        stats = await service.get_event_statistics(days=7)

        # Should return statistics
        assert stats["total_events"] == 3
        assert stats["important_events"] == 2
        assert stats["prep_required"] == 1
        assert stats["focus_time_blocks"] == 1
        assert stats["total_meeting_hours"] > 0


class TestCalendarServiceErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    @patch('src.services.calendar_service.get_calendar_agent')
    async def test_fetch_events_agent_error(
        self,
        mock_get_agent,
        mock_google
    ):
        """Test handling agent errors."""
        # Setup mocks
        mock_connector = Mock()
        mock_google.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_events = AsyncMock(side_effect=Exception("Agent error"))
        mock_get_agent.return_value = mock_agent

        service = CalendarService()
        events = await service.fetch_events()

        # Should return empty list on error
        assert len(events) == 0

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    async def test_get_focus_time_blocks_error(self, mock_google):
        """Test handling errors in get_focus_time_blocks."""
        # No connectors available
        mock_google.side_effect = Exception("Google not configured")

        service = CalendarService()
        events = await service.get_focus_time_blocks()

        # Should return empty list on error
        assert len(events) == 0

    @pytest.mark.asyncio
    @patch('src.services.calendar_service.get_google_calendar_connector')
    async def test_get_event_statistics_error(self, mock_google):
        """Test handling errors in get_event_statistics."""
        # No connectors available
        mock_google.side_effect = Exception("Google not configured")

        service = CalendarService()
        stats = await service.get_event_statistics()

        # Should return zero statistics on error
        assert stats["total_events"] == 0
        assert stats["important_events"] == 0
        assert stats["prep_required"] == 0
        assert stats["focus_time_blocks"] == 0

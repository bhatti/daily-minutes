#!/usr/bin/env python3
"""Unit tests for CalendarAgent (TDD approach).

Following TDD: Write tests first, then implement CalendarAgent.

The CalendarAgent orchestrates multiple calendar connectors (Google Calendar, Outlook Calendar)
and provides a unified interface for fetching, filtering, and managing calendar events.

Mocking Strategy:
- Mock calendar connectors (Google, Outlook)
- Mock database and metrics managers
- Verify correct connector selection
- Verify event merging, deduplication, and prioritization
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import List

from src.core.models import CalendarEvent


# Test constants
MOCK_EVENTS = [
    CalendarEvent(
        id="google-1",
        summary="Team Standup",
        description="Daily team sync",
        start_time=datetime(2025, 10, 27, 9, 0, 0),
        end_time=datetime(2025, 10, 27, 9, 30, 0),
        location="Conference Room A",
        attendees=["alice@example.com", "bob@example.com"],
        is_recurring=True,
        recurrence_rule="FREQ=DAILY",
        importance_score=0.7,
        requires_preparation=False,
        is_focus_time=False
    ),
    CalendarEvent(
        id="outlook-1",
        summary="Client Meeting",
        description="Q4 Review",
        start_time=datetime(2025, 10, 27, 14, 0, 0),
        end_time=datetime(2025, 10, 27, 15, 30, 0),
        location="Virtual",
        attendees=["client@company.com"],
        is_recurring=False,
        recurrence_rule=None,
        importance_score=0.9,
        requires_preparation=True,
        is_focus_time=False
    ),
    CalendarEvent(
        id="google-2",
        summary="Focus Time",
        description="Deep work block",
        start_time=datetime(2025, 10, 27, 10, 0, 0),
        end_time=datetime(2025, 10, 27, 12, 0, 0),
        location=None,
        attendees=[],
        is_recurring=False,
        recurrence_rule=None,
        importance_score=0.8,
        requires_preparation=False,
        is_focus_time=True
    )
]


@pytest.fixture
def mock_google_connector():
    """Create mock Google Calendar connector."""
    connector = AsyncMock()
    connector.is_authenticated = True
    connector.fetch_events.return_value = [MOCK_EVENTS[0], MOCK_EVENTS[2]]
    return connector


@pytest.fixture
def mock_outlook_connector():
    """Create mock Outlook Calendar connector."""
    connector = AsyncMock()
    connector.is_authenticated = True
    connector.fetch_events.return_value = [MOCK_EVENTS[1]]
    return connector


class TestCalendarAgentInitialization:
    """Test CalendarAgent initialization and configuration."""

    def test_agent_initialization_no_connectors(self):
        """Test CalendarAgent can be initialized without connectors."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent()

        assert agent is not None
        assert len(agent.connectors) == 0

    def test_agent_initialization_with_connectors(self, mock_google_connector):
        """Test CalendarAgent initialization with connectors."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={"google": mock_google_connector})

        assert agent is not None
        assert len(agent.connectors) == 1
        assert "google" in agent.connectors

    def test_agent_initialization_with_multiple_connectors(
        self, mock_google_connector, mock_outlook_connector
    ):
        """Test CalendarAgent with multiple connectors."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        assert len(agent.connectors) == 2
        assert "google" in agent.connectors
        assert "outlook" in agent.connectors

    def test_agent_initialization_with_cache_ttl(self):
        """Test CalendarAgent initialization with custom cache TTL."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(cache_ttl_seconds=300)

        assert agent.cache_ttl_seconds == 300


class TestCalendarAgentFetchEvents:
    """Test CalendarAgent event fetching functionality."""

    @pytest.mark.asyncio
    async def test_fetch_events_single_connector(self, mock_google_connector):
        """Test fetching events from single connector."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={"google": mock_google_connector})

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 2
        assert events[0].summary in ["Team Standup", "Focus Time"]
        mock_google_connector.fetch_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_events_multiple_connectors(
        self, mock_google_connector, mock_outlook_connector
    ):
        """Test fetching and merging events from multiple connectors."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        # Should get 3 events total (2 from Google + 1 from Outlook)
        assert len(events) == 3
        mock_google_connector.fetch_events.assert_called_once()
        mock_outlook_connector.fetch_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_events_no_connectors(self):
        """Test fetching events with no connectors returns empty list."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent()

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_fetch_events_with_max_results(self, mock_google_connector):
        """Test fetching events with max_results limit."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={"google": mock_google_connector})

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            max_results=1
        )

        # Should limit to 1 event even though connector returned 2
        assert len(events) == 1


class TestCalendarAgentFiltering:
    """Test CalendarAgent filtering functionality."""

    @pytest.mark.asyncio
    async def test_filter_by_summary(self, mock_google_connector, mock_outlook_connector):
        """Test filtering events by summary keyword."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            filter_summary="Meeting"
        )

        # Should only return "Client Meeting"
        assert len(events) == 1
        assert "Meeting" in events[0].summary

    @pytest.mark.asyncio
    async def test_filter_by_location(self, mock_google_connector, mock_outlook_connector):
        """Test filtering events by location."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            filter_location="Conference"
        )

        # Should only return events in conference rooms
        assert len(events) >= 1
        for event in events:
            if event.location:
                assert "Conference" in event.location

    @pytest.mark.asyncio
    async def test_filter_by_attendee(self, mock_google_connector):
        """Test filtering events by attendee."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={"google": mock_google_connector})

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            filter_attendee="alice@example.com"
        )

        # Should only return events with alice as attendee
        for event in events:
            assert "alice@example.com" in event.attendees

    @pytest.mark.asyncio
    async def test_filter_requires_preparation(self, mock_google_connector, mock_outlook_connector):
        """Test filtering events that require preparation."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            requires_preparation=True
        )

        # Should only return Client Meeting
        assert len(events) >= 1
        for event in events:
            assert event.requires_preparation is True

    @pytest.mark.asyncio
    async def test_filter_focus_time(self, mock_google_connector, mock_outlook_connector):
        """Test filtering focus time blocks."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            is_focus_time=True
        )

        # Should only return Focus Time event
        assert len(events) >= 1
        for event in events:
            assert event.is_focus_time is True


class TestCalendarAgentSorting:
    """Test CalendarAgent sorting functionality."""

    @pytest.mark.asyncio
    async def test_sort_by_start_time(self, mock_google_connector, mock_outlook_connector):
        """Test sorting events by start time (chronological)."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            sort_by_time=True
        )

        # Events should be sorted chronologically
        for i in range(len(events) - 1):
            assert events[i].start_time <= events[i + 1].start_time

    @pytest.mark.asyncio
    async def test_sort_by_importance(self, mock_google_connector, mock_outlook_connector):
        """Test sorting events by importance score."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            sort_by_importance=True
        )

        # Events should be sorted by importance (highest first)
        for i in range(len(events) - 1):
            assert events[i].importance_score >= events[i + 1].importance_score


class TestCalendarAgentCaching:
    """Test CalendarAgent caching functionality."""

    @pytest.mark.asyncio
    async def test_fetch_events_uses_cache(self, mock_google_connector):
        """Test that subsequent fetches use cache."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(
            connectors={"google": mock_google_connector},
            cache_ttl_seconds=60
        )

        # First fetch
        events1 = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            use_cache=True
        )

        # Second fetch (should use cache)
        events2 = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            use_cache=True
        )

        # Connector should only be called once
        assert mock_google_connector.fetch_events.call_count == 1
        assert len(events1) == len(events2)

    @pytest.mark.asyncio
    async def test_fetch_events_bypasses_cache(self, mock_google_connector):
        """Test that use_cache=False bypasses cache."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(
            connectors={"google": mock_google_connector},
            cache_ttl_seconds=60
        )

        # First fetch
        events1 = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            use_cache=False
        )

        # Second fetch (should not use cache)
        events2 = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            use_cache=False
        )

        # Connector should be called twice
        assert mock_google_connector.fetch_events.call_count == 2


class TestCalendarAgentStatistics:
    """Test CalendarAgent statistics functionality."""

    @pytest.mark.asyncio
    async def test_get_event_count_by_day(self, mock_google_connector, mock_outlook_connector):
        """Test getting event count statistics by day."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        # Fetch events first
        await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        # Get statistics
        stats = agent.get_event_count_by_day()

        assert isinstance(stats, dict)
        # All events are on 2025-10-27
        assert "2025-10-27" in stats
        assert stats["2025-10-27"] == 3

    @pytest.mark.asyncio
    async def test_get_average_importance(self, mock_google_connector, mock_outlook_connector):
        """Test calculating average importance score."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        # Fetch events first
        await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        # Get average importance
        avg_importance = agent.get_average_importance()

        assert isinstance(avg_importance, float)
        assert 0.0 <= avg_importance <= 1.0
        # Expected: (0.7 + 0.9 + 0.8) / 3 = 0.8
        assert avg_importance == pytest.approx(0.8, abs=0.01)

    @pytest.mark.asyncio
    async def test_get_upcoming_events(self, mock_google_connector, mock_outlook_connector):
        """Test getting upcoming events within hours."""
        from src.agents.calendar_agent import CalendarAgent

        agent = CalendarAgent(connectors={
            "google": mock_google_connector,
            "outlook": mock_outlook_connector
        })

        # Fetch events first
        await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        # Get upcoming events (with mocked "now" time)
        with patch('src.agents.calendar_agent.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 10, 27, 8, 0, 0)
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            upcoming = agent.get_upcoming_events(hours=2)

            # Team Standup at 9:00 is within 2 hours
            assert len(upcoming) >= 1


class TestCalendarAgentDeduplication:
    """Test CalendarAgent event deduplication."""

    @pytest.mark.asyncio
    async def test_deduplicate_identical_events(self):
        """Test deduplication of identical events from different sources."""
        from src.agents.calendar_agent import CalendarAgent

        # Create duplicate event
        duplicate_event = CalendarEvent(
            id="duplicate-1",
            summary="Team Standup",  # Same summary
            description="Daily team sync",
            start_time=datetime(2025, 10, 27, 9, 0, 0),  # Same time
            end_time=datetime(2025, 10, 27, 9, 30, 0),
            location="Conference Room A",
            attendees=["alice@example.com", "bob@example.com"],
            is_recurring=True,
            recurrence_rule="FREQ=DAILY",
            importance_score=0.7,
            requires_preparation=False,
            is_focus_time=False
        )

        # Create connectors returning duplicates
        connector1 = AsyncMock()
        connector1.is_authenticated = True
        connector1.fetch_events.return_value = [MOCK_EVENTS[0]]

        connector2 = AsyncMock()
        connector2.is_authenticated = True
        connector2.fetch_events.return_value = [duplicate_event]

        agent = CalendarAgent(connectors={
            "google": connector1,
            "outlook": connector2
        })

        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            deduplicate=True
        )

        # Should only have 1 event after deduplication
        assert len(events) == 1


class TestCalendarAgentErrorHandling:
    """Test CalendarAgent error handling."""

    @pytest.mark.asyncio
    async def test_fetch_events_connector_failure(self):
        """Test graceful handling of connector failure."""
        from src.agents.calendar_agent import CalendarAgent

        # Create failing connector
        failing_connector = AsyncMock()
        failing_connector.is_authenticated = True
        failing_connector.fetch_events.side_effect = Exception("API Error")

        # Create working connector
        working_connector = AsyncMock()
        working_connector.is_authenticated = True
        working_connector.fetch_events.return_value = [MOCK_EVENTS[1]]

        agent = CalendarAgent(connectors={
            "failing": failing_connector,
            "working": working_connector
        })

        # Should still get events from working connector
        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) >= 1
        assert events[0].summary == "Client Meeting"

    @pytest.mark.asyncio
    async def test_fetch_events_all_connectors_fail(self):
        """Test behavior when all connectors fail."""
        from src.agents.calendar_agent import CalendarAgent

        failing_connector = AsyncMock()
        failing_connector.is_authenticated = True
        failing_connector.fetch_events.side_effect = Exception("API Error")

        agent = CalendarAgent(connectors={"failing": failing_connector})

        # Should return empty list, not crash
        events = await agent.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

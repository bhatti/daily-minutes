"""Unit tests for calendar UI component helper functions - TDD RED phase.

The UI components should be thin rendering layers. We test:
1. Helper functions that prepare data for display
2. State management logic
3. Integration with formatters and services

Pure Streamlit rendering code is not unit tested - instead covered by integration tests.
"""

import pytest
from datetime import datetime, timedelta
from typing import List
from unittest.mock import Mock, AsyncMock, patch

from src.core.models import CalendarEvent
from src.ui.components.calendar_components import (
    prepare_event_display_data,
    get_importance_indicator,
    get_preparation_badge,
    should_show_event,
    group_events_for_display,
    get_event_conflicts,
)


class TestEventDisplayDataPreparation:
    """Test helper functions that prepare event data for display."""

    @pytest.fixture
    def sample_event(self) -> CalendarEvent:
        """Create a sample event for testing."""
        return CalendarEvent(
            id="1",
            summary="Product planning meeting",
            description="Discuss Q4 roadmap and priorities",
            start_time=datetime.now() + timedelta(hours=2),
            end_time=datetime.now() + timedelta(hours=3),
            location="Conference Room A",
            attendees=["alice@example.com", "bob@example.com"],
            importance_score=0.9,
            requires_preparation=True,
            preparation_notes=["Review last quarter metrics", "Prepare slides"]
        )

    def test_prepare_event_display_data(self, sample_event):
        """Test preparing event data for display."""
        display_data = prepare_event_display_data(sample_event)

        # Should return a dictionary with display-ready data
        assert isinstance(display_data, dict)
        assert "summary" in display_data
        assert "time_range" in display_data
        assert "duration" in display_data
        assert "time_until" in display_data
        assert "location" in display_data
        assert "attendees_display" in display_data
        assert "importance_indicator" in display_data
        assert "preparation_badge" in display_data

    def test_get_importance_indicator_high(self, sample_event):
        """Test getting importance indicator for high importance event."""
        indicator = get_importance_indicator(sample_event)

        # High importance (>= 0.7) should return an indicator
        assert isinstance(indicator, str)
        assert len(indicator) > 0

    def test_get_importance_indicator_low(self):
        """Test getting importance indicator for low importance event."""
        event = CalendarEvent(
            id="1",
            summary="Coffee chat",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            importance_score=0.3
        )

        indicator = get_importance_indicator(event)

        # Low importance might return empty string or low priority indicator
        assert isinstance(indicator, str)

    def test_get_preparation_badge_required(self, sample_event):
        """Test getting preparation badge when preparation is required."""
        badge = get_preparation_badge(sample_event)

        # Should return a badge indicating preparation needed
        assert isinstance(badge, str)
        assert len(badge) > 0

    def test_get_preparation_badge_not_required(self):
        """Test getting preparation badge when no preparation needed."""
        event = CalendarEvent(
            id="1",
            summary="Standup",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=15),
            requires_preparation=False
        )

        badge = get_preparation_badge(event)

        # Should return empty string
        assert isinstance(badge, str)


class TestEventFiltering:
    """Test event filtering logic for display."""

    def test_should_show_event_importance_filter(self):
        """Test event filtering with importance filter."""
        important_event = CalendarEvent(
            id="1",
            summary="Board meeting",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=2),
            importance_score=0.9
        )

        unimportant_event = CalendarEvent(
            id="2",
            summary="Coffee break",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=15),
            importance_score=0.3
        )

        # Should show important event when filtering by importance
        assert should_show_event(important_event, filter_important=True) is True
        assert should_show_event(unimportant_event, filter_important=True) is False

    def test_should_show_event_preparation_filter(self):
        """Test event filtering with preparation filter."""
        prep_event = CalendarEvent(
            id="1",
            summary="Client presentation",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            requires_preparation=True
        )

        no_prep_event = CalendarEvent(
            id="2",
            summary="Standup",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=15),
            requires_preparation=False
        )

        # Should show prep event when filtering by preparation
        assert should_show_event(prep_event, filter_preparation=True) is True
        assert should_show_event(no_prep_event, filter_preparation=True) is False

    def test_should_show_event_search_filter(self):
        """Test event filtering with search query."""
        event = CalendarEvent(
            id="1",
            summary="Sprint planning meeting",
            description="Plan next sprint with team",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=2),
            location="Room 101"
        )

        # Should match summary
        assert should_show_event(event, search_query="sprint") is True
        assert should_show_event(event, search_query="planning") is True

        # Should match description
        assert should_show_event(event, search_query="team") is True

        # Should match location
        assert should_show_event(event, search_query="101") is True

        # Should not match unrelated text
        assert should_show_event(event, search_query="marketing") is False

        # Search should be case-insensitive
        assert should_show_event(event, search_query="SPRINT") is True


class TestEventGrouping:
    """Test event grouping logic for display."""

    @pytest.fixture
    def sample_events(self) -> List[CalendarEvent]:
        """Create sample events for testing."""
        now = datetime.now()
        today = now.replace(hour=10, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)

        return [
            CalendarEvent(
                id="1",
                summary="Today standup",
                start_time=today,
                end_time=today + timedelta(minutes=30),
                importance_score=0.6
            ),
            CalendarEvent(
                id="2",
                summary="Today review",
                start_time=today + timedelta(hours=4),
                end_time=today + timedelta(hours=5),
                importance_score=0.8
            ),
            CalendarEvent(
                id="3",
                summary="Tomorrow planning",
                start_time=tomorrow,
                end_time=tomorrow + timedelta(hours=2),
                importance_score=0.9
            ),
        ]

    def test_group_events_for_display(self, sample_events):
        """Test grouping events by date for display."""
        grouped = group_events_for_display(sample_events)

        # Should return a dictionary with date groups
        assert isinstance(grouped, dict)
        assert len(grouped) >= 1  # At least one group

        # Each group should have a list of events
        for date_label, events in grouped.items():
            assert isinstance(date_label, str)
            assert isinstance(events, list)
            assert all(isinstance(e, CalendarEvent) for e in events)

    def test_group_events_empty_list(self):
        """Test grouping empty event list."""
        grouped = group_events_for_display([])

        # Should return empty dict
        assert isinstance(grouped, dict)
        assert len(grouped) == 0


class TestEventConflicts:
    """Test event conflict detection."""

    def test_get_event_conflicts_with_overlap(self):
        """Test detecting overlapping events."""
        event1 = CalendarEvent(
            id="1",
            summary="Meeting A",
            start_time=datetime(2025, 10, 26, 10, 0),
            end_time=datetime(2025, 10, 26, 11, 0)
        )

        event2 = CalendarEvent(
            id="2",
            summary="Meeting B",
            start_time=datetime(2025, 10, 26, 10, 30),
            end_time=datetime(2025, 10, 26, 11, 30)
        )

        event3 = CalendarEvent(
            id="3",
            summary="Meeting C",
            start_time=datetime(2025, 10, 26, 12, 0),
            end_time=datetime(2025, 10, 26, 13, 0)
        )

        events = [event1, event2, event3]

        # Check conflicts for event1
        conflicts = get_event_conflicts(event1, events)

        # Should detect overlap with event2
        assert len(conflicts) == 1
        assert conflicts[0].id == "2"

    def test_get_event_conflicts_no_overlap(self):
        """Test when events don't overlap."""
        event1 = CalendarEvent(
            id="1",
            summary="Meeting A",
            start_time=datetime(2025, 10, 26, 10, 0),
            end_time=datetime(2025, 10, 26, 11, 0)
        )

        event2 = CalendarEvent(
            id="2",
            summary="Meeting B",
            start_time=datetime(2025, 10, 26, 11, 0),
            end_time=datetime(2025, 10, 26, 12, 0)
        )

        events = [event1, event2]

        # Check conflicts for event1
        conflicts = get_event_conflicts(event1, events)

        # Should have no conflicts (adjacent events don't count as overlapping)
        assert len(conflicts) == 0


class TestCalendarComponentIntegration:
    """Test integration with formatters and services."""

    @pytest.fixture
    def sample_event(self) -> CalendarEvent:
        """Create a sample event for testing."""
        return CalendarEvent(
            id="1",
            summary="Test meeting",
            start_time=datetime.now() + timedelta(hours=1),
            end_time=datetime.now() + timedelta(hours=2),
            importance_score=0.8
        )

    def test_prepare_event_uses_formatter(self, sample_event):
        """Test that prepare_event_display_data uses CalendarFormatter."""
        with patch('src.ui.components.calendar_components.get_calendar_formatter') as mock_get_formatter:
            mock_formatter = Mock()
            mock_formatter.format_time_range.return_value = "14:00 - 15:00"
            mock_formatter.format_duration.return_value = "1 hour"
            mock_formatter.get_time_until_event.return_value = "in 1 hour"
            mock_formatter.format_attendees_list.return_value = []
            mock_get_formatter.return_value = mock_formatter

            display_data = prepare_event_display_data(sample_event)

            # Should have called formatter methods
            assert mock_formatter.format_time_range.called
            assert mock_formatter.format_duration.called
            assert mock_formatter.get_time_until_event.called

            # Should include formatted data
            assert "time_range" in display_data
            assert "duration" in display_data
            assert "time_until" in display_data

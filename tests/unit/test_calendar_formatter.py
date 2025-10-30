"""Unit tests for CalendarFormatter - TDD Red Phase.

The CalendarFormatter is responsible for business logic related to formatting
calendar event data for display. It should NOT contain any UI rendering code.

Business logic includes:
- Grouping events by date, type, or importance
- Calculating statistics (total events, meeting time, etc.)
- Formatting dates/times for display
- Extracting display-friendly summaries
- Priority calculations for display ordering
- Duration and time block calculations
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from src.core.models import CalendarEvent
from src.ui.formatters.calendar_formatter import CalendarFormatter


class TestCalendarFormatterInitialization:
    """Test CalendarFormatter initialization."""

    def test_initialization(self):
        """Test CalendarFormatter initializes successfully."""
        formatter = CalendarFormatter()
        assert formatter is not None

    def test_singleton_pattern(self):
        """Test get_calendar_formatter returns same instance."""
        from src.ui.formatters.calendar_formatter import get_calendar_formatter

        formatter1 = get_calendar_formatter()
        formatter2 = get_calendar_formatter()

        assert formatter1 is formatter2


class TestCalendarFormatterGrouping:
    """Test calendar event grouping logic."""

    @pytest.fixture
    def sample_events(self) -> List[CalendarEvent]:
        """Create sample calendar events for testing."""
        now = datetime.now()
        today = now.replace(hour=10, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)

        return [
            CalendarEvent(
                id="1",
                summary="Today's standup",
                start_time=today,
                end_time=today + timedelta(minutes=30),
                importance_score=0.7,
                requires_preparation=False
            ),
            CalendarEvent(
                id="2",
                summary="Tomorrow's planning",
                start_time=tomorrow,
                end_time=tomorrow + timedelta(hours=1),
                importance_score=0.9,
                requires_preparation=True
            ),
            CalendarEvent(
                id="3",
                summary="Today's review",
                start_time=today + timedelta(hours=2),
                end_time=today + timedelta(hours=3),
                importance_score=0.5,
                requires_preparation=False
            ),
        ]

    def test_group_by_date(self, sample_events):
        """Test grouping events by date."""
        formatter = CalendarFormatter()

        grouped = formatter.group_by_date(sample_events)

        # Should have 2 date groups (today and tomorrow)
        assert len(grouped) == 2

        # Check group keys are date strings
        for date_str, events in grouped.items():
            assert isinstance(date_str, str)
            assert isinstance(events, list)
            assert all(isinstance(e, CalendarEvent) for e in events)

    def test_group_by_importance(self, sample_events):
        """Test grouping events by importance level."""
        formatter = CalendarFormatter()

        grouped = formatter.group_by_importance(sample_events)

        # Should have importance levels: high, medium, low
        assert "high" in grouped or "medium" in grouped or "low" in grouped

        # High importance should include event with 0.9 score
        if "high" in grouped:
            assert len(grouped["high"]) >= 1

    def test_group_by_preparation_needed(self, sample_events):
        """Test grouping events by preparation requirement."""
        formatter = CalendarFormatter()

        grouped = formatter.group_by_preparation_needed(sample_events)

        # Should have two groups: requires_prep and no_prep
        assert "requires_prep" in grouped or "no_prep" in grouped

        # Should have 1 event requiring prep
        if "requires_prep" in grouped:
            assert len(grouped["requires_prep"]) == 1


class TestCalendarFormatterStatistics:
    """Test calendar statistics calculation."""

    @pytest.fixture
    def sample_events(self) -> List[CalendarEvent]:
        """Create sample events for testing."""
        now = datetime.now()
        today = now.replace(hour=10, minute=0, second=0, microsecond=0)

        return [
            CalendarEvent(
                id="1",
                summary="Meeting 1",
                start_time=today,
                end_time=today + timedelta(hours=1),
                importance_score=0.9,
                requires_preparation=True
            ),
            CalendarEvent(
                id="2",
                summary="Meeting 2",
                start_time=today + timedelta(hours=2),
                end_time=today + timedelta(hours=2, minutes=30),
                importance_score=0.5,
                requires_preparation=False,
                is_focus_time=True
            ),
            CalendarEvent(
                id="3",
                summary="Meeting 3",
                start_time=today + timedelta(hours=4),
                end_time=today + timedelta(hours=5),
                importance_score=0.7,
                requires_preparation=True
            ),
        ]

    def test_calculate_statistics(self, sample_events):
        """Test calculating event statistics."""
        formatter = CalendarFormatter()

        stats = formatter.calculate_statistics(sample_events)

        assert stats["total"] == 3
        assert stats["requires_preparation"] == 2
        assert stats["focus_time"] == 1
        assert stats["high_importance"] >= 1

    def test_get_total_meeting_time(self, sample_events):
        """Test calculating total meeting time in hours."""
        formatter = CalendarFormatter()

        total_hours = formatter.get_total_meeting_time(sample_events)

        # 1 hour + 0.5 hours + 1 hour = 2.5 hours
        assert total_hours == 2.5

    def test_get_preparation_count(self, sample_events):
        """Test counting events requiring preparation."""
        formatter = CalendarFormatter()

        prep_count = formatter.get_preparation_count(sample_events)

        assert prep_count == 2

    def test_get_focus_time_count(self, sample_events):
        """Test counting focus time blocks."""
        formatter = CalendarFormatter()

        focus_count = formatter.get_focus_time_count(sample_events)

        assert focus_count == 1


class TestCalendarFormatterFormatting:
    """Test calendar event formatting for display."""

    @pytest.fixture
    def sample_event(self) -> CalendarEvent:
        """Create a sample calendar event."""
        return CalendarEvent(
            id="1",
            summary="Quarterly Planning Meeting",
            description="Discuss Q4 goals and objectives",
            start_time=datetime(2025, 10, 26, 14, 0, 0),
            end_time=datetime(2025, 10, 26, 16, 0, 0),
            location="Conference Room A",
            attendees=["alice@example.com", "bob@example.com"],
            importance_score=0.9,
            requires_preparation=True
        )

    def test_format_time_range(self, sample_event):
        """Test formatting time range for display."""
        formatter = CalendarFormatter()

        formatted = formatter.format_time_range(
            sample_event.start_time,
            sample_event.end_time
        )

        # Should return a readable time range
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_duration(self, sample_event):
        """Test formatting event duration."""
        formatter = CalendarFormatter()

        duration = formatter.format_duration(sample_event)

        # Should return duration string (e.g., "2 hours")
        assert isinstance(duration, str)
        assert "hour" in duration.lower()

    def test_get_duration_minutes(self, sample_event):
        """Test calculating duration in minutes."""
        formatter = CalendarFormatter()

        minutes = formatter.get_duration_minutes(sample_event)

        # 2 hours = 120 minutes
        assert minutes == 120

    def test_format_attendees_list(self, sample_event):
        """Test formatting attendees list."""
        formatter = CalendarFormatter()

        formatted = formatter.format_attendees_list(sample_event.attendees)

        # Should return list of formatted attendees
        assert isinstance(formatted, list)
        assert len(formatted) == 2

    def test_get_time_until_event(self):
        """Test calculating time until event starts."""
        formatter = CalendarFormatter()

        # Event in 2 hours
        future_event = CalendarEvent(
            id="1",
            summary="Future meeting",
            start_time=datetime.now() + timedelta(hours=2),
            end_time=datetime.now() + timedelta(hours=3)
        )

        time_until = formatter.get_time_until_event(future_event)

        # Should return a string like "2 hours"
        assert isinstance(time_until, str)
        assert len(time_until) > 0


class TestCalendarFormatterSorting:
    """Test calendar event sorting logic."""

    @pytest.fixture
    def sample_events(self) -> List[CalendarEvent]:
        """Create sample events for testing."""
        now = datetime.now()
        base = now.replace(hour=10, minute=0, second=0, microsecond=0)

        return [
            CalendarEvent(
                id="1",
                summary="Low priority",
                start_time=base + timedelta(hours=1),
                end_time=base + timedelta(hours=2),
                importance_score=0.3
            ),
            CalendarEvent(
                id="2",
                summary="High priority",
                start_time=base + timedelta(hours=2),
                end_time=base + timedelta(hours=3),
                importance_score=0.9
            ),
            CalendarEvent(
                id="3",
                summary="Medium priority",
                start_time=base,
                end_time=base + timedelta(hours=1),
                importance_score=0.6
            ),
        ]

    def test_sort_by_importance(self, sample_events):
        """Test sorting events by importance score."""
        formatter = CalendarFormatter()

        sorted_events = formatter.sort_by_importance(sample_events)

        # Should be in descending order of importance
        assert len(sorted_events) == 3
        assert sorted_events[0].importance_score >= sorted_events[1].importance_score
        assert sorted_events[1].importance_score >= sorted_events[2].importance_score

    def test_sort_by_start_time(self, sample_events):
        """Test sorting events by start time."""
        formatter = CalendarFormatter()

        sorted_events = formatter.sort_by_start_time(sample_events)

        # Should be in ascending order of start time
        assert len(sorted_events) == 3
        assert sorted_events[0].start_time <= sorted_events[1].start_time
        assert sorted_events[1].start_time <= sorted_events[2].start_time

    def test_sort_by_duration(self, sample_events):
        """Test sorting events by duration."""
        formatter = CalendarFormatter()

        # Add event with different duration
        long_event = CalendarEvent(
            id="4",
            summary="Long meeting",
            start_time=sample_events[0].start_time,
            end_time=sample_events[0].start_time + timedelta(hours=3),
            importance_score=0.5
        )
        events_with_long = sample_events + [long_event]

        sorted_events = formatter.sort_by_duration(events_with_long, longest_first=True)

        # Longest should be first
        assert sorted_events[0].id == "4"


class TestCalendarFormatterFiltering:
    """Test calendar event filtering logic."""

    @pytest.fixture
    def sample_events(self) -> List[CalendarEvent]:
        """Create sample events for testing."""
        now = datetime.now()
        today = now.replace(hour=10, minute=0, second=0, microsecond=0)

        return [
            CalendarEvent(
                id="1",
                summary="Project planning meeting",
                start_time=today,
                end_time=today + timedelta(hours=1),
                importance_score=0.9,
                requires_preparation=True
            ),
            CalendarEvent(
                id="2",
                summary="Casual chat",
                start_time=today + timedelta(hours=2),
                end_time=today + timedelta(hours=2, minutes=30),
                importance_score=0.3,
                requires_preparation=False,
                is_focus_time=False
            ),
            CalendarEvent(
                id="3",
                summary="Project review",
                start_time=today + timedelta(hours=4),
                end_time=today + timedelta(hours=5),
                importance_score=0.7,
                requires_preparation=False,
                is_focus_time=True
            ),
        ]

    def test_filter_by_summary(self, sample_events):
        """Test filtering by summary keyword."""
        formatter = CalendarFormatter()

        project_events = formatter.filter_by_summary(sample_events, "project")

        # Should match both "Project planning" and "Project review"
        assert len(project_events) == 2
        assert all("project" in event.summary.lower() for event in project_events)

    def test_filter_requires_preparation(self, sample_events):
        """Test filtering events requiring preparation."""
        formatter = CalendarFormatter()

        prep_events = formatter.filter_requires_preparation(sample_events)

        assert len(prep_events) == 1
        assert all(event.requires_preparation for event in prep_events)

    def test_filter_focus_time(self, sample_events):
        """Test filtering focus time blocks."""
        formatter = CalendarFormatter()

        focus_events = formatter.filter_focus_time(sample_events)

        assert len(focus_events) == 1
        assert all(event.is_focus_time for event in focus_events)

    def test_filter_high_importance(self, sample_events):
        """Test filtering high importance events."""
        formatter = CalendarFormatter()

        high_importance = formatter.filter_high_importance(sample_events, threshold=0.7)

        # Should include events with score >= 0.7
        assert len(high_importance) == 2
        assert all(event.importance_score >= 0.7 for event in high_importance)

    def test_filter_today(self, sample_events):
        """Test filtering events happening today."""
        formatter = CalendarFormatter()

        today_events = formatter.filter_today(sample_events)

        # All sample events are today
        assert len(today_events) == 3

    def test_filter_upcoming(self, sample_events):
        """Test filtering upcoming events (future only)."""
        formatter = CalendarFormatter()

        # Create past event
        past_event = CalendarEvent(
            id="4",
            summary="Past meeting",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1)
        )

        all_events = sample_events + [past_event]
        upcoming = formatter.filter_upcoming(all_events)

        # Should exclude past event
        assert len(upcoming) <= len(sample_events)
        assert all(event.start_time >= datetime.now() - timedelta(minutes=5) for event in upcoming)

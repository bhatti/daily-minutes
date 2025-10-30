"""CalendarFormatter - Business logic for formatting calendar event data for display.

This module contains NO UI rendering code. It only provides business logic
for formatting, grouping, sorting, and filtering calendar event data.

UI components should use this formatter to prepare data for display.
"""

from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict

from src.core.models import CalendarEvent
from src.core.logging import get_logger

logger = get_logger(__name__)


class CalendarFormatter:
    """Helper class for formatting calendar event data for display.

    Provides business logic for:
    - Grouping events by various criteria
    - Calculating statistics and meeting time
    - Formatting timestamps and durations
    - Sorting and filtering
    """

    def __init__(self):
        """Initialize CalendarFormatter."""
        pass

    # Grouping methods
    def group_by_date(self, events: List[CalendarEvent]) -> Dict[str, List[CalendarEvent]]:
        """Group events by date.

        Args:
            events: List of calendar events

        Returns:
            Dictionary mapping date strings to event lists
        """
        grouped = defaultdict(list)

        for event in events:
            date_str = event.start_time.strftime("%Y-%m-%d")
            grouped[date_str].append(event)

        return dict(grouped)

    def group_by_importance(self, events: List[CalendarEvent]) -> Dict[str, List[CalendarEvent]]:
        """Group events by importance level.

        Importance levels:
        - high: score >= 0.7
        - medium: 0.4 <= score < 0.7
        - low: score < 0.4

        Args:
            events: List of calendar events

        Returns:
            Dictionary mapping importance levels to event lists
        """
        grouped = defaultdict(list)

        for event in events:
            if event.importance_score >= 0.7:
                grouped["high"].append(event)
            elif event.importance_score >= 0.4:
                grouped["medium"].append(event)
            else:
                grouped["low"].append(event)

        return dict(grouped)

    def group_by_preparation_needed(self, events: List[CalendarEvent]) -> Dict[str, List[CalendarEvent]]:
        """Group events by preparation requirement.

        Args:
            events: List of calendar events

        Returns:
            Dictionary with 'requires_prep' and 'no_prep' keys
        """
        grouped = defaultdict(list)

        for event in events:
            if event.requires_preparation:
                grouped["requires_prep"].append(event)
            else:
                grouped["no_prep"].append(event)

        return dict(grouped)

    # Statistics methods
    def calculate_statistics(self, events: List[CalendarEvent]) -> Dict[str, int]:
        """Calculate calendar event statistics.

        Args:
            events: List of calendar events

        Returns:
            Dictionary with statistics
        """
        total = len(events)
        requires_preparation = sum(1 for event in events if event.requires_preparation)
        focus_time = sum(1 for event in events if event.is_focus_time)
        high_importance = sum(1 for event in events if event.importance_score >= 0.7)

        return {
            "total": total,
            "requires_preparation": requires_preparation,
            "focus_time": focus_time,
            "high_importance": high_importance,
        }

    def get_total_meeting_time(self, events: List[CalendarEvent]) -> float:
        """Get total meeting time in hours.

        Args:
            events: List of calendar events

        Returns:
            Total meeting time in hours
        """
        total_minutes = sum(self.get_duration_minutes(event) for event in events)
        return total_minutes / 60.0

    def get_preparation_count(self, events: List[CalendarEvent]) -> int:
        """Get count of events requiring preparation.

        Args:
            events: List of calendar events

        Returns:
            Number of events requiring preparation
        """
        return sum(1 for event in events if event.requires_preparation)

    def get_focus_time_count(self, events: List[CalendarEvent]) -> int:
        """Get count of focus time blocks.

        Args:
            events: List of calendar events

        Returns:
            Number of focus time blocks
        """
        return sum(1 for event in events if event.is_focus_time)

    # Formatting methods
    def format_time_range(self, start_time: datetime, end_time: datetime) -> str:
        """Format time range for display.

        Args:
            start_time: Event start time
            end_time: Event end time

        Returns:
            Formatted time range string
        """
        start_str = start_time.strftime("%H:%M")
        end_str = end_time.strftime("%H:%M")
        return f"{start_str} - {end_str}"

    def format_duration(self, event: CalendarEvent) -> str:
        """Format event duration for display.

        Args:
            event: Calendar event

        Returns:
            Formatted duration string (e.g., "2 hours", "30 minutes")
        """
        minutes = self.get_duration_minutes(event)

        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"

        hours = minutes / 60.0
        if hours == int(hours):
            hours = int(hours)
            return f"{hours} hour{'s' if hours != 1 else ''}"

        return f"{hours:.1f} hours"

    def get_duration_minutes(self, event: CalendarEvent) -> int:
        """Calculate event duration in minutes.

        Args:
            event: Calendar event

        Returns:
            Duration in minutes
        """
        duration = event.end_time - event.start_time
        return int(duration.total_seconds() / 60)

    def format_attendees_list(self, attendees: List[str]) -> List[str]:
        """Format attendees list.

        Args:
            attendees: List of attendee email addresses

        Returns:
            Formatted attendees list
        """
        # For now, just return as-is
        # In the future, could extract names, add avatars, etc.
        return attendees

    def get_time_until_event(self, event: CalendarEvent) -> str:
        """Get time until event starts.

        Args:
            event: Calendar event

        Returns:
            Formatted time until event (e.g., "2 hours", "in 30 minutes")
        """
        now = datetime.now()
        diff = event.start_time - now

        if diff.total_seconds() < 0:
            return "started"

        if diff < timedelta(minutes=1):
            return "starting now"
        elif diff < timedelta(hours=1):
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        elif diff < timedelta(days=1):
            hours = int(diff.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''}"
        elif diff < timedelta(days=2):
            return "tomorrow"
        else:
            days = int(diff.total_seconds() / 86400)
            return f"{days} days"

    # Sorting methods
    def sort_by_importance(self, events: List[CalendarEvent]) -> List[CalendarEvent]:
        """Sort events by importance score (highest first).

        Args:
            events: List of calendar events

        Returns:
            Sorted event list
        """
        return sorted(events, key=lambda e: e.importance_score, reverse=True)

    def sort_by_start_time(self, events: List[CalendarEvent]) -> List[CalendarEvent]:
        """Sort events by start time (earliest first).

        Args:
            events: List of calendar events

        Returns:
            Sorted event list
        """
        def get_start_time(e):
            if isinstance(e, dict):
                start_time = e.get('start_time')
                if isinstance(start_time, str):
                    from datetime import datetime
                    return datetime.fromisoformat(start_time)
                return start_time
            return e.start_time

        return sorted(events, key=get_start_time)

    def sort_by_duration(
        self, events: List[CalendarEvent], longest_first: bool = True
    ) -> List[CalendarEvent]:
        """Sort events by duration.

        Args:
            events: List of calendar events
            longest_first: If True, longest events first. If False, shortest first.

        Returns:
            Sorted event list
        """
        return sorted(
            events,
            key=lambda e: self.get_duration_minutes(e),
            reverse=longest_first
        )

    # Filtering methods
    def filter_by_summary(self, events: List[CalendarEvent], keyword: str) -> List[CalendarEvent]:
        """Filter events by summary keyword (case-insensitive).

        Args:
            events: List of calendar events
            keyword: Summary keyword to search for

        Returns:
            Events matching keyword
        """
        keyword_lower = keyword.lower()
        return [event for event in events if keyword_lower in event.summary.lower()]

    def filter_requires_preparation(self, events: List[CalendarEvent]) -> List[CalendarEvent]:
        """Filter events requiring preparation.

        Args:
            events: List of calendar events

        Returns:
            Events requiring preparation
        """
        return [event for event in events if event.requires_preparation]

    def filter_focus_time(self, events: List[CalendarEvent]) -> List[CalendarEvent]:
        """Filter focus time blocks.

        Args:
            events: List of calendar events

        Returns:
            Focus time events
        """
        return [event for event in events if event.is_focus_time]

    def filter_high_importance(
        self, events: List[CalendarEvent], threshold: float = 0.7
    ) -> List[CalendarEvent]:
        """Filter high importance events.

        Args:
            events: List of calendar events
            threshold: Minimum importance score (default: 0.7)

        Returns:
            High importance events
        """
        return [event for event in events if event.importance_score >= threshold]

    def filter_today(self, events: List[CalendarEvent]) -> List[CalendarEvent]:
        """Filter events happening today.

        Args:
            events: List of calendar events

        Returns:
            Events happening today
        """
        today = datetime.now().date()
        return [event for event in events if event.start_time.date() == today]

    def filter_upcoming(self, events: List[CalendarEvent]) -> List[CalendarEvent]:
        """Filter upcoming events (future only).

        Args:
            events: List of calendar events

        Returns:
            Upcoming events
        """
        # Allow a 5-minute buffer for events that just started
        now = datetime.now() - timedelta(minutes=5)
        return [event for event in events if event.start_time >= now]


# Singleton instance
_calendar_formatter: CalendarFormatter = None


def get_calendar_formatter() -> CalendarFormatter:
    """Get singleton CalendarFormatter instance.

    Returns:
        CalendarFormatter singleton
    """
    global _calendar_formatter

    if _calendar_formatter is None:
        _calendar_formatter = CalendarFormatter()
        logger.debug("calendar_formatter_initialized")

    return _calendar_formatter

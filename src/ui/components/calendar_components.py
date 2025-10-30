"""Calendar UI Components - Pure rendering layer for calendar event display.

This module contains ONLY UI rendering code using Streamlit.
All business logic is delegated to:
- CalendarFormatter (src/ui/formatters/calendar_formatter.py)
- CalendarService (src/services/calendar_service.py)

Component helper functions prepare data for rendering and handle state management.
"""

import streamlit as st
from typing import Dict, List, Optional
from datetime import datetime

from src.core.models import CalendarEvent
from src.ui.formatters.calendar_formatter import get_calendar_formatter
from src.services.calendar_service import get_calendar_service
from src.core.logging import get_logger

logger = get_logger(__name__)


# ===================================================================
# Helper Functions - These have unit tests
# ===================================================================

def prepare_event_display_data(event: CalendarEvent) -> Dict[str, any]:
    """Prepare event data for display using CalendarFormatter.

    Args:
        event: Calendar event to prepare

    Returns:
        Dictionary with display-ready data
    """
    formatter = get_calendar_formatter()

    return {
        "summary": event.summary,
        "time_range": formatter.format_time_range(event.start_time, event.end_time),
        "duration": formatter.format_duration(event),
        "time_until": formatter.get_time_until_event(event),
        "location": event.location or "",
        "attendees_display": formatter.format_attendees_list(event.attendees) if event.attendees else [],
        "importance_indicator": get_importance_indicator(event),
        "preparation_badge": get_preparation_badge(event),
        "importance_score": event.importance_score,
        "requires_preparation": event.requires_preparation,
    }


def get_importance_indicator(event: CalendarEvent) -> str:
    """Get importance indicator display for event.

    Args:
        event: Calendar event

    Returns:
        Indicator string (emoji or text)
    """
    if event.importance_score >= 0.9:
        return "🔴 Critical"
    elif event.importance_score >= 0.7:
        return "🟠 Important"
    elif event.importance_score >= 0.5:
        return "🟡 Medium"
    else:
        return ""  # No indicator for low importance


def get_preparation_badge(event: CalendarEvent) -> str:
    """Get preparation badge display for event.

    Args:
        event: Calendar event

    Returns:
        Preparation badge string
    """
    if not event.requires_preparation:
        return ""

    # Show preparation notes if available
    if event.preparation_notes:
        count = len(event.preparation_notes)
        return f"📋 Prep Required ({count} item{'s' if count != 1 else ''})"
    else:
        return "📋 Prep Required"


def should_show_event(
    event: CalendarEvent,
    filter_important: bool = False,
    filter_preparation: bool = False,
    search_query: Optional[str] = None
) -> bool:
    """Determine if event should be shown based on filters.

    Args:
        event: Calendar event
        filter_important: Only show important events
        filter_preparation: Only show events requiring preparation
        search_query: Search query to match against

    Returns:
        True if event should be shown, False otherwise
    """
    # Apply importance filter
    if filter_important and event.importance_score < 0.7:
        return False

    # Apply preparation filter
    if filter_preparation and not event.requires_preparation:
        return False

    # Apply search filter
    if search_query:
        query_lower = search_query.lower()

        # Search in summary
        if query_lower in event.summary.lower():
            return True

        # Search in description
        if event.description and query_lower in event.description.lower():
            return True

        # Search in location
        if event.location and query_lower in event.location.lower():
            return True

        # No match found
        return False

    # All filters passed
    return True


def group_events_for_display(events: List[CalendarEvent]) -> Dict[str, List[CalendarEvent]]:
    """Group events by date for display.

    Args:
        events: List of events to group

    Returns:
        Dictionary mapping date labels to event lists
    """
    if not events:
        return {}

    formatter = get_calendar_formatter()
    return formatter.group_by_date(events)


def get_event_conflicts(event: CalendarEvent, all_events: List[CalendarEvent]) -> List[CalendarEvent]:
    """Detect overlapping events (scheduling conflicts).

    Args:
        event: Event to check for conflicts
        all_events: All events to check against

    Returns:
        List of conflicting events
    """
    conflicts = []

    for other in all_events:
        # Skip self
        if other.id == event.id:
            continue

        # Check for time overlap
        # Events overlap if: start1 < end2 AND start2 < end1
        if event.start_time < other.end_time and other.start_time < event.end_time:
            conflicts.append(other)

    return conflicts


# ===================================================================
# Streamlit Rendering Functions - Pure UI code
# ===================================================================

def render_calendar_filters() -> Dict[str, any]:
    """Render calendar filter controls.

    Returns:
        Dictionary with filter settings
    """
    st.subheader("📅 Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_important = st.checkbox("Important only", value=False, key="calendar_filter_important")

    with col2:
        filter_preparation = st.checkbox("Requires prep", value=False, key="calendar_filter_preparation")

    with col3:
        days_ahead = st.selectbox(
            "Days ahead",
            options=[1, 3, 7, 14, 30],
            index=2,  # Default to 7 days
            key="calendar_days_ahead"
        )

    search_query = st.text_input("🔍 Search events", placeholder="Search summary, description, or location...", key="calendar_search_query")

    return {
        "filter_important": filter_important,
        "filter_preparation": filter_preparation,
        "days_ahead": days_ahead,
        "search_query": search_query if search_query else None
    }


def render_event_card(event: CalendarEvent, show_conflicts: bool = True, all_events: Optional[List[CalendarEvent]] = None) -> None:
    """Render a single calendar event as a card.

    Args:
        event: Calendar event to render
        show_conflicts: Whether to show conflict warnings
        all_events: All events for conflict detection
    """
    display_data = prepare_event_display_data(event)

    # Container for the event card
    with st.container():
        # Header row with time and importance
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**{display_data['time_range']}** ({display_data['duration']})")

        with col2:
            st.caption(display_data["time_until"])

        # Summary with importance indicator
        summary_line = display_data["summary"]
        if display_data["importance_indicator"]:
            summary_line = f"{display_data['importance_indicator']} {summary_line}"

        st.markdown(f"### {summary_line}")

        # Preparation badge
        if display_data["preparation_badge"]:
            st.info(display_data["preparation_badge"])

            # Show preparation notes if available
            if event.preparation_notes:
                with st.expander("📝 Preparation Notes"):
                    for note in event.preparation_notes:
                        st.markdown(f"• {note}")

        # Location and attendees
        if display_data["location"]:
            st.markdown(f"📍 **Location:** {display_data['location']}")

        if display_data["attendees_display"]:
            attendee_count = len(display_data["attendees_display"])
            with st.expander(f"👥 Attendees ({attendee_count})"):
                for attendee in display_data["attendees_display"]:
                    st.markdown(f"• {attendee}")

        # Description
        if event.description:
            with st.expander("📄 Description"):
                st.markdown(event.description)

        # Check for conflicts
        if show_conflicts and all_events:
            conflicts = get_event_conflicts(event, all_events)
            if conflicts:
                st.warning(f"⚠️ Scheduling conflict with {len(conflicts)} other event(s)")
                with st.expander("View conflicts"):
                    for conflict in conflicts:
                        st.markdown(f"• {conflict.summary} ({conflict.start_time.strftime('%H:%M')} - {conflict.end_time.strftime('%H:%M')})")

        # Divider between events
        st.divider()


def render_event_list(
    events: List[CalendarEvent],
    filters: Optional[Dict[str, any]] = None
) -> None:
    """Render list of events with grouping and filtering.

    Args:
        events: List of events to render
        filters: Optional filter settings
    """
    if not events:
        st.info("No events to display")
        return

    # Apply filters if provided
    if filters:
        filtered_events = [
            event for event in events
            if should_show_event(
                event,
                filter_important=filters.get("filter_important", False),
                filter_preparation=filters.get("filter_preparation", False),
                search_query=filters.get("search_query")
            )
        ]

        # Sort events by start time (always chronological for calendar)
        formatter = get_calendar_formatter()
        filtered_events = formatter.sort_by_start_time(filtered_events)
    else:
        filtered_events = events

    # Show count
    st.caption(f"Showing {len(filtered_events)} of {len(events)} events")

    # Group by date
    grouped = group_events_for_display(filtered_events)

    # Render each group
    for date_label, group_events in grouped.items():
        st.subheader(date_label)

        for event in group_events:
            render_event_card(event, show_conflicts=True, all_events=events)


def render_calendar_statistics(events: List[CalendarEvent]) -> None:
    """Render calendar event statistics.

    Args:
        events: List of events to analyze
    """
    if not events:
        return

    formatter = get_calendar_formatter()
    stats = formatter.calculate_statistics(events)
    total_hours = formatter.get_total_meeting_time(events)

    # Display as metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Events", stats["total"])

    with col2:
        st.metric("Important", stats["high_importance"])

    with col3:
        st.metric("Prep Required", stats["requires_preparation"])

    with col4:
        st.metric("Focus Time", stats["focus_time"])

    with col5:
        st.metric("Meeting Hours", f"{total_hours:.1f}")


def render_daily_view(events: List[CalendarEvent]) -> None:
    """Render today's events in daily view format.

    Args:
        events: All calendar events
    """
    from datetime import datetime, timedelta
    import pandas as pd

    # Filter events for today
    today = datetime.now().date()
    today_events = []
    for e in events:
        # Handle both CalendarEvent objects and dicts
        if isinstance(e, dict):
            start_time = e.get('start_time')
            if isinstance(start_time, str):
                from datetime import datetime as dt
                start_time = dt.fromisoformat(start_time)
            if start_time and start_time.date() == today:
                today_events.append(e)
        else:
            if e.start_time.date() == today:
                today_events.append(e)

    if not today_events:
        st.info("No events scheduled for today")
        return

    # Sort by start time
    formatter = get_calendar_formatter()
    today_events = formatter.sort_by_start_time(today_events)

    # Create table data for today's events
    table_data = []
    for event in today_events:
        # Handle both CalendarEvent objects and dicts
        if isinstance(event, dict):
            start_time = event.get('start_time')
            end_time = event.get('end_time')
            if isinstance(start_time, str):
                from datetime import datetime as dt
                start_time = dt.fromisoformat(start_time)
            if isinstance(end_time, str):
                from datetime import datetime as dt
                end_time = dt.fromisoformat(end_time)

            time_range = f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"
            summary = event.get('summary', 'Untitled')
            description = event.get('description', '')
            location = event.get('location', '')
            importance_score = event.get('importance_score', 0)
            requires_preparation = event.get('requires_preparation', False)
        else:
            time_range = f"{event.start_time.strftime('%H:%M')} - {event.end_time.strftime('%H:%M')}"
            summary = event.summary
            description = event.description
            location = event.location
            importance_score = event.importance_score
            requires_preparation = event.requires_preparation

        # Importance indicator
        if importance_score >= 0.9:
            importance = "🔴 Critical"
        elif importance_score >= 0.7:
            importance = "🟠 Important"
        elif importance_score >= 0.5:
            importance = "🟡 Medium"
        else:
            importance = "🔵 Low"

        # TLDR/Overview (use first 80 chars of description or summary)
        overview = description[:80] + "..." if description and len(description) > 80 else (description or summary)

        # Preparation indicator
        prep = "✅ Yes" if requires_preparation else ""

        table_data.append({
            "Time": time_range,
            "Summary": summary[:50] + "..." if len(summary) > 50 else summary,
            "Importance": importance,
            "Overview": overview,
            "Prep": prep,
            "Location": location[:30] + "..." if location and len(location) > 30 else (location or ""),
        })

    # Display as DataFrame
    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "Time": st.column_config.TextColumn("Time", width="small"),
            "Summary": st.column_config.TextColumn("Summary", width="medium"),
            "Importance": st.column_config.TextColumn("Importance", width="small"),
            "Overview": st.column_config.TextColumn("Overview", width="large"),
            "Prep": st.column_config.TextColumn("Prep", width="small"),
            "Location": st.column_config.TextColumn("Location", width="medium"),
        }
    )


def render_weekly_view(events: List[CalendarEvent]) -> None:
    """Render next 7 days in weekly view format.

    Args:
        events: All calendar events
    """
    from datetime import datetime, timedelta
    import pandas as pd

    # Filter events for next 7 days
    today = datetime.now().date()
    week_end = today + timedelta(days=7)

    week_events = [e for e in events if today <= e.start_time.date() < week_end]

    if not week_events:
        st.info("No events scheduled for the next 7 days")
        return

    # Sort by start time
    formatter = get_calendar_formatter()
    week_events = formatter.sort_by_start_time(week_events)

    st.subheader(f"📅 Next 7 Days ({today.strftime('%b %d')} - {week_end.strftime('%b %d')})")

    # Create table data for weekly events
    table_data = []
    for event in week_events:
        # Date and day
        event_date = event.start_time.strftime('%a %b %d')

        # Time
        time = event.start_time.strftime('%H:%M')

        # Importance indicator
        importance = get_importance_indicator(event)
        if not importance:
            importance = "🔵 Low"

        # TLDR/Overview
        overview = event.description[:60] + "..." if event.description and len(event.description) > 60 else (event.description or event.summary)

        # Days until
        days_diff = (event.start_time.date() - today).days
        if days_diff == 0:
            days_until = "Today"
        elif days_diff == 1:
            days_until = "Tomorrow"
        else:
            days_until = f"In {days_diff} days"

        table_data.append({
            "Date": event_date,
            "Time": time,
            "Summary": event.summary[:45] + "..." if len(event.summary) > 45 else event.summary,
            "Importance": importance,
            "Overview": overview,
            "Days Until": days_until,
        })

    # Display as DataFrame
    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Time": st.column_config.TextColumn("Time", width="small"),
            "Summary": st.column_config.TextColumn("Summary", width="medium"),
            "Importance": st.column_config.TextColumn("Importance", width="small"),
            "Overview": st.column_config.TextColumn("Overview", width="large"),
            "Days Until": st.column_config.TextColumn("Days Until", width="small"),
        }
    )


def render_calendar_tab() -> None:
    """Render the complete calendar tab.

    This is the main entry point for the calendar UI.
    """
    st.header("📅 Calendar")

    # Show statistics if we have events
    if "calendar_events" in st.session_state:
        events = st.session_state["calendar_events"]

        # Statistics
        with st.expander("📊 Statistics", expanded=False):
            render_calendar_statistics(events)

        # Daily View - Today's events
        st.subheader("Today")
        render_daily_view(events)

        st.divider()

        # Weekly View - Next 7 days
        st.subheader("This Week")
        render_weekly_view(events)
    else:
        st.info("No calendar events loaded. Events are automatically fetched in the background.")

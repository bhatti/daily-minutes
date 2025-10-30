"""CalendarService - Centralized calendar fetching and management logic.

Following the NewsService pattern, this service provides a clean interface
for fetching calendar events from multiple providers.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Dict

from src.core.models import CalendarEvent
from src.agents.calendar_agent import get_calendar_agent
from src.connectors.calendar.google_calendar_connector import get_google_calendar_connector
from src.connectors.calendar.outlook_calendar_connector import get_outlook_calendar_connector
from src.core.logging import get_logger

logger = get_logger(__name__)


class CalendarService:
    """
    Service for fetching and managing calendar events.

    Handles:
    - Fetching from multiple calendar providers (Google Calendar, Outlook Calendar)
    - Using CalendarAgent for orchestration
    - Progress reporting
    - Error handling
    """

    def __init__(self):
        """Initialize CalendarService."""
        self._calendar_agent = None
        self._connectors_initialized = False

    def _initialize_connectors(self) -> Dict[str, any]:
        """Initialize calendar connectors based on configuration.

        Returns:
            Dictionary of initialized connectors
        """
        import os

        if self._connectors_initialized:
            return self._calendar_agent.connectors if self._calendar_agent else {}

        connectors = {}

        # Check if using mock calendar data
        use_mock = os.getenv('USE_MOCK_CALENDAR', 'false').lower() == 'true'
        if use_mock:
            logger.info("using_mock_calendar_data")
            # Don't initialize any connectors - we'll use mock data directly
            self._connectors_initialized = True
            return connectors

        # Try to initialize Google Calendar connector
        try:
            google_connector = get_google_calendar_connector()
            if google_connector:
                connectors["google"] = google_connector
                logger.info("google_calendar_connector_initialized")
        except Exception as e:
            logger.warning("google_calendar_connector_init_failed", error=str(e))

        # Try to initialize Outlook Calendar connector
        try:
            outlook_connector = get_outlook_calendar_connector()
            if outlook_connector:
                connectors["outlook"] = outlook_connector
                logger.info("outlook_calendar_connector_initialized")
        except Exception as e:
            logger.warning("outlook_calendar_connector_init_failed", error=str(e))

        self._connectors_initialized = True
        return connectors

    async def fetch_events(
        self,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: Optional[int] = 50,
        filter_important: bool = False,
        filter_preparation: bool = False,
        sort_by_time: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[CalendarEvent]:
        """
        Fetch calendar events from all configured providers.

        Args:
            time_min: Start of time range (default: now)
            time_max: End of time range (default: 7 days from now)
            max_results: Maximum number of events to return
            filter_important: Only return high importance events
            filter_preparation: Only return events requiring preparation
            sort_by_time: Sort by start time (chronological)
            progress_callback: Optional callback for progress updates

        Returns:
            List of calendar events
        """
        # Set default time range
        if time_min is None:
            time_min = datetime.now()
        if time_max is None:
            time_max = time_min + timedelta(days=7)

        logger.info("fetch_events_start",
                   time_min=time_min.isoformat(),
                   time_max=time_max.isoformat(),
                   max_results=max_results)

        # Update progress
        if progress_callback:
            await progress_callback(0.1, "Initializing calendar connectors...")

        # Initialize connectors
        connectors = self._initialize_connectors()

        # Check if using mock calendar data
        import os
        use_mock = os.getenv('USE_MOCK_CALENDAR', 'false').lower() == 'true'

        if use_mock:
            # Use mock calendar data generator
            logger.info("fetching_mock_calendar_events")
            if progress_callback:
                await progress_callback(0.3, "Generating mock calendar events...")

            try:
                from tests.mock_calendar_data import generate_mock_calendar_events
                days_ahead = (time_max - time_min).days
                events = generate_mock_calendar_events(
                    count=max_results,
                    start_date=time_min,
                    days_ahead=days_ahead
                )
                # Filter to time range
                events = [e for e in events if time_min <= e.start_time <= time_max]
                events = events[:max_results]
            except Exception as e:
                logger.error("mock_calendar_generation_failed", error=str(e))
                events = []
        elif not connectors:
            logger.warning("no_calendar_connectors_available")
            return []
        else:
            # Update progress
            if progress_callback:
                await progress_callback(0.2, f"Fetching events from {len(connectors)} provider(s)...")

            # Get or create calendar agent
            if self._calendar_agent is None:
                self._calendar_agent = get_calendar_agent()
                # Add connectors to agent
                for name, connector in connectors.items():
                    self._calendar_agent.add_connector(name, connector)

            # Fetch events using agent
            try:
                events = await self._calendar_agent.fetch_events(
                    time_min=time_min,
                    time_max=time_max,
                    max_results=max_results,
                    sort_by_time=sort_by_time,
                    use_cache=True,
                    deduplicate=True
                )
            except Exception as e:
                logger.error("fetch_events_failed", error=str(e), exc_info=True)
                events = []

        # Update progress
        if progress_callback:
            await progress_callback(0.6, f"Retrieved {len(events)} events")

        # Apply additional filters
        if filter_important:
            events = [e for e in events if e.importance_score >= 0.7]
            if progress_callback:
                await progress_callback(0.7, f"Filtered to {len(events)} important events")

        if filter_preparation:
            events = [e for e in events if e.requires_preparation]
            if progress_callback:
                await progress_callback(0.8, f"Filtered to {len(events)} events requiring prep")

        # Update progress
        if progress_callback:
            await progress_callback(1.0, f"Complete: {len(events)} events retrieved")

        provider_label = "mock" if use_mock else str(list(connectors.keys()))
        logger.info("fetch_events_complete",
                   total_events=len(events),
                   providers=provider_label)

        return events

    async def get_today_events(self) -> List[CalendarEvent]:
        """Get events happening today.

        Returns:
            List of today's events sorted by start time
        """
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        return await self.fetch_events(
            time_min=today_start,
            time_max=today_end,
            sort_by_time=True
        )

    async def get_upcoming_events(
        self,
        hours: int = 24,
        max_results: int = 10
    ) -> List[CalendarEvent]:
        """Get upcoming events within next N hours.

        Args:
            hours: Number of hours to look ahead
            max_results: Maximum number of events to return

        Returns:
            List of upcoming events sorted by start time
        """
        now = datetime.now()
        future = now + timedelta(hours=hours)

        return await self.fetch_events(
            time_min=now,
            time_max=future,
            max_results=max_results,
            sort_by_time=True
        )

    async def get_important_events(
        self,
        days: int = 7,
        max_results: int = 10
    ) -> List[CalendarEvent]:
        """Get most important events.

        Args:
            days: Number of days to look ahead
            max_results: Maximum number of events to return

        Returns:
            List of important events sorted by importance
        """
        now = datetime.now()
        future = now + timedelta(days=days)

        events = await self.fetch_events(
            time_min=now,
            time_max=future,
            filter_important=True,
            sort_by_time=False,  # Will sort by importance
            max_results=max_results * 2  # Fetch more to filter
        )

        # Sort by importance
        events.sort(key=lambda e: e.importance_score, reverse=True)

        return events[:max_results]

    async def get_events_requiring_preparation(
        self,
        days: int = 7
    ) -> List[CalendarEvent]:
        """Get events requiring preparation.

        Args:
            days: Number of days to look ahead

        Returns:
            List of events requiring preparation
        """
        now = datetime.now()
        future = now + timedelta(days=days)

        return await self.fetch_events(
            time_min=now,
            time_max=future,
            filter_preparation=True,
            sort_by_time=True
        )

    async def get_focus_time_blocks(
        self,
        days: int = 7
    ) -> List[CalendarEvent]:
        """Get focus time blocks.

        Args:
            days: Number of days to look ahead

        Returns:
            List of focus time events
        """
        try:
            now = datetime.now()
            future = now + timedelta(days=days)

            events = await self.fetch_events(
                time_min=now,
                time_max=future,
                sort_by_time=True
            )

            # Filter for focus time
            focus_events = [e for e in events if e.is_focus_time]

            return focus_events

        except Exception as e:
            logger.error("get_focus_time_blocks_failed", error=str(e))
            return []

    async def get_event_statistics(
        self,
        days: int = 7
    ) -> Dict[str, any]:
        """Get event statistics for the next N days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with event statistics
        """
        try:
            now = datetime.now()
            future = now + timedelta(days=days)

            events = await self.fetch_events(
                time_min=now,
                time_max=future,
                sort_by_time=False
            )

            # Calculate statistics
            total_events = len(events)
            important_events = sum(1 for e in events if e.importance_score >= 0.7)
            prep_required = sum(1 for e in events if e.requires_preparation)
            focus_time = sum(1 for e in events if e.is_focus_time)

            # Calculate total meeting time
            total_minutes = sum(
                (e.end_time - e.start_time).total_seconds() / 60
                for e in events
            )
            total_hours = total_minutes / 60.0

            return {
                "total_events": total_events,
                "important_events": important_events,
                "prep_required": prep_required,
                "focus_time_blocks": focus_time,
                "total_meeting_hours": round(total_hours, 1),
                "avg_meeting_duration_minutes": round(total_minutes / total_events, 1) if total_events > 0 else 0
            }

        except Exception as e:
            logger.error("get_event_statistics_failed", error=str(e))
            return {
                "total_events": 0,
                "important_events": 0,
                "prep_required": 0,
                "focus_time_blocks": 0,
                "total_meeting_hours": 0,
                "avg_meeting_duration_minutes": 0
            }


# Singleton instance
_calendar_service: CalendarService = None


def get_calendar_service() -> CalendarService:
    """Get singleton CalendarService instance.

    Returns:
        CalendarService singleton
    """
    global _calendar_service

    if _calendar_service is None:
        _calendar_service = CalendarService()
        logger.debug("calendar_service_initialized")

    return _calendar_service

"""CalendarAgent for orchestrating multiple calendar connectors.

The CalendarAgent provides a unified interface for fetching, filtering, and managing
calendar events from multiple calendar providers (Google Calendar, Outlook Calendar).

Features:
- Multi-connector support (Google Calendar, Outlook Calendar)
- Event fetching with parallel connector execution
- Importance-based sorting and prioritization
- Advanced filtering (summary, location, attendee, preparation, focus time)
- Event deduplication across providers
- Caching with configurable TTL
- Metrics emission and activity logging
- Event statistics and analytics

Example:
    ```python
    from src.agents.calendar_agent import CalendarAgent
    from src.connectors.calendar.google_calendar_connector import get_google_calendar_connector
    from datetime import datetime, timedelta

    # Create agent with Google Calendar connector
    connector = get_google_calendar_connector(
        credentials_file="credentials.json"
    )
    connector.authenticate()

    agent = CalendarAgent(connectors={"google": connector})

    # Fetch upcoming events
    events = await agent.fetch_events(
        time_min=datetime.now(),
        time_max=datetime.now() + timedelta(days=7),
        sort_by_importance=True
    )
    ```
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import Counter

from src.core.models import CalendarEvent
from src.core.logging import get_logger
from src.core.metrics import get_metrics_manager
from src.database.sqlite_manager import get_db_manager

logger = get_logger(__name__)


class CalendarAgent:
    """Agent for orchestrating multiple calendar connectors.

    Provides unified event fetching, filtering, sorting, and caching.

    Attributes:
        connectors: Dictionary of calendar connectors (key=name, value=connector)
        cache_ttl_seconds: Cache TTL in seconds
        last_fetch_time: Timestamp of last event fetch
        cached_events: Cached event list
    """

    def __init__(
        self,
        connectors: Optional[Dict[str, Any]] = None,
        cache_ttl_seconds: int = 300  # 5 minutes default
    ):
        """Initialize CalendarAgent.

        Args:
            connectors: Dictionary of calendar connectors {name: connector_instance}
            cache_ttl_seconds: Cache time-to-live in seconds (default: 300)
        """
        self.connectors = connectors or {}
        self.cache_ttl_seconds = cache_ttl_seconds

        # Cache
        self.last_fetch_time: Optional[datetime] = None
        self.cached_events: List[CalendarEvent] = []

        # Observability
        self.metrics_manager = get_metrics_manager()
        self.db_manager = get_db_manager()

        logger.info("calendar_agent_initialized",
                   connector_count=len(self.connectors),
                   cache_ttl=cache_ttl_seconds)

    def add_connector(self, name: str, connector: Any) -> None:
        """Add a calendar connector.

        Args:
            name: Connector name (e.g., "google", "outlook")
            connector: Connector instance
        """
        self.connectors[name] = connector
        logger.info("connector_added", name=name, total_connectors=len(self.connectors))

    async def fetch_events(
        self,
        time_min: datetime,
        time_max: datetime,
        max_results: Optional[int] = None,
        filter_summary: Optional[str] = None,
        filter_location: Optional[str] = None,
        filter_attendee: Optional[str] = None,
        requires_preparation: Optional[bool] = None,
        is_focus_time: Optional[bool] = None,
        sort_by_time: bool = False,
        sort_by_importance: bool = False,
        use_cache: bool = True,
        deduplicate: bool = True
    ) -> List[CalendarEvent]:
        """Fetch calendar events from all connectors.

        Args:
            time_min: Start of time range
            time_max: End of time range
            max_results: Maximum number of events to return
            filter_summary: Filter by summary keyword (case-insensitive)
            filter_location: Filter by location keyword (case-insensitive)
            filter_attendee: Filter by attendee email
            requires_preparation: Filter by requires_preparation flag
            is_focus_time: Filter by is_focus_time flag
            sort_by_time: Sort by start time (chronological)
            sort_by_importance: Sort by importance score (highest first)
            use_cache: Whether to use cached events
            deduplicate: Whether to deduplicate events

        Returns:
            List of CalendarEvent objects
        """
        start_time = datetime.now()
        logger.info("calendar_agent_fetching_events",
                   time_min=time_min.isoformat(),
                   time_max=time_max.isoformat(),
                   connector_count=len(self.connectors))

        # Check cache
        if use_cache and self._is_cache_valid():
            logger.info("calendar_agent_using_cache")
            events = self.cached_events
        else:
            # Fetch from all connectors
            events = await self._fetch_from_all_connectors(time_min, time_max)

            # Update cache
            if use_cache:
                self.cached_events = events
                self.last_fetch_time = datetime.now()

        # Apply filters
        events = self._apply_filters(
            events,
            filter_summary=filter_summary,
            filter_location=filter_location,
            filter_attendee=filter_attendee,
            requires_preparation=requires_preparation,
            is_focus_time=is_focus_time
        )

        # Deduplicate
        if deduplicate:
            events = self._deduplicate_events(events)

        # Apply sorting
        if sort_by_time:
            events = sorted(events, key=lambda e: e.start_time)
        elif sort_by_importance:
            events = sorted(events, key=lambda e: e.importance_score, reverse=True)

        # Apply max_results
        if max_results:
            events = events[:max_results]

        duration = (datetime.now() - start_time).total_seconds()
        logger.info("calendar_agent_fetch_complete",
                   event_count=len(events),
                   duration_seconds=duration)

        # Emit metrics
        self.metrics_manager.emit({
            "metric": "calendar_agent_fetch",
            "value": len(events),
            "timestamp": datetime.now().isoformat(),
            "labels": {"connector_count": len(self.connectors)}
        })

        return events

    async def _fetch_from_all_connectors(
        self,
        time_min: datetime,
        time_max: datetime
    ) -> List[CalendarEvent]:
        """Fetch events from all connectors in parallel.

        Args:
            time_min: Start of time range
            time_max: End of time range

        Returns:
            Combined list of events from all connectors
        """
        if not self.connectors:
            logger.warning("calendar_agent_no_connectors")
            return []

        # Fetch from all connectors concurrently
        tasks = []
        for name, connector in self.connectors.items():
            task = self._fetch_from_connector(name, connector, time_min, time_max)
            tasks.append(task)

        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all events
        all_events = []
        for result in results:
            if isinstance(result, list):
                all_events.extend(result)
            elif isinstance(result, Exception):
                logger.error("calendar_agent_connector_error", error=str(result))

        return all_events

    async def _fetch_from_connector(
        self,
        name: str,
        connector: Any,
        time_min: datetime,
        time_max: datetime
    ) -> List[CalendarEvent]:
        """Fetch events from a single connector.

        Args:
            name: Connector name
            connector: Connector instance
            time_min: Start of time range
            time_max: End of time range

        Returns:
            List of events from connector
        """
        try:
            logger.debug("calendar_agent_fetching_from_connector", connector=name)
            events = await connector.fetch_events(time_min=time_min, time_max=time_max)
            logger.info("calendar_agent_connector_success",
                       connector=name,
                       event_count=len(events))
            return events
        except Exception as e:
            logger.error("calendar_agent_connector_failed",
                        connector=name,
                        error=str(e))
            return []

    def _apply_filters(
        self,
        events: List[CalendarEvent],
        filter_summary: Optional[str] = None,
        filter_location: Optional[str] = None,
        filter_attendee: Optional[str] = None,
        requires_preparation: Optional[bool] = None,
        is_focus_time: Optional[bool] = None
    ) -> List[CalendarEvent]:
        """Apply filters to events.

        Args:
            events: List of events to filter
            filter_summary: Filter by summary keyword
            filter_location: Filter by location keyword
            filter_attendee: Filter by attendee email
            requires_preparation: Filter by requires_preparation flag
            is_focus_time: Filter by is_focus_time flag

        Returns:
            Filtered list of events
        """
        filtered = events

        if filter_summary:
            filtered = [e for e in filtered if filter_summary.lower() in e.summary.lower()]

        if filter_location:
            filtered = [e for e in filtered if e.location and filter_location.lower() in e.location.lower()]

        if filter_attendee:
            filtered = [e for e in filtered if filter_attendee in e.attendees]

        if requires_preparation is not None:
            filtered = [e for e in filtered if e.requires_preparation == requires_preparation]

        if is_focus_time is not None:
            filtered = [e for e in filtered if e.is_focus_time == is_focus_time]

        return filtered

    def _deduplicate_events(self, events: List[CalendarEvent]) -> List[CalendarEvent]:
        """Deduplicate events based on summary, start time, and end time.

        Args:
            events: List of events to deduplicate

        Returns:
            Deduplicated list of events
        """
        seen = set()
        deduplicated = []

        for event in events:
            # Create unique key from summary and times
            key = (event.summary, event.start_time, event.end_time)

            if key not in seen:
                seen.add(key)
                deduplicated.append(event)
            else:
                logger.debug("calendar_agent_duplicate_event",
                           summary=event.summary,
                           start_time=event.start_time.isoformat())

        logger.info("calendar_agent_deduplication",
                   original_count=len(events),
                   deduplicated_count=len(deduplicated),
                   removed=len(events) - len(deduplicated))

        return deduplicated

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid.

        Returns:
            True if cache is valid, False otherwise
        """
        if not self.last_fetch_time:
            return False

        age = (datetime.now() - self.last_fetch_time).total_seconds()
        return age < self.cache_ttl_seconds

    def get_event_count_by_day(self) -> Dict[str, int]:
        """Get event count statistics by day.

        Returns:
            Dictionary mapping date strings to event counts
        """
        day_counts: Dict[str, int] = Counter()

        for event in self.cached_events:
            day_str = event.start_time.strftime("%Y-%m-%d")
            day_counts[day_str] += 1

        return dict(day_counts)

    def get_average_importance(self) -> float:
        """Calculate average importance score of cached events.

        Returns:
            Average importance score (0.0 to 1.0)
        """
        if not self.cached_events:
            return 0.0

        total = sum(e.importance_score for e in self.cached_events)
        return total / len(self.cached_events)

    def get_upcoming_events(self, hours: int = 24) -> List[CalendarEvent]:
        """Get upcoming events within the next N hours.

        Args:
            hours: Number of hours to look ahead

        Returns:
            List of upcoming events sorted by start time
        """
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)

        upcoming = [
            e for e in self.cached_events
            if now <= e.start_time <= cutoff
        ]

        # Sort by start time
        upcoming.sort(key=lambda e: e.start_time)

        return upcoming


# Singleton instance
_calendar_agent: CalendarAgent = None


def get_calendar_agent() -> CalendarAgent:
    """Get singleton CalendarAgent instance.

    Returns:
        CalendarAgent singleton
    """
    global _calendar_agent

    if _calendar_agent is None:
        _calendar_agent = CalendarAgent()
        logger.debug("calendar_agent_singleton_initialized")

    return _calendar_agent

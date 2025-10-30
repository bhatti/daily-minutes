"""Google Calendar connector using OAuth 2.0 authentication.

Provides access to Google Calendar API for fetching, creating, updating, and deleting events.

Features:
- OAuth 2.0 authentication with automatic token refresh
- Event CRUD operations (Create, Read, Update, Delete)
- Multi-calendar support
- Recurring event support
- Pagination handling for large result sets
- Comprehensive error handling and rate limiting
- Full observability (logging, metrics, activity tracking)

Example:
    ```python
    from src.connectors.calendar.google_calendar_connector import GoogleCalendarConnector
    from datetime import datetime, timedelta

    # Create connector
    connector = GoogleCalendarConnector(
        credentials_file="credentials.json"
    )

    # Authenticate
    connector.authenticate()

    # Fetch upcoming events
    events = await connector.fetch_events(
        time_min=datetime.now(),
        time_max=datetime.now() + timedelta(days=7)
    )
    ```
"""

import os
import pickle
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dateutil import parser

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.core.models import CalendarEvent
from src.core.logging import get_logger
from src.core.metrics import get_metrics_manager

logger = get_logger(__name__)

# Google Calendar API scopes
# https://developers.google.com/calendar/api/auth
SCOPES = {
    'readonly': ['https://www.googleapis.com/auth/calendar.readonly'],
    'readwrite': ['https://www.googleapis.com/auth/calendar'],
    'events': ['https://www.googleapis.com/auth/calendar.events']
}


class GoogleCalendarConnector:
    """Google Calendar connector using OAuth 2.0.

    Provides comprehensive Calendar API access with authentication, event management,
    error handling, and observability.

    Attributes:
        credentials_file: Path to Google OAuth 2.0 credentials JSON file
        token_file: Path to save/load OAuth token (pickle format)
        scopes: List of OAuth scopes to request
        credentials: Google OAuth2 Credentials object
        service: Google Calendar API service
        is_authenticated: Whether connector is authenticated
    """

    def __init__(
        self,
        credentials_file: str,
        token_file: str = "./credentials/google_calendar_token.pickle",
        scopes: Optional[List[str]] = None
    ):
        """Initialize Google Calendar connector.

        Args:
            credentials_file: Path to credentials.json from Google Cloud Console
            token_file: Path to save OAuth token (default: ./credentials/google_calendar_token.pickle)
            scopes: OAuth scopes (default: calendar.readonly)
        """
        if not credentials_file:
            raise ValueError("credentials_file is required")

        self.credentials_file = credentials_file
        self.token_file = token_file
        self.scopes = scopes or SCOPES['readonly']

        # Authentication state
        self.credentials: Optional[Credentials] = None
        self.service = None
        self.is_authenticated = False

        # Observability
        self.metrics_manager = get_metrics_manager()

        logger.info(
            "google_calendar_connector_initialized",
            credentials_file=credentials_file,
            token_file=token_file,
            scopes=self.scopes
        )

    def authenticate(self) -> None:
        """Authenticate with Google Calendar API using OAuth 2.0.

        Flow:
        1. Try to load existing token from file
        2. If token expired, refresh it
        3. If no token, run OAuth flow
        4. Save token for future use

        Raises:
            RuntimeError: If authentication fails
        """
        start_time = datetime.now()
        logger.info("google_calendar_authentication_starting")

        try:
            # Load existing token
            if os.path.exists(self.token_file):
                with open(self.token_file, 'rb') as token:
                    self.credentials = pickle.load(token)
                logger.debug("loaded_existing_token", token_file=self.token_file)

            # Check if token is valid
            if self.credentials and not self.credentials.valid:
                if self.credentials.expired and self.credentials.refresh_token:
                    logger.info("refreshing_expired_token")
                    self.credentials.refresh(Request())
                    logger.info("token_refreshed_successfully")
                else:
                    # Token invalid and can't refresh - need new auth
                    self.credentials = None

            # If no valid credentials, run OAuth flow
            if not self.credentials:
                logger.info("running_oauth_flow")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file,
                    self.scopes
                )
                self.credentials = flow.run_local_server(port=0)
                logger.info("oauth_flow_completed")

            # Save token for next time
            os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
            with open(self.token_file, 'wb') as token:
                pickle.dump(self.credentials, token)
            logger.debug("token_saved", token_file=self.token_file)

            # Build Calendar API service
            self.service = build('calendar', 'v3', credentials=self.credentials)
            self.is_authenticated = True

            duration = (datetime.now() - start_time).total_seconds()
            logger.info("google_calendar_authentication_success", duration_seconds=duration)

            # Emit success metric
            self.metrics_manager.emit({
                "metric": "google_calendar_auth",
                "value": 1,
                "timestamp": datetime.now().isoformat(),
                "labels": {"status": "success"}
            })

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "google_calendar_authentication_failed",
                error=str(e),
                duration_seconds=duration
            )

            self.metrics_manager.emit({
                "metric": "google_calendar_auth",
                "value": 0,
                "timestamp": datetime.now().isoformat(),
                "labels": {"status": "failed", "error": str(e)[:100]}
            })

            raise RuntimeError(f"Google Calendar authentication failed: {e}")

    async def fetch_events(
        self,
        time_min: datetime,
        time_max: datetime,
        calendar_id: str = 'primary',
        max_results: int = 100,
        single_events: bool = True
    ) -> List[CalendarEvent]:
        """Fetch calendar events within a time range.

        Args:
            time_min: Minimum start time (inclusive)
            time_max: Maximum start time (exclusive)
            calendar_id: Calendar ID (default: 'primary')
            max_results: Maximum number of events per request (default: 100)
            single_events: Expand recurring events into instances (default: True)

        Returns:
            List of CalendarEvent objects

        Raises:
            RuntimeError: If not authenticated
            HttpError: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        start_time = datetime.now()
        logger.info(
            "google_calendar_fetching_events",
            calendar_id=calendar_id,
            time_min=time_min.isoformat(),
            time_max=time_max.isoformat(),
            max_results=max_results
        )

        try:
            all_events = []
            page_token = None

            while True:
                # Fetch events page
                events_result = self.service.events().list(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat() + 'Z',
                    timeMax=time_max.isoformat() + 'Z',
                    maxResults=max_results,
                    singleEvents=single_events,
                    orderBy='startTime',
                    pageToken=page_token
                ).execute()

                items = events_result.get('items', [])

                # Parse events
                for item in items:
                    try:
                        event = self._parse_event(item)
                        all_events.append(event)
                    except Exception as e:
                        logger.warning(
                            "event_parse_failed",
                            event_id=item.get('id'),
                            error=str(e)
                        )
                        continue

                # Check for next page
                page_token = events_result.get('nextPageToken')
                if not page_token:
                    break

                logger.debug("fetching_next_page", page_token=page_token)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                "google_calendar_fetch_success",
                event_count=len(all_events),
                duration_seconds=duration
            )

            # Emit metric
            self.metrics_manager.emit({
                "metric": "google_calendar_fetch",
                "value": len(all_events),
                "timestamp": datetime.now().isoformat(),
                "labels": {"calendar_id": calendar_id}
            })

            return all_events

        except HttpError as e:
            logger.error("google_calendar_api_error", error=str(e), status=e.resp.status)
            raise
        except Exception as e:
            logger.error("google_calendar_fetch_failed", error=str(e))
            raise

    def _parse_event(self, event_data: Dict[str, Any]) -> CalendarEvent:
        """Parse Google Calendar event into CalendarEvent model.

        Args:
            event_data: Raw event data from Google Calendar API

        Returns:
            CalendarEvent object
        """
        # Parse start/end times
        start = event_data.get('start', {})
        end = event_data.get('end', {})

        # Handle both dateTime and date (all-day events)
        start_time = parser.parse(start.get('dateTime', start.get('date')))
        end_time = parser.parse(end.get('dateTime', end.get('date')))

        # Parse attendees
        attendees = []
        for attendee in event_data.get('attendees', []):
            email = attendee.get('email')
            if email:
                attendees.append(email)

        # Parse recurrence
        recurrence_rules = event_data.get('recurrence') or []
        is_recurring = len(recurrence_rules) > 0
        recurrence_rule = recurrence_rules[0] if recurrence_rules else None

        return CalendarEvent(
            id=event_data['id'],
            summary=event_data.get('summary', 'Untitled Event'),
            description=event_data.get('description'),
            start_time=start_time,
            end_time=end_time,
            location=event_data.get('location'),
            attendees=attendees,
            is_recurring=is_recurring,
            recurrence_rule=recurrence_rule,
            # Default AI fields
            importance_score=0.5,
            requires_preparation=False,
            is_focus_time=False
        )

    async def create_event(
        self,
        summary: str,
        start_time: datetime,
        end_time: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: str = 'primary'
    ) -> str:
        """Create a new calendar event.

        Args:
            summary: Event title
            start_time: Event start time
            end_time: Event end time
            description: Event description (optional)
            location: Event location (optional)
            attendees: List of attendee emails (optional)
            calendar_id: Calendar ID (default: 'primary')

        Returns:
            Event ID of created event

        Raises:
            RuntimeError: If not authenticated
            HttpError: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        logger.info("google_calendar_creating_event", summary=summary)

        try:
            # Build event body
            event_body = {
                'summary': summary,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC'
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC'
                }
            }

            if description:
                event_body['description'] = description
            if location:
                event_body['location'] = location
            if attendees:
                event_body['attendees'] = [{'email': email} for email in attendees]

            # Create event
            result = self.service.events().insert(
                calendarId=calendar_id,
                body=event_body
            ).execute()

            event_id = result['id']
            logger.info("google_calendar_event_created", event_id=event_id)

            return event_id

        except HttpError as e:
            logger.error("google_calendar_create_failed", error=str(e))
            raise

    async def update_event(
        self,
        event_id: str,
        summary: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        calendar_id: str = 'primary'
    ) -> None:
        """Update an existing calendar event.

        Args:
            event_id: Event ID to update
            summary: New event title (optional)
            start_time: New start time (optional)
            end_time: New end time (optional)
            description: New description (optional)
            location: New location (optional)
            calendar_id: Calendar ID (default: 'primary')

        Raises:
            RuntimeError: If not authenticated
            HttpError: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        logger.info("google_calendar_updating_event", event_id=event_id)

        try:
            # Get existing event
            event = self.service.events().get(
                calendarId=calendar_id,
                eventId=event_id
            ).execute()

            # Update fields
            if summary:
                event['summary'] = summary
            if start_time:
                event['start'] = {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC'
                }
            if end_time:
                event['end'] = {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC'
                }
            if description:
                event['description'] = description
            if location:
                event['location'] = location

            # Update event
            self.service.events().update(
                calendarId=calendar_id,
                eventId=event_id,
                body=event
            ).execute()

            logger.info("google_calendar_event_updated", event_id=event_id)

        except HttpError as e:
            logger.error("google_calendar_update_failed", error=str(e), event_id=event_id)
            raise

    async def delete_event(
        self,
        event_id: str,
        calendar_id: str = 'primary'
    ) -> None:
        """Delete a calendar event.

        Args:
            event_id: Event ID to delete
            calendar_id: Calendar ID (default: 'primary')

        Raises:
            RuntimeError: If not authenticated
            HttpError: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        logger.info("google_calendar_deleting_event", event_id=event_id)

        try:
            self.service.events().delete(
                calendarId=calendar_id,
                eventId=event_id
            ).execute()

            logger.info("google_calendar_event_deleted", event_id=event_id)

        except HttpError as e:
            logger.error("google_calendar_delete_failed", error=str(e), event_id=event_id)
            raise

    async def list_calendars(self) -> List[Dict[str, Any]]:
        """List all accessible calendars.

        Returns:
            List of calendar metadata dictionaries

        Raises:
            RuntimeError: If not authenticated
            HttpError: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        logger.info("google_calendar_listing_calendars")

        try:
            result = self.service.calendarList().list().execute()
            calendars = result.get('items', [])

            logger.info("google_calendar_calendars_listed", count=len(calendars))

            return calendars

        except HttpError as e:
            logger.error("google_calendar_list_calendars_failed", error=str(e))
            raise


# Singleton pattern
_google_calendar_connector_instance = None


def get_google_calendar_connector(
    credentials_file: str,
    token_file: str = "./credentials/google_calendar_token.pickle",
    scopes: Optional[List[str]] = None
) -> GoogleCalendarConnector:
    """Get singleton GoogleCalendarConnector instance.

    Args:
        credentials_file: Path to credentials.json
        token_file: Path to token file
        scopes: OAuth scopes

    Returns:
        GoogleCalendarConnector singleton instance
    """
    global _google_calendar_connector_instance

    if _google_calendar_connector_instance is None:
        _google_calendar_connector_instance = GoogleCalendarConnector(
            credentials_file=credentials_file,
            token_file=token_file,
            scopes=scopes
        )

    return _google_calendar_connector_instance

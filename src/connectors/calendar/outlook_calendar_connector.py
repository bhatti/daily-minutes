"""Outlook/Microsoft 365 Calendar connector using Microsoft Graph API.

Provides access to Outlook/Microsoft 365 Calendar via Microsoft Graph API with OAuth 2.0.

Features:
- OAuth 2.0 authentication via MSAL (Microsoft Authentication Library)
- Event CRUD operations (Create, Read, Update, Delete)
- Multi-calendar support
- Recurring event support
- Pagination handling for large result sets
- Comprehensive error handling
- Full observability (logging, metrics, activity tracking)

OAuth Setup:
1. Register app in Azure AD portal (portal.azure.com)
2. Add "Calendars.Read" and "Calendars.ReadWrite" permissions
3. Configure redirect URI for your application
4. Save client_id, client_secret, and tenant_id

Example:
    ```python
    from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector
    from datetime import datetime, timedelta

    # Create connector
    connector = OutlookCalendarConnector(
        client_id="your_client_id",
        client_secret="your_client_secret",
        tenant_id="your_tenant_id"
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
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dateutil import parser

try:
    import msal
    import requests
except ImportError:
    raise ImportError(
        "Microsoft Graph API dependencies not installed. "
        "Install with: pip install msal requests"
    )

from src.core.models import CalendarEvent
from src.core.logging import get_logger
from src.core.metrics import get_metrics_manager

logger = get_logger(__name__)

# Microsoft Graph API configuration
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"
SCOPES = ["https://graph.microsoft.com/.default"]


class OutlookCalendarConnector:
    """Outlook/Microsoft 365 Calendar connector using Microsoft Graph API.

    Provides comprehensive Calendar API access with authentication, event management,
    error handling, and observability.

    Attributes:
        client_id: Azure AD application (client) ID
        client_secret: Azure AD application secret
        tenant_id: Azure AD tenant (directory) ID
        access_token: Current OAuth access token
        token_cache_path: Path to token cache file
        is_authenticated: Whether connector is authenticated
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        token_cache_path: str = "credentials/outlook_calendar_token_cache.json"
    ):
        """Initialize Outlook Calendar connector.

        Args:
            client_id: Azure AD application (client) ID
            client_secret: Azure AD application secret
            tenant_id: Azure AD tenant (directory) ID
            token_cache_path: Path to token cache file

        Raises:
            ValueError: If credentials are missing
        """
        if not client_id or not client_secret or not tenant_id:
            raise ValueError("client_id, client_secret, and tenant_id are required")

        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.token_cache_path = token_cache_path
        self.access_token: Optional[str] = None

        # Initialize MSAL app
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"
        self.msal_app: Optional[msal.ConfidentialClientApplication] = None

        # Observability
        self.metrics_manager = get_metrics_manager()

        logger.info(
            "outlook_calendar_connector_initialized",
            tenant_id=tenant_id,
            token_cache_path=token_cache_path
        )

    @property
    def is_authenticated(self) -> bool:
        """Check if connector is authenticated."""
        return self.access_token is not None

    def _get_msal_app(self) -> msal.ConfidentialClientApplication:
        """Get or create MSAL application with token cache.

        Returns:
            MSAL ConfidentialClientApplication instance
        """
        if self.msal_app is None:
            # Load token cache if exists
            cache = msal.SerializableTokenCache()
            if os.path.exists(self.token_cache_path):
                with open(self.token_cache_path, 'r') as f:
                    cache.deserialize(f.read())

            self.msal_app = msal.ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=self.authority,
                token_cache=cache
            )

        return self.msal_app

    def _save_token_cache(self):
        """Save token cache to disk."""
        if self.msal_app and self.msal_app.token_cache.has_state_changed:
            os.makedirs(os.path.dirname(self.token_cache_path), exist_ok=True)
            with open(self.token_cache_path, 'w') as f:
                f.write(self.msal_app.token_cache.serialize())

    def authenticate(self) -> None:
        """Authenticate with Microsoft Graph API using OAuth 2.0.

        Flow:
        1. Try to get token from cache (silent authentication)
        2. If no cached token, acquire token using client credentials
        3. Save token to cache for future use

        Raises:
            RuntimeError: If authentication fails
        """
        start_time = datetime.now()
        logger.info("outlook_calendar_authentication_starting")

        try:
            app = self._get_msal_app()

            # Try silent authentication first (from cache)
            accounts = app.get_accounts()
            if accounts:
                logger.debug("attempting_silent_authentication")
                result = app.acquire_token_silent(SCOPES, account=accounts[0])
                if result and "access_token" in result:
                    self.access_token = result["access_token"]
                    logger.info("silent_authentication_successful")
                    self._save_token_cache()

                    duration = (datetime.now() - start_time).total_seconds()
                    self.metrics_manager.emit({
                        "metric": "outlook_calendar_auth",
                        "value": 1,
                        "timestamp": datetime.now().isoformat(),
                        "labels": {"status": "success", "method": "silent"}
                    })
                    return

            # Fall back to client credentials flow
            logger.info("acquiring_token_via_client_credentials")
            result = app.acquire_token_for_client(scopes=SCOPES)

            if "access_token" in result:
                self.access_token = result["access_token"]
                self._save_token_cache()

                duration = (datetime.now() - start_time).total_seconds()
                logger.info("outlook_calendar_authentication_success", duration_seconds=duration)

                self.metrics_manager.emit({
                    "metric": "outlook_calendar_auth",
                    "value": 1,
                    "timestamp": datetime.now().isoformat(),
                    "labels": {"status": "success", "method": "client_credentials"}
                })
            else:
                error_msg = result.get("error_description", result.get("error", "Unknown error"))
                raise RuntimeError(f"Authentication failed: {error_msg}")

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "outlook_calendar_authentication_failed",
                error=str(e),
                duration_seconds=duration
            )

            self.metrics_manager.emit({
                "metric": "outlook_calendar_auth",
                "value": 0,
                "timestamp": datetime.now().isoformat(),
                "labels": {"status": "failed", "error": str(e)[:100]}
            })

            raise RuntimeError(f"Outlook Calendar authentication failed: {e}")

    async def fetch_events(
        self,
        time_min: datetime,
        time_max: datetime,
        calendar_id: str = 'me/calendar',
        max_results: int = 100
    ) -> List[CalendarEvent]:
        """Fetch calendar events within a time range.

        Args:
            time_min: Minimum start time (inclusive)
            time_max: Maximum start time (exclusive)
            calendar_id: Calendar ID (default: 'me/calendar' for primary calendar)
            max_results: Maximum number of events per request (default: 100)

        Returns:
            List of CalendarEvent objects

        Raises:
            RuntimeError: If not authenticated
            Exception: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        start_time = datetime.now()
        logger.info(
            "outlook_calendar_fetching_events",
            calendar_id=calendar_id,
            time_min=time_min.isoformat(),
            time_max=time_max.isoformat(),
            max_results=max_results
        )

        try:
            all_events = []

            # Build URL for calendarView endpoint (expands recurring events)
            base_url = f"{GRAPH_API_ENDPOINT}/{calendar_id}/calendarView"

            # Format times for Microsoft Graph API
            params = {
                "startDateTime": time_min.isoformat(),
                "endDateTime": time_max.isoformat(),
                "$top": max_results,
                "$orderby": "start/dateTime"
            }

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }

            url = base_url

            # Handle pagination
            while url:
                response = requests.get(url, params=params if url == base_url else None, headers=headers)
                response.raise_for_status()

                data = response.json()
                items = data.get("value", [])

                # Parse events
                for item in items:
                    try:
                        event = self._parse_event(item)
                        all_events.append(event)
                    except Exception as e:
                        logger.warning(
                            "event_parse_failed",
                            event_id=item.get("id"),
                            error=str(e)
                        )
                        continue

                # Check for next page
                url = data.get("@odata.nextLink")
                if url:
                    logger.debug("fetching_next_page", next_url=url)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                "outlook_calendar_fetch_success",
                event_count=len(all_events),
                duration_seconds=duration
            )

            # Emit metric
            self.metrics_manager.emit({
                "metric": "outlook_calendar_fetch",
                "value": len(all_events),
                "timestamp": datetime.now().isoformat(),
                "labels": {"calendar_id": calendar_id}
            })

            return all_events

        except Exception as e:
            logger.error("outlook_calendar_fetch_failed", error=str(e))
            raise

    def _parse_event(self, event_data: Dict[str, Any]) -> CalendarEvent:
        """Parse Microsoft Graph event into CalendarEvent model.

        Args:
            event_data: Raw event data from Microsoft Graph API

        Returns:
            CalendarEvent object
        """
        # Parse start/end times
        start = event_data.get("start", {})
        end = event_data.get("end", {})

        # Handle dateTime format from Microsoft Graph
        start_time = parser.parse(start.get("dateTime"))
        end_time = parser.parse(end.get("dateTime"))

        # Parse attendees
        attendees = []
        for attendee in event_data.get("attendees", []):
            email_address = attendee.get("emailAddress", {})
            email = email_address.get("address")
            if email:
                attendees.append(email)

        # Parse location
        location_obj = event_data.get("location", {})
        location = location_obj.get("displayName")

        # Parse recurrence
        recurrence = event_data.get("recurrence")
        is_recurring = recurrence is not None
        recurrence_rule = json.dumps(recurrence) if recurrence else None

        # Parse description
        description = event_data.get("bodyPreview") or event_data.get("body", {}).get("content")

        return CalendarEvent(
            id=event_data["id"],
            summary=event_data.get("subject", "Untitled Event"),
            description=description,
            start_time=start_time,
            end_time=end_time,
            location=location,
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
        calendar_id: str = 'me/calendar'
    ) -> str:
        """Create a new calendar event.

        Args:
            summary: Event title
            start_time: Event start time
            end_time: Event end time
            description: Event description (optional)
            location: Event location (optional)
            attendees: List of attendee emails (optional)
            calendar_id: Calendar ID (default: 'me/calendar')

        Returns:
            Event ID of created event

        Raises:
            RuntimeError: If not authenticated
            Exception: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        logger.info("outlook_calendar_creating_event", summary=summary)

        try:
            # Build event body
            event_body = {
                "subject": summary,
                "start": {
                    "dateTime": start_time.isoformat(),
                    "timeZone": "UTC"
                },
                "end": {
                    "dateTime": end_time.isoformat(),
                    "timeZone": "UTC"
                }
            }

            if description:
                event_body["body"] = {
                    "contentType": "text",
                    "content": description
                }

            if location:
                event_body["location"] = {
                    "displayName": location
                }

            if attendees:
                event_body["attendees"] = [
                    {
                        "emailAddress": {"address": email},
                        "type": "required"
                    }
                    for email in attendees
                ]

            # Create event
            url = f"{GRAPH_API_ENDPOINT}/{calendar_id}/events"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }

            response = requests.post(url, json=event_body, headers=headers)
            response.raise_for_status()

            result = response.json()
            event_id = result["id"]

            logger.info("outlook_calendar_event_created", event_id=event_id)

            return event_id

        except Exception as e:
            logger.error("outlook_calendar_create_failed", error=str(e))
            raise

    async def update_event(
        self,
        event_id: str,
        summary: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        calendar_id: str = 'me/calendar'
    ) -> None:
        """Update an existing calendar event.

        Args:
            event_id: Event ID to update
            summary: New event title (optional)
            start_time: New start time (optional)
            end_time: New end time (optional)
            description: New description (optional)
            location: New location (optional)
            calendar_id: Calendar ID (default: 'me/calendar')

        Raises:
            RuntimeError: If not authenticated
            Exception: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        logger.info("outlook_calendar_updating_event", event_id=event_id)

        try:
            # Build update body (only include fields to update)
            update_body = {}

            if summary:
                update_body["subject"] = summary
            if start_time:
                update_body["start"] = {
                    "dateTime": start_time.isoformat(),
                    "timeZone": "UTC"
                }
            if end_time:
                update_body["end"] = {
                    "dateTime": end_time.isoformat(),
                    "timeZone": "UTC"
                }
            if description:
                update_body["body"] = {
                    "contentType": "text",
                    "content": description
                }
            if location:
                update_body["location"] = {
                    "displayName": location
                }

            # Update event
            url = f"{GRAPH_API_ENDPOINT}/{calendar_id}/events/{event_id}"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }

            response = requests.patch(url, json=update_body, headers=headers)
            response.raise_for_status()

            logger.info("outlook_calendar_event_updated", event_id=event_id)

        except Exception as e:
            logger.error("outlook_calendar_update_failed", error=str(e), event_id=event_id)
            raise

    async def delete_event(
        self,
        event_id: str,
        calendar_id: str = 'me/calendar'
    ) -> None:
        """Delete a calendar event.

        Args:
            event_id: Event ID to delete
            calendar_id: Calendar ID (default: 'me/calendar')

        Raises:
            RuntimeError: If not authenticated
            Exception: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        logger.info("outlook_calendar_deleting_event", event_id=event_id)

        try:
            url = f"{GRAPH_API_ENDPOINT}/{calendar_id}/events/{event_id}"
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }

            response = requests.delete(url, headers=headers)
            response.raise_for_status()

            logger.info("outlook_calendar_event_deleted", event_id=event_id)

        except Exception as e:
            logger.error("outlook_calendar_delete_failed", error=str(e), event_id=event_id)
            raise

    async def list_calendars(self) -> List[Dict[str, Any]]:
        """List all accessible calendars.

        Returns:
            List of calendar metadata dictionaries

        Raises:
            RuntimeError: If not authenticated
            Exception: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        logger.info("outlook_calendar_listing_calendars")

        try:
            url = f"{GRAPH_API_ENDPOINT}/me/calendars"
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            result = response.json()
            calendars = result.get("value", [])

            logger.info("outlook_calendar_calendars_listed", count=len(calendars))

            return calendars

        except Exception as e:
            logger.error("outlook_calendar_list_calendars_failed", error=str(e))
            raise


# Singleton pattern
_outlook_calendar_connector_instance = None


def get_outlook_calendar_connector(
    client_id: str,
    client_secret: str,
    tenant_id: str,
    token_cache_path: str = "credentials/outlook_calendar_token_cache.json"
) -> OutlookCalendarConnector:
    """Get singleton OutlookCalendarConnector instance.

    Args:
        client_id: Azure AD application (client) ID
        client_secret: Azure AD application secret
        tenant_id: Azure AD tenant (directory) ID
        token_cache_path: Path to token cache file

    Returns:
        OutlookCalendarConnector singleton instance
    """
    global _outlook_calendar_connector_instance

    if _outlook_calendar_connector_instance is None:
        _outlook_calendar_connector_instance = OutlookCalendarConnector(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            token_cache_path=token_cache_path
        )

    return _outlook_calendar_connector_instance

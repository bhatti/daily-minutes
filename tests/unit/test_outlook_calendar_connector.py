#!/usr/bin/env python3
"""Unit tests for Outlook Calendar connector.

Tests the Microsoft Graph API-based calendar connector with OAuth 2.0 via MSAL.
All tests use mocks - no real API calls are made.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, mock_open
import json

from src.core.models import CalendarEvent


@pytest.fixture
def mock_msal_app():
    """Mock MSAL ConfidentialClientApplication."""
    app = Mock()
    app.token_cache = Mock()
    app.token_cache.has_state_changed = False
    app.token_cache.serialize = Mock(return_value="{}")
    return app


@pytest.fixture
def mock_access_token():
    """Mock access token."""
    return "mock_access_token_12345"


class TestOutlookCalendarConnectorInitialization:
    """Test Outlook Calendar connector initialization."""

    def test_init_with_credentials(self):
        """Test initialization with valid credentials."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )

        assert connector.client_id == "test_client_id"
        assert connector.client_secret == "test_client_secret"
        assert connector.tenant_id == "test_tenant_id"
        assert connector.access_token is None
        assert connector.is_authenticated is False

    def test_init_with_custom_token_cache_path(self):
        """Test initialization with custom token cache path."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id",
            token_cache_path="custom/path/token.json"
        )

        assert connector.token_cache_path == "custom/path/token.json"

    def test_init_without_client_id(self):
        """Test initialization without client_id raises error."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        with pytest.raises(ValueError, match="client_id, client_secret, and tenant_id are required"):
            OutlookCalendarConnector(
                client_id="",
                client_secret="secret",
                tenant_id="tenant"
            )

    def test_init_without_client_secret(self):
        """Test initialization without client_secret raises error."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        with pytest.raises(ValueError, match="client_id, client_secret, and tenant_id are required"):
            OutlookCalendarConnector(
                client_id="client",
                client_secret="",
                tenant_id="tenant"
            )

    def test_init_without_tenant_id(self):
        """Test initialization without tenant_id raises error."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        with pytest.raises(ValueError, match="client_id, client_secret, and tenant_id are required"):
            OutlookCalendarConnector(
                client_id="client",
                client_secret="secret",
                tenant_id=""
            )


class TestOutlookCalendarConnectorAuthentication:
    """Test Outlook Calendar connector authentication."""

    @patch('src.connectors.calendar.outlook_calendar_connector.msal.ConfidentialClientApplication')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_authenticate_with_client_credentials(
        self, mock_file, mock_makedirs, mock_exists, mock_msal_class, mock_msal_app
    ):
        """Test authentication using client credentials flow."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        # Mock token cache doesn't exist
        mock_exists.return_value = False

        # Mock MSAL app
        mock_msal_class.return_value = mock_msal_app
        mock_msal_app.get_accounts.return_value = []  # No cached accounts
        mock_msal_app.acquire_token_for_client.return_value = {
            "access_token": "new_access_token",
            "token_type": "Bearer"
        }

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )

        connector.authenticate()

        assert connector.is_authenticated is True
        assert connector.access_token == "new_access_token"
        mock_msal_app.acquire_token_for_client.assert_called_once()

    @patch('src.connectors.calendar.outlook_calendar_connector.msal.ConfidentialClientApplication')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"cached": "token"}')
    def test_authenticate_with_cached_token(
        self, mock_file, mock_exists, mock_msal_class, mock_msal_app
    ):
        """Test authentication with cached token."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        # Mock token cache exists
        mock_exists.return_value = True

        # Mock account
        mock_account = {"username": "test@example.com"}

        # Mock MSAL app with cached token
        mock_msal_class.return_value = mock_msal_app
        mock_msal_app.get_accounts.return_value = [mock_account]  # Has cached account
        mock_msal_app.acquire_token_silent.return_value = {
            "access_token": "cached_access_token",
            "token_type": "Bearer"
        }

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )

        connector.authenticate()

        assert connector.is_authenticated is True
        assert connector.access_token == "cached_access_token"

    @patch('src.connectors.calendar.outlook_calendar_connector.msal.ConfidentialClientApplication')
    @patch('os.path.exists')
    def test_authenticate_failure(self, mock_exists, mock_msal_class, mock_msal_app):
        """Test authentication failure."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_exists.return_value = False
        mock_msal_class.return_value = mock_msal_app
        mock_msal_app.get_accounts.return_value = []  # No cached accounts
        mock_msal_app.acquire_token_for_client.return_value = {
            "error": "invalid_client",
            "error_description": "Invalid client credentials"
        }

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )

        with pytest.raises(RuntimeError, match="authentication failed"):
            connector.authenticate()


class TestOutlookCalendarConnectorFetchEvents:
    """Test Outlook Calendar connector fetch events functionality."""

    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_fetch_events_basic(self, mock_get, mock_access_token):
        """Test fetching events from primary calendar."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "value": [
                {
                    "id": "event-1",
                    "subject": "Team Meeting",
                    "bodyPreview": "Discuss project updates",
                    "start": {"dateTime": "2025-10-27T10:00:00", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-10-27T11:00:00", "timeZone": "UTC"},
                    "location": {"displayName": "Conference Room A"},
                    "attendees": [
                        {"emailAddress": {"address": "alice@example.com"}},
                        {"emailAddress": {"address": "bob@example.com"}}
                    ],
                    "isAllDay": False,
                    "recurrence": None
                }
            ]
        }
        mock_get.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        events = await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 1
        assert events[0].summary == "Team Meeting"
        assert events[0].location == "Conference Room A"
        assert len(events[0].attendees) == 2

    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_fetch_events_with_calendar_id(self, mock_get, mock_access_token):
        """Test fetching events from specific calendar."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"value": []}
        mock_get.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            calendar_id="custom-calendar-id"
        )

        # Verify calendar_id was used in the URL
        call_args = mock_get.call_args
        assert "custom-calendar-id" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_events_not_authenticated(self):
        """Test fetch events without authentication raises error."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.fetch_events(
                time_min=datetime(2025, 10, 27, 0, 0, 0),
                time_max=datetime(2025, 10, 28, 0, 0, 0)
            )

    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_fetch_events_with_pagination(self, mock_get, mock_access_token):
        """Test fetching events with pagination."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        # Mock paginated responses
        mock_response_1 = Mock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            "value": [
                {
                    "id": "event-1",
                    "subject": "Event 1",
                    "start": {"dateTime": "2025-10-27T10:00:00", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-10-27T11:00:00", "timeZone": "UTC"},
                    "attendees": []
                }
            ],
            "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/calendar/calendarView?$skip=1"
        }

        mock_response_2 = Mock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            "value": [
                {
                    "id": "event-2",
                    "subject": "Event 2",
                    "start": {"dateTime": "2025-10-27T14:00:00", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-10-27T15:00:00", "timeZone": "UTC"},
                    "attendees": []
                }
            ]
        }

        mock_get.side_effect = [mock_response_1, mock_response_2]

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        events = await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 2
        assert events[0].summary == "Event 1"
        assert events[1].summary == "Event 2"

    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_fetch_events_with_recurring_event(self, mock_get, mock_access_token):
        """Test fetching recurring events."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "value": [
                {
                    "id": "event-recurring",
                    "subject": "Weekly Standup",
                    "start": {"dateTime": "2025-10-27T09:00:00", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-10-27T09:30:00", "timeZone": "UTC"},
                    "attendees": [],
                    "recurrence": {
                        "pattern": {
                            "type": "weekly",
                            "interval": 1,
                            "daysOfWeek": ["monday"]
                        }
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        events = await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 1
        assert events[0].is_recurring is True
        assert events[0].recurrence_rule is not None


class TestOutlookCalendarConnectorCreateEvent:
    """Test Outlook Calendar connector create event functionality."""

    @pytest.mark.asyncio
    @patch('requests.post')
    async def test_create_event_basic(self, mock_post, mock_access_token):
        """Test creating a basic event."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "new-event-123"}
        mock_post.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        event_id = await connector.create_event(
            summary="New Meeting",
            start_time=datetime(2025, 10, 27, 10, 0, 0),
            end_time=datetime(2025, 10, 27, 11, 0, 0)
        )

        assert event_id == "new-event-123"
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch('requests.post')
    async def test_create_event_with_attendees(self, mock_post, mock_access_token):
        """Test creating event with attendees."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "event-with-attendees"}
        mock_post.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        event_id = await connector.create_event(
            summary="Team Meeting",
            start_time=datetime(2025, 10, 27, 10, 0, 0),
            end_time=datetime(2025, 10, 27, 11, 0, 0),
            attendees=["alice@example.com", "bob@example.com"]
        )

        assert event_id == "event-with-attendees"

        # Verify attendees in request body
        call_args = mock_post.call_args
        body = call_args.kwargs['json']
        assert len(body['attendees']) == 2

    @pytest.mark.asyncio
    async def test_create_event_not_authenticated(self):
        """Test create event without authentication raises error."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.create_event(
                summary="Test Event",
                start_time=datetime(2025, 10, 27, 10, 0, 0),
                end_time=datetime(2025, 10, 27, 11, 0, 0)
            )


class TestOutlookCalendarConnectorUpdateEvent:
    """Test Outlook Calendar connector update event functionality."""

    @pytest.mark.asyncio
    @patch('requests.patch')
    async def test_update_event(self, mock_patch, mock_access_token):
        """Test updating an event."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 200
        mock_patch.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        await connector.update_event(
            event_id="event-123",
            summary="Updated Meeting Title"
        )

        mock_patch.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_event_not_authenticated(self):
        """Test update event without authentication raises error."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.update_event(
                event_id="event-123",
                summary="Updated Title"
            )


class TestOutlookCalendarConnectorDeleteEvent:
    """Test Outlook Calendar connector delete event functionality."""

    @pytest.mark.asyncio
    @patch('requests.delete')
    async def test_delete_event(self, mock_delete, mock_access_token):
        """Test deleting an event."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 204
        mock_delete.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        await connector.delete_event(event_id="event-123")

        mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_event_not_authenticated(self):
        """Test delete event without authentication raises error."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.delete_event(event_id="event-123")


class TestOutlookCalendarConnectorListCalendars:
    """Test Outlook Calendar connector list calendars functionality."""

    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_list_calendars(self, mock_get, mock_access_token):
        """Test listing all accessible calendars."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "value": [
                {"id": "cal-1", "name": "Primary Calendar"},
                {"id": "cal-2", "name": "Work Calendar"}
            ]
        }
        mock_get.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        calendars = await connector.list_calendars()

        assert len(calendars) == 2
        assert calendars[0]["name"] == "Primary Calendar"
        assert calendars[1]["name"] == "Work Calendar"

    @pytest.mark.asyncio
    async def test_list_calendars_not_authenticated(self):
        """Test list calendars without authentication raises error."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.list_calendars()


class TestOutlookCalendarConnectorErrorHandling:
    """Test Outlook Calendar connector error handling."""

    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_fetch_events_api_error(self, mock_get, mock_access_token):
        """Test handling API errors in fetch_events."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        with pytest.raises(Exception):
            await connector.fetch_events(
                time_min=datetime(2025, 10, 27, 0, 0, 0),
                time_max=datetime(2025, 10, 28, 0, 0, 0)
            )

    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_fetch_events_parse_error(self, mock_get, mock_access_token):
        """Test handling event parsing errors."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        # Mock response with malformed event data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "value": [
                {"id": "bad-event"},  # Missing required fields
                {
                    "id": "good-event",
                    "subject": "Valid Event",
                    "start": {"dateTime": "2025-10-27T10:00:00", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-10-27T11:00:00", "timeZone": "UTC"},
                    "attendees": []
                }
            ]
        }
        mock_get.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        # Should skip malformed event and return valid one
        events = await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 1
        assert events[0].summary == "Valid Event"

    @pytest.mark.asyncio
    @patch('requests.post')
    async def test_create_event_api_error(self, mock_post, mock_access_token):
        """Test handling API errors in create_event."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_response.raise_for_status.side_effect = Exception("Forbidden")
        mock_post.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        with pytest.raises(Exception):
            await connector.create_event(
                summary="Test Event",
                start_time=datetime(2025, 10, 27, 10, 0, 0),
                end_time=datetime(2025, 10, 27, 11, 0, 0)
            )

    @pytest.mark.asyncio
    @patch('requests.patch')
    async def test_update_event_api_error(self, mock_patch, mock_access_token):
        """Test handling API errors in update_event."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = Exception("Not Found")
        mock_patch.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        with pytest.raises(Exception):
            await connector.update_event(
                event_id="nonexistent",
                summary="Updated Title"
            )

    @pytest.mark.asyncio
    @patch('requests.delete')
    async def test_delete_event_api_error(self, mock_delete, mock_access_token):
        """Test handling API errors in delete_event."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = Exception("Not Found")
        mock_delete.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        with pytest.raises(Exception):
            await connector.delete_event(event_id="nonexistent")

    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_list_calendars_api_error(self, mock_get, mock_access_token):
        """Test handling API errors in list_calendars."""
        from src.connectors.calendar.outlook_calendar_connector import OutlookCalendarConnector

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = Exception("Server Error")
        mock_get.return_value = mock_response

        connector = OutlookCalendarConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        connector.access_token = mock_access_token

        with pytest.raises(Exception):
            await connector.list_calendars()

    def test_get_outlook_calendar_connector_singleton(self):
        """Test get_outlook_calendar_connector returns singleton instance."""
        from src.connectors.calendar.outlook_calendar_connector import get_outlook_calendar_connector
        import src.connectors.calendar.outlook_calendar_connector as ocal_module

        # Reset singleton for this test
        ocal_module._outlook_calendar_connector_instance = None

        # First call creates instance
        connector1 = get_outlook_calendar_connector(
            client_id="client1",
            client_secret="secret1",
            tenant_id="tenant1"
        )
        assert connector1 is not None
        assert connector1.client_id == "client1"

        # Second call returns same instance
        connector2 = get_outlook_calendar_connector(
            client_id="client2",
            client_secret="secret2",
            tenant_id="tenant2"
        )
        assert connector2 is connector1

        # Cleanup
        ocal_module._outlook_calendar_connector_instance = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

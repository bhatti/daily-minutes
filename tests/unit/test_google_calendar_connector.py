#!/usr/bin/env python3
"""Unit tests for Google Calendar connector.

Tests the GoogleCalendarConnector using mocks for the Google Calendar API.
Follows TDD approach - tests written first, implementation second.

Run with: pytest tests/unit/test_google_calendar_connector.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List

from src.connectors.calendar.google_calendar_connector import GoogleCalendarConnector
from src.core.models import CalendarEvent


@pytest.fixture
def mock_calendar_service():
    """Create a mock Google Calendar service."""
    service = Mock()

    # Mock events().list() for fetching events
    list_result = Mock()
    list_result.execute.return_value = {
        'items': [
            {
                'id': 'event1',
                'summary': 'Team Meeting',
                'description': 'Weekly sync meeting',
                'start': {'dateTime': '2025-10-27T10:00:00Z'},
                'end': {'dateTime': '2025-10-27T11:00:00Z'},
                'location': 'Conference Room A',
                'attendees': [
                    {'email': 'attendee1@example.com'},
                    {'email': 'attendee2@example.com'}
                ],
                'recurrence': None
            }
        ],
        'nextPageToken': None
    }
    service.events.return_value.list.return_value = list_result

    # Mock events().insert() for creating events
    insert_result = Mock()
    insert_result.execute.return_value = {
        'id': 'new_event_123',
        'summary': 'New Event',
        'start': {'dateTime': '2025-10-28T14:00:00Z'},
        'end': {'dateTime': '2025-10-28T15:00:00Z'}
    }
    service.events.return_value.insert.return_value = insert_result

    # Mock events().update() for updating events
    update_result = Mock()
    update_result.execute.return_value = {
        'id': 'event1',
        'summary': 'Updated Meeting',
        'start': {'dateTime': '2025-10-27T10:00:00Z'},
        'end': {'dateTime': '2025-10-27T11:00:00Z'}
    }
    service.events.return_value.update.return_value = update_result

    # Mock events().delete() for deleting events
    delete_result = Mock()
    delete_result.execute.return_value = {}
    service.events.return_value.delete.return_value = delete_result

    # Mock calendarList().list() for listing calendars
    calendar_list_result = Mock()
    calendar_list_result.execute.return_value = {
        'items': [
            {
                'id': 'primary',
                'summary': 'Primary Calendar',
                'primary': True
            },
            {
                'id': 'work@example.com',
                'summary': 'Work Calendar',
                'primary': False
            }
        ]
    }
    service.calendarList.return_value.list.return_value = calendar_list_result

    return service


@pytest.fixture
def mock_credentials():
    """Create mock Google credentials."""
    creds = Mock()
    creds.valid = True
    creds.expired = False
    creds.refresh_token = "mock_refresh_token"
    creds.token = "mock_access_token"
    return creds


class TestGoogleCalendarConnectorInitialization:
    """Tests for GoogleCalendarConnector initialization."""

    def test_init_with_credentials_file(self):
        """Test initialization with credentials file path."""
        connector = GoogleCalendarConnector(
            credentials_file="path/to/credentials.json"
        )

        assert connector.credentials_file == "path/to/credentials.json"
        assert connector.is_authenticated is False
        assert connector.service is None

    def test_init_with_token_file(self):
        """Test initialization with token file path."""
        connector = GoogleCalendarConnector(
            credentials_file="credentials.json",
            token_file="custom_token.json"
        )

        assert connector.token_file == "custom_token.json"

    def test_init_default_values(self):
        """Test default initialization values."""
        connector = GoogleCalendarConnector(
            credentials_file="credentials.json"
        )

        assert connector.scopes == ['https://www.googleapis.com/auth/calendar.readonly']
        assert connector.is_authenticated is False

    def test_init_without_credentials_file(self):
        """Test initialization without credentials file raises error."""
        with pytest.raises(ValueError, match="credentials_file is required"):
            GoogleCalendarConnector(credentials_file="")

        with pytest.raises(ValueError, match="credentials_file is required"):
            GoogleCalendarConnector(credentials_file=None)


class TestGoogleCalendarConnectorAuthentication:
    """Tests for Google Calendar authentication."""

    @patch('src.connectors.calendar.google_calendar_connector.build')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('pickle.load')
    @patch('pickle.dump')
    @patch('os.makedirs')
    @patch('google.auth.transport.requests.Request')
    def test_authenticate_with_existing_token(
        self, mock_request, mock_makedirs, mock_pickle_dump, mock_pickle_load, mock_open, mock_exists, mock_build, mock_credentials
    ):
        """Test authentication with existing valid token."""
        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_credentials
        mock_build.return_value = Mock()

        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.authenticate()

        assert connector.is_authenticated is True
        assert connector.credentials is not None
        mock_pickle_load.assert_called_once()

    @patch('src.connectors.calendar.google_calendar_connector.build')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('pickle.load')
    @patch('pickle.dump')
    @patch('os.makedirs')
    @patch('google.auth.transport.requests.Request')
    def test_authenticate_refresh_expired_token(
        self, mock_request, mock_makedirs, mock_pickle_dump, mock_pickle_load, mock_open, mock_exists, mock_build, mock_credentials
    ):
        """Test automatic token refresh when expired."""
        # Token exists but is expired
        mock_credentials.valid = False
        mock_credentials.expired = True
        mock_credentials.refresh_token = "valid_refresh_token"

        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_credentials
        mock_build.return_value = Mock()

        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.authenticate()

        mock_credentials.refresh.assert_called_once()
        assert connector.is_authenticated is True

    @patch('src.connectors.calendar.google_calendar_connector.build')
    @patch('os.path.exists')
    @patch('google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('pickle.dump')
    @patch('os.makedirs')
    def test_authenticate_new_flow(
        self, mock_makedirs, mock_pickle_dump, mock_open, mock_flow_class, mock_exists, mock_build, mock_credentials
    ):
        """Test authentication with new OAuth flow."""
        # Token doesn't exist
        def exists_side_effect(path):
            if 'token' in path:
                return False
            return True

        mock_exists.side_effect = exists_side_effect
        mock_build.return_value = Mock()

        # Mock the flow
        mock_flow = Mock()
        mock_flow.run_local_server.return_value = mock_credentials
        mock_flow_class.return_value = mock_flow

        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.authenticate()

        mock_flow.run_local_server.assert_called_once()
        assert connector.is_authenticated is True

    @patch('src.connectors.calendar.google_calendar_connector.build')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('pickle.load')
    @patch('pickle.dump')
    @patch('os.makedirs')
    @patch('google.auth.transport.requests.Request')
    def test_authenticate_with_invalid_credentials_no_refresh(
        self, mock_request, mock_makedirs, mock_pickle_dump, mock_pickle_load, mock_open, mock_exists, mock_build, mock_credentials
    ):
        """Test authentication with expired credentials but no refresh token."""
        # Token exists but is expired and has no refresh token
        mock_credentials.valid = False
        mock_credentials.expired = True
        mock_credentials.refresh_token = None  # No refresh token!

        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_credentials
        mock_build.return_value = Mock()

        # Mock flow for new authentication
        with patch('google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file') as mock_flow_class:
            mock_flow = Mock()
            mock_flow.run_local_server.return_value = mock_credentials
            mock_flow_class.return_value = mock_flow

            connector = GoogleCalendarConnector(credentials_file="credentials.json")
            connector.authenticate()

            # Should run new OAuth flow since credentials can't be refreshed
            mock_flow.run_local_server.assert_called_once()
            assert connector.is_authenticated is True


class TestGoogleCalendarConnectorFetchEvents:
    """Tests for fetching calendar events."""

    @pytest.mark.asyncio
    async def test_fetch_events_basic(self, mock_calendar_service, mock_credentials):
        """Test basic event fetching."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = mock_calendar_service
        connector.is_authenticated = True

        events = await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 1
        assert isinstance(events[0], CalendarEvent)
        assert events[0].summary == "Team Meeting"
        assert events[0].location == "Conference Room A"
        assert len(events[0].attendees) == 2

    @pytest.mark.asyncio
    async def test_fetch_events_with_calendar_id(self, mock_calendar_service, mock_credentials):
        """Test fetching events from specific calendar."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = mock_calendar_service
        connector.is_authenticated = True

        events = await connector.fetch_events(
            calendar_id="work@example.com",
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        # Verify calendar_id was passed to API call
        mock_calendar_service.events.return_value.list.assert_called()
        call_kwargs = mock_calendar_service.events.return_value.list.call_args[1]
        assert call_kwargs['calendarId'] == "work@example.com"

    @pytest.mark.asyncio
    async def test_fetch_events_with_max_results(self, mock_calendar_service, mock_credentials):
        """Test fetching events with max_results limit."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = mock_calendar_service
        connector.is_authenticated = True

        events = await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0),
            max_results=10
        )

        call_kwargs = mock_calendar_service.events.return_value.list.call_args[1]
        assert call_kwargs['maxResults'] == 10

    @pytest.mark.asyncio
    async def test_fetch_events_pagination(self, mock_calendar_service, mock_credentials):
        """Test handling pagination with nextPageToken."""
        # Mock multiple pages
        page1 = Mock()
        page1.execute.return_value = {
            'items': [
                {
                    'id': 'event1',
                    'summary': 'Event 1',
                    'start': {'dateTime': '2025-10-27T10:00:00Z'},
                    'end': {'dateTime': '2025-10-27T11:00:00Z'}
                }
            ],
            'nextPageToken': 'token123'
        }

        page2 = Mock()
        page2.execute.return_value = {
            'items': [
                {
                    'id': 'event2',
                    'summary': 'Event 2',
                    'start': {'dateTime': '2025-10-27T14:00:00Z'},
                    'end': {'dateTime': '2025-10-27T15:00:00Z'}
                }
            ],
            'nextPageToken': None
        }

        mock_calendar_service.events.return_value.list.side_effect = [page1, page2]

        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = mock_calendar_service
        connector.is_authenticated = True

        events = await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 2
        assert events[0].summary == "Event 1"
        assert events[1].summary == "Event 2"

    @pytest.mark.asyncio
    async def test_fetch_events_not_authenticated(self):
        """Test fetching events when not authenticated raises error."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.fetch_events(
                time_min=datetime(2025, 10, 27, 0, 0, 0),
                time_max=datetime(2025, 10, 28, 0, 0, 0)
            )

    @pytest.mark.asyncio
    async def test_fetch_events_with_recurring_event(self, mock_credentials):
        """Test fetching recurring events."""
        service = Mock()
        list_result = Mock()
        list_result.execute.return_value = {
            'items': [
                {
                    'id': 'recurring1',
                    'summary': 'Daily Standup',
                    'start': {'dateTime': '2025-10-27T09:00:00Z'},
                    'end': {'dateTime': '2025-10-27T09:15:00Z'},
                    'recurrence': ['RRULE:FREQ=DAILY']
                }
            ],
            'nextPageToken': None
        }
        service.events.return_value.list.return_value = list_result

        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = service
        connector.is_authenticated = True

        events = await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 1
        assert events[0].is_recurring is True
        assert events[0].recurrence_rule == 'RRULE:FREQ=DAILY'


class TestGoogleCalendarConnectorCreateEvent:
    """Tests for creating calendar events."""

    @pytest.mark.asyncio
    async def test_create_event_basic(self, mock_calendar_service, mock_credentials):
        """Test basic event creation."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = mock_calendar_service
        connector.is_authenticated = True

        event_id = await connector.create_event(
            summary="New Meeting",
            start_time=datetime(2025, 10, 28, 14, 0, 0),
            end_time=datetime(2025, 10, 28, 15, 0, 0)
        )

        assert event_id == "new_event_123"
        mock_calendar_service.events.return_value.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_event_with_attendees(self, mock_calendar_service, mock_credentials):
        """Test creating event with attendees."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = mock_calendar_service
        connector.is_authenticated = True

        event_id = await connector.create_event(
            summary="Team Sync",
            start_time=datetime(2025, 10, 28, 14, 0, 0),
            end_time=datetime(2025, 10, 28, 15, 0, 0),
            attendees=["user1@example.com", "user2@example.com"]
        )

        assert event_id == "new_event_123"
        call_kwargs = mock_calendar_service.events.return_value.insert.call_args[1]
        assert 'attendees' in call_kwargs['body']
        assert len(call_kwargs['body']['attendees']) == 2

    @pytest.mark.asyncio
    async def test_create_event_not_authenticated(self):
        """Test creating event when not authenticated raises error."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.create_event(
                summary="Meeting",
                start_time=datetime(2025, 10, 28, 14, 0, 0),
                end_time=datetime(2025, 10, 28, 15, 0, 0)
            )


class TestGoogleCalendarConnectorUpdateEvent:
    """Tests for updating calendar events."""

    @pytest.mark.asyncio
    async def test_update_event(self, mock_calendar_service, mock_credentials):
        """Test updating an existing event."""
        # Mock the get() call to return a proper dict (not a Mock)
        get_result = Mock()
        get_result.execute.return_value = {
            'id': 'event1',
            'summary': 'Old Meeting Title',
            'start': {'dateTime': '2025-10-27T10:00:00Z'},
            'end': {'dateTime': '2025-10-27T11:00:00Z'}
        }
        mock_calendar_service.events.return_value.get.return_value = get_result

        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = mock_calendar_service
        connector.is_authenticated = True

        await connector.update_event(
            event_id="event1",
            summary="Updated Meeting Title"
        )

        mock_calendar_service.events.return_value.update.assert_called_once()
        call_kwargs = mock_calendar_service.events.return_value.update.call_args[1]
        assert call_kwargs['eventId'] == "event1"
        assert call_kwargs['body']['summary'] == "Updated Meeting Title"

    @pytest.mark.asyncio
    async def test_update_event_not_authenticated(self):
        """Test updating event when not authenticated raises error."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.update_event(
                event_id="event1",
                summary="Updated"
            )


class TestGoogleCalendarConnectorDeleteEvent:
    """Tests for deleting calendar events."""

    @pytest.mark.asyncio
    async def test_delete_event(self, mock_calendar_service, mock_credentials):
        """Test deleting an event."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = mock_calendar_service
        connector.is_authenticated = True

        await connector.delete_event(event_id="event1")

        mock_calendar_service.events.return_value.delete.assert_called_once()
        call_kwargs = mock_calendar_service.events.return_value.delete.call_args[1]
        assert call_kwargs['eventId'] == "event1"

    @pytest.mark.asyncio
    async def test_delete_event_not_authenticated(self):
        """Test deleting event when not authenticated raises error."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.delete_event(event_id="event1")


class TestGoogleCalendarConnectorListCalendars:
    """Tests for listing calendars."""

    @pytest.mark.asyncio
    async def test_list_calendars(self, mock_calendar_service, mock_credentials):
        """Test listing all calendars."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = mock_calendar_service
        connector.is_authenticated = True

        calendars = await connector.list_calendars()

        assert len(calendars) == 2
        assert calendars[0]['id'] == 'primary'
        assert calendars[1]['id'] == 'work@example.com'

    @pytest.mark.asyncio
    async def test_list_calendars_not_authenticated(self):
        """Test listing calendars when not authenticated raises error."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.list_calendars()


class TestGoogleCalendarConnectorErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_fetch_events_api_error(self, mock_calendar_service, mock_credentials):
        """Test handling API errors gracefully."""
        from googleapiclient.errors import HttpError

        # Mock API error
        mock_calendar_service.events.return_value.list.side_effect = HttpError(
            resp=Mock(status=403),
            content=b'Rate limit exceeded'
        )

        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = mock_calendar_service
        connector.is_authenticated = True

        with pytest.raises(HttpError):
            await connector.fetch_events(
                time_min=datetime(2025, 10, 27, 0, 0, 0),
                time_max=datetime(2025, 10, 28, 0, 0, 0)
            )

    @pytest.mark.asyncio
    async def test_parse_event_missing_fields(self, mock_credentials):
        """Test parsing event with missing optional fields."""
        service = Mock()
        list_result = Mock()
        list_result.execute.return_value = {
            'items': [
                {
                    'id': 'minimal_event',
                    'summary': 'Minimal Event',
                    'start': {'dateTime': '2025-10-27T10:00:00Z'},
                    'end': {'dateTime': '2025-10-27T11:00:00Z'}
                    # Missing: description, location, attendees, recurrence
                }
            ],
            'nextPageToken': None
        }
        service.events.return_value.list.return_value = list_result

        connector = GoogleCalendarConnector(credentials_file="credentials.json")
        connector.credentials = mock_credentials
        connector.service = service
        connector.is_authenticated = True

        events = await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 1
        assert events[0].summary == "Minimal Event"
        assert events[0].description is None
        assert events[0].location is None
        assert len(events[0].attendees) == 0
        assert events[0].is_recurring is False

    @pytest.mark.asyncio
    async def test_fetch_events_parse_error(self, mock_credentials):
        """Test fetch_events continues when event parsing fails."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        # Create mock service with malformed event
        service = Mock()
        mock_events = Mock()
        mock_list = Mock()
        mock_execute = Mock()

        # First event is malformed (missing required fields), second is valid
        mock_execute.return_value = {
            'items': [
                {'id': 'bad-event'},  # Missing start/end times - will cause parsing error
                {
                    'id': 'event-2',
                    'summary': 'Valid Event',
                    'start': {'dateTime': '2025-10-27T10:00:00Z'},
                    'end': {'dateTime': '2025-10-27T11:00:00Z'}
                }
            ]
        }

        mock_list.return_value.execute = mock_execute
        mock_events.return_value.list = mock_list
        service.events = mock_events

        connector.credentials = mock_credentials
        connector.service = service
        connector.is_authenticated = True

        # Should skip bad event and return valid one
        events = await connector.fetch_events(
            time_min=datetime(2025, 10, 27, 0, 0, 0),
            time_max=datetime(2025, 10, 28, 0, 0, 0)
        )

        assert len(events) == 1
        assert events[0].summary == "Valid Event"

    @pytest.mark.asyncio
    async def test_fetch_events_general_exception(self, mock_credentials):
        """Test fetch_events handles general exceptions."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        service = Mock()
        service.events.return_value.list.side_effect = Exception("Network error")

        connector.credentials = mock_credentials
        connector.service = service
        connector.is_authenticated = True

        with pytest.raises(Exception, match="Network error"):
            await connector.fetch_events(
                time_min=datetime(2025, 10, 27, 0, 0, 0),
                time_max=datetime(2025, 10, 28, 0, 0, 0)
            )

    @pytest.mark.asyncio
    async def test_create_event_with_optional_fields(self, mock_credentials):
        """Test creating event with all optional fields."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        service = Mock()
        mock_insert = Mock()
        mock_insert.return_value.execute.return_value = {'id': 'new-event-123'}
        service.events.return_value.insert = mock_insert

        connector.credentials = mock_credentials
        connector.service = service
        connector.is_authenticated = True

        event_id = await connector.create_event(
            summary="Team Meeting",
            start_time=datetime(2025, 10, 27, 10, 0, 0),
            end_time=datetime(2025, 10, 27, 11, 0, 0),
            description="Discuss project updates",
            location="Conference Room A",
            attendees=["alice@example.com", "bob@example.com"]
        )

        assert event_id == "new-event-123"

        # Verify the event body included all optional fields
        call_args = mock_insert.call_args
        event_body = call_args.kwargs['body']
        assert event_body['description'] == "Discuss project updates"
        assert event_body['location'] == "Conference Room A"
        assert len(event_body['attendees']) == 2
        assert event_body['attendees'][0]['email'] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_create_event_http_error(self, mock_credentials):
        """Test create_event handles HttpError."""
        from googleapiclient.errors import HttpError
        from unittest.mock import Mock as MockResponse

        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        service = Mock()

        # Create mock HttpError
        resp = MockResponse()
        resp.status = 403
        http_error = HttpError(resp=resp, content=b'Forbidden')

        service.events.return_value.insert.return_value.execute.side_effect = http_error

        connector.credentials = mock_credentials
        connector.service = service
        connector.is_authenticated = True

        with pytest.raises(HttpError):
            await connector.create_event(
                summary="Test Event",
                start_time=datetime(2025, 10, 27, 10, 0, 0),
                end_time=datetime(2025, 10, 27, 11, 0, 0)
            )

    @pytest.mark.asyncio
    async def test_update_event_with_all_fields(self, mock_credentials):
        """Test updating event with all fields."""
        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        service = Mock()

        # Mock existing event
        existing_event = {
            'id': 'event-123',
            'summary': 'Old Summary',
            'start': {'dateTime': '2025-10-27T10:00:00Z'},
            'end': {'dateTime': '2025-10-27T11:00:00Z'}
        }

        service.events.return_value.get.return_value.execute.return_value = existing_event
        service.events.return_value.update.return_value.execute.return_value = {}

        connector.credentials = mock_credentials
        connector.service = service
        connector.is_authenticated = True

        await connector.update_event(
            event_id="event-123",
            summary="New Summary",
            start_time=datetime(2025, 10, 27, 14, 0, 0),
            end_time=datetime(2025, 10, 27, 15, 0, 0),
            description="Updated description",
            location="New Location"
        )

        # Verify update was called
        update_call = service.events.return_value.update.call_args
        updated_event = update_call.kwargs['body']
        assert updated_event['summary'] == "New Summary"
        assert updated_event['description'] == "Updated description"
        assert updated_event['location'] == "New Location"

    @pytest.mark.asyncio
    async def test_update_event_http_error(self, mock_credentials):
        """Test update_event handles HttpError."""
        from googleapiclient.errors import HttpError
        from unittest.mock import Mock as MockResponse

        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        service = Mock()

        # Mock get succeeds but update fails
        existing_event = {
            'id': 'event-123',
            'summary': 'Test Event'
        }
        service.events.return_value.get.return_value.execute.return_value = existing_event

        # Create mock HttpError for update
        resp = MockResponse()
        resp.status = 404
        http_error = HttpError(resp=resp, content=b'Not Found')
        service.events.return_value.update.return_value.execute.side_effect = http_error

        connector.credentials = mock_credentials
        connector.service = service
        connector.is_authenticated = True

        with pytest.raises(HttpError):
            await connector.update_event(
                event_id="event-123",
                summary="New Summary"
            )

    @pytest.mark.asyncio
    async def test_delete_event_http_error(self, mock_credentials):
        """Test delete_event handles HttpError."""
        from googleapiclient.errors import HttpError
        from unittest.mock import Mock as MockResponse

        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        service = Mock()

        # Create mock HttpError
        resp = MockResponse()
        resp.status = 404
        http_error = HttpError(resp=resp, content=b'Not Found')
        service.events.return_value.delete.return_value.execute.side_effect = http_error

        connector.credentials = mock_credentials
        connector.service = service
        connector.is_authenticated = True

        with pytest.raises(HttpError):
            await connector.delete_event(event_id="nonexistent-event")

    @pytest.mark.asyncio
    async def test_list_calendars_http_error(self, mock_credentials):
        """Test list_calendars handles HttpError."""
        from googleapiclient.errors import HttpError
        from unittest.mock import Mock as MockResponse

        connector = GoogleCalendarConnector(credentials_file="credentials.json")

        service = Mock()

        # Create mock HttpError
        resp = MockResponse()
        resp.status = 500
        http_error = HttpError(resp=resp, content=b'Internal Server Error')
        service.calendarList.return_value.list.return_value.execute.side_effect = http_error

        connector.credentials = mock_credentials
        connector.service = service
        connector.is_authenticated = True

        with pytest.raises(HttpError):
            await connector.list_calendars()

    def test_get_google_calendar_connector_singleton(self):
        """Test get_google_calendar_connector returns singleton instance."""
        from src.connectors.calendar.google_calendar_connector import get_google_calendar_connector, _google_calendar_connector_instance

        # Reset singleton for this test
        import src.connectors.calendar.google_calendar_connector as gcal_module
        gcal_module._google_calendar_connector_instance = None

        # First call creates instance
        connector1 = get_google_calendar_connector(credentials_file="creds.json")
        assert connector1 is not None
        assert connector1.credentials_file == "creds.json"

        # Second call returns same instance
        connector2 = get_google_calendar_connector(credentials_file="different.json")
        assert connector2 is connector1

        # Cleanup - reset singleton
        gcal_module._google_calendar_connector_instance = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

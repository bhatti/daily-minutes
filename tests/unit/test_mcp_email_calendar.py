"""Unit tests for MCP Server email and calendar tools.

Tests the email/calendar integration in MCP server using mocks.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.services.mcp_server import MCPServer, MCPResponse
from tests.load_mock_data import load_mock_emails
from tests.mock_calendar_data import generate_mock_calendar_events


class TestMCPEmailTools:
    """Test email tools in MCP server."""

    @pytest.mark.asyncio
    async def test_fetch_emails_with_gmail_connector(self):
        """Test fetching emails when Gmail connector is available."""
        # Create mock emails
        mock_emails = load_mock_emails(count=10)

        with patch('src.services.mcp_server.GmailConnector') as MockGmail:
            # Mock the connector
            mock_connector = MagicMock()
            mock_connector.is_authenticated = True
            mock_connector.fetch_unread_emails = AsyncMock(return_value=mock_emails)
            MockGmail.return_value = mock_connector

            # Mock os.path.exists to simulate credentials present
            with patch('os.path.exists', return_value=True):
                # Create MCP server (will initialize Gmail connector)
                server = MCPServer()

                # Verify Gmail connector was initialized
                assert server.connectors.get("gmail") is not None
                assert "fetch_emails" in server.tools

                # Execute fetch_emails tool
                response = await server.execute_tool("fetch_emails", {
                    "max_results": 10,
                    "query": "is:unread"
                })

                # Verify response
                assert response.success is True
                assert len(response.data) == 10
                assert all("subject" in email for email in response.data)
                assert all("sender" in email for email in response.data)
                assert all("importance_score" in email for email in response.data)
                assert all("has_action_items" in email for email in response.data)

    @pytest.mark.asyncio
    async def test_fetch_emails_without_credentials(self):
        """Test that email tool is not registered without credentials."""
        with patch('os.path.exists', return_value=False):
            # Create MCP server (credentials missing)
            server = MCPServer()

            # Verify Gmail connector was not registered as a tool
            assert "fetch_emails" not in server.tools

    @pytest.mark.asyncio
    async def test_fetch_emails_filters_by_importance(self):
        """Test that emails can be filtered by importance score."""
        # Create mock emails with varying importance
        mock_emails = load_mock_emails(count=20)

        with patch('src.services.mcp_server.GmailConnector') as MockGmail:
            mock_connector = MagicMock()
            mock_connector.is_authenticated = True
            mock_connector.fetch_unread_emails = AsyncMock(return_value=mock_emails)
            MockGmail.return_value = mock_connector

            with patch('os.path.exists', return_value=True):
                server = MCPServer()

                # Execute with default params
                response = await server.execute_tool("fetch_emails", {
                    "max_results": 20
                })

                assert response.success is True
                assert len(response.data) <= 20

    @pytest.mark.asyncio
    async def test_fetch_emails_error_handling(self):
        """Test error handling when Gmail connector fails."""
        with patch('src.services.mcp_server.GmailConnector') as MockGmail:
            mock_connector = MagicMock()
            mock_connector.is_authenticated = True
            mock_connector.fetch_unread_emails = AsyncMock(
                side_effect=Exception("Gmail API error")
            )
            MockGmail.return_value = mock_connector

            with patch('os.path.exists', return_value=True):
                server = MCPServer()

                # Execute should handle error gracefully
                response = await server.execute_tool("fetch_emails", {})

                assert response.success is False
                assert "Gmail API error" in response.error


class TestMCPCalendarTools:
    """Test calendar tools in MCP server."""

    @pytest.mark.asyncio
    async def test_fetch_calendar_events_with_connector(self):
        """Test fetching calendar events when Google Calendar connector is available."""
        # Create mock events
        mock_events = generate_mock_calendar_events(count=12, days_ahead=7)

        with patch('src.services.mcp_server.GoogleCalendarConnector') as MockCalendar:
            # Mock the connector
            mock_connector = MagicMock()
            mock_connector.is_authenticated = True
            mock_connector.fetch_events = AsyncMock(return_value=mock_events)
            MockCalendar.return_value = mock_connector

            # Mock os.path.exists to simulate credentials present
            with patch('os.path.exists', return_value=True):
                # Create MCP server (will initialize Calendar connector)
                server = MCPServer()

                # Verify Calendar connector was initialized
                assert server.connectors.get("google_calendar") is not None
                assert "fetch_calendar_events" in server.tools

                # Execute fetch_calendar_events tool
                response = await server.execute_tool("fetch_calendar_events", {
                    "days_ahead": 7,
                    "max_results": 12
                })

                # Verify response
                assert response.success is True
                assert len(response.data) <= 12  # May filter out past events
                assert all("summary" in event for event in response.data)
                assert all("start_time" in event for event in response.data)
                assert all("end_time" in event for event in response.data)
                assert all("importance_score" in event for event in response.data)
                assert all("requires_preparation" in event for event in response.data)

    @pytest.mark.asyncio
    async def test_fetch_calendar_events_without_credentials(self):
        """Test that calendar tool is not registered without credentials."""
        with patch('os.path.exists', return_value=False):
            # Create MCP server (credentials missing)
            server = MCPServer()

            # Verify Calendar tool was not registered
            assert "fetch_calendar_events" not in server.tools

    @pytest.mark.asyncio
    async def test_fetch_calendar_events_filters_upcoming(self):
        """Test that only upcoming events are returned."""
        # Create mock events (some in past, some future)
        now = datetime.now()
        mock_events = generate_mock_calendar_events(count=12, days_ahead=7)

        with patch('src.services.mcp_server.GoogleCalendarConnector') as MockCalendar:
            mock_connector = MagicMock()
            mock_connector.is_authenticated = True
            mock_connector.fetch_events = AsyncMock(return_value=mock_events)
            MockCalendar.return_value = mock_connector

            with patch('os.path.exists', return_value=True):
                server = MCPServer()

                response = await server.execute_tool("fetch_calendar_events", {
                    "days_ahead": 7
                })

                assert response.success is True
                # All returned events should be in the future
                for event in response.data:
                    event_time = datetime.fromisoformat(event["start_time"])
                    assert event_time >= now

    @pytest.mark.asyncio
    async def test_fetch_calendar_events_preparation_notes(self):
        """Test that preparation notes are included in response."""
        mock_events = generate_mock_calendar_events(count=12)

        with patch('src.services.mcp_server.GoogleCalendarConnector') as MockCalendar:
            mock_connector = MagicMock()
            mock_connector.is_authenticated = True
            mock_connector.fetch_events = AsyncMock(return_value=mock_events)
            MockCalendar.return_value = mock_connector

            with patch('os.path.exists', return_value=True):
                server = MCPServer()

                response = await server.execute_tool("fetch_calendar_events", {})

                assert response.success is True

                # Find events that require preparation
                prep_events = [e for e in response.data if e["requires_preparation"]]
                assert len(prep_events) > 0

                # Verify preparation notes are included
                for event in prep_events:
                    assert "preparation_notes" in event
                    assert isinstance(event["preparation_notes"], list)


class TestMCPServerIntegration:
    """Integration tests for MCP server with email and calendar."""

    @pytest.mark.asyncio
    async def test_list_tools_includes_email_calendar(self):
        """Test that email and calendar tools are listed when available."""
        with patch('os.path.exists', return_value=True):
            with patch('src.services.mcp_server.GmailConnector') as MockGmail, \
                 patch('src.services.mcp_server.GoogleCalendarConnector') as MockCalendar:

                mock_gmail = MagicMock()
                mock_gmail.is_authenticated = True
                MockGmail.return_value = mock_gmail

                mock_calendar = MagicMock()
                mock_calendar.is_authenticated = True
                MockCalendar.return_value = mock_calendar

                server = MCPServer()

                # List all tools
                tools = await server.list_tools()

                tool_names = [tool["name"] for tool in tools]

                # Verify email and calendar tools are listed
                assert "fetch_emails" in tool_names
                assert "fetch_calendar_events" in tool_names

                # Verify news and weather tools still work
                assert "fetch_hackernews" in tool_names
                assert "get_current_weather" in tool_names

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_credentials(self):
        """Test that server works normally without email/calendar credentials."""
        with patch('os.path.exists', return_value=False):
            # Create server without credentials
            server = MCPServer()

            # Should still have news and weather tools
            tools = await server.list_tools()
            tool_names = [tool["name"] for tool in tools]

            assert "fetch_hackernews" in tool_names
            assert "get_current_weather" in tool_names

            # Email and calendar tools should not be registered
            assert "fetch_emails" not in tool_names
            assert "fetch_calendar_events" not in tool_names

    @pytest.mark.asyncio
    async def test_mcp_protocol_request_handling(self):
        """Test MCP protocol request/response format."""
        mock_emails = load_mock_emails(count=5)

        with patch('src.services.mcp_server.GmailConnector') as MockGmail:
            mock_connector = MagicMock()
            mock_connector.is_authenticated = True
            mock_connector.fetch_unread_emails = AsyncMock(return_value=mock_emails)
            MockGmail.return_value = mock_connector

            with patch('os.path.exists', return_value=True):
                server = MCPServer()

                # Test MCP protocol request
                request = {
                    "method": "tools/call",
                    "params": {
                        "name": "fetch_emails",
                        "arguments": {"max_results": 5}
                    }
                }

                response = await server.handle_request(request)

                assert "success" in response
                assert response["success"] is True
                assert "result" in response
                assert len(response["result"]) == 5


class TestMCPDataConversion:
    """Test email and calendar data conversion to JSON."""

    def test_email_to_dict_conversion(self):
        """Test EmailMessage to dictionary conversion."""
        mock_emails = load_mock_emails(count=1)

        with patch('src.services.mcp_server.GmailConnector') as MockGmail:
            mock_connector = MagicMock()
            mock_connector.is_authenticated = True
            MockGmail.return_value = mock_connector

            with patch('os.path.exists', return_value=True):
                server = MCPServer()

                # Convert email to dict
                email_dict = server._email_to_dict(mock_emails[0])

                # Verify all required fields present
                assert "id" in email_dict
                assert "subject" in email_dict
                assert "sender" in email_dict
                assert "body" in email_dict
                assert "received_at" in email_dict
                assert "importance_score" in email_dict
                assert "has_action_items" in email_dict
                assert "action_items" in email_dict
                assert "is_read" in email_dict

                # Verify data types
                assert isinstance(email_dict["importance_score"], float)
                assert isinstance(email_dict["has_action_items"], bool)
                assert isinstance(email_dict["action_items"], list)

    def test_calendar_event_to_dict_conversion(self):
        """Test CalendarEvent to dictionary conversion."""
        mock_events = generate_mock_calendar_events(count=1)

        with patch('src.services.mcp_server.GoogleCalendarConnector') as MockCalendar:
            mock_connector = MagicMock()
            mock_connector.is_authenticated = True
            MockCalendar.return_value = mock_connector

            with patch('os.path.exists', return_value=True):
                server = MCPServer()

                # Convert event to dict
                event_dict = server._calendar_event_to_dict(mock_events[0])

                # Verify all required fields present
                assert "id" in event_dict
                assert "summary" in event_dict
                assert "description" in event_dict
                assert "start_time" in event_dict
                assert "end_time" in event_dict
                assert "location" in event_dict
                assert "attendees" in event_dict
                assert "importance_score" in event_dict
                assert "requires_preparation" in event_dict
                assert "preparation_notes" in event_dict
                assert "is_focus_time" in event_dict

                # Verify data types
                assert isinstance(event_dict["importance_score"], float)
                assert isinstance(event_dict["requires_preparation"], bool)
                assert isinstance(event_dict["preparation_notes"], list)
                assert isinstance(event_dict["is_focus_time"], bool)

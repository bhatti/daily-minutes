#!/usr/bin/env python3
"""Unit tests for IMAP Connector (TDD approach).

Following TDD: Write tests first, then implement IMAP connector.

Mocking Strategy:
- Mock imaplib.IMAP4_SSL for IMAP connection
- Mock email parsing (email.message_from_bytes)
- Mock IMAP responses and folder structures
- Verify metrics emission and observability

Key Features:
- Generic IMAP support (works with any provider)
- Username/password authentication (no OAuth)
- Fetching unread emails with IMAP search
- Parsing email messages with headers
- Batch operations (mark as read, delete)
- Rate limit handling
- Metrics and activity logging
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from pydantic import ValidationError

# Test constants
MOCK_EMAIL_ID = "1"
MOCK_SENDER = "sender@example.com"
MOCK_SUBJECT = "Test Email Subject"
MOCK_BODY = "This is the email body content."
MOCK_IMAP_SERVER = "imap.example.com"
MOCK_USERNAME = "user@example.com"
MOCK_PASSWORD = "password123"


@pytest.fixture
def mock_imap_connection():
    """Create mock IMAP4_SSL connection."""
    connection = Mock()

    # Mock login response
    connection.login.return_value = ('OK', [b'LOGIN completed'])

    # Mock select (select inbox folder)
    connection.select.return_value = ('OK', [b'10'])  # 10 messages in inbox

    # Mock search (find unread messages)
    connection.search.return_value = ('OK', [b'1 2 3'])  # Message IDs 1, 2, 3

    # Mock fetch (get email content)
    connection.fetch.return_value = ('OK', [(b'1 (RFC822 {1234}', create_mock_email_bytes())])

    # Mock store (mark as read, delete)
    connection.store.return_value = ('OK', [b'Flags set'])

    # Mock expunge (permanently delete)
    connection.expunge.return_value = ('OK', [])

    # Mock logout
    connection.logout.return_value = ('BYE', [b'Logging out'])

    # Mock close
    connection.close.return_value = ('OK', [b'Folder closed'])

    return connection


def create_mock_email_bytes():
    """Create mock email in RFC822 format as bytes."""
    email_content = f"""From: {MOCK_SENDER}
To: {MOCK_USERNAME}
Subject: {MOCK_SUBJECT}
Date: Sat, 26 Oct 2025 10:00:00 +0000
Message-ID: <{MOCK_EMAIL_ID}@example.com>
Content-Type: text/plain; charset=utf-8

{MOCK_BODY}
"""
    return email_content.encode('utf-8')


class TestIMAPConnectorInitialization:
    """Test IMAP connector initialization and configuration."""

    def test_connector_initialization(self):
        """Test IMAPConnector initializes correctly."""
        from src.connectors.email.imap_connector import IMAPConnector

        connector = IMAPConnector(
            imap_server=MOCK_IMAP_SERVER,
            username=MOCK_USERNAME,
            password=MOCK_PASSWORD
        )

        assert connector is not None
        assert connector.imap_server == MOCK_IMAP_SERVER
        assert connector.username == MOCK_USERNAME
        assert connector.is_authenticated is False

    def test_connector_requires_credentials(self):
        """Test IMAPConnector requires IMAP credentials."""
        from src.connectors.email.imap_connector import IMAPConnector

        with pytest.raises(ValueError, match="imap_server.*required"):
            IMAPConnector(imap_server="", username="user", password="pass")

    def test_connector_supports_ssl_and_non_ssl(self):
        """Test IMAPConnector supports both SSL and non-SSL connections."""
        from src.connectors.email.imap_connector import IMAPConnector

        # SSL (default)
        connector_ssl = IMAPConnector(
            imap_server=MOCK_IMAP_SERVER,
            username=MOCK_USERNAME,
            password=MOCK_PASSWORD,
            use_ssl=True
        )
        assert connector_ssl.use_ssl is True

        # Non-SSL
        connector_plain = IMAPConnector(
            imap_server=MOCK_IMAP_SERVER,
            username=MOCK_USERNAME,
            password=MOCK_PASSWORD,
            use_ssl=False,
            port=143
        )
        assert connector_plain.use_ssl is False
        assert connector_plain.port == 143


class TestIMAPAuthentication:
    """Test IMAP authentication flow."""

    @pytest.mark.asyncio
    async def test_authenticate_success(self, mock_imap_connection):
        """Test successful IMAP authentication."""
        from src.connectors.email.imap_connector import IMAPConnector

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )

            await connector.authenticate()

            assert connector.is_authenticated is True
            mock_imap_connection.login.assert_called_once_with(MOCK_USERNAME, MOCK_PASSWORD)

    @pytest.mark.asyncio
    async def test_authenticate_failure_invalid_credentials(self, mock_imap_connection):
        """Test authentication fails with invalid credentials."""
        from src.connectors.email.imap_connector import IMAPConnector
        import imaplib

        # Mock login failure
        mock_imap_connection.login.side_effect = imaplib.IMAP4.error("Authentication failed")

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password="wrong_password"
            )

            with pytest.raises(RuntimeError, match="Authentication failed"):
                await connector.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_connection_error(self):
        """Test authentication fails with connection error."""
        from src.connectors.email.imap_connector import IMAPConnector

        with patch('imaplib.IMAP4_SSL', side_effect=ConnectionRefusedError("Connection refused")):
            connector = IMAPConnector(
                imap_server="invalid.server.com",
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )

            with pytest.raises(RuntimeError, match="Connection.*failed"):
                await connector.authenticate()


class TestIMAPFetchEmails:
    """Test fetching emails from IMAP server."""

    @pytest.mark.asyncio
    async def test_fetch_unread_emails_success(self, mock_imap_connection):
        """Test successfully fetching unread emails."""
        from src.connectors.email.imap_connector import IMAPConnector

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            emails = await connector.fetch_unread_emails(max_results=10)

            assert len(emails) > 0
            assert emails[0].subject == MOCK_SUBJECT
            assert emails[0].sender == MOCK_SENDER
            assert emails[0].body == MOCK_BODY

    @pytest.mark.asyncio
    async def test_fetch_emails_requires_authentication(self):
        """Test fetch_unread_emails raises error if not authenticated."""
        from src.connectors.email.imap_connector import IMAPConnector

        connector = IMAPConnector(
            imap_server=MOCK_IMAP_SERVER,
            username=MOCK_USERNAME,
            password=MOCK_PASSWORD
        )

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.fetch_unread_emails()

    @pytest.mark.asyncio
    async def test_fetch_emails_from_custom_folder(self, mock_imap_connection):
        """Test fetching emails from custom folder (not INBOX)."""
        from src.connectors.email.imap_connector import IMAPConnector

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            await connector.fetch_unread_emails(folder="Sent")

            # Verify select was called with custom folder
            calls = [call[0][0] for call in mock_imap_connection.select.call_args_list]
            assert "Sent" in calls

    @pytest.mark.asyncio
    async def test_fetch_emails_with_limit(self, mock_imap_connection):
        """Test fetching emails respects max_results limit."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Mock search returns 10 messages
        mock_imap_connection.search.return_value = ('OK', [b'1 2 3 4 5 6 7 8 9 10'])

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            emails = await connector.fetch_unread_emails(max_results=5)

            # Should only fetch 5 emails, not all 10
            assert len(emails) <= 5


class TestIMAPEmailParsing:
    """Test parsing IMAP email messages into EmailMessage models."""

    @pytest.mark.asyncio
    async def test_parse_email_with_multipart_body(self, mock_imap_connection):
        """Test parsing email with multipart/alternative body."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Create multipart email
        multipart_email = b"""From: sender@example.com
To: user@example.com
Subject: Multipart Email
Content-Type: multipart/alternative; boundary="boundary123"

--boundary123
Content-Type: text/plain; charset=utf-8

Plain text body

--boundary123
Content-Type: text/html; charset=utf-8

<html><body>HTML body</body></html>

--boundary123--
"""
        mock_imap_connection.fetch.return_value = ('OK', [(b'1 (RFC822 {1234}', multipart_email)])

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            emails = await connector.fetch_unread_emails()

            # Should extract plain text from multipart
            assert len(emails) > 0
            assert "Plain text body" in emails[0].body or "HTML body" in emails[0].body

    @pytest.mark.asyncio
    async def test_parse_email_missing_fields(self, mock_imap_connection):
        """Test parsing email with missing headers."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Email with missing subject
        minimal_email = b"""From: sender@example.com
To: user@example.com
Date: Sat, 26 Oct 2025 10:00:00 +0000

Body content
"""
        mock_imap_connection.fetch.return_value = ('OK', [(b'1 (RFC822 {100}', minimal_email)])

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            emails = await connector.fetch_unread_emails()

            # Should handle missing subject gracefully
            assert len(emails) > 0
            assert emails[0].subject is not None  # Should have default subject


class TestIMAPBatchOperations:
    """Test batch operations on IMAP messages."""

    @pytest.mark.asyncio
    async def test_mark_as_read_single_message(self, mock_imap_connection):
        """Test marking a single message as read."""
        from src.connectors.email.imap_connector import IMAPConnector

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            await connector.mark_as_read(MOCK_EMAIL_ID)

            # Verify store was called with SEEN flag
            mock_imap_connection.store.assert_called()
            call_args = mock_imap_connection.store.call_args
            assert '\\Seen' in str(call_args) or 'SEEN' in str(call_args).upper()

    @pytest.mark.asyncio
    async def test_mark_as_read_batch(self, mock_imap_connection):
        """Test marking multiple messages as read in batch."""
        from src.connectors.email.imap_connector import IMAPConnector

        message_ids = ['1', '2', '3']

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            await connector.mark_as_read_batch(message_ids)

            # Should have called store for each message
            assert mock_imap_connection.store.call_count >= len(message_ids)

    @pytest.mark.asyncio
    async def test_mark_as_read_requires_authentication(self):
        """Test mark_as_read raises error if not authenticated."""
        from src.connectors.email.imap_connector import IMAPConnector

        connector = IMAPConnector(
            imap_server=MOCK_IMAP_SERVER,
            username=MOCK_USERNAME,
            password=MOCK_PASSWORD
        )
        # Not authenticated

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.mark_as_read(MOCK_EMAIL_ID)

    @pytest.mark.asyncio
    async def test_mark_as_read_batch_with_empty_list(self, mock_imap_connection):
        """Test mark_as_read_batch handles empty message list."""
        from src.connectors.email.imap_connector import IMAPConnector

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            # Should not raise error with empty list
            await connector.mark_as_read_batch([])


class TestIMAPErrorHandling:
    """Test error handling and retries."""

    @pytest.mark.asyncio
    async def test_fetch_emails_handles_imap_error(self, mock_imap_connection):
        """Test handling IMAP errors gracefully."""
        from src.connectors.email.imap_connector import IMAPConnector
        import imaplib

        # Mock IMAP error during search
        mock_imap_connection.search.side_effect = imaplib.IMAP4.error("Search failed")

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            with pytest.raises(RuntimeError):
                await connector.fetch_unread_emails()

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """Test handling connection timeouts."""
        from src.connectors.email.imap_connector import IMAPConnector
        import socket

        with patch('imaplib.IMAP4_SSL', side_effect=socket.timeout("Connection timeout")):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD,
                timeout=5
            )

            with pytest.raises(RuntimeError, match="timeout|failed"):
                await connector.authenticate()

    @pytest.mark.asyncio
    async def test_fetch_emails_select_folder_fails(self, mock_imap_connection):
        """Test handling folder selection failures."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Mock failed folder selection
        mock_imap_connection.select.return_value = ('NO', [b'Folder not found'])

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            with pytest.raises(RuntimeError, match="Failed to select folder"):
                await connector.fetch_unread_emails(folder="InvalidFolder")

    @pytest.mark.asyncio
    async def test_fetch_emails_handles_no_messages(self, mock_imap_connection):
        """Test handling empty search results."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Mock empty search result
        mock_imap_connection.search.return_value = ('OK', [b''])

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            emails = await connector.fetch_unread_emails()

            assert len(emails) == 0

    @pytest.mark.asyncio
    async def test_fetch_emails_handles_fetch_failure(self, mock_imap_connection):
        """Test handling individual message fetch failures."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Mock search returns messages but fetch fails
        mock_imap_connection.fetch.return_value = ('NO', [b'Fetch failed'])

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            emails = await connector.fetch_unread_emails()

            # Should skip failed messages and return empty list
            assert len(emails) == 0

    @pytest.mark.asyncio
    async def test_mark_as_read_handles_store_failure(self, mock_imap_connection):
        """Test handling store command failures."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Mock failed store operation
        mock_imap_connection.store.return_value = ('NO', [b'Store failed'])

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            with pytest.raises(RuntimeError, match="Failed to mark as read"):
                await connector.mark_as_read('1')

    @pytest.mark.asyncio
    async def test_authenticate_with_non_ssl(self):
        """Test authentication with non-SSL connection."""
        from src.connectors.email.imap_connector import IMAPConnector

        mock_connection = Mock()
        mock_connection.login.return_value = ('OK', [b'LOGIN completed'])
        mock_connection.sock = Mock()

        with patch('imaplib.IMAP4', return_value=mock_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD,
                use_ssl=False,
                port=143
            )

            await connector.authenticate()

            assert connector.is_authenticated is True

    @pytest.mark.asyncio
    async def test_authenticate_login_fails(self, mock_imap_connection):
        """Test authentication fails when login returns non-OK status."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Mock failed login (non-OK status)
        mock_imap_connection.login.return_value = ('NO', [b'Authentication failed'])

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )

            with pytest.raises(RuntimeError, match="Authentication failed"):
                await connector.authenticate()


class TestIMAPObservability:
    """Test observability and metrics emission."""

    @pytest.mark.asyncio
    async def test_fetch_emails_emits_metrics(self, mock_imap_connection):
        """Test fetch_unread_emails emits metrics."""
        from src.connectors.email.imap_connector import IMAPConnector

        mock_metrics_instance = Mock()

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            with patch('src.connectors.email.imap_connector.get_metrics_manager', return_value=mock_metrics_instance):
                connector = IMAPConnector(
                    imap_server=MOCK_IMAP_SERVER,
                    username=MOCK_USERNAME,
                    password=MOCK_PASSWORD
                )
                await connector.authenticate()

                await connector.fetch_unread_emails()

                # Verify metrics were emitted
                mock_metrics_instance.emit.assert_called()
                calls = [call.args[0] for call in mock_metrics_instance.emit.call_args_list]
                assert any('imap_emails_fetched' in str(call) for call in calls)


class TestIMAPDisconnect:
    """Test disconnection and cleanup."""

    @pytest.mark.asyncio
    async def test_disconnect_success(self, mock_imap_connection):
        """Test successful disconnection."""
        from src.connectors.email.imap_connector import IMAPConnector

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            connector.disconnect()

            # Verify close and logout were called
            mock_imap_connection.close.assert_called_once()
            mock_imap_connection.logout.assert_called_once()
            assert connector.connection is None

    @pytest.mark.asyncio
    async def test_disconnect_handles_errors(self, mock_imap_connection):
        """Test disconnect handles errors gracefully."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Mock close/logout failures
        mock_imap_connection.close.side_effect = Exception("Close failed")
        mock_imap_connection.logout.side_effect = Exception("Logout failed")

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            # Should not raise error
            connector.disconnect()

            # Connection should still be cleared
            assert connector.connection is None


class TestIMAPEmailParsingEdgeCases:
    """Test email parsing edge cases."""

    @pytest.mark.asyncio
    async def test_parse_email_with_malformed_bytes(self, mock_imap_connection):
        """Test parsing email when message parse fails."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Return malformed response that causes parse failure
        mock_imap_connection.fetch.side_effect = Exception("Parse error")

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            # Should skip failed messages and return empty list
            emails = await connector.fetch_unread_emails()
            assert len(emails) == 0

    @pytest.mark.asyncio
    async def test_parse_email_with_attachments(self, mock_imap_connection):
        """Test parsing email with attachments."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Email with attachment
        email_with_attachment = b"""From: sender@example.com
To: user@example.com
Subject: Email with Attachment
Content-Type: multipart/mixed; boundary="boundary123"

--boundary123
Content-Type: text/plain; charset=utf-8

Email body text

--boundary123
Content-Type: application/pdf; name="document.pdf"
Content-Disposition: attachment; filename="document.pdf"

Binary attachment data here

--boundary123--
"""
        mock_imap_connection.fetch.return_value = ('OK', [(b'1 (RFC822 {1234}', email_with_attachment)])

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            emails = await connector.fetch_unread_emails()

            # Should extract body text, skip attachment
            assert len(emails) > 0
            assert "Email body text" in emails[0].body

    @pytest.mark.asyncio
    async def test_parse_email_with_html_only(self, mock_imap_connection):
        """Test parsing email with HTML body only (no plain text)."""
        from src.connectors.email.imap_connector import IMAPConnector

        # Email with only HTML, no plain text
        html_only_email = b"""From: sender@example.com
To: user@example.com
Subject: HTML Email
Content-Type: multipart/alternative; boundary="boundary123"

--boundary123
Content-Type: text/html; charset=utf-8

<html><body><h1>HTML only email</h1></body></html>

--boundary123--
"""
        mock_imap_connection.fetch.return_value = ('OK', [(b'1 (RFC822 {1234}', html_only_email)])

        with patch('imaplib.IMAP4_SSL', return_value=mock_imap_connection):
            connector = IMAPConnector(
                imap_server=MOCK_IMAP_SERVER,
                username=MOCK_USERNAME,
                password=MOCK_PASSWORD
            )
            await connector.authenticate()

            emails = await connector.fetch_unread_emails()

            # Should extract HTML body
            assert len(emails) > 0
            assert "HTML" in emails[0].body or "html" in emails[0].body


class TestIMAPSingleton:
    """Test singleton pattern for IMAPConnector."""

    def test_get_imap_connector_returns_singleton(self):
        """Test get_imap_connector returns singleton instance."""
        from src.connectors.email.imap_connector import get_imap_connector

        connector1 = get_imap_connector(
            imap_server=MOCK_IMAP_SERVER,
            username=MOCK_USERNAME,
            password=MOCK_PASSWORD
        )
        connector2 = get_imap_connector(
            imap_server=MOCK_IMAP_SERVER,
            username=MOCK_USERNAME,
            password=MOCK_PASSWORD
        )

        assert connector1 is connector2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

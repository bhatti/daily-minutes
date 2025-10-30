#!/usr/bin/env python3
"""Unit tests for Outlook Connector (TDD approach).

Following TDD: Write tests first, then implement Outlook connector.

Mocking Strategy:
- Mock Microsoft Graph API client (requests library or msal)
- Mock OAuth credentials flow (MSAL - Microsoft Authentication Library)
- Mock Graph API responses (messages, folders, etc.)
- Verify metrics emission and observability

Key Differences from Gmail:
- Uses Microsoft Graph API instead of Gmail API
- OAuth via MSAL (Microsoft Authentication Library)
- Different API endpoints and response formats
- Uses /me/messages instead of users().messages()
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call, mock_open
from pydantic import ValidationError

# Test constants
MOCK_EMAIL_ID = "AAMkAGI2THVSAAA="
MOCK_SENDER = "sender@example.com"
MOCK_SUBJECT = "Test Email Subject"
MOCK_BODY = "This is the email body content."
MOCK_ACCESS_TOKEN = "mock_access_token_12345"


@pytest.fixture
def mock_msal_app():
    """Create mock MSAL confidential client application."""
    app = Mock()

    # Mock acquire_token_silent (for cached tokens)
    app.acquire_token_silent.return_value = {
        'access_token': MOCK_ACCESS_TOKEN,
        'token_type': 'Bearer',
        'expires_in': 3600
    }

    # Mock acquire_token_by_authorization_code (for OAuth flow)
    app.acquire_token_by_authorization_code.return_value = {
        'access_token': MOCK_ACCESS_TOKEN,
        'token_type': 'Bearer',
        'expires_in': 3600,
        'refresh_token': 'mock_refresh_token'
    }

    # Mock get_accounts
    app.get_accounts.return_value = []

    return app


@pytest.fixture
def mock_graph_response():
    """Create mock Microsoft Graph API response for emails."""
    return {
        'value': [
            {
                'id': MOCK_EMAIL_ID,
                'subject': MOCK_SUBJECT,
                'from': {
                    'emailAddress': {
                        'address': MOCK_SENDER,
                        'name': 'Test Sender'
                    }
                },
                'receivedDateTime': '2025-10-26T10:00:00Z',
                'body': {
                    'contentType': 'text',
                    'content': MOCK_BODY
                },
                'isRead': False,
                'importance': 'normal',
                'categories': []
            }
        ],
        '@odata.nextLink': None
    }


class TestOutlookConnectorInitialization:
    """Test Outlook connector initialization and configuration."""

    def test_connector_initialization(self):
        """Test OutlookConnector initializes correctly."""
        from src.connectors.email.outlook_connector import OutlookConnector

        connector = OutlookConnector(
            client_id="mock_client_id",
            client_secret="mock_client_secret",
            tenant_id="mock_tenant_id"
        )

        assert connector is not None
        assert connector.client_id == "mock_client_id"
        assert connector.access_token is None
        assert connector.is_authenticated is False

    def test_connector_requires_credentials(self):
        """Test OutlookConnector requires OAuth credentials."""
        from src.connectors.email.outlook_connector import OutlookConnector

        with pytest.raises(ValueError, match="client_id.*required"):
            OutlookConnector(client_id="", client_secret="secret", tenant_id="tenant")


class TestOutlookAuthentication:
    """Test Outlook OAuth authentication flow using MSAL."""

    def test_authenticate_with_cached_token(self, mock_msal_app):
        """Test authentication with cached access token."""
        from src.connectors.email.outlook_connector import OutlookConnector

        # Mock get_accounts to return a cached account
        mock_msal_app.get_accounts.return_value = [{'username': 'user@example.com'}]

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.open', mock_open(read_data='{"account": "data"}')):
                    connector = OutlookConnector(
                        client_id="client_id",
                        client_secret="secret",
                        tenant_id="tenant"
                    )
                    connector.authenticate()

                    assert connector.is_authenticated is True
                    assert connector.access_token == MOCK_ACCESS_TOKEN

    def test_authenticate_fails_with_no_token_no_code(self, mock_msal_app):
        """Test authentication fails when no cached token and no auth code."""
        from src.connectors.email.outlook_connector import OutlookConnector

        # No cached accounts
        mock_msal_app.get_accounts.return_value = []

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('os.path.exists', return_value=False):
                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )

                # Try to authenticate without providing auth code
                with pytest.raises(RuntimeError, match="No cached token found"):
                    connector.authenticate()

    def test_authenticate_fails_with_bad_auth_code(self, mock_msal_app):
        """Test authentication fails with invalid authorization code."""
        from src.connectors.email.outlook_connector import OutlookConnector

        # Mock failed auth response
        mock_msal_app.acquire_token_by_authorization_code.return_value = {
            'error': 'invalid_grant',
            'error_description': 'Invalid authorization code'
        }

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('os.path.exists', return_value=False):
                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )

                # Try to authenticate with bad code
                with pytest.raises(RuntimeError, match="Authentication failed"):
                    connector.authenticate(authorization_code="bad_code")

    def test_authenticate_with_authorization_code(self, mock_msal_app):
        """Test authentication with OAuth authorization code flow."""
        from src.connectors.email.outlook_connector import OutlookConnector

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('os.path.exists', return_value=False):
                with patch('builtins.open', mock_open()):
                    with patch('os.makedirs'):
                        connector = OutlookConnector(
                            client_id="client_id",
                            client_secret="secret",
                            tenant_id="tenant"
                        )

                        # Mock the authorization code flow
                        connector.authenticate(authorization_code="mock_auth_code")

                        assert connector.is_authenticated is True
                        mock_msal_app.acquire_token_by_authorization_code.assert_called_once()

    def test_authenticate_token_refresh(self, mock_msal_app):
        """Test automatic token refresh when expired."""
        from src.connectors.email.outlook_connector import OutlookConnector

        # Mock get_accounts to return a cached account
        mock_msal_app.get_accounts.return_value = [{'username': 'user@example.com'}]

        # Mock expired token then successful refresh
        mock_msal_app.acquire_token_silent.side_effect = [
            None,  # First call returns None (expired)
            {'access_token': 'new_token', 'token_type': 'Bearer', 'expires_in': 3600}
        ]

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('os.path.exists', return_value=False):
                with patch('builtins.open', mock_open()):
                    with patch('os.makedirs'):
                        connector = OutlookConnector(
                            client_id="client_id",
                            client_secret="secret",
                            tenant_id="tenant"
                        )

                        # Provide auth code since silent auth will fail first time
                        connector.authenticate(authorization_code="mock_code")

                        # Should use authorization code flow
                        assert connector.is_authenticated is True


class TestOutlookFetchEmails:
    """Test fetching emails from Outlook/Microsoft 365."""

    @pytest.mark.asyncio
    async def test_fetch_unread_emails_success(self, mock_msal_app, mock_graph_response):
        """Test successfully fetching unread emails."""
        from src.connectors.email.outlook_connector import OutlookConnector

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.get') as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = mock_graph_response

                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )
                connector.access_token = MOCK_ACCESS_TOKEN

                emails = await connector.fetch_unread_emails(max_results=10)

                assert len(emails) == 1
                assert emails[0].id == MOCK_EMAIL_ID
                assert emails[0].subject == MOCK_SUBJECT
                assert emails[0].sender == MOCK_SENDER
                assert emails[0].body == MOCK_BODY

    @pytest.mark.asyncio
    async def test_fetch_emails_requires_authentication(self):
        """Test fetch_unread_emails raises error if not authenticated."""
        from src.connectors.email.outlook_connector import OutlookConnector

        connector = OutlookConnector(
            client_id="client_id",
            client_secret="secret",
            tenant_id="tenant"
        )

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.fetch_unread_emails()

    @pytest.mark.asyncio
    async def test_fetch_emails_with_filter(self, mock_msal_app, mock_graph_response):
        """Test fetching emails with OData filter."""
        from src.connectors.email.outlook_connector import OutlookConnector

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.get') as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = mock_graph_response

                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )
                connector.access_token = MOCK_ACCESS_TOKEN

                await connector.fetch_unread_emails(
                    filter_query="from/emailAddress/address eq 'boss@company.com'"
                )

                # Verify filter was passed in query params
                call_args = mock_get.call_args
                assert '$filter' in call_args[1]['params']

    @pytest.mark.asyncio
    async def test_fetch_emails_with_pagination(self, mock_msal_app):
        """Test fetching emails with pagination (@odata.nextLink)."""
        from src.connectors.email.outlook_connector import OutlookConnector

        # Page 1 response
        page1_response = {
            'value': [{'id': 'msg_1', 'subject': 'Email 1', 'from': {'emailAddress': {'address': 'a@example.com'}},
                      'receivedDateTime': '2025-10-26T10:00:00Z', 'body': {'contentType': 'text', 'content': 'Body 1'},
                      'isRead': False}],
            '@odata.nextLink': 'https://graph.microsoft.com/v1.0/me/messages?$skip=10'
        }

        # Page 2 response
        page2_response = {
            'value': [{'id': 'msg_2', 'subject': 'Email 2', 'from': {'emailAddress': {'address': 'b@example.com'}},
                      'receivedDateTime': '2025-10-26T11:00:00Z', 'body': {'contentType': 'text', 'content': 'Body 2'},
                      'isRead': False}],
            '@odata.nextLink': None
        }

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.get') as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.side_effect = [page1_response, page2_response]

                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )
                connector.access_token = MOCK_ACCESS_TOKEN

                emails = await connector.fetch_unread_emails(max_results=50)

                # Should have fetched both pages
                assert len(emails) == 2
                assert mock_get.call_count == 2


class TestOutlookEmailParsing:
    """Test parsing Microsoft Graph API responses into EmailMessage models."""

    @pytest.mark.asyncio
    async def test_parse_email_with_html_body(self, mock_msal_app):
        """Test parsing email with HTML body."""
        from src.connectors.email.outlook_connector import OutlookConnector

        html_response = {
            'value': [{
                'id': 'msg_html',
                'subject': 'HTML Email',
                'from': {'emailAddress': {'address': MOCK_SENDER}},
                'receivedDateTime': '2025-10-26T10:00:00Z',
                'body': {
                    'contentType': 'html',
                    'content': '<html><body>HTML body</body></html>'
                },
                'isRead': False,
                'importance': 'normal',
                'categories': []
            }]
        }

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.get') as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = html_response

                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )
                connector.access_token = MOCK_ACCESS_TOKEN

                emails = await connector.fetch_unread_emails()

                assert '<html>' in emails[0].body or 'HTML body' in emails[0].body

    @pytest.mark.asyncio
    async def test_parse_email_with_high_importance(self, mock_msal_app):
        """Test parsing email with high importance boosts score."""
        from src.connectors.email.outlook_connector import OutlookConnector

        important_response = {
            'value': [{
                'id': 'msg_important',
                'subject': 'URGENT',
                'from': {'emailAddress': {'address': MOCK_SENDER}},
                'receivedDateTime': '2025-10-26T10:00:00Z',
                'body': {'contentType': 'text', 'content': 'Urgent action needed'},
                'isRead': False,
                'importance': 'high',
                'categories': ['Important']
            }]
        }

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.get') as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = important_response

                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )
                connector.access_token = MOCK_ACCESS_TOKEN

                emails = await connector.fetch_unread_emails()

                # Should have boosted importance score
                assert emails[0].importance_score > 0.5

    @pytest.mark.asyncio
    async def test_parse_email_with_low_importance(self, mock_msal_app):
        """Test parsing email with low importance reduces score."""
        from src.connectors.email.outlook_connector import OutlookConnector

        low_importance_response = {
            'value': [{
                'id': 'msg_low',
                'subject': 'FYI',
                'from': {'emailAddress': {'address': MOCK_SENDER}},
                'receivedDateTime': '2025-10-26T10:00:00Z',
                'body': {'contentType': 'text', 'content': 'Just for your information'},
                'isRead': False,
                'importance': 'low',
                'categories': []
            }]
        }

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.get') as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = low_importance_response

                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )
                connector.access_token = MOCK_ACCESS_TOKEN

                emails = await connector.fetch_unread_emails()

                # Should have reduced importance score
                assert emails[0].importance_score < 0.5


class TestOutlookBatchOperations:
    """Test batch operations on Outlook messages."""

    @pytest.mark.asyncio
    async def test_mark_as_read_single_message(self, mock_msal_app):
        """Test marking a single message as read."""
        from src.connectors.email.outlook_connector import OutlookConnector

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.patch') as mock_patch:
                mock_patch.return_value.status_code = 200

                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )
                connector.access_token = MOCK_ACCESS_TOKEN

                await connector.mark_as_read(MOCK_EMAIL_ID)

                # Verify PATCH was called with isRead=true
                mock_patch.assert_called_once()
                call_args = mock_patch.call_args
                assert call_args[1]['json']['isRead'] is True

    @pytest.mark.asyncio
    async def test_mark_as_read_batch(self, mock_msal_app):
        """Test marking multiple messages as read in batch."""
        from src.connectors.email.outlook_connector import OutlookConnector

        message_ids = ['msg_1', 'msg_2', 'msg_3']

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.patch') as mock_patch:
                mock_patch.return_value.status_code = 200

                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )
                connector.access_token = MOCK_ACCESS_TOKEN

                await connector.mark_as_read_batch(message_ids)

                # Should have called PATCH for each message
                assert mock_patch.call_count == len(message_ids)

    @pytest.mark.asyncio
    async def test_mark_as_read_requires_authentication(self, mock_msal_app):
        """Test mark_as_read raises error if not authenticated."""
        from src.connectors.email.outlook_connector import OutlookConnector

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            connector = OutlookConnector(
                client_id="client_id",
                client_secret="secret",
                tenant_id="tenant"
            )
            # Not authenticated

            with pytest.raises(RuntimeError, match="Not authenticated"):
                await connector.mark_as_read(MOCK_EMAIL_ID)

    @pytest.mark.asyncio
    async def test_mark_as_read_handles_api_error(self, mock_msal_app):
        """Test mark_as_read handles API errors."""
        from src.connectors.email.outlook_connector import OutlookConnector
        import requests

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.patch') as mock_patch:
                mock_patch.return_value.status_code = 500
                mock_patch.return_value.raise_for_status.side_effect = requests.HTTPError("Server error")

                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )
                connector.access_token = MOCK_ACCESS_TOKEN

                with pytest.raises(requests.HTTPError):
                    await connector.mark_as_read(MOCK_EMAIL_ID)

    @pytest.mark.asyncio
    async def test_mark_as_read_batch_with_empty_list(self, mock_msal_app):
        """Test mark_as_read_batch handles empty message list."""
        from src.connectors.email.outlook_connector import OutlookConnector

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            connector = OutlookConnector(
                client_id="client_id",
                client_secret="secret",
                tenant_id="tenant"
            )
            connector.access_token = MOCK_ACCESS_TOKEN

            # Should not raise error with empty list
            await connector.mark_as_read_batch([])

    @pytest.mark.asyncio
    async def test_mark_as_read_batch_requires_authentication(self, mock_msal_app):
        """Test mark_as_read_batch raises error if not authenticated."""
        from src.connectors.email.outlook_connector import OutlookConnector

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            connector = OutlookConnector(
                client_id="client_id",
                client_secret="secret",
                tenant_id="tenant"
            )
            # Not authenticated

            with pytest.raises(RuntimeError, match="Not authenticated"):
                await connector.mark_as_read_batch(['msg_1'])


class TestOutlookErrorHandling:
    """Test error handling and retries."""

    @pytest.mark.asyncio
    async def test_fetch_emails_handles_api_error(self, mock_msal_app):
        """Test handling Graph API errors gracefully."""
        from src.connectors.email.outlook_connector import OutlookConnector
        import requests

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.get') as mock_get:
                mock_get.return_value.status_code = 500
                mock_get.return_value.raise_for_status.side_effect = requests.HTTPError("Server error")

                connector = OutlookConnector(
                    client_id="client_id",
                    client_secret="secret",
                    tenant_id="tenant"
                )
                connector.access_token = MOCK_ACCESS_TOKEN

                with pytest.raises(requests.HTTPError):
                    await connector.fetch_unread_emails()

    @pytest.mark.asyncio
    async def test_fetch_emails_retries_on_rate_limit(self, mock_msal_app, mock_graph_response):
        """Test retries on rate limit errors (429)."""
        from src.connectors.email.outlook_connector import OutlookConnector

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('requests.get') as mock_get:
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    # First call: 429, second call: success
                    mock_response_429 = Mock()
                    mock_response_429.status_code = 429
                    mock_response_429.headers = {'Retry-After': '1'}

                    mock_response_200 = Mock()
                    mock_response_200.status_code = 200
                    mock_response_200.json.return_value = mock_graph_response

                    mock_get.side_effect = [mock_response_429, mock_response_200]

                    connector = OutlookConnector(
                        client_id="client_id",
                        client_secret="secret",
                        tenant_id="tenant"
                    )
                    connector.access_token = MOCK_ACCESS_TOKEN

                    emails = await connector.fetch_unread_emails(max_retries=3)

                    # Should have retried and succeeded
                    assert len(emails) == 1
                    assert mock_get.call_count == 2


class TestOutlookObservability:
    """Test observability and metrics emission."""

    @pytest.mark.asyncio
    async def test_fetch_emails_emits_metrics(self, mock_msal_app, mock_graph_response):
        """Test fetch_unread_emails emits metrics."""
        from src.connectors.email.outlook_connector import OutlookConnector

        mock_metrics_instance = Mock()

        with patch('msal.ConfidentialClientApplication', return_value=mock_msal_app):
            with patch('src.connectors.email.outlook_connector.get_metrics_manager', return_value=mock_metrics_instance):
                with patch('requests.get') as mock_get:
                    mock_get.return_value.status_code = 200
                    mock_get.return_value.json.return_value = mock_graph_response

                    connector = OutlookConnector(
                        client_id="client_id",
                        client_secret="secret",
                        tenant_id="tenant"
                    )
                    connector.access_token = MOCK_ACCESS_TOKEN

                    await connector.fetch_unread_emails()

                    # Verify metrics were emitted
                    mock_metrics_instance.emit.assert_called()
                    calls = [call.args[0] for call in mock_metrics_instance.emit.call_args_list]
                    assert any('outlook_emails_fetched' in str(call) for call in calls)


class TestOutlookSingleton:
    """Test singleton pattern for OutlookConnector."""

    def test_get_outlook_connector_returns_singleton(self):
        """Test get_outlook_connector returns singleton instance."""
        from src.connectors.email.outlook_connector import get_outlook_connector

        connector1 = get_outlook_connector(client_id="id", client_secret="secret", tenant_id="tenant")
        connector2 = get_outlook_connector(client_id="id", client_secret="secret", tenant_id="tenant")

        assert connector1 is connector2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

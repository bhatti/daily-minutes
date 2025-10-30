#!/usr/bin/env python3
"""Unit tests for Gmail Connector (TDD approach).

Following TDD: Write tests first, then implement Gmail connector.

Mocking Strategy:
- Mock Google API client (googleapiclient.discovery.build)
- Mock OAuth credentials flow
- Mock Gmail API responses (messages.list, messages.get, etc.)
- Verify metrics emission and observability
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call, mock_open
from pydantic import ValidationError

# Test constants
MOCK_EMAIL_ID = "msg_12345abcde"
MOCK_THREAD_ID = "thread_abc123"
MOCK_SENDER = "sender@example.com"
MOCK_SUBJECT = "Test Email Subject"
MOCK_BODY = "This is the email body content."


@pytest.fixture
def mock_gmail_service():
    """Create mock Gmail API service with common responses."""
    service = Mock()

    # Mock messages().list() for fetching message IDs
    list_mock = Mock()
    list_mock.execute.return_value = {
        'messages': [
            {'id': MOCK_EMAIL_ID, 'threadId': MOCK_THREAD_ID}
        ],
        'resultSizeEstimate': 1
    }
    service.users().messages().list.return_value = list_mock

    # Mock messages().get() for fetching full message
    get_mock = Mock()
    get_mock.execute.return_value = {
        'id': MOCK_EMAIL_ID,
        'threadId': MOCK_THREAD_ID,
        'labelIds': ['INBOX', 'UNREAD'],
        'snippet': 'Email preview text...',
        'internalDate': '1234567890000',  # Unix timestamp in milliseconds
        'payload': {
            'headers': [
                {'name': 'From', 'value': MOCK_SENDER},
                {'name': 'Subject', 'value': MOCK_SUBJECT},
                {'name': 'Date', 'value': 'Mon, 26 Oct 2025 10:00:00 -0700'}
            ],
            'body': {
                'size': len(MOCK_BODY),
                'data': 'VGhpcyBpcyB0aGUgZW1haWwgYm9keSBjb250ZW50Lg=='  # base64 encoded
            }
        }
    }
    service.users().messages().get.return_value = get_mock

    # Mock messages().modify() for marking as read
    modify_mock = Mock()
    modify_mock.execute.return_value = {'id': MOCK_EMAIL_ID}
    service.users().messages().modify.return_value = modify_mock

    # Mock messages().batchModify() for batch operations
    batch_modify_mock = Mock()
    batch_modify_mock.execute.return_value = {}
    service.users().messages().batchModify.return_value = batch_modify_mock

    return service


@pytest.fixture
def mock_credentials():
    """Create mock OAuth credentials."""
    creds = Mock()
    creds.valid = True
    creds.expired = False
    creds.refresh_token = "mock_refresh_token"
    creds.token = "mock_access_token"
    creds.to_json.return_value = '{"token": "mock_access_token"}'
    return creds


class TestGmailConnectorInitialization:
    """Test Gmail connector initialization and configuration."""

    def test_connector_initialization(self):
        """Test GmailConnector initializes correctly."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()

        assert connector is not None
        assert connector.service is None  # Not authenticated yet
        assert connector.credentials is None
        assert connector.user_id == "me"  # Default user

    def test_connector_with_custom_user_id(self):
        """Test GmailConnector with custom user ID."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector(user_id="test@example.com")

        assert connector.user_id == "test@example.com"

    def test_connector_requires_credentials_file(self):
        """Test connector validates credentials file exists."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()

        # Should raise error if credentials file doesn't exist
        with pytest.raises(FileNotFoundError):
            connector._load_credentials_from_file("/nonexistent/credentials.json")


class TestGmailAuthentication:
    """Test Gmail OAuth authentication flow."""

    def test_authenticate_with_valid_credentials(self, mock_credentials, mock_gmail_service):
        """Test authentication with valid stored credentials."""
        from src.connectors.email.gmail_connector import GmailConnector

        with patch('os.path.exists', return_value=True):
            with patch('google.oauth2.credentials.Credentials.from_authorized_user_file',
                       return_value=mock_credentials):
                with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
                    with patch('builtins.open', mock_open()):  # Mock file write
                        connector = GmailConnector()
                        connector.authenticate(credentials_path="mock_token.json")

                        assert connector.credentials == mock_credentials
                        assert connector.service == mock_gmail_service
                        assert connector.is_authenticated is True

    def test_authenticate_with_expired_credentials_refresh(self, mock_credentials, mock_gmail_service):
        """Test authentication refreshes expired credentials."""
        from src.connectors.email.gmail_connector import GmailConnector

        # Make credentials expired but with refresh token
        mock_credentials.valid = False
        mock_credentials.expired = True
        mock_credentials.refresh_token = "valid_refresh_token"

        with patch('os.path.exists', return_value=True):
            with patch('google.oauth2.credentials.Credentials.from_authorized_user_file',
                       return_value=mock_credentials):
                with patch('google.auth.transport.requests.Request'):
                    with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
                        with patch('builtins.open', mock_open()):
                            with patch('os.makedirs'):
                                connector = GmailConnector()
                                connector.authenticate(credentials_path="mock_token.json")

                                # Should have called refresh
                                mock_credentials.refresh.assert_called_once()
                                assert connector.is_authenticated is True

    def test_authenticate_new_user_oauth_flow(self, mock_credentials, mock_gmail_service):
        """Test OAuth flow for new user without stored credentials."""
        from src.connectors.email.gmail_connector import GmailConnector

        # Simulate no stored credentials
        with patch('os.path.exists', return_value=False):  # No existing token file
            with patch('google.oauth2.credentials.Credentials.from_authorized_user_file',
                       side_effect=FileNotFoundError):
                with patch('google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file') as mock_flow:
                    mock_flow_instance = Mock()
                    mock_flow_instance.run_local_server.return_value = mock_credentials
                    mock_flow.return_value = mock_flow_instance

                    with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
                        with patch('builtins.open', mock_open()):
                            with patch('os.makedirs'):
                                connector = GmailConnector()
                                connector.authenticate(
                                    credentials_path="mock_token.json",
                                    client_secrets_path="credentials/client_secret.json"
                                )

                                # Should have run OAuth flow
                                mock_flow_instance.run_local_server.assert_called_once()
                                assert connector.is_authenticated is True

    def test_authenticate_without_refresh_token_reruns_oauth(self, mock_gmail_service):
        """Test re-runs OAuth if expired without refresh token."""
        from src.connectors.email.gmail_connector import GmailConnector

        # Expired credentials without refresh token
        expired_creds = Mock()
        expired_creds.valid = False
        expired_creds.expired = True
        expired_creds.refresh_token = None

        new_creds = Mock()
        new_creds.valid = True
        new_creds.expired = False

        with patch('os.path.exists', return_value=True):
            with patch('google.oauth2.credentials.Credentials.from_authorized_user_file',
                       return_value=expired_creds):
                with patch('google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file') as mock_flow:
                    mock_flow_instance = Mock()
                    mock_flow_instance.run_local_server.return_value = new_creds
                    mock_flow.return_value = mock_flow_instance

                    with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
                        with patch('builtins.open', mock_open()):
                            with patch('os.makedirs'):
                                connector = GmailConnector()
                                connector.authenticate(
                                    credentials_path="mock_token.json",
                                    client_secrets_path="credentials/client_secret.json"
                                )

                                # Should have re-run OAuth flow
                                mock_flow_instance.run_local_server.assert_called_once()
                                assert connector.credentials == new_creds


class TestGmailFetchEmails:
    """Test fetching emails from Gmail."""

    @pytest.mark.asyncio
    async def test_fetch_unread_emails_success(self, mock_gmail_service, mock_credentials):
        """Test successfully fetching unread emails."""
        from src.connectors.email.gmail_connector import GmailConnector

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            emails = await connector.fetch_unread_emails(max_results=10)

            assert len(emails) == 1
            assert emails[0].id == MOCK_EMAIL_ID
            assert emails[0].subject == MOCK_SUBJECT
            assert emails[0].sender == MOCK_SENDER
            assert emails[0].body == MOCK_BODY
            assert 'UNREAD' in emails[0].labels

    @pytest.mark.asyncio
    async def test_fetch_unread_emails_with_query_filter(self, mock_gmail_service, mock_credentials):
        """Test fetching emails with custom query filter."""
        from src.connectors.email.gmail_connector import GmailConnector

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            await connector.fetch_unread_emails(
                max_results=10,
                query="from:boss@company.com is:unread"
            )

            # Verify the query was passed correctly
            call_args = mock_gmail_service.users().messages().list.call_args
            assert call_args[1]['q'] == "from:boss@company.com is:unread"

    @pytest.mark.asyncio
    async def test_fetch_emails_requires_authentication(self):
        """Test fetch_unread_emails raises error if not authenticated."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()

        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector.fetch_unread_emails()

    @pytest.mark.asyncio
    async def test_fetch_emails_handles_no_messages(self, mock_gmail_service, mock_credentials):
        """Test handling empty inbox (no unread messages)."""
        from src.connectors.email.gmail_connector import GmailConnector

        # Mock empty response
        list_mock = Mock()
        list_mock.execute.return_value = {'resultSizeEstimate': 0}
        mock_gmail_service.users().messages().list.return_value = list_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            emails = await connector.fetch_unread_emails()

            assert emails == []

    @pytest.mark.asyncio
    async def test_fetch_emails_with_pagination(self, mock_gmail_service, mock_credentials):
        """Test fetching emails with pagination (nextPageToken)."""
        from src.connectors.email.gmail_connector import GmailConnector

        # First page with nextPageToken
        list_mock_page1 = Mock()
        list_mock_page1.execute.return_value = {
            'messages': [{'id': 'msg_1', 'threadId': 'thread_1'}],
            'nextPageToken': 'token_page2',
            'resultSizeEstimate': 2
        }

        # Second page without nextPageToken
        list_mock_page2 = Mock()
        list_mock_page2.execute.return_value = {
            'messages': [{'id': 'msg_2', 'threadId': 'thread_2'}],
            'resultSizeEstimate': 2
        }

        mock_gmail_service.users().messages().list.side_effect = [
            list_mock_page1,
            list_mock_page2
        ]

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            emails = await connector.fetch_unread_emails(max_results=50)

            # Should have fetched both pages
            assert mock_gmail_service.users().messages().list.call_count == 2
            # Note: get() would be called for each message, so 2 times

    @pytest.mark.asyncio
    async def test_fetch_emails_respects_max_results(self, mock_gmail_service, mock_credentials):
        """Test max_results parameter limits returned emails."""
        from src.connectors.email.gmail_connector import GmailConnector

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            await connector.fetch_unread_emails(max_results=5)

            # Verify maxResults parameter
            call_args = mock_gmail_service.users().messages().list.call_args
            assert call_args[1]['maxResults'] == 5


class TestGmailEmailParsing:
    """Test parsing Gmail API responses into EmailMessage models."""

    @pytest.mark.asyncio
    async def test_parse_email_with_text_body(self, mock_gmail_service):
        """Test parsing email with plain text body."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()
        connector.service = mock_gmail_service

        email_msg = await connector._parse_email_message({
            'id': MOCK_EMAIL_ID,
            'threadId': MOCK_THREAD_ID,
            'labelIds': ['INBOX', 'UNREAD'],
            'snippet': 'Preview text...',
            'internalDate': '1729954800000',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': MOCK_SENDER},
                    {'name': 'Subject', 'value': MOCK_SUBJECT},
                    {'name': 'Date', 'value': 'Mon, 26 Oct 2025 10:00:00 -0700'}
                ],
                'body': {
                    'data': 'VGhpcyBpcyB0aGUgZW1haWwgYm9keSBjb250ZW50Lg=='
                }
            }
        })

        assert email_msg.id == MOCK_EMAIL_ID
        assert email_msg.subject == MOCK_SUBJECT
        assert email_msg.sender == MOCK_SENDER
        assert email_msg.body == MOCK_BODY
        assert email_msg.snippet == 'Preview text...'
        assert 'UNREAD' in email_msg.labels

    @pytest.mark.asyncio
    async def test_parse_email_with_multipart_body(self, mock_gmail_service):
        """Test parsing email with multipart/alternative body (text + HTML)."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()
        connector.service = mock_gmail_service

        # Multipart email structure
        multipart_payload = {
            'id': MOCK_EMAIL_ID,
            'threadId': MOCK_THREAD_ID,
            'labelIds': ['INBOX'],
            'snippet': 'Preview...',
            'internalDate': '1729954800000',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': MOCK_SENDER},
                    {'name': 'Subject', 'value': MOCK_SUBJECT}
                ],
                'mimeType': 'multipart/alternative',
                'parts': [
                    {
                        'mimeType': 'text/plain',
                        'body': {'data': 'UGxhaW4gdGV4dCBib2R5'}  # "Plain text body"
                    },
                    {
                        'mimeType': 'text/html',
                        'body': {'data': 'PGh0bWw+SFRNTCBib2R5PC9odG1sPg=='}  # "<html>HTML body</html>"
                    }
                ]
            }
        }

        email_msg = await connector._parse_email_message(multipart_payload)

        # Should prefer plain text over HTML
        assert email_msg.body == "Plain text body"

    @pytest.mark.asyncio
    async def test_parse_email_extracts_importance_keywords(self, mock_gmail_service):
        """Test parsing extracts importance keywords for scoring."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()
        connector.service = mock_gmail_service

        urgent_email = {
            'id': 'msg_urgent',
            'threadId': 'thread_urgent',
            'labelIds': ['INBOX', 'IMPORTANT'],
            'snippet': 'URGENT: Action required',
            'internalDate': '1729954800000',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'boss@company.com'},
                    {'name': 'Subject', 'value': 'URGENT: Client meeting tomorrow'},
                    {'name': 'Importance', 'value': 'high'}
                ],
                'body': {'data': 'VXJnZW50IGFjdGlvbiByZXF1aXJlZA=='}
            }
        }

        email_msg = await connector._parse_email_message(urgent_email)

        # Should have higher importance score due to URGENT, IMPORTANT label
        # Default is 0.5, should be boosted
        assert email_msg.importance_score > 0.5

    @pytest.mark.asyncio
    async def test_parse_email_handles_missing_headers(self, mock_gmail_service):
        """Test parsing handles emails with missing headers gracefully."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()
        connector.service = mock_gmail_service

        minimal_email = {
            'id': 'msg_minimal',
            'threadId': 'thread_minimal',
            'labelIds': ['INBOX'],
            'snippet': 'No subject',
            'internalDate': '1729954800000',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'sender@example.com'}
                    # Missing Subject header
                ],
                'body': {'data': 'Qm9keQ=='}  # "Body"
            }
        }

        email_msg = await connector._parse_email_message(minimal_email)

        assert email_msg.subject == "(No Subject)"  # Default subject
        assert email_msg.sender == "sender@example.com"


class TestGmailBatchOperations:
    """Test batch operations on Gmail messages."""

    @pytest.mark.asyncio
    async def test_mark_as_read_single_message(self, mock_gmail_service, mock_credentials):
        """Test marking a single message as read."""
        from src.connectors.email.gmail_connector import GmailConnector

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            await connector.mark_as_read(MOCK_EMAIL_ID)

            # Verify modify was called with correct parameters
            mock_gmail_service.users().messages().modify.assert_called_once()
            call_args = mock_gmail_service.users().messages().modify.call_args
            assert call_args[1]['id'] == MOCK_EMAIL_ID
            assert 'UNREAD' in call_args[1]['body']['removeLabelIds']

    @pytest.mark.asyncio
    async def test_mark_as_read_batch(self, mock_gmail_service, mock_credentials):
        """Test marking multiple messages as read in batch."""
        from src.connectors.email.gmail_connector import GmailConnector

        message_ids = ['msg_1', 'msg_2', 'msg_3']

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            await connector.mark_as_read_batch(message_ids)

            # Verify batchModify was called
            mock_gmail_service.users().messages().batchModify.assert_called_once()
            call_args = mock_gmail_service.users().messages().batchModify.call_args
            assert call_args[1]['body']['ids'] == message_ids
            assert 'UNREAD' in call_args[1]['body']['removeLabelIds']

    @pytest.mark.asyncio
    async def test_add_labels_to_messages(self, mock_gmail_service, mock_credentials):
        """Test adding labels to messages."""
        from src.connectors.email.gmail_connector import GmailConnector

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            await connector.add_labels(MOCK_EMAIL_ID, ['STARRED', 'IMPORTANT'])

            # Verify modify was called with addLabelIds
            call_args = mock_gmail_service.users().messages().modify.call_args
            assert 'STARRED' in call_args[1]['body']['addLabelIds']
            assert 'IMPORTANT' in call_args[1]['body']['addLabelIds']


class TestGmailErrorHandling:
    """Test error handling and retries."""

    @pytest.mark.asyncio
    async def test_fetch_emails_handles_api_error(self, mock_gmail_service, mock_credentials):
        """Test handling Gmail API errors gracefully."""
        from src.connectors.email.gmail_connector import GmailConnector
        from googleapiclient.errors import HttpError

        # Mock API error
        list_mock = Mock()
        list_mock.execute.side_effect = HttpError(
            resp=Mock(status=500),
            content=b'Internal Server Error'
        )
        mock_gmail_service.users().messages().list.return_value = list_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            with pytest.raises(HttpError):
                await connector.fetch_unread_emails()

    @pytest.mark.asyncio
    async def test_fetch_emails_retries_on_rate_limit(self, mock_gmail_service, mock_credentials):
        """Test retries on rate limit errors (429)."""
        from src.connectors.email.gmail_connector import GmailConnector
        from googleapiclient.errors import HttpError

        # First call: rate limit error
        # Second call: success
        list_mock = Mock()
        list_mock.execute.side_effect = [
            HttpError(resp=Mock(status=429), content=b'Rate limit exceeded'),
            {'messages': [{'id': MOCK_EMAIL_ID, 'threadId': MOCK_THREAD_ID}]}
        ]
        mock_gmail_service.users().messages().list.return_value = list_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            with patch('asyncio.sleep', new_callable=AsyncMock):  # Mock sleep to speed up test
                connector = GmailConnector()
                connector.credentials = mock_credentials
                connector.service = mock_gmail_service

                emails = await connector.fetch_unread_emails(max_retries=3)

                # Should have retried and succeeded
                assert list_mock.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_emails_fails_after_max_retries(self, mock_gmail_service, mock_credentials):
        """Test fails after exceeding max retries."""
        from src.connectors.email.gmail_connector import GmailConnector
        from googleapiclient.errors import HttpError

        # Always fail with rate limit
        list_mock = Mock()
        list_mock.execute.side_effect = HttpError(
            resp=Mock(status=429),
            content=b'Rate limit exceeded'
        )
        mock_gmail_service.users().messages().list.return_value = list_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                connector = GmailConnector()
                connector.credentials = mock_credentials
                connector.service = mock_gmail_service

                with pytest.raises(HttpError):
                    await connector.fetch_unread_emails(max_retries=2)

                # Should have tried 3 times (initial + 2 retries)
                assert list_mock.execute.call_count == 3


class TestGmailObservabilityAndMetrics:
    """Test observability and metrics emission."""

    @pytest.mark.asyncio
    async def test_fetch_emails_emits_metrics(self, mock_gmail_service, mock_credentials):
        """Test fetch_unread_emails emits metrics."""
        from src.connectors.email.gmail_connector import GmailConnector

        mock_metrics_instance = Mock()

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            with patch('src.connectors.email.gmail_connector.get_metrics_manager', return_value=mock_metrics_instance):
                connector = GmailConnector()
                connector.credentials = mock_credentials
                connector.service = mock_gmail_service

                await connector.fetch_unread_emails()

                # Verify metrics were emitted
                mock_metrics_instance.emit.assert_called()
                # Check for fetch count metric
                calls = [call.args[0] for call in mock_metrics_instance.emit.call_args_list]
                assert any('gmail_emails_fetched' in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_fetch_emails_logs_activity(self, mock_gmail_service, mock_credentials):
        """Test fetch_unread_emails logs activity to database."""
        from src.connectors.email.gmail_connector import GmailConnector

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            with patch.object(GmailConnector, '_log_activity', new_callable=AsyncMock) as mock_log:
                connector = GmailConnector()
                connector.credentials = mock_credentials
                connector.service = mock_gmail_service

                await connector.fetch_unread_emails()

                # Verify activity was logged
                mock_log.assert_called_once()
                call_args = mock_log.call_args[1]
                assert call_args['activity_type'] == 'gmail_fetch_emails'
                assert call_args['status'] == 'success'

    @pytest.mark.asyncio
    async def test_authentication_logs_activity(self, mock_credentials, mock_gmail_service):
        """Test authentication logs activity."""
        from src.connectors.email.gmail_connector import GmailConnector

        with patch('os.path.exists', return_value=True):
            with patch('google.oauth2.credentials.Credentials.from_authorized_user_file',
                       return_value=mock_credentials):
                with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
                    with patch('builtins.open', mock_open()):
                        with patch.object(GmailConnector, '_log_activity', new_callable=AsyncMock) as mock_log:
                            connector = GmailConnector()
                            connector.authenticate(credentials_path="mock_token.json")

                            # Verify authentication was logged
                            mock_log.assert_called_once()
                            call_args = mock_log.call_args[1]
                            assert call_args['activity_type'] == 'gmail_authentication'
                            assert call_args['status'] == 'success'

    @pytest.mark.asyncio
    async def test_error_logs_with_details(self, mock_gmail_service, mock_credentials):
        """Test errors are logged with details."""
        from src.connectors.email.gmail_connector import GmailConnector
        from googleapiclient.errors import HttpError

        list_mock = Mock()
        list_mock.execute.side_effect = HttpError(
            resp=Mock(status=403),
            content=b'Insufficient permissions'
        )
        mock_gmail_service.users().messages().list.return_value = list_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            with patch.object(GmailConnector, '_log_activity', new_callable=AsyncMock) as mock_log:
                connector = GmailConnector()
                connector.credentials = mock_credentials
                connector.service = mock_gmail_service

                try:
                    await connector.fetch_unread_emails()
                except HttpError:
                    pass

                # Verify error was logged
                mock_log.assert_called_once()
                call_args = mock_log.call_args[1]
                assert call_args['status'] == 'failed'
                assert 'error' in call_args


class TestGmailEdgeCases:
    """Test edge cases and error paths for better coverage."""

    @pytest.mark.asyncio
    async def test_parse_email_with_sender_in_angle_brackets(self, mock_gmail_service):
        """Test parsing sender with angle brackets format."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()
        connector.service = mock_gmail_service

        email_data = {
            'id': 'msg_angle',
            'threadId': 'thread_angle',
            'labelIds': ['INBOX'],
            'snippet': 'Test',
            'internalDate': '1729954800000',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'John Doe <john@example.com>'},
                    {'name': 'Subject', 'value': 'Test'}
                ],
                'body': {'data': 'VGVzdA=='}
            }
        }

        email_msg = await connector._parse_email_message(email_data)
        assert email_msg.sender == 'john@example.com'

    @pytest.mark.asyncio
    async def test_parse_email_with_html_body_in_multipart(self, mock_gmail_service):
        """Test extracting HTML body from multipart email."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()
        connector.service = mock_gmail_service

        html_email = {
            'id': 'msg_html',
            'threadId': 'thread_html',
            'labelIds': ['INBOX'],
            'snippet': 'HTML email',
            'internalDate': '1729954800000',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'sender@example.com'},
                    {'name': 'Subject', 'value': 'HTML Email'}
                ],
                'mimeType': 'multipart/alternative',
                'parts': [
                    {
                        'mimeType': 'text/html',
                        'body': {'data': 'PGh0bWw+VGVzdDwvaHRtbD4='}  # <html>Test</html>
                    }
                ]
            }
        }

        email_msg = await connector._parse_email_message(html_email)
        assert '<html>Test</html>' in email_msg.body

    @pytest.mark.asyncio
    async def test_parse_email_with_nested_multipart_body(self, mock_gmail_service):
        """Test extracting body from nested multipart structure."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()
        connector.service = mock_gmail_service

        nested_email = {
            'id': 'msg_nested',
            'threadId': 'thread_nested',
            'labelIds': ['INBOX'],
            'snippet': 'Nested',
            'internalDate': '1729954800000',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'sender@example.com'},
                    {'name': 'Subject', 'value': 'Nested'}
                ],
                'mimeType': 'multipart/mixed',
                'parts': [
                    {
                        'mimeType': 'multipart/alternative',
                        'parts': [
                            {
                                'mimeType': 'text/plain',
                                'body': {'data': 'TmVzdGVkIGJvZHk='}  # "Nested body"
                            }
                        ]
                    }
                ]
            }
        }

        email_msg = await connector._parse_email_message(nested_email)
        assert email_msg.body == 'Nested body'

    @pytest.mark.asyncio
    async def test_parse_email_with_no_body_content(self, mock_gmail_service):
        """Test parsing email with no body returns default message."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()
        connector.service = mock_gmail_service

        no_body_email = {
            'id': 'msg_nobody',
            'threadId': 'thread_nobody',
            'labelIds': ['INBOX'],
            'snippet': 'Empty',
            'internalDate': '1729954800000',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'sender@example.com'},
                    {'name': 'Subject', 'value': 'Empty'}
                ],
                'mimeType': 'multipart/alternative',
                'parts': []
            }
        }

        email_msg = await connector._parse_email_message(no_body_email)
        assert email_msg.body == '(No body content)'

    @pytest.mark.asyncio
    async def test_parse_email_with_important_label_boosts_score(self, mock_gmail_service):
        """Test IMPORTANT label increases importance score."""
        from src.connectors.email.gmail_connector import GmailConnector

        connector = GmailConnector()
        connector.service = mock_gmail_service

        important_email = {
            'id': 'msg_important',
            'threadId': 'thread_important',
            'labelIds': ['INBOX', 'IMPORTANT'],
            'snippet': 'Important',
            'internalDate': '1729954800000',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'sender@example.com'},
                    {'name': 'Subject', 'value': 'Important'}
                ],
                'body': {'data': 'VGVzdA=='}
            }
        }

        email_msg = await connector._parse_email_message(important_email)
        # Base score 0.5 + 0.1 for IMPORTANT label = 0.6
        assert email_msg.importance_score >= 0.6

    @pytest.mark.asyncio
    async def test_fetch_emails_handles_parse_failure(self, mock_gmail_service, mock_credentials):
        """Test that parse failures are logged but don't crash fetch."""
        from src.connectors.email.gmail_connector import GmailConnector

        # Mock list to return two message IDs
        list_mock = Mock()
        list_mock.execute.return_value = {
            'messages': [
                {'id': 'msg_valid', 'threadId': 'thread_1'},
                {'id': 'msg_invalid', 'threadId': 'thread_2'}
            ],
            'resultSizeEstimate': 2
        }
        mock_gmail_service.users().messages().list.return_value = list_mock

        # First message: valid
        # Second message: will raise exception during parsing
        get_mock = Mock()
        get_mock.execute.side_effect = [
            {
                'id': 'msg_valid',
                'threadId': 'thread_1',
                'labelIds': ['INBOX'],
                'snippet': 'Valid',
                'internalDate': '1729954800000',
                'payload': {
                    'headers': [
                        {'name': 'From', 'value': 'sender@example.com'},
                        {'name': 'Subject', 'value': 'Valid'}
                    ],
                    'body': {'data': 'VGVzdA=='}
                }
            },
            Exception("Parse error")  # This will trigger the exception handler
        ]
        mock_gmail_service.users().messages().get.return_value = get_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            # Should not crash, just log warning and skip invalid email
            emails = await connector.fetch_unread_emails()

            # Should have 1 valid email (invalid one was skipped)
            assert len(emails) == 1
            assert emails[0].id == 'msg_valid'

    @pytest.mark.asyncio
    async def test_fetch_message_details_retries_on_rate_limit(self, mock_gmail_service, mock_credentials):
        """Test _fetch_message_details retries on 429 errors."""
        from src.connectors.email.gmail_connector import GmailConnector
        from googleapiclient.errors import HttpError

        # First call: 429, second call: success
        get_mock = Mock()
        get_mock.execute.side_effect = [
            HttpError(resp=Mock(status=429), content=b'Rate limit'),
            {
                'id': MOCK_EMAIL_ID,
                'threadId': MOCK_THREAD_ID,
                'labelIds': ['INBOX'],
                'snippet': 'Test',
                'internalDate': '1729954800000',
                'payload': {
                    'headers': [
                        {'name': 'From', 'value': MOCK_SENDER},
                        {'name': 'Subject', 'value': MOCK_SUBJECT}
                    ],
                    'body': {'data': 'VGVzdA=='}
                }
            }
        ]
        mock_gmail_service.users().messages().get.return_value = get_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                connector = GmailConnector()
                connector.credentials = mock_credentials
                connector.service = mock_gmail_service

                msg_data = await connector._fetch_message_details(MOCK_EMAIL_ID, max_retries=3)

                assert msg_data['id'] == MOCK_EMAIL_ID
                assert get_mock.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_message_details_fails_after_max_retries(self, mock_gmail_service, mock_credentials):
        """Test _fetch_message_details raises HttpError after max retries."""
        from src.connectors.email.gmail_connector import GmailConnector
        from googleapiclient.errors import HttpError

        get_mock = Mock()
        get_mock.execute.side_effect = HttpError(resp=Mock(status=429), content=b'Rate limit')
        mock_gmail_service.users().messages().get.return_value = get_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                connector = GmailConnector()
                connector.credentials = mock_credentials
                connector.service = mock_gmail_service

                with pytest.raises(HttpError):
                    await connector._fetch_message_details(MOCK_EMAIL_ID, max_retries=2)

                # Verify it tried 3 times (initial + 2 retries)
                assert get_mock.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_mark_as_read_handles_errors(self, mock_gmail_service, mock_credentials):
        """Test mark_as_read logs and raises errors."""
        from src.connectors.email.gmail_connector import GmailConnector
        from googleapiclient.errors import HttpError

        modify_mock = Mock()
        modify_mock.execute.side_effect = HttpError(resp=Mock(status=500), content=b'Error')
        mock_gmail_service.users().messages().modify.return_value = modify_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            with pytest.raises(HttpError):
                await connector.mark_as_read(MOCK_EMAIL_ID)

    @pytest.mark.asyncio
    async def test_mark_as_read_batch_with_empty_list(self, mock_gmail_service, mock_credentials):
        """Test mark_as_read_batch handles empty list."""
        from src.connectors.email.gmail_connector import GmailConnector

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            # Should return early without making API call
            await connector.mark_as_read_batch([])

            mock_gmail_service.users().messages().batchModify.assert_not_called()

    @pytest.mark.asyncio
    async def test_mark_as_read_batch_handles_errors(self, mock_gmail_service, mock_credentials):
        """Test mark_as_read_batch logs and raises errors."""
        from src.connectors.email.gmail_connector import GmailConnector
        from googleapiclient.errors import HttpError

        batch_mock = Mock()
        batch_mock.execute.side_effect = HttpError(resp=Mock(status=500), content=b'Error')
        mock_gmail_service.users().messages().batchModify.return_value = batch_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            with pytest.raises(HttpError):
                await connector.mark_as_read_batch(['msg_1', 'msg_2'])

    @pytest.mark.asyncio
    async def test_add_labels_handles_errors(self, mock_gmail_service, mock_credentials):
        """Test add_labels logs and raises errors."""
        from src.connectors.email.gmail_connector import GmailConnector
        from googleapiclient.errors import HttpError

        modify_mock = Mock()
        modify_mock.execute.side_effect = HttpError(resp=Mock(status=500), content=b'Error')
        mock_gmail_service.users().messages().modify.return_value = modify_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            with pytest.raises(HttpError):
                await connector.add_labels(MOCK_EMAIL_ID, ['STARRED'])

    def test_authenticate_handles_general_exception(self, mock_gmail_service):
        """Test authentication handles general exceptions."""
        from src.connectors.email.gmail_connector import GmailConnector

        with patch('os.path.exists', return_value=True):
            with patch('google.oauth2.credentials.Credentials.from_authorized_user_file',
                       side_effect=Exception("Unexpected error")):
                connector = GmailConnector()

                with pytest.raises(Exception, match="Unexpected error"):
                    connector.authenticate(credentials_path="mock_token.json")

    @pytest.mark.asyncio
    async def test_fetch_message_details_handles_non_rate_limit_error(self, mock_gmail_service, mock_credentials):
        """Test _fetch_message_details raises non-429 errors immediately."""
        from src.connectors.email.gmail_connector import GmailConnector
        from googleapiclient.errors import HttpError

        get_mock = Mock()
        get_mock.execute.side_effect = HttpError(resp=Mock(status=500), content=b'Server error')
        mock_gmail_service.users().messages().get.return_value = get_mock

        with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
            connector = GmailConnector()
            connector.credentials = mock_credentials
            connector.service = mock_gmail_service

            with pytest.raises(HttpError):
                await connector._fetch_message_details(MOCK_EMAIL_ID, max_retries=3)

            # Should not retry on non-429 errors
            assert get_mock.execute.call_count == 1


class TestGmailSingleton:
    """Test singleton pattern for GmailConnector."""

    def test_get_gmail_connector_returns_singleton(self):
        """Test get_gmail_connector returns singleton instance."""
        from src.connectors.email.gmail_connector import get_gmail_connector

        connector1 = get_gmail_connector()
        connector2 = get_gmail_connector()

        assert connector1 is connector2

    def test_singleton_preserves_authentication(self, mock_credentials, mock_gmail_service):
        """Test singleton preserves authentication state."""
        from src.connectors.email.gmail_connector import get_gmail_connector

        with patch('os.path.exists', return_value=True):
            with patch('google.oauth2.credentials.Credentials.from_authorized_user_file',
                       return_value=mock_credentials):
                with patch('src.connectors.email.gmail_connector.build', return_value=mock_gmail_service):
                    with patch('builtins.open', mock_open()):
                        connector1 = get_gmail_connector()
                        connector1.authenticate(credentials_path="mock_token.json")

                        connector2 = get_gmail_connector()

                        # Should be the same instance with same auth state
                        assert connector2.is_authenticated is True
                        assert connector2.credentials == mock_credentials


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

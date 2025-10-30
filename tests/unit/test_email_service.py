"""Unit tests for EmailService - TDD validation.

Tests the service layer that orchestrates email fetching from multiple providers.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from src.core.models import EmailMessage
from src.services.email_service import EmailService, get_email_service


class TestEmailServiceInitialization:
    """Test EmailService initialization."""

    def test_initialization(self):
        """Test EmailService initializes successfully."""
        service = EmailService()
        assert service is not None
        assert service._email_agent is None
        assert service._connectors_initialized is False

    def test_singleton_pattern(self):
        """Test get_email_service returns same instance."""
        service1 = get_email_service()
        service2 = get_email_service()

        assert service1 is service2


class TestEmailServiceConnectorInitialization:
    """Test connector initialization logic."""

    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_outlook_connector')
    @patch('src.services.email_service.get_imap_connector')
    def test_initialize_all_connectors_success(
        self,
        mock_imap,
        mock_outlook,
        mock_gmail
    ):
        """Test initializing all connectors successfully."""
        # Setup mocks
        mock_gmail.return_value = Mock()
        mock_outlook.return_value = Mock()
        mock_imap.return_value = Mock()

        service = EmailService()
        connectors = service._initialize_connectors()

        # Should have all 3 connectors
        assert len(connectors) == 3
        assert "gmail" in connectors
        assert "outlook" in connectors
        assert "imap" in connectors

    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_outlook_connector')
    @patch('src.services.email_service.get_imap_connector')
    def test_initialize_partial_connectors(
        self,
        mock_imap,
        mock_outlook,
        mock_gmail
    ):
        """Test initializing with some connectors failing."""
        # Gmail succeeds, others fail
        mock_gmail.return_value = Mock()
        mock_outlook.side_effect = Exception("Outlook not configured")
        mock_imap.side_effect = Exception("IMAP not configured")

        service = EmailService()
        connectors = service._initialize_connectors()

        # Should only have Gmail
        assert len(connectors) == 1
        assert "gmail" in connectors

    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_outlook_connector')
    @patch('src.services.email_service.get_imap_connector')
    def test_initialize_no_connectors(
        self,
        mock_imap,
        mock_outlook,
        mock_gmail
    ):
        """Test initialization with no available connectors."""
        # All connectors fail
        mock_gmail.side_effect = Exception("Gmail not configured")
        mock_outlook.side_effect = Exception("Outlook not configured")
        mock_imap.side_effect = Exception("IMAP not configured")

        service = EmailService()
        connectors = service._initialize_connectors()

        # Should have no connectors
        assert len(connectors) == 0


class TestEmailServiceFetchEmails:
    """Test email fetching functionality."""

    @pytest.fixture
    def sample_emails(self) -> List[EmailMessage]:
        """Create sample emails for testing."""
        now = datetime.now()

        return [
            EmailMessage(
                id="1",
                subject="Important meeting",
                sender="alice@example.com",
                received_at=now,
                body="Meeting details",
                importance_score=0.9,
                is_read=False
            ),
            EmailMessage(
                id="2",
                subject="Newsletter",
                sender="news@example.com",
                received_at=now - timedelta(hours=1),
                body="Weekly newsletter",
                importance_score=0.3,
                is_read=True
            ),
            EmailMessage(
                id="3",
                subject="Action required",
                sender="bob@example.com",
                received_at=now - timedelta(hours=2),
                body="Please review",
                importance_score=0.8,
                has_action_items=True,
                action_items=["Review document"],
                is_read=False
            ),
        ]

    @pytest.mark.asyncio
    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_email_agent')
    async def test_fetch_emails_success(
        self,
        mock_get_agent,
        mock_gmail,
        sample_emails
    ):
        """Test successful email fetching."""
        # Setup mocks
        mock_connector = Mock()
        mock_gmail.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_emails = AsyncMock(return_value=sample_emails)
        mock_get_agent.return_value = mock_agent

        service = EmailService()
        emails = await service.fetch_emails(max_results=10)

        # Should return all emails
        assert len(emails) == 3
        assert mock_agent.fetch_emails.called

    @pytest.mark.asyncio
    @patch('src.services.email_service.get_gmail_connector')
    async def test_fetch_emails_no_connectors(self, mock_gmail):
        """Test fetching with no available connectors."""
        # No connectors available
        mock_gmail.side_effect = Exception("Gmail not configured")

        service = EmailService()
        emails = await service.fetch_emails()

        # Should return empty list
        assert len(emails) == 0

    @pytest.mark.asyncio
    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_email_agent')
    async def test_fetch_emails_with_filters(
        self,
        mock_get_agent,
        mock_gmail,
        sample_emails
    ):
        """Test fetching with filters applied."""
        # Setup mocks
        mock_connector = Mock()
        mock_gmail.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_emails = AsyncMock(return_value=sample_emails)
        mock_get_agent.return_value = mock_agent

        service = EmailService()

        # Filter for unread only
        emails = await service.fetch_emails(
            filter_unread=True,
            max_results=10
        )

        # Should only return unread emails (2 out of 3)
        assert len(emails) == 2
        assert all(not e.is_read for e in emails)

    @pytest.mark.asyncio
    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_email_agent')
    async def test_fetch_emails_with_importance_filter(
        self,
        mock_get_agent,
        mock_gmail,
        sample_emails
    ):
        """Test fetching with importance filter."""
        # Setup mocks
        mock_connector = Mock()
        mock_gmail.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_emails = AsyncMock(return_value=sample_emails)
        mock_get_agent.return_value = mock_agent

        service = EmailService()

        # Filter for important only (>= 0.7)
        emails = await service.fetch_emails(
            filter_important=True,
            max_results=10
        )

        # Should only return important emails (2 out of 3)
        assert len(emails) == 2
        assert all(e.importance_score >= 0.7 for e in emails)

    @pytest.mark.asyncio
    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_email_agent')
    async def test_fetch_emails_with_progress_callback(
        self,
        mock_get_agent,
        mock_gmail,
        sample_emails
    ):
        """Test fetching with progress callback."""
        # Setup mocks
        mock_connector = Mock()
        mock_gmail.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_emails = AsyncMock(return_value=sample_emails)
        mock_get_agent.return_value = mock_agent

        # Track progress callbacks
        progress_calls = []

        async def progress_callback(progress, message):
            progress_calls.append((progress, message))

        service = EmailService()
        emails = await service.fetch_emails(
            progress_callback=progress_callback
        )

        # Should have called progress callback multiple times
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 1.0  # Final progress should be 100%


class TestEmailServiceConvenienceMethods:
    """Test convenience methods."""

    @pytest.fixture
    def sample_emails(self) -> List[EmailMessage]:
        """Create sample emails for testing."""
        now = datetime.now()

        return [
            EmailMessage(
                id="1",
                subject="Important meeting",
                sender="alice@example.com",
                received_at=now,
                body="Meeting details",
                importance_score=0.9,
                is_read=False
            ),
            EmailMessage(
                id="2",
                subject="Action required",
                sender="bob@example.com",
                received_at=now - timedelta(hours=1),
                body="Please review",
                importance_score=0.8,
                has_action_items=True,
                action_items=["Review document"],
                is_read=False
            ),
            EmailMessage(
                id="3",
                subject="Newsletter",
                sender="news@example.com",
                received_at=now - timedelta(hours=2),
                body="Weekly newsletter",
                importance_score=0.3,
                is_read=True
            ),
        ]

    @pytest.mark.asyncio
    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_email_agent')
    async def test_get_unread_count(
        self,
        mock_get_agent,
        mock_gmail,
        sample_emails
    ):
        """Test getting unread count."""
        # Setup mocks
        mock_connector = Mock()
        mock_gmail.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_emails = AsyncMock(return_value=sample_emails)
        mock_get_agent.return_value = mock_agent

        service = EmailService()
        count = await service.get_unread_count()

        # Should count only unread (2 out of 3)
        assert count == 2

    @pytest.mark.asyncio
    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_email_agent')
    async def test_get_important_emails(
        self,
        mock_get_agent,
        mock_gmail,
        sample_emails
    ):
        """Test getting important emails."""
        # Setup mocks
        mock_connector = Mock()
        mock_gmail.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_emails = AsyncMock(return_value=sample_emails)
        mock_get_agent.return_value = mock_agent

        service = EmailService()
        emails = await service.get_important_emails(max_results=10)

        # Should return important emails (>= 0.7)
        assert len(emails) == 2
        assert all(e.importance_score >= 0.7 for e in emails)

    @pytest.mark.asyncio
    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_email_agent')
    async def test_get_emails_with_action_items(
        self,
        mock_get_agent,
        mock_gmail,
        sample_emails
    ):
        """Test getting emails with action items."""
        # Setup mocks
        mock_connector = Mock()
        mock_gmail.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_emails = AsyncMock(return_value=sample_emails)
        mock_get_agent.return_value = mock_agent

        service = EmailService()
        emails = await service.get_emails_with_action_items(max_results=10)

        # Should return emails with action items (1 out of 3)
        assert len(emails) == 1
        assert all(e.has_action_items for e in emails)


class TestEmailServiceErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    @patch('src.services.email_service.get_gmail_connector')
    @patch('src.services.email_service.get_email_agent')
    async def test_fetch_emails_agent_error(
        self,
        mock_get_agent,
        mock_gmail
    ):
        """Test handling agent errors."""
        # Setup mocks
        mock_connector = Mock()
        mock_gmail.return_value = mock_connector

        mock_agent = Mock()
        mock_agent.connectors = {}
        mock_agent.add_connector = Mock()
        mock_agent.fetch_emails = AsyncMock(side_effect=Exception("Agent error"))
        mock_get_agent.return_value = mock_agent

        service = EmailService()
        emails = await service.fetch_emails()

        # Should return empty list on error
        assert len(emails) == 0

    @pytest.mark.asyncio
    @patch('src.services.email_service.get_gmail_connector')
    async def test_get_unread_count_error(self, mock_gmail):
        """Test handling errors in get_unread_count."""
        # No connectors available
        mock_gmail.side_effect = Exception("Gmail not configured")

        service = EmailService()
        count = await service.get_unread_count()

        # Should return 0 on error
        assert count == 0

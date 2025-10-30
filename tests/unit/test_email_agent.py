#!/usr/bin/env python3
"""Unit tests for EmailAgent (TDD approach).

Following TDD: Write tests first, then implement EmailAgent.

The EmailAgent orchestrates multiple email connectors (Gmail, Outlook, IMAP)
and provides a unified interface for fetching, filtering, and prioritizing emails.

Mocking Strategy:
- Mock email connectors (Gmail, Outlook, IMAP)
- Mock database and metrics managers
- Verify correct connector selection
- Verify email prioritization and filtering
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import List

from src.core.models import EmailMessage


# Test constants
MOCK_EMAILS = [
    EmailMessage(
        id="1",
        subject="Urgent: Project Deadline",
        sender="boss@company.com",
        received_at=datetime.now(),
        body="This is urgent",
        snippet="This is urgent",
        labels={"UNREAD", "IMPORTANT"},
        importance_score=0.9
    ),
    EmailMessage(
        id="2",
        subject="FYI: Newsletter",
        sender="newsletter@updates.com",
        received_at=datetime.now() - timedelta(hours=2),
        body="Weekly update",
        snippet="Weekly update",
        labels={"UNREAD"},
        importance_score=0.3
    ),
    EmailMessage(
        id="3",
        subject="Meeting Invite",
        sender="colleague@company.com",
        received_at=datetime.now() - timedelta(hours=1),
        body="Team meeting tomorrow",
        snippet="Team meeting tomorrow",
        labels={"UNREAD"},
        importance_score=0.6
    )
]


@pytest.fixture
def mock_gmail_connector():
    """Create mock Gmail connector."""
    connector = AsyncMock()
    connector.is_authenticated = True
    connector.fetch_unread_emails.return_value = [MOCK_EMAILS[0]]
    return connector


@pytest.fixture
def mock_outlook_connector():
    """Create mock Outlook connector."""
    connector = AsyncMock()
    connector.is_authenticated = True
    connector.fetch_unread_emails.return_value = [MOCK_EMAILS[1]]
    return connector


@pytest.fixture
def mock_imap_connector():
    """Create mock IMAP connector."""
    connector = AsyncMock()
    connector.is_authenticated = True
    connector.fetch_unread_emails.return_value = [MOCK_EMAILS[2]]
    return connector


class TestEmailAgentInitialization:
    """Test EmailAgent initialization and configuration."""

    def test_agent_initialization_no_connectors(self):
        """Test EmailAgent can be initialized without connectors."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent()

        assert agent is not None
        assert len(agent.connectors) == 0

    def test_agent_initialization_with_connectors(self, mock_gmail_connector):
        """Test EmailAgent initialization with connectors."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={"gmail": mock_gmail_connector})

        assert agent is not None
        assert len(agent.connectors) == 1
        assert "gmail" in agent.connectors

    def test_add_connector(self, mock_gmail_connector):
        """Test adding a connector after initialization."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent()
        agent.add_connector("gmail", mock_gmail_connector)

        assert len(agent.connectors) == 1
        assert "gmail" in agent.connectors


class TestEmailAgentFetching:
    """Test fetching emails from multiple connectors."""

    @pytest.mark.asyncio
    async def test_fetch_emails_from_single_connector(self, mock_gmail_connector):
        """Test fetching emails from a single connector."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={"gmail": mock_gmail_connector})
        emails = await agent.fetch_emails(max_results=10)

        assert len(emails) == 1
        assert emails[0].subject == "Urgent: Project Deadline"
        mock_gmail_connector.fetch_unread_emails.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_emails_from_multiple_connectors(
        self, mock_gmail_connector, mock_outlook_connector, mock_imap_connector
    ):
        """Test fetching emails from multiple connectors."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={
            "gmail": mock_gmail_connector,
            "outlook": mock_outlook_connector,
            "imap": mock_imap_connector
        })

        emails = await agent.fetch_emails(max_results=10)

        # Should fetch from all connectors and merge results
        assert len(emails) == 3
        mock_gmail_connector.fetch_unread_emails.assert_called_once()
        mock_outlook_connector.fetch_unread_emails.assert_called_once()
        mock_imap_connector.fetch_unread_emails.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_emails_sorted_by_importance(
        self, mock_gmail_connector, mock_outlook_connector, mock_imap_connector
    ):
        """Test emails are sorted by importance score (descending)."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={
            "gmail": mock_gmail_connector,
            "outlook": mock_outlook_connector,
            "imap": mock_imap_connector
        })

        emails = await agent.fetch_emails(max_results=10, sort_by_importance=True)

        # Should be sorted: 0.9 (gmail), 0.6 (imap), 0.3 (outlook)
        assert len(emails) == 3
        assert emails[0].importance_score == 0.9
        assert emails[1].importance_score == 0.6
        assert emails[2].importance_score == 0.3

    @pytest.mark.asyncio
    async def test_fetch_emails_respects_max_results(
        self, mock_gmail_connector, mock_outlook_connector
    ):
        """Test fetch respects max_results limit."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={
            "gmail": mock_gmail_connector,
            "outlook": mock_outlook_connector
        })

        emails = await agent.fetch_emails(max_results=1)

        # Should return only 1 email even though 2 connectors
        assert len(emails) <= 1

    @pytest.mark.asyncio
    async def test_fetch_emails_handles_connector_failure(
        self, mock_gmail_connector, mock_outlook_connector
    ):
        """Test fetch continues even if one connector fails."""
        from src.agents.email_agent import EmailAgent

        # Make Gmail fail
        mock_gmail_connector.fetch_unread_emails.side_effect = Exception("Gmail failed")

        agent = EmailAgent(connectors={
            "gmail": mock_gmail_connector,
            "outlook": mock_outlook_connector
        })

        emails = await agent.fetch_emails(max_results=10)

        # Should still get emails from Outlook
        assert len(emails) == 1
        assert emails[0].sender == "newsletter@updates.com"


class TestEmailAgentFiltering:
    """Test email filtering capabilities."""

    @pytest.mark.asyncio
    async def test_filter_by_sender(
        self, mock_gmail_connector, mock_outlook_connector, mock_imap_connector
    ):
        """Test filtering emails by sender."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={
            "gmail": mock_gmail_connector,
            "outlook": mock_outlook_connector,
            "imap": mock_imap_connector
        })

        emails = await agent.fetch_emails(
            max_results=10,
            filter_sender="boss@company.com"
        )

        # Should only return emails from boss
        assert len(emails) == 1
        assert emails[0].sender == "boss@company.com"

    @pytest.mark.asyncio
    async def test_filter_by_subject_keyword(
        self, mock_gmail_connector, mock_outlook_connector, mock_imap_connector
    ):
        """Test filtering emails by subject keyword."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={
            "gmail": mock_gmail_connector,
            "outlook": mock_outlook_connector,
            "imap": mock_imap_connector
        })

        emails = await agent.fetch_emails(
            max_results=10,
            filter_subject="urgent"
        )

        # Should only return emails with "urgent" in subject
        assert len(emails) == 1
        assert "Urgent" in emails[0].subject

    @pytest.mark.asyncio
    async def test_filter_by_minimum_importance(
        self, mock_gmail_connector, mock_outlook_connector, mock_imap_connector
    ):
        """Test filtering emails by minimum importance score."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={
            "gmail": mock_gmail_connector,
            "outlook": mock_outlook_connector,
            "imap": mock_imap_connector
        })

        emails = await agent.fetch_emails(
            max_results=10,
            min_importance=0.5
        )

        # Should only return emails with importance >= 0.5
        assert len(emails) == 2
        assert all(email.importance_score >= 0.5 for email in emails)


class TestEmailAgentCaching:
    """Test email caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_emails(self, mock_gmail_connector):
        """Test emails are cached after fetching."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={"gmail": mock_gmail_connector})

        # First fetch
        emails1 = await agent.fetch_emails(max_results=10)

        # Second fetch should use cache
        emails2 = await agent.fetch_emails(max_results=10, use_cache=True)

        # Should only call connector once
        assert mock_gmail_connector.fetch_unread_emails.call_count == 1
        assert len(emails1) == len(emails2)

    @pytest.mark.asyncio
    async def test_cache_expiration(self, mock_gmail_connector):
        """Test cache expires after TTL."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(
            connectors={"gmail": mock_gmail_connector},
            cache_ttl_seconds=1  # 1 second TTL
        )

        # First fetch
        await agent.fetch_emails(max_results=10, use_cache=True)

        # Wait for cache to expire
        import asyncio
        await asyncio.sleep(1.5)

        # Second fetch should not use cache
        await agent.fetch_emails(max_results=10, use_cache=True)

        # Should call connector twice (cache expired)
        assert mock_gmail_connector.fetch_unread_emails.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_gmail_connector):
        """Test fetching without cache."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={"gmail": mock_gmail_connector})

        # Fetch twice without cache
        await agent.fetch_emails(max_results=10, use_cache=False)
        await agent.fetch_emails(max_results=10, use_cache=False)

        # Should call connector twice
        assert mock_gmail_connector.fetch_unread_emails.call_count == 2


class TestEmailAgentObservability:
    """Test metrics and logging."""

    @pytest.mark.asyncio
    async def test_fetch_emails_emits_metrics(self, mock_gmail_connector):
        """Test fetch_emails emits metrics."""
        from src.agents.email_agent import EmailAgent

        mock_metrics = Mock()

        with patch('src.agents.email_agent.get_metrics_manager', return_value=mock_metrics):
            agent = EmailAgent(connectors={"gmail": mock_gmail_connector})
            await agent.fetch_emails(max_results=10)

            # Verify metrics were emitted
            mock_metrics.emit.assert_called()

    @pytest.mark.asyncio
    async def test_fetch_emails_logs_activity(self, mock_gmail_connector):
        """Test fetch_emails logs activity to database."""
        from src.agents.email_agent import EmailAgent

        mock_db = AsyncMock()

        with patch('src.agents.email_agent.get_db_manager') as mock_get_db:
            mock_get_db.return_value.get_connection.return_value.__aenter__.return_value = mock_db

            agent = EmailAgent(connectors={"gmail": mock_gmail_connector})
            await agent.fetch_emails(max_results=10)

            # Verify activity was logged
            mock_db.execute.assert_called()


class TestEmailAgentStatistics:
    """Test email statistics and analytics."""

    @pytest.mark.asyncio
    async def test_get_email_count_by_sender(
        self, mock_gmail_connector, mock_outlook_connector, mock_imap_connector
    ):
        """Test getting email count grouped by sender."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={
            "gmail": mock_gmail_connector,
            "outlook": mock_outlook_connector,
            "imap": mock_imap_connector
        })

        # Fetch emails first
        await agent.fetch_emails(max_results=10)

        # Get statistics
        stats = agent.get_sender_statistics()

        assert len(stats) == 3
        assert "boss@company.com" in stats
        assert "newsletter@updates.com" in stats
        assert "colleague@company.com" in stats

    @pytest.mark.asyncio
    async def test_get_average_importance_score(
        self, mock_gmail_connector, mock_outlook_connector, mock_imap_connector
    ):
        """Test calculating average importance score."""
        from src.agents.email_agent import EmailAgent

        agent = EmailAgent(connectors={
            "gmail": mock_gmail_connector,
            "outlook": mock_outlook_connector,
            "imap": mock_imap_connector
        })

        # Fetch emails first
        await agent.fetch_emails(max_results=10)

        # Get average importance
        avg_importance = agent.get_average_importance()

        # Average of 0.9, 0.3, 0.6 = 0.6
        assert avg_importance == pytest.approx(0.6, rel=0.1)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

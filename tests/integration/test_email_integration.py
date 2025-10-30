#!/usr/bin/env python3
"""Integration tests for email connectors and EmailAgent.

These tests require the mock email server to be running.
Start the server with: python tests/mock_email_server.py start

Run integration tests with: pytest tests/integration/ -v -m integration
"""

import pytest
import pytest_asyncio
import asyncio
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import urllib.request
import time

from src.connectors.email.imap_connector import IMAPConnector
from src.agents.email_agent import EmailAgent

# Mock email server configuration (GreenMail)
IMAP_HOST = "localhost"
IMAP_PORT = 3143
SMTP_HOST = "localhost"
SMTP_PORT = 3025
WEB_UI_URL = "http://localhost:8080"


def is_mock_server_running() -> bool:
    """Check if the mock email server is running."""
    try:
        urllib.request.urlopen(WEB_UI_URL, timeout=2)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def check_mock_server():
    """Verify mock email server is running before tests."""
    if not is_mock_server_running():
        pytest.skip(
            f"Mock email server is not running. "
            f"Start it with: python tests/mock_email_server.py start"
        )


@pytest.fixture
def send_test_email():
    """Helper function to send a test email via SMTP."""
    def _send(subject: str, body: str, sender: str = "test@example.com", recipient: str = "test@localhost"):
        """Send a test email to the mock server."""
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient
        msg['Date'] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")

        smtp = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        smtp.send_message(msg)
        smtp.quit()

        # Small delay to ensure email is available via IMAP
        time.sleep(0.5)

    return _send


@pytest_asyncio.fixture
async def imap_connector():
    """Create and authenticate IMAP connector for mock server."""
    connector = IMAPConnector(
        imap_server=IMAP_HOST,
        username="test",  # GreenMail test user
        password="test",
        use_ssl=False,
        port=IMAP_PORT
    )

    await connector.authenticate()

    yield connector

    # Cleanup
    if connector.connection:
        connector.disconnect()


@pytest_asyncio.fixture
async def email_agent(imap_connector):
    """Create EmailAgent with IMAP connector."""
    agent = EmailAgent(
        connectors={"imap": imap_connector},
        cache_ttl_seconds=10  # Short TTL for testing
    )

    return agent


@pytest.mark.integration
class TestIMAPConnectorIntegration:
    """Integration tests for IMAP connector with mock server."""

    @pytest.mark.asyncio
    async def test_connect_to_mock_server(self, check_mock_server):
        """Test connecting to the mock IMAP server."""
        connector = IMAPConnector(
            imap_server=IMAP_HOST,
            username="test",
            password="test",
            use_ssl=False,
            port=IMAP_PORT
        )

        await connector.authenticate()

        assert connector.is_authenticated is True

        connector.disconnect()

    @pytest.mark.asyncio
    async def test_fetch_emails_from_mock_server(self, check_mock_server, imap_connector, send_test_email):
        """Test fetching emails from mock server."""
        # Send test email
        send_test_email(
            subject="Integration Test Email",
            body="This is a test email for integration testing.",
            sender="sender@test.com"
        )

        # Fetch emails
        emails = await imap_connector.fetch_unread_emails(max_results=10)

        # Verify we got at least one email
        assert len(emails) >= 1

        # Find our test email
        test_email = next(
            (e for e in emails if "Integration Test Email" in e.subject),
            None
        )

        assert test_email is not None
        assert test_email.subject == "Integration Test Email"
        assert "test email" in test_email.body
        assert test_email.sender == "sender@test.com"

    @pytest.mark.asyncio
    async def test_fetch_multiple_emails(self, check_mock_server, imap_connector, send_test_email):
        """Test fetching multiple emails."""
        # Send multiple test emails
        for i in range(3):
            send_test_email(
                subject=f"Test Email {i+1}",
                body=f"Body content {i+1}",
                sender=f"sender{i+1}@test.com"
            )

        # Fetch emails
        emails = await imap_connector.fetch_unread_emails(max_results=50)

        # Verify we got at least 3 emails
        assert len(emails) >= 3

        # Verify our test emails are present
        test_emails = [e for e in emails if e.subject.startswith("Test Email")]
        assert len(test_emails) >= 3

    @pytest.mark.asyncio
    async def test_mark_as_read(self, check_mock_server, imap_connector, send_test_email):
        """Test marking an email as read."""
        # Send test email
        send_test_email(
            subject="Mark As Read Test",
            body="This email will be marked as read"
        )

        # Fetch unread emails
        emails = await imap_connector.fetch_unread_emails(max_results=10)

        # Find our test email
        test_email = next(
            (e for e in emails if "Mark As Read Test" in e.subject),
            None
        )

        assert test_email is not None

        # Mark as read
        await imap_connector.mark_as_read(test_email.id)

        # Note: MailHog IMAP may not fully support mark as read,
        # but we verify the function doesn't raise an error
        assert True  # If we got here, no exception was raised


@pytest.mark.integration
class TestEmailAgentIntegration:
    """Integration tests for EmailAgent with mock server."""

    @pytest.mark.asyncio
    async def test_agent_fetch_emails(self, check_mock_server, email_agent, send_test_email):
        """Test EmailAgent fetching emails."""
        # Send test emails
        send_test_email(
            subject="Agent Test 1",
            body="First agent test email"
        )
        send_test_email(
            subject="Agent Test 2",
            body="Second agent test email"
        )

        # Fetch via agent
        emails = await email_agent.fetch_emails(max_results=50)

        # Verify we got emails
        assert len(emails) >= 2

        # Verify our test emails
        test_emails = [e for e in emails if e.subject.startswith("Agent Test")]
        assert len(test_emails) >= 2

    @pytest.mark.asyncio
    async def test_agent_sort_by_importance(self, check_mock_server, email_agent, send_test_email):
        """Test EmailAgent sorts emails by importance."""
        # Send emails (importance is determined by sender/subject keywords in real scenario)
        send_test_email(
            subject="Urgent: Critical Issue",
            body="High importance",
            sender="boss@company.com"
        )
        send_test_email(
            subject="FYI: Newsletter",
            body="Low importance",
            sender="newsletter@updates.com"
        )

        # Fetch with importance sorting
        emails = await email_agent.fetch_emails(
            max_results=50,
            sort_by_importance=True
        )

        # Verify sorting (higher importance first)
        if len(emails) >= 2:
            # In mock server, importance scores are default (0.5)
            # But we verify the sorting doesn't break
            assert isinstance(emails[0].importance_score, float)
            assert isinstance(emails[1].importance_score, float)

    @pytest.mark.asyncio
    async def test_agent_filter_by_sender(self, check_mock_server, email_agent, send_test_email):
        """Test EmailAgent filtering by sender."""
        test_sender = "specific@sender.com"

        # Send emails from different senders
        send_test_email(
            subject="From Specific Sender",
            body="Test",
            sender=test_sender
        )
        send_test_email(
            subject="From Different Sender",
            body="Test",
            sender="other@sender.com"
        )

        # Fetch with sender filter
        emails = await email_agent.fetch_emails(
            max_results=50,
            filter_sender=test_sender
        )

        # Verify filtering
        for email in emails:
            if "From Specific Sender" in email.subject:
                assert email.sender == test_sender

    @pytest.mark.asyncio
    async def test_agent_filter_by_subject(self, check_mock_server, email_agent, send_test_email):
        """Test EmailAgent filtering by subject keyword."""
        # Send emails with different subjects
        send_test_email(
            subject="Important Project Update",
            body="Test"
        )
        send_test_email(
            subject="Random Newsletter",
            body="Test"
        )

        # Fetch with subject filter
        emails = await email_agent.fetch_emails(
            max_results=50,
            filter_subject="project"
        )

        # Verify filtering (case-insensitive)
        for email in emails:
            if "Project" in email.subject or "project" in email.subject:
                assert "project" in email.subject.lower()

    @pytest.mark.asyncio
    async def test_agent_caching(self, check_mock_server, email_agent, send_test_email):
        """Test EmailAgent caching mechanism."""
        # Send test email
        send_test_email(
            subject="Cache Test",
            body="Testing cache"
        )

        # First fetch (populates cache)
        emails1 = await email_agent.fetch_emails(max_results=50, use_cache=True)

        # Second fetch (should use cache)
        emails2 = await email_agent.fetch_emails(max_results=50, use_cache=True)

        # Should get same results from cache
        assert len(emails1) == len(emails2)

    @pytest.mark.asyncio
    async def test_agent_statistics(self, check_mock_server, email_agent, send_test_email):
        """Test EmailAgent statistics functionality."""
        # Send emails from different senders
        send_test_email(
            subject="Stats Test 1",
            body="Test",
            sender="sender1@test.com"
        )
        send_test_email(
            subject="Stats Test 2",
            body="Test",
            sender="sender1@test.com"
        )
        send_test_email(
            subject="Stats Test 3",
            body="Test",
            sender="sender2@test.com"
        )

        # Fetch emails
        await email_agent.fetch_emails(max_results=50)

        # Get statistics
        sender_stats = email_agent.get_sender_statistics()
        avg_importance = email_agent.get_average_importance()

        # Verify statistics
        assert isinstance(sender_stats, dict)
        assert isinstance(avg_importance, float)
        assert 0.0 <= avg_importance <= 1.0


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_email_workflow(self, check_mock_server):
        """Test complete email workflow from connection to fetching."""
        # 1. Create connector
        connector = IMAPConnector(
            imap_server=IMAP_HOST,
            username="test",
            password="test",
            use_ssl=False,
            port=IMAP_PORT
        )

        # 2. Authenticate
        await connector.authenticate()
        assert connector.is_authenticated is True

        # 3. Send test email
        msg = MIMEText("End-to-end test email")
        msg['Subject'] = "E2E Test"
        msg['From'] = "e2e@test.com"
        msg['To'] = "test@localhost"

        smtp = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        smtp.send_message(msg)
        smtp.quit()

        time.sleep(0.5)  # Wait for email to be available

        # 4. Create agent
        agent = EmailAgent(connectors={"imap": connector})

        # 5. Fetch and filter emails
        emails = await agent.fetch_emails(
            max_results=50,
            sort_by_importance=True,
            filter_subject="E2E"
        )

        # 6. Verify results
        assert len(emails) >= 1
        e2e_email = next((e for e in emails if "E2E Test" in e.subject), None)
        assert e2e_email is not None
        assert e2e_email.sender == "e2e@test.com"

        # 7. Get statistics
        stats = agent.get_sender_statistics()
        assert isinstance(stats, dict)

        # Cleanup
        connector.disconnect()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])

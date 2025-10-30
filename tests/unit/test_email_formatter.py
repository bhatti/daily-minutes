"""Unit tests for EmailFormatter - TDD Red Phase.

The EmailFormatter is responsible for business logic related to formatting
email data for display. It should NOT contain any UI rendering code.

Business logic includes:
- Grouping emails by date, sender, or importance
- Calculating statistics (unread count, total size, etc.)
- Formatting dates/times for display
- Extracting display-friendly summaries
- Priority calculations for display ordering
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from src.core.models import EmailMessage
from src.ui.formatters.email_formatter import EmailFormatter


class TestEmailFormatterInitialization:
    """Test EmailFormatter initialization."""

    def test_initialization(self):
        """Test EmailFormatter initializes successfully."""
        formatter = EmailFormatter()
        assert formatter is not None

    def test_singleton_pattern(self):
        """Test get_email_formatter returns same instance."""
        from src.ui.formatters.email_formatter import get_email_formatter

        formatter1 = get_email_formatter()
        formatter2 = get_email_formatter()

        assert formatter1 is formatter2


class TestEmailFormatterGrouping:
    """Test email grouping logic."""

    @pytest.fixture
    def sample_emails(self) -> List[EmailMessage]:
        """Create sample emails for testing."""
        now = datetime.now()

        return [
            EmailMessage(
                id="1",
                subject="Today's meeting",
                sender="alice@example.com",
                received_at=now,
                body="Meeting discussion",
                importance_score=0.9,
                has_action_items=True,
                action_items=["Review proposal"]
            ),
            EmailMessage(
                id="2",
                subject="Yesterday's update",
                sender="bob@example.com",
                received_at=now - timedelta(days=1),
                body="Status update",
                importance_score=0.5,
                has_action_items=False
            ),
            EmailMessage(
                id="3",
                subject="Another today item",
                sender="alice@example.com",
                received_at=now - timedelta(hours=2),
                body="Today's content",
                importance_score=0.7,
                has_action_items=False
            ),
        ]

    def test_group_by_date(self, sample_emails):
        """Test grouping emails by date."""
        formatter = EmailFormatter()

        grouped = formatter.group_by_date(sample_emails)

        # Should have 2 date groups (today and yesterday)
        assert len(grouped) == 2

        # Check group keys are date strings
        for date_str, emails in grouped.items():
            assert isinstance(date_str, str)
            assert isinstance(emails, list)
            assert all(isinstance(e, EmailMessage) for e in emails)

    def test_group_by_sender(self, sample_emails):
        """Test grouping emails by sender."""
        formatter = EmailFormatter()

        grouped = formatter.group_by_sender(sample_emails)

        # Should have 2 senders (alice and bob)
        assert len(grouped) == 2
        assert "alice@example.com" in grouped
        assert "bob@example.com" in grouped

        # Alice sent 2 emails
        assert len(grouped["alice@example.com"]) == 2
        # Bob sent 1 email
        assert len(grouped["bob@example.com"]) == 1

    def test_group_by_importance(self, sample_emails):
        """Test grouping emails by importance level."""
        formatter = EmailFormatter()

        grouped = formatter.group_by_importance(sample_emails)

        # Should have 3 importance levels: high, medium, low
        assert "high" in grouped  # importance_score >= 0.7
        assert "medium" in grouped or "low" in grouped

        # High importance should include email with 0.9 and 0.7 scores
        high_importance = grouped.get("high", [])
        assert len(high_importance) >= 1


class TestEmailFormatterStatistics:
    """Test email statistics calculation."""

    @pytest.fixture
    def sample_emails(self) -> List[EmailMessage]:
        """Create sample emails for testing."""
        now = datetime.now()

        return [
            EmailMessage(
                id="1",
                subject="Unread 1",
                sender="alice@example.com",
                received_at=now,
                body="Unread email body",
                importance_score=0.9,
                has_action_items=True,
                action_items=["Review proposal"],
                is_read=False
            ),
            EmailMessage(
                id="2",
                subject="Read email",
                sender="bob@example.com",
                received_at=now,
                body="Read email body",
                importance_score=0.5,
                has_action_items=False,
                is_read=True
            ),
            EmailMessage(
                id="3",
                subject="Unread 2",
                sender="charlie@example.com",
                received_at=now,
                body="Another unread email",
                importance_score=0.7,
                has_action_items=True,
                action_items=["Schedule call"],
                is_read=False
            ),
        ]

    def test_calculate_statistics(self, sample_emails):
        """Test calculating email statistics."""
        formatter = EmailFormatter()

        stats = formatter.calculate_statistics(sample_emails)

        assert stats["total"] == 3
        assert stats["unread"] == 2
        assert stats["read"] == 1
        assert stats["with_action_items"] == 2
        assert stats["high_importance"] >= 1  # At least one high importance email

    def test_get_unread_count(self, sample_emails):
        """Test getting unread count."""
        formatter = EmailFormatter()

        unread_count = formatter.get_unread_count(sample_emails)

        assert unread_count == 2

    def test_get_action_items_count(self, sample_emails):
        """Test counting emails with action items."""
        formatter = EmailFormatter()

        action_count = formatter.get_action_items_count(sample_emails)

        assert action_count == 2


class TestEmailFormatterFormatting:
    """Test email data formatting for display."""

    @pytest.fixture
    def sample_email(self) -> EmailMessage:
        """Create a sample email."""
        return EmailMessage(
            id="1",
            subject="Meeting tomorrow at 2pm",
            sender="alice@example.com",
            received_at=datetime(2025, 10, 26, 14, 30, 0),
            body="Let's discuss the quarterly results.",
            importance_score=0.9,
            has_action_items=True,
            action_items=["Review agenda", "Prepare slides"]
        )

    def test_format_timestamp(self, sample_email):
        """Test formatting timestamp for display."""
        formatter = EmailFormatter()

        formatted = formatter.format_timestamp(sample_email.received_at)

        # Should return a human-readable string
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_relative_time(self, sample_email):
        """Test formatting relative time (e.g., '2 hours ago')."""
        formatter = EmailFormatter()

        # Test with recent email
        recent_email = EmailMessage(
            id="1",
            subject="Test",
            sender="test@example.com",
            received_at=datetime.now() - timedelta(hours=2),
            body="Test email body"
        )

        relative = formatter.format_relative_time(recent_email.received_at)

        # Should contain time reference
        assert isinstance(relative, str)
        # Examples: "2 hours ago", "just now", "yesterday"
        assert len(relative) > 0

    def test_format_sender_display(self, sample_email):
        """Test formatting sender for display."""
        formatter = EmailFormatter()

        display = formatter.format_sender_display(sample_email.sender)

        # Should extract name or email
        assert isinstance(display, str)
        assert len(display) > 0

    def test_get_preview_text(self, sample_email):
        """Test getting preview text from email body."""
        formatter = EmailFormatter()

        preview = formatter.get_preview_text(sample_email, max_length=50)

        # Should return truncated text
        assert isinstance(preview, str)
        assert len(preview) <= 50 + 3  # +3 for potential "..."

    def test_format_action_items_list(self, sample_email):
        """Test formatting action items as list."""
        formatter = EmailFormatter()

        formatted = formatter.format_action_items_list(sample_email.action_items)

        # Should return list of formatted strings
        assert isinstance(formatted, list)
        assert len(formatted) == 2
        assert all(isinstance(item, str) for item in formatted)


class TestEmailFormatterSorting:
    """Test email sorting logic."""

    @pytest.fixture
    def sample_emails(self) -> List[EmailMessage]:
        """Create sample emails for testing."""
        now = datetime.now()

        return [
            EmailMessage(
                id="1",
                subject="Low priority",
                sender="alice@example.com",
                received_at=now - timedelta(hours=1),
                body="Low priority email",
                importance_score=0.3
            ),
            EmailMessage(
                id="2",
                subject="High priority",
                sender="bob@example.com",
                received_at=now - timedelta(hours=2),
                body="High priority email",
                importance_score=0.9
            ),
            EmailMessage(
                id="3",
                subject="Medium priority",
                sender="charlie@example.com",
                received_at=now,
                body="Medium priority email",
                importance_score=0.6
            ),
        ]

    def test_sort_by_importance(self, sample_emails):
        """Test sorting emails by importance score."""
        formatter = EmailFormatter()

        sorted_emails = formatter.sort_by_importance(sample_emails)

        # Should be in descending order of importance
        assert len(sorted_emails) == 3
        assert sorted_emails[0].importance_score >= sorted_emails[1].importance_score
        assert sorted_emails[1].importance_score >= sorted_emails[2].importance_score

    def test_sort_by_timestamp(self, sample_emails):
        """Test sorting emails by timestamp (newest first)."""
        formatter = EmailFormatter()

        sorted_emails = formatter.sort_by_timestamp(sample_emails, newest_first=True)

        # Should be in descending order of timestamp
        assert len(sorted_emails) == 3
        assert sorted_emails[0].received_at >= sorted_emails[1].received_at
        assert sorted_emails[1].received_at >= sorted_emails[2].received_at

    def test_sort_by_timestamp_oldest_first(self, sample_emails):
        """Test sorting emails by timestamp (oldest first)."""
        formatter = EmailFormatter()

        sorted_emails = formatter.sort_by_timestamp(sample_emails, newest_first=False)

        # Should be in ascending order of timestamp
        assert len(sorted_emails) == 3
        assert sorted_emails[0].received_at <= sorted_emails[1].received_at
        assert sorted_emails[1].received_at <= sorted_emails[2].received_at


class TestEmailFormatterFiltering:
    """Test email filtering logic."""

    @pytest.fixture
    def sample_emails(self) -> List[EmailMessage]:
        """Create sample emails for testing."""
        now = datetime.now()

        return [
            EmailMessage(
                id="1",
                subject="Meeting about project",
                sender="alice@example.com",
                received_at=now,
                body="Project meeting email",
                importance_score=0.9,
                is_read=False
            ),
            EmailMessage(
                id="2",
                subject="Lunch plans",
                sender="bob@example.com",
                received_at=now,
                body="Lunch email",
                importance_score=0.3,
                is_read=True
            ),
            EmailMessage(
                id="3",
                subject="Project update",
                sender="charlie@example.com",
                received_at=now,
                body="Project update email",
                importance_score=0.7,
                is_read=False
            ),
        ]

    def test_filter_unread(self, sample_emails):
        """Test filtering unread emails."""
        formatter = EmailFormatter()

        unread = formatter.filter_unread(sample_emails)

        assert len(unread) == 2
        assert all(not email.is_read for email in unread)

    def test_filter_by_sender(self, sample_emails):
        """Test filtering by sender."""
        formatter = EmailFormatter()

        alice_emails = formatter.filter_by_sender(sample_emails, "alice@example.com")

        assert len(alice_emails) == 1
        assert alice_emails[0].sender == "alice@example.com"

    def test_filter_by_subject(self, sample_emails):
        """Test filtering by subject keyword."""
        formatter = EmailFormatter()

        project_emails = formatter.filter_by_subject(sample_emails, "project")

        # Should match both "Meeting about project" and "Project update"
        assert len(project_emails) == 2
        assert all("project" in email.subject.lower() for email in project_emails)

    def test_filter_high_importance(self, sample_emails):
        """Test filtering high importance emails."""
        formatter = EmailFormatter()

        high_importance = formatter.filter_high_importance(sample_emails, threshold=0.7)

        # Should include emails with score >= 0.7
        assert len(high_importance) == 2
        assert all(email.importance_score >= 0.7 for email in high_importance)

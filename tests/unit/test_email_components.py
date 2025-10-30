"""Unit tests for email UI component helper functions - TDD RED phase.

The UI components should be thin rendering layers. We test:
1. Helper functions that prepare data for display
2. State management logic
3. Integration with formatters and services

Pure Streamlit rendering code is not unit tested - instead covered by integration tests.
"""

import pytest
from datetime import datetime, timedelta
from typing import List
from unittest.mock import Mock, AsyncMock, patch

from src.core.models import EmailMessage
from src.ui.components.email_components import (
    prepare_email_display_data,
    get_importance_badge,
    get_action_items_display,
    should_show_email,
    group_emails_for_display,
)


class TestEmailDisplayDataPreparation:
    """Test helper functions that prepare email data for display."""

    @pytest.fixture
    def sample_email(self) -> EmailMessage:
        """Create a sample email for testing."""
        return EmailMessage(
            id="1",
            subject="Important meeting tomorrow",
            sender="alice@example.com",
            received_at=datetime.now() - timedelta(hours=2),
            body="Please review the attached document before our meeting.",
            importance_score=0.9,
            has_action_items=True,
            action_items=["Review document"],
            is_read=False
        )

    def test_prepare_email_display_data(self, sample_email):
        """Test preparing email data for display."""
        display_data = prepare_email_display_data(sample_email)

        # Should return a dictionary with display-ready data
        assert isinstance(display_data, dict)
        assert "subject" in display_data
        assert "sender_display" in display_data
        assert "time_display" in display_data
        assert "preview_text" in display_data
        assert "importance_badge" in display_data
        assert "action_items_display" in display_data

    def test_get_importance_badge_high(self, sample_email):
        """Test getting importance badge for high importance email."""
        badge = get_importance_badge(sample_email)

        # High importance (>= 0.7) should return a badge
        assert isinstance(badge, str)
        assert len(badge) > 0

    def test_get_importance_badge_low(self):
        """Test getting importance badge for low importance email."""
        email = EmailMessage(
            id="1",
            subject="Newsletter",
            sender="news@example.com",
            received_at=datetime.now(),
            body="Weekly updates",
            importance_score=0.3,
            is_read=True
        )

        badge = get_importance_badge(email)

        # Low importance might return empty string or low priority badge
        assert isinstance(badge, str)

    def test_get_action_items_display_with_items(self, sample_email):
        """Test getting action items display when email has action items."""
        display = get_action_items_display(sample_email)

        # Should return formatted action items
        assert isinstance(display, str)
        assert len(display) > 0
        assert "Review document" in display

    def test_get_action_items_display_no_items(self):
        """Test getting action items display when email has no action items."""
        email = EmailMessage(
            id="1",
            subject="FYI",
            sender="bob@example.com",
            received_at=datetime.now(),
            body="Just keeping you informed",
            has_action_items=False,
            is_read=False
        )

        display = get_action_items_display(email)

        # Should return empty string or "No action items"
        assert isinstance(display, str)


class TestEmailFiltering:
    """Test email filtering logic for display."""

    def test_should_show_email_unread_filter(self):
        """Test email filtering with unread filter."""
        unread_email = EmailMessage(
            id="1",
            subject="Unread",
            sender="alice@example.com",
            received_at=datetime.now(),
            body="Content",
            is_read=False
        )

        read_email = EmailMessage(
            id="2",
            subject="Read",
            sender="bob@example.com",
            received_at=datetime.now(),
            body="Content",
            is_read=True
        )

        # Should show unread email when filtering by unread
        assert should_show_email(unread_email, filter_unread=True) is True
        assert should_show_email(read_email, filter_unread=True) is False

        # Should show both when not filtering
        assert should_show_email(unread_email, filter_unread=False) is True
        assert should_show_email(read_email, filter_unread=False) is True

    def test_should_show_email_importance_filter(self):
        """Test email filtering with importance filter."""
        important_email = EmailMessage(
            id="1",
            subject="Important",
            sender="alice@example.com",
            received_at=datetime.now(),
            body="Content",
            importance_score=0.9,
            is_read=False
        )

        unimportant_email = EmailMessage(
            id="2",
            subject="Not important",
            sender="bob@example.com",
            received_at=datetime.now(),
            body="Content",
            importance_score=0.3,
            is_read=False
        )

        # Should show important email when filtering by importance
        assert should_show_email(important_email, filter_important=True) is True
        assert should_show_email(unimportant_email, filter_important=True) is False

    def test_should_show_email_search_filter(self):
        """Test email filtering with search query."""
        email = EmailMessage(
            id="1",
            subject="Meeting about project alpha",
            sender="alice@example.com",
            received_at=datetime.now(),
            body="Let's discuss the alpha project roadmap",
            is_read=False
        )

        # Should match subject
        assert should_show_email(email, search_query="project") is True
        assert should_show_email(email, search_query="alpha") is True

        # Should match sender
        assert should_show_email(email, search_query="alice") is True

        # Should not match unrelated text
        assert should_show_email(email, search_query="beta") is False

        # Search should be case-insensitive
        assert should_show_email(email, search_query="ALPHA") is True


class TestEmailGrouping:
    """Test email grouping logic for display."""

    @pytest.fixture
    def sample_emails(self) -> List[EmailMessage]:
        """Create sample emails for testing."""
        now = datetime.now()
        today = now.replace(hour=12, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)

        return [
            EmailMessage(
                id="1",
                subject="Today email 1",
                sender="alice@example.com",
                received_at=today,
                body="Content",
                importance_score=0.9,
                is_read=False
            ),
            EmailMessage(
                id="2",
                subject="Today email 2",
                sender="bob@example.com",
                received_at=today + timedelta(hours=1),
                body="Content",
                importance_score=0.7,
                is_read=False
            ),
            EmailMessage(
                id="3",
                subject="Yesterday email",
                sender="carol@example.com",
                received_at=yesterday,
                body="Content",
                importance_score=0.5,
                is_read=True
            ),
        ]

    def test_group_emails_for_display(self, sample_emails):
        """Test grouping emails by date for display."""
        grouped = group_emails_for_display(sample_emails)

        # Should return a dictionary with date groups
        assert isinstance(grouped, dict)
        assert len(grouped) >= 1  # At least one group

        # Each group should have a list of emails
        for date_label, emails in grouped.items():
            assert isinstance(date_label, str)
            assert isinstance(emails, list)
            assert all(isinstance(e, EmailMessage) for e in emails)

    def test_group_emails_empty_list(self):
        """Test grouping empty email list."""
        grouped = group_emails_for_display([])

        # Should return empty dict
        assert isinstance(grouped, dict)
        assert len(grouped) == 0


class TestEmailComponentIntegration:
    """Test integration with formatters and services."""

    @pytest.fixture
    def sample_email(self) -> EmailMessage:
        """Create a sample email for testing."""
        return EmailMessage(
            id="1",
            subject="Test email",
            sender="test@example.com",
            received_at=datetime.now(),
            body="Test body content",
            importance_score=0.8,
            is_read=False
        )

    def test_prepare_email_uses_formatter(self, sample_email):
        """Test that prepare_email_display_data uses EmailFormatter."""
        with patch('src.ui.components.email_components.get_email_formatter') as mock_get_formatter:
            mock_formatter = Mock()
            mock_formatter.format_sender_display.return_value = "Test <test@example.com>"
            mock_formatter.format_relative_time.return_value = "2 hours ago"
            mock_formatter.get_preview_text.return_value = "Test body content..."
            mock_get_formatter.return_value = mock_formatter

            display_data = prepare_email_display_data(sample_email)

            # Should have called formatter methods
            assert mock_formatter.format_sender_display.called
            assert mock_formatter.format_relative_time.called
            assert mock_formatter.get_preview_text.called

            # Should include formatted data
            assert "sender_display" in display_data
            assert "time_display" in display_data
            assert "preview_text" in display_data

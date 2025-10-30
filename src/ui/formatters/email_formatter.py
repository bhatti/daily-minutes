"""EmailFormatter - Business logic for formatting email data for display.

This module contains NO UI rendering code. It only provides business logic
for formatting, grouping, sorting, and filtering email data.

UI components should use this formatter to prepare data for display.
"""

from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict

from src.core.models import EmailMessage
from src.core.logging import get_logger

logger = get_logger(__name__)


class EmailFormatter:
    """Helper class for formatting email data for display.

    Provides business logic for:
    - Grouping emails by various criteria
    - Calculating statistics
    - Formatting timestamps and text
    - Sorting and filtering
    """

    def __init__(self):
        """Initialize EmailFormatter."""
        pass

    # Grouping methods
    def group_by_date(self, emails: List[EmailMessage]) -> Dict[str, List[EmailMessage]]:
        """Group emails by date.

        Args:
            emails: List of email messages

        Returns:
            Dictionary mapping date strings to email lists
        """
        grouped = defaultdict(list)

        for email in emails:
            date_str = email.received_at.strftime("%Y-%m-%d")
            grouped[date_str].append(email)

        return dict(grouped)

    def group_by_sender(self, emails: List[EmailMessage]) -> Dict[str, List[EmailMessage]]:
        """Group emails by sender.

        Args:
            emails: List of email messages

        Returns:
            Dictionary mapping sender addresses to email lists
        """
        grouped = defaultdict(list)

        for email in emails:
            grouped[email.sender].append(email)

        return dict(grouped)

    def group_by_importance(self, emails: List[EmailMessage]) -> Dict[str, List[EmailMessage]]:
        """Group emails by importance level.

        Importance levels:
        - high: score >= 0.7
        - medium: 0.4 <= score < 0.7
        - low: score < 0.4

        Args:
            emails: List of email messages

        Returns:
            Dictionary mapping importance levels to email lists
        """
        grouped = defaultdict(list)

        for email in emails:
            if email.importance_score >= 0.7:
                grouped["high"].append(email)
            elif email.importance_score >= 0.4:
                grouped["medium"].append(email)
            else:
                grouped["low"].append(email)

        return dict(grouped)

    # Statistics methods
    def calculate_statistics(self, emails: List[EmailMessage]) -> Dict[str, int]:
        """Calculate email statistics.

        Args:
            emails: List of email messages

        Returns:
            Dictionary with statistics
        """
        total = len(emails)
        unread = sum(1 for email in emails if not email.is_read)
        read = total - unread
        with_action_items = sum(1 for email in emails if email.has_action_items)
        high_importance = sum(1 for email in emails if email.importance_score >= 0.7)

        return {
            "total": total,
            "unread": unread,
            "read": read,
            "with_action_items": with_action_items,
            "high_importance": high_importance,
        }

    def get_unread_count(self, emails: List[EmailMessage]) -> int:
        """Get count of unread emails.

        Args:
            emails: List of email messages

        Returns:
            Number of unread emails
        """
        return sum(1 for email in emails if not email.is_read)

    def get_action_items_count(self, emails: List[EmailMessage]) -> int:
        """Get count of emails with action items.

        Args:
            emails: List of email messages

        Returns:
            Number of emails with action items
        """
        return sum(1 for email in emails if email.has_action_items)

    # Formatting methods
    def format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp for display.

        Args:
            timestamp: Email timestamp

        Returns:
            Formatted timestamp string
        """
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def format_relative_time(self, timestamp: datetime) -> str:
        """Format timestamp as relative time (e.g., '2 hours ago').

        Args:
            timestamp: Email timestamp (datetime or ISO string)

        Returns:
            Relative time string
        """
        # Handle both datetime and string timestamps
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        now = datetime.now()
        diff = now - timestamp

        if diff < timedelta(minutes=1):
            return "just now"
        elif diff < timedelta(hours=1):
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif diff < timedelta(days=1):
            hours = int(diff.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff < timedelta(days=2):
            return "yesterday"
        elif diff < timedelta(days=7):
            days = int(diff.total_seconds() / 86400)
            return f"{days} days ago"
        else:
            return timestamp.strftime("%Y-%m-%d")

    def format_sender_display(self, sender: str) -> str:
        """Format sender for display.

        Args:
            sender: Sender email address

        Returns:
            Display-friendly sender name
        """
        # For now, just return the email
        # In the future, could extract name from "Name <email@example.com>" format
        return sender

    def get_preview_text(self, email: EmailMessage, max_length: int = 100) -> str:
        """Get preview text from email body.

        Args:
            email: Email message
            max_length: Maximum length of preview

        Returns:
            Preview text (truncated if necessary)
        """
        if not email.body:
            return ""

        text = email.body.strip()

        if len(text) <= max_length:
            return text

        return text[:max_length] + "..."

    def format_action_items_list(self, action_items: List[str]) -> List[str]:
        """Format action items as list.

        Args:
            action_items: List of action items

        Returns:
            Formatted action items
        """
        # For now, just return as-is
        # In the future, could add bullets, icons, etc.
        return action_items

    # Sorting methods
    def sort_by_importance(self, emails: List[EmailMessage]) -> List[EmailMessage]:
        """Sort emails by importance score (highest first).

        Args:
            emails: List of email messages

        Returns:
            Sorted email list
        """
        return sorted(emails, key=lambda e: e.importance_score, reverse=True)

    def sort_by_timestamp(
        self, emails: List[EmailMessage], newest_first: bool = True
    ) -> List[EmailMessage]:
        """Sort emails by timestamp.

        Args:
            emails: List of email messages
            newest_first: If True, newest emails first. If False, oldest first.

        Returns:
            Sorted email list
        """
        return sorted(emails, key=lambda e: e.received_at, reverse=newest_first)

    # Filtering methods
    def filter_unread(self, emails: List[EmailMessage]) -> List[EmailMessage]:
        """Filter unread emails.

        Args:
            emails: List of email messages

        Returns:
            Unread emails only
        """
        return [email for email in emails if not email.is_read]

    def filter_by_sender(self, emails: List[EmailMessage], sender: str) -> List[EmailMessage]:
        """Filter emails by sender.

        Args:
            emails: List of email messages
            sender: Sender email address

        Returns:
            Emails from specified sender
        """
        return [email for email in emails if email.sender == sender]

    def filter_by_subject(self, emails: List[EmailMessage], keyword: str) -> List[EmailMessage]:
        """Filter emails by subject keyword (case-insensitive).

        Args:
            emails: List of email messages
            keyword: Subject keyword to search for

        Returns:
            Emails matching keyword
        """
        keyword_lower = keyword.lower()
        return [email for email in emails if keyword_lower in email.subject.lower()]

    def filter_high_importance(
        self, emails: List[EmailMessage], threshold: float = 0.7
    ) -> List[EmailMessage]:
        """Filter high importance emails.

        Args:
            emails: List of email messages
            threshold: Minimum importance score (default: 0.7)

        Returns:
            High importance emails
        """
        return [email for email in emails if email.importance_score >= threshold]


# Singleton instance
_email_formatter: EmailFormatter = None


def get_email_formatter() -> EmailFormatter:
    """Get singleton EmailFormatter instance.

    Returns:
        EmailFormatter singleton
    """
    global _email_formatter

    if _email_formatter is None:
        _email_formatter = EmailFormatter()
        logger.debug("email_formatter_initialized")

    return _email_formatter

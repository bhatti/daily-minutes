"""EmailService - Centralized email fetching and management logic.

Following the NewsService pattern, this service provides a clean interface
for fetching emails from multiple providers.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Dict

from src.core.models import EmailMessage
from src.agents.email_agent import get_email_agent
from src.connectors.email.gmail_connector import get_gmail_connector
from src.connectors.email.outlook_connector import get_outlook_connector
from src.connectors.email.imap_connector import get_imap_connector
from src.core.logging import get_logger

logger = get_logger(__name__)


class EmailService:
    """
    Service for fetching and managing emails.

    Handles:
    - Fetching from multiple email providers (Gmail, Outlook, IMAP)
    - Using EmailAgent for orchestration
    - Progress reporting
    - Error handling
    """

    def __init__(self):
        """Initialize EmailService."""
        self._email_agent = None
        self._connectors_initialized = False

    def _initialize_connectors(self) -> Dict[str, any]:
        """Initialize email connectors based on configuration.

        Returns:
            Dictionary of initialized connectors
        """
        import os

        if self._connectors_initialized:
            return self._email_agent.connectors if self._email_agent else {}

        connectors = {}

        # Check if using mock email data
        use_mock = os.getenv('USE_MOCK_EMAIL', 'false').lower() == 'true'
        if use_mock:
            logger.info("using_mock_email_data")
            # Don't initialize any connectors - we'll use mock data directly
            self._connectors_initialized = True
            return connectors

        # Try to initialize Gmail connector
        try:
            gmail_connector = get_gmail_connector()
            if gmail_connector:
                connectors["gmail"] = gmail_connector
                logger.info("gmail_connector_initialized")
        except Exception as e:
            logger.warning("gmail_connector_init_failed", error=str(e))

        # Try to initialize Outlook connector
        try:
            outlook_connector = get_outlook_connector()
            if outlook_connector:
                connectors["outlook"] = outlook_connector
                logger.info("outlook_connector_initialized")
        except Exception as e:
            logger.warning("outlook_connector_init_failed", error=str(e))

        # Try to initialize IMAP connector
        try:
            imap_connector = get_imap_connector()
            if imap_connector:
                connectors["imap"] = imap_connector
                logger.info("imap_connector_initialized")
        except Exception as e:
            logger.warning("imap_connector_init_failed", error=str(e))

        self._connectors_initialized = True
        return connectors

    async def fetch_emails(
        self,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: Optional[int] = 50,
        filter_unread: bool = False,
        filter_important: bool = False,
        sort_by_importance: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[EmailMessage]:
        """
        Fetch emails from all configured providers.

        Args:
            time_min: Start of time range (default: 7 days ago)
            time_max: End of time range (default: now)
            max_results: Maximum number of emails to return
            filter_unread: Only return unread emails
            filter_important: Only return high importance emails
            sort_by_importance: Sort by importance score
            progress_callback: Optional callback for progress updates

        Returns:
            List of email messages
        """
        # Set default time range
        if time_max is None:
            time_max = datetime.now()
        if time_min is None:
            time_min = time_max - timedelta(days=7)

        logger.info("fetch_emails_start",
                   time_min=time_min.isoformat(),
                   time_max=time_max.isoformat(),
                   max_results=max_results)

        # Update progress
        if progress_callback:
            await progress_callback(0.1, "Initializing email connectors...")

        # Initialize connectors
        connectors = self._initialize_connectors()

        # Check if using mock email data
        import os
        use_mock = os.getenv('USE_MOCK_EMAIL', 'false').lower() == 'true'

        if use_mock:
            # Use mock email data generator
            logger.info("fetching_mock_email_data")
            if progress_callback:
                await progress_callback(0.3, "Generating mock email data...")

            try:
                from tests.load_mock_data import load_mock_emails
                emails = load_mock_emails(count=max_results)

                # Filter by time range if specified
                if time_min or time_max:
                    filtered = []
                    for email in emails:
                        if time_min and email.received_at < time_min:
                            continue
                        if time_max and email.received_at > time_max:
                            continue
                        filtered.append(email)
                    emails = filtered
            except Exception as e:
                logger.error("mock_email_generation_failed", error=str(e))
                emails = []
        elif not connectors:
            logger.warning("no_email_connectors_available")
            return []
        else:
            # Update progress
            if progress_callback:
                await progress_callback(0.2, f"Fetching emails from {len(connectors)} provider(s)...")

            # Get or create email agent
            if self._email_agent is None:
                self._email_agent = get_email_agent()
                # Add connectors to agent
                for name, connector in connectors.items():
                    self._email_agent.add_connector(name, connector)

            # Fetch emails using agent
            # Note: time_min/time_max filtering happens at connector level, not agent level
            try:
                emails = await self._email_agent.fetch_emails(
                    max_results=max_results,
                    sort_by_importance=sort_by_importance,
                    use_cache=True
                )
            except Exception as e:
                logger.error("fetch_emails_failed", error=str(e), exc_info=True)
                if progress_callback:
                    await progress_callback(1.0, f"Error: {str(e)}")
                return []

        # Update progress
        if progress_callback:
            await progress_callback(0.6, f"Retrieved {len(emails)} emails")

        # Apply additional filters
        if filter_unread:
            emails = [e for e in emails if not e.is_read]
            if progress_callback:
                await progress_callback(0.7, f"Filtered to {len(emails)} unread emails")

        if filter_important:
            emails = [e for e in emails if e.importance_score >= 0.7]
            if progress_callback:
                await progress_callback(0.8, f"Filtered to {len(emails)} important emails")

        # Update progress
        if progress_callback:
            await progress_callback(1.0, f"Complete: {len(emails)} emails retrieved")

        provider_label = "mock" if use_mock else str(list(connectors.keys()))
        logger.info("fetch_emails_complete",
                   total_emails=len(emails),
                   providers=provider_label)

        return emails

    async def get_unread_count(self) -> int:
        """Get count of unread emails.

        Returns:
            Number of unread emails
        """
        try:
            emails = await self.fetch_emails(
                max_results=100,
                filter_unread=True,
                sort_by_importance=False
            )
            return len(emails)
        except Exception as e:
            logger.error("get_unread_count_failed", error=str(e))
            return 0

    async def get_important_emails(
        self,
        max_results: int = 10
    ) -> List[EmailMessage]:
        """Get most important emails.

        Args:
            max_results: Maximum number of emails to return

        Returns:
            List of important emails sorted by importance
        """
        return await self.fetch_emails(
            max_results=max_results,
            filter_important=True,
            sort_by_importance=True
        )

    async def get_emails_with_action_items(
        self,
        max_results: int = 20
    ) -> List[EmailMessage]:
        """Get emails containing action items.

        Args:
            max_results: Maximum number of emails to return

        Returns:
            List of emails with action items
        """
        try:
            emails = await self.fetch_emails(
                max_results=max_results * 2,  # Fetch more to filter
                sort_by_importance=True
            )

            # Filter for action items
            emails_with_actions = [e for e in emails if e.has_action_items]

            return emails_with_actions[:max_results]

        except Exception as e:
            logger.error("get_emails_with_action_items_failed", error=str(e))
            return []


# Singleton instance
_email_service: EmailService = None


def get_email_service() -> EmailService:
    """Get singleton EmailService instance.

    Returns:
        EmailService singleton
    """
    global _email_service

    if _email_service is None:
        _email_service = EmailService()
        logger.debug("email_service_initialized")

    return _email_service

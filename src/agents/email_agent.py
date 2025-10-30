"""EmailAgent for orchestrating multiple email connectors.

The EmailAgent provides a unified interface for fetching, filtering, and prioritizing
emails from multiple email providers (Gmail, Outlook, IMAP).

Features:
- Multi-connector support (Gmail, Outlook, IMAP)
- Email fetching with parallel connector execution
- Importance-based sorting and prioritization
- Advanced filtering (sender, subject, importance)
- Caching with configurable TTL
- Metrics emission and activity logging
- Email statistics and analytics

Example:
    ```python
    from src.agents.email_agent import EmailAgent
    from src.connectors.email.imap_connector import get_imap_connector

    # Create agent with IMAP connector
    connector = get_imap_connector(
        imap_server="localhost",
        username="test",
        password="test",
        port=1143
    )

    agent = EmailAgent(connectors={"imap": connector})

    # Fetch and prioritize emails
    emails = await agent.fetch_emails(
        max_results=50,
        sort_by_importance=True,
        min_importance=0.5
    )
    ```
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import Counter

from src.core.models import EmailMessage
from src.core.logging import get_logger
from src.core.metrics import get_metrics_manager
from src.database.sqlite_manager import get_db_manager

logger = get_logger(__name__)


class EmailAgent:
    """Agent for orchestrating multiple email connectors.

    Provides unified email fetching, filtering, prioritization, and caching.

    Attributes:
        connectors: Dictionary of email connectors (key=name, value=connector)
        cache_ttl_seconds: Cache TTL in seconds
        last_fetch_time: Timestamp of last email fetch
        cached_emails: Cached email list
    """

    def __init__(
        self,
        connectors: Optional[Dict[str, Any]] = None,
        cache_ttl_seconds: int = 300  # 5 minutes default
    ):
        """Initialize EmailAgent.

        Args:
            connectors: Dictionary of email connectors {name: connector_instance}
            cache_ttl_seconds: Cache time-to-live in seconds (default: 300)
        """
        self.connectors = connectors or {}
        self.cache_ttl_seconds = cache_ttl_seconds

        # Cache
        self.last_fetch_time: Optional[datetime] = None
        self.cached_emails: List[EmailMessage] = []

        # Observability
        self.metrics_manager = get_metrics_manager()
        self.db_manager = get_db_manager()

        logger.info("email_agent_initialized",
                   connector_count=len(self.connectors),
                   cache_ttl=cache_ttl_seconds)

    def add_connector(self, name: str, connector: Any) -> None:
        """Add an email connector.

        Args:
            name: Connector name (e.g., "gmail", "outlook", "imap")
            connector: Connector instance
        """
        self.connectors[name] = connector
        logger.info("connector_added", name=name, total_connectors=len(self.connectors))

    async def fetch_emails(
        self,
        max_results: int = 50,
        sort_by_importance: bool = True,
        use_cache: bool = True,
        filter_sender: Optional[str] = None,
        filter_subject: Optional[str] = None,
        min_importance: Optional[float] = None
    ) -> List[EmailMessage]:
        """Fetch emails from all configured connectors.

        Args:
            max_results: Maximum number of emails to return
            sort_by_importance: Sort emails by importance score (descending)
            use_cache: Use cached results if available
            filter_sender: Filter by sender email address
            filter_subject: Filter by subject keyword (case-insensitive)
            min_importance: Minimum importance score filter

        Returns:
            List of EmailMessage objects
        """
        start_time = datetime.now()

        try:
            # Check cache
            if use_cache and self._is_cache_valid():
                logger.info("email_fetch_using_cache", cached_count=len(self.cached_emails))
                emails = self.cached_emails
            else:
                # Fetch from all connectors
                emails = await self._fetch_from_connectors(max_results)

                # Update cache
                self.cached_emails = emails
                self.last_fetch_time = datetime.now()

            # Apply filters
            emails = self._apply_filters(
                emails,
                filter_sender=filter_sender,
                filter_subject=filter_subject,
                min_importance=min_importance
            )

            # Sort by importance if requested
            if sort_by_importance:
                emails = sorted(emails, key=lambda e: e.importance_score, reverse=True)

            # Limit results
            emails = emails[:max_results]

            duration = (datetime.now() - start_time).total_seconds()
            logger.info("email_fetch_success",
                       email_count=len(emails),
                       duration_seconds=duration,
                       used_cache=use_cache and self._is_cache_valid())

            # Emit metrics
            self.metrics_manager.emit({
                "metric": "email_agent_fetch",
                "value": len(emails),
                "timestamp": datetime.now().isoformat(),
                "labels": {
                    "connectors": len(self.connectors),
                    "cached": str(use_cache and self._is_cache_valid())
                }
            })

            # Log activity
            await self._log_activity(
                activity_type="email_fetch",
                status="success",
                details={
                    "email_count": len(emails),
                    "duration_seconds": duration,
                    "connectors": list(self.connectors.keys())
                }
            )

            return emails

        except Exception as e:
            logger.error("email_fetch_failed", error=str(e))
            await self._log_activity(
                activity_type="email_fetch",
                status="failed",
                error=str(e)
            )
            raise

    async def _fetch_from_connectors(self, max_results: int) -> List[EmailMessage]:
        """Fetch emails from all connectors in parallel.

        Args:
            max_results: Maximum results per connector

        Returns:
            Combined list of emails from all connectors
        """
        if not self.connectors:
            logger.warning("no_connectors_configured")
            return []

        # Fetch from each connector in parallel
        tasks = []
        for name, connector in self.connectors.items():
            tasks.append(self._fetch_from_single_connector(name, connector, max_results))

        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine emails (filter out exceptions)
        all_emails = []
        for name, result in zip(self.connectors.keys(), results):
            if isinstance(result, Exception):
                logger.error("connector_fetch_failed", connector=name, error=str(result))
            else:
                all_emails.extend(result)

        logger.info("multi_connector_fetch_complete",
                   total_emails=len(all_emails),
                   connector_count=len(self.connectors))

        return all_emails

    async def _fetch_from_single_connector(
        self,
        name: str,
        connector: Any,
        max_results: int
    ) -> List[EmailMessage]:
        """Fetch emails from a single connector.

        Args:
            name: Connector name
            connector: Connector instance
            max_results: Maximum results to fetch

        Returns:
            List of emails from this connector
        """
        try:
            logger.debug("fetching_from_connector", connector=name)

            # Check if connector is authenticated
            if not connector.is_authenticated:
                logger.warning("connector_not_authenticated", connector=name)
                return []

            # Fetch emails
            emails = await connector.fetch_unread_emails(max_results=max_results)

            logger.info("connector_fetch_success",
                       connector=name,
                       email_count=len(emails))

            return emails

        except Exception as e:
            logger.error("connector_fetch_error",
                        connector=name,
                        error=str(e))
            # Return empty list instead of raising (allow other connectors to succeed)
            return []

    def _apply_filters(
        self,
        emails: List[EmailMessage],
        filter_sender: Optional[str] = None,
        filter_subject: Optional[str] = None,
        min_importance: Optional[float] = None
    ) -> List[EmailMessage]:
        """Apply filters to email list.

        Args:
            emails: List of emails to filter
            filter_sender: Filter by sender
            filter_subject: Filter by subject keyword
            min_importance: Minimum importance score

        Returns:
            Filtered email list
        """
        filtered = emails

        # Filter by sender
        if filter_sender:
            filtered = [e for e in filtered if e.sender == filter_sender]
            logger.debug("filter_by_sender", sender=filter_sender, count=len(filtered))

        # Filter by subject
        if filter_subject:
            filtered = [
                e for e in filtered
                if filter_subject.lower() in e.subject.lower()
            ]
            logger.debug("filter_by_subject", subject=filter_subject, count=len(filtered))

        # Filter by importance
        if min_importance is not None:
            filtered = [e for e in filtered if e.importance_score >= min_importance]
            logger.debug("filter_by_importance",
                        min_importance=min_importance,
                        count=len(filtered))

        return filtered

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid.

        Returns:
            True if cache is valid, False otherwise
        """
        if self.last_fetch_time is None:
            return False

        age = (datetime.now() - self.last_fetch_time).total_seconds()
        is_valid = age < self.cache_ttl_seconds

        logger.debug("cache_check",
                    age_seconds=age,
                    ttl_seconds=self.cache_ttl_seconds,
                    is_valid=is_valid)

        return is_valid

    def get_sender_statistics(self) -> Dict[str, int]:
        """Get email count grouped by sender.

        Returns:
            Dictionary mapping sender to email count
        """
        if not self.cached_emails:
            return {}

        senders = [email.sender for email in self.cached_emails]
        return dict(Counter(senders))

    def get_average_importance(self) -> float:
        """Calculate average importance score of cached emails.

        Returns:
            Average importance score (0.0 if no emails)
        """
        if not self.cached_emails:
            return 0.0

        total = sum(email.importance_score for email in self.cached_emails)
        return total / len(self.cached_emails)

    async def _log_activity(
        self,
        activity_type: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Log activity to database.

        Args:
            activity_type: Type of activity
            status: Status (success/failed)
            details: Optional activity details
            error: Optional error message
        """
        # Activity logging is not yet implemented in SQLiteManager
        # TODO: Implement activity_log table and logging
        logger.debug("activity_log",
                    activity=activity_type,
                    status=status,
                    details=details,
                    error=error)


# Singleton instance
_email_agent: EmailAgent = None


def get_email_agent() -> EmailAgent:
    """Get singleton EmailAgent instance.

    Returns:
        EmailAgent singleton
    """
    global _email_agent

    if _email_agent is None:
        _email_agent = EmailAgent()
        logger.debug("email_agent_singleton_initialized")

    return _email_agent

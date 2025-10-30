"""Gmail connector for fetching and managing emails.

Provides OAuth-authenticated access to Gmail API with:
- Email fetching and parsing
- Batch operations (mark as read, labels)
- Error handling and retries
- Observability (metrics + activity logging)

Dependencies:
    pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
"""

import asyncio
import base64
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

# Google API imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    raise ImportError(
        "Gmail connector requires Google API libraries. "
        "Install with: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
    )

from src.core.models import EmailMessage
from src.core.logging import get_logger
from src.core.metrics import get_metrics_manager
from src.database.sqlite_manager import get_db_manager

logger = get_logger(__name__)

# Gmail API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify'
]


class GmailConnector:
    """Gmail API connector with OAuth authentication and observability.

    Features:
    - OAuth 2.0 authentication with token refresh
    - Fetch unread emails with filtering
    - Parse emails into EmailMessage models
    - Batch operations (mark as read, labels)
    - Rate limit handling with exponential backoff
    - Comprehensive error handling
    - Metrics emission and activity logging

    Example:
        ```python
        connector = GmailConnector()
        connector.authenticate(
            credentials_path="credentials/token.json",
            client_secrets_path="credentials/client_secret.json"
        )

        emails = await connector.fetch_unread_emails(max_results=50)
        await connector.mark_as_read_batch([email.id for email in emails[:10]])
        ```
    """

    def __init__(self, user_id: str = "me"):
        """Initialize Gmail connector.

        Args:
            user_id: Gmail user ID (default: "me" for authenticated user)
        """
        self.user_id = user_id
        self.credentials: Optional[Credentials] = None
        self.service = None
        self.metrics_manager = get_metrics_manager()
        self.db_manager = get_db_manager()

        logger.info("gmail_connector_initialized", user_id=user_id)

    @property
    def is_authenticated(self) -> bool:
        """Check if connector is authenticated."""
        return self.credentials is not None and self.service is not None

    def _load_credentials_from_file(self, credentials_path: str) -> Credentials:
        """Load credentials from file.

        Args:
            credentials_path: Path to credentials JSON file

        Returns:
            Loaded credentials

        Raises:
            FileNotFoundError: If credentials file doesn't exist
        """
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found: {credentials_path}")

        return Credentials.from_authorized_user_file(credentials_path, SCOPES)

    def authenticate(
        self,
        credentials_path: str = "credentials/gmail_token.json",
        client_secrets_path: str = "credentials/client_secret.json"
    ) -> None:
        """Authenticate with Gmail API using OAuth 2.0.

        Authentication flow:
        1. Try to load existing credentials from file
        2. If credentials exist but expired, refresh them
        3. If no credentials or refresh fails, run OAuth flow
        4. Save credentials for future use

        Args:
            credentials_path: Path to store/load OAuth token
            client_secrets_path: Path to OAuth client secrets file

        Raises:
            FileNotFoundError: If client_secrets_path doesn't exist (for new users)
        """
        start_time = datetime.now()
        creds = None

        try:
            # Try to load existing credentials
            if os.path.exists(credentials_path):
                logger.debug("gmail_auth_loading_existing_credentials", path=credentials_path)
                creds = self._load_credentials_from_file(credentials_path)

            # Refresh or re-authenticate if needed
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    logger.info("gmail_auth_refreshing_token")
                    creds.refresh(Request())
                else:
                    # Run OAuth flow for new authentication
                    logger.info("gmail_auth_running_oauth_flow")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        client_secrets_path,
                        SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save credentials for future use
                os.makedirs(os.path.dirname(credentials_path), exist_ok=True)
                with open(credentials_path, 'w') as token:
                    token.write(creds.to_json())
                logger.info("gmail_auth_credentials_saved", path=credentials_path)

            # Build Gmail API service
            self.credentials = creds
            self.service = build('gmail', 'v1', credentials=creds)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info("gmail_authentication_success", duration_seconds=duration)

            # Log activity (skip async logging in sync method)
            # TODO: Add sync logging method or make authenticate() async
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._log_activity(
                        activity_type="gmail_authentication",
                        status="success",
                        details={"duration_seconds": duration}
                    ))
            except RuntimeError:
                pass  # No event loop, skip async logging

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error("gmail_authentication_failed", error=str(e), duration_seconds=duration)

            # Log failed activity (skip async logging in sync method)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._log_activity(
                        activity_type="gmail_authentication",
                        status="failed",
                        error=str(e)
                    ))
            except RuntimeError:
                pass  # No event loop, skip async logging

            raise

    async def fetch_unread_emails(
        self,
        max_results: int = 50,
        query: str = "is:unread",
        max_retries: int = 3
    ) -> List[EmailMessage]:
        """Fetch unread emails from Gmail.

        Args:
            max_results: Maximum number of emails to fetch
            query: Gmail search query (default: "is:unread")
            max_retries: Maximum retry attempts on rate limit errors

        Returns:
            List of EmailMessage objects

        Raises:
            RuntimeError: If not authenticated
            HttpError: On API errors after retries exhausted
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        start_time = datetime.now()
        all_emails = []

        try:
            logger.info("gmail_fetching_emails", max_results=max_results, query=query)

            # Fetch message IDs with pagination
            message_ids = await self._fetch_message_ids(
                query=query,
                max_results=max_results,
                max_retries=max_retries
            )

            logger.debug("gmail_message_ids_fetched", count=len(message_ids))

            # Fetch full message details for each ID
            for msg_id in message_ids[:max_results]:
                try:
                    msg_data = await self._fetch_message_details(msg_id, max_retries)
                    email_msg = await self._parse_email_message(msg_data)
                    all_emails.append(email_msg)
                except Exception as e:
                    logger.warning("gmail_email_parse_failed", msg_id=msg_id, error=str(e))
                    continue

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                "gmail_fetch_success",
                emails_fetched=len(all_emails),
                duration_seconds=duration
            )

            # Emit metrics
            self.metrics_manager.emit({
                "metric": "gmail_emails_fetched",
                "value": len(all_emails),
                "timestamp": datetime.now().isoformat(),
                "labels": {"query": query}
            })

            # Log activity
            await self._log_activity(
                activity_type="gmail_fetch_emails",
                status="success",
                details={
                    "emails_count": len(all_emails),
                    "query": query,
                    "duration_seconds": duration
                }
            )

            return all_emails

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error("gmail_fetch_failed", error=str(e), duration_seconds=duration)

            # Log failed activity
            await self._log_activity(
                activity_type="gmail_fetch_emails",
                status="failed",
                error=str(e)
            )

            raise

    async def _fetch_message_ids(
        self,
        query: str,
        max_results: int,
        max_retries: int
    ) -> List[str]:
        """Fetch message IDs matching query with pagination.

        Args:
            query: Gmail search query
            max_results: Maximum results to fetch
            max_retries: Maximum retry attempts

        Returns:
            List of message IDs
        """
        message_ids = []
        next_page_token = None

        while len(message_ids) < max_results:
            retry_count = 0

            while retry_count <= max_retries:
                try:
                    # Fetch page of results
                    results = self.service.users().messages().list(
                        userId=self.user_id,
                        q=query,
                        maxResults=min(max_results - len(message_ids), 100),
                        pageToken=next_page_token
                    ).execute()

                    messages = results.get('messages', [])
                    message_ids.extend([msg['id'] for msg in messages])

                    next_page_token = results.get('nextPageToken')

                    # Break retry loop on success
                    break

                except HttpError as e:
                    if e.resp.status == 429:  # Rate limit
                        retry_count += 1
                        if retry_count > max_retries:
                            raise

                        wait_time = 2 ** retry_count  # Exponential backoff
                        logger.warning(
                            "gmail_rate_limit_retry",
                            retry=retry_count,
                            wait_seconds=wait_time
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise

            # Stop if no more pages
            if not next_page_token:
                break

        return message_ids

    async def _fetch_message_details(self, msg_id: str, max_retries: int) -> Dict[str, Any]:
        """Fetch full message details with retry logic.

        Args:
            msg_id: Message ID to fetch
            max_retries: Maximum retry attempts

        Returns:
            Message data dictionary
        """
        retry_count = 0

        while retry_count <= max_retries:
            try:
                return self.service.users().messages().get(
                    userId=self.user_id,
                    id=msg_id,
                    format='full'
                ).execute()

            except HttpError as e:
                if e.resp.status == 429:  # Rate limit
                    retry_count += 1
                    if retry_count > max_retries:
                        raise

                    wait_time = 2 ** retry_count
                    await asyncio.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed to fetch message {msg_id} after {max_retries} retries")

    async def _parse_email_message(self, msg_data: Dict[str, Any]) -> EmailMessage:
        """Parse Gmail API message into EmailMessage model.

        Args:
            msg_data: Raw message data from Gmail API

        Returns:
            Parsed EmailMessage object
        """
        # Extract headers
        headers = {
            h['name'].lower(): h['value']
            for h in msg_data['payload'].get('headers', [])
        }

        # Parse basic fields
        email_id = msg_data['id']
        subject = headers.get('subject', '(No Subject)')
        sender = headers.get('from', 'unknown@example.com')

        # Extract sender email if in "Name <email>" format
        if '<' in sender and '>' in sender:
            sender = sender[sender.index('<')+1:sender.index('>')]

        # Parse received date
        internal_date = int(msg_data.get('internalDate', 0)) / 1000
        received_at = datetime.fromtimestamp(internal_date) if internal_date > 0 else datetime.now()

        # Extract body
        body = self._extract_body(msg_data['payload'])

        # Extract snippet
        snippet = msg_data.get('snippet')

        # Extract labels
        labels = msg_data.get('labelIds', [])

        # Calculate importance score based on labels and headers
        importance_score = self._calculate_importance(headers, labels)

        return EmailMessage(
            id=email_id,
            subject=subject,
            sender=sender,
            received_at=received_at,
            body=body,
            snippet=snippet,
            labels=labels,
            importance_score=importance_score
        )

    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """Extract email body from payload (handles text and multipart).

        Args:
            payload: Email payload from Gmail API

        Returns:
            Email body text
        """
        # Check if body is directly in payload
        if 'body' in payload and 'data' in payload['body']:
            return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')

        # Handle multipart messages
        if 'parts' in payload:
            for part in payload['parts']:
                # Prefer text/plain over text/html
                if part.get('mimeType') == 'text/plain':
                    if 'data' in part.get('body', {}):
                        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')

            # Fallback to HTML if no plain text
            for part in payload['parts']:
                if part.get('mimeType') == 'text/html':
                    if 'data' in part.get('body', {}):
                        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')

            # Recursively search nested parts
            for part in payload['parts']:
                if 'parts' in part:
                    body = self._extract_body(part)
                    if body:
                        return body

        return "(No body content)"

    def _calculate_importance(self, headers: Dict[str, str], labels: List[str]) -> float:
        """Calculate importance score based on headers and labels.

        Args:
            headers: Email headers dictionary
            labels: Gmail label IDs

        Returns:
            Importance score (0.0-1.0)
        """
        score = 0.5  # Base score

        # Boost for IMPORTANT label
        if 'IMPORTANT' in labels:
            score += 0.2

        # Boost for STARRED
        if 'STARRED' in labels:
            score += 0.1

        # Boost for high importance header
        importance_header = headers.get('importance', '').lower()
        if importance_header == 'high':
            score += 0.15

        # Boost for urgent keywords in subject
        subject = headers.get('subject', '').lower()
        urgent_keywords = ['urgent', 'important', 'asap', 'critical']
        if any(keyword in subject for keyword in urgent_keywords):
            score += 0.1

        # Clamp to valid range
        return max(0.0, min(1.0, score))

    async def mark_as_read(self, message_id: str) -> None:
        """Mark a single email as read.

        Args:
            message_id: Email message ID to mark as read
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated")

        try:
            self.service.users().messages().modify(
                userId=self.user_id,
                id=message_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()

            logger.debug("gmail_message_marked_read", message_id=message_id)

        except Exception as e:
            logger.error("gmail_mark_read_failed", message_id=message_id, error=str(e))
            raise

    async def mark_as_read_batch(self, message_ids: List[str]) -> None:
        """Mark multiple emails as read in a single batch operation.

        Args:
            message_ids: List of email message IDs to mark as read
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated")

        if not message_ids:
            return

        try:
            self.service.users().messages().batchModify(
                userId=self.user_id,
                body={
                    'ids': message_ids,
                    'removeLabelIds': ['UNREAD']
                }
            ).execute()

            logger.info("gmail_batch_marked_read", count=len(message_ids))

        except Exception as e:
            logger.error("gmail_batch_mark_read_failed", count=len(message_ids), error=str(e))
            raise

    async def add_labels(self, message_id: str, labels: List[str]) -> None:
        """Add labels to an email.

        Args:
            message_id: Email message ID
            labels: List of label IDs to add
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated")

        try:
            self.service.users().messages().modify(
                userId=self.user_id,
                id=message_id,
                body={'addLabelIds': labels}
            ).execute()

            logger.debug("gmail_labels_added", message_id=message_id, labels=labels)

        except Exception as e:
            logger.error("gmail_add_labels_failed", message_id=message_id, error=str(e))
            raise

    async def _log_activity(
        self,
        activity_type: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Log activity to database.

        Args:
            activity_type: Type of activity (e.g., "gmail_fetch_emails")
            status: Activity status ("success", "failed", "partial")
            details: Additional details dictionary
            error: Error message if status is "failed"
        """
        try:
            await self.db_manager.initialize()

            async with self.db_manager._get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO activity_log (activity_type, status, details, error_message)
                    VALUES (?, ?, ?, ?)
                    """,
                    (activity_type, status, str(details) if details else None, error)
                )
                await db.commit()

        except Exception as e:
            logger.warning("gmail_activity_log_failed", error=str(e))


# Singleton instance
_gmail_connector_instance: Optional[GmailConnector] = None


def get_gmail_connector() -> GmailConnector:
    """Get singleton GmailConnector instance.

    Returns:
        Global GmailConnector instance
    """
    global _gmail_connector_instance

    if _gmail_connector_instance is None:
        _gmail_connector_instance = GmailConnector()

    return _gmail_connector_instance

"""IMAP Email Connector.

Provides generic email integration with any IMAP server using standard IMAP protocol.

Features:
- Standard IMAP authentication (username/password - no OAuth)
- Works with any email provider (Gmail, Outlook, Yahoo, custom servers)
- Fetch unread emails with IMAP search
- Parse emails from RFC822 format into EmailMessage models
- Batch operations (mark as read)
- Metrics emission for observability
- Activity logging to database

IMAP Setup:
1. Enable IMAP access in your email provider settings
2. For Gmail: Enable "Allow less secure apps" or use App Passwords
3. For Outlook: Enable IMAP in settings
4. Get IMAP server address and port (usually imap.gmail.com:993, outlook.office365.com:993)

Example:
    ```python
    from src.connectors.email.imap_connector import get_imap_connector

    connector = get_imap_connector(
        imap_server="imap.gmail.com",
        username="user@gmail.com",
        password="app_password"
    )
    await connector.authenticate()
    emails = await connector.fetch_unread_emails(max_results=50)
    ```
"""

import asyncio
import email
import imaplib
import json
import socket
from datetime import datetime
from email.header import decode_header
from typing import List, Optional, Any, Dict

from src.core.models import EmailMessage
from src.core.logging import get_logger
from src.core.metrics import get_metrics_manager
from src.database.sqlite_manager import get_db_manager

logger = get_logger(__name__)


class IMAPConnector:
    """IMAP email connector using standard IMAP protocol.

    Supports:
    - Any IMAP server (Gmail, Outlook, Yahoo, custom)
    - Username/password authentication (no OAuth)
    - Fetching unread emails with IMAP search
    - Parsing RFC822 emails with multipart support
    - Batch operations
    - Metrics and activity logging

    Attributes:
        imap_server: IMAP server hostname
        username: Email username
        password: Email password
        use_ssl: Use SSL/TLS (default True)
        port: IMAP port (993 for SSL, 143 for non-SSL)
        timeout: Connection timeout in seconds
    """

    def __init__(
        self,
        imap_server: str,
        username: str,
        password: str,
        use_ssl: bool = True,
        port: Optional[int] = None,
        timeout: int = 30
    ):
        """Initialize IMAP connector.

        Args:
            imap_server: IMAP server hostname (e.g., "imap.gmail.com")
            username: Email username
            password: Email password or app password
            use_ssl: Use SSL/TLS connection (default True)
            port: IMAP port (defaults to 993 for SSL, 143 for non-SSL)
            timeout: Connection timeout in seconds

        Raises:
            ValueError: If credentials are missing
        """
        if not imap_server or not username or not password:
            raise ValueError("imap_server, username, and password are required")

        self.imap_server = imap_server
        self.username = username
        self.password = password
        self.use_ssl = use_ssl
        self.port = port or (993 if use_ssl else 143)
        self.timeout = timeout

        self.connection: Optional[Any] = None

        # Observability
        self.metrics_manager = get_metrics_manager()
        self.db_manager = get_db_manager()

        logger.info("imap_connector_initialized", server=imap_server, username=username)

    @property
    def is_authenticated(self) -> bool:
        """Check if connector is authenticated."""
        return self.connection is not None

    async def authenticate(self) -> None:
        """Authenticate with IMAP server.

        Raises:
            RuntimeError: If authentication fails
        """
        start_time = datetime.now()

        try:
            logger.info("imap_authentication_starting", server=self.imap_server, username=self.username)

            # Create IMAP connection
            if self.use_ssl:
                self.connection = imaplib.IMAP4_SSL(self.imap_server, self.port)
            else:
                self.connection = imaplib.IMAP4(self.imap_server, self.port)

            # Set timeout
            if self.connection.sock:
                self.connection.sock.settimeout(self.timeout)

            # Login
            status, response = self.connection.login(self.username, self.password)

            if status != 'OK':
                raise RuntimeError(f"Authentication failed: {response}")

            duration = (datetime.now() - start_time).total_seconds()
            logger.info("imap_authentication_success", duration_seconds=duration)

        except (imaplib.IMAP4.error, socket.error, socket.timeout, ConnectionRefusedError) as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error("imap_authentication_failed", error=str(e), duration_seconds=duration)
            raise RuntimeError(f"Connection or authentication failed: {e}")
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error("imap_authentication_failed", error=str(e), duration_seconds=duration)
            raise

    async def fetch_unread_emails(
        self,
        max_results: int = 50,
        folder: str = "INBOX",
        search_criteria: str = "UNSEEN"
    ) -> List[EmailMessage]:
        """Fetch unread emails from IMAP server.

        Args:
            max_results: Maximum number of emails to fetch
            folder: IMAP folder to search (default "INBOX")
            search_criteria: IMAP search criteria (default "UNSEEN" for unread)

        Returns:
            List of EmailMessage objects

        Raises:
            RuntimeError: If not authenticated or IMAP operation fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        start_time = datetime.now()
        all_emails: List[EmailMessage] = []

        try:
            logger.info("imap_fetching_emails", folder=folder, max_results=max_results)

            # Select folder
            status, response = self.connection.select(folder)
            if status != 'OK':
                raise RuntimeError(f"Failed to select folder {folder}: {response}")

            # Search for messages
            status, message_ids_data = self.connection.search(None, search_criteria)
            if status != 'OK':
                raise RuntimeError(f"Search failed: {message_ids_data}")

            # Parse message IDs
            message_ids = message_ids_data[0].split()
            if not message_ids:
                logger.info("imap_no_messages_found", folder=folder)
                return []

            # Limit to max_results
            message_ids = message_ids[:max_results]

            # Fetch each message
            for msg_id in message_ids:
                if len(all_emails) >= max_results:
                    break

                try:
                    status, msg_data = self.connection.fetch(msg_id, '(RFC822)')
                    if status != 'OK':
                        logger.warning("imap_fetch_message_failed", message_id=msg_id.decode())
                        continue

                    # Parse email
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            email_msg = self._parse_email_message(response_part[1], msg_id.decode())
                            all_emails.append(email_msg)
                            break

                except Exception as e:
                    logger.warning("imap_parse_email_failed", message_id=msg_id.decode(), error=str(e))
                    continue

            duration = (datetime.now() - start_time).total_seconds()
            logger.info("imap_fetch_success", emails_fetched=len(all_emails), duration_seconds=duration)

            # Emit metrics
            self.metrics_manager.emit({
                "metric": "imap_emails_fetched",
                "value": len(all_emails),
                "timestamp": datetime.now().isoformat(),
                "labels": {"server": self.imap_server, "folder": folder}
            })

            # Log activity
            await self._log_activity(
                activity_type="imap_fetch_emails",
                status="success",
                details={"emails_count": len(all_emails), "duration_seconds": duration}
            )

            return all_emails

        except imaplib.IMAP4.error as e:
            logger.error("imap_fetch_failed", error=str(e))
            await self._log_activity(
                activity_type="imap_fetch_emails",
                status="failed",
                error=str(e)
            )
            raise RuntimeError(f"IMAP operation failed: {e}")
        except Exception as e:
            logger.error("imap_fetch_failed", error=str(e))
            await self._log_activity(
                activity_type="imap_fetch_emails",
                status="failed",
                error=str(e)
            )
            raise

    def _parse_email_message(self, raw_email: bytes, message_id: str) -> EmailMessage:
        """Parse RFC822 email into EmailMessage model.

        Args:
            raw_email: Raw email bytes in RFC822 format
            message_id: IMAP message ID

        Returns:
            EmailMessage object
        """
        # Parse email
        msg = email.message_from_bytes(raw_email)

        # Extract subject
        subject = self._decode_header(msg.get('Subject', '(No Subject)'))

        # Extract sender
        sender = self._decode_header(msg.get('From', 'unknown@example.com'))
        # Extract email address from "Name <email>" format
        if '<' in sender and '>' in sender:
            sender = sender[sender.find('<') + 1:sender.find('>')]

        # Extract date
        date_str = msg.get('Date', '')
        try:
            from email.utils import parsedate_to_datetime
            received_at = parsedate_to_datetime(date_str) if date_str else datetime.now()
        except Exception:
            received_at = datetime.now()

        # Extract body
        body = self._extract_body(msg)

        # Extract snippet (first 100 chars of body)
        snippet = body[:100] if body else subject

        # Default importance score
        importance_score = 0.5

        # Create EmailMessage
        email_msg = EmailMessage(
            id=message_id,
            subject=subject,
            sender=sender,
            received_at=received_at,
            body=body,
            snippet=snippet,
            labels={'UNREAD'},
            importance_score=importance_score
        )

        logger.debug("imap_email_parsed", message_id=message_id, subject=subject)

        return email_msg

    def _decode_header(self, header_value: str) -> str:
        """Decode email header (handles encoding like UTF-8, quoted-printable).

        Args:
            header_value: Raw header value

        Returns:
            Decoded header string
        """
        if not header_value:
            return ""

        try:
            decoded_parts = decode_header(header_value)
            result = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    result += part.decode(encoding or 'utf-8', errors='replace')
                else:
                    result += part
            return result
        except Exception as e:
            logger.warning("header_decode_failed", error=str(e))
            return str(header_value)

    def _extract_body(self, msg: email.message.Message) -> str:
        """Extract body text from email message.

        Handles multipart messages by extracting text/plain or text/html parts.

        Args:
            msg: Parsed email message

        Returns:
            Email body text
        """
        body = ""

        try:
            if msg.is_multipart():
                # Multipart message - extract text parts
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))

                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue

                    # Extract text/plain or text/html
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode(errors='replace')
                            break  # Prefer plain text
                    elif content_type == "text/html" and not body:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode(errors='replace')
            else:
                # Simple message - get payload directly
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors='replace')

        except Exception as e:
            logger.warning("body_extraction_failed", error=str(e))
            body = "(Failed to extract body)"

        return body.strip()

    async def mark_as_read(self, message_id: str) -> None:
        """Mark a single email as read.

        Args:
            message_id: IMAP message ID to mark as read

        Raises:
            RuntimeError: If not authenticated
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated")

        try:
            status, response = self.connection.store(message_id, '+FLAGS', '\\Seen')
            if status != 'OK':
                raise RuntimeError(f"Failed to mark as read: {response}")

            logger.debug("imap_message_marked_read", message_id=message_id)

        except Exception as e:
            logger.error("imap_mark_read_failed", message_id=message_id, error=str(e))
            raise

    async def mark_as_read_batch(self, message_ids: List[str]) -> None:
        """Mark multiple emails as read.

        Args:
            message_ids: List of IMAP message IDs to mark as read
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated")

        if not message_ids:
            return

        try:
            # Execute in sequence (IMAP doesn't support parallel operations)
            for msg_id in message_ids:
                await self.mark_as_read(msg_id)

            logger.info("imap_batch_mark_read", count=len(message_ids))

        except Exception as e:
            logger.error("imap_batch_mark_read_failed", count=len(message_ids), error=str(e))
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
            activity_type: Type of activity (e.g., "imap_fetch_emails")
            status: Status ("success" or "failed")
            details: Optional activity details
            error: Optional error message
        """
        try:
            async with self.db_manager.get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO activity_log (timestamp, connector, activity, status, details)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(),
                        "imap",
                        activity_type,
                        status,
                        json.dumps(details or {}) if not error else json.dumps({"error": error})
                    )
                )
                await db.commit()
        except Exception as e:
            logger.warning("imap_activity_log_failed", error=str(e))

    def disconnect(self) -> None:
        """Disconnect from IMAP server."""
        if self.connection:
            try:
                self.connection.close()
                self.connection.logout()
                logger.info("imap_disconnected")
            except Exception as e:
                logger.warning("imap_disconnect_failed", error=str(e))
            finally:
                self.connection = None


# Singleton instance
_imap_connector_instance: Optional[IMAPConnector] = None


def get_imap_connector(
    imap_server: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs
) -> Optional[IMAPConnector]:
    """Get singleton IMAPConnector instance.

    Args:
        imap_server: IMAP server hostname (defaults to IMAP_SERVER env var)
        username: Email username (defaults to IMAP_USERNAME env var)
        password: Email password (defaults to IMAP_PASSWORD env var)
        **kwargs: Additional IMAPConnector arguments

    Returns:
        Global IMAPConnector instance or None if credentials not configured
    """
    import os

    global _imap_connector_instance

    if _imap_connector_instance is None:
        # Read from environment variables if not provided
        imap_server = imap_server or os.getenv('IMAP_SERVER')
        username = username or os.getenv('IMAP_USERNAME')
        password = password or os.getenv('IMAP_PASSWORD')

        # If still no credentials, return None
        if not imap_server or not username or not password:
            logger.debug("imap_connector_not_configured",
                        has_server=bool(imap_server),
                        has_username=bool(username),
                        has_password=bool(password))
            return None

        # Read additional settings from environment
        use_ssl = os.getenv('IMAP_USE_SSL', 'true').lower() == 'true'
        port = int(os.getenv('IMAP_PORT', '993' if use_ssl else '143'))

        # Merge kwargs with env settings
        final_kwargs = {
            'use_ssl': use_ssl,
            'port': port,
            **kwargs
        }

        _imap_connector_instance = IMAPConnector(
            imap_server=imap_server,
            username=username,
            password=password,
            **final_kwargs
        )

    return _imap_connector_instance

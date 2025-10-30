"""Outlook/Microsoft 365 Email Connector.

Provides email integration with Outlook/Microsoft 365 using Microsoft Graph API.

Features:
- OAuth 2.0 authentication via MSAL (Microsoft Authentication Library)
- Fetch unread emails with pagination
- Parse emails into EmailMessage models with RLHF support
- Batch operations (mark as read, categories)
- Rate limit handling with exponential backoff
- Metrics emission for observability
- Activity logging to database

OAuth Setup:
1. Register app in Azure AD portal (portal.azure.com)
2. Add "Mail.Read" and "Mail.ReadWrite" permissions
3. Configure redirect URI for your application
4. Save client_id, client_secret, and tenant_id

Example:
    ```python
    from src.connectors.email.outlook_connector import get_outlook_connector

    connector = get_outlook_connector(
        client_id="your_client_id",
        client_secret="your_client_secret",
        tenant_id="your_tenant_id"
    )
    connector.authenticate(authorization_code="...")
    emails = await connector.fetch_unread_emails(max_results=50)
    ```
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dateutil import parser as date_parser

try:
    import msal
    import requests
except ImportError:
    raise ImportError(
        "Microsoft Graph API dependencies not installed. "
        "Install with: pip install msal requests"
    )

from src.core.models import EmailMessage
from src.core.logging import get_logger
from src.core.metrics import get_metrics_manager
from src.database.sqlite_manager import get_db_manager

logger = get_logger(__name__)

# Microsoft Graph API configuration
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"
SCOPES = ["Mail.Read", "Mail.ReadWrite"]


class OutlookConnector:
    """Outlook/Microsoft 365 email connector using Microsoft Graph API.

    Supports:
    - OAuth 2.0 via MSAL
    - Fetching unread emails with pagination
    - Parsing emails with importance scoring
    - Batch operations
    - Rate limit handling
    - Metrics and activity logging

    Attributes:
        client_id: Azure AD application (client) ID
        client_secret: Azure AD application secret
        tenant_id: Azure AD tenant (directory) ID
        access_token: Current OAuth access token
        token_cache_path: Path to token cache file
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        token_cache_path: str = "credentials/outlook_token_cache.json"
    ):
        """Initialize Outlook connector.

        Args:
            client_id: Azure AD application (client) ID
            client_secret: Azure AD application secret
            tenant_id: Azure AD tenant (directory) ID
            token_cache_path: Path to token cache file

        Raises:
            ValueError: If credentials are missing
        """
        if not client_id or not client_secret or not tenant_id:
            raise ValueError("client_id, client_secret, and tenant_id are required")

        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.token_cache_path = token_cache_path
        self.access_token: Optional[str] = None

        # Initialize MSAL app
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"
        self.msal_app: Optional[msal.ConfidentialClientApplication] = None

        # Observability
        self.metrics_manager = get_metrics_manager()
        self.db_manager = get_db_manager()

        logger.info("outlook_connector_initialized", tenant_id=tenant_id)

    @property
    def is_authenticated(self) -> bool:
        """Check if connector is authenticated."""
        return self.access_token is not None

    def _get_msal_app(self) -> msal.ConfidentialClientApplication:
        """Get or create MSAL application with token cache.

        Returns:
            MSAL ConfidentialClientApplication instance
        """
        if self.msal_app is None:
            # Load token cache if exists
            cache = msal.SerializableTokenCache()
            if os.path.exists(self.token_cache_path):
                with open(self.token_cache_path, 'r') as f:
                    cache.deserialize(f.read())

            self.msal_app = msal.ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=self.authority,
                token_cache=cache
            )

        return self.msal_app

    def _save_token_cache(self):
        """Save token cache to disk."""
        if self.msal_app and self.msal_app.token_cache.has_state_changed:
            os.makedirs(os.path.dirname(self.token_cache_path), exist_ok=True)
            with open(self.token_cache_path, 'w') as f:
                f.write(self.msal_app.token_cache.serialize())

    def authenticate(self, authorization_code: Optional[str] = None, redirect_uri: str = "http://localhost"):
        """Authenticate with Microsoft Graph API using OAuth 2.0.

        Tries to use cached token first, then falls back to authorization code flow.

        Args:
            authorization_code: OAuth authorization code (for first-time auth)
            redirect_uri: Redirect URI configured in Azure AD app
        """
        start_time = datetime.now()

        try:
            app = self._get_msal_app()

            # Try to get cached token
            accounts = app.get_accounts()
            if accounts:
                logger.info("outlook_auth_trying_cached_token")
                result = app.acquire_token_silent(SCOPES, account=accounts[0])
                if result and 'access_token' in result:
                    self.access_token = result['access_token']
                    logger.info("outlook_auth_cached_token_success")
                    self._save_token_cache()
                    return

            # Use authorization code flow if provided
            if authorization_code:
                logger.info("outlook_auth_using_authorization_code")
                result = app.acquire_token_by_authorization_code(
                    code=authorization_code,
                    scopes=SCOPES,
                    redirect_uri=redirect_uri
                )

                if 'access_token' in result:
                    self.access_token = result['access_token']
                    self._save_token_cache()
                    logger.info("outlook_authentication_success",
                               duration_seconds=(datetime.now() - start_time).total_seconds())
                    return
                else:
                    error = result.get('error_description', result.get('error', 'Unknown error'))
                    raise RuntimeError(f"Authentication failed: {error}")

            # If we get here, no cached token and no auth code
            raise RuntimeError(
                "No cached token found and no authorization code provided. "
                "Please provide an authorization code for first-time authentication."
            )

        except Exception as e:
            logger.error("outlook_authentication_failed", error=str(e),
                        duration_seconds=(datetime.now() - start_time).total_seconds())
            raise

    async def fetch_unread_emails(
        self,
        max_results: int = 50,
        filter_query: Optional[str] = None,
        max_retries: int = 3
    ) -> List[EmailMessage]:
        """Fetch unread emails from Outlook/Microsoft 365.

        Args:
            max_results: Maximum number of emails to fetch
            filter_query: OData filter query (e.g., "from/emailAddress/address eq 'user@example.com'")
            max_retries: Maximum retry attempts for rate limits

        Returns:
            List of EmailMessage objects

        Raises:
            RuntimeError: If not authenticated
            requests.HTTPError: If API call fails
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        start_time = datetime.now()
        all_emails: List[EmailMessage] = []

        try:
            logger.info("outlook_fetching_emails", max_results=max_results, filter_query=filter_query)

            # Build query parameters
            params = {
                '$top': min(max_results, 100),  # Graph API max is 100 per page
                '$select': 'id,subject,from,receivedDateTime,body,isRead,importance,categories',
                '$orderby': 'receivedDateTime DESC'
            }

            # Add filter for unread emails
            base_filter = "isRead eq false"
            if filter_query:
                params['$filter'] = f"({base_filter}) and ({filter_query})"
            else:
                params['$filter'] = base_filter

            # Fetch emails with pagination
            url = f"{GRAPH_API_ENDPOINT}/me/messages"
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }

            while url and len(all_emails) < max_results:
                retry_count = 0

                while retry_count <= max_retries:
                    try:
                        response = requests.get(url, headers=headers, params=params if retry_count == 0 else None)

                        if response.status_code == 429:
                            # Rate limit - respect Retry-After header
                            retry_after = int(response.headers.get('Retry-After', 2 ** retry_count))
                            logger.warning("outlook_rate_limit", retry=retry_count, wait_seconds=retry_after)
                            await asyncio.sleep(retry_after)
                            retry_count += 1
                            continue

                        response.raise_for_status()
                        data = response.json()

                        # Parse emails
                        for email_data in data.get('value', []):
                            if len(all_emails) >= max_results:
                                break

                            email_msg = self._parse_email_message(email_data)
                            all_emails.append(email_msg)

                        # Get next page URL
                        url = data.get('@odata.nextLink')
                        params = None  # Params are included in nextLink
                        break  # Success, exit retry loop

                    except requests.HTTPError as e:
                        # Check if it's a rate limit error
                        if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                            retry_count += 1
                            if retry_count > max_retries:
                                raise
                        else:
                            raise

            duration = (datetime.now() - start_time).total_seconds()
            logger.info("outlook_fetch_success", emails_fetched=len(all_emails), duration_seconds=duration)

            # Emit metrics
            self.metrics_manager.emit({
                "metric": "outlook_emails_fetched",
                "value": len(all_emails),
                "timestamp": datetime.now().isoformat(),
                "labels": {"filter": filter_query or "unread"}
            })

            # Log activity
            await self._log_activity(
                activity_type="outlook_fetch_emails",
                status="success",
                details={"emails_count": len(all_emails), "duration_seconds": duration}
            )

            return all_emails

        except Exception as e:
            logger.error("outlook_fetch_failed", error=str(e))
            await self._log_activity(
                activity_type="outlook_fetch_emails",
                status="failed",
                error=str(e)
            )
            raise

    def _parse_email_message(self, email_data: Dict[str, Any]) -> EmailMessage:
        """Parse Microsoft Graph API email into EmailMessage model.

        Args:
            email_data: Raw email data from Graph API

        Returns:
            EmailMessage object with RLHF support
        """
        # Extract basic fields
        email_id = email_data.get('id', '')
        subject = email_data.get('subject', '(No Subject)')

        # Parse sender
        from_data = email_data.get('from', {}).get('emailAddress', {})
        sender = from_data.get('address', 'unknown@example.com')

        # Parse received date
        received_str = email_data.get('receivedDateTime', '')
        received_at = date_parser.isoparse(received_str) if received_str else datetime.now()

        # Extract body
        body_data = email_data.get('body', {})
        body = body_data.get('content', '(No body content)')

        # Parse labels/categories
        categories = email_data.get('categories', [])
        labels = list(categories)
        if not email_data.get('isRead', True):
            labels.append('UNREAD')

        # Calculate importance score
        importance = email_data.get('importance', 'normal').lower()
        importance_score = 0.5  # Base score

        if importance == 'high':
            importance_score += 0.2
        elif importance == 'low':
            importance_score -= 0.1

        if 'Important' in categories:
            importance_score += 0.1

        # Clamp to 0.0-1.0
        importance_score = max(0.0, min(1.0, importance_score))

        # Create EmailMessage
        email_msg = EmailMessage(
            id=email_id,
            subject=subject,
            sender=sender,
            received_at=received_at,
            body=body,
            snippet=subject,  # Graph API doesn't have snippet, use subject
            labels=set(labels),
            importance_score=importance_score
        )

        logger.debug("outlook_email_parsed", email_id=email_id, subject=subject, importance=importance_score)

        return email_msg

    async def mark_as_read(self, message_id: str) -> None:
        """Mark a single email as read.

        Args:
            message_id: Email message ID to mark as read

        Raises:
            RuntimeError: If not authenticated
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated")

        try:
            url = f"{GRAPH_API_ENDPOINT}/me/messages/{message_id}"
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            data = {'isRead': True}

            response = requests.patch(url, headers=headers, json=data)
            response.raise_for_status()

            logger.debug("outlook_message_marked_read", message_id=message_id)

        except Exception as e:
            logger.error("outlook_mark_read_failed", message_id=message_id, error=str(e))
            raise

    async def mark_as_read_batch(self, message_ids: List[str]) -> None:
        """Mark multiple emails as read.

        Note: Graph API doesn't have native batch modify for mail,
        so we use individual PATCH requests in parallel.

        Args:
            message_ids: List of email message IDs to mark as read
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated")

        if not message_ids:
            return

        try:
            # Execute in parallel using asyncio.gather
            tasks = [self.mark_as_read(msg_id) for msg_id in message_ids]
            await asyncio.gather(*tasks, return_exceptions=True)

            logger.info("outlook_batch_mark_read", count=len(message_ids))

        except Exception as e:
            logger.error("outlook_batch_mark_read_failed", count=len(message_ids), error=str(e))
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
            activity_type: Type of activity (e.g., "outlook_fetch_emails")
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
                        "outlook",
                        activity_type,
                        status,
                        json.dumps(details or {}) if not error else json.dumps({"error": error})
                    )
                )
                await db.commit()
        except Exception as e:
            logger.warning("outlook_activity_log_failed", error=str(e))


# Singleton instance
_outlook_connector_instance: Optional[OutlookConnector] = None


def get_outlook_connector(
    client_id: str,
    client_secret: str,
    tenant_id: str
) -> OutlookConnector:
    """Get singleton OutlookConnector instance.

    Args:
        client_id: Azure AD application (client) ID
        client_secret: Azure AD application secret
        tenant_id: Azure AD tenant (directory) ID

    Returns:
        Global OutlookConnector instance
    """
    global _outlook_connector_instance

    if _outlook_connector_instance is None:
        _outlook_connector_instance = OutlookConnector(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id
        )

    return _outlook_connector_instance

"""Credential management service for OAuth tokens and API keys.

This service provides secure storage and retrieval of OAuth credentials using
encrypted storage with salt+pepper encryption.

Reference: CREDENTIAL_MANAGEMENT_PLAN.md
"""

import json
import ulid
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.database.sqlite_manager import SQLiteManager
from src.core.logging import get_logger

logger = get_logger(__name__)


class CredentialService:
    """Service for managing encrypted OAuth credentials.

    Features:
    - Secure encrypted storage using EncryptionService
    - CRUD operations for credentials
    - Service-based credential lookup
    - Status tracking (active/expired/revoked)
    - Metadata management separate from tokens
    """

    def __init__(self):
        """Initialize CredentialService."""
        self.db = None

    async def initialize(self, db_path: str = None):
        """Initialize database connection.

        Args:
            db_path: Optional database path. If None, uses default path.
                    Use ":memory:" only for tests.
        """
        if self.db is None:
            if db_path is None:
                from src.database.sqlite_manager import get_db_manager
                self.db = get_db_manager()
            else:
                self.db = SQLiteManager(db_path)
            await self.db.initialize()

    async def add_credential(
        self,
        service_type: str,
        account_email: str,
        oauth_token: Dict[str, Any],
        pepper_key: str
    ) -> str:
        """Add a new encrypted credential.

        Args:
            service_type: Type of service (gmail, google_calendar, outlook, etc.)
            account_email: Account email address
            oauth_token: OAuth token data (will be encrypted)
            pepper_key: User's pepper key for encryption

        Returns:
            Credential ID (ULID)

        Example:
            >>> service = CredentialService()
            >>> await service.initialize()
            >>> token = {
            ...     "access_token": "ya29.a0AfB...",
            ...     "refresh_token": "1//0gHZ9K...",
            ...     "scopes": ["https://www.googleapis.com/auth/gmail.readonly"]
            ... }
            >>> cred_id = await service.add_credential(
            ...     "gmail",
            ...     "user@gmail.com",
            ...     token,
            ...     "my-pepper-key"
            ... )
        """
        await self.initialize()

        # Generate credential ID
        cred_id = str(ulid.ULID())

        # Store metadata in kv_store
        metadata = {
            'id': cred_id,
            'service_type': service_type,
            'account_email': account_email,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }

        await self.db.set_setting(f"credential_meta_{cred_id}", metadata)

        # Store encrypted OAuth token
        token_json = json.dumps(oauth_token)
        await self.db.set_encrypted_setting(
            f"credential_token_{cred_id}",
            token_json,
            pepper_key=pepper_key
        )

        logger.info(
            "credential_added",
            cred_id=cred_id,
            service_type=service_type,
            account_email=account_email
        )

        return cred_id

    async def list_credentials(self) -> List[Dict[str, Any]]:
        """List all stored credentials (metadata only, no tokens).

        Returns:
            List of credential metadata dictionaries (without oauth_token field)

        Example:
            >>> credentials = await service.list_credentials()
            >>> for cred in credentials:
            ...     print(f"{cred['service_type']}: {cred['account_email']}")
        """
        await self.initialize()

        # Get all settings and filter for credential_meta_* keys
        all_settings = await self.db.get_all_settings()
        credentials = []

        for key, value in all_settings.items():
            if key.startswith("credential_meta_"):
                credentials.append(value)

        return credentials

    async def get_credential(
        self,
        cred_id: str,
        pepper_key: str
    ) -> Optional[Dict[str, Any]]:
        """Get credential with decrypted token.

        Args:
            cred_id: Credential ID
            pepper_key: Pepper key for decryption

        Returns:
            Credential dict with 'oauth_token' field containing decrypted token,
            or None if not found

        Raises:
            Exception: If decryption fails (wrong pepper, corrupted data)
        """
        await self.initialize()

        # Get metadata
        metadata = await self.db.get_setting(f"credential_meta_{cred_id}")
        if not metadata:
            return None

        # Get decrypted token
        token_json = await self.db.get_encrypted_setting(
            f"credential_token_{cred_id}",
            pepper_key=pepper_key
        )

        if token_json:
            metadata['oauth_token'] = json.loads(token_json)

        return metadata

    async def get_credential_by_service(
        self,
        service_type: str,
        pepper_key: str
    ) -> Optional[Dict[str, Any]]:
        """Get credential by service type.

        If multiple credentials exist for the same service, returns the first one.

        Args:
            service_type: Service type (gmail, google_calendar, etc.)
            pepper_key: Pepper key for decryption

        Returns:
            Credential dict or None if not found
        """
        credentials = await self.list_credentials()

        for cred in credentials:
            if cred['service_type'] == service_type:
                return await self.get_credential(cred['id'], pepper_key)

        return None

    async def update_credential(
        self,
        cred_id: str,
        oauth_token: Dict[str, Any],
        pepper_key: str
    ):
        """Update credential token (e.g., after token refresh).

        Args:
            cred_id: Credential ID
            oauth_token: New OAuth token data
            pepper_key: Pepper key for encryption
        """
        await self.initialize()

        # Update token
        token_json = json.dumps(oauth_token)
        await self.db.set_encrypted_setting(
            f"credential_token_{cred_id}",
            token_json,
            pepper_key=pepper_key
        )

        # Update metadata timestamp
        metadata = await self.db.get_setting(f"credential_meta_{cred_id}")
        if metadata:
            metadata['updated_at'] = datetime.utcnow().isoformat()
            await self.db.set_setting(f"credential_meta_{cred_id}", metadata)

        logger.info("credential_updated", cred_id=cred_id)

    async def remove_credential(self, cred_id: str):
        """Remove credential (both metadata and encrypted token).

        Args:
            cred_id: Credential ID
        """
        await self.initialize()

        # Delete metadata and token
        await self.db.delete_setting(f"credential_meta_{cred_id}")
        await self.db.delete_setting(f"credential_token_{cred_id}")

        logger.info("credential_removed", cred_id=cred_id)

    async def update_status(self, cred_id: str, status: str):
        """Update credential status.

        Args:
            cred_id: Credential ID
            status: New status (active, expired, revoked, error)
        """
        await self.initialize()

        metadata = await self.db.get_setting(f"credential_meta_{cred_id}")
        if metadata:
            metadata['status'] = status
            metadata['updated_at'] = datetime.utcnow().isoformat()
            await self.db.set_setting(f"credential_meta_{cred_id}", metadata)

        logger.info("credential_status_updated", cred_id=cred_id, status=status)

    async def get_credential_raw(self, cred_id: str) -> Optional[Dict[str, Any]]:
        """Get raw encrypted credential (for testing).

        Args:
            cred_id: Credential ID

        Returns:
            Raw encrypted data dict or None if not found
        """
        await self.initialize()
        return await self.db.get_setting(f"credential_token_{cred_id}")


# Singleton instance
_credential_service: Optional[CredentialService] = None


def get_credential_service() -> CredentialService:
    """Get or create CredentialService singleton.

    Returns:
        CredentialService instance
    """
    global _credential_service
    if _credential_service is None:
        _credential_service = CredentialService()
    return _credential_service

"""Unit tests for CredentialService - Secure credential management."""

import pytest
import json
from src.services.credential_service import CredentialService


class TestCredentialServiceInitialization:
    """Test CredentialService initialization."""

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test CredentialService initializes correctly."""
        service = CredentialService()
        assert service is not None
        await service.initialize()
        assert service.db is not None


class TestAddCredential:
    """Test adding credentials."""

    @pytest.mark.asyncio
    async def test_add_gmail_credential(self):
        """Test adding Gmail OAuth credential."""
        service = CredentialService()
        await service.initialize()

        oauth_token = {
            "access_token": "ya29.a0AfB_test",
            "refresh_token": "1//0gHZ9K_test",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "123.apps.googleusercontent.com",
            "client_secret": "secret123",
            "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
            "expiry": "2025-10-27T12:00:00Z"
        }

        cred_id = await service.add_credential(
            service_type="gmail",
            account_email="user@gmail.com",
            oauth_token=oauth_token,
            pepper_key="user-pepper"
        )

        assert cred_id is not None
        assert len(cred_id) > 0  # Should be a ULID

    @pytest.mark.asyncio
    async def test_add_calendar_credential(self):
        """Test adding Google Calendar credential."""
        service = CredentialService()
        await service.initialize()

        oauth_token = {
            "access_token": "ya29.calendar_test",
            "refresh_token": "1//calendar_refresh",
            "scopes": ["https://www.googleapis.com/auth/calendar.readonly"]
        }

        cred_id = await service.add_credential(
            service_type="google_calendar",
            account_email="user@gmail.com",
            oauth_token=oauth_token,
            pepper_key="user-pepper"
        )

        assert cred_id is not None

    @pytest.mark.asyncio
    async def test_add_credential_stores_encrypted(self):
        """Test that credential token is stored encrypted."""
        service = CredentialService()
        await service.initialize()

        oauth_token = {"access_token": "secret_token_123"}

        cred_id = await service.add_credential(
            service_type="gmail",
            account_email="test@gmail.com",
            oauth_token=oauth_token,
            pepper_key="test-pepper"
        )

        # Verify token is encrypted in storage
        raw_token = await service.get_credential_raw(cred_id)
        assert raw_token is not None
        assert raw_token.get('is_encrypted') is True
        # Should be base64-encoded string, not plain dict
        assert isinstance(raw_token.get('ciphertext'), str)


class TestListCredentials:
    """Test listing stored credentials."""

    @pytest.mark.asyncio
    async def test_list_empty_credentials(self):
        """Test listing when no credentials stored."""
        service = CredentialService()
        await service.initialize()

        credentials = await service.list_credentials()

        assert credentials == []

    @pytest.mark.asyncio
    async def test_list_multiple_credentials(self):
        """Test listing multiple stored credentials."""
        service = CredentialService()
        await service.initialize()

        # Add two credentials
        await service.add_credential("gmail", "user1@gmail.com", {"token": "1"}, "pepper1")
        await service.add_credential("google_calendar", "user2@gmail.com", {"token": "2"}, "pepper2")

        credentials = await service.list_credentials()

        assert len(credentials) == 2
        # Should return metadata only, not tokens
        assert 'oauth_token' not in credentials[0]
        assert 'oauth_token' not in credentials[1]

    @pytest.mark.asyncio
    async def test_list_credentials_shows_metadata(self):
        """Test that list returns credential metadata."""
        service = CredentialService()
        await service.initialize()

        await service.add_credential("gmail", "user@gmail.com", {"token": "abc"}, "pepper")

        credentials = await service.list_credentials()

        assert len(credentials) == 1
        cred = credentials[0]
        assert cred['service_type'] == "gmail"
        assert cred['account_email'] == "user@gmail.com"
        assert cred['status'] == "active"
        assert 'created_at' in cred
        assert 'updated_at' in cred
        assert 'id' in cred


class TestGetCredential:
    """Test retrieving credentials."""

    @pytest.mark.asyncio
    async def test_get_credential_decrypted(self):
        """Test retrieving and decrypting credential."""
        service = CredentialService()
        await service.initialize()

        token = {"access_token": "secret123", "refresh_token": "refresh456"}
        cred_id = await service.add_credential("gmail", "user@gmail.com", token, "pepper")

        decrypted = await service.get_credential(cred_id, pepper_key="pepper")

        assert decrypted is not None
        assert decrypted['oauth_token']['access_token'] == "secret123"
        assert decrypted['oauth_token']['refresh_token'] == "refresh456"

    @pytest.mark.asyncio
    async def test_get_nonexistent_credential(self):
        """Test getting non-existent credential returns None."""
        service = CredentialService()
        await service.initialize()

        result = await service.get_credential("nonexistent_id", pepper_key="pepper")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_credential_wrong_pepper_fails(self):
        """Test getting credential with wrong pepper fails."""
        service = CredentialService()
        await service.initialize()

        cred_id = await service.add_credential("gmail", "user@gmail.com", {"token": "abc"}, "correct_pepper")

        with pytest.raises(Exception):
            await service.get_credential(cred_id, pepper_key="wrong_pepper")


class TestGetCredentialByService:
    """Test getting credential by service type."""

    @pytest.mark.asyncio
    async def test_get_credential_by_service(self):
        """Test retrieving credential by service type."""
        service = CredentialService()
        await service.initialize()

        token = {"access_token": "gmail_token"}
        await service.add_credential("gmail", "user@gmail.com", token, "pepper")

        cred = await service.get_credential_by_service("gmail", pepper_key="pepper")

        assert cred is not None
        assert cred['service_type'] == "gmail"
        assert cred['oauth_token']['access_token'] == "gmail_token"

    @pytest.mark.asyncio
    async def test_get_credential_by_nonexistent_service(self):
        """Test getting credential for non-existent service."""
        service = CredentialService()
        await service.initialize()

        await service.add_credential("gmail", "user@gmail.com", {"token": "abc"}, "pepper")

        result = await service.get_credential_by_service("outlook", pepper_key="pepper")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_first_credential_when_multiple_same_service(self):
        """Test getting first credential when multiple exist for same service."""
        service = CredentialService()
        await service.initialize()

        # Add two Gmail credentials
        await service.add_credential("gmail", "user1@gmail.com", {"token": "first"}, "pepper")
        await service.add_credential("gmail", "user2@gmail.com", {"token": "second"}, "pepper")

        cred = await service.get_credential_by_service("gmail", pepper_key="pepper")

        # Should return the first one
        assert cred is not None
        assert cred['oauth_token']['token'] == "first"


class TestUpdateCredential:
    """Test updating credentials."""

    @pytest.mark.asyncio
    async def test_update_credential_token(self):
        """Test updating credential token (e.g., after refresh)."""
        service = CredentialService()
        await service.initialize()

        token = {"access_token": "old_token"}
        cred_id = await service.add_credential("gmail", "user@gmail.com", token, "pepper")

        new_token = {"access_token": "new_token", "refresh_token": "new_refresh"}
        await service.update_credential(cred_id, new_token, pepper_key="pepper")

        updated = await service.get_credential(cred_id, pepper_key="pepper")
        assert updated['oauth_token']['access_token'] == "new_token"
        assert updated['oauth_token']['refresh_token'] == "new_refresh"

    @pytest.mark.asyncio
    async def test_update_credential_updates_timestamp(self):
        """Test that updating credential updates the timestamp."""
        service = CredentialService()
        await service.initialize()

        cred_id = await service.add_credential("gmail", "user@gmail.com", {"token": "abc"}, "pepper")

        original = await service.get_credential(cred_id, pepper_key="pepper")
        original_updated_at = original['updated_at']

        import asyncio
        await asyncio.sleep(0.1)  # Small delay to ensure timestamp changes

        await service.update_credential(cred_id, {"token": "xyz"}, pepper_key="pepper")

        updated = await service.get_credential(cred_id, pepper_key="pepper")
        assert updated['updated_at'] > original_updated_at


class TestRemoveCredential:
    """Test removing credentials."""

    @pytest.mark.asyncio
    async def test_remove_credential(self):
        """Test removing credential."""
        service = CredentialService()
        await service.initialize()

        cred_id = await service.add_credential("gmail", "user@gmail.com", {"token": "abc"}, "pepper")

        await service.remove_credential(cred_id)

        credentials = await service.list_credentials()
        assert len(credentials) == 0

    @pytest.mark.asyncio
    async def test_remove_credential_removes_token(self):
        """Test that removing credential also removes encrypted token."""
        service = CredentialService()
        await service.initialize()

        cred_id = await service.add_credential("gmail", "user@gmail.com", {"token": "abc"}, "pepper")

        await service.remove_credential(cred_id)

        # Verify token is also removed
        raw_token = await service.get_credential_raw(cred_id)
        assert raw_token is None

    @pytest.mark.asyncio
    async def test_remove_one_credential_keeps_others(self):
        """Test removing one credential doesn't affect others."""
        service = CredentialService()
        await service.initialize()

        cred_id1 = await service.add_credential("gmail", "user1@gmail.com", {"token": "1"}, "pepper")
        cred_id2 = await service.add_credential("google_calendar", "user2@gmail.com", {"token": "2"}, "pepper")

        await service.remove_credential(cred_id1)

        credentials = await service.list_credentials()
        assert len(credentials) == 1
        assert credentials[0]['service_type'] == "google_calendar"


class TestCredentialStatus:
    """Test credential status management."""

    @pytest.mark.asyncio
    async def test_update_credential_status(self):
        """Test updating credential status."""
        service = CredentialService()
        await service.initialize()

        cred_id = await service.add_credential("gmail", "user@gmail.com", {"token": "abc"}, "pepper")

        # Mark as expired
        await service.update_status(cred_id, "expired")

        cred = await service.get_credential(cred_id, pepper_key="pepper")
        assert cred['status'] == "expired"

    @pytest.mark.asyncio
    async def test_credential_status_values(self):
        """Test different credential status values."""
        service = CredentialService()
        await service.initialize()

        cred_id = await service.add_credential("gmail", "user@gmail.com", {"token": "abc"}, "pepper")

        # Test different statuses
        for status in ['active', 'expired', 'revoked', 'error']:
            await service.update_status(cred_id, status)
            cred = await service.get_credential(cred_id, pepper_key="pepper")
            assert cred['status'] == status


class TestCredentialServiceSingleton:
    """Test singleton pattern."""

    def test_get_credential_service_singleton(self):
        """Test get_credential_service returns singleton."""
        from src.services.credential_service import get_credential_service

        service1 = get_credential_service()
        service2 = get_credential_service()

        assert service1 is service2

"""E2E tests for credential management flow.

This module tests the complete end-to-end flow:
1. OAuth authorization with mock server
2. Token exchange
3. Storing credentials with CredentialService
4. Using credentials to access Gmail API (mock)

Mock servers must be running on ports 5001 (OAuth) and 5002 (Gmail).
"""

import pytest
import pytest_asyncio
import requests
import json
import base64
from urllib.parse import urlparse, parse_qs

from src.services.credential_service import CredentialService


# Mock server URLs
OAUTH_SERVER_URL = "http://localhost:5001"
GMAIL_SERVER_URL = "http://localhost:5002"


@pytest_asyncio.fixture(scope="function")
async def credential_service():
    """Get credential service with clean database."""
    service = CredentialService()
    await service.initialize()

    # Clean up any existing test credentials
    all_settings = await service.db.get_all_settings()
    for key in all_settings.keys():
        if key.startswith("credential_"):
            await service.db.delete_setting(key)

    yield service

    # Cleanup after test
    all_settings = await service.db.get_all_settings()
    for key in all_settings.keys():
        if key.startswith("credential_"):
            await service.db.delete_setting(key)


class TestMockServerConnection:
    """Test connectivity to mock servers."""

    def test_oauth_server_health(self):
        """Test OAuth mock server is running."""
        response = requests.get(f"{OAUTH_SERVER_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "oauth_mock_server"

    def test_gmail_server_health(self):
        """Test Gmail mock server is running."""
        response = requests.get(f"{GMAIL_SERVER_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "gmail_mock_api"


class TestOAuthFlow:
    """Test OAuth authorization flow with mock server."""

    def test_oauth_authorization_request(self):
        """Test OAuth authorization endpoint returns redirect."""
        params = {
            "client_id": "test_client",
            "redirect_uri": "http://localhost:8501/callback",
            "scope": "gmail.readonly",
            "state": "test_state_123",
            "response_type": "code"
        }

        # Don't follow redirects so we can check the response
        response = requests.get(
            f"{OAUTH_SERVER_URL}/oauth/authorize",
            params=params,
            allow_redirects=False
        )

        assert response.status_code == 302
        assert "Location" in response.headers

        # Parse redirect URL
        location = response.headers["Location"]
        assert location.startswith("http://localhost:8501/callback")

        # Parse query parameters from redirect
        parsed = urlparse(location)
        query_params = parse_qs(parsed.query)

        assert "code" in query_params
        assert "state" in query_params
        assert query_params["state"][0] == "test_state_123"

        # Store auth code for next test
        auth_code = query_params["code"][0]
        assert auth_code.startswith("mock_auth_")

    def test_oauth_token_exchange(self):
        """Test exchanging auth code for tokens."""
        # First get an auth code
        params = {
            "client_id": "test_client",
            "redirect_uri": "http://localhost:8501/callback",
            "scope": "gmail.readonly",
            "state": "test_state",
            "response_type": "code"
        }

        auth_response = requests.get(
            f"{OAUTH_SERVER_URL}/oauth/authorize",
            params=params,
            allow_redirects=False
        )

        location = auth_response.headers["Location"]
        parsed = urlparse(location)
        query_params = parse_qs(parsed.query)
        auth_code = query_params["code"][0]

        # Exchange code for tokens
        token_data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": "test_client",
            "client_secret": "test_secret",
            "redirect_uri": "http://localhost:8501/callback"
        }

        token_response = requests.post(
            f"{OAUTH_SERVER_URL}/oauth/token",
            data=token_data
        )

        assert token_response.status_code == 200
        tokens = token_response.json()

        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert "token_type" in tokens
        assert "expires_in" in tokens
        assert "scope" in tokens

        assert tokens["token_type"] == "Bearer"
        assert tokens["expires_in"] == 3600
        assert tokens["scope"] == "gmail.readonly"
        assert tokens["access_token"].startswith("ya29.mock_")
        assert tokens["refresh_token"].startswith("1//mock_")

    def test_oauth_token_refresh(self):
        """Test refreshing access token with refresh token."""
        # Get initial tokens
        params = {
            "client_id": "test_client",
            "redirect_uri": "http://localhost:8501/callback",
            "scope": "gmail.readonly",
            "state": "test_state",
            "response_type": "code"
        }

        auth_response = requests.get(
            f"{OAUTH_SERVER_URL}/oauth/authorize",
            params=params,
            allow_redirects=False
        )

        location = auth_response.headers["Location"]
        parsed = urlparse(location)
        query_params = parse_qs(parsed.query)
        auth_code = query_params["code"][0]

        token_data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": "test_client",
            "client_secret": "test_secret",
            "redirect_uri": "http://localhost:8501/callback"
        }

        token_response = requests.post(f"{OAUTH_SERVER_URL}/oauth/token", data=token_data)
        tokens = token_response.json()
        refresh_token = tokens["refresh_token"]

        # Use refresh token to get new access token
        refresh_data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": "test_client",
            "client_secret": "test_secret"
        }

        refresh_response = requests.post(
            f"{OAUTH_SERVER_URL}/oauth/token",
            data=refresh_data
        )

        assert refresh_response.status_code == 200
        new_tokens = refresh_response.json()

        assert "access_token" in new_tokens
        assert new_tokens["token_type"] == "Bearer"
        assert new_tokens["expires_in"] == 3600
        assert new_tokens["access_token"].startswith("ya29.mock_")
        # New access token should be different
        assert new_tokens["access_token"] != tokens["access_token"]


class TestGmailAPI:
    """Test Gmail API with mock server."""

    def test_list_gmail_messages(self):
        """Test listing Gmail messages with mock server."""
        headers = {"Authorization": "Bearer test_token"}

        response = requests.get(
            f"{GMAIL_SERVER_URL}/gmail/v1/users/me/messages",
            headers=headers,
            params={"maxResults": 5}
        )

        assert response.status_code == 200
        data = response.json()

        assert "messages" in data
        assert "resultSizeEstimate" in data
        assert len(data["messages"]) == 5
        assert data["resultSizeEstimate"] == 5

        # Check message format
        for msg in data["messages"]:
            assert "id" in msg
            assert "threadId" in msg

    def test_get_gmail_message_by_id(self):
        """Test getting specific Gmail message."""
        headers = {"Authorization": "Bearer test_token"}

        response = requests.get(
            f"{GMAIL_SERVER_URL}/gmail/v1/users/me/messages/msg001",
            headers=headers
        )

        assert response.status_code == 200
        message = response.json()

        assert message["id"] == "msg001"
        assert "payload" in message
        assert "headers" in message["payload"]
        assert "body" in message["payload"]
        assert "labelIds" in message

        # Check headers
        headers_list = message["payload"]["headers"]
        subjects = [h for h in headers_list if h["name"] == "Subject"]
        assert len(subjects) == 1
        assert "URGENT" in subjects[0]["value"]

        # Check body is base64 encoded
        body_data = message["payload"]["body"]["data"]
        assert isinstance(body_data, str)
        # Decode to verify it's valid base64
        decoded = base64.b64decode(body_data).decode('utf-8')
        assert "Q4 report" in decoded

    def test_gmail_unauthorized_without_token(self):
        """Test Gmail API returns 401 without auth token."""
        response = requests.get(
            f"{GMAIL_SERVER_URL}/gmail/v1/users/me/messages"
        )

        assert response.status_code == 401


class TestCredentialIntegration:
    """Test complete credential flow integration."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_oauth_credentials(self, credential_service):
        """Test storing OAuth credentials after flow."""
        # Simulate OAuth flow
        params = {
            "client_id": "test_client",
            "redirect_uri": "http://localhost:8501/callback",
            "scope": "gmail.readonly",
            "state": "test_state",
            "response_type": "code"
        }

        auth_response = requests.get(
            f"{OAUTH_SERVER_URL}/oauth/authorize",
            params=params,
            allow_redirects=False
        )

        location = auth_response.headers["Location"]
        parsed = urlparse(location)
        query_params = parse_qs(parsed.query)
        auth_code = query_params["code"][0]

        token_data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": "test_client",
            "client_secret": "test_secret",
            "redirect_uri": "http://localhost:8501/callback"
        }

        token_response = requests.post(f"{OAUTH_SERVER_URL}/oauth/token", data=token_data)
        oauth_tokens = token_response.json()

        # Store credentials
        cred_id = await credential_service.add_credential(
            service_type="gmail",
            account_email="test@gmail.com",
            oauth_token=oauth_tokens,
            pepper_key="test_pepper"
        )

        assert cred_id is not None
        assert len(cred_id) > 0

        # Retrieve credentials
        retrieved = await credential_service.get_credential(cred_id, pepper_key="test_pepper")

        assert retrieved is not None
        assert retrieved["service_type"] == "gmail"
        assert retrieved["account_email"] == "test@gmail.com"
        assert retrieved["status"] == "active"
        assert "oauth_token" in retrieved

        # Verify token was decrypted correctly
        retrieved_token = retrieved["oauth_token"]
        assert retrieved_token["access_token"] == oauth_tokens["access_token"]
        assert retrieved_token["refresh_token"] == oauth_tokens["refresh_token"]
        assert retrieved_token["token_type"] == oauth_tokens["token_type"]

    @pytest.mark.asyncio
    async def test_use_stored_credentials_for_gmail_api(self, credential_service):
        """Test complete flow: OAuth → Store → Use for Gmail API."""
        # 1. OAuth flow
        params = {
            "client_id": "test_client",
            "redirect_uri": "http://localhost:8501/callback",
            "scope": "gmail.readonly",
            "state": "test_state",
            "response_type": "code"
        }

        auth_response = requests.get(
            f"{OAUTH_SERVER_URL}/oauth/authorize",
            params=params,
            allow_redirects=False
        )

        location = auth_response.headers["Location"]
        parsed = urlparse(location)
        query_params = parse_qs(parsed.query)
        auth_code = query_params["code"][0]

        token_data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": "test_client",
            "client_secret": "test_secret",
            "redirect_uri": "http://localhost:8501/callback"
        }

        token_response = requests.post(f"{OAUTH_SERVER_URL}/oauth/token", data=token_data)
        oauth_tokens = token_response.json()

        # 2. Store credentials
        cred_id = await credential_service.add_credential(
            service_type="gmail",
            account_email="test@gmail.com",
            oauth_token=oauth_tokens,
            pepper_key="test_pepper"
        )

        # 3. Retrieve credentials
        stored_cred = await credential_service.get_credential(cred_id, pepper_key="test_pepper")
        access_token = stored_cred["oauth_token"]["access_token"]

        # 4. Use access token to fetch Gmail messages
        headers = {"Authorization": f"Bearer {access_token}"}

        gmail_response = requests.get(
            f"{GMAIL_SERVER_URL}/gmail/v1/users/me/messages",
            headers=headers,
            params={"maxResults": 5}
        )

        assert gmail_response.status_code == 200
        messages = gmail_response.json()
        assert "messages" in messages
        assert len(messages["messages"]) == 5

    @pytest.mark.asyncio
    async def test_update_credential_after_token_refresh(self, credential_service):
        """Test updating credentials after refreshing token."""
        # Get initial OAuth tokens
        params = {
            "client_id": "test_client",
            "redirect_uri": "http://localhost:8501/callback",
            "scope": "gmail.readonly",
            "state": "test_state",
            "response_type": "code"
        }

        auth_response = requests.get(
            f"{OAUTH_SERVER_URL}/oauth/authorize",
            params=params,
            allow_redirects=False
        )

        location = auth_response.headers["Location"]
        parsed = urlparse(location)
        query_params = parse_qs(parsed.query)
        auth_code = query_params["code"][0]

        token_data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": "test_client",
            "client_secret": "test_secret",
            "redirect_uri": "http://localhost:8501/callback"
        }

        token_response = requests.post(f"{OAUTH_SERVER_URL}/oauth/token", data=token_data)
        oauth_tokens = token_response.json()

        # Store credentials
        cred_id = await credential_service.add_credential(
            service_type="gmail",
            account_email="test@gmail.com",
            oauth_token=oauth_tokens,
            pepper_key="test_pepper"
        )

        # Refresh token
        refresh_data = {
            "grant_type": "refresh_token",
            "refresh_token": oauth_tokens["refresh_token"],
            "client_id": "test_client",
            "client_secret": "test_secret"
        }

        refresh_response = requests.post(f"{OAUTH_SERVER_URL}/oauth/token", data=refresh_data)
        new_tokens = refresh_response.json()

        # Update stored credentials with new access token
        updated_oauth_token = oauth_tokens.copy()
        updated_oauth_token["access_token"] = new_tokens["access_token"]

        await credential_service.update_credential(
            cred_id=cred_id,
            oauth_token=updated_oauth_token,
            pepper_key="test_pepper"
        )

        # Verify update
        retrieved = await credential_service.get_credential(cred_id, pepper_key="test_pepper")
        assert retrieved["oauth_token"]["access_token"] == new_tokens["access_token"]
        # Refresh token should remain the same
        assert retrieved["oauth_token"]["refresh_token"] == oauth_tokens["refresh_token"]


class TestErrorHandling:
    """Test error handling in integration flow."""

    def test_oauth_invalid_auth_code(self):
        """Test token exchange with invalid auth code."""
        token_data = {
            "grant_type": "authorization_code",
            "code": "invalid_code",
            "client_id": "test_client",
            "client_secret": "test_secret",
            "redirect_uri": "http://localhost:8501/callback"
        }

        response = requests.post(f"{OAUTH_SERVER_URL}/oauth/token", data=token_data)

        assert response.status_code == 400
        error = response.json()
        assert "error" in error

    @pytest.mark.asyncio
    async def test_retrieve_credential_with_wrong_pepper(self, credential_service):
        """Test retrieving credential with wrong pepper key fails."""
        # Store credential
        oauth_token = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "token_type": "Bearer",
            "expires_in": 3600
        }

        cred_id = await credential_service.add_credential(
            service_type="gmail",
            account_email="test@gmail.com",
            oauth_token=oauth_token,
            pepper_key="correct_pepper"
        )

        # Try to retrieve with wrong pepper
        with pytest.raises(Exception):
            await credential_service.get_credential(cred_id, pepper_key="wrong_pepper")

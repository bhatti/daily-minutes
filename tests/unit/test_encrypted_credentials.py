"""Unit tests for encrypted credential storage in SQLiteManager."""

import pytest
import json
from src.database.sqlite_manager import SQLiteManager
from src.security.encryption_service import EncryptionService


class TestEncryptedCredentialStorage:
    """Test encrypted get/set methods in SQLiteManager."""

    @pytest.mark.asyncio
    async def test_set_encrypted_setting(self):
        """Test storing encrypted setting in kv_store."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        key = "test_encrypted_credential"
        value = "my-secret-password"
        pepper = "test-pepper-key"

        await db_mgr.set_encrypted_setting(key, value, pepper_key=pepper)

        # Verify it's stored (should be encrypted)
        raw_value = await db_mgr.get_setting(key)
        assert raw_value is not None
        assert isinstance(raw_value, dict)
        assert raw_value['is_encrypted'] is True
        assert raw_value['ciphertext'] != value.encode()

    @pytest.mark.asyncio
    async def test_get_encrypted_setting(self):
        """Test retrieving and decrypting setting from kv_store."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        key = "test_encrypted_credential"
        value = "my-secret-password"
        pepper = "test-pepper-key"

        await db_mgr.set_encrypted_setting(key, value, pepper_key=pepper)
        decrypted = await db_mgr.get_encrypted_setting(key, pepper_key=pepper)

        assert decrypted == value

    @pytest.mark.asyncio
    async def test_encrypted_oauth_token_storage(self):
        """Test storing OAuth tokens as encrypted credentials."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        oauth_token = {
            "access_token": "ya29.a0AfB_byA...",
            "refresh_token": "1//0gHZ9K...",
            "expires_at": "2025-10-27T12:00:00Z",
            "account_email": "user@gmail.com"
        }

        key = "gmail_oauth_token"
        value = json.dumps(oauth_token)
        pepper = "my-pepper"

        await db_mgr.set_encrypted_setting(key, value, pepper_key=pepper)
        decrypted = await db_mgr.get_encrypted_setting(key, pepper_key=pepper)

        assert decrypted == value
        decrypted_token = json.loads(decrypted)
        assert decrypted_token['access_token'] == oauth_token['access_token']
        assert decrypted_token['refresh_token'] == oauth_token['refresh_token']

    @pytest.mark.asyncio
    async def test_decrypt_with_wrong_pepper_fails(self):
        """Test decryption fails with wrong pepper key."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        key = "test_encrypted_credential"
        value = "my-secret-password"
        correct_pepper = "correct-pepper"
        wrong_pepper = "wrong-pepper"

        await db_mgr.set_encrypted_setting(key, value, pepper_key=correct_pepper)

        with pytest.raises(Exception):
            await db_mgr.get_encrypted_setting(key, pepper_key=wrong_pepper)

    @pytest.mark.asyncio
    async def test_encrypted_setting_uses_default_pepper(self):
        """Test encrypted setting works with default pepper."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        key = "test_credential"
        value = "my-secret"

        # Store without explicit pepper (uses default)
        await db_mgr.set_encrypted_setting(key, value)

        # Retrieve without explicit pepper (uses same default)
        decrypted = await db_mgr.get_encrypted_setting(key)

        assert decrypted == value

    @pytest.mark.asyncio
    async def test_get_nonexistent_encrypted_setting_returns_none(self):
        """Test getting non-existent encrypted setting returns None."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        result = await db_mgr.get_encrypted_setting("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_encrypted_setting(self):
        """Test updating an encrypted setting."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        key = "test_credential"
        value1 = "old-password"
        value2 = "new-password"
        pepper = "test-pepper"

        # Set initial value
        await db_mgr.set_encrypted_setting(key, value1, pepper_key=pepper)
        assert await db_mgr.get_encrypted_setting(key, pepper_key=pepper) == value1

        # Update value
        await db_mgr.set_encrypted_setting(key, value2, pepper_key=pepper)
        assert await db_mgr.get_encrypted_setting(key, pepper_key=pepper) == value2

    @pytest.mark.asyncio
    async def test_encrypted_setting_different_salts(self):
        """Test same value encrypted twice produces different ciphertexts."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        key1 = "credential1"
        key2 = "credential2"
        value = "same-password"
        pepper = "test-pepper"

        await db_mgr.set_encrypted_setting(key1, value, pepper_key=pepper)
        await db_mgr.set_encrypted_setting(key2, value, pepper_key=pepper)

        # Get raw encrypted values
        raw1 = await db_mgr.get_setting(key1)
        raw2 = await db_mgr.get_setting(key2)

        # Ciphertexts should be different (different salts)
        assert raw1['ciphertext'] != raw2['ciphertext']
        assert raw1['salt'] != raw2['salt']

        # But decrypted values should be the same
        decrypted1 = await db_mgr.get_encrypted_setting(key1, pepper_key=pepper)
        decrypted2 = await db_mgr.get_encrypted_setting(key2, pepper_key=pepper)
        assert decrypted1 == decrypted2 == value

    @pytest.mark.asyncio
    async def test_encrypted_empty_string(self):
        """Test encrypting and decrypting empty string."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        key = "empty_credential"
        value = ""
        pepper = "test-pepper"

        await db_mgr.set_encrypted_setting(key, value, pepper_key=pepper)
        decrypted = await db_mgr.get_encrypted_setting(key, pepper_key=pepper)

        assert decrypted == value

    @pytest.mark.asyncio
    async def test_encrypted_unicode_credential(self):
        """Test encrypting and decrypting unicode credentials."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        key = "unicode_credential"
        value = "密码🔐 with émojis"
        pepper = "test-pepper"

        await db_mgr.set_encrypted_setting(key, value, pepper_key=pepper)
        decrypted = await db_mgr.get_encrypted_setting(key, pepper_key=pepper)

        assert decrypted == value

    @pytest.mark.asyncio
    async def test_encrypted_long_credential(self):
        """Test encrypting and decrypting long credentials."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        key = "long_credential"
        value = "x" * 10000  # 10k characters
        pepper = "test-pepper"

        await db_mgr.set_encrypted_setting(key, value, pepper_key=pepper)
        decrypted = await db_mgr.get_encrypted_setting(key, pepper_key=pepper)

        assert decrypted == value
        assert len(decrypted) == 10000


class TestEncryptedCredentialMetadata:
    """Test encrypted credential metadata in kv_store."""

    @pytest.mark.asyncio
    async def test_encrypted_metadata_stored_correctly(self):
        """Test encrypted setting stores complete metadata."""
        db_mgr = SQLiteManager(":memory:")
        await db_mgr.initialize()

        key = "test_credential"
        value = "my-secret"
        pepper = "test-pepper"

        await db_mgr.set_encrypted_setting(key, value, pepper_key=pepper)

        # Get raw value
        raw = await db_mgr.get_setting(key)

        assert raw is not None
        assert isinstance(raw, dict)
        assert raw['is_encrypted'] is True
        assert raw['algorithm'] == 'AES-256-GCM'
        assert raw['version'] == 1
        assert 'salt' in raw
        assert 'ciphertext' in raw
        # Salt and ciphertext should be base64-encoded strings (for JSON serialization)
        assert isinstance(raw['salt'], str)
        assert isinstance(raw['ciphertext'], str)

"""Unit tests for EncryptionService - Secure credential encryption."""

import pytest
from src.security.encryption_service import EncryptionService


class TestEncryptionService:
    """Test EncryptionService initialization and basic operations."""

    def test_initialization_with_custom_pepper(self):
        """Test EncryptionService initializes with custom pepper key."""
        service = EncryptionService(pepper_key="my-secret-pepper")
        assert service is not None
        assert service.pepper_key == "my-secret-pepper"

    def test_initialization_with_default_pepper(self):
        """Test EncryptionService initializes with deterministic pepper."""
        service = EncryptionService()
        assert service is not None
        assert service.pepper_key is not None
        assert len(service.pepper_key) > 0

    def test_deterministic_pepper_is_consistent(self):
        """Test deterministic pepper generation is consistent."""
        service1 = EncryptionService()
        service2 = EncryptionService()
        assert service1.pepper_key == service2.pepper_key


class TestEncryptionDecryption:
    """Test encryption and decryption operations."""

    def test_encrypt_decrypt_roundtrip(self):
        """Test encrypt and decrypt produce original plaintext."""
        service = EncryptionService(pepper_key="test-pepper")
        plaintext = "my-secret-credential"

        encrypted = service.encrypt_credential(plaintext)
        assert encrypted['is_encrypted'] is True
        assert encrypted['ciphertext'] != plaintext
        assert encrypted['salt'] is not None
        assert encrypted['algorithm'] == 'AES-256-GCM'
        assert encrypted['version'] == 1

        decrypted = service.decrypt_credential(encrypted)
        assert decrypted == plaintext

    def test_different_salts_produce_different_ciphertext(self):
        """Test same plaintext with different salts produces different ciphertext."""
        service = EncryptionService(pepper_key="test-pepper")
        plaintext = "my-secret-credential"

        encrypted1 = service.encrypt_credential(plaintext)
        encrypted2 = service.encrypt_credential(plaintext)

        assert encrypted1['salt'] != encrypted2['salt']
        assert encrypted1['ciphertext'] != encrypted2['ciphertext']

        # Both should decrypt to same plaintext
        assert service.decrypt_credential(encrypted1) == plaintext
        assert service.decrypt_credential(encrypted2) == plaintext

    def test_wrong_pepper_fails_decryption(self):
        """Test decryption fails with wrong pepper key."""
        service1 = EncryptionService(pepper_key="correct-pepper")
        service2 = EncryptionService(pepper_key="wrong-pepper")

        plaintext = "my-secret-credential"
        encrypted = service1.encrypt_credential(plaintext)

        with pytest.raises(Exception):  # Should fail to decrypt
            service2.decrypt_credential(encrypted)

    def test_custom_salt_can_be_provided(self):
        """Test encryption accepts custom salt parameter."""
        service = EncryptionService(pepper_key="test-pepper")
        plaintext = "my-secret-credential"
        custom_salt = b'0' * 16  # 16 bytes

        encrypted = service.encrypt_credential(plaintext, salt=custom_salt)
        assert encrypted['salt'] == custom_salt

        decrypted = service.decrypt_credential(encrypted)
        assert decrypted == plaintext

    def test_encrypt_empty_string(self):
        """Test encryption handles empty string."""
        service = EncryptionService(pepper_key="test-pepper")
        plaintext = ""

        encrypted = service.encrypt_credential(plaintext)
        decrypted = service.decrypt_credential(encrypted)
        assert decrypted == plaintext

    def test_encrypt_unicode_string(self):
        """Test encryption handles unicode characters."""
        service = EncryptionService(pepper_key="test-pepper")
        plaintext = "🔐 Secret with émojis and àccents 中文"

        encrypted = service.encrypt_credential(plaintext)
        decrypted = service.decrypt_credential(encrypted)
        assert decrypted == plaintext


class TestKeyDerivation:
    """Test key derivation from salt and pepper."""

    def test_key_derivation_iterations(self):
        """Test key derivation uses sufficient iterations (100k+)."""
        service = EncryptionService(pepper_key="test-pepper")
        assert service.pbkdf2_iterations >= 100000

    def test_derived_key_length(self):
        """Test derived key is 32 bytes (256 bits for AES-256)."""
        service = EncryptionService(pepper_key="test-pepper")
        salt = b'0' * 16
        key = service._derive_encryption_key(salt)
        assert len(key) == 32  # 256 bits

    def test_same_salt_pepper_produces_same_key(self):
        """Test same salt+pepper always produces same key."""
        service = EncryptionService(pepper_key="test-pepper")
        salt = b'0' * 16

        key1 = service._derive_encryption_key(salt)
        key2 = service._derive_encryption_key(salt)
        assert key1 == key2

    def test_different_salt_produces_different_key(self):
        """Test different salts produce different keys."""
        service = EncryptionService(pepper_key="test-pepper")
        salt1 = b'0' * 16
        salt2 = b'1' * 16

        key1 = service._derive_encryption_key(salt1)
        key2 = service._derive_encryption_key(salt2)
        assert key1 != key2


class TestEncryptedMetadataStructure:
    """Test encrypted data structure and metadata."""

    def test_encrypted_metadata_structure(self):
        """Test encrypted data contains all required fields."""
        service = EncryptionService(pepper_key="test-pepper")
        plaintext = "my-secret"

        encrypted = service.encrypt_credential(plaintext)

        assert 'ciphertext' in encrypted
        assert 'salt' in encrypted
        assert 'is_encrypted' in encrypted
        assert 'algorithm' in encrypted
        assert 'version' in encrypted

        assert encrypted['is_encrypted'] is True
        assert encrypted['algorithm'] == 'AES-256-GCM'
        assert encrypted['version'] == 1
        assert isinstance(encrypted['salt'], bytes)
        assert isinstance(encrypted['ciphertext'], bytes)

    def test_salt_is_16_bytes(self):
        """Test salt is 16 bytes (128 bits)."""
        service = EncryptionService(pepper_key="test-pepper")
        plaintext = "my-secret"

        encrypted = service.encrypt_credential(plaintext)
        assert len(encrypted['salt']) == 16


class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_decrypt_invalid_structure_fails(self):
        """Test decryption fails gracefully with invalid structure."""
        service = EncryptionService(pepper_key="test-pepper")

        invalid_data = {'invalid': 'structure'}
        with pytest.raises(Exception):
            service.decrypt_credential(invalid_data)

    def test_decrypt_corrupted_ciphertext_fails(self):
        """Test decryption fails with corrupted ciphertext."""
        service = EncryptionService(pepper_key="test-pepper")
        plaintext = "my-secret"

        encrypted = service.encrypt_credential(plaintext)
        # Corrupt the ciphertext
        encrypted['ciphertext'] = b'corrupted_data'

        with pytest.raises(Exception):
            service.decrypt_credential(encrypted)

    def test_decrypt_wrong_salt_fails(self):
        """Test decryption fails with wrong salt."""
        service = EncryptionService(pepper_key="test-pepper")
        plaintext = "my-secret"

        encrypted = service.encrypt_credential(plaintext)
        # Change the salt
        encrypted['salt'] = b'0' * 16

        with pytest.raises(Exception):
            service.decrypt_credential(encrypted)


class TestDeterministicPepperGeneration:
    """Test deterministic pepper generation from ULID."""

    def test_deterministic_ulid_pepper_generation(self):
        """Test pepper is generated deterministically from system state."""
        service = EncryptionService()
        pepper = service.pepper_key

        # Pepper should be consistent across instances
        assert pepper is not None
        assert len(pepper) > 0
        assert isinstance(pepper, str)

    def test_pepper_is_derived_from_ulid(self):
        """Test pepper appears to be ULID-based (timestamp-based)."""
        service = EncryptionService()
        pepper = service.pepper_key

        # Should be alphanumeric (ULID format)
        assert pepper.replace('-', '').isalnum()

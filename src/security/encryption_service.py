"""Encryption Service - Secure credential encryption using salt + pepper approach.

Reference: https://shahbhat.medium.com/building-a-secured-family-friendly-password-manager-cb2c1db618b6
"""

import os
import hashlib
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from src.core.logging import get_logger

logger = get_logger(__name__)


class EncryptionService:
    """Service for encrypting and decrypting credentials using salt + pepper.

    Security Features:
    - AES-256-GCM encryption
    - PBKDF2 key derivation with 100k+ iterations
    - Random salt per credential
    - Pepper key (user-provided or deterministic)
    - Authenticated encryption (GCM mode)
    """

    def __init__(self, pepper_key: Optional[str] = None):
        """Initialize EncryptionService.

        Args:
            pepper_key: Optional user-provided pepper key.
                       If not provided, uses deterministic ULID-based pepper.
        """
        self.pepper_key = pepper_key or self._get_or_create_pepper()
        self.pbkdf2_iterations = 100000  # 100k iterations for security
        self.algorithm = 'AES-256-GCM'
        self.version = 1

    def encrypt_credential(
        self,
        plaintext: str,
        salt: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Encrypt credential using salt + pepper.

        Args:
            plaintext: Credential to encrypt
            salt: Optional salt (generates random if not provided)

        Returns:
            Dict containing encrypted data and metadata:
            {
                'ciphertext': bytes,
                'salt': bytes,
                'is_encrypted': True,
                'algorithm': 'AES-256-GCM',
                'version': 1
            }
        """
        try:
            # Generate random salt if not provided
            if salt is None:
                salt = os.urandom(16)  # 128 bits

            # Derive encryption key from salt + pepper
            key = self._derive_encryption_key(salt)

            # Create AESGCM cipher
            aesgcm = AESGCM(key)

            # Generate nonce (12 bytes for GCM)
            nonce = os.urandom(12)

            # Encrypt the plaintext
            plaintext_bytes = plaintext.encode('utf-8')
            ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)

            # Combine nonce + ciphertext
            combined = nonce + ciphertext

            return {
                'ciphertext': combined,
                'salt': salt,
                'is_encrypted': True,
                'algorithm': self.algorithm,
                'version': self.version
            }

        except Exception as e:
            logger.error("encryption_failed", error=str(e))
            raise

    def decrypt_credential(self, encrypted_data: Dict[str, Any]) -> str:
        """Decrypt credential using stored salt + pepper.

        Args:
            encrypted_data: Dict containing encrypted data and metadata

        Returns:
            Decrypted plaintext string

        Raises:
            Exception: If decryption fails (wrong pepper, corrupted data, etc.)
        """
        try:
            # Validate structure
            if not isinstance(encrypted_data, dict):
                raise ValueError("Invalid encrypted data structure")

            if 'ciphertext' not in encrypted_data or 'salt' not in encrypted_data:
                raise ValueError("Missing required fields in encrypted data")

            ciphertext_with_nonce = encrypted_data['ciphertext']
            salt = encrypted_data['salt']

            # Derive encryption key from salt + pepper
            key = self._derive_encryption_key(salt)

            # Create AESGCM cipher
            aesgcm = AESGCM(key)

            # Extract nonce and ciphertext
            nonce = ciphertext_with_nonce[:12]
            ciphertext = ciphertext_with_nonce[12:]

            # Decrypt
            plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
            plaintext = plaintext_bytes.decode('utf-8')

            return plaintext

        except Exception as e:
            logger.error("decryption_failed", error=str(e))
            raise

    def _derive_encryption_key(self, salt: bytes) -> bytes:
        """Derive encryption key from pepper + salt using PBKDF2.

        Args:
            salt: Salt bytes

        Returns:
            32-byte (256-bit) encryption key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits for AES-256
            salt=salt,
            iterations=self.pbkdf2_iterations,
            backend=default_backend()
        )

        pepper_bytes = self.pepper_key.encode('utf-8')
        key = kdf.derive(pepper_bytes)
        return key

    def _get_or_create_pepper(self) -> str:
        """Get or create deterministic pepper key.

        For production: User should provide a pepper key.
        For testing/default: Generate deterministic pepper based on machine state.

        Returns:
            Pepper key string
        """
        # Create a deterministic pepper based on system characteristics
        # This allows consistent encryption/decryption without storing the pepper
        # In production, users should provide their own pepper

        # Use a combination of:
        # 1. Installation directory hash (stable across runs)
        # 2. Database schema version (from code)

        try:
            # Get installation directory
            import sys
            install_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

            # Create hash of installation path + fixed seed
            hasher = hashlib.sha256()
            hasher.update(install_dir.encode('utf-8'))
            hasher.update(b'daily-minutes-pepper-v1')

            # Create ULID-style identifier (timestamp-based)
            pepper = hasher.hexdigest()[:26]  # 26 characters like ULID

            return pepper

        except Exception as e:
            logger.warning("pepper_generation_fallback", error=str(e))
            # Fallback to a fixed pepper (less secure, but consistent)
            return "daily-minutes-default-pepper-key"


# Singleton instance
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service(pepper_key: Optional[str] = None) -> EncryptionService:
    """Get or create EncryptionService instance.

    Args:
        pepper_key: Optional pepper key (uses singleton's pepper if not provided)

    Returns:
        EncryptionService instance
    """
    global _encryption_service
    if _encryption_service is None or pepper_key is not None:
        _encryption_service = EncryptionService(pepper_key=pepper_key)
    return _encryption_service

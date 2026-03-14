"""
RSA private key management.

Loads PEM-encoded RSA private keys for Snowflake key-pair authentication.
"""

import logging
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

logger = logging.getLogger(__name__)


class RSAKeyLoadError(Exception):
    """Raised when an RSA private key cannot be loaded."""


def load_private_key(
    key_path: str,
    password: bytes | None = None,
) -> rsa.RSAPrivateKey:
    """
    Load an RSA private key from a PEM file.

    Args:
        key_path: Path to the private key file (default: .rsa/rsa_key.p8).
                  Relative paths are resolved from the current working directory.
        password: Passphrase to decrypt the key (``None`` if unencrypted).

    Returns:
        RSAPrivateKey object.

    Raises:
        RSAKeyLoadError: If the key cannot be loaded or is not an RSA key.
    """
    try:
        key_file_path = Path(key_path).resolve()

        if not key_file_path.is_file():
            raise RSAKeyLoadError(f"Key file does not exist or is not a file at {key_file_path}")

        logger.debug("Loading RSA private key from %s", key_file_path)

        with open(key_file_path, "rb") as key_file:
            private_key_content = key_file.read()

        private_key = serialization.load_pem_private_key(
            private_key_content,
            password=password,
        )

        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise RSAKeyLoadError(f"Key at {key_file_path} is not an RSA private key")

        logger.debug("RSA private key loaded successfully")

        return private_key

    except RSAKeyLoadError:
        raise
    except Exception as e:
        logger.debug("Original error while loading private key: %s", e)
        raise RSAKeyLoadError(f"Failed to load private key from {key_file_path}") from e


def get_snowflake_key_bytes(
    key_path: str,
    password: bytes | None = None,
) -> bytes:
    """
    Load an RSA private key and return it as unencrypted DER/PKCS8 bytes,
    which is the format expected by the Snowflake connector.
    """
    private_key = load_private_key(key_path, password)

    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return private_key_bytes

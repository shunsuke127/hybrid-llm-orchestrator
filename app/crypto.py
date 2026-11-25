"""AES-256-CBC response encryption with a fresh random IV for every message."""

import base64
import hashlib
import os
from typing import Optional, Tuple

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

KEY_SIZE_BYTES = 32  # 256-bit key
IV_SIZE_BYTES = 16  # one AES block
BLOCK_SIZE_BITS = 128


class EncryptionNotConfiguredError(RuntimeError):
    """Raised when encryption is requested but no shared secret is configured."""


def get_aes_key() -> bytes:
    """Derive the 32-byte AES key from the AES_SHARED_SECRET environment variable.

    Fails closed: there is no built-in fallback key.
    """
    secret = os.getenv("AES_SHARED_SECRET", "")
    if not secret:
        raise EncryptionNotConfiguredError(
            "AES_SHARED_SECRET is not set; refusing to encrypt with a default key."
        )
    return hashlib.sha256(secret.encode("utf-8")).digest()


def _check_key(key: bytes) -> bytes:
    if len(key) != KEY_SIZE_BYTES:
        raise ValueError(f"AES-256 needs a {KEY_SIZE_BYTES}-byte key, got {len(key)} bytes.")
    return key


def encrypt_aes256_cbc(plaintext: str, key: Optional[bytes] = None) -> Tuple[str, str]:
    """Encrypt UTF-8 text with AES-256-CBC + PKCS7 padding and a fresh 16-byte IV.

    Returns (ciphertext_b64, iv_b64).
    """
    key_bytes = _check_key(key if key is not None else get_aes_key())
    iv = os.urandom(IV_SIZE_BYTES)
    padder = padding.PKCS7(BLOCK_SIZE_BITS).padder()
    padded = padder.update(plaintext.encode("utf-8")) + padder.finalize()
    encryptor = Cipher(algorithms.AES(key_bytes), modes.CBC(iv)).encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(ciphertext).decode("ascii"), base64.b64encode(iv).decode("ascii")


def decrypt_aes256_cbc(ciphertext_b64: str, iv_b64: str, key: Optional[bytes] = None) -> str:
    """Decrypt a Base64 payload produced by encrypt_aes256_cbc (client-side helper)."""
    key_bytes = _check_key(key if key is not None else get_aes_key())
    iv = base64.b64decode(iv_b64)
    ciphertext = base64.b64decode(ciphertext_b64)
    decryptor = Cipher(algorithms.AES(key_bytes), modes.CBC(iv)).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(BLOCK_SIZE_BITS).unpadder()
    return (unpadder.update(padded) + unpadder.finalize()).decode("utf-8")

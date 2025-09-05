from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manolo_client.client import ManoloClient


class EncryptionMixin:
    def encrypt_data(self: "ManoloClient", data: bytes) -> bytes:
        """
        Encrypts data using AES in CBC mode.

        Args:
            data (bytes): The data to encrypt.

        Returns:
            bytes: The encrypted data with IV prepended.
        """
        self.logger.debug("Encrypting data...")
        iv = get_random_bytes(16)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        pad_len = 16 - (len(data) % 16)
        padded = data + bytes([pad_len] * pad_len)
        encrypted = cipher.encrypt(padded)
        return iv + encrypted

    def decrypt_data(self: "ManoloClient", ciphertext: bytes) -> bytes:
        """
        Decrypts data using AES in CBC mode.

        Args:
            ciphertext (bytes): The ciphertext to decrypt (IV + encrypted data).

        Returns:
            bytes: The decrypted data without padding.

        Raises:
            ValueError: If padding is invalid.
        """
        self.logger.debug("Decrypting data...")
        if len(ciphertext) < 16:
            self.logger.error("Ciphertext too short to contain IV")
            raise ValueError("Invalid ciphertext length")

        iv = ciphertext[:16]
        encrypted = ciphertext[16:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        padded = cipher.decrypt(encrypted)

        pad_len = padded[-1]
        if pad_len < 1 or pad_len > 16:
            self.logger.error(f"Invalid padding length: {pad_len}")
            raise ValueError("Invalid padding length")

        if padded[-pad_len:] != bytes([pad_len] * pad_len):
            self.logger.error("Invalid padding bytes")
            raise ValueError("Invalid padding bytes")

        return padded[:-pad_len]

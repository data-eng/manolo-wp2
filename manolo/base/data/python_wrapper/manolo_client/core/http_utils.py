import json
import re
from types import SimpleNamespace
import niquests

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client import ManoloClient


class HttpUtilsMixin:
    def __init__(self: "ManoloClient"):
        self.session = niquests.Session()

        self.session.on_quic_ticket = self._save_ticket

        if self.key:
            if len(self.key) not in (16, 24, 32):
                self.logger.error(
                    "Invalid AES key size. Must be 16, 24, or 32 bytes.")
                raise ValueError(
                    "Invalid AES key size. Must be 16, 24, or 32 bytes.")

    def _save_ticket(self, ticket: bytes):
        """Store QUIC session ticket for 0-RTT resumption."""
        self._quic_tickets.append(ticket)

    def _url(self: "ManoloClient", endpoint: str):
        """
        Construct full URL from endpoint.

        Args:
            endpoint (str): API endpoint.

        Returns:
            str: Full API URL.
        """
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def request(self: "ManoloClient", endpoint: str, **kwargs):
        """
        Make a GET/POST request reusing the session.
        """
        url = self._url(endpoint)

        if self._quic_tickets and "early_data" not in kwargs:
            kwargs["early_data"] = b""

        response = self.session.get(url, **kwargs)
        return self._check_response(response)

    def _check_response(self: "ManoloClient", response):

        self.logger.debug(
            f"Response HTTP version: {response.conn_info.http_version}")

        """Raise exception if response is not 2xx."""
        if not response.ok:
            content_type = response.headers.get("Content-Type", "")
            message = response.text if "application/json" in content_type else response.content[:200]
            self.logger.error(
                f"HTTP {response.status_code} Error: {message}")
            return f"HTTP {response.status_code} Error: {message}"

        text = response.text.strip()
        if re.match(r"^\d{3}\s*-\s*", text):
            self.logger.error(f"Logical error (200 OK): {text}")
            return f"Logical error (200 OK): {text}"

        try:
            return json.loads(text, object_hook=lambda d: SimpleNamespace(**d))
        except json.JSONDecodeError:
            return text

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client import ManoloClient


class AuthMixin:
    def login(self: "ManoloClient"):
        """Authenticate with the API and start a session."""
        params = {"Username": self.username, "Password": self.password}
        self.logger.info("Logging in as %s...", self.username)
        response = self.session.post(self._url("login"), params=params)
        return self._check_response(response)

    def logout(self: "ManoloClient"):
        """Logout and close the session."""
        self.logger.info("Logging out...")
        response = self.session.get(self._url("logout"))
        return self._check_response(response)

    def __enter__(self: "ManoloClient"):
        """Enable use of the client in a context manager."""
        return self

    def __exit__(self: "ManoloClient", exc_type, exc_value, traceback):
        """Logout when exiting the context."""
        self.logout()

import datetime
import mimetypes

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manolo_client.client import ManoloClient
class MetadataMixin:
    def get_mime_type(self: "ManoloClient", file_path):
        """
        Get the MIME type of a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The MIME type of the file.
        """
        self.logger.debug(f"Getting MIME type for {file_path}")
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    def get_date_of_upload(self: "ManoloClient", file_path):
        """
        Get the date of upload for a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The date of upload in UTC ISO 8601 format.
        """
        self.logger.debug(f"Getting date of upload for {file_path}")
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

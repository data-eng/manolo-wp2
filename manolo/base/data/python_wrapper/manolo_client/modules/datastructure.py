import json

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manolo_client.client import ManoloClient


class DataStructureMixin:

    def create_datastructure(self: "ManoloClient", dsn: int, name: str, kind: str):
        """
        Create a new data structure. `dsn` can vary depending on the type of structure being created.

        Args:
            dsn (int): Data structure number or identifier.
            name (str): Name of the data structure.
            kind (str): Type of the data structure.
        """
        self.logger.debug(
            f"Creating data structure {name} with kind {kind} and dsn {dsn}")
        params = {"Dsn": dsn, "Name": name, "Kind": kind}
        response = self.session.post(
            self._url("createDataStructure"), params=params)
        return self._check_response(response)

    def delete_datastructure(self: "ManoloClient", id: str, dsn: int, name: str):
        """
        Delete a data structure.

        Args:
            id (str): ID of the structure.
            dsn (int): Data structure number.
            name (str): Name of the structure.
        """
        self.logger.debug(
            f"Deleting data structure with id={id}, dsn={dsn}, name={name}")
        params = {"Id": id, "Dsn": dsn, "Name": name}
        response = self.session.delete(
            self._url("deleteDataStructure"), params=params)
        return self._check_response(response)

    def get_datastructure(self: "ManoloClient", id: str = None, dsn: int = None, name: str = None):
        """
        Get details of a specific data structure.

        Args:
            id (str, optional): Structure ID.
            dsn (int, optional): Data structure number.
            name (str, optional): Name of the structure.
        """
        self.logger.debug(
            f"Getting data structure with id={id}, dsn={dsn}, name={name}")
        params = {}
        if id is not None:
            params["Id"] = id
        if dsn is not None:
            params["Dsn"] = dsn
        if name is not None:
            params["Name"] = name

        if not params:
            self.logger.error(
                "At least one of id, dsn, or name must be provided.")
            raise ValueError(
                "At least one of id, dsn, or name must be provided.")

        response = self.session.get(
            self._url("getDataStructure"), params=params)
        return self._check_response(response)

    def get_datastructures(self):
        """Retrieve a list of all data structures."""
        self.logger.debug("Getting all data structures")
        response = self.session.get(self._url("getDataStructures"))
        return json.loads(response.text.strip())

    def restore_datastructure(self: "ManoloClient", id: str, dsn: int, name: str):
        """
        Restore a previously deleted data structure.

        Args:
            id (str): ID of the structure.
            dsn (int): Data structure number.
            name (str): Name of the structure.
        """
        self.logger.debug(
            f"Restoring data structure with id={id}, dsn={dsn}, name={name}")
        params = {"Id": id, "Dsn": dsn, "Name": name}
        response = self.session.put(
            self._url("restoreDataStructure"), params=params)
        return self._check_response(response)

    def update_datastructure(self: "ManoloClient", id: str, dsn: int, name: str, kind: str):
        """
        Update a data structure's properties.

        Args:
            id (str): Structure ID.
            dsn (int): Data structure number.
            name (str): Name of the structure.
            kind (str): New type/kind.
        """
        self.logger.debug(
            f"Updating data structure with id={id}, dsn={dsn}, name={name}, kind={kind}")
        params = {"Id": id, "Dsn": dsn, "Name": name, "Kind": kind}
        response = self.session.put(
            self._url("updateDataStructure"), params=params)
        return self._check_response(response)

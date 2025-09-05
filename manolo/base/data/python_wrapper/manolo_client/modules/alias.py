from typing import TYPE_CHECKING

from manolo_client.enums.DefaultDatastucts import DefaultDatastucts

if TYPE_CHECKING:
    from manolo_client.client import ManoloClient


class AliasMixin:
    def upsert_alias(self: "ManoloClient", id: str, alias: str):
        """
        Create or update an alias for a given ID.

        Args:
            id (str): The target object ID.
            alias (str): Alias name to associate.
        """
        self.logger.debug(f"Upserting alias {alias} for ID {id}")
        params = {"Id": id, "Alias": alias}
        response = self.session.post(
            self._url("createUpdateAlias"), params=params)
        return self._check_response(response)

    def delete_alias(self: "ManoloClient", alias: str):
        """
        Delete an alias.

        Args:
            alias (str): Alias to delete.
        """
        self.logger.debug(f"Deleting alias {alias}")
        response = self.session.delete(
            self._url("deleteAlias"), params={"Alias": alias})
        return self._check_response(response)

    def get_alias(self: "ManoloClient", id: str):
        """
        Get alias by ID.

        Args:
            id (str): ID to lookup.
        """
        self.logger.debug(f"Getting alias for ID {id}")
        response = self.session.get(self._url("getAlias"), params={"Id": id})
        return self._check_response(response)

    def get_id(self: "ManoloClient", alias: str):
        """
        Get ID from alias.

        Args:
            alias (str): Alias to resolve.
        """
        self.logger.debug(f"Getting ID for alias {alias}")
        response = self.session.get(
            self._url("getId"), params={"Alias": alias})
        return self._check_response(response)

    def ensure_alias(self: "ManoloClient", alias: str = None, id: str = None, framework: DefaultDatastucts = DefaultDatastucts.MLFLOW):
        """
        Check if an alias or ID exists for the given framework.
        If not found, it will create the item and associate the alias.

        Args:
            alias (str, optional): Alias to check and create if missing.
            id (str, optional): ID to check and create if missing.
            framework (Frameworks): The framework context (default is mlflow).
        """
        if not alias and not id:
            msg = "Either 'alias' or 'id' must be provided."
            self.logger.error(msg)
            raise ValueError(msg)

        if alias:
            self.logger.debug(
                f"Checking if alias '{alias}' exists for framework '{framework}'")
            response = self.get_id(alias)
            if response is None or isinstance(response, str):
                new_id = self.create_item_raw(framework.value, "dummy")
                self.upsert_alias(new_id, alias)
            return response

        self.logger.debug(
            f"Checking if ID '{id}' exists for framework '{framework}'")
        response_alias = self.get_alias(id)
        response_id = self.get_id(id)

        if "logical error" in response_alias.lower() and "logical error" in response_id.lower():
            self.logger.info(
                f"ID '{id}' not found for framework '{framework}'")
            self.logger.info(
                f"Creating new item for ID '{id}' for framework '{framework}'")
            new_id = self.create_item_raw(framework.value, "dummy")

            self.logger.info(
                f"Upserting alias '{id}' for ID '{new_id}' for framework '{framework}'")
            self.upsert_alias(new_id, id)

        response = self.get_id(id)
        if response is None or "logical error" in response.lower():
            self.logger.error(
                f"ID '{id}' not found for framework '{framework}'")
            raise ValueError(
                f"ID '{id}' not found for framework '{framework}'")
        return response

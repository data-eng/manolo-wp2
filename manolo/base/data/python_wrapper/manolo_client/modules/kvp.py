from types import SimpleNamespace
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from manolo_client.enums.DefaultDatastucts import DefaultDatastucts
if TYPE_CHECKING:
    from manolo_client.client import ManoloClient


class KeyValueMixin:
    def create_kvp(self: "ManoloClient", object_id: str, key: str, value: str):
        """Create or update a key-value pair for an object."""
        self.logger.debug(f"Creating key-value pair for object {object_id}")
        params = {"Object": object_id, "Key": key, "Value": value}
        response = self.session.post(
            self._url("createUpdateKeyValue"), params=params)
        return self._check_response(response)

    def delete_kvp(self: "ManoloClient", object_id: str, key: str):
        """Delete a key-value pair from an object."""
        self.logger.debug(f"Deleting key-value pair for object {object_id}")
        params = {"Object": object_id, "Key": key}
        response = self.session.delete(
            self._url("deleteKeyValue"), params=params)
        return self._check_response(response)

    def create_update_kvp_batch(self: "ManoloClient", object_id: str, keys: list[str], values: list[str]):
        """
        Create or update multiple key-value pairs for an object in batch.

        Args:
            object_id (str): The object identifier.
            keys (list[str]): List of keys.
            values (list[str]): List of values corresponding to keys.
        """
        self.logger.debug(
            f"Creating/updating batch key-value pairs for object {object_id}")
        json_payload = {
            "Object": object_id,
            "Keys": keys,
            "Values": values
        }
        response = self.session.post(
            self._url("createUpdateKeyValueBatch"), params=json_payload)
        return self._check_response(response)

    def delete_kvp_batch(self: "ManoloClient", object_id: str, keys: list[str]):
        """
        Delete multiple key-value pairs from an object in batch.

        Args:
            object_id (str): The object identifier.
            keys (list[str]): List of keys to delete.
        """
        self.logger.debug(
            f"Deleting batch key-value pairs for object {object_id}")
        json_payload = {
            "Object": object_id,
            "Keys": keys
        }
        response = self.session.delete(
            self._url("deleteKeyValueBatch"), params=json_payload)
        return self._check_response(response)

    def get_keys(self: "ManoloClient", object_id: str):
        """Get all keys associated with an object."""
        self.logger.debug(f"Getting keys for object {object_id}")
        response = self.session.get(
            self._url("getKeys"), params={"Object": object_id})
        return self._check_response(response)

    def get_values(self: "ManoloClient", object_id: str, key: str):
        """Get values for a specific key from an object."""
        self.logger.debug(
            f"Getting values for key {key} from object {object_id}")
        response = self.session.get(self._url("getValue"), params={
                                    "Object": object_id, "Key": key})
        return self._check_response(response)

    def get_kvps(self: "ManoloClient", object_id: str):
        """Get a all key-value pairs from an object."""
        self.logger.debug(f"Getting key-value pairs for object {object_id}")
        response = self.session.get(
            self._url(f"/getKeyValuePerObject/{object_id}"))
        return self._check_response(response)

    def create_kvps_from_object(self: "ManoloClient", obj: SimpleNamespace,
                                framework: DefaultDatastucts = DefaultDatastucts.MLFLOW, num_workers: int = None):
        """
        Converts a nested object into key-value pairs and sends them in batches.
        Automatically uses the top-level key as the object_id.
        Supports parallel creation of key-value pairs.

        Args:
            obj (SimpleNamespace): The object to convert.
            framework (DefaultDatastucts): Framework for alias resolution.
            num_workers (int, optional): Number of parallel workers. Defaults to half CPU cores.
        """
        import multiprocessing

        if not isinstance(obj, SimpleNamespace):
            self.logger.error("Expected a SimpleNamespace as input")
            raise TypeError("Expected a SimpleNamespace as input")

        top_keys = vars(obj)
        if not top_keys:
            self.logger.error("Empty object provided")
            raise ValueError("Empty object provided")

        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() // 2)

        def create_for_object(top_key, nested_obj):
            object_id = top_key
            kvps = self._flatten_to_kvps(nested_obj, prefix=object_id)
            resolved_id = self.ensure_alias(id=object_id, framework=framework)
            self.logger.debug(
                f"Creating batch key-value pairs for object {resolved_id}")

            keys, values = zip(*kvps) if kvps else ([], [])
            if keys and values:
                self.create_update_kvp_batch(
                    resolved_id, list(keys), list(values))
            else:
                self.logger.warning(
                    f"No key-value pairs to send for object {resolved_id}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(create_for_object, k, v)
                       for k, v in top_keys.items()]
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    self.logger.error(f"Error in parallel execution: {exc}")

    def create_object_from_kvps(self: "ManoloClient", kvps: list[tuple[str, any]]) -> dict:
        """
        Given a list of (key, value) pairs where keys are dot-separated paths,
        reconstruct a nested dictionary representing the original object.

        Args:
            kvps (list[tuple[str, any]]): A list of (key, value) pairs.

        Returns:
            dict: A nested dictionary representing the original object.
        """
        result = {}
        self.logger.debug("Converting key-value pairs to object")

        for full_key, value in kvps:
            parts = full_key.split('.')
            current = result

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    self.logger.warning(
                        f"Overwriting key {part} which is not a dict")
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return result

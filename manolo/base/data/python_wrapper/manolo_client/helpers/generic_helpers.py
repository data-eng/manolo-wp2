import asyncio
import mimetypes
import os
import json
import types
from types import SimpleNamespace
from typing import TYPE_CHECKING, Union, List, Any
from concurrent.futures import ThreadPoolExecutor
import imghdr

from ..enums.DefaultDatastucts import DefaultDatastucts

if TYPE_CHECKING:
    from client import ManoloClient


class GenericHelpers:
    def check_id_alias(self: "ManoloClient", string: str = None) -> list[str]:
        """
        Extracts parts from a mlflow:// URI based on the code.

        - If code == 29: return a list with everything after the last slash.
        - Else: return all parts split by '/' (excluding the 'mlflow://' scheme).
        """
        self.logger.debug(f"Checking ID or alias: {string}")
        if len(string) == 29:
            response = self.ensure_alias(
                id=string, framework=DefaultDatastucts.MLFLOW)
            clean = response.replace("mlflow://", "")
            self.logger.debug(f"Resolved ID or alias: {clean}")
            return [clean.rsplit('/', 1)[-1]]
        else:
            self.ensure_alias(alias=string, framework=DefaultDatastucts.MLFLOW)
            self.logger.debug(f"Resolved ID or alias: {string}")
            return string.split('/')

    def load_json_as_object(self: "ManoloClient", json_or_path: str) -> Any:
        """
        Loads a JSON file or string as a Python object.

        Args:
            json_or_path (str): The JSON data or file path.

        Returns:
            object: The loaded Python object.
        """
        self.logger.debug(f"Loading JSON: {json_or_path}")
        if not isinstance(json_or_path, str):
            raise TypeError("Expected input to be a str")

        stripped = json_or_path.strip()
        if not stripped:
            raise ValueError("Empty string provided")

        try:
            return json.loads(stripped, object_hook=lambda d: SimpleNamespace(**d))
        except json.JSONDecodeError:
            self.logger.debug("Not raw JSON, trying file path")

        path = os.path.expanduser(stripped)
        if not os.path.isfile(path):
            raise ValueError("Input is not valid JSON or a valid file path")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        except FileNotFoundError:
            raise ValueError(f"File not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file '{path}': {e.msg}") from e

    def _flatten_to_kvps(self: "ManoloClient", obj: Any, prefix: str = "", num_workers: int = None) -> List[tuple]:
        """
        Flattens a nested object into a list of key-value pairs.

        Args:
            obj (Any): The object to flatten.
            prefix (str, optional): The prefix to use for the keys. Defaults to "".
            num_workers (int, optional): The number of workers to use for parallel processing. Defaults to None.

        Returns:
            List[tuple]: A list of key-value pairs.
        """
        kvps = []

        if isinstance(obj, SimpleNamespace):
            obj = vars(obj)

        num_workers = num_workers or max(1, os.cpu_count() // 2)

        if isinstance(obj, dict):
            if len(obj) > 20:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    results = executor.map(
                        lambda kv: self._flatten_to_kvps(
                            kv[1], f"{prefix}.{kv[0]}" if prefix else kv[0], num_workers),
                        obj.items()
                    )
                    for res in results:
                        kvps.extend(res)
            else:
                for k, v in obj.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    kvps.extend(self._flatten_to_kvps(
                        v, new_prefix, num_workers))

        elif isinstance(obj, list):
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = executor.map(
                    lambda idx_val: self._flatten_to_kvps(
                        idx_val[1], f"{prefix}.{idx_val[0]}", num_workers),
                    enumerate(obj)
                )
                for res in results:
                    kvps.extend(res)

        elif obj is not None:
            kvps.append((prefix, str(obj)))

        return kvps

    def parse_value(self: "ManoloClient", value: Any) -> Any:
        """Convert string to appropriate Python types."""
        if isinstance(value, str):
            lower = value.lower()
            if lower == "true":
                return True
            if lower == "false":
                return False
            if lower in ("null", "none"):
                return None
            try:
                return int(value) if '.' not in value else float(value)
            except ValueError:
                return value
        return value

    def parse_key(self: "ManoloClient", key: str) -> List[Union[str, int]]:
        """Parse a key into a list of strings and integers."""
        parts = []
        for part in key.split('.'):
            parts.append(int(part) if part.isdigit() else part)
        return parts

    def set_nested_value(self: "ManoloClient", data: Any, keys: List[Union[str, int]], value: Any):
        """Set a value in a nested dictionary or list."""
        prev_key = None
        d = data
        for i, key in enumerate(keys):
            is_last = (i == len(keys) - 1)
            if isinstance(key, int):
                if not isinstance(d, list):
                    d_parent = d
                    d = []
                    d_parent[prev_key] = d
                while len(d) <= key:
                    d.append(None)
                if is_last:
                    d[key] = value
                else:
                    if d[key] is None:
                        d[key] = {} if isinstance(keys[i + 1], str) else []
                    d = d[key]
            else:
                if is_last:
                    d[key] = value
                else:
                    if key not in d or not isinstance(d[key], (dict, list)):
                        d[key] = {} if isinstance(keys[i + 1], str) else []
                    d = d[key]
            prev_key = key

    def unflatten_kvps(self: "ManoloClient", kvp_list: List[dict]) -> dict:
        """Unflatten a list of key-value pairs into a nested dictionary."""
        result = {}
        for item in kvp_list:
            keys = self.parse_key(item["Key"])
            value = self.parse_value(item["Value"])
            self.set_nested_value(result, keys, value)
        return result

    def convert_to_dict(self: "ManoloClient", obj: Any, num_workers: int = None) -> Any:
        """Converts a nested object to a dictionary."""
        if isinstance(obj, types.SimpleNamespace):
            return {k: self.convert_to_dict(v, num_workers) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            if len(obj) > 20:
                num_workers = num_workers or max(1, os.cpu_count() // 2)
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    results = executor.map(
                        lambda kv: (kv[0], self.convert_to_dict(
                            kv[1], num_workers)),
                        obj.items()
                    )
                    return dict(results)
            else:
                return {k: self.convert_to_dict(v, num_workers) for k, v in obj.items()}
        elif isinstance(obj, list):
            num_workers = num_workers or max(1, os.cpu_count() // 2)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                return list(executor.map(lambda x: self.convert_to_dict(x, num_workers), obj))
        else:
            return obj

    def save_decrypted_item_with_metadata(
        self: "ManoloClient",
        item_id: str,
        raw_data: bytes,
        output_dir: str
    ):
        """
        Saves a decrypted item to disk with correct folder structure and extension
        based on metadata.
        """
        ext = ".bin"
        rel_path_parts = []

        try:
            kvps = self.get_kvps(item_id)

            mime_kvp = next(
                (kvp for kvp in kvps if getattr(kvp, "Key", "") == "mime_type"),
                None
            )
            if mime_kvp:
                mime_type = mime_kvp.Value
                guessed_ext = mimetypes.guess_extension(mime_type)
                ext = guessed_ext if guessed_ext else ext
            else:
                guessed = imghdr.what(None, h=raw_data)
                ext = f".{guessed}" if guessed else ext

            folder_parts = {
                int(kvp.Key.split("_")[1]): kvp.Value
                for kvp in kvps
                if kvp.Key.startswith("folder_") and kvp.Key.split("_")[1].isdigit()
            }
            rel_path_parts = [folder_parts[i]
                              for i in sorted(folder_parts.keys())]

        except Exception as meta_err:
            self.logger.warning(
                f"Could not parse metadata for {item_id}: {meta_err}"
            )

        rel_dir = os.path.join(*rel_path_parts) if rel_path_parts else ""
        full_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(full_dir, exist_ok=True)

        file_path = os.path.join(full_dir, f"{item_id}{ext}")
        with open(file_path, "wb") as f:
            f.write(raw_data)

        self.logger.info(f"Decrypted item saved to {file_path}")

    async def _ensure_connection(self: "ManoloClient"):
        if self._signalr_connected:
            return

        connection_ready_event = asyncio.Event()

        def on_open():
            self.logger.info("SignalR connection opened")
            self._signalr_connected = True
            connection_ready_event.set()

        def on_close():
            self.logger.warning("SignalR connection closed")
            self._signalr_connected = False

        self.hub_connection.on_open(on_open)
        self.hub_connection.on_close(on_close)

        self.hub_connection.start()

        try:
            await asyncio.wait_for(connection_ready_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            raise Exception("SignalR connection timed out while starting")

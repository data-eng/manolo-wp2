from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
import shutil
from types import SimpleNamespace
from typing import Optional, Union

from typing import TYPE_CHECKING

from manolo_client.enums.ManifestKeys import ManifestKeys

if TYPE_CHECKING:
    from manolo_client.client import ManoloClient


class FileUtilsMixin:
    def load_manifest(self: "ManoloClient", manifest_path: str) -> SimpleNamespace:
        """
        Loads a manifest from a file.

        Args:
            manifest_path (str): Path to the manifest file.

        Returns:
            SimpleNamespace: Loaded manifest as an object with attribute access.
        """
        self.logger.debug(f"Loading manifest from {manifest_path}...")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    return json.load(f, object_hook=lambda d: SimpleNamespace(**d))
            except json.JSONDecodeError:
                self.logger.error(
                    f"Failed to load manifest from {manifest_path} (invalid JSON), creating backup and returning empty manifest"
                )
                backup_path = manifest_path + ".bak"
                try:
                    shutil.copy2(manifest_path, backup_path)
                    self.logger.debug(f"Backup created at {backup_path}")
                except Exception as e:
                    self.logger.error(
                        f"Failed to create backup of manifest: {e} - returning empty manifest")
                return SimpleNamespace(items=[])
        else:
            self.logger.debug(
                f"No manifest found at {manifest_path} - returning empty manifest")
            return SimpleNamespace(items=[])

    def save_manifest(self: "ManoloClient", manifest: SimpleNamespace, manifest_path: str = "manifest.json", num_workers: Optional[int] = None):
        """
        Saves a manifest to a file, processing items in parallel.

        Args:
            manifest (SimpleNamespace): Manifest to save.
            manifest_path (str): Path to the manifest file.
            num_workers (int, optional): Number of parallel workers to use.
        """
        self.logger.debug(f"Saving manifest to {manifest_path}")
        manifest_dict = vars(manifest)

        items_key = ManifestKeys.ITEMS.value
        items = manifest_dict.get(items_key)

        if isinstance(items, list):
            if num_workers is None:
                num_workers = max(1, multiprocessing.cpu_count() // 2)

            def convert(item):
                return vars(item) if isinstance(item, SimpleNamespace) else item

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                manifest_dict[items_key] = list(executor.map(convert, items))

        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest_dict, f, indent=2)

    def get_item_from_manifest(self: "ManoloClient", manifest: Union[SimpleNamespace, dict], file_path: str) -> Optional[dict]:
        """
        Gets an item from a manifest by file path.

        Args:
            manifest (SimpleNamespace or dict): Manifest to search.
            file_path (str): File path to search for.

        Returns:
            dict: Item found in the manifest, or None if not found.
        """
        self.logger.debug(f"Getting item from manifest for {file_path}")
        items = getattr(manifest, ManifestKeys.ITEMS.value, None)
        if items is None and isinstance(manifest, dict):
            items = manifest.get(ManifestKeys.ITEMS.value, [])
        if items is None:
            return None

        for item in items:
            if isinstance(item, SimpleNamespace):
                if getattr(item, "file_path", None) == file_path:
                    return vars(item)
            elif isinstance(item, dict):
                if item.get("file_path") == file_path:
                    return item
        return None

    def export_json_from_object(self: "ManoloClient", obj: str, file_path: Optional[str] = None):
        """
        Exports an object to a JSON file by retrieving key-value pairs and converting them into a nested JSON structure.

        Args:
            obj (str): Object ID to export.
            file_path (str, optional): Destination path or filename.
        """
        if file_path is None:
            file_path = os.path.join("exports", f"{obj}.json")

        file_path = os.path.expanduser(file_path)

        if os.path.isdir(file_path) or file_path.endswith(os.path.sep):
            file_path = os.path.join(file_path, f"{obj}.json")

        if not os.path.isabs(file_path) and os.path.dirname(file_path) == "":
            file_path = os.path.join("exports", file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.logger.debug(f"Exporting JSON from object {obj} to {file_path}")
        kvps = self.get_kvps(obj)
        kvps_dict = self.convert_to_dict(kvps)
        nested_json = self.unflatten_kvps(kvp_list=kvps_dict)

        with open(file_path, "w") as f:
            json.dump(nested_json, f, indent=2)

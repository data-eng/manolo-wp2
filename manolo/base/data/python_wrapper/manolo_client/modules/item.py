import base64
import hashlib
import json
import mimetypes
import os
import imghdr
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading
from typing import TYPE_CHECKING, List, Optional
import asyncio

from enums.ManifestKeys import ManifestKeys
if TYPE_CHECKING:
    from client import ManoloClient


class ItemMixin:
    def create_item_file(self: "ManoloClient", dsn: int, item_path: str):
        """
        Upload an item from a file using multipart/form-data.

        Args:
            dsn (int): Data structure number.
            item_path (str): Local path to the file.
        """
        self.logger.debug(f"Uploading item from {item_path} to DSN {dsn}")
        mime_type, _ = mimetypes.guess_type(item_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(item_path, 'rb') as f:
            files = {'DataFile': (os.path.basename(item_path), f, mime_type)}
            response = self.session.post(
                self._url("createItem"), files=files, data={'Dsn': str(dsn)})
            return self._check_response(response)

    def create_item_raw(self: "ManoloClient", dsn: int, data: str):
        """
        Upload raw item data in JSON body.

        Args:
            dsn (int): Data structure number.
            data (str): Raw string content (e.g. base64 encoded).
        """
        self.logger.debug(f"Uploading raw item data to DSN {dsn}")
        json_payload = {"Dsn": dsn, "Data": data}
        response = self.session.post(
            self._url("createItem"), json=json_payload)
        return self._check_response(response)

    def delete_item(self: "ManoloClient", dsn: int, id: str):
        """
        Delete an item.

        Args:
            dsn (int): Data structure number.
            id (str): ID of the item.
        """
        self.logger.debug(f"Deleting item with id={id} from DSN {dsn}")
        params = {"Dsn": dsn, "Id": id}
        response = self.session.delete(self._url("deleteItem"), params=params)
        return self._check_response(response)

    def download_item_data(self: "ManoloClient", dsn: int, id: str, output_path: str):
        """
        Download item data and save to a file.
        """
        self.logger.debug(
            f"Downloading item with id={id} from DSN {dsn} to {output_path}")
        params = {"Dsn": dsn, "Id": id}
        response = self.session.get(
            self._url("downloadItemData"), params=params)

        content_type = response.headers.get("Content-Type", "")
        if "application/octet-stream" not in content_type:
            raise Exception(
                f"Unexpected content type: {content_type}, response: {response.text[:200]}")

        with open(output_path, 'wb') as f:
            f.write(response.content)
        return output_path

    def get_item(self: "ManoloClient", dsn: int, id: str):
        """
        Get item metadata.

        Args:
            dsn (int): Data structure number.
            id (str): Item ID.
        """
        self.logger.debug(f"Getting item with id={id} from DSN {dsn}")
        params = {"Dsn": dsn, "Id": id}
        response = self.session.get(self._url("getItem"), params=params)
        return self._check_response(response)

    def get_item_data(self: "ManoloClient", dsn: int, id: str):
        """
        Get raw data for an item.

        Args:
            dsn (int): Data structure number.
            id (str): Item ID.
        """
        self.logger.debug(
            f"Getting raw data for item with id={id} from DSN {dsn}")
        params = {"Dsn": dsn, "Id": id}
        response = self.session.get(self._url("getItemData"), params=params)
        return self._check_response(response)

    def get_items(self: "ManoloClient", dsn: int):
        """
        Retrieve all items for a given DSN.

        Args:
            dsn (int): Data structure number.
        """
        self.logger.debug(f"Getting items for DSN {dsn}")
        response = self.session.get(self._url("getItems"), params={"Dsn": dsn})
        return json.loads(response.text.strip())

    def restore_item(self: "ManoloClient", dsn: int, id: str):
        """
        Restore a deleted item.

        Args:
            dsn (int): Data structure number.
            id (str): Item ID.
        """
        self.logger.debug(f"Restoring item with id={id} from DSN {dsn}")
        params = {"Dsn": dsn, "Id": id}
        response = self.session.put(self._url("restoreItem"), params=params)
        return self._check_response(response)

    def update_item_file(self: "ManoloClient", dsn: int, id: str, image_path: str):
        """
        Update an item using a new file.

        Args:
            dsn (int): Data structure number.
            id (str): Item ID.
            image_path (str): Path to the new file.
        """
        self.logger.debug(
            f"Updating item with id={id} from DSN {dsn} with {image_path}")
        with open(image_path, 'rb') as f:
            files = {'DataFile': (image_path, f, 'image/png')}
            params = {'Dsn': dsn, 'Id': id,
                      'Data': f"Uploaded from: {image_path}"}
            response = self.session.put(
                self._url("updateItem"), params=params, files=files)
            return self._check_response(response)

    def update_item_raw(self: "ManoloClient", dsn: int, id: str, data: str):
        """
        Update an item using raw data.

        Args:
            dsn (int): Data structure number.
            id (str): Item ID.
            data (str): Raw string data.
        """
        self.logger.debug(f"Updating item with id={id} from DSN {dsn}")
        params = {"Dsn": dsn, "Id": id, "Data": data}
        response = self.session.put(self._url("updateItem"), params=params)
        return self._check_response(response)

    def create_item_batch_files(self: "ManoloClient", dsn: int, item_paths: list[str]):
        """
        Upload multiple items from files using multipart/form-data batch.

        Args:
            dsn (int): Data structure number.
            item_paths (list[str]): List of local file paths to upload.
        """
        self.logger.debug(f"Uploading batch items from files to DSN {dsn}")

        files = {}
        for i, path in enumerate(item_paths):
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type is None:
                mime_type = "application/octet-stream"
            with open(path, 'rb') as f:
                files[f'DataFile[{i}]'] = (
                    os.path.basename(path), f.read(), mime_type)

        data = {'Dsn': str(dsn)}
        response = self.session.post(
            self._url("createItemBatch"), files=files, json=data)
        return self._check_response(response)

    def create_item_batch_raw(self: "ManoloClient", dsn: int, data_list: list[str], batch_size: int = 100, max_workers: int = None):
        """
        Upload multiple raw items in JSON body batches, with optional parallelism.

        Args:
            dsn (int): Data structure number.
            data_list (list[str]): List of raw string contents.
            batch_size (int): Number of items per request.
            max_workers (int, optional): Thread pool size (default: half CPU cores).
        """
        self.logger.debug(
            f"Uploading {len(data_list)} items in batches of {batch_size} to DSN {dsn}"
        )

        if not max_workers:
            max_workers = max(1, multiprocessing.cpu_count() // 2)

        def upload_batch(start_idx: int, batch: list[str]):
            self.logger.debug(f"Uploading batch starting at index {start_idx}")
            try:
                json_payload = {"Dsn": dsn, "Data": batch}
                response = self.session.post(
                    self._url("createItemBatch"), json=json_payload)
                return self._check_response(response)
            except Exception as e:
                self.logger.error(f"Batch starting at {start_idx} failed: {e}")
                raise

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                futures.append(executor.submit(upload_batch, i, batch))

            results = []
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def delete_item_batch(self: "ManoloClient", dsn: int, ids: list[str]):
        """
        Delete multiple items by their IDs in batch.

        Args:
            dsn (int): Data structure number.
            ids (list[str]): List of item IDs to delete.
        """
        self.logger.debug(
            f"Deleting batch items with ids={ids} from DSN {dsn}")
        json_payload = {"Dsn": dsn, "Ids": ids}
        response = self.session.delete(
            self._url("deleteItemBatch"), json=json_payload)
        return self._check_response(response)

    def upload_items_with_encryption(self: "ManoloClient", manifest,
                                     allowed_extensions: set[str], folder: str, dsn: int,
                                     metadata: list = [], allow_mime_type_metadata: bool = True, allow_upload_date_metadata: bool = True,
                                     max_workers: int = None):
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() // 2)

        lock = threading.Lock()
        path_to_item = {
            getattr(i, ManifestKeys.FILE_PATH.value, None): i for i in manifest.items}
        hash_to_item = {
            getattr(i, ManifestKeys.FILE_SHA256.value, None): i for i in manifest.items}

        all_files = []
        for dirpath, _, filenames in os.walk(folder):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in allowed_extensions:
                    all_files.append(os.path.join(dirpath, filename))
        self.logger.info(
            f"Found {len(all_files)} files to consider for upload")

        def prepare_file_data(item_path):
            try:
                if not os.path.isfile(item_path):
                    self.logger.warning(f"File not found: {item_path}")
                    return None

                with open(item_path, "rb") as f:
                    raw_data = f.read()
                file_hash = hashlib.sha256(raw_data).hexdigest()
                rel_path = os.path.relpath(item_path, folder)

                with lock:
                    existing_same_path = path_to_item.get(rel_path)
                    if existing_same_path and getattr(existing_same_path, ManifestKeys.FILE_SHA256.value) == file_hash:
                        self.logger.debug(
                            f"No change for {rel_path}, skipping")
                        return None
                    if existing_same_path and getattr(existing_same_path, ManifestKeys.FILE_SHA256.value) != file_hash:
                        self.logger.debug(
                            f"Hash changed for {rel_path}, removing old")
                        self.delete_item(dsn, existing_same_path.object_id)
                        manifest.items.remove(existing_same_path)
                        del path_to_item[rel_path]
                        if file_hash in hash_to_item:
                            del hash_to_item[file_hash]

                    existing_same_hash = hash_to_item.get(file_hash)
                    if existing_same_hash and getattr(existing_same_hash, ManifestKeys.FILE_PATH.value) != rel_path:
                        self.logger.debug(
                            f"Duplicate content found at {getattr(existing_same_hash, ManifestKeys.FILE_PATH.value)}, removing duplicate")
                        self.delete_item(dsn, existing_same_hash.object_id)
                        manifest.items.remove(existing_same_hash)
                        del path_to_item[getattr(
                            existing_same_hash, ManifestKeys.FILE_PATH.value)]
                        del hash_to_item[file_hash]

                item_metadata = list(metadata)
                if allow_mime_type_metadata:
                    mime_type = self.get_mime_type(item_path)
                    item_metadata.append(
                        {"key": "mime_type", "value": mime_type})
                if allow_upload_date_metadata:
                    upload_date = self.get_date_of_upload(item_path)
                    item_metadata.append(
                        {"key": "upload_date", "value": upload_date})

                encrypted_data = self.encrypt_data(raw_data)
                encrypted_b64 = base64.b64encode(encrypted_data).decode()

                return {
                    "rel_path": rel_path,
                    "file_hash": file_hash,
                    "encrypted_b64": encrypted_b64,
                    "metadata": item_metadata
                }

            except Exception as e:
                self.logger.error(f"Failed to prepare {item_path}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            prepared_files = list(
                filter(None, executor.map(prepare_file_data, all_files)))

        if not prepared_files:
            self.logger.info("No new or changed files to upload.")
            return

        encrypted_datas = [file["encrypted_b64"] for file in prepared_files]
        object_ids = self.create_item_batch_raw(
            dsn, encrypted_datas, batch_size=100, max_workers=max_workers)

        if isinstance(object_ids, str):
            object_ids = [object_ids]
        elif isinstance(object_ids, list) and all(isinstance(i, list) for i in object_ids):
            object_ids = [item for sublist in object_ids for item in sublist]

        batch_keys = []
        batch_values = []
        batch_object_ids = []

        for obj_id, file in zip(object_ids, prepared_files):
            parts = file["rel_path"].split(os.sep)
            keys = [f"folder_{i}" for i in range(len(parts) - 1)]
            values = parts[:-1]

            for entry in file["metadata"]:
                keys.append(entry["key"])
                values.append(entry["value"])

            batch_object_ids.extend([obj_id] * len(keys))
            batch_keys.extend(keys)
            batch_values.extend(values)

        self.create_update_kvp_batch(
            batch_object_ids, batch_keys, batch_values)

        with lock:
            for obj_id, file in zip(object_ids, prepared_files):
                item_ns = SimpleNamespace(
                    object_id=obj_id,
                    file_path=file["rel_path"],
                    file_sha256=file["file_hash"]
                )
                manifest.items.append(item_ns)
                path_to_item[file["rel_path"]] = item_ns
                hash_to_item[file["file_hash"]] = item_ns

        self.logger.info(f"Uploaded {len(prepared_files)} files successfully.")

    def upload_encrypted_item(self: "ManoloClient", manifest, item_path: str, dsn: int, metadata: list, root_folder: str = None,):
        """
        Upload a single file with encryption, avoiding duplicates by comparing hashes.

        Args:
            manifest (dict): The manifest dictionary tracking uploaded items.
            item_path (str): Full path to the file to upload.
            dsn (int): Data structure number.
            metadata (list): List of metadata dicts (e.g. [{"key": "...", "value": "..."}]).
            root_folder (str, optional): Used to compute relative path for folder tags.
        """
        try:
            if not os.path.isfile(item_path):
                self.logger.warning(f"File not found: {item_path}")
                return

            with open(item_path, "rb") as f:
                raw_data = f.read()

            file_hash = hashlib.sha256(raw_data).hexdigest()

            rel_path = os.path.relpath(
                item_path, root_folder) if root_folder else item_path

            self.logger.debug(f"Checking manifest for {rel_path}")

            existing_same_path = next(
                (item for item in manifest.items
                 if getattr(item, ManifestKeys.FILE_PATH.value, None) == rel_path),
                None
            )
            if existing_same_path:
                if getattr(existing_same_path, ManifestKeys.FILE_SHA256.value, None) == file_hash:
                    self.logger.debug(
                        f"No change detected for {rel_path}. Skipping.")
                    return
                self.logger.debug(
                    f"Hash changed for {rel_path}. Replacing item.")
                self.delete_item(dsn, existing_same_path.object_id)
                manifest.items.remove(existing_same_path)

            existing_same_hash = next(
                (item for item in manifest.items
                 if getattr(item, ManifestKeys.FILE_SHA256.value, None) == file_hash and getattr(item, ManifestKeys.FILE_PATH.value, None) != rel_path),
                None
            )
            if existing_same_hash:
                self.logger.debug(
                    f"Duplicate content found at {existing_same_hash.file_path}. Replacing.")
                self.delete_item(dsn, existing_same_hash.object_id)
                manifest.items.remove(existing_same_hash)

            encrypted_data = self.encrypt_data(raw_data)
            encrypted_b64 = base64.b64encode(encrypted_data).decode()
            object_id = self.create_item_raw(dsn, encrypted_b64)

            parts = rel_path.split(os.sep)
            for i, part in enumerate(parts[:-1]):
                self.create_kvp(object_id, f"folder_{i}", part)

            for entry in metadata:
                self.create_kvp(object_id, entry["key"], entry["value"])

            manifest.items.append(SimpleNamespace(
                object_id=object_id,
                file_path=rel_path,
                file_sha256=file_hash
            ))

            self.logger.info(
                f"Uploaded {rel_path} -> Object ID: {object_id}")

        except Exception as e:
            self.logger.error(f"Failed to upload {item_path}: {e}")

    def download_and_decrypt_item(self: "ManoloClient", item_id: str, output_dir: str, dsn: int):
        """
        Downloads, decrypts, and writes an item to disk using folder structure and MIME metadata.

        Args:
            item_id (str): Item ID.
            output_dir (str): Root output directory.
            dsn (int): Data structure number.
        """
        self.logger.debug(f"Downloading and saving item with id={item_id}")

        try:
            raw_data = self.get_and_decrypt_item(item_id, dsn)

            ext = ".bin"
            rel_path_parts = []

            try:
                kvps = self.get_kvps(item_id)

                mime_kvp = next((kvp for kvp in kvps if getattr(
                    kvp, "Key", "") == "mime_type"), None)
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
                    f"Could not parse metadata for {item_id}: {meta_err}")

            rel_dir = os.path.join(*rel_path_parts) if rel_path_parts else ""
            full_dir = os.path.join(output_dir, rel_dir)
            os.makedirs(full_dir, exist_ok=True)

            file_path = os.path.join(full_dir, f"{item_id}{ext}")
            with open(file_path, "wb") as f:
                f.write(raw_data)

            self.logger.debug(f"Decrypted item saved to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to download/decrypt {item_id}: {e}")

    def get_and_decrypt_item(self: "ManoloClient", item_id: str, dsn: int) -> bytes:
        """
        Fetches and decrypts an item's raw data.

        Args:
            item_id (str): Item ID.
            dsn (int): Data structure number.

        Returns:
            bytes: Decrypted raw binary data.
        """
        self.logger.debug(f"Fetching and decrypting item with id={item_id}")

        encrypted_b64 = self.get_item_data(dsn, item_id)
        encrypted_data = base64.b64decode(encrypted_b64)
        raw_data = self.decrypt_data(encrypted_data)

        self.logger.debug(f"Decrypted item with id={item_id}")

        return raw_data

    # Async methods

    async def download_items_via_signalr(self: "ManoloClient", dsn, output_dir, item_ids=None):

        self._download_events = {item_id: asyncio.Event()
                                 for item_id in item_ids}

        loop = asyncio.get_running_loop()
        await self._ensure_connection()

        self._pending_items = asyncio.Queue()
        if item_ids:
            for item_id in item_ids:
                await self._pending_items.put(item_id)

        def on_item_received(item_data_array):
            try:
                item_id = self._pending_items.get_nowait()
                self.logger.debug(f"Dequeued item_id: {item_id}")
            except asyncio.QueueEmpty:
                self.logger.error(
                    "Received data but no pending item ID available")
                return

            try:
                if all(isinstance(chunk, str) for chunk in item_data_array):
                    full_b64 = "".join(item_data_array)
                    encrypted_bytes = base64.b64decode(full_b64, validate=True)
                else:
                    raise TypeError(
                        f"Unexpected chunk types: {[type(c) for c in item_data_array]}")
            except Exception as e:
                self.logger.error(
                    f"Failed to prepare encrypted bytes for {item_id}: {e}")
                return

            try:
                raw_data = self.decrypt_data(encrypted_bytes)
                self.save_decrypted_item_with_metadata(
                    item_id, raw_data, output_dir)
                self.logger.debug(f"Decrypted and saved item {item_id}")
                if item_id in self._download_events:
                    self.logger.debug(
                        f"Scheduling download event set for {item_id} on the event loop")
                    loop.call_soon_threadsafe(
                        self._download_events[item_id].set)
                    remaining = [
                        eid for eid, ev in self._download_events.items() if not ev.is_set()]
                    self.logger.debug(
                        f"Remaining unset download events: {remaining}")
                    self.logger.debug(
                        f"Download event scheduled for {item_id}")
                else:
                    self.logger.error(
                        f"Item ID {item_id} not in _download_events")

            except Exception as e:
                self.logger.error(
                    f"Failed to decrypt and save item {item_id}: {e}")
                if item_id in self._download_events:
                    self.logger.debug(
                        f"Setting download event for {item_id} after failure")
                    self._download_events[item_id].set()

        self.hub_connection.on("SignalRISuccess", on_item_received)
        self.hub_connection.on(
            "SignalRError", lambda msg: self.logger.error(
                f"SignalR error: {msg}")
        )

        if item_ids:
            for item_id in item_ids:
                self.hub_connection.send(
                    "RequestItemData", [str(item_id), dsn])

        if self._download_events:
            await asyncio.gather(*[event.wait() for event in self._download_events.values()])
            self.logger.debug(
                "All download events awaited successfully (parallel)")

        self.hub_connection.stop()

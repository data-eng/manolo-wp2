import base64
import logging
import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))
from manolo_client.client import ManoloClient




IMAGE_ROOT = "manolo/base/data/python_wrapper/manolo_client/demo/images"
DSN = 10001  #
key_b64 = "kVnA8+8nFbGHWJ9fAptF6Bp22E5h8lfUPJ1/jjvgL0c="

MANIFEST_PATH = "manolo/base/data/python_wrapper/manolo_client/demo/manifest.json"
DOWNLOAD_DIR = "manolo/base/data/python_wrapper/manolo_client/demo/output"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif",
                    ".tiff", ".webp"}
if __name__ == "__main__":

    client = ManoloClient(
        base_url="http://localhost:5001",  # Replace with your Manolo server / port
        username="manolo",
        password="manolo",
        key=base64.b64decode(key_b64),
        logging_level=logging.DEBUG,
        log_dir="manolo/base/data/python_wrapper/manolo_client/demo/logs",
        log_to_file=True,
        log_to_console=False
    )

    client.login()

    manifest = client.load_manifest(manifest_path=MANIFEST_PATH)

    dataset = client.get_datastructure(dsn=DSN)

    if dataset is None or isinstance(dataset, str):
        client.create_datastructure(DSN, "TestPhotos", "image")

    client.upload_items_with_encryption(
        manifest=manifest,
        allowed_extensions=IMAGE_EXTENSIONS,
        folder=IMAGE_ROOT,
        dsn=DSN,
        allow_mime_type_metadata=True,
        allow_upload_date_metadata=True
    )

    # Save the manifest after the upload
    client.save_manifest(manifest, manifest_path=MANIFEST_PATH)

    items = client.get_items(dsn=DSN)  # Get the items

    for item_id in items:
        print(f"[INFO] Downloading item {item_id}")
        client.download_and_decrypt_item(
            item_id=item_id, output_dir=DOWNLOAD_DIR, dsn=DSN)

    client.logout()

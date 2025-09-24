import logging
import sys
import os

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))
from manolo_client.client import ManoloClient

IMAGE_ROOT = "manolo/base/data/python_wrapper/manolo_client/demo/images"
DSN = 10001  #

DOWNLOAD_DIR = "manolo/base/data/python_wrapper/manolo_client/demo/output"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif",
                    ".tiff", ".webp"}
if __name__ == "__main__":

    client = ManoloClient(
        base_url="http://localhost:5001",  # Replace with your Manolo server / port
        username="manolo",
        password="manolo",
        logging_level=logging.DEBUG,
        log_dir="manolo/base/data/python_wrapper/manolo_client/demo/logs",
        log_to_file=True,
        log_to_console=False
    )

    client.login()

    manifest = client.load_manifest()

    dataset = client.get_datastructure(dsn=DSN)

    if dataset is None or isinstance(dataset, str):
        client.create_datastructure(DSN, "TestPhotos", "image")

    client.create_item_batch_files(
        manifest=manifest,
        item_paths=IMAGE_ROOT,
        dsn=DSN,
    )

    client.save_manifest(manifest)

    items = client.get_items(dsn=DSN)  # Get the items

    for item_id in items:
        print(f"[INFO] Downloading item {item_id}")
        client.download_item_data(
            id=item_id, output_path=DOWNLOAD_DIR, dsn=DSN)

    client.logout()

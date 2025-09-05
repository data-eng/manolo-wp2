import base64
import logging

from manolo_client.client import ManoloClient

DSN = 1000  # Data structure number
key_b64 = "kVnA8+8nFbGHWJ9fAptF6Bp22E5h8lfUPJ1/jjvgL0c="  # Base64 encoded key
MANIFEST_PATH = "manifest.json"  # Path to the manifest file
DOWNLOAD_DIR = "output"  # Path to the output directory

if __name__ == "__main__":

    client = ManoloClient(
        base_url="http://localhost:5002",  # Replace with your Manolo server / port
        username="manolo",
        password="manolo",
        key=base64.b64decode(key_b64),
        logging_level=logging.DEBUG,
        log_dir="logs",
        log_to_file=True,
        log_to_console=False
    )

    manifest = client.load_manifest(
        manifest_path=MANIFEST_PATH)  # Load the manifest

    dataset = client.get_datastructure(dsn=DSN)  # Get the datastructure

    if dataset is None or isinstance(dataset, str):
        raise Exception("Datastructure not found.")

    items = client.get_items(dsn=DSN)  # Get the items

    if not items:
        print("[INFO] No items found to download.")
    else:
        for item_id in items:
            print(f"[INFO] Downloading item {item_id}")
            client.download_and_decrypt_item(
                item_id=item_id, output_dir=DOWNLOAD_DIR, dsn=DSN)

    client.logout()

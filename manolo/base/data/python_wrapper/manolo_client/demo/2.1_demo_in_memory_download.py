import base64
import logging
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client import ManoloClient

DSN = 1000  # Data structure number
key_b64 = "kVnA8+8nFbGHWJ9fAptF6Bp22E5h8lfUPJ1/jjvgL0c="  # Base64 encoded key
MANIFEST_PATH = "manifest.json"  # Path to the manifest file

if __name__ == "__main__":

    # Initialize client
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

    client.login()

    # Load manifest
    manifest = client.load_manifest(manifest_path=MANIFEST_PATH)

    # Check dataset
    dataset = client.get_datastructure(dsn=DSN)

    if dataset is None or isinstance(dataset, str):
        raise Exception("Datastructure not found.")

    # Get items
    items = client.get_items(dsn=DSN)

    if items:
        # Get and decrypt first item
        item_data = client.get_and_decrypt_item(dsn=DSN, item_id=items[0])

    # Logout
    client.logout()

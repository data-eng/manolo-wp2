import base64
import logging
from manolo_client.client import ManoloClient

IMAGE_ROOT = "images"  # Path to the folder containing the images change to your path

DSN = 1000  # Data structure number can be any number

# Base64 encoded key replace with your own if needed
key_b64 = "kVnA8+8nFbGHWJ9fAptF6Bp22E5h8lfUPJ1/jjvgL0c="

MANIFEST_PATH = "manifest.json"  # Path to the manifest file

DOWNLOAD_DIR = "output"  # Path to the output directory

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif",
                    ".tiff", ".webp"}  # Set of allowed image extensions

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

    client.login()

    manifest = client.load_manifest(manifest_path=MANIFEST_PATH)

    dataset = client.get_datastructure(dsn=DSN)

    if dataset is None or isinstance(dataset, str):
        # Create the datastructure if it doesn't exist
        client.create_datastructure(DSN, "ArxPhotos", "image")

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

    client.logout()

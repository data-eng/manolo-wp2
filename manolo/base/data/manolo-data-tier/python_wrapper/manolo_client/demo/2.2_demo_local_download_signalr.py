import base64
import logging
import asyncio
from manolo_client.client import ManoloClient

DSN = 1000  # Data structure number
key_b64 = "kVnA8+8nFbGHWJ9fAptF6Bp22E5h8lfUPJ1/jjvgL0c="  # Base64 encoded key
DOWNLOAD_DIR = "output"     # Path to the output directory


async def main():

    client = ManoloClient(
        base_url="http://localhost:5002",  # Replace with your Manolo server / port
        username="manolo",
        password="manolo",
        key=base64.b64decode(key_b64),
        logging_level=logging.DEBUG,
        log_dir="logs",
        log_to_file=True,
        log_to_console=False,
        max_workers=500
    )

    client.login()

    dataset = client.get_datastructure(dsn=DSN)  # Get the datastructure

    if dataset is None or isinstance(dataset, str):
        raise Exception("Datastructure not found.")

    items = client.get_items(dsn=DSN)  # Get the items

    if not items:
        print("[INFO] No items found to download.")
    else:

        client._download_events = {item_id: asyncio.Event()
                                   for item_id in items}

        # Download the items using SignalR
        await client.download_items_via_signalr(
            dsn=DSN,
            output_dir=DOWNLOAD_DIR,
            item_ids=items
        )

if __name__ == "__main__":
    asyncio.run(main())

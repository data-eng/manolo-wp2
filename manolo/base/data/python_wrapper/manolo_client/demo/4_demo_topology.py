import base64
import logging

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client import ManoloClient
from ..enums.DefaultDatastucts import DefaultDatastucts

key_b64 = "kVnA8+8nFbGHWJ9fAptF6Bp22E5h8lfUPJ1/jjvgL0c="  # Base64 encoded key
# Path to the JSON file change to your path
json = "~/Downloads/UC2_updated_topo.json"

if __name__ == "__main__":

    client = ManoloClient(
        base_url="http://localhost:5002",  # Replace with your Manolo server / port
        username="manolo",
        password="manolo",
        key=base64.b64decode(key_b64),
        logging_level=logging.INFO,
        log_dir="logs",
        log_to_file=True,
        log_to_console=False,
        performance_monitor=True,
        log_args=False
    )

    client.login()

    loaded_json = client.load_json_as_object(
        json)  # Load the JSON file as an object

    client.create_kvps_from_object(
        obj=loaded_json, framework=DefaultDatastucts.TOPOLOGY)  # Create the key-value pairs

    # Export the JSON file replace node_1 with your actual node
    client.export_json_from_object("node_1")

    client.logout()

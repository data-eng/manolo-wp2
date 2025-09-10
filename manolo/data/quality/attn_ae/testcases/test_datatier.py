import base64
import logging
import json

import manolo_client.client

with open( "testcases/server.json", "r" ) as f:
    creds = json.load( f )

data_service = manolo_client.client.ManoloClient(
    base_url=creds["url"],
    username=creds["username"],
    password=creds["password"],
    key=base64.b64decode( creds["key"] ),
    logging_level=logging.DEBUG,
    log_dir="logs",
    log_to_file=True,
    log_to_console=False
)

data_service.login()

data_service.logout()


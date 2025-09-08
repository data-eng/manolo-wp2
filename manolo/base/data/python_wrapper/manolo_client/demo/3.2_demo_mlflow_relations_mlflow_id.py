import base64
import logging

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client import ManoloClient

from enums.MlflowSearchParams import MlflowSearchParams
from helpers.mlflow_helpers import MlflowHelper

key_b64 = "kVnA8+8nFbGHWJ9fAptF6Bp22E5h8lfUPJ1/jjvgL0c="  # Base64 encoded key

# MLflow alias replace RUNID with your actual run ID
ALIASTEST = "mlflow://run/RUNID}"

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

    response = client.get_from_mlflow(
        search_for=MlflowSearchParams.RUN, runId=ALIASTEST)  # Get the item data using MLflow

    # Create a new key-value pair
    client.create_kvp(
        object_id=ALIASTEST,
        key="Energy Consumption",
        value="22 PJ"
    )

    # Print the metrics from the MLflow response replace KEYNAME with your actual metric key
    print("The metric is:",
          MlflowHelper.get_metric_value(response, "KEYNAME"))

    energy = client.get_values(
        object_id=ALIASTEST, key="Energy Consumption")
    print("The energy consumption is", energy[-1].Value)

    client.logout()

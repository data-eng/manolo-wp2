#!/bin/bash

PASSWD=$(cat /run/secrets/db_password)

/usr/local/bin/mlflow server --backend-store-uri postgresql://manolo:${PASSWD}@db:5432/mlflow_db --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5000


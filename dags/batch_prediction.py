"""
Batch Prediction DAG.

Runs batch predictions using an H2O model from an MLflow run.
Accepts an optional run_id via DAG conf; defaults to the latest run.
"""

from __future__ import annotations

import os
from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


with DAG(
    dag_id="batch_prediction",
    description="Run batch predictions using an MLflow-logged H2O model",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "prediction", "batch"],
    params={"run_id": ""},
) as dag:

    predict = DockerOperator(
        task_id="batch_predict",
        image="ml-platform-model-serving:latest",
        command=(
            "/opt/scripts/serve.py"
            " {{ '--run-id ' ~ params.run_id if params.run_id else '' }}"
        ),
        docker_url="unix://var/run/docker.sock",
        network_mode="ml-platform_default",
        mounts=[
            Mount(
                source=os.path.join(
                    os.environ.get("HOST_PROJECT_DIR", "/Users/satrijandi/code/ml-platform"),
                    "scripts",
                ),
                target="/opt/scripts",
                type="bind",
                read_only=True,
            ),
        ],
        environment={
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", "mlplatform-access-key"),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", "mlplatform-secret-key"),
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MLFLOW_S3_ENDPOINT_URL": "http://seaweedfs:8333",
        },
        mount_tmp_dir=False,
        mem_limit="1g",
        auto_remove="success",
        retrieve_output=True,
    )

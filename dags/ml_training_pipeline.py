"""
ML Training Pipeline DAG.

Submits a PySparkling training job via DockerOperator.
The job trains an H2O AutoML model, saves data to SeaweedFS,
and logs the model artifact to MLflow.
"""

from __future__ import annotations

import os
from datetime import datetime

from airflow.sdk import DAG, Variable
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


with DAG(
    dag_id="ml_training_pipeline",
    description="Train ML model with PySparkling, save to SeaweedFS, log to MLflow",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "training", "pysparkling"],
) as dag:

    train = DockerOperator(
        task_id="train_model",
        image="pysparkling:latest",
        entrypoint="python",
        command="/opt/scripts/train.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="ml-platform_default",
        mounts=[
            Mount(
                source=os.path.join(os.environ.get("HOST_PROJECT_DIR", "/Users/satrijandi/code/ml-platform"), "scripts"),
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

"""
Batch prediction script.

Loads an H2O MOJO model from an MLflow run and runs batch predictions
on the training data stored in SeaweedFS, saving results back to S3.

Usage (inside the serving container via Airflow DockerOperator):
    python /opt/scripts/serve.py --run-id <mlflow_run_id>

If --run-id is omitted, the latest finished run is used.
"""

import argparse
import glob
import os
import tempfile
from datetime import datetime

import h2o
import mlflow
import pandas as pd


def get_run(run_id=None):
    """Resolve the MLflow run, defaulting to the latest finished run."""
    experiment = mlflow.get_experiment_by_name("ml-platform-training")
    if experiment is None:
        raise RuntimeError("MLflow experiment 'ml-platform-training' not found")

    if run_id:
        return mlflow.get_run(run_id)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No finished runs found")
    return mlflow.get_run(runs.iloc[0]["run_id"])


def read_parquet_from_s3(s3, bucket, prefix):
    """Download and read all parquet files under an S3 prefix."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    parquet_keys = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".parquet")]
    dfs = []
    for key in parquet_keys:
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            s3.download_file(bucket, key, tmp.name)
            dfs.append(pd.read_parquet(tmp.name))
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Batch prediction using an MLflow-logged H2O MOJO")
    parser.add_argument("--run-id", default=None, help="MLflow run ID (default: latest finished run)")
    parser.add_argument("--input", default=None,
                        help="S3 path to input parquet (default: latest training data)")
    parser.add_argument("--output", default=None,
                        help="S3 path for output parquet (default: s3a://mlplatform-data/batch_predictions/<timestamp>)")
    args = parser.parse_args()

    s3_endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]
    s3_access_key = os.environ["AWS_ACCESS_KEY_ID"]
    s3_secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Resolve run and model
    run = get_run(args.run_id)
    run_id = run.info.run_id
    best_model_id = run.data.params["best_model_id"]
    n_features = int(run.data.params["n_features"])
    feature_names = [f"C{i+1}" for i in range(n_features)]
    print(f"Run: {run_id}, Model: {best_model_id}")

    # Download MOJO artifact
    artifact_dir = tempfile.mkdtemp()
    model_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="h2o_model",
        dst_path=artifact_dir,
    )
    mojo_files = glob.glob(os.path.join(model_dir, "*.zip"))
    if not mojo_files:
        raise RuntimeError(f"No MOJO zip found in {model_dir}")
    mojo_path = mojo_files[0]

    # Init H2O and import MOJO
    h2o.init()
    model = h2o.import_mojo(mojo_path)
    print(f"MOJO loaded: {model.model_id}")

    # Resolve input data
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    if args.input:
        input_path = args.input
    else:
        resp = s3.list_objects_v2(Bucket="mlplatform-data", Prefix="training/")
        parquet_keys = [
            obj["Key"] for obj in resp.get("Contents", [])
            if obj["Key"].endswith(".parquet")
        ]
        if not parquet_keys:
            raise RuntimeError("No training data found in s3://mlplatform-data/training/")
        latest_key = sorted(parquet_keys)[-1]
        input_path = "s3a://mlplatform-data/" + "/".join(latest_key.split("/")[:2])
    print(f"Input: {input_path}")

    # Read input parquet
    path_no_scheme = input_path.replace("s3a://", "").replace("s3://", "")
    bucket = path_no_scheme.split("/")[0]
    prefix = "/".join(path_no_scheme.split("/")[1:])
    input_df = read_parquet_from_s3(s3, bucket, prefix)
    print(f"Loaded {len(input_df)} rows")

    # Select feature columns and predict
    predict_cols = [c for c in feature_names if c in input_df.columns]
    frame = h2o.H2OFrame(input_df[predict_cols])

    print("Running batch predictions...")
    preds = model.predict(frame)
    preds_df = preds.as_data_frame()

    # Combine input + predictions
    result_df = pd.concat([input_df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)

    # Write output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"s3a://mlplatform-data/batch_predictions/{ts}"
    path_no_scheme = output_path.replace("s3a://", "").replace("s3://", "")
    out_bucket = path_no_scheme.split("/")[0]
    out_prefix = "/".join(path_no_scheme.split("/")[1:])
    out_key = f"{out_prefix}/predictions.parquet"

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        result_df.to_parquet(tmp.name, index=False)
        s3.upload_file(tmp.name, out_bucket, out_key)
    print(f"Predictions saved to s3://{out_bucket}/{out_key}")

    # Log batch prediction run to MLflow
    with mlflow.start_run(run_name=f"batch-predict-{ts}"):
        mlflow.log_param("source_run_id", run_id)
        mlflow.log_param("model_id", best_model_id)
        mlflow.log_param("input_path", input_path)
        mlflow.log_param("output_path", f"s3://{out_bucket}/{out_key}")
        mlflow.log_metric("n_rows", len(result_df))

    print(f"Batch prediction complete: {len(result_df)} rows")

    h2o.cluster().shutdown()


if __name__ == "__main__":
    main()

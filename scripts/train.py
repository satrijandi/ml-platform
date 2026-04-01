"""
PySparkling ML training pipeline.

Runs inside the pysparkling container via Airflow DockerOperator.
- Generates synthetic data
- Trains an H2O AutoML model via PySparkling
- Saves predictions to SeaweedFS (S3)
- Logs model artifact and metrics to MLflow
"""

import os
from datetime import datetime

import mlflow
from pyspark.sql import SparkSession
from pysparkling import H2OContext
import h2o
from h2o.automl import H2OAutoML


def main():
    s3_endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]
    s3_access_key = os.environ["AWS_ACCESS_KEY_ID"]
    s3_secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("ml-platform-training")
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", s3_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", s3_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .getOrCreate()
    )

    hc = H2OContext.getOrCreate()

    # Generate synthetic classification dataset
    print("Generating synthetic dataset...")
    train_h2o = h2o.create_frame(
        rows=1000,
        cols=10,
        real_fraction=0.7,
        integer_fraction=0.3,
        categorical_fraction=0,
        missing_fraction=0.02,
        seed=42,
    )
    # Add binary target column
    import random
    random.seed(42)
    target = h2o.H2OFrame(
        [[random.choice([0, 1])] for _ in range(train_h2o.nrows)],
        column_names=["target"],
    )
    target["target"] = target["target"].asfactor()
    train_h2o = train_h2o.cbind(target)

    predictors = [c for c in train_h2o.columns if c != "target"]

    # Train with H2O AutoML
    print("Training H2O AutoML model...")
    aml = H2OAutoML(
        max_models=5,
        max_runtime_secs=120,
        seed=42,
        sort_metric="AUC",
    )
    aml.train(x=predictors, y="target", training_frame=train_h2o)

    leader = aml.leader
    perf = leader.model_performance()
    auc = perf.auc()
    logloss = perf.logloss()
    print(f"Best model: {leader.model_id}, AUC={auc:.4f}, LogLoss={logloss:.4f}")

    # Generate predictions and save to SeaweedFS as parquet
    # Use unique run paths to avoid .mode("overwrite") which triggers _temporary
    # dir deletes that SeaweedFS doesn't handle well (returns HTTP 500).
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Saving predictions to SeaweedFS...")
    preds_h2o = leader.predict(train_h2o)
    preds_spark = hc.asSparkFrame(preds_h2o)
    preds_path = f"s3a://mlplatform-data/predictions/{run_ts}"
    preds_spark.write.parquet(preds_path)
    print(f"Predictions saved to {preds_path}")

    # Save training data to SeaweedFS
    train_spark = hc.asSparkFrame(train_h2o)
    train_path = f"s3a://mlplatform-data/training/{run_ts}"
    train_spark.write.parquet(train_path)
    print(f"Training data saved to {train_path}")

    # Log to MLflow
    print("Logging to MLflow...")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("ml-platform-training")

    with mlflow.start_run(run_name="pysparkling-automl") as run:
        mlflow.log_param("max_models", 5)
        mlflow.log_param("max_runtime_secs", 120)
        mlflow.log_param("sort_metric", "AUC")
        mlflow.log_param("n_rows", train_h2o.nrows)
        mlflow.log_param("n_features", len(predictors))
        mlflow.log_param("best_model_id", leader.model_id)

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("logloss", logloss)

        # Save H2O model as MOJO (self-contained, no base model dependencies)
        mojo_path = leader.download_mojo(path="/tmp", get_genmodel_jar=True)
        mlflow.log_artifact(mojo_path, artifact_path="h2o_model")
        mlflow.log_artifact("/tmp/h2o-genmodel.jar", artifact_path="h2o_model")

        print(f"MLflow run ID: {run.info.run_id}")
        print(f"MLflow experiment: ml-platform-training")

    print("Pipeline complete.")

    hc.stop()
    spark.stop()


if __name__ == "__main__":
    main()

"""
experiment_tracker.py
Wraps MLflow logging. Reusable across any training job.
Replaces + enhances: mlflow/exepriment_tracker.py
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost


class ExperimentTracker:

    def __init__(self, model_name: str, model_version: str):
        self.model_name    = model_name
        self.model_version = model_version

    def start_run(self):
        if mlflow.active_run():
            mlflow.end_run()
        return mlflow.start_run(run_name=f"{self.model_name}_{self.model_version}")

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_model(self, model, framework: str = "sklearn"):
        if framework == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path=self.model_name)
        else:
            mlflow.sklearn.log_model(model, artifact_path=self.model_name)

    def log_artifact(self, path: str):
        mlflow.log_artifact(path)

    def end_run(self):
        mlflow.end_run()

    @property
    def run_id(self):
        run = mlflow.active_run()
        return run.info.run_id if run else None

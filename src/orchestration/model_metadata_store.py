"""
model_metadata_store.py
Tracks model lineage, signatures, and registry transitions.
"""
import mlflow
from mlflow.tracking import MlflowClient


class ModelMetadataStore:

    def __init__(self):
        self.client = MlflowClient()

    def register(self, run_id: str, model_name: str, artifact_name: str = None) -> str:
        artifact = artifact_name or model_name
        model_uri = f"runs:/{run_id}/{artifact}"  # ← was "model", now uses actual name
        result    = mlflow.register_model(model_uri, model_name)
        print(f"Registered: {model_name} v{result.version}")
        return result.version

    def transition(self, model_name: str, version: str, stage: str):
        self.client.transition_model_version_stage(
            name=model_name, version=version, stage=stage,
            archive_existing_versions=(stage == "Production"))
        print(f"Transitioned {model_name} v{version} → {stage}")

    def list_versions(self, model_name: str):
        return self.client.search_model_versions(f"name='{model_name}'")

    def promote_latest(self, model_name: str):
        versions = self.list_versions(model_name)
        latest   = max(versions, key=lambda v: int(v.version))
        self.transition(model_name, latest.version, "Production")
        return latest.version

    def rollback(self, model_name: str, version: str):
        self.transition(model_name, version, "Production")

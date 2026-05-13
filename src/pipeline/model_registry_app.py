"""
model_registry_app.py
High-level interface to MLflow Model Registry.
Replaces: model registry calls scattered in train_model.py + pricing_api.py
"""
from orchestration.model_metadata_store import ModelMetadataStore

_store = ModelMetadataStore()


def register(run_id: str, model_name: str) -> str:
    return _store.register(run_id, model_name)


def promote(model_name: str, version: str):
    _store.transition(model_name, version, "Production")


def promote_latest(model_name: str):
    return _store.promote_latest(model_name)


def rollback(model_name: str, version: str):
    _store.rollback(model_name, version)


def archive(model_name: str, version: str):
    _store.transition(model_name, version, "Archived")


def list_versions(model_name: str):
    return _store.list_versions(model_name)

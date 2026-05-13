"""
mlflow_connector.py
Low-level MLflow client utilities. Reusable across all training jobs.

Two tracking modes (set in YAML mlflow.tracking_uri):
  1. Local file  → tracking_uri: "mlruns"          (no server needed)
  2. Remote HTTP → tracking_uri: "http://localhost:5000"  (server must be running)
"""
import os
import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(config: dict):
    mlflow_cfg   = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")

    if tracking_uri.startswith("http"):
        # ── Remote server mode ────────────────────────────────────────────────
        # Validate the server is actually reachable before setting it.
        # If not reachable, fall back to local file tracking with a clear warning.
        if not _is_server_reachable(tracking_uri):
            print(
                f"\n[MLflow] ⚠  WARNING: MLflow server not reachable at {tracking_uri}\n"
                f"[MLflow]    Falling back to local file tracking.\n"
                f"[MLflow]    To start the server run:\n"
                f"[MLflow]      mlflow server --host 127.0.0.1 --port 5000\n"
            )
            tracking_uri = _setup_local_tracking()
        else:
            print(f"[MLflow] Connected to remote server: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)

    else:
        # ── Local file mode ───────────────────────────────────────────────────
        tracking_uri = _setup_local_tracking(tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_cfg.get("experiment_name", "default")
    mlflow.set_experiment(experiment_name)

    print(f"[MLflow] Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"[MLflow] Experiment   : {experiment_name}")
    return experiment_name


def _setup_local_tracking(relative_path: str = "mlruns") -> str:
    """
    Resolve a local mlruns folder relative to PROJECT_ROOT and return
    a file:/// URI that MLflow accepts on Windows.
    """
    from bootstrap.train_config_loader import PROJECT_ROOT
    abs_path = os.path.join(PROJECT_ROOT, relative_path)
    os.makedirs(abs_path, exist_ok=True)
    # Windows requires forward slashes in file:/// URIs
    uri = "file:///" + abs_path.replace(os.sep, "/")
    print(f"[MLflow] Local tracking folder: {abs_path}")
    return uri


def _is_server_reachable(uri: str, timeout: int = 3) -> bool:
    """
    Quick TCP check — returns True if the MLflow server is accepting connections.
    Does NOT authenticate — just checks if the port is open.
    """
    import socket
    from urllib.parse import urlparse
    try:
        parsed = urlparse(uri)
        host   = parsed.hostname or "localhost"
        port   = parsed.port    or 5000
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def get_client() -> MlflowClient:
    return MlflowClient()
"""
train_context.py
Manages MLflow session + Ray initialisation + optional config file watcher.
Replaces: config/file_watcher.py + mlflow/mlflow_manager.py
"""
import os
import time
import yaml
import mlflow
import ray
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from bootstrap.train_config_loader import create_new_config_version, load_training_config
from bootstrap.train_config_loader import get_active_config, PROJECT_ROOT



# ── MLflow setup ──────────────────────────────────────────────────────────────

def setup_mlflow(config: dict) -> str:
    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    if not tracking_uri.startswith("http"):
        tracking_uri = os.path.join(PROJECT_ROOT, tracking_uri)
        os.makedirs(tracking_uri, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_cfg.get("experiment_name", "default")
    mlflow.set_experiment(experiment_name)
    return experiment_name


# ── Ray initialisation ────────────────────────────────────────────────────────

def setup_ray(config: dict):
    """
    Initialise Ray from config.
    In dev (address=null) → local mode.
    In prod (address=ray://...) → connects to remote cluster.
    """
    ray_cfg  = config.get("ray", {})
    address  = ray_cfg.get("address")          # None = local
    num_cpus = ray_cfg.get("num_cpus", 4)
    num_gpus = ray_cfg.get("num_gpus", 0)

    if not ray.is_initialized():
        if address:
            ray.init(address=address)
        else:
            ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)
        print(f"Ray initialised — address={address or 'local'}, cpus={num_cpus}")
    return ray


# ── Config file watcher (from file_watcher.py) ────────────────────────────────

class _ConfigWatcher(FileSystemEventHandler):
    def __init__(self):
        self._last = 0

    def on_modified(self, event):
        if not event.src_path.endswith("base_config.yaml"):
            return
        now = time.time()
        if now - self._last < 2:
            return
        self._last = now
        print("YAML config updated — refreshing versioned config")
        with open(os.path.join(PROJECT_ROOT, "config", "base_config.yaml")) as f:
            new_cfg = yaml.safe_load(f)
        create_new_config_version(new_cfg, "Updated via YAML file watcher")


def start_config_watcher():
    observer = Observer()
    config_path = os.path.join(PROJECT_ROOT, "config")
    if not os.path.exists(config_path):
        print(f"Config watcher skipped — path not found: {config_path}")
        return None
    observer.schedule(_ConfigWatcher(), path=config_path, recursive=False)
    observer.start()
    print("Config file watcher started")
    return observer

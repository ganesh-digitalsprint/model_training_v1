"""
train_config_loader.py
Loads and merges base_config + env override + job YAML.
Replaces: config/settings.py + config/config_service.py (YAML bootstrap part)
"""
import os
import yaml
import json
import sqlite3
from threading import Lock
from pathlib import Path
from datetime import datetime

_config_lock = Lock()
_runtime_config = None

# ── Resolve root paths relative to THIS file's location ──────────────────────
# __file__ = .../training/src/bootstrap/train_config_loader.py
# PROJECT_ROOT = .../training/
_THIS_FILE   = os.path.abspath(__file__)
_BOOTSTRAP   = os.path.dirname(_THIS_FILE)   # .../src/bootstrap/
_SRC         = os.path.dirname(_BOOTSTRAP)   # .../src/
PROJECT_ROOT = os.path.dirname(_SRC)         # .../training/

DB_PATH      = os.path.join(PROJECT_ROOT, "database", "config_store.db")
DEFAULT_YAML = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")


def _merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def load_training_config(job: str, env: str = "dev") -> dict:
    """
    Merge priority (highest wins):
      job YAML > env YAML > base_config YAML
    """
    root = Path(__file__).resolve()

    # Walk up until we find the 'config' folder
    while not (root / "config").exists():
        root = root.parent

    print("FINAL ROOT:", root)

    base_path = os.path.join(root, "config", "base_config.yaml")
    env_path  = os.path.join(root, "config", "environments", f"{env}.yaml")
    job_path  = os.path.join(root, "config", "jobs", f"{job}.yaml")

    with open(base_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            config = _merge(config, yaml.safe_load(f))

    if os.path.exists(job_path):
        with open(job_path, encoding="utf-8") as f:
            config = _merge(config, yaml.safe_load(f))

    return config


# ── SQLite-backed versioned config ────────────────────────────────────────────

def init_config_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS config_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT, config_json TEXT,
            is_active INTEGER, created_at TEXT)""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS config_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT, change_description TEXT, changed_at TEXT)""")
    conn.commit()
    conn.close()


def bootstrap_config_from_yaml(yaml_path: str = "config/base_config.yaml"):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM config_versions")
    if cur.fetchone()[0] > 0:
        conn.close()
        return
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cur.execute(
        "INSERT INTO config_versions (version,config_json,is_active,created_at) VALUES (?,?,?,?)",
        ("v1.0", json.dumps(cfg), 1, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_active_config() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(
        "SELECT config_json FROM config_versions WHERE is_active=1 ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    raise RuntimeError("No active config found in DB. Run init_config_db() and bootstrap_config_from_yaml() first.")


def create_new_config_version(new_config: dict, description: str):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    version = f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    cur.execute("UPDATE config_versions SET is_active=0")
    cur.execute(
        "INSERT INTO config_versions (version,config_json,is_active,created_at) VALUES (?,?,?,?)",
        (version, json.dumps(new_config), 1, datetime.utcnow().isoformat()))
    cur.execute(
        "INSERT INTO config_audit (version,change_description,changed_at) VALUES (?,?,?)",
        (version, description, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    print(f"Config version created: {version}")
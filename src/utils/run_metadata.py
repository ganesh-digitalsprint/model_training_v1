"""
run_metadata.py
Generates unique run metadata (timestamp, run_id).
Replaces: src/utils/run_metadata.py
"""
import uuid
from datetime import datetime
from utils.constants import (
    META_TIMESTAMP, META_MODEL_VERSION, META_RUN_ID,
    META_TIMESTAMP_FMT, VERSION,
)


def generate_run_metadata(model_version: str = VERSION) -> dict:
    return {
        META_TIMESTAMP:     datetime.now().strftime(META_TIMESTAMP_FMT),
        META_MODEL_VERSION: model_version,
        META_RUN_ID:        uuid.uuid4().hex[:6].upper(),
    }
"""
json_sanitizer.py
Recursively converts numpy types and NaN/Inf to JSON-safe Python types.
Replaces: src/utils/json_sanitizer.py
"""
import math
import numpy as np


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cleaned = [sanitize_for_json(v) for v in obj]
        return type(obj)(cleaned)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return 0.0 if (math.isnan(float(obj)) or math.isinf(float(obj))) else float(obj)
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj

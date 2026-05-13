"""
model_utils.py
Model serialisation helpers (joblib / ONNX).
"""
import joblib
import os


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved: {path}")


def load_model(path: str):
    return joblib.load(path)

"""
train_launcher.py
-----------------
CLI entry point for ALL jobs — training, pricing, and any future job.

Config is loaded directly from YAML files.  No SQLite, no DB bootstrap.

Usage
-----
# Any training job — just swap the YAML, zero code change
python src/train_launcher.py --job price_prediction_training --env dev
python src/train_launcher.py --job price_prediction_training --env prod --model xgboost --version v2.0
python src/train_launcher.py --job price_prediction_training --env dev --csv-fallback data/raw/data/golden_training_data.csv
python src/train_launcher.py --job catalog_gap_analysis --env dev

# Pricing
python src/train_launcher.py --job pricing_job --env dev
python src/train_launcher.py --job pricing_job --env prod --mode hybrid
python src/train_launcher.py --job pricing_job --env dev  --mode rule
python src/train_launcher.py --job pricing_job --env dev  --mode ai --model xgboost --version v1.0
python src/train_launcher.py --job pricing_job --env dev  --golden-fallback data/raw/data/golden_training_data.csv

How to add a new training job
------------------------------
1. Create  config/jobs/<new_job>.yaml  with training_data / delta_lake / mlflow / ml_training sections.
2. Run:    python src/train_launcher.py --job <new_job> --env dev
   train_executor.py handles it automatically — no other file needs to change.
"""

from __future__ import annotations

import argparse
import os
import sys

# ── make sure src/ is on sys.path so sibling packages resolve ────────────────
_THIS_FILE = os.path.abspath(__file__)
_SRC_DIR   = os.path.dirname(_THIS_FILE)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.environ["PYTHONIOENCODING"] = "utf-8"

from bootstrap.train_config_loader import load_training_config
from utils.constants import (
    JOB_PRICING,
    ENV_DEV, ENV_QA, ENV_PROD,
    CFG_MODEL, CFG_MODEL_VERSION, CFG_PRICING_MODE,
    CFG_PRICING, CFG_ML, DELTA_LAKE,
    CFG_TOLERANCE, CFG_ENVIRONMENT, CFG_GOLDEN_CSV_FALLBACK,
    MODE_RULE, MODE_AI, MODE_HYBRID,
    XGBOOST, VERSION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Executor registry
# Only pricing needs special dispatch — everything else is a training job
# and goes straight to run_training in train_executor.py.
# ─────────────────────────────────────────────────────────────────────────────

def _get_executor(job: str):
    """
    Return the executor callable for *job*.

    - pricing_job   → price_executor.run_pricing_job
    - everything else → train_executor.run_training
      (catalog_gap_analysis, price_prediction_training, any future training job)
    """
    if job == JOB_PRICING:
        from pipeline.price_executor import run_pricing_job
        return run_pricing_job

    # All training-style jobs share the same executor.
    # The YAML is the only thing that changes between them.
    from pipeline.train_executor import run_training
    return run_training


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DSAI Launcher — training and pricing jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--job",
        default="price_prediction_training",
        help="Job name — must match config/jobs/<job>.yaml",
    )
    parser.add_argument(
        "--env",
        default=ENV_DEV,
        choices=[ENV_DEV, ENV_QA, ENV_PROD],
        help="Target environment",
    )
    # ── shared overrides ──────────────────────────────────────────────────────
    parser.add_argument("--model",   default=None,
                        help="Override model name: linear | rf | xgboost")
    parser.add_argument("--version", default=None,
                        help="Override model version e.g. v2.0")

    # ── training-specific ─────────────────────────────────────────────────────
    parser.add_argument("--csv-fallback", default=None, dest="csv_fallback",
                        help="[Training] Dev-only: golden CSV if Delta Lake unavailable")

    # ── pricing-specific ──────────────────────────────────────────────────────
    parser.add_argument("--mode", default=None,
                        choices=[MODE_RULE, MODE_AI, MODE_HYBRID],
                        help="[Pricing] Override engine mode: rule | ai | hybrid")
    parser.add_argument("--golden-fallback", default=None, dest="golden_fallback",
                        help="[Pricing] Dev-only: golden CSV if Delta Lake unavailable")

    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> dict:
    args   = parse_args(argv)
    config = load_training_config(job=args.job, env=args.env)

    # CLI overrides win over YAML; YAML wins over hard-coded defaults
    model_name = args.model   or config.get(CFG_MODEL,         XGBOOST)
    version    = args.version or config.get(CFG_MODEL_VERSION, VERSION)

    print(f"\n{'='*60}")
    print(f"  Job   : {args.job}")
    print(f"  Env   : {args.env}")
    print(f"  Model : {model_name}  v{version}")
    if args.mode:
        print(f"  Mode  : {args.mode}")
    print(f"{'='*60}\n")

    executor = _get_executor(args.job)

    # ── Pricing — needs its own set of config keys ────────────────────────────
    if args.job == JOB_PRICING:
        pricing_cfg = config.get(CFG_PRICING, {})
        ml_cfg      = config.get(CFG_ML, {})
        delta_cfg   = config.get(DELTA_LAKE, {})

        result = executor(
            config          = config,
            mode            = args.mode or config.get(CFG_PRICING_MODE, MODE_HYBRID),
            model_name      = model_name,
            model_version   = version,
            environment     = ml_cfg.get(CFG_ENVIRONMENT, args.env),
            tolerance       = float(pricing_cfg.get(CFG_TOLERANCE, 0.05)),
            golden_fallback = args.golden_fallback or delta_cfg.get(CFG_GOLDEN_CSV_FALLBACK),
        )

    # ── All training jobs ─────────────────────────────────────────────────────
    # price_prediction_training, catalog_gap_analysis, anything new —
    # they all go to the same run_training().  Only the YAML differs.
    else:
        result = executor(
            config            = config,
            model_name        = model_name,
            version           = version,
            csv_fallback_path = args.csv_fallback,
        )

    print(f"\nJob complete: {result}")
    return result


if __name__ == "__main__":
    main()
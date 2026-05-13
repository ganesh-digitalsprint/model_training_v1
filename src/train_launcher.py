"""
train_launcher.py
CLI entry point for ALL jobs — training and pricing.

Usage
-----
# Training
python src/train_launcher.py --job price_prediction_training --env dev
python src/train_launcher.py --job price_prediction_training --env prod --model xgboost --version v2.0
python src/train_launcher.py --job price_prediction_training --env dev --csv-fallback data/raw/data/golden_training_data.csv

# Pricing
python src/train_launcher.py --job pricing_job --env dev
python src/train_launcher.py --job pricing_job --env prod --mode hybrid
python src/train_launcher.py --job pricing_job --env dev  --mode rule
python src/train_launcher.py --job pricing_job --env dev  --mode ai --model xgboost --version v1.0
python src/train_launcher.py --job pricing_job --env dev  --golden-fallback data/raw/data/golden_training_data.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_THIS_FILE = os.path.abspath(__file__)
_SRC_DIR   = os.path.dirname(_THIS_FILE)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from bootstrap.train_config_loader import (
    load_training_config, init_config_db, bootstrap_config_from_yaml,
    create_new_config_version)
from bootstrap.train_context import setup_mlflow, setup_ray, start_config_watcher
from utils.constants import (
    JOB_TRAINING, JOB_PRICING,
    ENV_DEV, ENV_QA, ENV_PROD,
    CFG_MODEL, CFG_MODEL_VERSION, CFG_PRICING_MODE,
    CFG_PRICING, CFG_ML, DELTA_LAKE,
    CFG_TOLERANCE, CFG_ENVIRONMENT, CFG_GOLDEN_CSV_FALLBACK,
    MODE_RULE, MODE_AI, MODE_HYBRID,
    XGBOOST, VERSION, DEFAULT_ENVIRONMENT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Job registry
# Maps job_name (must match config/jobs/<job>.yaml) to executor function.
# To add a new job: create the YAML + executor, then register it here.
# ─────────────────────────────────────────────────────────────────────────────

def _get_executor(job: str):
    """
    Return the executor callable for the given job name.
    Imports are deferred so unrelated dependencies are never loaded.
    """
    if job == JOB_TRAINING:
        from pipeline.train_executor import run_training
        return run_training

    if job == JOB_PRICING:
        from pipeline.price_executor import run_pricing_job
        return run_pricing_job

    raise ValueError(
        f"Unknown job '{job}'. "
        f"Register it in _get_executor() and add config/jobs/{job}.yaml."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DSAI Launcher — training and pricing jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--job",
        default=JOB_TRAINING,
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

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_training_config(job=args.job, env=args.env)

    # Sync config to versioned DB (same bootstrap for every job)
    init_config_db()
    bootstrap_config_from_yaml()
    create_new_config_version(config, f"Launched job={args.job} env={args.env}")

    watcher = None  # config watcher hook (start_config_watcher for live YAML edits)

    # ── Resolve common overrides (CLI > YAML) ─────────────────────────────────
    model_name = args.model   or config.get(CFG_MODEL,         XGBOOST)
    version    = args.version or config.get(CFG_MODEL_VERSION, VERSION)


    if args.mode:
        print(f"  Mode  : {args.mode}")
    print(f"{'='*60}\n")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    executor = _get_executor(args.job)

    if args.job == JOB_TRAINING:
        result = executor(
            model_name        = model_name,
            version           = version,
            csv_fallback_path = args.csv_fallback,
        )

    elif args.job == JOB_PRICING:
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

    else:
        # Generic fallback — executors that only need (config,) can be added here
        result = executor(config=config)

    print(f"\nJob complete: {result}")

    if watcher:
        watcher.stop()


if __name__ == "__main__":
    main()
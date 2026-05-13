"""
pricing_service.py
Main pricing orchestrator — selects engine mode and runs full pipeline.
Replaces: src/services/pricing_service.py
"""
import os
import json
import pandas as pd
from bootstrap.train_config_loader import get_active_config
from processors.custom.services.rule_service                    import run_rule_service
from processors.custom.services.ai_service                      import run_ai_service
from processors.custom.hybrid_engine                            import generate_hybrid_output
from processors.custom.champion_challenger_engine               import evaluate_champion_challenger
from processors.custom.simulation.revenue_simulator             import run_revenue_simulator
from processors.custom.simulation.ai_vs_rule_simulator          import compare_ai_vs_rule, compare_current_vs_ai
from processors.custom.simulation.champion_challenger_simulator import compare_champion_challenger
from processors.custom.simulation.pricing_dashboard             import log_pricing_dashboard
from utils.run_metadata  import generate_run_metadata
from utils.output_writer import save_output
from utils.constants import (
    CFG_PRICING, CFG_TOLERANCE, CFG_ML, CFG_DEFAULT_MODEL,
    CFG_ACTIVE_MODEL_VERSION, CFG_FEATURES,
    CFG_ENABLE_RULE_PRICING, CFG_ENABLE_AI_PRICING, CFG_ENABLE_HYBRID_MODE,
    MODE_RULE, MODE_AI, MODE_HYBRID,
    XGBOOST, VERSION, DEFAULT_TRACKING_URI,
    COL_DEMAND_QTY, COL_RUN_DATE, COL_MODEL_VERSION, COL_RUN_ID,
    META_TIMESTAMP, META_MODEL_VERSION, META_RUN_ID,
)


def run_pricing(master_df: pd.DataFrame,
                mode: str, model_name: str = None,
                model_version: str = None, tolerance: float = 0.05) -> None:

    config  = get_active_config()
    tol     = float(config.get(CFG_PRICING, {}).get(CFG_TOLERANCE, tolerance))
    ml_cfg  = config.get(CFG_ML, {})
    model_name    = model_name    or ml_cfg.get(CFG_DEFAULT_MODEL, XGBOOST)
    model_version = model_version or ml_cfg.get(CFG_ACTIVE_MODEL_VERSION, VERSION)

    features = config.get(CFG_FEATURES, {})

    if COL_DEMAND_QTY not in master_df.columns:
        raise ValueError(
            "'demand_qty' column is missing from master_df. "
            "The golden dataset loaded by delta_lake_reader must include it. "
            "Check EXPECTED_COLUMNS in delta_lake_reader.py and your Delta table."
        )

    if mode == MODE_RULE:
        assert json.loads(str(features.get(CFG_ENABLE_RULE_PRICING, "true")).lower()), \
            "Rule pricing disabled in config"
        final_df = run_rule_service(master_df)

    elif mode == MODE_AI:
        assert json.loads(str(features.get(CFG_ENABLE_AI_PRICING, "true")).lower()), \
            "AI pricing disabled in config"
        final_df = run_ai_service(master_df, model_name, model_version, tol)

    elif mode == MODE_HYBRID:
        assert json.loads(str(features.get(CFG_ENABLE_HYBRID_MODE, "true")).lower()), \
            "Hybrid mode disabled in config"
        final_df = generate_hybrid_output(master_df, model_name, model_version, tol)

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose: rule | ai | hybrid")

    metadata = generate_run_metadata(model_version)
    final_df[COL_RUN_DATE]      = metadata[META_TIMESTAMP]
    final_df[COL_MODEL_VERSION] = metadata[META_MODEL_VERSION]
    final_df[COL_RUN_ID]        = metadata[META_RUN_ID]

    rev       = run_revenue_simulator(final_df)
    ai_rule   = compare_ai_vs_rule(final_df)      if mode in [MODE_AI, MODE_HYBRID] else None
    curr_ai   = compare_current_vs_ai(final_df)   if mode in [MODE_AI, MODE_HYBRID] else None
    cc_df     = evaluate_champion_challenger(final_df, model_name)
    cc_sum    = compare_champion_challenger(cc_df)

    log_pricing_dashboard(rev["summary"], ai_rule["summary"] if ai_rule else {},
                          curr_ai or {}, cc_sum, model_name, model_version
                        )

    save_output(final_df, mode, metadata)
    print("Pricing Service Completed")
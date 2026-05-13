"""
champion_challenger_engine.py
Evaluates champion vs challenger model revenue side-by-side.
Replaces: src/engine/champion_challenger_engine.py
"""
import pandas as pd
import ray
from processors.custom.inference import predict_demand
from processors.common.trainers.model_factory import load_champion_challenger_models
from utils.constants import (
    COL_CURRENT_PRICE, COL_COMPETITOR_PRICE, COL_TOTAL_INVENTORY, COL_PRICE_DIFF_PCT,
    COL_CHAMPION_DEMAND, COL_CHALLENGER_DEMAND,
    COL_CHAMPION_REVENUE, COL_CHALLENGER_REVENUE,
    XGBOOST,
)


@ray.remote
def _score_batch(rows: pd.DataFrame, champion_ref, challenger_ref) -> pd.DataFrame:
    champion   = ray.get(champion_ref)
    challenger = ray.get(challenger_ref)
    rows = rows.copy()
    rows[COL_CHAMPION_DEMAND]   = rows.apply(
        lambda x: predict_demand(champion,   x[COL_CURRENT_PRICE], x.get(COL_COMPETITOR_PRICE),
                                 x.get(COL_TOTAL_INVENTORY, 0), x.get(COL_PRICE_DIFF_PCT, 0)), axis=1)
    rows[COL_CHALLENGER_DEMAND] = rows.apply(
        lambda x: predict_demand(challenger, x[COL_CURRENT_PRICE], x.get(COL_COMPETITOR_PRICE),
                                 x.get(COL_TOTAL_INVENTORY, 0), x.get(COL_PRICE_DIFF_PCT, 0)), axis=1)
    rows[COL_CHAMPION_REVENUE]   = rows[COL_CHAMPION_DEMAND]   * rows[COL_CURRENT_PRICE]
    rows[COL_CHALLENGER_REVENUE] = rows[COL_CHALLENGER_DEMAND] * rows[COL_CURRENT_PRICE]
    return rows


def evaluate_champion_challenger(master_df: pd.DataFrame,
                                  model_name: str = XGBOOST) -> pd.DataFrame:
    champion, challenger = load_champion_challenger_models(model_name)
    if champion is None or challenger is None:
        print("Skipping Champion-Challenger — models not available")
        return master_df

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    champ_ref  = ray.put(champion)
    chall_ref  = ray.put(challenger)

    import numpy as np
    batches = [g for _, g in master_df.groupby(
        pd.cut(range(len(master_df)), bins=max(1, len(master_df) // 50),
               labels=False))]
    parts   = ray.get([_score_batch.remote(b, champ_ref, chall_ref) for b in batches])
    return pd.concat(parts, ignore_index=True)
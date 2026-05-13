"""
xgb_trainer.py
XGBoost trainer with Ray Train distributed support.
"""
import ray.train
from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig
import pandas as pd
from processors.base.train_base_stage import TrainBaseStage


class XGBTrainer(TrainBaseStage):

    def __init__(self, params: dict, num_workers: int = 2):
        self.params      = params
        self.num_workers = num_workers

    def run(self, X_train: pd.DataFrame, y_train: pd.Series):
        import ray
        import ray.data
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        train_df          = X_train.copy()
        train_df["label"] = y_train.values
        ds = ray.data.from_pandas(train_df)

        trainer = XGBoostTrainer(
            scaling_config=ScalingConfig(num_workers=self.num_workers, use_gpu=False),
            label_column="label",
            params=self.params,
            datasets={"train": ds},
        )
        result = trainer.fit()
        print(f"XGBoost Ray Train completed. Metrics: {result.metrics}")
        return result

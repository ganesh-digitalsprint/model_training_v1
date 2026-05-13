"""
lgbm_trainer.py
LightGBM trainer stub — extend with Ray Train LightGBM integration.
"""
import lightgbm as lgb
import pandas as pd
from processors.base.train_base_stage import TrainBaseStage


class LGBMTrainer(TrainBaseStage):

    def __init__(self, params: dict):
        self.params = params
        self.model  = None

    def run(self, X_train, y_train):
        dtrain     = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(self.params, dtrain)
        print("LightGBM training complete")
        return self.model

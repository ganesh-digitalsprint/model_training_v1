"""
sklearn_adapter.py
Thin adapter to run any sklearn estimator inside the framework pipeline.
"""
from processors.base.train_base_stage import TrainBaseStage


class SklearnAdapter(TrainBaseStage):

    def __init__(self, estimator):
        self.estimator = estimator

    def run(self, X_train, y_train):
        self.estimator.fit(X_train, y_train)
        return self.estimator

"""
train_base_stage.py
Abstract base class for all Trainers, Evaluators, and Tuners.
"""
from abc import ABC, abstractmethod


class TrainBaseStage(ABC):

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute this stage."""

    def validate_inputs(self, *args, **kwargs):
        """Optional hook — override in subclass for input checks."""

    def on_success(self, result):
        """Optional hook — called after successful run."""

    def on_failure(self, error: Exception):
        """Optional hook — called on stage failure."""
        raise error

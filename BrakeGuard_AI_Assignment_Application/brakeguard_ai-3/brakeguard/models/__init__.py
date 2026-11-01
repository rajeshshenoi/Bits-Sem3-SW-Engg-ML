"""Model training, persistence (registry) and inference."""

from .predictor import BrakeFaultPredictor, Prediction
from .registry import ModelBundle, ModelRegistry
from .trainer import ModelTrainer, TrainingResult

__all__ = [
    "BrakeFaultPredictor",
    "ModelBundle",
    "ModelRegistry",
    "ModelTrainer",
    "Prediction",
    "TrainingResult",
]

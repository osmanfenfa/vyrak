"""
Machine learning training, evaluation, and lifecycle management layer.
"""

from ml.trainer import ModelTrainer, TrainingResult
from ml.evaluator import ModelEvaluator, EvaluationReport
from ml.adversarial import AdversarialTrainer, AdversarialConfig
from ml.feature_store import FeatureStore, FeatureRecord
from ml.registry import ModelRegistry, ModelVersion, ModelState

__all__ = [
    "ModelTrainer",
    "TrainingResult",
    "ModelEvaluator",
    "EvaluationReport",
    "AdversarialTrainer",
    "AdversarialConfig",
    "FeatureStore",
    "FeatureRecord",
    "ModelRegistry",
    "ModelVersion",
    "ModelState",
]

__version__ = "2.0.0"
__author__  = "Vyrak"
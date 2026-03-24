"""
Machine learning training, evaluation, and lifecycle management layer.
"""

from importlib import import_module

_EXPORT_MAP = {
    "ModelTrainer": ("ml.trainer", "ModelTrainer"),
    "TrainingResult": ("ml.trainer", "TrainingResult"),
    "ModelEvaluator": ("ml.evaluator", "ModelEvaluator"),
    "EvaluationReport": ("ml.evaluator", "EvaluationReport"),
    "AdversarialTrainer": ("ml.adversarial", "AdversarialTrainer"),
    "AdversarialConfig": ("ml.adversarial", "AdversarialConfig"),
    "FeatureStore": ("ml.feature_store", "FeatureStore"),
    "FeatureRecord": ("ml.feature_store", "FeatureRecord"),
    "ModelRegistry": ("ml.registry", "ModelRegistry"),
    "ModelVersion": ("ml.registry", "ModelVersion"),
    "ModelState": ("ml.registry", "ModelState"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'ml' has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__version__ = "2.0.0"
__author__ = "Vyrak"

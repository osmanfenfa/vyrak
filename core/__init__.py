"""
Core package exports for the Zelvryn ABSA system.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "UserAction",
    "BehavioralProfile",
    "FeatureExtractor",
    "RandomForestClassifier",
    "HoeffdingTreeClassifier",
    "ABSAv2Engine",
    "DecisionResult",
    "constants",
    "math_utils",
]

_EXPORTS = {
    "UserAction": ("core.action", "UserAction"),
    "BehavioralProfile": ("core.profile", "BehavioralProfile"),
    "FeatureExtractor": ("core.features", "FeatureExtractor"),
    "RandomForestClassifier": ("core.classifier", "RandomForestClassifier"),
    "HoeffdingTreeClassifier": ("core.engine", "HoeffdingTreeClassifier"),
    "ABSAv2Engine": ("core.engine", "ABSAv2Engine"),
    "DecisionResult": ("core.engine", "DecisionResult"),
    "constants": ("core.constants", None),
    "math_utils": ("core.math_utils", None),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'core' has no attribute {name!r}")
    module_name, attr = target
    module = import_module(module_name)
    if attr is None:
        return module
    return getattr(module, attr)


__version__ = "2.0.0"
__author__ = "Vyrak"
__system__ = "Zelvryn - Adaptive Behavioral Signature Authentication"

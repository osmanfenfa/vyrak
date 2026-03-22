"""
The core package is the algorithm heart of the Zelvryn ABSA system. It is
deliberately import-independent: zero imports from api/, db/, ingestion/,
ml/, profiles/, or any other project folder. This isolation ensures:
"""

from core.action import UserAction
from core.profile import BehavioralProfile
from core.features import FeatureExtractor
from core.classifier import RandomForestClassifier
from core.hoeffding import HoeffdingTreeClassifier
from core.engine import ABSAv2Engine, DecisionResult
from core import constants
from core import math_utils


__all__ = [
    # Core data structures
    "UserAction",
    "BehavioralProfile",
    # Feature extraction
    "FeatureExtractor",
    # ML classifiers
    "RandomForestClassifier",
    "HoeffdingTreeClassifier",
    # Main engine and result type
    "ABSAv2Engine",
    "DecisionResult",
    # Submodules (for direct access when needed)
    "constants",
    "math_utils",
]

__version__ = "2.0.0"
__author__  = "Vyrak"
__system__  = "Zelvryn — Adaptive Behavioral Signature Authentication"
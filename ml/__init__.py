"""
zelvryn/ml/__init__.py
======================
Machine learning training, evaluation, and lifecycle management layer.

This package owns everything related to the ML classifier lifecycle:
training, cross-validation, adversarial hardening, feature persistence,
and model versioning. It sits above core/ (which owns inference) and
below api/ (which owns serving).

Dependency rule
---------------
ml/ imports from core/ only.
core/ never imports from ml/.
api/, profiles/, and ingestion/ may import from ml/ for training triggers
and model loading, but ml/ never imports from them.

Components
----------
  ModelTrainer (trainer.py)
    Trains scikit-learn RandomForestClassifier on labeled feature vectors.
    Wraps cross-validation, model serialisation, and retraining scheduling.

  ModelEvaluator (evaluator.py)
    Computes precision, recall, F1, FAR, FRR, ROC curve data, and confusion
    matrices. Used by the registry to gate model promotion decisions.

  AdversarialTrainer (adversarial.py)
    GAN-based adversarial hardening loop. A generator network learns to
    produce feature vectors that fool the Random Forest. The classifier
    retrains against these adversarial examples, producing a model that
    is robust to evasion strategies that static training cannot anticipate.

  FeatureStore (feature_store.py)
    Persists labeled (feature_vector, label) pairs for offline training.
    In-memory dict backend for testing; PostgreSQL backend for production.
    Provides batch retrieval for ModelTrainer.

  ModelRegistry (registry.py)
    Tracks all trained model versions with their metrics. Manages the
    active / canary / retired state machine. Supports atomic promotion
    and instant rollback.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
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
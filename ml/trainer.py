"""
zelvryn/ml/trainer.py
======================
ModelTrainer — trains, validates, and persists the Random Forest classifier
that powers Zelvryn's ML-based anomaly detection.

Responsibilities
----------------
  1. Train a RandomForestClassifier on labeled (X, y) feature datasets
  2. Run stratified k-fold cross-validation to produce reliable metric estimates
  3. Save trained models to disk (joblib) and register them in the ModelRegistry
  4. Manage the retraining schedule: trigger retraining after N new labeled
     samples have accumulated in the FeatureStore
  5. Log all training events with metrics for audit and debugging

Training data sources
---------------------
  The trainer draws from three sources, merged before each training run:
    a) Shadow session exports (data/shadow_exports/) — real human and flagged
       bot sessions from production observation
    b) Attack corpus features (data/processed/) — CICIDS2018, CTU-13
    c) Profile-level training data (BehavioralProfile.ml_training_data) —
       accumulated heuristic labels from live sessions

Class balance
-------------
The training dataset is class-balanced by default (equal human/bot samples)
to prevent the classifier from becoming biased toward the majority class.
The `class_weight="balanced"` parameter in sklearn's RF also handles this
within each tree, providing a second layer of imbalance correction.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.classifier import RandomForestClassifier
from core.constants import (
    ML_NUM_TREES,
    ML_MAX_DEPTH,
    ML_MIN_SAMPLES_SPLIT,
    ML_MIN_TRAIN_SAMPLES,
    ML_FEATURE_DIM,
)
from core.features import FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """
    Result of a single model training run.

    Attributes
    ----------
    version : str
        Version string assigned to this trained model (e.g., "rf_v3").
    model_path : str
        Path to the saved .joblib file.
    n_samples : int
        Total training samples used.
    n_human : int
        Human-class samples.
    n_bot : int
        Bot-class samples.
    training_duration_s : float
        Wall-clock seconds for the training call.
    cv_metrics : dict
        Cross-validation metrics: precision, recall, f1, accuracy, far, frr.
    feature_importances : dict
        {feature_name: importance_score} after permutation importance.
    top_features : list of tuple
        Top 5 (feature_name, importance) pairs.
    trained_at : float
        Unix timestamp of training completion.
    success : bool
        True if training completed without error.
    error : str or None
        Error message if training failed.
    """
    version:              str
    model_path:           str                  = ""
    n_samples:            int                  = 0
    n_human:              int                  = 0
    n_bot:                int                  = 0
    training_duration_s:  float                = 0.0
    cv_metrics:           Dict[str, float]     = field(default_factory=dict)
    feature_importances:  Dict[str, float]     = field(default_factory=dict)
    top_features:         List[Tuple[str,float]] = field(default_factory=list)
    trained_at:           float                = field(default_factory=time.time)
    success:              bool                 = True
    error:                Optional[str]        = None

    def summary(self) -> str:
        """One-line human-readable summary of training results."""
        if not self.success:
            return f"TrainingResult({self.version}: FAILED — {self.error})"
        cv = self.cv_metrics
        return (
            f"TrainingResult({self.version}: "
            f"n={self.n_samples} ({self.n_human}H/{self.n_bot}B) "
            f"F1={cv.get('f1', 0):.3f} "
            f"FAR={cv.get('far', 0):.3f} "
            f"FRR={cv.get('frr', 0):.3f} "
            f"dur={self.training_duration_s:.1f}s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version":             self.version,
            "model_path":          self.model_path,
            "n_samples":           self.n_samples,
            "n_human":             self.n_human,
            "n_bot":               self.n_bot,
            "training_duration_s": round(self.training_duration_s, 2),
            "cv_metrics":          {k: round(v, 4) for k, v in self.cv_metrics.items()},
            "top_features":        [(n, round(i, 4)) for n, i in self.top_features],
            "trained_at":          self.trained_at,
            "success":             self.success,
            "error":               self.error,
        }


class ModelTrainer:
    """
    Trains and manages Zelvryn's Random Forest classifier.

    Parameters
    ----------
    model_dir : str, optional
        Directory where trained model files are saved. Default "models/".
    n_trees : int, optional
        Number of trees in each Random Forest. Default ML_NUM_TREES.
    max_depth : int, optional
        Maximum tree depth. Default ML_MAX_DEPTH.
    cv_folds : int, optional
        Number of folds for cross-validation. Default 5.
    balance_classes : bool, optional
        If True, downsample the majority class to match the minority class
        before training. Default True.
    random_seed : int, optional
        Random seed for reproducibility. Default 42.
    min_train_samples : int, optional
        Minimum samples required to start training. Default ML_MIN_TRAIN_SAMPLES.
    """

    def __init__(
        self,
        model_dir:          str  = "models",
        n_trees:            int  = ML_NUM_TREES,
        max_depth:          int  = ML_MAX_DEPTH,
        cv_folds:           int  = 5,
        balance_classes:    bool = True,
        random_seed:        int  = 42,
        min_train_samples:  int  = ML_MIN_TRAIN_SAMPLES,
    ) -> None:
        self.model_dir          = model_dir
        self.n_trees            = n_trees
        self.max_depth          = max_depth
        self.cv_folds           = cv_folds
        self.balance_classes    = balance_classes
        self.random_seed        = random_seed
        self.min_train_samples  = min_train_samples

        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Track version numbering
        self._version_counter: int = self._detect_existing_versions()

        logger.info(
            "ModelTrainer initialised: dir=%s trees=%d depth=%d cv=%d",
            model_dir, n_trees, max_depth, cv_folds
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMARY INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        X: List[List[float]],
        y: List[int],
        version_tag: Optional[str] = None,
        run_cv: bool = True,
    ) -> TrainingResult:
        """
        Train a new Random Forest on the provided labeled dataset.

        Steps:
          1. Validate input size and class distribution
          2. Optionally balance classes
          3. Fit the RandomForestClassifier
          4. Run cross-validation (if run_cv=True)
          5. Save model to disk
          6. Return TrainingResult with all metrics

        Parameters
        ----------
        X : list of list of float
            Feature matrix. Each row is a 20-dim feature vector.
        y : list of int
            Labels: 0 = human, 1 = bot.
        version_tag : str, optional
            Custom version string (e.g., "rf_v5_post_shadow"). If None,
            auto-increments (rf_v1, rf_v2, ...).
        run_cv : bool, optional
            Whether to run cross-validation. Default True. Set False for
            rapid iteration during development.

        Returns
        -------
        TrainingResult
        """
        version = version_tag or self._next_version()
        t_start = time.time()

        logger.info(
            "Training started: version=%s samples=%d (human=%d bot=%d)",
            version, len(X), y.count(0), y.count(1)
        )

        # ── Validation ────────────────────────────────────────────────────
        validation_error = self._validate_dataset(X, y)
        if validation_error:
            logger.error("Training aborted: %s", validation_error)
            return TrainingResult(
                version=version, success=False, error=validation_error
            )

        # ── Class balancing ───────────────────────────────────────────────
        if self.balance_classes:
            X, y = self._balance_classes(X, y)
            logger.info(
                "After balancing: %d samples (%d human, %d bot)",
                len(X), y.count(0), y.count(1)
            )

        # ── Cross-validation ──────────────────────────────────────────────
        cv_metrics: Dict[str, float] = {}
        if run_cv and len(X) >= self.cv_folds * 4:
            logger.info("Running %d-fold cross-validation...", self.cv_folds)
            cv_clf = RandomForestClassifier(
                n_trees=self.n_trees,
                max_depth=self.max_depth,
                min_samples_split=ML_MIN_SAMPLES_SPLIT,
                random_state=self.random_seed,
            )
            try:
                cv_metrics = cv_clf.cross_validate(X, y, k=self.cv_folds)
                logger.info(
                    "CV results: F1=%.3f FAR=%.3f FRR=%.3f Acc=%.3f",
                    cv_metrics.get("f1", 0),
                    cv_metrics.get("far", 0),
                    cv_metrics.get("frr", 0),
                    cv_metrics.get("accuracy", 0),
                )
            except Exception as e:
                logger.warning("Cross-validation failed: %s. Continuing without CV.", e)

        # ── Full training ─────────────────────────────────────────────────
        clf = RandomForestClassifier(
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            min_samples_split=ML_MIN_SAMPLES_SPLIT,
            random_state=self.random_seed,
        )
        try:
            clf.fit(X, y)
        except Exception as e:
            logger.error("Training failed: %s", e, exc_info=True)
            return TrainingResult(
                version=version,
                success=False,
                error=str(e),
                n_samples=len(X),
                n_human=y.count(0),
                n_bot=y.count(1),
                training_duration_s=time.time() - t_start,
            )

        # ── Save model ────────────────────────────────────────────────────
        model_filename = f"{version}.joblib"
        model_path     = os.path.join(self.model_dir, model_filename)
        try:
            clf.save_model(model_path)
        except Exception as e:
            logger.warning("Model save failed: %s", e)
            model_path = ""

        # ── Build result ──────────────────────────────────────────────────
        top_features = clf.get_top_features(5)
        all_importances = clf.all_feature_importances()
        duration = time.time() - t_start

        result = TrainingResult(
            version             = version,
            model_path          = model_path,
            n_samples           = len(X),
            n_human             = y.count(0),
            n_bot               = y.count(1),
            training_duration_s = duration,
            cv_metrics          = cv_metrics,
            feature_importances = all_importances,
            top_features        = top_features,
            trained_at          = time.time(),
            success             = True,
        )

        self._save_training_log(result)

        logger.info(result.summary())
        return result

    def train_from_files(
        self,
        feature_paths: List[str],
        label_paths:   List[str],
        version_tag:   Optional[str] = None,
    ) -> TrainingResult:
        """
        Load feature matrices and labels from JSON files then train.

        Parameters
        ----------
        feature_paths : list of str
            Paths to JSON files containing lists of feature vectors.
        label_paths : list of str
            Paths to JSON files containing lists of integer labels.
        version_tag : str, optional

        Returns
        -------
        TrainingResult
        """
        X_all: List[List[float]] = []
        y_all: List[int]         = []

        for fp, lp in zip(feature_paths, label_paths):
            try:
                with open(fp) as f:
                    X_chunk = json.load(f)
                with open(lp) as f:
                    y_chunk = json.load(f)
                X_all.extend(X_chunk)
                y_all.extend(y_chunk)
                logger.info(
                    "Loaded %d samples from %s / %s",
                    len(X_chunk), fp, lp
                )
            except Exception as e:
                logger.warning("Failed to load %s / %s: %s", fp, lp, e)

        if not X_all:
            return TrainingResult(
                version=version_tag or self._next_version(),
                success=False,
                error="No data loaded from provided file paths."
            )

        return self.train(X_all, y_all, version_tag=version_tag)

    def should_retrain(
        self,
        new_samples_since_last: int,
        retrain_interval: int = 50,
    ) -> bool:
        """
        Determine whether retraining should be triggered.

        Parameters
        ----------
        new_samples_since_last : int
            Number of new labeled samples accumulated since last training.
        retrain_interval : int, optional
            Threshold. Default 50.

        Returns
        -------
        bool
        """
        return new_samples_since_last >= retrain_interval

    def load_model(self, model_path: str) -> RandomForestClassifier:
        """
        Load a previously trained model from disk.

        Parameters
        ----------
        model_path : str

        Returns
        -------
        RandomForestClassifier
        """
        return RandomForestClassifier.load_model(model_path)

    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all .joblib model files in the model directory.

        Returns
        -------
        list of dict
            Each dict: {version, path, size_kb, modified_at}
        """
        models = []
        if not os.path.exists(self.model_dir):
            return models

        for fname in sorted(os.listdir(self.model_dir)):
            if fname.endswith(".joblib"):
                full_path = os.path.join(self.model_dir, fname)
                stat      = os.stat(full_path)
                models.append({
                    "version":     fname.replace(".joblib", ""),
                    "path":        full_path,
                    "size_kb":     round(stat.st_size / 1024, 1),
                    "modified_at": stat.st_mtime,
                })
        return models

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_dataset(
        self,
        X: List[List[float]],
        y: List[int],
    ) -> Optional[str]:
        """
        Return an error message if the dataset is invalid, else None.
        """
        if len(X) < self.min_train_samples:
            return (
                f"Insufficient training samples: {len(X)} < "
                f"{self.min_train_samples} required."
            )
        if len(X) != len(y):
            return f"X/y length mismatch: {len(X)} != {len(y)}."
        if len(set(y)) < 2:
            return (
                f"Only one class present ({set(y)}). "
                f"Need both 0=human and 1=bot samples."
            )
        if X and len(X[0]) != ML_FEATURE_DIM:
            return (
                f"Feature vector dimension mismatch: "
                f"expected {ML_FEATURE_DIM}, got {len(X[0])}."
            )
        return None

    def _balance_classes(
        self,
        X: List[List[float]],
        y: List[int],
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Downsample the majority class to match the minority class count.
        """
        indices_0 = [i for i, lbl in enumerate(y) if lbl == 0]
        indices_1 = [i for i, lbl in enumerate(y) if lbl == 1]

        min_count = min(len(indices_0), len(indices_1))
        if min_count == 0:
            return X, y

        rng = random.Random(self.random_seed)
        sampled_0 = rng.sample(indices_0, min_count)
        sampled_1 = rng.sample(indices_1, min_count)

        all_indices = sampled_0 + sampled_1
        rng.shuffle(all_indices)

        X_bal = [X[i] for i in all_indices]
        y_bal = [y[i] for i in all_indices]
        return X_bal, y_bal

    def _next_version(self) -> str:
        """Return the next auto-incremented version string."""
        self._version_counter += 1
        return f"rf_v{self._version_counter}"

    def _detect_existing_versions(self) -> int:
        """
        Scan the model directory to find the highest existing version number.
        """
        if not os.path.exists(self.model_dir):
            return 0
        max_v = 0
        for fname in os.listdir(self.model_dir):
            if fname.startswith("rf_v") and fname.endswith(".joblib"):
                try:
                    n = int(fname[4:].replace(".joblib", ""))
                    max_v = max(max_v, n)
                except ValueError:
                    pass
        return max_v

    def _save_training_log(self, result: TrainingResult) -> None:
        """Append a training result to the training history log."""
        log_path = os.path.join(self.model_dir, "training_history.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except OSError as e:
            logger.warning("Failed to write training log: %s", e)

    def __repr__(self) -> str:
        return (
            f"ModelTrainer("
            f"dir={self.model_dir!r}, "
            f"trees={self.n_trees}, "
            f"depth={self.max_depth}, "
            f"cv_folds={self.cv_folds})"
        )

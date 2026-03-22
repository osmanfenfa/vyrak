"""
HoeffdingTreeClassifier — incremental online learning for Zelvryn.
"""

import logging
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from core.constants import (
    ML_FEATURE_DIM,
    ML_BOT_CONFIDENCE_THRESH,
    ML_MIN_TRAIN_SAMPLES,
)
from core.features import FeatureExtractor

logger = logging.getLogger(__name__)


class HoeffdingTreeClassifier:
    """
    Incremental online Hoeffding Tree classifier for human vs bot detection.
    """

    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "nba",
    ) -> None:
        self.grace_period       = grace_period
        self.split_confidence   = split_confidence
        self.tie_threshold      = tie_threshold
        self.leaf_prediction    = leaf_prediction

        self.is_trained: bool                   = False
        self.n_samples: int                     = 0
        self.training_timestamp: Optional[float] = None
        self.feature_importances_: List[float]   = []   # not available for HTCs

        self._model: Any = None
        self._use_river: bool = self._try_import_river()

        # Fallback Naive Bayes state (used when river is not installed)
        self._nb_class_counts: Dict[int, int] = {0: 0, 1: 0}
        self._nb_feature_sums: Dict[int, List[float]] = {
            0: [0.0] * ML_FEATURE_DIM,
            1: [0.0] * ML_FEATURE_DIM,
        }
        self._nb_feature_sq_sums: Dict[int, List[float]] = {
            0: [0.0] * ML_FEATURE_DIM,
            1: [0.0] * ML_FEATURE_DIM,
        }

    # RIVER DETECTION
    def _try_import_river(self) -> bool:
        """
        Attempt to import river. Returns True if successful, False otherwise.
        Logs a warning if river is unavailable so operators know to install it.
        """
        try:
            import river  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "river library not installed. HoeffdingTreeClassifier will use "
                "a simple Gaussian Naive Bayes fallback. For best performance, "
                "install river: pip install river"
            )
            return False

    # ONLINE LEARNING INTERFACE
    def learn_one(self, features: List[float], label: int) -> None:
        """
        Update the model with a single labeled example.
        """
        if len(features) != ML_FEATURE_DIM:
            raise ValueError(
                f"Feature vector must have dimension {ML_FEATURE_DIM}, "
                f"got {len(features)}"
            )

        if self._use_river:
            self._river_learn_one(features, label)
        else:
            self._nb_learn_one(features, label)

        self.n_samples += 1
        if not self.is_trained:
            self.is_trained = True
            self.training_timestamp = time.time()

    def predict_proba_one(self, features: List[float]) -> float:
        """
        Return P(bot) for a single feature vector using the current model state.
        """
        if not self.is_trained:
            return 0.0

        if self._use_river:
            return self._river_predict_proba_one(features)
        else:
            return self._nb_predict_proba_one(features)

    # BATCH INTERFACE (matches RandomForestClassifier API)
    def fit(
        self,
        X: List[List[float]],
        y: List[int],
    ) -> "HoeffdingTreeClassifier":
        """
        Train from a batch of samples by calling learn_one() for each.
        """
        if len(X) < ML_MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"Batch training requires at least {ML_MIN_TRAIN_SAMPLES} samples. "
                f"Got {len(X)}."
            )

        logger.info(
            "HoeffdingTree batch fit: processing %d samples one at a time.",
            len(X)
        )

        for xi, yi in zip(X, y):
            self.learn_one(xi, yi)

        self.training_timestamp = time.time()
        logger.info("HoeffdingTree batch fit complete. Total samples: %d", self.n_samples)
        return self

    def predict_proba(self, x: List[float]) -> float:
        """
        Return P(bot) for a single feature vector.
        Matches RandomForestClassifier.predict_proba() interface.
        """
        return self.predict_proba_one(x)

    def predict(self, x: List[float]) -> int:
        """
        Return predicted class (0 or 1).
        Matches RandomForestClassifier.predict() interface.
        """
        return 1 if self.predict_proba_one(x) >= ML_BOT_CONFIDENCE_THRESH else 0

    def predict_batch(self, X: List[List[float]]) -> List[float]:
        """
        Return P(bot) for a batch of feature vectors.
        """
        return [self.predict_proba_one(x) for x in X]

    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Hoeffding Trees do not support permutation importance.
        Returns all features with equal importance as a placeholder.
        """
        names = FeatureExtractor.feature_names()
        equal_importance = 1.0 / ML_FEATURE_DIM
        return [(name, equal_importance) for name in names[:n]]

    def all_feature_importances(self) -> Dict[str, float]:
        """Returns equal importance for all features (placeholder)."""
        names = FeatureExtractor.feature_names()
        equal = 1.0 / ML_FEATURE_DIM
        return {name: equal for name in names}

    # PERSISTENCE
    def save_model(self, path: str) -> None:
        """
        Persist the Hoeffding Tree model to disk.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model. Call learn_one() first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model":              self._model,
            "grace_period":       self.grace_period,
            "split_confidence":   self.split_confidence,
            "tie_threshold":      self.tie_threshold,
            "leaf_prediction":    self.leaf_prediction,
            "n_samples":          self.n_samples,
            "training_timestamp": self.training_timestamp,
            "use_river":          self._use_river,
            # Naive Bayes fallback state
            "nb_class_counts":    self._nb_class_counts,
            "nb_feature_sums":    self._nb_feature_sums,
            "nb_feature_sq_sums": self._nb_feature_sq_sums,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        logger.info("HoeffdingTree model saved to %s", path)

    @classmethod
    def load_model(cls, path: str) -> "HoeffdingTreeClassifier":
        """
        Load a persisted Hoeffding Tree model from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(
            grace_period=payload["grace_period"],
            split_confidence=payload["split_confidence"],
            tie_threshold=payload["tie_threshold"],
            leaf_prediction=payload["leaf_prediction"],
        )
        instance._model              = payload["model"]
        instance.n_samples           = payload["n_samples"]
        instance.training_timestamp  = payload["training_timestamp"]
        instance._use_river          = payload.get("use_river", False)
        instance._nb_class_counts    = payload.get("nb_class_counts", {0: 0, 1: 0})
        instance._nb_feature_sums    = payload.get("nb_feature_sums",
                                                    {0: [0.0]*ML_FEATURE_DIM,
                                                     1: [0.0]*ML_FEATURE_DIM})
        instance._nb_feature_sq_sums = payload.get("nb_feature_sq_sums",
                                                    {0: [0.0]*ML_FEATURE_DIM,
                                                     1: [0.0]*ML_FEATURE_DIM})
        instance.is_trained          = instance.n_samples > 0

        logger.info(
            "HoeffdingTree loaded from %s (%d samples)",
            path, instance.n_samples
        )
        return instance
    
    # RIVER BACKEND
    def _river_learn_one(self, features: List[float], label: int) -> None:
        """Internal: update via river HoeffdingTreeClassifier."""
        if self._model is None:
            from river.tree import HoeffdingTreeClassifier as _RiverHTC
            self._model = _RiverHTC(
                grace_period=self.grace_period,
                split_confidence=self.split_confidence,
                tie_threshold=self.tie_threshold,
                leaf_prediction=self.leaf_prediction,
            )

        # river expects a dict of {feature_name: value}
        x_dict = {
            name: val
            for name, val in zip(FeatureExtractor.feature_names(), features)
        }
        self._model.learn_one(x_dict, label)

    def _river_predict_proba_one(self, features: List[float]) -> float:
        """Internal: get P(bot) from river model."""
        if self._model is None:
            return 0.0

        x_dict = {
            name: val
            for name, val in zip(FeatureExtractor.feature_names(), features)
        }
        try:
            proba_dict = self._model.predict_proba_one(x_dict)
            # proba_dict: {0: P(human), 1: P(bot)}
            return float(proba_dict.get(1, 0.0))
        except Exception as e:
            logger.warning("River predict failed: %s", e)
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # NAIVE BAYES FALLBACK (no external dependencies)
    # Used when river is not installed.
    # Implements Gaussian Naive Bayes: P(x|class) ~ Normal(μ, σ²)
    # ─────────────────────────────────────────────────────────────────────────

    def _nb_learn_one(self, features: List[float], label: int) -> None:
        """
        Update Gaussian Naive Bayes statistics with one sample.
        """
        if label not in (0, 1):
            return

        self._nb_class_counts[label] += 1
        for i, val in enumerate(features):
            self._nb_feature_sums[label][i]    += val
            self._nb_feature_sq_sums[label][i] += val * val

    def _nb_predict_proba_one(self, features: List[float]) -> float:
        """
        Predict P(bot) using Gaussian Naive Bayes.

        Applies Bayes' theorem:
          P(class | x) ∝ P(class) * Π_i P(x_i | class)

        Returns the normalised posterior P(class=1 | x).
        Uses log-space arithmetic to prevent underflow with 20 features.
        """
        total_samples = sum(self._nb_class_counts.values())
        if total_samples < 2:
            return 0.0

        log_posteriors = {}
        for c in (0, 1):
            count = self._nb_class_counts[c]
            if count == 0:
                log_posteriors[c] = float("-inf")
                continue

            # Log prior
            log_prior = math.log(count / total_samples)
            log_likelihood = 0.0

            for i, val in enumerate(features):
                mu    = self._nb_feature_sums[c][i] / count
                sq_mu = self._nb_feature_sq_sums[c][i] / count
                var   = max(sq_mu - mu * mu, 1e-9)  # prevent zero variance
                sigma = math.sqrt(var)

                # Log N(val; mu, sigma) = -log(sigma * sqrt(2π)) - (val-mu)²/(2σ²)
                log_likelihood += (
                    -math.log(sigma * math.sqrt(2 * math.pi))
                    - ((val - mu) ** 2) / (2 * var)
                )

            log_posteriors[c] = log_prior + log_likelihood

        # Softmax over log posteriors → probabilities
        log_bot   = log_posteriors[1]
        log_human = log_posteriors[0]

        # Numerically stable softmax
        log_max = max(log_bot, log_human)
        p_bot   = math.exp(log_bot - log_max)
        p_human = math.exp(log_human - log_max)
        denom   = p_bot + p_human

        if denom == 0:
            return 0.5

        return min(1.0, max(0.0, p_bot / denom))

    # Dunder methods

    def __repr__(self) -> str:
        backend = "river" if self._use_river else "NaiveBayes"
        status = (
            f"trained on {self.n_samples} samples"
            if self.is_trained else "untrained"
        )
        return (
            f"HoeffdingTreeClassifier("
            f"backend={backend}, "
            f"grace_period={self.grace_period}, "
            f"{status})"
        )
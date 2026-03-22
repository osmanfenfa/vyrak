"""
Production Random Forest classifier — the ML heart of Zelvryn v2.

Why scikit-learn instead of the scratch RF from v1?
----------------------------------------------------
The scratch Random Forest in the original ABSA algorithm was built to illustrate
the mechanism without external dependencies — valuable for the final year project
prototype.
"""

import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.constants import (
    ML_NUM_TREES,
    ML_MAX_DEPTH,
    ML_MIN_SAMPLES_SPLIT,
    ML_MIN_TRAIN_SAMPLES,
    ML_FEATURE_DIM,
    ML_BOT_CONFIDENCE_THRESH,
)
from core.features import FeatureExtractor

logger = logging.getLogger(__name__)


class RandomForestClassifier:
    """
    Production Random Forest classifier for human vs bot behavioural classification.
    """

    def __init__(
        self,
        n_trees: int = ML_NUM_TREES,
        max_depth: int = ML_MAX_DEPTH,
        min_samples_split: int = ML_MIN_SAMPLES_SPLIT,
        random_state: int = 42,
    ) -> None:
        self.n_trees           = n_trees
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.random_state      = random_state

        # Training state
        self.is_trained: bool                 = False
        self.n_samples: int                   = 0
        self.training_timestamp: Optional[float] = None
        self.feature_importances_: List[float]   = []

        # Internal sklearn model (lazy import to keep core/ free of hard deps
        self._model = None

    # PRIMARY INTERFACE
    def fit(
        self,
        X: List[List[float]],
        y: List[int],
    ) -> "RandomForestClassifier":
        """
        Train the Random Forest on labeled feature vectors.
        """
        if len(X) < ML_MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"Training requires at least {ML_MIN_TRAIN_SAMPLES} samples. "
                f"Got {len(X)}. Accumulate more labeled data before training."
            )
        if len(set(y)) < 2:
            raise ValueError(
                f"Training requires both classes (0=human, 1=bot). "
                f"Got only class(es): {set(y)}"
            )
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got {len(X)} and {len(y)}."
            )

        try:
            from sklearn.ensemble import RandomForestClassifier as _SKLearnRF
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for RandomForestClassifier. "
                "Install it with: pip install scikit-learn"
            ) from e

        logger.info(
            "Training Random Forest: %d trees, max_depth=%d, %d samples "
            "(%d human, %d bot)",
            self.n_trees, self.max_depth, len(X), y.count(0), y.count(1)
        )

        t_start = time.time()

        self._model = _SKLearnRF(
            n_estimators=self.n_trees,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features="sqrt",          
            bootstrap=True,            
            class_weight="balanced",    
            n_jobs=-1,                
            random_state=self.random_state,
        )
        self._model.fit(X, y)

        self.is_trained          = True
        self.n_samples           = len(X)
        self.training_timestamp  = time.time()
        elapsed                  = self.training_timestamp - t_start

        logger.info("Random Forest trained in %.2f seconds.", elapsed)
        self._compute_permutation_importance(X, y)

        return self

    def predict_proba(self, x: List[float]) -> float:
        """
        Return P(bot) for a single 20-dimensional feature vector.

        Aggregates probability estimates from all trees via soft voting:
          P(bot | x) = mean over trees of P(class=1 | x, tree_t)
        """
        if not self.is_trained or self._model is None:
            return 0.0

        try:
            # sklearn expects 2D input
            proba = self._model.predict_proba([x])
            # proba shape: (1, 2) where [:, 1] is P(class=1)
            return float(proba[0][1])
        except Exception as e:
            logger.warning("predict_proba failed: %s. Returning 0.0.", e)
            return 0.0

    def predict(self, x: List[float]) -> int:
        """
        Return the predicted class (0=human, 1=bot) for a feature vector.
        Uses ML_BOT_CONFIDENCE_THRESH as the decision boundary.
        """
        return 1 if self.predict_proba(x) >= ML_BOT_CONFIDENCE_THRESH else 0

    def predict_batch(
        self,
        X: List[List[float]],
    ) -> List[float]:
        """
        Return P(bot) for a batch of feature vectors.
        """
        if not self.is_trained or self._model is None or not X:
            return [0.0] * len(X)
        try:
            proba = self._model.predict_proba(X)
            return [float(p[1]) for p in proba]
        except Exception as e:
            logger.warning("predict_batch failed: %s. Returning zeros.", e)
            return [0.0] * len(X)

    # MODEL EVALUATION
    def cross_validate(
        self,
        X: List[List[float]],
        y: List[int],
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate the classifier using k-fold cross-validation.
        """
        try:
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import precision_recall_fscore_support
        except ImportError:
            raise ImportError("scikit-learn required for cross_validate()")

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        metrics = {"precision": [], "recall": [], "f1": [], "accuracy": [],
                   "far": [], "frr": []}

        for train_idx, val_idx in skf.split(X, y):
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val   = [X[i] for i in val_idx]
            y_val   = [y[i] for i in val_idx]

            # Train a temporary model for this fold
            fold_clf = RandomForestClassifier(
                n_trees=self.n_trees,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
            try:
                fold_clf.fit(X_train, y_train)
            except ValueError:
                continue   # skip fold if it lacks both classes

            y_pred = [fold_clf.predict(x) for x in X_val]

            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average="binary", zero_division=0
            )

            # FAR: bot predicted as human / total bots
            bots    = [1 for y_t in y_val if y_t == 1]
            far_num = sum(1 for y_t, y_p in zip(y_val, y_pred) if y_t == 1 and y_p == 0)
            far     = far_num / max(len(bots), 1)

            # FRR: human predicted as bot / total humans
            humans  = [1 for y_t in y_val if y_t == 0]
            frr_num = sum(1 for y_t, y_p in zip(y_val, y_pred) if y_t == 0 and y_p == 1)
            frr     = frr_num / max(len(humans), 1)

            accuracy = sum(1 for y_t, y_p in zip(y_val, y_pred) if y_t == y_p) / len(y_val)

            metrics["precision"].append(prec)
            metrics["recall"].append(rec)
            metrics["f1"].append(f1)
            metrics["accuracy"].append(accuracy)
            metrics["far"].append(far)
            metrics["frr"].append(frr)

        return {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}

    def accuracy(
        self,
        X: List[List[float]],
        y: List[int],
    ) -> float:
        """
        Compute classification accuracy on a labeled dataset.
        """
        if not self.is_trained or not X:
            return 0.0
        predictions = [self.predict(x) for x in X]
        return sum(p == t for p, t in zip(predictions, y)) / len(y)

    # FEATURE IMPORTANCE
    def get_top_features(
        self,
        n: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-n most important features by permutation importance.
        Feature importance is computed after each fit() call. 
        """
        names = FeatureExtractor.feature_names()
        if not self.feature_importances_:
            return [(name, 0.0) for name in names[:n]]

        ranked = sorted(
            zip(names, self.feature_importances_),
            key=lambda t: t[1],
            reverse=True,
        )
        return ranked[:n]

    def all_feature_importances(self) -> Dict[str, float]:
        """
        Return all feature importance scores keyed by feature name.
        """
        names = FeatureExtractor.feature_names()
        if not self.feature_importances_:
            return {name: 0.0 for name in names}
        return dict(zip(names, self.feature_importances_))
    
    # PERSISTENCE
    def save_model(self, path: str) -> None:
        """
        Serialise the trained model to disk using joblib.
        joblib is preferred over pickle for sklearn models because it handles
        large numpy arrays more efficiently.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model. Call fit() first.")

        try:
            import joblib
        except ImportError:
            raise ImportError("joblib required for model persistence. pip install joblib")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "sklearn_model":       self._model,
            "n_trees":             self.n_trees,
            "max_depth":           self.max_depth,
            "min_samples_split":   self.min_samples_split,
            "random_state":        self.random_state,
            "n_samples":           self.n_samples,
            "training_timestamp":  self.training_timestamp,
            "feature_importances": self.feature_importances_,
        }
        joblib.dump(payload, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load_model(cls, path: str) -> "RandomForestClassifier":
        """
        Load a previously trained model from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            import joblib
        except ImportError:
            raise ImportError("joblib required for model loading. pip install joblib")

        payload = joblib.load(path)

        instance = cls(
            n_trees=payload["n_trees"],
            max_depth=payload["max_depth"],
            min_samples_split=payload["min_samples_split"],
            random_state=payload["random_state"],
        )
        instance._model              = payload["sklearn_model"]
        instance.n_samples           = payload["n_samples"]
        instance.training_timestamp  = payload["training_timestamp"]
        instance.feature_importances_ = payload.get("feature_importances", [])
        instance.is_trained          = True

        logger.info(
            "Model loaded from %s (trained on %d samples at %s)",
            path, instance.n_samples,
            time.strftime("%Y-%m-%d %H:%M:%S",
                          time.localtime(instance.training_timestamp))
        )
        return instance

    # PRIVATE HELPERS

    def _compute_permutation_importance(
        self,
        X: List[List[float]],
        y: List[int],
    ) -> None:
        """
        Compute permutation feature importance on the training data.
        """
        if not self.is_trained or not X:
            self.feature_importances_ = [0.0] * ML_FEATURE_DIM
            return

        # Use a random subsample for speed if dataset is large
        sample_size = min(len(X), 500)
        indices = random.sample(range(len(X)), sample_size)
        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]

        baseline = self.accuracy(X_sample, y_sample)
        importances = []

        for feat_idx in range(ML_FEATURE_DIM):
            # Shuffle this feature column
            X_shuffled = [row[:] for row in X_sample]
            col_values = [row[feat_idx] for row in X_shuffled]
            random.shuffle(col_values)
            for row, val in zip(X_shuffled, col_values):
                row[feat_idx] = val

            shuffled_acc = self.accuracy(X_shuffled, y_sample)
            importances.append(max(0.0, baseline - shuffled_acc))

        # Normalise
        total = sum(importances)
        if total > 0:
            self.feature_importances_ = [v / total for v in importances]
        else:
            self.feature_importances_ = [1.0 / ML_FEATURE_DIM] * ML_FEATURE_DIM

    # Dunder methods

    def __repr__(self) -> str:
        status = (
            f"trained on {self.n_samples} samples"
            if self.is_trained else "untrained"
        )
        return (
            f"RandomForestClassifier("
            f"n_trees={self.n_trees}, "
            f"max_depth={self.max_depth}, "
            f"{status})"
        )
"""
HoeffdingTreeClassifier — incremental online learning for Zelvryn.
"""

import logging
import math
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from coldstart.ramp_manager import RampUpManager
from core.action import UserAction
from core.constants import (
    ANOMALY_DUAL_SIGNAL_ESCALATION,
    ANOMALY_HIGH_ENTROPY,
    ANOMALY_LOW_SIMILARITY,
    ANOMALY_ML_BOT_DETECTED,
    ANOMALY_TEMPORAL_DRIFT,
    BEHAVIOR_WINDOW_SIZE,
    COSINE_MODERATE_THRESHOLD,
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_CHALLENGE,
    ENTROPY_HIGH_THRESHOLD,
    ENTROPY_MIN_WINDOW,
    ML_FEATURE_DIM,
    ML_BOT_CONFIDENCE_THRESH,
    ML_MIN_TRAIN_SAMPLES,
    TEMPORAL_ZSCORE_LIMIT,
    TRUST_BLOCK_THRESHOLD,
    TRUST_CHALLENGE_THRESHOLD,
    TRUST_DECAY_AI_BOT,
    TRUST_DECAY_MILD,
    TRUST_DECAY_MODERATE,
    TRUST_DECAY_SEVERE,
    TRUST_INITIAL,
    TRUST_RECOVERY_RATE,
)
from core.features import FeatureExtractor
from core.math_utils import (
    build_live_vector,
    compute_risk_label,
    cosine_similarity,
    shannon_entropy,
    temporal_zscore,
)
from core.profile import BehavioralProfile

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


@dataclass
class DecisionResult:
    """
    Per-action evaluation output from ABSAv2Engine.
    """

    decision: str
    trust_score: float
    bot_prob: float
    similarity: float
    entropy: float
    temporal_z: float
    anomalies: List[str] = field(default_factory=list)
    ramp_active: bool = False
    action_type: str = "read"
    timestamp: float = 0.0

    def is_flagged(self) -> bool:
        return bool(self.anomalies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "trust_score": self.trust_score,
            "bot_prob": self.bot_prob,
            "similarity": self.similarity,
            "entropy": self.entropy,
            "temporal_z": self.temporal_z,
            "anomalies": list(self.anomalies),
            "ramp_active": self.ramp_active,
            "action_type": self.action_type,
            "timestamp": self.timestamp,
        }


class ABSAv2Engine:
    """
    Minimal, production-safe ABSA evaluation engine used by the API layer.
    """

    def __init__(
        self,
        profile: BehavioralProfile,
        classifier: Optional[Any] = None,
        session_id: str = "",
        shadow_mode: bool = False,
    ) -> None:
        self.profile = profile
        self.classifier = classifier
        self.session_id = session_id or ""
        self.shadow_mode = shadow_mode

        self.trust_score: float = TRUST_INITIAL
        self.action_window: deque = deque(maxlen=BEHAVIOR_WINDOW_SIZE)
        self.result_history: List[DecisionResult] = []
        self.decision_log: List[str] = []
        self.anomaly_log: List[Dict[str, Any]] = []

        self._last_action_ts: Optional[float] = None
        self._started_at: float = time.time()
        self._ramp = RampUpManager()

    def evaluate_action(self, action: UserAction) -> DecisionResult:
        """
        Evaluate one action and update trust + profile state.
        """
        self.action_window.append(action)
        action_count = self.profile.vector_count
        ramp_active = self._ramp.is_warming_up(action_count)
        cosine_enabled = self._ramp.cosine_check_enabled(action_count)

        similarity = self._compute_similarity(cosine_enabled)
        entropy = shannon_entropy([a.action_type for a in self.action_window])
        temporal_z = self._compute_temporal_z(action, action_count)
        bot_prob = self._compute_bot_prob()

        ml_thresh = self._ramp.get_ml_threshold(action_count)
        temporal_limit = self._ramp.get_temporal_limit(action_count)
        sim_thresh = self.profile.adaptive_threshold

        anomalies: List[str] = []
        if cosine_enabled and similarity < sim_thresh:
            anomalies.append(ANOMALY_LOW_SIMILARITY)
        if len(self.action_window) >= ENTROPY_MIN_WINDOW:
            if entropy > ENTROPY_HIGH_THRESHOLD:
                anomalies.append(ANOMALY_HIGH_ENTROPY)
        if temporal_z > temporal_limit:
            anomalies.append(ANOMALY_TEMPORAL_DRIFT)
        if bot_prob >= ml_thresh:
            anomalies.append(ANOMALY_ML_BOT_DETECTED)
        if len(anomalies) >= 2:
            anomalies.append(ANOMALY_DUAL_SIGNAL_ESCALATION)

        self._apply_trust_update(action, anomalies, similarity, bot_prob, ml_thresh)
        decision = self._compute_decision()
        if self.shadow_mode:
            decision = DECISION_ALLOW

        result = DecisionResult(
            decision=decision,
            trust_score=round(self.trust_score, 3),
            bot_prob=max(0.0, min(1.0, float(bot_prob))),
            similarity=max(0.0, min(1.0, float(similarity))),
            entropy=max(0.0, min(1.0, float(entropy))),
            temporal_z=max(0.0, float(temporal_z)),
            anomalies=anomalies,
            ramp_active=ramp_active,
            action_type=action.action_type,
            timestamp=action.timestamp,
        )
        self.result_history.append(result)
        self.decision_log.append(result.decision)
        if anomalies:
            self.anomaly_log.append(
                {
                    "action_type": action.action_type,
                    "timestamp": action.timestamp,
                    "trust_score": result.trust_score,
                    "anomalies": anomalies,
                    "bot_prob": result.bot_prob,
                    "decision": result.decision,
                }
            )

        self._last_action_ts = action.timestamp
        return result

    def restore_trust(self, increment: float = 20.0) -> None:
        self.trust_score = min(100.0, self.trust_score + max(0.0, increment))

    def get_session_summary(self) -> Dict[str, Any]:
        total = len(self.result_history)
        allowed = self.decision_log.count(DECISION_ALLOW)
        challenged = self.decision_log.count(DECISION_CHALLENGE)
        blocked = self.decision_log.count(DECISION_BLOCK)

        top_features: List[Tuple[str, float]] = []
        if self.classifier and hasattr(self.classifier, "get_top_features"):
            try:
                top_features = list(self.classifier.get_top_features(5))
            except Exception:
                top_features = []

        ml_trained = bool(getattr(self.classifier, "is_trained", False))
        ml_samples = int(getattr(self.classifier, "n_samples", 0) or 0)

        return {
            "total_actions": total,
            "allowed": allowed,
            "challenged": challenged,
            "blocked": blocked,
            "anomaly_count": len(self.anomaly_log),
            "final_trust": round(self.trust_score, 2),
            "risk_rating": compute_risk_label(self.trust_score),
            "ml_trained": ml_trained,
            "ml_train_samples": ml_samples,
            "top_features": top_features,
            "session_duration_s": max(0.0, time.time() - self._started_at),
            "shadow_mode": bool(self.shadow_mode),
            "anomaly_log": list(self.anomaly_log),
        }

    def _compute_decision(self) -> str:
        if self.trust_score <= TRUST_BLOCK_THRESHOLD:
            return DECISION_BLOCK
        if self.trust_score <= TRUST_CHALLENGE_THRESHOLD:
            return DECISION_CHALLENGE
        return DECISION_ALLOW

    def _compute_similarity(self, cosine_enabled: bool) -> float:
        if not cosine_enabled:
            return 1.0
        profile_vec = self.profile.get_mean_vector()
        live_vec = build_live_vector(
            list(self.action_window),
            num_types=UserAction.type_count(),
        )
        return cosine_similarity(live_vec, profile_vec)

    def _compute_temporal_z(self, action: UserAction, action_count: int) -> float:
        if self._last_action_ts is None:
            return 0.0
        interval = max(0.0, action.timestamp - self._last_action_ts)
        mean_iv, std_iv = self.profile.get_timing_stats()
        base_limit = self._ramp.get_temporal_limit(action_count)
        z = temporal_zscore(interval, mean_iv, std_iv)
        return z if base_limit >= TEMPORAL_ZSCORE_LIMIT else max(0.0, z)

    def _compute_bot_prob(self) -> float:
        if not self.classifier:
            return 0.0
        if not getattr(self.classifier, "is_trained", False):
            return 0.0
        try:
            features = FeatureExtractor.extract(
                list(self.action_window),
                profile_intervals=self.profile.get_recent_intervals(20),
            )
            return float(self.classifier.predict_proba(features))
        except Exception:
            return 0.0

    def _apply_trust_update(
        self,
        action: UserAction,
        anomalies: List[str],
        similarity: float,
        bot_prob: float,
        ml_thresh: float,
    ) -> None:
        if anomalies:
            decay = TRUST_DECAY_MILD
            if ANOMALY_LOW_SIMILARITY in anomalies:
                if similarity < COSINE_MODERATE_THRESHOLD:
                    decay = max(decay, TRUST_DECAY_SEVERE)
                else:
                    decay = max(decay, TRUST_DECAY_MODERATE)
            if ANOMALY_ML_BOT_DETECTED in anomalies or bot_prob >= ml_thresh:
                decay = max(decay, TRUST_DECAY_AI_BOT)
            if len(anomalies) >= 2:
                decay = max(decay, TRUST_DECAY_MODERATE)
            if action.is_sensitive():
                decay = max(decay, TRUST_DECAY_MODERATE)
            self.trust_score = max(0.0, self.trust_score - decay)
            return

        recovery = self._ramp.get_trust_recovery_rate(self.profile.vector_count)
        self.trust_score = min(100.0, self.trust_score + max(recovery, TRUST_RECOVERY_RATE))
        self.profile.record_action(action)

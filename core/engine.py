"""
Zelvryn evaluation engine.

- ABSAv2Engine: per-session evaluator combining rule signals, label-free
  detectors (sequence, volume, peer group) and an optional ML classifier
  into a trust score and an ALLOW / CHALLENGE / BLOCK decision.
- HoeffdingTreeClassifier: incremental online classifier (river, with a
  Gaussian Naive Bayes fallback).
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
    ANOMALY_MACHINE_TIMING,
    ANOMALY_NEW_IDENTITY_SENSITIVE_BURST,
    MACHINE_TIMING_MAX_CV,
    MACHINE_TIMING_MIN_INTERVALS,
    NEW_IDENTITY_BURST_SHARE,
    NEW_IDENTITY_SENSITIVE_SHARE,
    ANOMALY_NEW_PRIVILEGED_ACTION,
    ANOMALY_PEER_DEVIATION,
    ANOMALY_UNUSUAL_SEQUENCE,
    ANOMALY_VOLUME_SPIKE,
    IDENTITY_KIND_HUMAN,
    PEER_MIN_IDENTITIES,
    PEER_MIN_WINDOWS,
    PEER_Z_LIMIT,
    SEQ_MIN_TRANSITIONS,
    SEQ_SURPRISE_MARGIN_BITS,
    VOLUME_FLOOR_DELETE,
    VOLUME_FLOOR_MULTIPLIER_AGENT,
    VOLUME_FLOOR_MULTIPLIER_SERVICE,
    VOLUME_FLOOR_EXPORT,
    VOLUME_FLOOR_READ,
    VOLUME_K_SIGMA,
    VOLUME_MIN_BUCKETS,
    VOLUME_RATIO,
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
    action_space_entropy,
    build_live_vector,
    compute_risk_label,
    cosine_similarity,
    shannon_entropy,
    temporal_zscore,
)
from core.profile import BehavioralProfile
from detectors.peer import PeerGroupBaseline

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



# First-ever use of these by an established identity is worth an alert.
# (Exports/deletes are covered by the volume detector instead.)
_PRIVILEGED_TYPES = frozenset({"admin", "config"})

# Single low-severity signals that may still be learned from when the session
# stays in good standing; otherwise a legitimate rare action would be flagged
# forever (it could never enter the baseline). Severe signals never are.
_LEARNABLE_ANOMALIES = frozenset({
    "NEW_PRIVILEGED_ACTION", "TEMPORAL_DRIFT", "UNUSUAL_SEQUENCE", "HIGH_ENTROPY",
})


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
    reasons: List[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)
    learned: bool = False
    # 20-dim feature vector of the window (kept off the wire and out of
    # stored history; attached to alerts so labels become training data)
    features: List[float] = field(default_factory=list, repr=False)

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
            "reasons": list(self.reasons),
            "signals": dict(self.signals),
            "learned": self.learned,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionResult":
        return cls(
            decision=d["decision"],
            trust_score=float(d["trust_score"]),
            bot_prob=float(d.get("bot_prob", 0.0)),
            similarity=float(d.get("similarity", 1.0)),
            entropy=float(d.get("entropy", 0.0)),
            temporal_z=float(d.get("temporal_z", 0.0)),
            anomalies=list(d.get("anomalies") or []),
            ramp_active=bool(d.get("ramp_active", False)),
            action_type=d.get("action_type", "read"),
            timestamp=float(d.get("timestamp", 0.0)),
            reasons=list(d.get("reasons") or []),
            signals=dict(d.get("signals") or {}),
            learned=bool(d.get("learned", False)),
        )


class ABSAv2Engine:
    """
    Per-session evaluation engine.

    Signals
    -------
    Rule signals (per identity):   cosine similarity of action mix, entropy,
                                   timing z-score.
    Label-free detectors:          action-sequence surprise (Markov chain),
                                   data-volume rate, first-ever privileged
                                   action, peer-group deviation.
    Supervised (optional):         automation classifier P(bot), for human
                                   identities only. Agents and services are
                                   automated by design, so "is it a bot?" is
                                   the wrong question for them; they are
                                   judged against their own baseline instead.

    Learning is gated: an action updates the identity's baseline only if it
    raised no anomaly AND the session is in good standing (trust above the
    challenge threshold). This limits slow profile poisoning.
    """

    RESULT_HISTORY_CAP = 500
    MIN_COSINE_WINDOW = 5
    HUMAN_MIN_TIMING_CV = 0.15
    ANOMALY_LOG_CAP = 200

    def __init__(
        self,
        profile: BehavioralProfile,
        classifier: Optional[Any] = None,
        session_id: str = "",
        shadow_mode: bool = False,
        identity_kind: Optional[str] = None,
        peer_baseline: Optional[PeerGroupBaseline] = None,
        peer_group: Optional[str] = None,
    ) -> None:
        self.profile = profile
        self.classifier = classifier
        self.session_id = session_id or ""
        self.shadow_mode = shadow_mode
        self.identity_kind = identity_kind or getattr(profile, "identity_kind", IDENTITY_KIND_HUMAN)
        self.peer_baseline = peer_baseline
        self.peer_group = peer_group or self.identity_kind
        self.peer_dirty = False

        self.trust_score: float = TRUST_INITIAL
        self.action_window: deque = deque(maxlen=BEHAVIOR_WINDOW_SIZE)
        self.result_history: List[DecisionResult] = []
        self.decision_log: List[str] = []
        self.anomaly_log: List[Dict[str, Any]] = []

        self._last_action_ts: Optional[float] = None
        self._started_at: float = time.time()
        self._ramp = RampUpManager()
        self._actions_since_peer_update: int = 0
        self._decision_counts: Dict[str, int] = {
            DECISION_ALLOW: 0, DECISION_CHALLENGE: 0, DECISION_BLOCK: 0,
        }

    @property
    def is_human(self) -> bool:
        return self.identity_kind == IDENTITY_KIND_HUMAN

    # ─────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────
    def evaluate_action(self, action: UserAction) -> DecisionResult:
        """
        Evaluate one action and update trust + profile state.
        """
        self.action_window.append(action)
        action_count = self.profile.vector_count
        ramp_active = self._ramp.is_warming_up(action_count)
        cosine_enabled = self._ramp.cosine_check_enabled(action_count)

        # A 1-4 action window is too small for a stable action-mix comparison
        cosine_enabled = cosine_enabled and len(self.action_window) >= self.MIN_COSINE_WINDOW
        similarity = self._compute_similarity(cosine_enabled)
        entropy = action_space_entropy(
            [a.action_type for a in self.action_window], UserAction.type_count()
        )
        temporal_z = self._compute_temporal_z(action, action_count)
        features = self._extract_features()
        bot_prob = self._compute_bot_prob(features) if self.is_human else 0.0

        ml_thresh = self._ramp.get_ml_threshold(action_count)
        temporal_limit = self._ramp.get_temporal_limit(action_count)
        sim_thresh = self.profile.adaptive_threshold

        anomalies: List[str] = []
        reasons: List[str] = []
        signals: Dict[str, Any] = {"identity_kind": self.identity_kind}

        # ── rule signals ─────────────────────────────────────────────────
        if cosine_enabled and similarity < sim_thresh:
            anomalies.append(ANOMALY_LOW_SIMILARITY)
            reasons.append(
                f"Action mix similarity {similarity:.2f} is below this identity's "
                f"threshold {sim_thresh:.2f}."
            )
        if len(self.action_window) >= ENTROPY_MIN_WINDOW and entropy > ENTROPY_HIGH_THRESHOLD:
            anomalies.append(ANOMALY_HIGH_ENTROPY)
            reasons.append(f"Actions are unusually random (entropy {entropy:.2f}).")
        if temporal_z > temporal_limit:
            anomalies.append(ANOMALY_TEMPORAL_DRIFT)
            reasons.append(
                f"Acting much faster than this identity's rhythm: the gap since the "
                f"previous action is {temporal_z:.1f} standard deviations shorter than usual."
            )
        if self.is_human and bot_prob >= ml_thresh:
            anomalies.append(ANOMALY_ML_BOT_DETECTED)
            reasons.append(f"Automation classifier estimates P(bot) = {bot_prob:.2f}.")

        # Undeclared automation: a "human" whose timing never varies
        if self.is_human:
            cv = self._window_interval_cv()
            if cv is not None and cv < MACHINE_TIMING_MAX_CV:
                anomalies.append(ANOMALY_MACHINE_TIMING)
                reasons.append(
                    f"Timing is machine-regular for an identity declared as human "
                    f"(variation {cv:.3f} over {len(self.action_window) - 1} gaps; "
                    f"people are rarely below {MACHINE_TIMING_MAX_CV})."
                )

        # Cold start: a brand-new identity that goes straight for sensitive data
        if (
            ramp_active
            and len(self.action_window) >= ENTROPY_MIN_WINDOW
            and features[14] >= NEW_IDENTITY_SENSITIVE_SHARE
            and features[17] >= NEW_IDENTITY_BURST_SHARE
        ):
            anomalies.append(ANOMALY_NEW_IDENTITY_SENSITIVE_BURST)
            reasons.append(
                f"New identity with no history: {features[14]:.0%} of recent actions "
                f"are delete/export/config/admin and {features[17]:.0%} came less "
                f"than half a second apart."
            )

        # ── label-free detectors ─────────────────────────────────────────
        seq = self.profile.sequence_model.assess(
            [a.action_type for a in self.action_window],
            min_transitions=SEQ_MIN_TRANSITIONS,
            margin_bits=SEQ_SURPRISE_MARGIN_BITS,
        )
        signals["sequence_excess_bits"] = round(seq.excess_bits, 3) if seq.active else None
        if seq.anomalous:
            anomalies.append(ANOMALY_UNUSUAL_SEQUENCE)
            reasons.extend(seq.reasons)

        vol = self.profile.volume.observe(
            action.action_type,
            action.metadata,
            action.timestamp,
            min_buckets=VOLUME_MIN_BUCKETS,
            k_sigma=VOLUME_K_SIGMA,
            ratio=VOLUME_RATIO,
            floors=self._volume_floors(),
        )
        if vol.metric:
            signals["volume"] = {
                "metric": vol.metric,
                "window_total": round(vol.live_total, 1),
                "limit": round(vol.limit, 1),
                "baseline_mature": vol.mature,
            }
        self._volume_overshoot = 0.0
        if vol.anomalous:
            anomalies.append(ANOMALY_VOLUME_SPIKE)
            reasons.extend(vol.reasons)
            self._volume_overshoot = vol.live_total / max(vol.limit, 1.0)

        if (
            action.action_type in _PRIVILEGED_TYPES
            and self.profile.is_mature(min_actions=self._ramp.ramp_actions)
            and self.profile.action_counts.get(action.action_type, 0) == 0
        ):
            anomalies.append(ANOMALY_NEW_PRIVILEGED_ACTION)
            reasons.append(
                f"First '{action.action_type}' action ever by this identity after "
                f"{self.profile.vector_count} normal actions."
            )

        if self.peer_baseline is not None and len(self.action_window) >= ENTROPY_MIN_WINDOW:
            # Stricter for established identities, which have their own baseline
            z_limit = PEER_Z_LIMIT if ramp_active else PEER_Z_LIMIT * 1.5
            peer = self.peer_baseline.assess(
                features,
                min_windows=PEER_MIN_WINDOWS,
                min_identities=PEER_MIN_IDENTITIES,
                z_limit=z_limit,
                group_name=self.peer_group,
            )
            signals["peer_score"] = round(peer.score, 3) if peer.active else None
            if peer.anomalous:
                anomalies.append(ANOMALY_PEER_DEVIATION)
                reasons.extend(peer.reasons)

        if len(anomalies) >= 2:
            anomalies.append(ANOMALY_DUAL_SIGNAL_ESCALATION)

        learned = self._apply_trust_update(action, anomalies, similarity, bot_prob, ml_thresh)
        if learned:
            self._learn_from_clean_action(action, features)

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
            reasons=reasons,
            signals=signals,
            learned=learned,
            features=[round(float(v), 6) for v in features],
        )
        self._record_result(result)
        self._last_action_ts = action.timestamp
        return result

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC HELPERS
    # ─────────────────────────────────────────────────────────────────────
    def restore_trust(self, increment: float = 20.0) -> None:
        self.trust_score = min(100.0, self.trust_score + max(0.0, increment))

    def current_decision(self) -> str:
        return self._compute_decision()

    def get_session_summary(self) -> Dict[str, Any]:
        top_features: List[Tuple[str, float]] = []
        if self.is_human and self.classifier and hasattr(self.classifier, "get_top_features"):
            try:
                top_features = list(self.classifier.get_top_features(5))
            except Exception:
                top_features = []

        ml_trained = bool(getattr(self.classifier, "is_trained", False))
        ml_samples = int(getattr(self.classifier, "n_samples", 0) or 0)

        return {
            "total_actions": sum(self._decision_counts.values()),
            "allowed": self._decision_counts[DECISION_ALLOW],
            "challenged": self._decision_counts[DECISION_CHALLENGE],
            "blocked": self._decision_counts[DECISION_BLOCK],
            "anomaly_count": len(self.anomaly_log),
            "final_trust": round(self.trust_score, 2),
            "risk_rating": compute_risk_label(self.trust_score),
            "ml_trained": ml_trained,
            "ml_train_samples": ml_samples,
            "top_features": top_features,
            "session_duration_s": max(0.0, time.time() - self._started_at),
            "shadow_mode": bool(self.shadow_mode),
            "identity_kind": self.identity_kind,
            "anomaly_log": list(self.anomaly_log),
        }

    # ─────────────────────────────────────────────────────────────────────
    # STATE (for storing sessions outside the process, e.g. Redis)
    # ─────────────────────────────────────────────────────────────────────
    def export_state(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "shadow_mode": self.shadow_mode,
            "identity_kind": self.identity_kind,
            "peer_group": self.peer_group,
            "trust_score": self.trust_score,
            "action_window": [a.to_dict() for a in self.action_window],
            "result_history": [r.to_dict() for r in self.result_history[-self.RESULT_HISTORY_CAP:]],
            "decision_counts": dict(self._decision_counts),
            "anomaly_log": self.anomaly_log[-self.ANOMALY_LOG_CAP:],
            "last_action_ts": self._last_action_ts,
            "started_at": self._started_at,
            "actions_since_peer_update": self._actions_since_peer_update,
            "ramp_actions": self._ramp.ramp_actions,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self.shadow_mode = bool(state.get("shadow_mode", self.shadow_mode))
        self.identity_kind = state.get("identity_kind", self.identity_kind)
        self.peer_group = state.get("peer_group", self.peer_group)
        self.trust_score = float(state.get("trust_score", TRUST_INITIAL))
        self.action_window.clear()
        for a in state.get("action_window") or []:
            self.action_window.append(UserAction.from_dict(a))
        self.result_history = [DecisionResult.from_dict(r) for r in state.get("result_history") or []]
        self.decision_log = [r.decision for r in self.result_history]
        counts = state.get("decision_counts") or {}
        for k in self._decision_counts:
            self._decision_counts[k] = int(counts.get(k, 0))
        self.anomaly_log = list(state.get("anomaly_log") or [])
        self._last_action_ts = state.get("last_action_ts")
        self._started_at = float(state.get("started_at", self._started_at))
        self._actions_since_peer_update = int(state.get("actions_since_peer_update", 0))
        ramp_actions = state.get("ramp_actions")
        if ramp_actions:
            self.set_ramp_actions(int(ramp_actions))

    def set_ramp_actions(self, ramp_actions: int) -> None:
        self._ramp.ramp_actions = max(1, int(ramp_actions))
        self._ramp._interp_start = int(
            self._ramp.ramp_actions * (1 - self._ramp.interpolation_fraction)
        )

    # ─────────────────────────────────────────────────────────────────────
    # INTERNALS
    # ─────────────────────────────────────────────────────────────────────
    def _record_result(self, result: DecisionResult) -> None:
        self.result_history.append(result)
        if len(self.result_history) > self.RESULT_HISTORY_CAP:
            self.result_history = self.result_history[-self.RESULT_HISTORY_CAP:]
        self.decision_log.append(result.decision)
        if len(self.decision_log) > self.RESULT_HISTORY_CAP:
            self.decision_log = self.decision_log[-self.RESULT_HISTORY_CAP:]
        self._decision_counts[result.decision] = self._decision_counts.get(result.decision, 0) + 1
        if result.anomalies:
            self.anomaly_log.append(
                {
                    "action_type": result.action_type,
                    "timestamp": result.timestamp,
                    "trust_score": result.trust_score,
                    "anomalies": list(result.anomalies),
                    "bot_prob": result.bot_prob,
                    "decision": result.decision,
                    "reasons": list(result.reasons),
                }
            )
            if len(self.anomaly_log) > self.ANOMALY_LOG_CAP:
                self.anomaly_log = self.anomaly_log[-self.ANOMALY_LOG_CAP:]

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
        # One-sided: pauses (coffee, meetings) are benign; acting much FASTER
        # than this identity's rhythm is the hijack / automation signal.
        if interval >= mean_iv:
            return 0.0
        # A tiny learned spread turns ordinary jitter into "drift": floor the
        # spread relative to the mean (people vary more than agents do).
        rel_floor = self.HUMAN_MIN_TIMING_CV if self.is_human else 0.25
        std_iv = max(std_iv, rel_floor * mean_iv)
        return temporal_zscore(interval, mean_iv, std_iv)

    def _volume_floors(self) -> Dict[str, float]:
        mult = 1.0
        if self.identity_kind == "service":
            mult = VOLUME_FLOOR_MULTIPLIER_SERVICE
        elif self.identity_kind == "agent":
            mult = VOLUME_FLOOR_MULTIPLIER_AGENT
        return {
            "read_units": VOLUME_FLOOR_READ * mult,
            "export_units": VOLUME_FLOOR_EXPORT * mult,
            "delete_units": VOLUME_FLOOR_DELETE * mult,
        }

    def _window_interval_cv(self) -> Optional[float]:
        acts = list(self.action_window)
        if len(acts) - 1 < MACHINE_TIMING_MIN_INTERVALS:
            return None
        gaps = [max(0.0, b.timestamp - a.timestamp) for a, b in zip(acts[:-1], acts[1:])]
        mean = sum(gaps) / len(gaps)
        if mean <= 0:
            return 0.0
        var = sum((g - mean) ** 2 for g in gaps) / (len(gaps) - 1)
        return math.sqrt(var) / mean

    def _extract_features(self) -> List[float]:
        try:
            return FeatureExtractor.extract(
                list(self.action_window),
                profile_intervals=self.profile.get_recent_intervals(20),
            )
        except Exception:
            return [0.0] * ML_FEATURE_DIM

    def _compute_bot_prob(self, features: List[float]) -> float:
        if not self.classifier:
            return 0.0
        if not getattr(self.classifier, "is_trained", False):
            return 0.0
        try:
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
    ) -> bool:
        """Update trust. Returns True when the action may be learned from."""
        if anomalies:
            decay = TRUST_DECAY_MILD
            if ANOMALY_LOW_SIMILARITY in anomalies:
                if similarity < COSINE_MODERATE_THRESHOLD:
                    decay = max(decay, TRUST_DECAY_SEVERE)
                else:
                    decay = max(decay, TRUST_DECAY_MODERATE)
            if ANOMALY_ML_BOT_DETECTED in anomalies:
                decay = max(decay, TRUST_DECAY_AI_BOT)
            if ANOMALY_NEW_IDENTITY_SENSITIVE_BURST in anomalies:
                decay = max(decay, TRUST_DECAY_SEVERE)
            if ANOMALY_VOLUME_SPIKE in anomalies:
                decay = max(decay, TRUST_DECAY_SEVERE)
                # Magnitude-aware: 3x or more over the limit challenges at once
                if getattr(self, "_volume_overshoot", 0.0) >= 3.0:
                    decay = max(decay, TRUST_DECAY_SEVERE + TRUST_DECAY_MODERATE)
            if ANOMALY_UNUSUAL_SEQUENCE in anomalies or ANOMALY_PEER_DEVIATION in anomalies:
                decay = max(decay, TRUST_DECAY_MODERATE)
            # NEW_PRIVILEGED_ACTION alone stays mild: worth an alert, not a block
            if len(anomalies) >= 2:
                decay = max(decay, TRUST_DECAY_MODERATE)
            if action.is_sensitive() and not (
                len(anomalies) == 1 and anomalies[0] == ANOMALY_NEW_PRIVILEGED_ACTION
            ):
                decay = max(decay, TRUST_DECAY_MODERATE)
            self.trust_score = max(0.0, self.trust_score - decay)
            return (
                len(anomalies) == 1
                and anomalies[0] in _LEARNABLE_ANOMALIES
                and self.trust_score > TRUST_CHALLENGE_THRESHOLD
            )

        recovery = self._ramp.get_trust_recovery_rate(self.profile.vector_count)
        self.trust_score = min(100.0, self.trust_score + max(recovery, TRUST_RECOVERY_RATE))
        # Anti-poisoning: only learn while the session is in good standing
        return self.trust_score > TRUST_CHALLENGE_THRESHOLD

    def _learn_from_clean_action(self, action: UserAction, features: List[float]) -> None:
        self.profile.record_action(action)
        self.profile.volume.commit_clean(action.action_type, action.metadata)

        if self.peer_baseline is None:
            return
        self._actions_since_peer_update += 1
        # Non-overlapping full windows keep peer samples roughly independent
        if (
            len(self.action_window) >= BEHAVIOR_WINDOW_SIZE
            and self._actions_since_peer_update >= BEHAVIOR_WINDOW_SIZE
        ):
            self.peer_baseline.update(features, self.profile.user_id)
            self._actions_since_peer_update = 0
            self.peer_dirty = True

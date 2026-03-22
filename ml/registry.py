"""
zelvryn/ml/registry.py
=======================
ModelRegistry — tracks all trained model versions, manages the active/canary
state machine, and provides atomic promotion and instant rollback.

Model lifecycle state machine
-------------------------------
Every model version moves through these states:

  TRAINED  → trained and persisted, not yet deployed
  CANARY   → receiving a small fraction of traffic (CANARY_TRAFFIC_PERCENT%)
             alongside the active model for A/B comparison
  ACTIVE   → the primary model receiving all (or most) traffic
  RETIRED  → superseded by a newer ACTIVE model, kept for rollback

State transitions:
  TRAINED  --promote_to_canary()--> CANARY
  CANARY   --promote()--> ACTIVE  (previous ACTIVE → RETIRED)
  CANARY   --demote()--> TRAINED  (back to staging if canary metrics are poor)
  ACTIVE   --rollback(v)--> previous RETIRED model becomes ACTIVE again

Canary deployment
-----------------
Canary mode routes CANARY_TRAFFIC_PERCENT% of sessions to the new model
and the rest to the active model. The API layer (api/dependencies.py)
selects which model to use per session by calling get_serving_model().

After enough canary sessions, the operator compares FAR/FRR/F1 between
active and canary in the dashboard, then calls promote() or demote().

Persistence
-----------
The registry state is stored in models/model_registry.json. On startup,
the API loads this file and restores the active/canary pointers. Model
weight files (.joblib, .pkl) are stored in models/ and referenced by path.

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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.classifier import RandomForestClassifier
from core.constants import CANARY_TRAFFIC_PERCENT

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "model_registry.json"


class ModelState(str, Enum):
    """Lifecycle state of a model version."""
    TRAINED  = "trained"    # persisted, not deployed
    CANARY   = "canary"     # receiving a fraction of traffic
    ACTIVE   = "active"     # primary production model
    RETIRED  = "retired"    # superseded, kept for rollback


@dataclass
class ModelVersion:
    """
    Metadata for a single trained model version.

    Attributes
    ----------
    version : str
        Unique version string (e.g., "rf_v3").
    model_path : str
        Path to the serialised model file.
    state : ModelState
    n_samples : int
        Training sample count.
    n_human : int
    n_bot : int
    cv_f1 : float
        Cross-validation F1 score at training time.
    cv_far : float
        Cross-validation FAR.
    cv_frr : float
        Cross-validation FRR.
    trained_at : float
        Unix timestamp of training.
    promoted_at : float or None
        Timestamp of last promotion.
    retired_at : float or None
        Timestamp of retirement.
    notes : str
        Optional human-readable notes (e.g., "post-adversarial hardening").
    """
    version:     str
    model_path:  str
    state:       str            = ModelState.TRAINED
    n_samples:   int            = 0
    n_human:     int            = 0
    n_bot:       int            = 0
    cv_f1:       float          = 0.0
    cv_far:      float          = 0.0
    cv_frr:      float          = 0.0
    trained_at:  float          = field(default_factory=time.time)
    promoted_at: Optional[float] = None
    retired_at:  Optional[float] = None
    notes:       str            = ""

    def is_deployable(self) -> bool:
        """Return True if the model is in a state that can be promoted."""
        return self.state in (ModelState.TRAINED, ModelState.CANARY)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version":     self.version,
            "model_path":  self.model_path,
            "state":       self.state,
            "n_samples":   self.n_samples,
            "n_human":     self.n_human,
            "n_bot":       self.n_bot,
            "cv_f1":       round(self.cv_f1, 4),
            "cv_far":      round(self.cv_far, 4),
            "cv_frr":      round(self.cv_frr, 4),
            "trained_at":  self.trained_at,
            "promoted_at": self.promoted_at,
            "retired_at":  self.retired_at,
            "notes":       self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelVersion":
        return cls(
            version     = d["version"],
            model_path  = d["model_path"],
            state       = d.get("state", ModelState.TRAINED),
            n_samples   = d.get("n_samples", 0),
            n_human     = d.get("n_human", 0),
            n_bot       = d.get("n_bot", 0),
            cv_f1       = d.get("cv_f1", 0.0),
            cv_far      = d.get("cv_far", 0.0),
            cv_frr      = d.get("cv_frr", 0.0),
            trained_at  = d.get("trained_at", 0.0),
            promoted_at = d.get("promoted_at"),
            retired_at  = d.get("retired_at"),
            notes       = d.get("notes", ""),
        )

    def __repr__(self) -> str:
        return (
            f"ModelVersion({self.version} "
            f"state={self.state} "
            f"F1={self.cv_f1:.3f} "
            f"FAR={self.cv_far:.3f})"
        )


class ModelRegistry:
    """
    Manages the lifecycle of all trained Zelvryn model versions.

    Provides:
      - Version registration (register)
      - Canary promotion (promote_to_canary)
      - Full production promotion (promote)
      - Emergency rollback (rollback)
      - Serving model selection (get_serving_model)
      - JSON-backed persistence (save / load)

    Parameters
    ----------
    registry_dir : str, optional
        Directory containing model files and the registry JSON.
        Default "models/".
    canary_traffic_percent : int, optional
        Percentage of sessions routed to the canary model. Default from
        constants.CANARY_TRAFFIC_PERCENT.
    """

    def __init__(
        self,
        registry_dir:            str = "models",
        canary_traffic_percent:  int = CANARY_TRAFFIC_PERCENT,
    ) -> None:
        self.registry_dir            = registry_dir
        self.canary_traffic_percent  = canary_traffic_percent

        self._versions:    Dict[str, ModelVersion]              = {}
        self._active_ver:  Optional[str]                        = None
        self._canary_ver:  Optional[str]                        = None
        self._loaded_models: Dict[str, RandomForestClassifier]  = {}

        Path(registry_dir).mkdir(parents=True, exist_ok=True)
        self._load_registry()

        logger.info(
            "ModelRegistry: dir=%s active=%s canary=%s versions=%d",
            registry_dir, self._active_ver, self._canary_ver,
            len(self._versions)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # REGISTRATION
    # ─────────────────────────────────────────────────────────────────────────

    def register(
        self,
        version:    str,
        model_path: str,
        n_samples:  int   = 0,
        n_human:    int   = 0,
        n_bot:      int   = 0,
        cv_f1:      float = 0.0,
        cv_far:     float = 0.0,
        cv_frr:     float = 0.0,
        notes:      str   = "",
    ) -> ModelVersion:
        """
        Register a newly trained model version in the registry.

        Parameters
        ----------
        version : str
            Unique version string (e.g., "rf_v4").
        model_path : str
            Path to the saved .joblib file.
        n_samples, n_human, n_bot : int
            Training dataset statistics.
        cv_f1, cv_far, cv_frr : float
            Cross-validation metrics.
        notes : str, optional
            Human-readable notes.

        Returns
        -------
        ModelVersion
        """
        if version in self._versions:
            logger.warning(
                "Version %s already registered. Overwriting.", version
            )

        mv = ModelVersion(
            version    = version,
            model_path = model_path,
            state      = ModelState.TRAINED,
            n_samples  = n_samples,
            n_human    = n_human,
            n_bot      = n_bot,
            cv_f1      = cv_f1,
            cv_far     = cv_far,
            cv_frr     = cv_frr,
            notes      = notes,
        )
        self._versions[version] = mv
        self._save_registry()

        logger.info(
            "Registered: %s (n=%d F1=%.3f FAR=%.3f FRR=%.3f)",
            version, n_samples, cv_f1, cv_far, cv_frr
        )
        return mv

    def register_from_training_result(
        self,
        result: Any,   # TrainingResult from ml/trainer.py
        notes: str = "",
    ) -> ModelVersion:
        """
        Register a model using a TrainingResult object from ModelTrainer.

        Parameters
        ----------
        result : TrainingResult
        notes : str, optional

        Returns
        -------
        ModelVersion
        """
        cv = result.cv_metrics
        return self.register(
            version    = result.version,
            model_path = result.model_path,
            n_samples  = result.n_samples,
            n_human    = result.n_human,
            n_bot      = result.n_bot,
            cv_f1      = cv.get("f1", 0.0),
            cv_far     = cv.get("far", 0.0),
            cv_frr     = cv.get("frr", 0.0),
            notes      = notes,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PROMOTION
    # ─────────────────────────────────────────────────────────────────────────

    def promote_to_canary(self, version: str) -> ModelVersion:
        """
        Promote a TRAINED model to CANARY status.

        The canary model will start receiving canary_traffic_percent% of
        traffic via get_serving_model(). Any existing CANARY model is
        demoted back to TRAINED.

        Parameters
        ----------
        version : str

        Returns
        -------
        ModelVersion

        Raises
        ------
        KeyError
            If version is not registered.
        ValueError
            If version is not in a promotable state.
        """
        mv = self._get_version(version)
        if not mv.is_deployable():
            raise ValueError(
                f"Version {version} is in state {mv.state!r} "
                f"and cannot be promoted to canary."
            )

        # Demote existing canary
        if self._canary_ver and self._canary_ver != version:
            self._versions[self._canary_ver].state = ModelState.TRAINED
            logger.info("Previous canary %s demoted to TRAINED.", self._canary_ver)

        mv.state       = ModelState.CANARY
        mv.promoted_at = time.time()
        self._canary_ver = version
        self._save_registry()

        logger.info(
            "Canary promoted: %s (traffic=%d%%)",
            version, self.canary_traffic_percent
        )
        return mv

    def promote(self, version: str, notes: str = "") -> ModelVersion:
        """
        Promote a model to ACTIVE status (full production deployment).

        The current ACTIVE model is retired. Any CANARY model (if different
        from version) is also retired.

        Parameters
        ----------
        version : str
            Must be in TRAINED or CANARY state.
        notes : str, optional
            Reason for promotion (written to the ModelVersion notes field).

        Returns
        -------
        ModelVersion

        Raises
        ------
        KeyError, ValueError
        """
        mv = self._get_version(version)
        if not mv.is_deployable() and mv.state != ModelState.CANARY:
            raise ValueError(
                f"Cannot promote {version}: state is {mv.state!r}."
            )

        now = time.time()

        # Retire current active model
        if self._active_ver and self._active_ver != version:
            old_active = self._versions[self._active_ver]
            old_active.state      = ModelState.RETIRED
            old_active.retired_at = now
            logger.info("Retired previous active model: %s", self._active_ver)

        # Retire canary if it's a different version
        if self._canary_ver and self._canary_ver != version:
            self._versions[self._canary_ver].state = ModelState.RETIRED
            self._versions[self._canary_ver].retired_at = now

        mv.state        = ModelState.ACTIVE
        mv.promoted_at  = now
        if notes:
            mv.notes = notes
        self._active_ver = version
        self._canary_ver = None
        self._save_registry()

        # Preload model into memory cache
        self._ensure_loaded(version)

        logger.info("Promoted to ACTIVE: %s", version)
        return mv

    def demote_canary(self) -> Optional[str]:
        """
        Demote the current canary model back to TRAINED.

        Called when canary metrics are worse than the active model.

        Returns
        -------
        str or None
            Version that was demoted, or None if no canary existed.
        """
        if not self._canary_ver:
            logger.info("No canary to demote.")
            return None

        version = self._canary_ver
        self._versions[version].state = ModelState.TRAINED
        self._canary_ver = None
        self._save_registry()

        logger.info("Canary demoted to TRAINED: %s", version)
        return version

    def rollback(self, target_version: Optional[str] = None) -> ModelVersion:
        """
        Roll back to a previous RETIRED model version (or to target_version).

        The current ACTIVE model is retired. The target model is promoted
        to ACTIVE immediately, without a canary phase.

        Parameters
        ----------
        target_version : str, optional
            Specific version to roll back to. If None, rolls back to the
            most recently retired model.

        Returns
        -------
        ModelVersion

        Raises
        ------
        ValueError
            If no suitable rollback target is found.
        """
        if target_version:
            target = self._get_version(target_version)
            if target.state not in (ModelState.RETIRED, ModelState.TRAINED):
                raise ValueError(
                    f"Cannot rollback to {target_version}: state={target.state!r}. "
                    f"Only RETIRED or TRAINED models can be rolled back to."
                )
        else:
            # Find most recently retired version
            retired = [
                mv for mv in self._versions.values()
                if mv.state == ModelState.RETIRED
                and mv.version != self._active_ver
            ]
            if not retired:
                raise ValueError("No retired models available for rollback.")
            target = max(retired, key=lambda mv: mv.retired_at or 0)

        logger.warning(
            "ROLLBACK initiated: active=%s → target=%s",
            self._active_ver, target.version
        )
        return self.promote(target.version, notes=f"rollback from {self._active_ver}")

    # ─────────────────────────────────────────────────────────────────────────
    # SERVING MODEL SELECTION
    # ─────────────────────────────────────────────────────────────────────────

    def get_serving_model(
        self,
        session_id: str = "",
    ) -> Optional[RandomForestClassifier]:
        """
        Return the classifier to use for a given session.

        Applies canary routing: canary_traffic_percent% of sessions are
        routed to the canary model based on a deterministic hash of session_id
        (so the same session always gets the same model within a session).

        Parameters
        ----------
        session_id : str
            Used for deterministic canary routing. Empty string → random.

        Returns
        -------
        RandomForestClassifier or None
            None if no active model is available.
        """
        # Canary routing
        if self._canary_ver and self.canary_traffic_percent > 0:
            if session_id:
                # Deterministic: hash session_id to decide routing
                h = hash(session_id) % 100
                use_canary = h < self.canary_traffic_percent
            else:
                use_canary = random.random() < (self.canary_traffic_percent / 100.0)

            if use_canary:
                return self._ensure_loaded(self._canary_ver)

        # Active model
        if self._active_ver:
            return self._ensure_loaded(self._active_ver)

        logger.warning("No active model in registry — returning None.")
        return None

    def get_active_model(self) -> Optional[RandomForestClassifier]:
        """Return the active model (bypasses canary routing)."""
        if not self._active_ver:
            return None
        return self._ensure_loaded(self._active_ver)

    def get_canary_model(self) -> Optional[RandomForestClassifier]:
        """Return the canary model, or None if no canary is deployed."""
        if not self._canary_ver:
            return None
        return self._ensure_loaded(self._canary_ver)

    # ─────────────────────────────────────────────────────────────────────────
    # INSPECTION
    # ─────────────────────────────────────────────────────────────────────────

    def list_versions(self) -> List[ModelVersion]:
        """Return all registered versions sorted by training time (newest first)."""
        return sorted(
            self._versions.values(),
            key=lambda mv: mv.trained_at,
            reverse=True,
        )

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Return a specific ModelVersion by name, or None."""
        return self._versions.get(version)

    def active_version(self) -> Optional[str]:
        """Return the active version string."""
        return self._active_ver

    def canary_version(self) -> Optional[str]:
        """Return the canary version string."""
        return self._canary_ver

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for the health endpoint."""
        versions_by_state: Dict[str, List[str]] = {}
        for mv in self._versions.values():
            versions_by_state.setdefault(mv.state, []).append(mv.version)

        active_mv = self._versions.get(self._active_ver) if self._active_ver else None
        canary_mv = self._versions.get(self._canary_ver) if self._canary_ver else None

        return {
            "active_version":         self._active_ver,
            "canary_version":         self._canary_ver,
            "canary_traffic_percent": self.canary_traffic_percent,
            "total_versions":         len(self._versions),
            "versions_by_state":      versions_by_state,
            "active_metrics": {
                "cv_f1":  round(active_mv.cv_f1,  4) if active_mv else None,
                "cv_far": round(active_mv.cv_far, 4) if active_mv else None,
                "cv_frr": round(active_mv.cv_frr, 4) if active_mv else None,
            } if active_mv else None,
            "canary_metrics": {
                "cv_f1":  round(canary_mv.cv_f1,  4) if canary_mv else None,
                "cv_far": round(canary_mv.cv_far, 4) if canary_mv else None,
                "cv_frr": round(canary_mv.cv_frr, 4) if canary_mv else None,
            } if canary_mv else None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _save_registry(self) -> None:
        """Write the registry state to models/model_registry.json."""
        path = os.path.join(self.registry_dir, REGISTRY_FILENAME)
        data = {
            "active_version":        self._active_ver,
            "canary_version":        self._canary_ver,
            "canary_traffic_percent": self.canary_traffic_percent,
            "saved_at":              time.time(),
            "versions":              {
                v: mv.to_dict()
                for v, mv in self._versions.items()
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error("Failed to save registry: %s", e)

    def _load_registry(self) -> None:
        """Load registry state from models/model_registry.json if it exists."""
        path = os.path.join(self.registry_dir, REGISTRY_FILENAME)
        if not os.path.exists(path):
            logger.info("No existing registry found at %s — starting fresh.", path)
            return

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            self._active_ver = data.get("active_version")
            self._canary_ver = data.get("canary_version")

            for vstr, vdata in data.get("versions", {}).items():
                self._versions[vstr] = ModelVersion.from_dict(vdata)

            logger.info(
                "Registry loaded: %d versions, active=%s, canary=%s",
                len(self._versions), self._active_ver, self._canary_ver
            )
        except Exception as e:
            logger.error("Failed to load registry: %s", e, exc_info=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL LOADING CACHE
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_loaded(
        self,
        version: str,
    ) -> Optional[RandomForestClassifier]:
        """
        Return the loaded classifier for a version, loading from disk if needed.

        Parameters
        ----------
        version : str

        Returns
        -------
        RandomForestClassifier or None
        """
        if version in self._loaded_models:
            return self._loaded_models[version]

        mv = self._versions.get(version)
        if not mv:
            logger.error("Version %s not in registry.", version)
            return None

        if not os.path.exists(mv.model_path):
            logger.error(
                "Model file not found: %s for version %s",
                mv.model_path, version
            )
            return None

        try:
            clf = RandomForestClassifier.load_model(mv.model_path)
            self._loaded_models[version] = clf
            logger.info("Loaded model into memory: %s", version)
            return clf
        except Exception as e:
            logger.error("Failed to load model %s: %s", version, e)
            return None

    def _get_version(self, version: str) -> ModelVersion:
        """Return a ModelVersion or raise KeyError."""
        if version not in self._versions:
            raise KeyError(
                f"Version {version!r} not registered. "
                f"Available: {list(self._versions.keys())}"
            )
        return self._versions[version]

    def __repr__(self) -> str:
        return (
            f"ModelRegistry("
            f"dir={self.registry_dir!r}, "
            f"active={self._active_ver!r}, "
            f"canary={self._canary_ver!r}, "
            f"versions={len(self._versions)})"
        )

"""
zelvryn/ingestion/shadow_mode.py
==================================
ShadowModeLogger — runs Zelvryn's full evaluation pipeline on live sessions
without enforcing any CHALLENGE or BLOCK decisions.

Shadow mode is the recommended first deployment phase. It allows Zelvryn to:
  1. Observe real human behavioural patterns in the specific production context
  2. Log what decisions WOULD have been made under full enforcement
  3. Accumulate genuine labeled training data for the ML classifier
  4. Validate that false positive rates are acceptable before enforcement begins

Deployment workflow
-------------------
  Week 1-4 : Deploy with SHADOW_MODE_ENABLED=true. All responses to the
              application are ALLOW regardless of what Zelvryn detects.
  Week 5   : Review shadow logs. Check false positive rate across normal sessions.
              Tune thresholds if needed via constants.py environment variables.
  Week 6   : Export shadow data via export_labeled_sessions().
              Use it to retrain the Random Forest on real-world data.
  Week 7   : Set SHADOW_MODE_ENABLED=false. Enforcement begins.

What shadow mode logs
---------------------
Every evaluate_action() call produces a ShadowLogEntry containing:
  - The real decision (what WOULD have been enforced)
  - Trust score at evaluation time
  - All signal values (similarity, entropy, temporal Z-score, P(bot))
  - All triggered anomaly codes
  - The 20-dimensional feature vector
  - Heuristic label (0=human, 1=bot) for training

This log is stored in-memory during the session and written to disk
(data/shadow_exports/) when the session ends or on explicit export_session().

Privacy note
-----------
Shadow logs contain feature vectors (statistical abstractions), not raw action
content. Resource IDs are already hashed in UserAction. User IDs in exported
files should be pseudonymised before sharing outside the security team.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.action import UserAction
from core.profile import BehavioralProfile
from core.features import FeatureExtractor
from core.engine import ABSAv2Engine, DecisionResult
from core.classifier import RandomForestClassifier
from core.constants import (
    SHADOW_MODE_ENABLED,
    ML_FEATURE_DIM,
)

logger = logging.getLogger(__name__)

# Default directory for shadow session exports
DEFAULT_EXPORT_DIR = "data/shadow_exports"

# Maximum in-memory log entries before auto-flush to disk
MAX_IN_MEMORY_ENTRIES = 5000


@dataclass
class ShadowLogEntry:
    """
    A single logged evaluation result from a shadow mode session.

    Attributes
    ----------
    session_id : str
    user_id : str
    action_type : str
    timestamp : float
    real_decision : str
        The decision Zelvryn WOULD have enforced (ALLOW/CHALLENGE/BLOCK).
    trust_score : float
    bot_prob : float
    similarity : float
    entropy : float
    temporal_z : float
    anomalies : list of str
    feature_vector : list of float
        The 20-dimensional feature vector for this action window.
    heuristic_label : int
        0 = human (no anomalies), 1 = bot (any anomaly detected).
        Used for ML training; can be overridden by analyst review.
    verified_label : int or None
        Set by analyst review to override the heuristic label.
    ramp_active : bool
    """
    session_id:      str
    user_id:         str
    action_type:     str
    timestamp:       float
    real_decision:   str
    trust_score:     float
    bot_prob:        float
    similarity:      float
    entropy:         float
    temporal_z:      float
    anomalies:       List[str]           = field(default_factory=list)
    feature_vector:  List[float]         = field(default_factory=list)
    heuristic_label: int                 = 0
    verified_label:  Optional[int]       = None
    ramp_active:     bool                = False

    def effective_label(self) -> int:
        """Return verified_label if set, otherwise heuristic_label."""
        return self.verified_label if self.verified_label is not None else self.heuristic_label

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["effective_label"] = self.effective_label()
        return d

    def to_training_pair(self) -> Tuple[List[float], int]:
        """Return (feature_vector, label) for ML training."""
        return (self.feature_vector, self.effective_label())


class ShadowModeLogger:
    """
    Runs Zelvryn's full evaluation stack in observation-only mode.

    Wraps ABSAv2Engine to:
      1. Force all returned decisions to ALLOW (no enforcement)
      2. Record every evaluation result with its feature vector
      3. Provide export utilities for training data extraction

    Parameters
    ----------
    engine : ABSAv2Engine
        The evaluation engine to run in shadow mode. The engine's own
        shadow_mode flag will be overridden to True.
    export_dir : str, optional
        Directory for writing session export files. Default data/shadow_exports/.
    auto_flush_threshold : int, optional
        Number of in-memory log entries before auto-flushing to disk.
        Default MAX_IN_MEMORY_ENTRIES (5000).
    """

    def __init__(
        self,
        engine: ABSAv2Engine,
        export_dir: str = DEFAULT_EXPORT_DIR,
        auto_flush_threshold: int = MAX_IN_MEMORY_ENTRIES,
    ) -> None:
        self.engine               = engine
        self.export_dir           = export_dir
        self.auto_flush_threshold = auto_flush_threshold

        # Override engine's shadow mode flag
        self.engine.shadow_mode = True

        # In-memory log
        self._log: List[ShadowLogEntry] = []

        # Session counters
        self._total_evaluated: int   = 0
        self._total_flushed:   int   = 0
        self._session_start:   float = time.time()

        Path(export_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "ShadowModeLogger started for user=%s session=%s export_dir=%s",
            engine.profile.user_id, engine.session_id, export_dir
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMARY INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_silently(self, action: UserAction) -> DecisionResult:
        """
        Evaluate an action through the full Zelvryn engine in shadow mode.

        The engine's result contains the REAL decision (what would be enforced).
        This method wraps it to:
          1. Always return ALLOW to the caller (zero enforcement impact)
          2. Log the real decision and feature vector for training data

        Parameters
        ----------
        action : UserAction
            The action to evaluate.

        Returns
        -------
        DecisionResult
            Always has decision="ALLOW" regardless of what the engine detected.
            The real_decision is stored in the ShadowLogEntry.
        """
        # Extract feature vector BEFORE engine evaluation (same window state)
        window_snapshot = list(self.engine.action_window) + [action]
        features = FeatureExtractor.extract(
            window_snapshot,
            profile_intervals=self.engine.profile.get_recent_intervals(20)
        )

        # Run full engine evaluation
        result = self.engine.evaluate_action(action)

        # Determine heuristic label
        heuristic_label = 1 if result.is_flagged() else 0

        # Create log entry
        entry = ShadowLogEntry(
            session_id      = self.engine.session_id,
            user_id         = self.engine.profile.user_id,
            action_type     = action.action_type,
            timestamp       = action.timestamp,
            real_decision   = result.decision,  # the real (enforced) decision
            trust_score     = result.trust_score,
            bot_prob        = result.bot_prob,
            similarity      = result.similarity,
            entropy         = result.entropy,
            temporal_z      = result.temporal_z,
            anomalies       = list(result.anomalies),
            feature_vector  = features,
            heuristic_label = heuristic_label,
            ramp_active     = result.ramp_active,
        )
        self._log.append(entry)
        self._total_evaluated += 1

        # Auto-flush if in-memory log is too large
        if len(self._log) >= self.auto_flush_threshold:
            self._flush_to_disk(partial=True)

        # Return ALLOW regardless of real decision (shadow mode — no enforcement)
        from copy import copy
        shadow_result = copy(result)
        shadow_result.decision = "ALLOW"
        return shadow_result

    def set_verified_label(
        self,
        timestamp: float,
        label: int,
        tolerance_seconds: float = 1.0,
    ) -> bool:
        """
        Override the heuristic label for a log entry with an analyst-verified label.

        Finds the log entry closest in time to `timestamp` and sets its
        verified_label. This is called from the analyst review queue when
        a security analyst confirms or overrides a detection.

        Parameters
        ----------
        timestamp : float
            Approximate timestamp of the entry to update.
        label : int
            0 = human, 1 = bot.
        tolerance_seconds : float
            Maximum time difference to accept as a match.

        Returns
        -------
        bool
            True if a matching entry was found and updated.
        """
        best_entry = None
        best_diff  = float("inf")

        for entry in self._log:
            diff = abs(entry.timestamp - timestamp)
            if diff < best_diff:
                best_diff  = diff
                best_entry = entry

        if best_entry and best_diff <= tolerance_seconds:
            best_entry.verified_label = label
            logger.debug(
                "Verified label set: session=%s ts=%.3f label=%d",
                self.engine.session_id, timestamp, label
            )
            return True

        logger.warning(
            "No log entry found within %.2fs of timestamp %.3f",
            tolerance_seconds, timestamp
        )
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def export_session(
        self,
        format: str = "jsonl",
        pseudonymise: bool = True,
    ) -> str:
        """
        Write the current session's shadow log to disk.

        Parameters
        ----------
        format : str, optional
            Output format: "jsonl" (one JSON object per line) or "csv".
            Default "jsonl".
        pseudonymise : bool, optional
            If True, replace user_id and session_id with SHA-256 hashes before
            writing. Default True (recommended for logs shared with ML team).

        Returns
        -------
        str
            Path of the written file.
        """
        if not self._log:
            logger.info("Shadow log is empty — nothing to export.")
            return ""

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        safe_session  = self.engine.session_id.replace("/", "_")[:20]
        filename      = f"shadow_{safe_session}_{timestamp_str}.{format}"
        filepath      = os.path.join(self.export_dir, filename)

        entries = self._prepare_export_entries(pseudonymise)

        if format == "jsonl":
            self._write_jsonl(filepath, entries)
        elif format == "csv":
            self._write_csv(filepath, entries)
        else:
            raise ValueError(f"Unsupported export format: {format!r}. Use 'jsonl' or 'csv'.")

        logger.info(
            "Shadow session exported: %s (%d entries, format=%s, pseudo=%s)",
            filepath, len(entries), format, pseudonymise
        )
        return filepath

    def export_training_dataset(
        self,
        output_path: str,
        verified_only: bool = False,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Extract (X, y) training arrays from the shadow log.

        Parameters
        ----------
        output_path : str
            Path to write the dataset as JSON (for reproducibility).
        verified_only : bool, optional
            If True, only includes entries with analyst-verified labels.

        Returns
        -------
        tuple of (list of list, list of int)
            (X, y) where X[i] is a 20-dim feature vector and y[i] is its label.
        """
        entries = self._log
        if verified_only:
            entries = [e for e in entries if e.verified_label is not None]

        if not entries:
            logger.warning("No %s entries in shadow log for training export.",
                           "verified" if verified_only else "")
            return ([], [])

        X = [e.feature_vector for e in entries]
        y = [e.effective_label() for e in entries]

        dataset = {
            "session_id":    self.engine.session_id,
            "user_id":       self.engine.profile.user_id,
            "exported_at":   time.time(),
            "n_samples":     len(X),
            "n_human":       y.count(0),
            "n_bot":         y.count(1),
            "verified_only": verified_only,
            "feature_names": FeatureExtractor.feature_names(),
            "X":             X,
            "y":             y,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        logger.info(
            "Training dataset exported: %s (%d samples: %d human, %d bot)",
            output_path, len(X), y.count(0), y.count(1)
        )
        return (X, y)

    # ─────────────────────────────────────────────────────────────────────────
    # STATISTICS AND REPORTING
    # ─────────────────────────────────────────────────────────────────────────

    def get_shadow_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the shadow session's evaluation results.

        Returns
        -------
        dict
            Counts of real decisions, anomaly rates, avg trust score, etc.
        """
        if not self._log:
            return {"total_evaluated": 0, "message": "No evaluations logged."}

        real_decisions = [e.real_decision for e in self._log]
        labels         = [e.heuristic_label for e in self._log]
        bot_probs      = [e.bot_prob for e in self._log]
        trust_scores   = [e.trust_score for e in self._log]

        avg_trust = sum(trust_scores) / len(trust_scores)
        avg_bot_p = sum(bot_probs) / len(bot_probs)

        return {
            "session_id":         self.engine.session_id,
            "user_id":            self.engine.profile.user_id,
            "total_evaluated":    self._total_evaluated,
            "real_allow":         real_decisions.count("ALLOW"),
            "real_challenge":     real_decisions.count("CHALLENGE"),
            "real_block":         real_decisions.count("BLOCK"),
            "heuristic_human":    labels.count(0),
            "heuristic_bot":      labels.count(1),
            "verified_labels":    sum(1 for e in self._log if e.verified_label is not None),
            "avg_trust_score":    round(avg_trust, 2),
            "avg_bot_prob":       round(avg_bot_p, 4),
            "session_duration_s": round(time.time() - self._session_start, 1),
            "log_size":           len(self._log),
        }

    @property
    def log_entries(self) -> List[ShadowLogEntry]:
        """Read-only access to all log entries."""
        return list(self._log)

    @property
    def entry_count(self) -> int:
        """Total number of logged entries."""
        return len(self._log)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _flush_to_disk(self, partial: bool = False) -> None:
        """Write current in-memory log to disk and clear the in-memory buffer."""
        if not self._log:
            return

        self._total_flushed += len(self._log)
        self.export_session(format="jsonl", pseudonymise=True)

        if partial:
            self._log.clear()
            logger.debug(
                "Shadow log auto-flushed: %d total entries persisted.",
                self._total_flushed
            )

    def _prepare_export_entries(self, pseudonymise: bool) -> List[Dict[str, Any]]:
        """Convert log entries to dicts, optionally pseudonymising IDs."""
        import hashlib
        entries = []
        for entry in self._log:
            d = entry.to_dict()
            if pseudonymise:
                d["user_id"]    = hashlib.sha256(
                    entry.user_id.encode()).hexdigest()[:16]
                d["session_id"] = hashlib.sha256(
                    entry.session_id.encode()).hexdigest()[:16]
            entries.append(d)
        return entries

    @staticmethod
    def _write_jsonl(filepath: str, entries: List[Dict]) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _write_csv(filepath: str, entries: List[Dict]) -> None:
        if not entries:
            return
        fieldnames = [k for k in entries[0].keys() if k != "feature_vector"]
        # Add feature vector columns as f_0, f_1, ..., f_19
        fv_cols = [f"f_{i}" for i in range(ML_FEATURE_DIM)]
        fieldnames.extend(fv_cols)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                row = {k: v for k, v in entry.items() if k != "feature_vector"}
                fv  = entry.get("feature_vector", [0.0] * ML_FEATURE_DIM)
                for i, col in enumerate(fv_cols):
                    row[col] = fv[i] if i < len(fv) else 0.0
                # Serialise list fields to strings for CSV
                row["anomalies"] = "|".join(row.get("anomalies", []))
                writer.writerow(row)

    # ─────────────────────────────────────────────────────────────────────────
    # Dunder methods
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ShadowModeLogger("
            f"user={self.engine.profile.user_id!r}, "
            f"session={self.engine.session_id!r}, "
            f"entries={len(self._log)}, "
            f"export_dir={self.export_dir!r})"
        )
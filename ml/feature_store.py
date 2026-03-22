"""
FeatureStore — persists labeled (feature_vector, label) pairs for offline
ML training. Provides the training data pipeline between the live evaluation
engine and the ModelTrainer.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.constants import ML_FEATURE_DIM, ML_MIN_TRAIN_SAMPLES

logger = logging.getLogger(__name__)


class LabelSource(str, Enum):
    """Origin classification for training labels."""
    LIVE    = "live"     # heuristic label from live session engine
    SHADOW  = "shadow"   # heuristic label from shadow mode session
    CORPUS  = "corpus"   # label from external attack corpus (CICIDS, CTU-13)
    ANALYST = "analyst"  # manually verified by security analyst


@dataclass
class FeatureRecord:
    """
    A single labeled training record in the feature store.
    """
    user_id:    str
    session_id: str
    vector:     List[float]
    label:      int
    verified:   bool        = False
    source:     str         = LabelSource.LIVE
    created_at: float       = field(default_factory=time.time)
    record_id:  str         = field(default_factory=lambda: f"rec_{time.time_ns()}")

    def is_high_quality(self) -> bool:
        """Return True if this is a verified or corpus record."""
        return self.verified or self.source in (LabelSource.CORPUS, LabelSource.ANALYST)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id":  self.record_id,
            "user_id":    self.user_id,
            "session_id": self.session_id,
            "vector":     self.vector,
            "label":      self.label,
            "verified":   self.verified,
            "source":     self.source,
            "created_at": self.created_at,
        }

    def to_training_pair(self) -> Tuple[List[float], int]:
        return (self.vector, self.label)


class FeatureStore:
    """
    Persists labeled feature vectors for ML training.
    """

    def __init__(
        self,
        db_url:      Optional[str] = None,
        max_records: int           = 100_000,
    ) -> None:
        self._db_url     = db_url
        self._max_records = max_records

        # In-memory storage (always initialised; replaced by DB queries in prod)
        self._records: List[FeatureRecord] = []

        # Statistics
        self._total_saved:     int = 0
        self._total_retrieved: int = 0

        if db_url:
            logger.info(
                "FeatureStore: PostgreSQL backend configured. "
                "DB writes require async context (use save_async)."
            )
        else:
            logger.info("FeatureStore: Using in-memory backend.")

    # PRIMARY INTERFACE
    def save(
        self,
        user_id:    str,
        vector:     List[float],
        label:      int,
        session_id: str        = "",
        verified:   bool       = False,
        source:     str        = LabelSource.LIVE,
    ) -> FeatureRecord:
        """
        Save a labeled feature vector to the store.
        """
        if len(vector) != ML_FEATURE_DIM:
            raise ValueError(
                f"Feature vector must have {ML_FEATURE_DIM} dimensions, "
                f"got {len(vector)}."
            )
        if label not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {label!r}.")

        record = FeatureRecord(
            user_id    = user_id,
            session_id = session_id,
            vector     = list(vector),
            label      = label,
            verified   = verified,
            source     = source,
        )
        self._records.append(record)
        self._total_saved += 1

        # Evict oldest records if at capacity
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

        return record

    def save_batch(
        self,
        records_data: List[Dict[str, Any]],
    ) -> List[FeatureRecord]:
        """
        Save a batch of records.
        """
        saved = []
        for rd in records_data:
            try:
                rec = self.save(
                    user_id    = rd["user_id"],
                    vector     = rd["vector"],
                    label      = rd["label"],
                    session_id = rd.get("session_id", ""),
                    verified   = rd.get("verified", False),
                    source     = rd.get("source", LabelSource.LIVE),
                )
                saved.append(rec)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping invalid record: %s", e)
        return saved

    def verify_label(
        self,
        record_id: str,
        new_label: int,
    ) -> bool:
        """
        Set a record's label as analyst-verified.
        """
        for rec in self._records:
            if rec.record_id == record_id:
                rec.label    = new_label
                rec.verified = True
                rec.source   = LabelSource.ANALYST
                logger.info(
                    "Record verified: id=%s label=%d", record_id, new_label
                )
                return True
        logger.warning("Record not found for verification: %s", record_id)
        return False

    # RETRIEVAL
    def get_training_batch(
        self,
        min_samples:    int  = ML_MIN_TRAIN_SAMPLES,
        verified_only:  bool = False,
        max_samples:    Optional[int] = None,
        balance:        bool = True,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Retrieve a labeled (X, y) training batch from the store.
        tuple of (list of list, list of int)
        """
        records = self._filter_records(verified_only=verified_only)

        if len(records) < min_samples:
            logger.debug(
                "Insufficient training data: %d < %d required.",
                len(records), min_samples
            )
            return ([], [])

        if max_samples and len(records) > max_samples:
            records = random.sample(records, max_samples)

        if balance:
            records = self._balance_records(records)

        X = [r.vector for r in records]
        y = [r.label  for r in records]

        self._total_retrieved += len(X)
        logger.info(
            "Training batch retrieved: %d samples (%d human, %d bot, "
            "verified=%s)",
            len(X), y.count(0), y.count(1), verified_only
        )
        return (X, y)

    def get_unverified_records(
        self,
        limit: int = 100,
    ) -> List[FeatureRecord]:
        """
        Return records that have not been analyst-verified.
        """
        unverified = [r for r in self._records if not r.verified]
        # Return highest-confidence bot predictions first (most useful for review)
        return unverified[-limit:]

    def get_records_for_user(
        self,
        user_id: str,
    ) -> List[FeatureRecord]:
        """
        Return all stored records for a specific user.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        list of FeatureRecord
        """
        return [r for r in self._records if r.user_id == user_id]

    def delete_user_records(self, user_id: str) -> int:
        """
        Delete all records for a user (GDPR right to erasure).
        """
        before = len(self._records)
        self._records = [r for r in self._records if r.user_id != user_id]
        deleted = before - len(self._records)
        logger.info(
            "GDPR erasure: deleted %d records for user=%s", deleted, user_id
        )
        return deleted

    # STATISTICS
    def stats(self) -> Dict[str, Any]:
        """
        Return store statistics.
        """
        total   = len(self._records)
        human   = sum(1 for r in self._records if r.label == 0)
        bot     = sum(1 for r in self._records if r.label == 1)
        verified = sum(1 for r in self._records if r.verified)

        by_source: Dict[str, int] = {}
        for r in self._records:
            by_source[r.source] = by_source.get(r.source, 0) + 1

        return {
            "total_records":     total,
            "human_records":     human,
            "bot_records":       bot,
            "verified_records":  verified,
            "unverified":        total - verified,
            "by_source":         by_source,
            "total_saved":       self._total_saved,
            "total_retrieved":   self._total_retrieved,
            "capacity_used_pct": round(100 * total / self._max_records, 1),
        }

    def export_jsonl(self, path: str, verified_only: bool = False) -> int:
        """
        Export all (or verified-only) records to a JSONL file.
        """
        records = self._filter_records(verified_only=verified_only)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")
        logger.info("Exported %d records to %s", len(records), path)
        return len(records)

    @classmethod
    def import_jsonl(
        cls,
        path: str,
        db_url: Optional[str] = None,
    ) -> "FeatureStore":
        """
        Create a FeatureStore and populate it from a JSONL export file.
        FeatureStore
        """
        store = cls(db_url=db_url)
        loaded = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    store.save(
                        user_id    = data["user_id"],
                        vector     = data["vector"],
                        label      = data["label"],
                        session_id = data.get("session_id", ""),
                        verified   = data.get("verified", False),
                        source     = data.get("source", LabelSource.LIVE),
                    )
                    loaded += 1
                except Exception as e:
                    logger.debug("Skipping malformed record: %s", e)
        logger.info("Imported %d records from %s", loaded, path)
        return store
    # PRIVATE HELPERS
    def _filter_records(
        self,
        verified_only: bool = False,
    ) -> List[FeatureRecord]:
        """Filter the record list based on criteria."""
        if verified_only:
            return [r for r in self._records if r.verified]
        return list(self._records)

    @staticmethod
    def _balance_records(records: List[FeatureRecord]) -> List[FeatureRecord]:
        """Downsample the majority class to produce a balanced list."""
        human   = [r for r in records if r.label == 0]
        bots    = [r for r in records if r.label == 1]
        min_n   = min(len(human), len(bots))
        if min_n == 0:
            return records
        sampled_h = random.sample(human, min_n)
        sampled_b = random.sample(bots,  min_n)
        combined  = sampled_h + sampled_b
        random.shuffle(combined)
        return combined

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"FeatureStore("
            f"total={s['total_records']}, "
            f"human={s['human_records']}, "
            f"bot={s['bot_records']}, "
            f"verified={s['verified_records']}, "
            f"backend={'postgresql' if self._db_url else 'memory'})"
        )
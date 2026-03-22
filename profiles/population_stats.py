"""
zelvryn/profiles/population_stats.py
======================================
PopulationStats — aggregates anonymised behavioural statistics across all
user profiles to maintain the population prior used by cold start initialisation.

Role in the system
-------------------
PopulationStats is the data producer for coldstart/population_prior.py.
It answers the question: "What does a typical legitimate user in this
deployment look like?" — expressed as statistical aggregates that can be
assigned to new users who have no personal profile yet.

Update strategy
---------------
The stats are maintained through two update paths:

  1. Batch rebuild (scripts/build_population_prior.py):
     Iterates all profiles in Redis, computes fresh aggregates from scratch.
     Run weekly via cron. Produces the most accurate prior.

  2. Incremental update (update_with_profile):
     Called when a user profile matures (crosses the COLD_START_RAMP_ACTIONS
     threshold). Applies a small EWMA weight to the incoming profile's
     statistics without touching the rest of the aggregate. Keeps the prior
     fresh between weekly rebuilds at negligible compute cost.

Privacy design
--------------
PopulationStats stores ONLY aggregate statistics — means, standard deviations,
medians. No individual user_id, session_id, action content, or resource_id is
stored. The output is mathematically indistinguishable from a population mean
computed over any sufficiently large group, providing strong anonymity.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.profile import BehavioralProfile
from core.action import UserAction
from core.constants import EWMA_DECAY_FACTOR, TEMPORAL_DEFAULT_MEAN, TEMPORAL_DEFAULT_STDEV

logger = logging.getLogger(__name__)

# Minimum profiles needed to produce a reliable population prior
MIN_PROFILES_FOR_STATS: int = 10

# EWMA weight for incremental updates (very small — individual users
# should not dominate the population aggregate)
INCREMENTAL_UPDATE_WEIGHT: float = 0.02

# Output file path for the JSON export
DEFAULT_STATS_PATH: str = "data/processed/population_prior.json"


@dataclass
class PopulationStats:
    """
    Aggregated anonymised statistics across all user profiles.

    This class both computes and stores the statistics. It can be built
    fresh from all profiles via build_from_profiles() or updated
    incrementally via update_with_profile().

    Attributes
    ----------
    mean_action_vector : list of float
        Population-average normalised action type frequency vector (8-dim).
    std_action_vector : list of float
        Standard deviation of each frequency component across users.
    mean_interval_s : float
        Population-average inter-action interval in seconds.
    std_interval_s : float
        Standard deviation of inter-action intervals across users.
    median_threshold : float
        Median adaptive cosine similarity threshold across users.
    n_profiles : int
        Number of profiles incorporated in these statistics.
    last_rebuilt : float
        Unix timestamp of last full rebuild.
    last_incremental_update : float
        Unix timestamp of last incremental update.
    """
    mean_action_vector:      List[float]    = field(default_factory=lambda: [1.0/8]*8)
    std_action_vector:       List[float]    = field(default_factory=lambda: [0.05]*8)
    mean_interval_s:         float          = TEMPORAL_DEFAULT_MEAN
    std_interval_s:          float          = TEMPORAL_DEFAULT_STDEV
    median_threshold:        float          = 0.75
    n_profiles:              int            = 0
    last_rebuilt:            float          = 0.0
    last_incremental_update: float          = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD FROM ALL PROFILES
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def build_from_profiles(
        cls,
        profiles: List[BehavioralProfile],
        min_actions: int = 20,
    ) -> "PopulationStats":
        """
        Compute population statistics from a list of BehavioralProfile objects.

        Only mature profiles (>= min_actions) are included.

        Parameters
        ----------
        profiles : list of BehavioralProfile
        min_actions : int
            Minimum recorded actions to include a profile. Default 20.

        Returns
        -------
        PopulationStats
        """
        mature = [p for p in profiles if p.vector_count >= min_actions]

        if len(mature) < MIN_PROFILES_FOR_STATS:
            logger.warning(
                "Only %d mature profiles available (min %d required). "
                "Using defaults.",
                len(mature), MIN_PROFILES_FOR_STATS
            )
            stats = cls()
            stats.n_profiles  = len(mature)
            stats.last_rebuilt = time.time()
            return stats

        logger.info(
            "Building PopulationStats from %d profiles", len(mature)
        )

        n_types = UserAction.type_count()

        # ── Action frequency vectors ───────────────────────────────────────
        all_vecs = [p.get_mean_vector() for p in mature]

        mean_vec = [
            sum(v[i] for v in all_vecs) / len(all_vecs)
            for i in range(n_types)
        ]
        # Normalise
        total = sum(mean_vec)
        if total > 0:
            mean_vec = [v / total for v in mean_vec]

        std_vec = []
        for i in range(n_types):
            vals = [v[i] for v in all_vecs]
            mu   = mean_vec[i]
            variance = sum((v - mu)**2 for v in vals) / max(len(vals)-1, 1)
            std_vec.append(math.sqrt(variance))

        # ── Timing statistics ──────────────────────────────────────────────
        all_means = []
        all_stds  = []
        for p in mature:
            m, s = p.get_timing_stats()
            if m > 0:
                all_means.append(m)
                all_stds.append(s)

        pop_mean_iv = (
            sum(all_means) / len(all_means)
            if all_means else TEMPORAL_DEFAULT_MEAN
        )
        pop_std_iv = (
            sum(all_stds) / len(all_stds)
            if all_stds else TEMPORAL_DEFAULT_STDEV
        )

        # ── Adaptive threshold median ──────────────────────────────────────
        thresholds = sorted(p.adaptive_threshold for p in mature)
        n          = len(thresholds)
        median_thresh = (
            thresholds[n // 2]
            if n % 2 == 1
            else (thresholds[n//2 - 1] + thresholds[n//2]) / 2.0
        )

        stats = cls(
            mean_action_vector = mean_vec,
            std_action_vector  = std_vec,
            mean_interval_s    = pop_mean_iv,
            std_interval_s     = pop_std_iv,
            median_threshold   = median_thresh,
            n_profiles         = len(mature),
            last_rebuilt       = time.time(),
        )

        logger.info(
            "PopulationStats built: n=%d mean_interval=%.2fs "
            "threshold=%.3f vector=%s",
            stats.n_profiles, stats.mean_interval_s,
            stats.median_threshold,
            [round(v, 3) for v in stats.mean_action_vector]
        )
        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # INCREMENTAL UPDATE
    # ─────────────────────────────────────────────────────────────────────────

    def update_with_profile(
        self,
        profile: BehavioralProfile,
        weight: float = INCREMENTAL_UPDATE_WEIGHT,
    ) -> None:
        """
        Incorporate a newly matured profile into the population stats.

        Uses EWMA with a very small weight so no single user can significantly
        shift the population aggregate.

        Parameters
        ----------
        profile : BehavioralProfile
            A profile that has just crossed the maturity threshold.
        weight : float, optional
            EWMA weight. Default 0.02 (profile contributes ~2% of aggregate).
        """
        if not profile.is_mature():
            return

        n_types = UserAction.type_count()
        new_vec = profile.get_mean_vector()

        # Update mean action vector
        for i in range(n_types):
            self.mean_action_vector[i] = (
                (1 - weight) * self.mean_action_vector[i]
                + weight * new_vec[i]
            )

        # Re-normalise
        total = sum(self.mean_action_vector)
        if total > 0:
            self.mean_action_vector = [v / total for v in self.mean_action_vector]

        # Update timing
        new_mean_iv, new_std_iv = profile.get_timing_stats()
        self.mean_interval_s = (
            (1 - weight) * self.mean_interval_s + weight * new_mean_iv
        )
        self.std_interval_s = (
            (1 - weight) * self.std_interval_s + weight * new_std_iv
        )

        # Update threshold (rolling average)
        self.median_threshold = (
            (1 - weight) * self.median_threshold
            + weight * profile.adaptive_threshold
        )

        self.n_profiles += 1
        self.last_incremental_update = time.time()

        logger.debug(
            "PopulationStats incremental update: user=%s n_profiles=%d",
            profile.user_id, self.n_profiles
        )

    # ─────────────────────────────────────────────────────────────────────────
    # QUERIES
    # ─────────────────────────────────────────────────────────────────────────

    def get_prior_vector(self) -> List[float]:
        """
        Return the population mean action frequency vector.

        Returns
        -------
        list of float
            8-dimensional normalised frequency vector.
        """
        return list(self.mean_action_vector)

    def get_timing_prior(self) -> tuple:
        """
        Return (mean_interval, std_interval) for the population.

        Returns
        -------
        tuple of (float, float)
        """
        return (self.mean_interval_s, self.std_interval_s)

    def get_threshold_prior(self) -> float:
        """
        Return the population median adaptive threshold.

        Returns
        -------
        float
        """
        return self.median_threshold

    def is_stale(self, max_age_days: int = 7) -> bool:
        """
        Return True if the stats were last rebuilt more than max_age_days ago.
        """
        if self.last_rebuilt == 0:
            return True
        return (time.time() - self.last_rebuilt) > (max_age_days * 86400)

    def age_hours(self) -> float:
        """Return the age of the last full rebuild in hours."""
        if self.last_rebuilt == 0:
            return float("inf")
        return (time.time() - self.last_rebuilt) / 3600.0

    def diversity_score(self) -> float:
        """
        Compute a diversity score for the population action distribution.

        A uniform distribution has maximum diversity (score = 1.0).
        A distribution where all users perform one action type has score = 0.0.

        Returns
        -------
        float in [0.0, 1.0]
        """
        n = len(self.mean_action_vector)
        if n == 0:
            return 0.0
        uniform = 1.0 / n
        deviation = sum(abs(v - uniform) for v in self.mean_action_vector)
        max_deviation = 2 * (1 - uniform)   # worst case: all weight on one type
        if max_deviation == 0:
            return 1.0
        return max(0.0, 1.0 - deviation / max_deviation)

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str = DEFAULT_STATS_PATH) -> None:
        """
        Save population statistics to a JSON file.

        Parameters
        ----------
        path : str
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("PopulationStats saved to %s (n=%d)", path, self.n_profiles)

    @classmethod
    def load(cls, path: str = DEFAULT_STATS_PATH) -> "PopulationStats":
        """
        Load population statistics from a JSON file.

        Returns default stats if the file does not exist.

        Parameters
        ----------
        path : str

        Returns
        -------
        PopulationStats
        """
        if not os.path.exists(path):
            logger.warning(
                "PopulationStats file not found at %s. "
                "Using defaults. Run scripts/build_population_prior.py.",
                path
            )
            return cls()

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        stats = cls(
            mean_action_vector      = data.get("mean_action_vector", [1.0/8]*8),
            std_action_vector       = data.get("std_action_vector",  [0.05]*8),
            mean_interval_s         = data.get("mean_interval_s",     TEMPORAL_DEFAULT_MEAN),
            std_interval_s          = data.get("std_interval_s",      TEMPORAL_DEFAULT_STDEV),
            median_threshold        = data.get("median_threshold",     0.75),
            n_profiles              = data.get("n_profiles",           0),
            last_rebuilt            = data.get("last_rebuilt",         0.0),
            last_incremental_update = data.get("last_incremental_update", 0.0),
        )

        logger.info(
            "PopulationStats loaded: n=%d age=%.1fh stale=%s",
            stats.n_profiles, stats.age_hours(), stats.is_stale()
        )
        return stats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_action_vector":      [round(v, 6) for v in self.mean_action_vector],
            "std_action_vector":       [round(v, 6) for v in self.std_action_vector],
            "mean_interval_s":         round(self.mean_interval_s, 4),
            "std_interval_s":          round(self.std_interval_s, 4),
            "median_threshold":        round(self.median_threshold, 4),
            "n_profiles":              self.n_profiles,
            "last_rebuilt":            self.last_rebuilt,
            "last_incremental_update": self.last_incremental_update,
            "action_types":            UserAction.ACTION_TYPES,
            "diversity_score":         round(self.diversity_score(), 4),
            "age_hours":               round(self.age_hours(), 2),
            "is_stale":                self.is_stale(),
        }

    def __repr__(self) -> str:
        return (
            f"PopulationStats("
            f"n={self.n_profiles}, "
            f"mean_iv={self.mean_interval_s:.2f}s, "
            f"threshold={self.median_threshold:.3f}, "
            f"stale={self.is_stale()})"
        )

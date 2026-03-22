"""
zelvryn/coldstart/population_prior.py
======================================
PopulationPriorProfile — derives and stores a population-average behavioural
baseline used to warm-start new user profiles during the cold start period.

The cold start problem
----------------------
A BehavioralProfile requires observed actions before it can compute a meaningful
mean vector, timing statistics, or adaptive threshold. A brand-new user has none
of these. Without a fallback, the cosine similarity check returns 1.0 for every
window (comparing against a zero vector is undefined), the timing Z-score cannot
be computed, and the adaptive threshold stays at the system default regardless
of whether that default fits this particular application's usage pattern.

The population prior solves this by aggregating anonymised statistics across
ALL existing users and providing the result as a starting baseline for new
users. A new user's profile is initialised as if they had already been observed
behaving like a typical legitimate user in this specific deployment.

How it works
------------
  1. `build_from_profiles(profiles)` — aggregates statistics across a list of
     BehavioralProfile objects. Computes:
       - Mean action type frequency vector (element-wise average of all profiles'
         mean vectors)
       - Mean inter-action timing (seconds), pooled across all users
       - Pooled standard deviation of timing
       - Suggested adaptive threshold (median across all users)

  2. `save(path)` — serialises the prior to a JSON file at
     data/processed/population_prior.json. Run weekly via
     scripts/build_population_prior.py.

  3. `load(path)` — deserialises from JSON. Called at API startup to have the
     prior ready for immediate assignment to new users.

  4. `create_new_user_profile(user_id)` — returns a BehavioralProfile whose
     statistics are initialised to the population prior values. The profile's
     vector_count is set to a sentinel value (PRIOR_SENTINEL_COUNT) so the
     engine knows it is a prior-seeded profile, not a fully matured one.

Privacy design
--------------
The prior is computed only from aggregate statistics — no raw action logs,
resource IDs, or timing sequences from individual users are stored. The
output is a set of floating-point averages that cannot be traced back to
any individual user.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.action import UserAction
from core.profile import BehavioralProfile
from core.constants import (
    SIMILARITY_THRESHOLD,
    TEMPORAL_DEFAULT_MEAN,
    TEMPORAL_DEFAULT_STDEV,
    EWMA_DECAY_FACTOR,
    COLD_START_RAMP_ACTIONS,
)

logger = logging.getLogger(__name__)

# Sentinel action count given to prior-seeded profiles.
# The engine checks `profile.vector_count >= COLD_START_RAMP_ACTIONS` to decide
# whether to apply ramp-up rules. By setting the sentinel below the ramp threshold,
# we ensure new users always enter ramp-up mode even with a prior.
PRIOR_SENTINEL_COUNT: int = 0

# Minimum number of existing profiles required to compute a meaningful prior.
# Below this threshold, the build() method logs a warning and uses system defaults.
MIN_PROFILES_FOR_PRIOR: int = 10


class PopulationPriorProfile:
    """
    Aggregated population-average behavioural baseline for cold start initialisation.

    Attributes
    ----------
    mean_vector : list of float
        Population-average normalised action type frequency vector (8-dimensional).
        Element i = average proportion of action type i across all source profiles.
    mean_interval : float
        Population-average inter-action interval in seconds.
    std_interval : float
        Pooled standard deviation of inter-action intervals.
    median_adaptive_threshold : float
        Median adaptive cosine similarity threshold across all source profiles.
    source_profile_count : int
        Number of profiles used to build this prior.
    built_at : float
        Unix timestamp when this prior was last computed.
    deployment_context : str
        Optional label describing the application context (e.g., "enterprise_saas").
    """

    def __init__(self) -> None:
        self.mean_vector: List[float]              = [1.0 / UserAction.type_count()] * UserAction.type_count()
        self.mean_interval: float                  = TEMPORAL_DEFAULT_MEAN
        self.std_interval: float                   = TEMPORAL_DEFAULT_STDEV
        self.median_adaptive_threshold: float      = SIMILARITY_THRESHOLD
        self.source_profile_count: int             = 0
        self.built_at: float                       = time.time()
        self.deployment_context: str               = "default"

    # ─────────────────────────────────────────────────────────────────────────
    # BUILDING THE PRIOR
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def build_from_profiles(
        cls,
        profiles: List[BehavioralProfile],
        deployment_context: str = "default",
        min_actions_per_profile: int = 20,
    ) -> "PopulationPriorProfile":
        """
        Build a population prior from a list of existing user profiles.

        Only profiles with at least `min_actions_per_profile` recorded actions
        are included — immature profiles skew the statistics and should be excluded.

        Parameters
        ----------
        profiles : list of BehavioralProfile
            All existing user profiles in the deployment.
        deployment_context : str, optional
            Descriptive label for this prior (e.g., "enterprise_saas", "fintech").
        min_actions_per_profile : int, optional
            Minimum recorded actions for a profile to be included. Default 20.

        Returns
        -------
        PopulationPriorProfile
            Computed prior ready for serialisation or immediate use.
        """
        prior = cls()
        prior.deployment_context = deployment_context
        prior.built_at = time.time()

        # Filter to mature profiles only
        mature = [p for p in profiles if p.vector_count >= min_actions_per_profile]

        if len(mature) < MIN_PROFILES_FOR_PRIOR:
            logger.warning(
                "Only %d mature profiles available (minimum %d required for "
                "reliable prior). Using system defaults. Accumulate more users "
                "before building a meaningful prior.",
                len(mature), MIN_PROFILES_FOR_PRIOR
            )
            prior.source_profile_count = len(mature)
            return prior   # return with system defaults

        logger.info(
            "Building population prior from %d profiles (context=%s)",
            len(mature), deployment_context
        )

        # ── Mean action frequency vector ──────────────────────────────────
        # Average each component of the frequency vector across all profiles
        n_types = UserAction.type_count()
        component_means = []
        for i in range(n_types):
            values = []
            for p in mature:
                vec = p.get_mean_vector()
                if vec:
                    values.append(vec[i])
            component_means.append(statistics.mean(values) if values else 0.0)

        # Normalise so components sum to 1.0
        total = sum(component_means)
        if total > 0:
            prior.mean_vector = [v / total for v in component_means]
        else:
            prior.mean_vector = [1.0 / n_types] * n_types

        # ── Pooled timing statistics ───────────────────────────────────────
        all_mean_intervals = []
        all_std_intervals = []
        for p in mature:
            mean_iv, std_iv = p.get_timing_stats()
            if mean_iv > 0:
                all_mean_intervals.append(mean_iv)
                all_std_intervals.append(std_iv)

        prior.mean_interval = (
            statistics.mean(all_mean_intervals)
            if all_mean_intervals else TEMPORAL_DEFAULT_MEAN
        )
        prior.std_interval = (
            statistics.mean(all_std_intervals)
            if all_std_intervals else TEMPORAL_DEFAULT_STDEV
        )

        # ── Median adaptive threshold ──────────────────────────────────────
        thresholds = [p.adaptive_threshold for p in mature]
        prior.median_adaptive_threshold = statistics.median(thresholds)

        prior.source_profile_count = len(mature)

        logger.info(
            "Population prior built: mean_interval=%.2fs std=%.2fs "
            "threshold=%.3f vector=%s",
            prior.mean_interval, prior.std_interval,
            prior.median_adaptive_threshold,
            [round(v, 3) for v in prior.mean_vector]
        )

        return prior

    @classmethod
    def build_from_aggregates(
        cls,
        aggregate_data: Dict[str, Any],
    ) -> "PopulationPriorProfile":
        """
        Build a prior from pre-computed aggregate statistics.

        Used when individual BehavioralProfile objects are not available
        (e.g., when loading from an external analytics pipeline). The aggregate
        dict must contain the same keys as produced by to_dict().

        Parameters
        ----------
        aggregate_data : dict
            Pre-computed aggregate stats. Expected keys:
              mean_vector, mean_interval, std_interval,
              median_adaptive_threshold, source_profile_count.

        Returns
        -------
        PopulationPriorProfile
        """
        prior = cls()
        prior.mean_vector                = aggregate_data.get("mean_vector", prior.mean_vector)
        prior.mean_interval              = aggregate_data.get("mean_interval", prior.mean_interval)
        prior.std_interval               = aggregate_data.get("std_interval", prior.std_interval)
        prior.median_adaptive_threshold  = aggregate_data.get("median_adaptive_threshold", prior.median_adaptive_threshold)
        prior.source_profile_count       = aggregate_data.get("source_profile_count", 0)
        prior.built_at                   = aggregate_data.get("built_at", time.time())
        prior.deployment_context         = aggregate_data.get("deployment_context", "default")
        return prior

    # ─────────────────────────────────────────────────────────────────────────
    # APPLYING THE PRIOR TO NEW USERS
    # ─────────────────────────────────────────────────────────────────────────

    def create_new_user_profile(
        self,
        user_id: str,
        n_virtual_actions: int = 30,
    ) -> BehavioralProfile:
        """
        Create a BehavioralProfile pre-seeded with population prior statistics.

        The returned profile behaves as if the user has already been observed
        making `n_virtual_actions` actions matching the population average.
        This gives the anomaly engine enough signal to make meaningful
        comparisons from the very first real action.

        The profile's `vector_count` is set to `n_virtual_actions` so the
        RampUpManager still correctly identifies it as in cold start (since
        n_virtual_actions < COLD_START_RAMP_ACTIONS).

        Parameters
        ----------
        user_id : str
            Unique identifier for the new user.
        n_virtual_actions : int, optional
            Number of virtual prior actions to simulate. Must be less than
            COLD_START_RAMP_ACTIONS (default 50) to keep cold start active.
            Default 30.

        Returns
        -------
        BehavioralProfile
            A profile pre-seeded with population average statistics.
        """
        if n_virtual_actions >= COLD_START_RAMP_ACTIONS:
            logger.warning(
                "n_virtual_actions=%d >= COLD_START_RAMP_ACTIONS=%d. "
                "New user would skip cold start immediately. Clamping to %d.",
                n_virtual_actions, COLD_START_RAMP_ACTIONS,
                COLD_START_RAMP_ACTIONS - 10
            )
            n_virtual_actions = COLD_START_RAMP_ACTIONS - 10

        profile = BehavioralProfile(user_id=user_id)

        # Seed the vector_sum using the population mean vector
        # vector_sum[i] represents total count of action type i
        n_types = UserAction.type_count()
        profile.vector_sum   = [v * n_virtual_actions for v in self.mean_vector]
        profile.vector_count = n_virtual_actions

        # Seed EWMA vector
        profile.ewma_vector = list(profile.vector_sum)

        # Seed timing statistics
        profile.ewma_mean_interval = self.mean_interval
        profile.ewma_std_interval  = self.std_interval

        # Seed simulated interval history (used for std dev calculation)
        import random
        rng = random.Random(hash(user_id) % (2**32))
        for _ in range(min(n_virtual_actions, 20)):
            iv = max(0.1, rng.gauss(self.mean_interval, self.std_interval))
            profile.time_intervals.append(iv)

        # Apply population median adaptive threshold
        profile.adaptive_threshold = self.median_adaptive_threshold

        # Seed action counts proportionally
        for i, t in enumerate(UserAction.ACTION_TYPES):
            count = round(self.mean_vector[i] * n_virtual_actions)
            if count > 0:
                profile.action_counts[t] = count

        logger.debug(
            "Created prior-seeded profile for user=%s "
            "(virtual_actions=%d mean_interval=%.2fs threshold=%.3f)",
            user_id, n_virtual_actions,
            profile.ewma_mean_interval, profile.adaptive_threshold
        )

        return profile

    def is_stale(self, max_age_days: int = 7) -> bool:
        """
        Return True if the prior is older than max_age_days.

        A stale prior should be rebuilt via scripts/build_population_prior.py
        to reflect the current user population's behaviour.

        Parameters
        ----------
        max_age_days : int
            Maximum acceptable age in days. Default 7.

        Returns
        -------
        bool
        """
        age_seconds = time.time() - self.built_at
        return age_seconds > (max_age_days * 86400)

    def update_incrementally(
        self,
        new_profile: BehavioralProfile,
        weight: float = 0.02,
    ) -> None:
        """
        Incrementally update the prior with a new user's statistics.

        Uses a small EWMA weight so individual users do not dominate the prior.
        Called after each user's profile matures enough to contribute.

        Parameters
        ----------
        new_profile : BehavioralProfile
            A matured user profile to incorporate.
        weight : float, optional
            EWMA weight for the new observation. Default 0.02 (very light
            contribution — roughly equivalent to 1 user in 50).
        """
        if not new_profile.is_mature():
            return

        new_vec = new_profile.get_mean_vector()
        for i in range(len(self.mean_vector)):
            self.mean_vector[i] = (
                (1 - weight) * self.mean_vector[i]
                + weight * new_vec[i]
            )

        new_mean_iv, new_std_iv = new_profile.get_timing_stats()
        self.mean_interval  = (1 - weight) * self.mean_interval  + weight * new_mean_iv
        self.std_interval   = (1 - weight) * self.std_interval   + weight * new_std_iv
        self.source_profile_count += 1

        logger.debug(
            "Prior incrementally updated with user=%s (total sources=%d)",
            new_profile.user_id, self.source_profile_count
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the population prior to a JSON file.

        Parameters
        ----------
        path : str
            Output file path. Typically data/processed/population_prior.json.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Population prior saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "PopulationPriorProfile":
        """
        Load a population prior from a JSON file.

        Falls back to system defaults if the file does not exist, logging
        a warning so the operator knows to run build_population_prior.py.

        Parameters
        ----------
        path : str
            Path to the JSON file.

        Returns
        -------
        PopulationPriorProfile
        """
        if not os.path.exists(path):
            logger.warning(
                "Population prior file not found at %s. "
                "Using system defaults. Run scripts/build_population_prior.py "
                "to generate a deployment-specific prior.",
                path
            )
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prior = cls.build_from_aggregates(data)
        logger.info(
            "Population prior loaded from %s "
            "(sources=%d built=%s stale=%s)",
            path,
            prior.source_profile_count,
            time.strftime("%Y-%m-%d", time.localtime(prior.built_at)),
            prior.is_stale()
        )
        return prior

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise to a plain dict (JSON-safe).

        Returns
        -------
        dict
        """
        return {
            "mean_vector":               self.mean_vector,
            "mean_interval":             self.mean_interval,
            "std_interval":              self.std_interval,
            "median_adaptive_threshold": self.median_adaptive_threshold,
            "source_profile_count":      self.source_profile_count,
            "built_at":                  self.built_at,
            "deployment_context":        self.deployment_context,
            "action_types":              UserAction.ACTION_TYPES,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Dunder methods
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"PopulationPriorProfile("
            f"sources={self.source_profile_count}, "
            f"context={self.deployment_context!r}, "
            f"mean_interval={self.mean_interval:.2f}s, "
            f"stale={self.is_stale()})"
        )
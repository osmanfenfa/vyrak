"""
zelvryn/profiles/shared_device_detector.py
============================================
SharedDeviceDetector — analyses a behavioural profile's timing interval
distribution for bimodal structure indicating that two or more people share
the same account.

The shared account problem
----------------------------
When two people share an account (common in households, small teams, or
legacy enterprise systems with shared service accounts), Zelvryn's profile
builds a confused mixture of two distinct behavioural signatures. This causes:

  - Chronic false positives: User A's patterns flag as anomalous against
    User B's baseline and vice versa.
  - Degraded ML accuracy: The training samples include mixed behaviour,
    producing a centroid that fits neither user well.
  - Unreliable timing statistics: The timing distribution is bimodal
    (two peaks), making mean and stdev meaningless.

Detection: bimodal clustering on inter-action intervals
---------------------------------------------------------
Human typing and navigation rhythms are distinctive: a fast typist acts at
~1-2 second intervals; a careful reader at ~5-15 seconds. When two people
share an account, the interval histogram shows two clusters — one per person.

The detector runs k-means with k=2 on the recorded inter-action intervals.
A high silhouette score (cluster separation measure) indicates the two
clusters are well-separated and likely represent two distinct users.

When bimodality is detected:
  1. The account is flagged and the user is notified.
  2. Two "sub-profile blueprints" are generated — the timing and action
     distributions associated with each cluster.
  3. These blueprints can be used to initialise two new sub-profiles if
     the user confirms the split via the appeal/account settings flow.

Silhouette score
----------------
The silhouette score measures how well each data point fits its assigned
cluster vs the alternative cluster:

  s(i) = (b(i) - a(i)) / max(a(i), b(i))
  where:
    a(i) = mean distance to other points in same cluster (cohesion)
    b(i) = mean distance to nearest other cluster (separation)

  s = 1.0 : perfect separation (clearly two distinct clusters)
  s = 0.0 : ambiguous (clusters overlap)
  s = -1.0: misclassified (wrong cluster)

We flag bimodality when the mean silhouette score exceeds a threshold.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.profile import BehavioralProfile
from core.action import UserAction

logger = logging.getLogger(__name__)

# Minimum inter-action intervals required for bimodal analysis
MIN_INTERVALS_FOR_ANALYSIS: int = 40

# Silhouette score threshold above which shared device is declared
BIMODAL_SILHOUETTE_THRESHOLD: float = 0.45

# Minimum gap between cluster means (seconds) to be considered two people
MIN_CLUSTER_MEAN_GAP_S: float = 3.0

# K-means iterations
KMEANS_MAX_ITER: int = 50
KMEANS_N_INIT:   int = 5    # multiple restarts to avoid local optima


@dataclass
class SubProfileBlueprint:
    """
    Blueprint for a sub-profile derived from one cluster of shared-account data.

    Attributes
    ----------
    cluster_id : int
        0 or 1 — which k-means cluster this blueprint represents.
    mean_interval : float
        Estimated mean inter-action interval for this sub-user.
    std_interval : float
        Estimated interval standard deviation.
    interval_count : int
        Number of intervals in this cluster.
    action_proportion : float
        Fraction of total intervals belonging to this cluster.
    estimated_actions : list of str
        Action types disproportionately associated with this cluster
        (inferred from which sessions produced these intervals).
    """
    cluster_id:          int
    mean_interval:       float
    std_interval:        float
    interval_count:      int
    action_proportion:   float
    estimated_actions:   List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id":        self.cluster_id,
            "mean_interval":     round(self.mean_interval, 3),
            "std_interval":      round(self.std_interval, 3),
            "interval_count":    self.interval_count,
            "action_proportion": round(self.action_proportion, 3),
            "estimated_actions": self.estimated_actions,
        }


@dataclass
class SharedDeviceReport:
    """
    Result of a shared-device detection analysis.

    Attributes
    ----------
    user_id : str
    is_shared : bool
        True if bimodal timing distribution was detected.
    silhouette_score : float
        Cluster separation quality (0=none, 1=perfect).
    cluster_means : list of float
        Mean interval for each cluster.
    cluster_gap_s : float
        Gap between cluster means (seconds).
    sub_profile_blueprints : list of SubProfileBlueprint
        One blueprint per detected user, for sub-profile creation.
    confidence : str
        "HIGH", "MEDIUM", or "LOW" based on silhouette score.
    intervals_analysed : int
    analysed_at : float
    """
    user_id:                 str
    is_shared:               bool                       = False
    silhouette_score:        float                      = 0.0
    cluster_means:           List[float]                = field(default_factory=list)
    cluster_gap_s:           float                      = 0.0
    sub_profile_blueprints:  List[SubProfileBlueprint]  = field(default_factory=list)
    confidence:              str                        = "LOW"
    intervals_analysed:      int                        = 0
    analysed_at:             float                      = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"SharedDeviceReport(user={self.user_id} "
            f"shared={self.is_shared} "
            f"silhouette={self.silhouette_score:.3f} "
            f"gap={self.cluster_gap_s:.1f}s "
            f"confidence={self.confidence})"
        )


class SharedDeviceDetector:
    """
    Detects shared-account usage via bimodal timing interval clustering.

    Parameters
    ----------
    silhouette_threshold : float, optional
        Minimum silhouette score to declare shared device. Default 0.45.
    min_cluster_gap_s : float, optional
        Minimum seconds between cluster means. Default 3.0.
    min_intervals : int, optional
        Minimum intervals needed for analysis. Default 40.
    random_seed : int, optional
        Seed for k-means initialisation. Default 42.
    """

    def __init__(
        self,
        silhouette_threshold: float = BIMODAL_SILHOUETTE_THRESHOLD,
        min_cluster_gap_s:    float = MIN_CLUSTER_MEAN_GAP_S,
        min_intervals:        int   = MIN_INTERVALS_FOR_ANALYSIS,
        random_seed:          int   = 42,
    ) -> None:
        self.silhouette_threshold = silhouette_threshold
        self.min_cluster_gap_s   = min_cluster_gap_s
        self.min_intervals       = min_intervals
        self.random_seed         = random_seed

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMARY INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def analyse(self, profile: BehavioralProfile) -> SharedDeviceReport:
        """
        Analyse a profile's timing distribution for bimodal structure.

        Parameters
        ----------
        profile : BehavioralProfile

        Returns
        -------
        SharedDeviceReport
        """
        user_id   = profile.user_id
        intervals = profile.time_intervals

        if len(intervals) < self.min_intervals:
            return SharedDeviceReport(
                user_id            = user_id,
                is_shared          = False,
                intervals_analysed = len(intervals),
            )

        # ── Run k-means (k=2) with multiple restarts ──────────────────────
        best_labels: List[int]   = []
        best_centres: List[float] = [0.0, 0.0]
        best_inertia: float      = float("inf")

        for init_num in range(KMEANS_N_INIT):
            labels, centres, inertia = self._kmeans_1d(
                intervals, k=2, seed=self.random_seed + init_num
            )
            if inertia < best_inertia:
                best_inertia  = inertia
                best_labels   = labels
                best_centres  = centres

        # ── Compute silhouette score ───────────────────────────────────────
        sil_score = self._silhouette_score(intervals, best_labels)

        # ── Check cluster separation ───────────────────────────────────────
        cluster_gap = abs(best_centres[0] - best_centres[1])

        is_shared = (
            sil_score  >= self.silhouette_threshold
            and cluster_gap >= self.min_cluster_gap_s
        )

        confidence = self._confidence_label(sil_score)

        # ── Build sub-profile blueprints ───────────────────────────────────
        blueprints: List[SubProfileBlueprint] = []
        if is_shared:
            blueprints = self._build_blueprints(
                intervals, best_labels, best_centres, profile
            )
            logger.info(
                "Shared device detected: %s",
                SharedDeviceReport(
                    user_id=user_id,
                    is_shared=True,
                    silhouette_score=sil_score,
                    cluster_means=sorted(best_centres),
                    cluster_gap_s=cluster_gap,
                    confidence=confidence,
                ).summary()
            )

        return SharedDeviceReport(
            user_id                = user_id,
            is_shared              = is_shared,
            silhouette_score       = round(sil_score, 4),
            cluster_means          = [round(c, 3) for c in sorted(best_centres)],
            cluster_gap_s          = round(cluster_gap, 3),
            sub_profile_blueprints = blueprints,
            confidence             = confidence,
            intervals_analysed     = len(intervals),
        )

    def suggest_split(
        self,
        profile: BehavioralProfile,
    ) -> Optional[Tuple[SubProfileBlueprint, SubProfileBlueprint]]:
        """
        Return two sub-profile blueprints if the account appears shared.

        Parameters
        ----------
        profile : BehavioralProfile

        Returns
        -------
        tuple of (SubProfileBlueprint, SubProfileBlueprint) or None
            None if no shared device pattern was detected.
        """
        report = self.analyse(profile)
        if not report.is_shared or len(report.sub_profile_blueprints) < 2:
            return None
        return (report.sub_profile_blueprints[0], report.sub_profile_blueprints[1])

    # ─────────────────────────────────────────────────────────────────────────
    # K-MEANS (1D, implemented from scratch to avoid numpy dependency)
    # ─────────────────────────────────────────────────────────────────────────

    def _kmeans_1d(
        self,
        data: List[float],
        k: int = 2,
        seed: int = 42,
    ) -> Tuple[List[int], List[float], float]:
        """
        Run k-means clustering on a 1D dataset.

        Parameters
        ----------
        data : list of float
        k : int
        seed : int

        Returns
        -------
        tuple of (labels, centres, inertia)
            labels: cluster assignment per data point
            centres: cluster centroid values
            inertia: sum of squared distances to nearest centroid
        """
        rng = random.Random(seed)
        n   = len(data)

        # Initialise centres using k-means++ strategy
        centres = [rng.choice(data)]
        while len(centres) < k:
            # Choose next centre with probability proportional to distance²
            distances = [
                min((x - c) ** 2 for c in centres)
                for x in data
            ]
            total = sum(distances)
            if total == 0:
                centres.append(rng.choice(data))
                continue
            probs = [d / total for d in distances]
            cumulative = []
            acc = 0.0
            for p in probs:
                acc += p
                cumulative.append(acc)
            r = rng.random()
            for i, cp in enumerate(cumulative):
                if r <= cp:
                    centres.append(data[i])
                    break
            else:
                centres.append(data[-1])

        # Iterate
        labels: List[int] = [0] * n
        for _ in range(KMEANS_MAX_ITER):
            # Assignment step
            new_labels = [
                min(range(k), key=lambda j: (x - centres[j]) ** 2)
                for x in data
            ]

            # Check for convergence
            if new_labels == labels:
                break
            labels = new_labels

            # Update step
            for j in range(k):
                cluster_points = [data[i] for i in range(n) if labels[i] == j]
                if cluster_points:
                    centres[j] = sum(cluster_points) / len(cluster_points)

        # Compute inertia
        inertia = sum(
            (data[i] - centres[labels[i]]) ** 2
            for i in range(n)
        )
        return (labels, centres, inertia)

    # ─────────────────────────────────────────────────────────────────────────
    # SILHOUETTE SCORE
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _silhouette_score(data: List[float], labels: List[int]) -> float:
        """
        Compute mean silhouette score for a 1D clustering result.

        For each point i:
          a(i) = mean distance to other points in the same cluster
          b(i) = mean distance to points in the nearest other cluster
          s(i) = (b(i) - a(i)) / max(a(i), b(i))

        Returns mean s(i) across all points.
        """
        n = len(data)
        if n < 4:
            return 0.0

        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            return 0.0

        # Group indices by cluster
        clusters: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            clusters.setdefault(lbl, []).append(i)

        silhouettes = []
        for i in range(n):
            my_label   = labels[i]
            my_cluster = [j for j in clusters[my_label] if j != i]

            if not my_cluster:
                silhouettes.append(0.0)
                continue

            a = sum(abs(data[i] - data[j]) for j in my_cluster) / len(my_cluster)

            # Find nearest other cluster
            b = float("inf")
            for other_label, other_indices in clusters.items():
                if other_label == my_label:
                    continue
                mean_dist = sum(
                    abs(data[i] - data[j]) for j in other_indices
                ) / len(other_indices)
                b = min(b, mean_dist)

            if b == float("inf"):
                silhouettes.append(0.0)
                continue

            denom = max(a, b)
            s = (b - a) / denom if denom > 0 else 0.0
            silhouettes.append(s)

        return sum(silhouettes) / len(silhouettes) if silhouettes else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # BLUEPRINT CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def _build_blueprints(
        self,
        intervals: List[float],
        labels:    List[int],
        centres:   List[float],
        profile:   BehavioralProfile,
    ) -> List[SubProfileBlueprint]:
        """
        Build sub-profile blueprints from the k-means clustering result.
        """
        n = len(intervals)
        blueprints = []

        for cluster_id in range(2):
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not cluster_indices:
                continue

            cluster_intervals = [intervals[i] for i in cluster_indices]
            mean_iv = sum(cluster_intervals) / len(cluster_intervals)
            variance = sum((x - mean_iv)**2 for x in cluster_intervals) / max(len(cluster_intervals)-1, 1)
            std_iv  = math.sqrt(variance)
            proportion = len(cluster_intervals) / n

            # Estimate action types from recent action history if available
            # Use a simple heuristic: the cluster with shorter intervals
            # is more likely to be associated with read-heavy quick sessions
            recent = list(profile.action_history)
            if recent and len(recent) >= len(intervals) // 2:
                # Assign actions to clusters based on timing
                cluster_actions: Dict[str, int] = {}
                for i in cluster_indices:
                    if i < len(recent):
                        at = recent[i].action_type
                        cluster_actions[at] = cluster_actions.get(at, 0) + 1
                top_actions = sorted(
                    cluster_actions, key=cluster_actions.get, reverse=True
                )[:3]
            else:
                top_actions = []

            blueprints.append(SubProfileBlueprint(
                cluster_id        = cluster_id,
                mean_interval     = round(mean_iv, 3),
                std_interval      = round(std_iv, 3),
                interval_count    = len(cluster_intervals),
                action_proportion = round(proportion, 3),
                estimated_actions = top_actions,
            ))

        return blueprints

    @staticmethod
    def _confidence_label(silhouette: float) -> str:
        if silhouette >= 0.65:
            return "HIGH"
        elif silhouette >= 0.45:
            return "MEDIUM"
        else:
            return "LOW"

    def __repr__(self) -> str:
        return (
            f"SharedDeviceDetector("
            f"sil_threshold={self.silhouette_threshold}, "
            f"min_gap={self.min_cluster_gap_s}s, "
            f"min_intervals={self.min_intervals})"
        )

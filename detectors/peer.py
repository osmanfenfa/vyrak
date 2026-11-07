"""
PeerGroupBaseline — compare an identity with similar identities.

Why: a brand-new identity has no history of its own, so per-identity checks
are switched off during cold start. Peers fill that gap: a new support agent
should look like other support agents; a new AI agent configured for ticket
triage should look like other triage agents.

How it works
------------
For each (tenant, peer group) we keep running mean/variance (Welford) of a
subset of the 20-dimensional window features, learned only from clean,
full windows. A live window is scored by the mean of its three largest
absolute z-scores across those features. No labels needed.

Peer groups are whatever the customer passes as ``peer_group`` when a
session starts (e.g. a role or agent type); default is the identity kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Feature indices (see core.features.FeatureExtractor.feature_names)
PEER_FEATURES: Dict[int, str] = {
    1: "share of reads",
    2: "share of writes",
    3: "share of deletes",
    5: "share of exports",
    6: "share of config changes",
    7: "share of admin actions",
    14: "share of sensitive actions",
    16: "resource diversity",
    17: "burst rate",
    18: "repeat-run length",
}

_MIN_STD = {
    # floors keep tiny-variance features from exploding the z-score
    16: 0.05, 17: 0.05, 18: 0.25,
}


@dataclass
class PeerAssessment:
    active: bool
    score: float
    anomalous: bool
    top_features: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


class PeerGroupBaseline:
    MAX_TRACKED_IDENTITIES = 1000

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: Dict[int, float] = {i: 0.0 for i in PEER_FEATURES}
        self.m2: Dict[int, float] = {i: 0.0 for i in PEER_FEATURES}
        self.identities: List[str] = []

    def update(self, features: Sequence[float], identity_id: str) -> None:
        self.n += 1
        for i in PEER_FEATURES:
            x = float(features[i])
            delta = x - self.mean[i]
            self.mean[i] += delta / self.n
            self.m2[i] += delta * (x - self.mean[i])
        if identity_id not in self.identities and len(self.identities) < self.MAX_TRACKED_IDENTITIES:
            self.identities.append(identity_id)

    def std(self, i: int) -> float:
        var = self.m2[i] / (self.n - 1) if self.n > 1 else 0.0
        return max(math.sqrt(max(var, 0.0)), _MIN_STD.get(i, 0.03))

    def assess(
        self,
        features: Sequence[float],
        min_windows: int,
        min_identities: int,
        z_limit: float,
        group_name: str,
    ) -> PeerAssessment:
        if self.n < min_windows or len(self.identities) < min_identities:
            return PeerAssessment(active=False, score=0.0, anomalous=False)

        zs = []
        for i, label in PEER_FEATURES.items():
            z = (float(features[i]) - self.mean[i]) / self.std(i)
            zs.append((abs(z), i, label, float(features[i])))
        zs.sort(reverse=True)
        top = zs[:3]
        score = sum(z for z, *_ in top) / len(top)
        anomalous = score > z_limit
        reasons: List[str] = []
        if anomalous:
            parts = [
                f"{label} {val:.2f} vs peer average {self.mean[i]:.2f}"
                for _, i, label, val in top
            ]
            reasons.append(
                f"Behaves unlike other identities in peer group '{group_name}': "
                + "; ".join(parts) + "."
            )
        return PeerAssessment(
            active=True, score=score, anomalous=anomalous,
            top_features=[label for _, _, label, _ in top], reasons=reasons,
        )

    def to_dict(self) -> Dict:
        return {
            "n": self.n,
            "mean": {str(k): v for k, v in self.mean.items()},
            "m2": {str(k): v for k, v in self.m2.items()},
            "identities": self.identities,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "PeerGroupBaseline":
        pg = cls()
        if not data:
            return pg
        pg.n = int(data.get("n", 0))
        for attr in ("mean", "m2"):
            stored = data.get(attr) or {}
            setattr(pg, attr, {i: float(stored.get(str(i), 0.0)) for i in PEER_FEATURES})
        pg.identities = list(data.get("identities") or [])
        return pg

"""
SequenceModel — a per-identity first-order Markov chain over action types.

Why: the 20 window features summarise *what* happened but barely capture
*order*. "read → export → admin → delete" and the same actions shuffled look
alike to the classifier. A Markov chain learns each identity's own transition
habits from clean traffic and measures how surprising recent transitions are.

Scoring
-------
For a transition a → b the surprise is  −log2 P(b | a)  with symmetric
Dirichlet smoothing. Under the identity's own model, the *expected* surprise
per step is the conditional entropy H (bits). If the mean surprise of the
recent transitions exceeds H by more than ``margin`` bits, the recent
sequence is much less likely than this identity's normal behaviour.

No labels needed; explanations name the rarest transition observed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

ACTION_TYPES = ("login", "read", "write", "delete", "search", "export", "config", "admin")
_IDX = {t: i for i, t in enumerate(ACTION_TYPES)}
_N = len(ACTION_TYPES)


@dataclass
class SequenceAssessment:
    active: bool                  # model had enough history to judge
    mean_surprise: float          # bits per transition, recent window
    expected_surprise: float      # conditional entropy H of the model (bits)
    excess_bits: float            # mean_surprise - expected_surprise
    anomalous: bool
    rarest_transition: Optional[str] = None
    rarest_count: int = 0
    rarest_row_total: int = 0
    reasons: List[str] = field(default_factory=list)


class SequenceModel:
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.counts: List[List[float]] = [[0.0] * _N for _ in range(_N)]

    # ── learning ──────────────────────────────────────────────────────────
    def update(self, prev_type: str, cur_type: str) -> None:
        i, j = _IDX.get(prev_type), _IDX.get(cur_type)
        if i is None or j is None:
            return
        self.counts[i][j] += 1.0

    @property
    def total_transitions(self) -> int:
        return int(sum(sum(r) for r in self.counts))

    # ── probabilities ─────────────────────────────────────────────────────
    def prob(self, prev_type: str, cur_type: str) -> float:
        i, j = _IDX[prev_type], _IDX[cur_type]
        row = self.counts[i]
        return (row[j] + self.alpha) / (sum(row) + self.alpha * _N)

    def surprise(self, prev_type: str, cur_type: str) -> float:
        return -math.log2(self.prob(prev_type, cur_type))

    def conditional_entropy(self) -> float:
        """H(next | prev), weighted by how often each prev state occurs."""
        total = self.total_transitions
        if total == 0:
            return math.log2(_N)
        h = 0.0
        for i in range(_N):
            row_total = sum(self.counts[i])
            if row_total == 0:
                continue
            denom = row_total + self.alpha * _N
            row_h = 0.0
            for j in range(_N):
                p = (self.counts[i][j] + self.alpha) / denom
                row_h -= p * math.log2(p)
            h += (row_total / total) * row_h
        return h

    # ── scoring ───────────────────────────────────────────────────────────
    def assess(
        self,
        recent_types: Sequence[str],
        min_transitions: int,
        margin_bits: float,
        max_steps: int = 10,
    ) -> SequenceAssessment:
        types = [t for t in recent_types if t in _IDX]
        pairs = list(zip(types[:-1], types[1:]))[-max_steps:]
        expected = self.conditional_entropy()

        if self.total_transitions < min_transitions or len(pairs) < 3:
            return SequenceAssessment(
                active=False, mean_surprise=0.0, expected_surprise=expected,
                excess_bits=0.0, anomalous=False,
            )

        surprises = [self.surprise(a, b) for a, b in pairs]
        mean_s = sum(surprises) / len(surprises)
        excess = mean_s - expected

        worst_idx = max(range(len(pairs)), key=lambda k: surprises[k])
        a, b = pairs[worst_idx]
        row_total = int(sum(self.counts[_IDX[a]]))
        count = int(self.counts[_IDX[a]][_IDX[b]])

        anomalous = excess > margin_bits
        reasons: List[str] = []
        if anomalous:
            reasons.append(
                f"Action order is unusual for this identity: recent transitions "
                f"average {mean_s:.1f} bits of surprise vs {expected:.1f} normally. "
                f"Rarest: {a}→{b} (seen {count} of {row_total} times after '{a}')."
            )
        return SequenceAssessment(
            active=True, mean_surprise=mean_s, expected_surprise=expected,
            excess_bits=excess, anomalous=anomalous,
            rarest_transition=f"{a}->{b}", rarest_count=count,
            rarest_row_total=row_total, reasons=reasons,
        )

    # ── serialisation ─────────────────────────────────────────────────────
    def to_dict(self) -> Dict:
        return {"alpha": self.alpha, "counts": self.counts}

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "SequenceModel":
        m = cls(alpha=float((data or {}).get("alpha", 0.5)))
        counts = (data or {}).get("counts")
        if counts and len(counts) == _N and all(len(r) == _N for r in counts):
            m.counts = [[float(v) for v in r] for r in counts]
        return m

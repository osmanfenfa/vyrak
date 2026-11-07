"""
VolumeBaseline — per-identity data-volume rate baseline.

Why: the strongest exfiltration signal is *how much* data an identity touches
in a short period, not which verbs it uses. A stolen token that exports
40,000 customer records looks like a normal "export" to an action-type model.

How it works
------------
Time is cut into fixed buckets (default 10 minutes). For each bucket we sum
three metrics for the identity across all of its sessions:

    read_units    records read (or 1 per read/search if no count is sent)
    export_units  records exported
    delete_units  records deleted / overwritten (delete + config)

When a bucket closes, its *clean* totals (only actions that raised no
anomaly) update an EWMA mean and variance per metric — so attacks don't
teach the baseline. The *live* totals of the current bucket (all actions)
are checked against

    limit = max(floor, mean + k * sqrt(max(var, mean)), ratio * mean)

before the baseline is mature, a conservative absolute floor is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

METRICS = ("read_units", "export_units", "delete_units")

_ACTION_METRIC = {
    "read": "read_units",
    "search": "read_units",
    "export": "export_units",
    "delete": "delete_units",
    "config": "delete_units",
}

_METRIC_LABEL = {
    "read_units": "records read",
    "export_units": "records exported",
    "delete_units": "records deleted or changed",
}


@dataclass
class VolumeAssessment:
    metric: Optional[str]
    live_total: float
    limit: float
    baseline_mean: float
    mature: bool
    anomalous: bool
    reasons: List[str] = field(default_factory=list)


class VolumeBaseline:
    def __init__(self, bucket_seconds: int = 600, decay: float = 0.9) -> None:
        self.bucket_seconds = int(bucket_seconds)
        self.decay = float(decay)
        self.bucket_start: Optional[float] = None
        self.live: Dict[str, float] = {m: 0.0 for m in METRICS}
        self.clean: Dict[str, float] = {m: 0.0 for m in METRICS}
        self.mean: Dict[str, float] = {m: 0.0 for m in METRICS}
        self.var: Dict[str, float] = {m: 0.0 for m in METRICS}
        self.completed_buckets: int = 0

    @staticmethod
    def units_for(action_type: str, metadata: Optional[Dict]) -> float:
        """Records touched by one action; falls back to 1 when not reported."""
        meta = metadata or {}
        for key in ("records", "record_count", "rows"):
            val = meta.get(key)
            if isinstance(val, (int, float)) and val >= 0:
                return float(val)
        return 1.0

    # ── bucket management ────────────────────────────────────────────────
    def _roll(self, ts: float) -> None:
        if self.bucket_start is None:
            self.bucket_start = ts - (ts % self.bucket_seconds)
            return
        if ts < self.bucket_start + self.bucket_seconds:
            return
        # Close the current bucket: learn from clean totals of active buckets
        if any(self.clean[m] > 0 for m in METRICS):
            for m in METRICS:
                x = self.clean[m]
                if self.completed_buckets == 0:
                    self.mean[m] = x
                    self.var[m] = 0.0
                else:
                    diff = x - self.mean[m]
                    self.mean[m] = self.decay * self.mean[m] + (1 - self.decay) * x
                    self.var[m] = self.decay * (self.var[m] + (1 - self.decay) * diff * diff)
            self.completed_buckets += 1
        self.bucket_start = ts - (ts % self.bucket_seconds)
        self.live = {m: 0.0 for m in METRICS}
        self.clean = {m: 0.0 for m in METRICS}

    # ── per-action API ───────────────────────────────────────────────────
    def observe(
        self,
        action_type: str,
        metadata: Optional[Dict],
        ts: float,
        min_buckets: int,
        k_sigma: float,
        ratio: float,
        floors: Dict[str, float],
    ) -> VolumeAssessment:
        """Add the action to the live bucket and assess it."""
        self._roll(ts)
        metric = _ACTION_METRIC.get(action_type)
        if metric is None:
            return VolumeAssessment(None, 0.0, 0.0, 0.0, self.mature(min_buckets), False)

        self.live[metric] += self.units_for(action_type, metadata)
        live_total = self.live[metric]
        mature = self.mature(min_buckets)
        floor = float(floors.get(metric, 100.0))

        if mature:
            mean = self.mean[metric]
            spread = math.sqrt(max(self.var[metric], mean, 1.0))
            limit = max(floor * 0.2, mean + k_sigma * spread, ratio * mean)
        else:
            mean = self.mean[metric]
            limit = floor

        anomalous = live_total > limit
        reasons: List[str] = []
        if anomalous:
            minutes = self.bucket_seconds // 60
            if mature:
                reasons.append(
                    f"{int(live_total):,} {_METRIC_LABEL[metric]} in the current "
                    f"{minutes}-minute window; this identity normally does about "
                    f"{mean:,.0f} (limit {limit:,.0f})."
                )
            else:
                reasons.append(
                    f"{int(live_total):,} {_METRIC_LABEL[metric]} in the current "
                    f"{minutes}-minute window exceeds the new-identity limit of "
                    f"{limit:,.0f}."
                )
        return VolumeAssessment(metric, live_total, limit, mean, mature, anomalous, reasons)

    def commit_clean(self, action_type: str, metadata: Optional[Dict]) -> None:
        """Count an action toward the learnable (clean) totals."""
        metric = _ACTION_METRIC.get(action_type)
        if metric is not None:
            self.clean[metric] += self.units_for(action_type, metadata)

    def mature(self, min_buckets: int) -> bool:
        return self.completed_buckets >= min_buckets

    # ── serialisation ────────────────────────────────────────────────────
    def to_dict(self) -> Dict:
        return {
            "bucket_seconds": self.bucket_seconds,
            "decay": self.decay,
            "bucket_start": self.bucket_start,
            "live": self.live,
            "clean": self.clean,
            "mean": self.mean,
            "var": self.var,
            "completed_buckets": self.completed_buckets,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict], bucket_seconds: int = 600) -> "VolumeBaseline":
        data = data or {}
        vb = cls(
            bucket_seconds=int(data.get("bucket_seconds", bucket_seconds)),
            decay=float(data.get("decay", 0.9)),
        )
        vb.bucket_start = data.get("bucket_start")
        for attr in ("live", "clean", "mean", "var"):
            stored = data.get(attr) or {}
            setattr(vb, attr, {m: float(stored.get(m, 0.0)) for m in METRICS})
        vb.completed_buckets = int(data.get("completed_buckets", 0))
        return vb

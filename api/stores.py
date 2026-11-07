"""
Redis-backed shared state for the API: sessions, alerts, peer baselines,
tenant settings and rate limits.

Everything here lives in Redis (or the in-memory fallback in development),
so any API worker can serve any request and a restart loses nothing.
All keys are scoped by tenant.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from detectors.peer import PeerGroupBaseline

SESSION_TTL_SECONDS = 24 * 3600
ALERT_INDEX_CAP = 10_000
ALERT_EVENTS_CAP = 50

ALERT_STATUSES = ("open", "confirmed_malicious", "false_positive", "benign_expected")


def _t(tenant_id: str) -> str:
    return f"zelvryn:t:{tenant_id}"


class SessionStateStore:
    """Serialized engine state per session (see ABSAv2Engine.export_state)."""

    def __init__(self, redis_client, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    def _key(self, tenant_id: str, session_id: str) -> str:
        return f"{_t(tenant_id)}:session:{session_id}"

    def load(self, tenant_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(self._key(tenant_id, session_id))
        return json.loads(raw) if raw else None

    def save(self, tenant_id: str, session_id: str, state: Dict[str, Any]) -> None:
        self._redis.set(self._key(tenant_id, session_id), json.dumps(state), ex=self._ttl)

    def delete(self, tenant_id: str, session_id: str) -> bool:
        return bool(self._redis.delete(self._key(tenant_id, session_id)))

    def count(self, tenant_id: Optional[str] = None) -> int:
        pattern = f"{_t(tenant_id)}:session:*" if tenant_id else "zelvryn:t:*:session:*"
        _, keys = self._redis.scan(0, match=pattern, count=10_000)
        return len(keys)


class PeerStore:
    """Peer-group baselines per (tenant, group)."""

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def _key(self, tenant_id: str, group: str) -> str:
        return f"{_t(tenant_id)}:peer:{group}"

    def load(self, tenant_id: str, group: str) -> PeerGroupBaseline:
        raw = self._redis.get(self._key(tenant_id, group))
        return PeerGroupBaseline.from_dict(json.loads(raw) if raw else None)

    def save(self, tenant_id: str, group: str, baseline: PeerGroupBaseline) -> None:
        self._redis.set(self._key(tenant_id, group), json.dumps(baseline.to_dict()))


class TenantSettingsStore:
    """Per-tenant settings. New tenants start in shadow mode (observe only)."""

    DEFAULTS: Dict[str, Any] = {"shadow_mode": True}

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def _key(self, tenant_id: str) -> str:
        return f"{_t(tenant_id)}:settings"

    def get(self, tenant_id: str, fallback_shadow: Optional[bool] = None) -> Dict[str, Any]:
        raw = self._redis.get(self._key(tenant_id))
        settings = dict(self.DEFAULTS)
        if fallback_shadow is not None:
            settings["shadow_mode"] = bool(fallback_shadow)
        if raw:
            settings.update(json.loads(raw))
        return settings

    def update(self, tenant_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        raw = self._redis.get(self._key(tenant_id))
        stored = json.loads(raw) if raw else {}
        stored.update(patch)
        stored["updated_at"] = time.time()
        self._redis.set(self._key(tenant_id), json.dumps(stored))
        return self.get(tenant_id)

    def is_configured(self, tenant_id: str) -> bool:
        return bool(self._redis.exists(self._key(tenant_id)))


class AlertStore:
    """
    One alert per session, opened on the first anomaly and updated as more
    anomalies arrive. Analysts label alerts; labelled alerts become
    analyst-verified training data for the supervised model.
    """

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def _key(self, tenant_id: str, alert_id: str) -> str:
        return f"{_t(tenant_id)}:alert:{alert_id}"

    def _index(self, tenant_id: str) -> str:
        return f"{_t(tenant_id)}:alerts"

    @staticmethod
    def severity_for(decision: str) -> str:
        return {"BLOCK": "high", "CHALLENGE": "medium"}.get(decision, "low")

    def record(
        self,
        tenant_id: str,
        alert_id: Optional[str],
        *,
        session_id: str,
        identity_id: str,
        identity_kind: str,
        decision: str,
        enforced: bool,
        trust_score: float,
        anomalies: List[str],
        reasons: List[str],
        action_type: str,
        timestamp: float,
        features: Optional[List[float]] = None,
    ) -> str:
        """Create or update the session's alert. Returns the alert id."""
        now = time.time()
        alert = self.get(tenant_id, alert_id) if alert_id else None
        if alert is None:
            alert_id = uuid.uuid4().hex
            alert = {
                "alert_id": alert_id,
                "tenant_id": tenant_id,
                "session_id": session_id,
                "identity_id": identity_id,
                "identity_kind": identity_kind,
                "status": "open",
                "severity": "low",
                "created_at": now,
                "events": [],
                "label": None,
            }
            self._redis.lpush(self._index(tenant_id), alert_id)
            self._redis.ltrim(self._index(tenant_id), 0, ALERT_INDEX_CAP - 1)

        sev_rank = {"low": 0, "medium": 1, "high": 2}
        sev = self.severity_for(decision)
        if sev_rank[sev] > sev_rank[alert["severity"]]:
            alert["severity"] = sev
        alert["updated_at"] = now
        alert["latest_trust"] = trust_score
        alert["latest_decision"] = decision
        alert["enforced"] = enforced
        alert["anomaly_types"] = sorted(set(alert.get("anomaly_types", [])) | set(anomalies))
        alert["events"].append({
            "timestamp": timestamp,
            "action_type": action_type,
            "decision": decision,
            "trust_score": trust_score,
            "anomalies": anomalies,
            "reasons": reasons,
            "features": features,
        })
        alert["events"] = alert["events"][-ALERT_EVENTS_CAP:]
        alert["summary"] = reasons[0] if reasons else ", ".join(anomalies)
        self._redis.set(self._key(tenant_id, alert_id), json.dumps(alert))
        return alert_id

    def get(self, tenant_id: str, alert_id: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(self._key(tenant_id, alert_id))
        return json.loads(raw) if raw else None

    def list(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        identity_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        ids = self._redis.lrange(self._index(tenant_id), 0, -1)
        items = []
        for aid in ids:
            a = self.get(tenant_id, aid)
            if not a:
                continue
            if status and a["status"] != status:
                continue
            if severity and a["severity"] != severity:
                continue
            if identity_id and a["identity_id"] != identity_id:
                continue
            items.append(a)
        page = items[offset: offset + limit]
        for a in page:   # list view: drop bulky per-event features
            a["events"] = [{k: v for k, v in e.items() if k != "features"} for e in a["events"]]
        return {"total": len(items), "items": page}

    def label(
        self, tenant_id: str, alert_id: str, status: str, analyst: str, note: str = ""
    ) -> Optional[Dict[str, Any]]:
        if status not in ALERT_STATUSES:
            raise ValueError(f"status must be one of {ALERT_STATUSES}")
        alert = self.get(tenant_id, alert_id)
        if alert is None:
            return None
        alert["status"] = status
        alert["label"] = {"status": status, "analyst": analyst, "note": note, "at": time.time()}
        self._redis.set(self._key(tenant_id, alert_id), json.dumps(alert))
        return alert

    def training_samples(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Analyst-verified samples: confirmed_malicious → 1, false_positive /
        benign_expected → 0. Only events that carry a feature vector.
        """
        out = []
        for aid in self._redis.lrange(self._index(tenant_id), 0, -1):
            a = self.get(tenant_id, aid)
            if not a or a["status"] == "open":
                continue
            label = 1 if a["status"] == "confirmed_malicious" else 0
            for e in a["events"]:
                if e.get("features"):
                    out.append({
                        "features": e["features"], "label": label,
                        "identity_kind": a["identity_kind"], "alert_id": aid,
                        "anomalies": e["anomalies"],
                    })
        return out


class RateLimiter:
    """Fixed-window limiter in Redis: consistent across workers."""

    def __init__(self, redis_client, limit_per_minute: int) -> None:
        self._redis = redis_client
        self.limit = int(limit_per_minute)

    def hit(self, bucket: str) -> int:
        window = int(time.time() // 60)
        key = f"zelvryn:rl:{bucket}:{window}"
        count = int(self._redis.incr(key))
        if count == 1:
            self._redis.expire(key, 120)
        return count

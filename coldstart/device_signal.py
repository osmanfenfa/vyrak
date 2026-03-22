"""
zelvryn/coldstart/device_signals.py
=====================================
DeviceSignalBridge — supplementary risk signals that operate without
behavioural history, bridging the cold start gap from action zero.

Why device signals are needed
------------------------------
The ramp-up period (RampUpManager) relaxes thresholds but cannot eliminate
the cold start gap entirely. A sophisticated attacker who knows Zelvryn is
deployed could attempt to exploit the ramp-up window by attacking immediately
after account creation before any profile has been established.

Device signals provide a parallel risk channel that requires zero behavioural
history. They answer three questions that can be answered from the very first
request:

  1. Is this IP address known to be malicious?
     (IP reputation via external lookup or local blocklist)

  2. Is this device consistent with past logins?
     (Device fingerprint hashing — browser, OS, screen, timezone)

  3. Is this account logging in from two locations too fast to be human?
     (Geo-velocity: same account, two distant locations, short time window)

Scoring
-------
Each signal contributes a risk score in [0.0, 1.0]:
  0.0 = no risk detected
  0.5 = moderate risk (worthy of monitoring)
  1.0 = maximum risk (strong indicator of attack)

The combined DeviceRiskScore aggregates these into a single composite score
and a list of triggered signal codes, which ABSAv2Engine incorporates as an
additional trust penalty independent of the behavioural window.

Integration design
------------------
DeviceSignalBridge is intentionally decoupled from ABSAv2Engine. The engine
calls it optionally via the API layer (api/dependencies.py) rather than
directly. This keeps core/ free of device-signal logic, and allows device
signals to be disabled for deployments where IP and geolocation data is
unavailable or prohibited by privacy policy.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Maximum acceptable distance (km) per hour to flag geo-velocity anomaly.
# A legitimate user cannot travel faster than ~900 km/h (fastest commercial flight).
# We use 1200 km/h as the limit to account for international flight connections.
GEO_VELOCITY_LIMIT_KMH: float = 1200.0

# Default TTL (seconds) for IP reputation cache entries. 1 hour.
IP_REPUTATION_CACHE_TTL: int = 3600

# Minimum time between the same account's logins from different locations (seconds).
GEO_VELOCITY_MIN_WINDOW: int = 300   # 5 minutes

# Risk score thresholds for composite assessment
COMPOSITE_RISK_LOW:    float = 0.25
COMPOSITE_RISK_MEDIUM: float = 0.50
COMPOSITE_RISK_HIGH:   float = 0.75


@dataclass
class DeviceRiskScore:
    """
    Result of a device signal assessment for a single request.

    Attributes
    ----------
    composite_score : float
        Weighted average of all individual signal scores. Range [0.0, 1.0].
    ip_risk : float
        Risk score from IP reputation check. 0.0 = clean, 1.0 = malicious.
    fingerprint_risk : float
        Risk score from device fingerprint check. 0.0 = known device, 1.0 = unknown.
    geo_velocity_risk : float
        Risk score from geo-velocity check. 0.0 = plausible, 1.0 = physically impossible.
    triggered_signals : list of str
        Human-readable codes for each signal that contributed a non-zero risk.
    risk_label : str
        "LOW", "MEDIUM", "HIGH", or "CRITICAL".
    assessed_at : float
        Unix timestamp of the assessment.
    """
    composite_score:    float       = 0.0
    ip_risk:            float       = 0.0
    fingerprint_risk:   float       = 0.0
    geo_velocity_risk:  float       = 0.0
    triggered_signals:  List[str]   = field(default_factory=list)
    risk_label:         str         = "LOW"
    assessed_at:        float       = field(default_factory=time.time)

    def is_high_risk(self) -> bool:
        """Return True if composite_score exceeds the HIGH threshold."""
        return self.composite_score >= COMPOSITE_RISK_HIGH

    def is_medium_risk(self) -> bool:
        """Return True if composite_score is in the MEDIUM zone."""
        return COMPOSITE_RISK_MEDIUM <= self.composite_score < COMPOSITE_RISK_HIGH

    def to_dict(self) -> Dict:
        return {
            "composite_score":   round(self.composite_score, 4),
            "ip_risk":           round(self.ip_risk, 4),
            "fingerprint_risk":  round(self.fingerprint_risk, 4),
            "geo_velocity_risk": round(self.geo_velocity_risk, 4),
            "triggered_signals": self.triggered_signals,
            "risk_label":        self.risk_label,
            "assessed_at":       self.assessed_at,
        }

    def __repr__(self) -> str:
        return (
            f"DeviceRiskScore("
            f"composite={self.composite_score:.3f}, "
            f"label={self.risk_label}, "
            f"signals={self.triggered_signals})"
        )


class DeviceSignalBridge:
    """
    Assesses device-level risk signals for incoming authentication requests.

    All assessment is done synchronously and in-process. In production,
    the IP reputation check can be upgraded to call an external API (e.g.,
    AbuseIPDB, IPQualityScore) by overriding _check_ip_reputation_remote().

    Parameters
    ----------
    ip_blocklist : set of str, optional
        Set of known malicious IP addresses (from threat intel feeds).
        Updated at runtime via add_to_blocklist() / remove_from_blocklist().
    ip_allowlist : set of str, optional
        Set of trusted IP addresses that bypass reputation checks (e.g.,
        corporate office NAT IPs). Always score 0.0 for IP risk.
    geo_velocity_enabled : bool, optional
        Whether to perform geo-velocity checks. Requires that the caller
        provides location data in assess(). Default True.
    cache_ttl : int, optional
        TTL in seconds for the IP reputation cache. Default 3600 (1 hour).

    Attributes
    ----------
    _ip_blocklist : set of str
    _ip_allowlist : set of str
    _ip_cache : dict
        Short-term cache of {ip: (risk_score, expires_at)} to avoid
        redundant lookups within the TTL window.
    _user_location_history : dict
        {user_id: [(latitude, longitude, timestamp), ...]}
        Rolling window of recent login locations for geo-velocity checks.
    _device_fingerprint_db : dict
        {user_id: set of fingerprint hashes}
        Known device fingerprints for each user.
    """

    def __init__(
        self,
        ip_blocklist: Optional[Set[str]] = None,
        ip_allowlist: Optional[Set[str]] = None,
        geo_velocity_enabled: bool = True,
        cache_ttl: int = IP_REPUTATION_CACHE_TTL,
    ) -> None:
        self._ip_blocklist:           Set[str]          = ip_blocklist or set()
        self._ip_allowlist:           Set[str]          = ip_allowlist or set()
        self.geo_velocity_enabled:    bool              = geo_velocity_enabled
        self._cache_ttl:              int               = cache_ttl

        self._ip_cache:               Dict[str, Tuple[float, float]] = {}
        self._user_location_history:  Dict[str, List[Tuple[float, float, float]]] = {}
        self._device_fingerprint_db:  Dict[str, Set[str]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMARY INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def assess(
        self,
        user_id: str,
        ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        screen_resolution: Optional[str] = None,
        timezone_offset: Optional[int] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> DeviceRiskScore:
        """
        Perform a full device signal risk assessment for a login request.

        Runs three independent checks and aggregates their scores into a
        composite DeviceRiskScore.

        Parameters
        ----------
        user_id : str
            The authenticated user's ID (used for per-user history lookups).
        ip : str, optional
            Client IP address. IPv4 or IPv6.
        user_agent : str, optional
            Browser User-Agent string.
        screen_resolution : str, optional
            Client screen resolution (e.g., "1920x1080"). Included in fingerprint.
        timezone_offset : int, optional
            UTC offset in minutes (e.g., 60 = UTC+1).
        latitude : float, optional
            Client latitude for geo-velocity check.
        longitude : float, optional
            Client longitude for geo-velocity check.
        timestamp : float, optional
            Request timestamp. Defaults to time.time().

        Returns
        -------
        DeviceRiskScore
            Aggregated risk assessment with per-signal scores and labels.
        """
        ts = timestamp or time.time()
        signals: List[str] = []

        # ── Signal 1: IP Reputation ───────────────────────────────────────
        ip_risk = 0.0
        if ip:
            ip_risk, ip_signals = self._check_ip_reputation(ip)
            signals.extend(ip_signals)

        # ── Signal 2: Device Fingerprint ──────────────────────────────────
        fp_risk = 0.0
        if any(v is not None for v in [user_agent, screen_resolution, timezone_offset]):
            fp_hash = self._compute_fingerprint(
                user_agent=user_agent,
                screen_resolution=screen_resolution,
                timezone_offset=timezone_offset,
            )
            fp_risk, fp_signals = self._check_fingerprint(user_id, fp_hash)
            signals.extend(fp_signals)

            # Register this fingerprint in the known-device database
            self._register_fingerprint(user_id, fp_hash)

        # ── Signal 3: Geo-velocity ────────────────────────────────────────
        geo_risk = 0.0
        if (self.geo_velocity_enabled
                and latitude is not None
                and longitude is not None):
            geo_risk, geo_signals = self._check_geo_velocity(
                user_id, latitude, longitude, ts
            )
            signals.extend(geo_signals)

            # Record this location in history
            self._record_location(user_id, latitude, longitude, ts)

        # ── Composite score ───────────────────────────────────────────────
        # Weights: IP reputation is most reliable (0.5), fingerprint moderate
        # (0.3), geo-velocity context-dependent (0.2)
        weights = {"ip": 0.5, "fp": 0.3, "geo": 0.2}
        composite = (
            weights["ip"]  * ip_risk
            + weights["fp"]  * fp_risk
            + weights["geo"] * geo_risk
        )
        composite = min(1.0, composite)

        label = self._risk_label(composite)

        result = DeviceRiskScore(
            composite_score   = composite,
            ip_risk           = ip_risk,
            fingerprint_risk  = fp_risk,
            geo_velocity_risk = geo_risk,
            triggered_signals = signals,
            risk_label        = label,
            assessed_at       = ts,
        )

        if signals:
            logger.info(
                "Device risk assessment: user=%s composite=%.3f label=%s signals=%s",
                user_id, composite, label, signals
            )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCKLIST AND ALLOWLIST MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def add_to_blocklist(self, ip: str) -> None:
        """Add an IP to the local blocklist and invalidate its cache entry."""
        self._ip_blocklist.add(ip)
        self._ip_cache.pop(ip, None)
        logger.info("IP added to blocklist: %s", ip)

    def remove_from_blocklist(self, ip: str) -> None:
        """Remove an IP from the local blocklist."""
        self._ip_blocklist.discard(ip)
        logger.info("IP removed from blocklist: %s", ip)

    def add_to_allowlist(self, ip: str) -> None:
        """Add an IP to the allowlist (bypasses reputation checks)."""
        self._ip_allowlist.add(ip)
        self._ip_cache.pop(ip, None)
        logger.info("IP added to allowlist: %s", ip)

    def register_known_device(self, user_id: str, fingerprint_hash: str) -> None:
        """
        Pre-register a device fingerprint as trusted for a user.

        Used during account setup or when a user explicitly trusts a device
        via the appeal mechanism.

        Parameters
        ----------
        user_id : str
        fingerprint_hash : str
            SHA-256 hash of the device fingerprint components.
        """
        if user_id not in self._device_fingerprint_db:
            self._device_fingerprint_db[user_id] = set()
        self._device_fingerprint_db[user_id].add(fingerprint_hash)
        logger.info("Registered trusted device for user=%s", user_id)

    def clear_user_history(self, user_id: str) -> None:
        """
        Remove all device and location history for a user.

        Called on profile deletion (GDPR right to erasure).

        Parameters
        ----------
        user_id : str
        """
        self._device_fingerprint_db.pop(user_id, None)
        self._user_location_history.pop(user_id, None)
        logger.info("Device history cleared for user=%s (GDPR erasure)", user_id)

    def known_device_count(self, user_id: str) -> int:
        """Return the number of registered device fingerprints for a user."""
        return len(self._device_fingerprint_db.get(user_id, set()))

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: IP REPUTATION
    # ─────────────────────────────────────────────────────────────────────────

    def _check_ip_reputation(self, ip: str) -> Tuple[float, List[str]]:
        """
        Check IP reputation using the local blocklist and cache.

        In production, override or extend _check_ip_reputation_remote() to
        call an external threat intel API (AbuseIPDB, IPQualityScore, etc.).

        Parameters
        ----------
        ip : str

        Returns
        -------
        tuple of (float, list of str)
            (risk_score, signal_codes)
        """
        signals = []

        # Allowlisted IPs always score 0.0
        if ip in self._ip_allowlist:
            return (0.0, [])

        # Check blocklist first (cheapest)
        if ip in self._ip_blocklist:
            signals.append(f"IP_BLOCKLISTED:{ip}")
            return (1.0, signals)

        # Check cache
        if ip in self._ip_cache:
            cached_score, expires_at = self._ip_cache[ip]
            if time.time() < expires_at:
                if cached_score > 0:
                    signals.append(f"IP_CACHED_RISK:{ip}:{cached_score:.2f}")
                return (cached_score, signals)

        # Local heuristics: private/loopback addresses are trusted
        if self._is_private_ip(ip):
            self._ip_cache[ip] = (0.0, time.time() + self._cache_ttl)
            return (0.0, [])

        # In production, call external API here:
        # score = self._check_ip_reputation_remote(ip)
        score = 0.0   # default: no information = no risk

        self._ip_cache[ip] = (score, time.time() + self._cache_ttl)

        if score > 0.3:
            signals.append(f"IP_REPUTATION_RISK:{ip}:{score:.2f}")

        return (score, signals)

    def _check_ip_reputation_remote(self, ip: str) -> float:
        """
        Override this method to integrate a real IP reputation API.

        Example integration (AbuseIPDB):
          import requests
          resp = requests.get(
              "https://api.abuseipdb.com/api/v2/check",
              headers={"Key": API_KEY, "Accept": "application/json"},
              params={"ipAddress": ip, "maxAgeInDays": 90}
          )
          data = resp.json()["data"]
          return data["abuseConfidenceScore"] / 100.0

        Parameters
        ----------
        ip : str

        Returns
        -------
        float
            Risk score in [0.0, 1.0].
        """
        return 0.0   # placeholder — no external call in base implementation

    @staticmethod
    def _is_private_ip(ip: str) -> bool:
        """Return True if the IP is a private, loopback, or link-local address."""
        private_prefixes = (
            "10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
            "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
            "127.", "::1", "fe80:", "fc00:", "fd"
        )
        return any(ip.startswith(p) for p in private_prefixes)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: DEVICE FINGERPRINTING
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_fingerprint(
        user_agent:        Optional[str],
        screen_resolution: Optional[str],
        timezone_offset:   Optional[int],
    ) -> str:
        """
        Compute a SHA-256 fingerprint hash from available device signals.

        The fingerprint captures the device's software and display configuration.
        It is not a true unique identifier (many users share identical fingerprints)
        but is sufficient to flag when a user's session transitions to a completely
        different device type (desktop → mobile, or specific browser switch).

        Parameters
        ----------
        user_agent : str or None
        screen_resolution : str or None
        timezone_offset : int or None

        Returns
        -------
        str
            First 32 hex characters of the SHA-256 hash.
        """
        components = [
            user_agent        or "unknown_ua",
            screen_resolution or "unknown_res",
            str(timezone_offset) if timezone_offset is not None else "unknown_tz",
        ]
        raw = "|".join(components).lower()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _check_fingerprint(
        self,
        user_id: str,
        fp_hash: str,
    ) -> Tuple[float, List[str]]:
        """
        Check a device fingerprint against the user's known devices.

        Parameters
        ----------
        user_id : str
        fp_hash : str
            Computed fingerprint hash.

        Returns
        -------
        tuple of (float, list of str)
            (risk_score, signal_codes)
        """
        signals = []
        known_fingerprints = self._device_fingerprint_db.get(user_id, set())

        if not known_fingerprints:
            # No history yet — first device is always trusted (can't flag
            # an unknown against nothing). Return low but non-zero risk.
            return (0.1, ["FINGERPRINT_FIRST_DEVICE"])

        if fp_hash in known_fingerprints:
            return (0.0, [])   # known device

        # Unknown device — risk scales with how many known devices already exist.
        # A user with 3 known devices switching to a 4th is more suspicious
        # than a user with 0 known devices seeing device 1.
        n_known = len(known_fingerprints)
        base_risk = 0.4
        scaling   = min(1.0, 0.1 * n_known)
        risk      = min(0.9, base_risk + scaling)

        signals.append(f"FINGERPRINT_UNKNOWN_DEVICE:known={n_known}")
        return (risk, signals)

    def _register_fingerprint(self, user_id: str, fp_hash: str) -> None:
        """Add a fingerprint to the user's known device set."""
        if user_id not in self._device_fingerprint_db:
            self._device_fingerprint_db[user_id] = set()
        self._device_fingerprint_db[user_id].add(fp_hash)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: GEO-VELOCITY
    # ─────────────────────────────────────────────────────────────────────────

    def _check_geo_velocity(
        self,
        user_id: str,
        lat: float,
        lon: float,
        ts: float,
    ) -> Tuple[float, List[str]]:
        """
        Check whether the user's current location is physically reachable
        from their most recent login location given the elapsed time.

        Uses the Haversine formula to compute great-circle distance between
        two coordinates, then checks whether the required travel speed exceeds
        GEO_VELOCITY_LIMIT_KMH.

        Parameters
        ----------
        user_id : str
        lat : float
            Current latitude.
        lon : float
            Current longitude.
        ts : float
            Current request timestamp.

        Returns
        -------
        tuple of (float, list of str)
        """
        signals = []
        history = self._user_location_history.get(user_id, [])

        if not history:
            return (0.0, [])   # no prior location to compare

        # Find the most recent location entry
        prev_lat, prev_lon, prev_ts = history[-1]

        elapsed_seconds = ts - prev_ts
        if elapsed_seconds < GEO_VELOCITY_MIN_WINDOW:
            # Locations too close in time — geo-velocity check unreliable
            return (0.0, [])

        distance_km = self._haversine_km(prev_lat, prev_lon, lat, lon)

        if distance_km < 50:
            # Less than 50km — same city, no concern
            return (0.0, [])

        elapsed_hours = elapsed_seconds / 3600.0
        required_speed_kmh = distance_km / elapsed_hours if elapsed_hours > 0 else float("inf")

        if required_speed_kmh <= GEO_VELOCITY_LIMIT_KMH:
            return (0.0, [])   # plausible travel speed

        # Impossible or highly suspicious velocity
        # Risk scales with how much the limit is exceeded
        excess_ratio = required_speed_kmh / GEO_VELOCITY_LIMIT_KMH
        risk = min(1.0, 0.4 + 0.2 * math.log2(excess_ratio))

        signals.append(
            f"GEO_VELOCITY:{required_speed_kmh:.0f}km/h "
            f"(limit={GEO_VELOCITY_LIMIT_KMH:.0f} dist={distance_km:.0f}km)"
        )

        logger.warning(
            "Geo-velocity anomaly: user=%s speed=%.0f km/h distance=%.0f km "
            "elapsed=%.0f s risk=%.3f",
            user_id, required_speed_kmh, distance_km, elapsed_seconds, risk
        )

        return (risk, signals)

    def _record_location(
        self,
        user_id: str,
        lat: float,
        lon: float,
        ts: float,
    ) -> None:
        """
        Record a location entry in the user's login history.

        Keeps only the last 10 entries to bound memory growth.

        Parameters
        ----------
        user_id : str
        lat : float
        lon : float
        ts : float
        """
        if user_id not in self._user_location_history:
            self._user_location_history[user_id] = []

        self._user_location_history[user_id].append((lat, lon, ts))

        # Keep only last 10 location records per user
        if len(self._user_location_history[user_id]) > 10:
            self._user_location_history[user_id] = (
                self._user_location_history[user_id][-10:]
            )

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Compute the great-circle distance between two coordinate pairs.

        Uses the Haversine formula for accuracy over long distances.

        Parameters
        ----------
        lat1, lon1, lat2, lon2 : float
            Coordinates in decimal degrees.

        Returns
        -------
        float
            Distance in kilometres.
        """
        R = 6371.0   # Earth's radius in km

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: RISK LABEL
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _risk_label(score: float) -> str:
        if score >= COMPOSITE_RISK_HIGH:
            return "CRITICAL" if score >= 0.9 else "HIGH"
        elif score >= COMPOSITE_RISK_MEDIUM:
            return "MEDIUM"
        elif score >= COMPOSITE_RISK_LOW:
            return "LOW"
        return "CLEAN"

    # ─────────────────────────────────────────────────────────────────────────
    # Dunder methods
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DeviceSignalBridge("
            f"blocklist_size={len(self._ip_blocklist)}, "
            f"allowlist_size={len(self._ip_allowlist)}, "
            f"geo_velocity={'on' if self.geo_velocity_enabled else 'off'}, "
            f"tracked_users={len(self._device_fingerprint_db)})"
        )
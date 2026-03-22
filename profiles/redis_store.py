"""
zelvryn/profiles/redis_store.py
=================================
RedisProfileStore — persists BehavioralProfile objects in Redis for
sub-millisecond read/write access across a horizontally scaled API fleet.

Why Redis for profiles
-----------------------
The ABSAv2Engine is stateless by design: each evaluate_action() call reads
the profile at the start and writes it at the end. For this to work across
multiple API server instances (horizontal scaling), the profile must live
in a shared, fast store — not in each server's local memory.

Redis provides:
  - Sub-millisecond reads (< 1ms at p99 for hash gets)
  - Atomic writes (HSET is atomic — no partial updates)
  - TTL-based auto-expiry (inactive profiles expire after 90 days)
  - Cluster mode for horizontal Redis scaling
  - AOF persistence (prevents profile loss on Redis restart)

Serialisation strategy
-----------------------
BehavioralProfile objects are serialised to JSON for portability and
debugging ease. In high-throughput deployments (>100k sessions/hour),
replace JSON with msgpack for ~3x smaller payloads and faster encode/decode.

The serialised profile includes all statistical fields but NOT the
ml_training_data (which can be very large). Training data is managed
separately by the FeatureStore (ml/feature_store.py).

Connection management
---------------------
The store accepts a pre-built redis.Redis client, enabling:
  - Connection pooling (recommended for production)
  - Cluster mode (redis.RedisCluster)
  - TLS connections (redis.Redis(ssl=True))
  - Testing with fakeredis (a drop-in in-memory Redis replacement)

GDPR compliance
---------------
delete(user_id) removes the profile and all associated keys permanently.
This is the technical implementation of the right to erasure (GDPR Article 17).
The API layer ensures deletion happens within 72 hours of a user request.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from core.profile import BehavioralProfile
from core.action import UserAction
from core.constants import EWMA_DECAY_FACTOR

logger = logging.getLogger(__name__)

# Redis key prefixes
_KEY_PREFIX   = "zelvryn:profile:"
_META_PREFIX  = "zelvryn:meta:"
_LOCK_PREFIX  = "zelvryn:lock:"

# Default TTL for profile keys (90 days)
DEFAULT_PROFILE_TTL: int = 90 * 24 * 3600

# Timeout for profile lock acquisition (milliseconds)
LOCK_TIMEOUT_MS: int = 500


class ProfileStorageError(Exception):
    """Raised when a profile storage operation fails."""
    pass


class RedisProfileStore:
    """
    Persists and retrieves BehavioralProfile objects using Redis.

    Parameters
    ----------
    redis_client : redis.Redis or compatible, optional
        Pre-built Redis client. If None, a default localhost client is created.
        Pass fakeredis.FakeRedis() for unit testing without infrastructure.
    ttl_seconds : int, optional
        Profile key TTL in seconds. Default 90 days.
    key_prefix : str, optional
        Namespace prefix for all keys. Default "zelvryn:profile:".

    Example
    -------
    Production (with connection pool):
      import redis
      pool = redis.ConnectionPool.from_url(REDIS_URL, max_connections=50)
      client = redis.Redis(connection_pool=pool)
      store = RedisProfileStore(redis_client=client)

    Testing (in-memory, no Redis needed):
      import fakeredis
      store = RedisProfileStore(redis_client=fakeredis.FakeRedis())

    Async usage (FastAPI):
      import redis.asyncio as aioredis
      client = aioredis.from_url(REDIS_URL)
      store = RedisProfileStore(redis_client=client)
      # Use await store.get_async(user_id) etc.
    """

    def __init__(
        self,
        redis_client = None,
        ttl_seconds:  int = DEFAULT_PROFILE_TTL,
        key_prefix:   str = _KEY_PREFIX,
    ) -> None:
        self._redis      = redis_client
        self._ttl        = ttl_seconds
        self._key_prefix = key_prefix

        if self._redis is None:
            self._redis = self._create_default_client()

        logger.info(
            "RedisProfileStore: ttl=%ds prefix=%s",
            ttl_seconds, key_prefix
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SYNCHRONOUS INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def get(self, user_id: str) -> Optional[BehavioralProfile]:
        """
        Load a BehavioralProfile from Redis.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        BehavioralProfile or None
            None if no profile exists for this user.

        Raises
        ------
        ProfileStorageError
            If Redis is unreachable or deserialisation fails.
        """
        key = self._profile_key(user_id)
        try:
            raw = self._redis.get(key)
        except Exception as e:
            raise ProfileStorageError(
                f"Redis GET failed for user={user_id}: {e}"
            ) from e

        if raw is None:
            return None

        try:
            data    = json.loads(raw)
            profile = BehavioralProfile.from_dict(data)
            logger.debug(
                "Profile loaded: user=%s vector_count=%d",
                user_id, profile.vector_count
            )
            return profile
        except Exception as e:
            logger.error(
                "Profile deserialisation failed for user=%s: %s",
                user_id, e, exc_info=True
            )
            raise ProfileStorageError(
                f"Deserialisation failed for user={user_id}: {e}"
            ) from e

    def save(self, user_id: str, profile: BehavioralProfile) -> bool:
        """
        Persist a BehavioralProfile to Redis.

        Serialises the profile to JSON and stores it with TTL. The TTL is
        extended on every save, so active users never have their profile expire.

        Parameters
        ----------
        user_id : str
        profile : BehavioralProfile

        Returns
        -------
        bool
            True on success.

        Raises
        ------
        ProfileStorageError
        """
        key = self._profile_key(user_id)
        try:
            data = profile.to_dict()
            raw  = json.dumps(data)
            self._redis.set(key, raw, ex=self._ttl)
            self._update_meta(user_id, profile)
            logger.debug(
                "Profile saved: user=%s vector_count=%d size_bytes=%d",
                user_id, profile.vector_count, len(raw)
            )
            return True
        except Exception as e:
            raise ProfileStorageError(
                f"Redis SET failed for user={user_id}: {e}"
            ) from e

    def delete(self, user_id: str) -> bool:
        """
        Delete a user's profile from Redis (GDPR right to erasure).

        Removes the profile key, metadata key, and any lock keys.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        bool
            True if a profile was found and deleted. False if not found.
        """
        keys_to_delete = [
            self._profile_key(user_id),
            self._meta_key(user_id),
            self._lock_key(user_id),
        ]
        try:
            deleted = self._redis.delete(*keys_to_delete)
            logger.info(
                "Profile deleted (GDPR erasure): user=%s keys_removed=%d",
                user_id, deleted
            )
            return deleted > 0
        except Exception as e:
            raise ProfileStorageError(
                f"Redis DELETE failed for user={user_id}: {e}"
            ) from e

    def exists(self, user_id: str) -> bool:
        """
        Check if a profile exists for the given user.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        bool
        """
        try:
            return bool(self._redis.exists(self._profile_key(user_id)))
        except Exception as e:
            logger.warning("Redis EXISTS check failed for user=%s: %s", user_id, e)
            return False

    def ttl_extend(self, user_id: str) -> bool:
        """
        Reset the TTL for a user's profile without modifying the profile data.

        Called at session end to ensure active users' profiles don't expire.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        bool
            True if the key existed and its TTL was extended.
        """
        try:
            result = self._redis.expire(self._profile_key(user_id), self._ttl)
            return bool(result)
        except Exception as e:
            logger.warning(
                "TTL extend failed for user=%s: %s", user_id, e
            )
            return False

    def get_or_create(
        self,
        user_id: str,
        fallback_profile: Optional[BehavioralProfile] = None,
    ) -> BehavioralProfile:
        """
        Load a profile or create a new one if none exists.

        Parameters
        ----------
        user_id : str
        fallback_profile : BehavioralProfile, optional
            Profile to return (and save) if no existing profile is found.
            If None, creates an empty profile.

        Returns
        -------
        BehavioralProfile
        """
        profile = self.get(user_id)
        if profile is not None:
            return profile

        # Create and persist a new profile
        new_profile = fallback_profile or BehavioralProfile(user_id=user_id)
        try:
            self.save(user_id, new_profile)
        except ProfileStorageError as e:
            logger.warning(
                "Could not persist new profile for user=%s: %s. "
                "Using in-memory only.", user_id, e
            )
        return new_profile

    # ─────────────────────────────────────────────────────────────────────────
    # METADATA AND ADMIN
    # ─────────────────────────────────────────────────────────────────────────

    def get_profile_meta(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Return lightweight metadata for a profile without loading the full data.

        Metadata includes: vector_count, last_updated, adaptive_threshold,
        profile_age_days, ttl_remaining_s.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        dict or None
        """
        try:
            raw = self._redis.get(self._meta_key(user_id))
            if raw is None:
                return None
            meta = json.loads(raw)
            # Add live TTL
            ttl_remaining = self._redis.ttl(self._profile_key(user_id))
            meta["ttl_remaining_s"] = ttl_remaining
            return meta
        except Exception as e:
            logger.warning("Meta fetch failed for user=%s: %s", user_id, e)
            return None

    def list_active_users(
        self,
        pattern: str = "*",
        max_results: int = 1000,
    ) -> List[str]:
        """
        Return a list of user IDs that have profiles in Redis.

        Uses SCAN (non-blocking) rather than KEYS (blocking).

        Parameters
        ----------
        pattern : str
            Glob pattern for user IDs. Default "*" = all users.
        max_results : int
            Maximum user IDs to return.

        Returns
        -------
        list of str
        """
        full_pattern = self._profile_key(pattern)
        prefix_len   = len(self._key_prefix)
        user_ids     = []

        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(
                    cursor=cursor, match=full_pattern, count=100
                )
                for key in keys:
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")
                    user_ids.append(key[prefix_len:])
                    if len(user_ids) >= max_results:
                        return user_ids
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning("SCAN failed: %s", e)

        return user_ids

    def store_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the profile store.

        Returns
        -------
        dict
        """
        try:
            info = self._redis.info("memory")
            keyspace = self._redis.info("keyspace")
            profile_count = len(self.list_active_users(max_results=10000))
            return {
                "profile_count":     profile_count,
                "used_memory_mb":    round(
                    info.get("used_memory", 0) / (1024 ** 2), 2
                ),
                "used_memory_peak_mb": round(
                    info.get("used_memory_peak", 0) / (1024 ** 2), 2
                ),
                "ttl_seconds":       self._ttl,
                "key_prefix":        self._key_prefix,
            }
        except Exception as e:
            return {"error": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # ASYNC WRAPPERS (for FastAPI / asyncio deployments)
    # ─────────────────────────────────────────────────────────────────────────

    async def get_async(self, user_id: str) -> Optional[BehavioralProfile]:
        """
        Async version of get(). Use with redis.asyncio clients.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        BehavioralProfile or None
        """
        key = self._profile_key(user_id)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        return BehavioralProfile.from_dict(data)

    async def save_async(
        self,
        user_id: str,
        profile: BehavioralProfile,
    ) -> bool:
        """
        Async version of save(). Use with redis.asyncio clients.
        """
        key = self._profile_key(user_id)
        raw = json.dumps(profile.to_dict())
        await self._redis.set(key, raw, ex=self._ttl)
        return True

    async def delete_async(self, user_id: str) -> bool:
        """
        Async version of delete(). Use with redis.asyncio clients.
        """
        keys = [
            self._profile_key(user_id),
            self._meta_key(user_id),
        ]
        deleted = await self._redis.delete(*keys)
        return deleted > 0

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _profile_key(self, user_id: str) -> str:
        return f"{self._key_prefix}{user_id}"

    def _meta_key(self, user_id: str) -> str:
        return f"{_META_PREFIX}{user_id}"

    def _lock_key(self, user_id: str) -> str:
        return f"{_LOCK_PREFIX}{user_id}"

    def _update_meta(
        self,
        user_id: str,
        profile: BehavioralProfile,
    ) -> None:
        """
        Store lightweight profile metadata in a separate key for fast access.
        This avoids loading the full (potentially large) profile JSON just to
        check staleness or display dashboard information.
        """
        meta = {
            "user_id":            user_id,
            "vector_count":       profile.vector_count,
            "last_updated":       profile.last_updated,
            "adaptive_threshold": round(profile.adaptive_threshold, 4),
            "is_mature":          profile.is_mature(),
            "training_samples":   len(profile.ml_training_data),
            "meta_updated_at":    time.time(),
        }
        try:
            self._redis.set(
                self._meta_key(user_id),
                json.dumps(meta),
                ex=self._ttl
            )
        except Exception:
            pass   # metadata is best-effort — don't fail the main save

    @staticmethod
    def _create_default_client():
        """
        Create a default Redis client pointing to localhost:6379.

        In production, always pass a pre-built client with connection pooling.
        This default is suitable for development and testing only.
        """
        try:
            import redis
            client = redis.Redis(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            client.ping()   # test connectivity
            logger.info("Default Redis client connected: localhost:6379")
            return client
        except Exception as e:
            logger.warning(
                "Could not connect to default Redis (localhost:6379): %s. "
                "Pass a redis_client explicitly or use fakeredis for testing.",
                e
            )
            # Return a no-op stub so the store doesn't crash at import time
            return _NoOpRedisStub()

    def __repr__(self) -> str:
        return (
            f"RedisProfileStore("
            f"ttl={self._ttl}s, "
            f"prefix={self._key_prefix!r})"
        )


class _NoOpRedisStub:
    """
    A minimal no-op stub used when Redis is unavailable during testing.
    All operations return safe defaults without raising exceptions.
    """
    _store: Dict[str, str] = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: str, ex: int = 0):
        self._store[key] = value
        return True

    def delete(self, *keys):
        count = sum(1 for k in keys if k in self._store)
        for k in keys:
            self._store.pop(k, None)
        return count

    def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    def expire(self, key: str, seconds: int) -> int:
        return 1 if key in self._store else 0

    def ttl(self, key: str) -> int:
        return -1

    def scan(self, cursor: int = 0, match: str = "*", count: int = 100):
        return (0, [])

    def info(self, section: str = "all") -> Dict:
        return {}

    def ping(self) -> bool:
        return True

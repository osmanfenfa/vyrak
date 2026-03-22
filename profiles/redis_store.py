"""
RedisProfileStore — persists BehavioralProfile objects in Redis for
sub-millisecond read/write access across a horizontally scaled API fleet.
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

    # SYNCHRONOUS INTERFACE
    def get(self, user_id: str) -> Optional[BehavioralProfile]:
        """
        Load a BehavioralProfile from Redis by user ID
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
        """
        try:
            return bool(self._redis.exists(self._profile_key(user_id)))
        except Exception as e:
            logger.warning("Redis EXISTS check failed for user=%s: %s", user_id, e)
            return False

    def ttl_extend(self, user_id: str) -> bool:
        """
        Reset the TTL for a user's profile without modifying the profile data.
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

    # METADATA AND ADMIN

    def get_profile_meta(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Return lightweight metadata for a profile without loading the full data.
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
        
    # ASYNC WRAPPERS (for FastAPI / asyncio deployments)

    async def get_async(self, user_id: str) -> Optional[BehavioralProfile]:
        """
        Async version of get(). Use with redis.asyncio clients.
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
    
    # PRIVATE HELPERS
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
        Create a default Redis client
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

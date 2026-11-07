"""
MemoryRedis — a small, thread-safe, in-process stand-in for the subset of the
redis-py API Zelvryn uses. Development and tests only: data is lost on restart
and is not shared between worker processes. Production refuses to start
without a real Redis (see api.main).
"""

from __future__ import annotations

import fnmatch
import threading
import time
from typing import Dict, List, Optional, Tuple


class MemoryRedis:
    def __init__(self) -> None:
        self._data: Dict[str, str] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()

    # ── internals ──────────────────────────────────────────────────────────
    def _alive(self, key: str) -> bool:
        exp = self._expiry.get(key)
        if exp is not None and exp <= time.time():
            self._data.pop(key, None)
            self._expiry.pop(key, None)
            return False
        return key in self._data

    # ── strings ────────────────────────────────────────────────────────────
    def get(self, key: str) -> Optional[str]:
        with self._lock:
            return self._data[key] if self._alive(key) else None

    def set(self, key: str, value, ex: Optional[int] = None, px: Optional[int] = None,
            nx: bool = False, **_kw) -> Optional[bool]:
        with self._lock:
            if nx and self._alive(key):
                return None
            self._data[key] = value if isinstance(value, str) else str(value)
            self._expiry.pop(key, None)
            if ex:
                self._expiry[key] = time.time() + float(ex)
            elif px:
                self._expiry[key] = time.time() + float(px) / 1000.0
            return True

    def incr(self, key: str, amount: int = 1) -> int:
        with self._lock:
            cur = int(self._data[key]) if self._alive(key) else 0
            cur += amount
            self._data[key] = str(cur)
            return cur

    def delete(self, *keys: str) -> int:
        with self._lock:
            n = 0
            for k in keys:
                if self._alive(k):
                    n += 1
                self._data.pop(k, None)
                self._expiry.pop(k, None)
            return n

    def exists(self, *keys: str) -> int:
        with self._lock:
            return sum(1 for k in keys if self._alive(k))

    def expire(self, key: str, seconds: int) -> bool:
        with self._lock:
            if not self._alive(key):
                return False
            self._expiry[key] = time.time() + float(seconds)
            return True

    def ttl(self, key: str) -> int:
        with self._lock:
            if not self._alive(key):
                return -2
            exp = self._expiry.get(key)
            return -1 if exp is None else max(0, int(exp - time.time()))

    # ── lists (stored JSON-encoded so string semantics stay uniform) ────────
    def _list(self, key: str) -> List[str]:
        import json
        raw = self._data[key] if self._alive(key) else None
        return json.loads(raw) if raw else []

    def _put_list(self, key: str, items: List[str]) -> None:
        import json
        self._data[key] = json.dumps(items)

    def lpush(self, key: str, *values) -> int:
        with self._lock:
            items = self._list(key)
            for v in values:
                items.insert(0, v if isinstance(v, str) else str(v))
            self._put_list(key, items)
            return len(items)

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        with self._lock:
            items = self._list(key)
            end = len(items) - 1 if end == -1 else end
            return items[start:end + 1]

    def ltrim(self, key: str, start: int, end: int) -> bool:
        with self._lock:
            items = self._list(key)
            end = len(items) - 1 if end == -1 else end
            self._put_list(key, items[start:end + 1])
            return True

    def llen(self, key: str) -> int:
        with self._lock:
            return len(self._list(key))

    # ── iteration ──────────────────────────────────────────────────────────
    def scan(self, cursor: int = 0, match: str = "*", count: int = 100) -> Tuple[int, List[str]]:
        with self._lock:
            keys = [k for k in list(self._data) if self._alive(k) and fnmatch.fnmatchcase(k, match)]
            return 0, keys

    def scan_iter(self, match: str = "*", count: int = 100):
        _, keys = self.scan(match=match)
        return iter(keys)

    # ── misc ───────────────────────────────────────────────────────────────
    def ping(self) -> bool:
        return True

    def info(self, section: str = "all") -> Dict:
        with self._lock:
            return {"backend": "memory", "keys": len(self._data)}

    def flushall(self) -> bool:
        with self._lock:
            self._data.clear()
            self._expiry.clear()
            return True

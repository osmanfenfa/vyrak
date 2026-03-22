"""
Every interaction a user has with the protected application is represented
as a UserAction before it is evaluated by the engine.
"""

import hashlib
import time
from typing import Dict, Optional, Any

class UserAction:
    """
    Represents a single user action event within a Zelvryn-monitored session.
    """

    # ORDER IS FIXED.  Index in this list == numeric encoding in feature vector.
    ACTION_TYPES: tuple = (
        "login",   
        "read",    
        "write",   
        "delete",  
        "search",  
        "export", 
        "config",  
        "admin",   
    )

    # Pre-built lookup for O(1) index retrieval
    _TYPE_INDEX: Dict[str, int] = {t: i for i, t in enumerate(ACTION_TYPES)}
    SENSITIVE_TYPES: frozenset = frozenset({"delete", "export", "admin", "config"})

    def __init__(
        self,
        action_type: str,
        resource_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        _timestamp: Optional[float] = None, 
    ) -> None:
        # Type validation 
        if not isinstance(action_type, str):
            raise TypeError(
                f"action_type must be str, got {type(action_type).__name__}"
            )
        if not isinstance(resource_id, str):
            raise TypeError(
                f"resource_id must be str, got {type(resource_id).__name__}"
            )

        # Value validation
        action_type = action_type.strip().lower()
        if action_type not in self._TYPE_INDEX:
            raise ValueError(
                f"Unknown action_type '{action_type}'. "
                f"Must be one of: {', '.join(self.ACTION_TYPES)}"
            )

        self.action_type: str = action_type

        # Resource ID hashing
        # Truncate to 16 hex chars (64-bit collision space).
        self.resource_id: str = (
            hashlib.sha256(resource_id.encode("utf-8", errors="replace"))
            .hexdigest()[:16]
        )

        # Timestamp
        self.timestamp: float = _timestamp if _timestamp is not None else time.time()
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

    # Numeric encoding

    def to_index(self) -> int:
        """
        Return the integer index of this action type in ACTION_TYPES.
        """
        return self._TYPE_INDEX[self.action_type]

    @property
    def is_sensitive(self) -> bool:
        """True if this action type is in the sensitive set (delete/export/admin/config)."""
        return self.action_type in self.SENSITIVE_TYPES

    @property
    def is_destructive(self) -> bool:
        """True for actions that remove or overwrite data (delete, config)."""
        return self.action_type in {"delete", "config"}

    @property
    def is_exfiltration_risk(self) -> bool:
        """True for actions associated with data exfiltration (export, admin)."""
        return self.action_type in {"export", "admin"}

    # Serialisation
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise to a plain dictionary suitable for JSON encoding or
        storage in the database.
        """
        return {
            "action_type": self.action_type,
            "resource_id": self.resource_id,
            "timestamp":   self.timestamp,
            "metadata":    self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserAction":
        """
        Reconstruct a UserAction from a serialised dictionary.
        UserAction
        """
        obj = cls.__new__(cls)
        action_type = str(data["action_type"]).strip().lower()
        if action_type not in cls._TYPE_INDEX:
            raise ValueError(
                f"Unknown action_type '{action_type}' in from_dict(). "
                f"Must be one of: {', '.join(cls.ACTION_TYPES)}"
            )
        obj.action_type = action_type
        obj.resource_id = str(data["resource_id"])  # already hashed
        obj.timestamp   = float(data["timestamp"])
        obj.metadata    = dict(data.get("metadata") or {})
        return obj

    # Dunder methods 
    def __repr__(self) -> str:
        return (
            f"UserAction(type={self.action_type!r}, "
            f"resource={self.resource_id[:8]}..., "
            f"t={self.timestamp:.3f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UserAction):
            return NotImplemented
        return (
            self.action_type == other.action_type
            and self.resource_id == other.resource_id
            and abs(self.timestamp - other.timestamp) < 1e-6
        )

    def __hash__(self) -> int:
        return hash((self.action_type, self.resource_id, round(self.timestamp, 6)))

    def __lt__(self, other: "UserAction") -> bool:
        """Allow sorting by timestamp."""
        if not isinstance(other, UserAction):
            return NotImplemented
        return self.timestamp < other.timestamp

# Module-level convenience
def make_action(
    action_type: str,
    resource_id: str = "resource",
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> UserAction:
    """
    Convenience factory.  Shorter to type than UserAction(...) directly.
    Primarily used in test code and simulation scripts.
    """
    return UserAction(
        action_type=action_type,
        resource_id=resource_id,
        metadata=metadata,
        _timestamp=timestamp,
    )

def validate_action_type(action_type: str) -> bool:
    """
    Check whether a string is a valid action type without constructing
    a full UserAction object.
    """
    return action_type.strip().lower() in UserAction._TYPE_INDEX
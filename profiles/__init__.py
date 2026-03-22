"""
Profile persistence, drift detection, and population statistics layer.
"""

from profiles.redis_store import RedisProfileStore, ProfileStorageError
from profiles.drift_detector import DriftDetector, DriftReport
from profiles.shared_device_detector import SharedDeviceDetector, SharedDeviceReport
from profiles.population_stats import PopulationStats

__all__ = [
    "RedisProfileStore",
    "ProfileStorageError",
    "DriftDetector",
    "DriftReport",
    "SharedDeviceDetector",
    "SharedDeviceReport",
    "PopulationStats",
]

__version__ = "2.0.0"
__author__  = "Vyrak"
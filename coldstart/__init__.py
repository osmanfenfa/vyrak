"""
Cold start management layer for the Zelvryn ABSA system.
"""

from coldstart.population_prior import PopulationPriorProfile
from coldstart.ramp_manager import RampUpManager
from coldstart.device_signals import DeviceSignalBridge, DeviceRiskScore

__all__ = [
    "PopulationPriorProfile",
    "RampUpManager",
    "DeviceSignalBridge",
    "DeviceRiskScore",
]

__version__ = "2.0.0"
__author__  = "Vyrak"
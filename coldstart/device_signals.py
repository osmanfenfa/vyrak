"""
Compatibility shim for the historical module typo `device_signal.py`.
"""

from coldstart.device_signal import DeviceRiskScore, DeviceSignalBridge

__all__ = ["DeviceSignalBridge", "DeviceRiskScore"]
